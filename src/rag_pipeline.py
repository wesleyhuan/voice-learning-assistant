"""
RAG Pipeline
- Loads preloaded JSONL corpus
- Chunking with RecursiveCharacterTextSplitter
- Embedding with all-MiniLM-L6-v2
- Hybrid search: BM25 + Vector + RRF fusion
- FlashRank reranking

The preloaded corpus is built once into a shared, read-only base index.
User uploads go into separate per-session indexes (see build_index /
add_to_index) so one visitor's documents never leak into another's
knowledge base, and the shared index is never mutated concurrently.
"""

import hashlib
import json
import os
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 64
TOP_K_RETRIEVE  = 10   # candidates before rerank
TOP_K_FINAL     = 5    # final results after rerank

_ranker = None
_embeddings = None

def _get_ranker():
    global _ranker
    if _ranker is None:
        _ranker = Ranker(
            model_name="ms-marco-MiniLM-L-12-v2",
            cache_dir="/tmp/flashrank"
        )
    return _ranker


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def _build_bm25(chunks: list) -> BM25Okapi:
    return BM25Okapi([c.page_content.lower().split() for c in chunks])


def build_index_from_jsonl(path: str = "preloaded_corpus.jsonl",
                           cache_dir: str | None = None):
    """Load corpus → chunk → embed → build BM25 + Chroma indexes.

    When cache_dir is set, the Chroma collection is persisted to disk keyed
    on the corpus content + chunking/embedding params, so a process restart
    skips the slow embedding step. Chunking and BM25 are always rebuilt —
    they take a fraction of a second.
    """
    with open(path, "rb") as f:
        raw = f.read()

    docs = []
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if line:
            item = json.loads(line)
            if item.get("content"):
                docs.append(item)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for doc in docs:
        splits = splitter.create_documents(
            texts=[doc["content"]],
            metadatas=[{
                "source": doc.get("source", "Unknown"),
                "url":    doc.get("url", "")
            }]
        )
        chunks.extend(splits)

    embeddings = _get_embeddings()

    if cache_dir:
        params = f"|{EMBEDDING_MODEL}|{CHUNK_SIZE}|{CHUNK_OVERLAP}".encode("utf-8")
        key = hashlib.md5(raw + params).hexdigest()
        persist_dir = os.path.join(cache_dir, key)
        os.makedirs(persist_dir, exist_ok=True)
        vectorstore = Chroma(
            collection_name="base",
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        if vectorstore._collection.count() == 0:
            vectorstore.add_documents(chunks)
    else:
        vectorstore = Chroma.from_documents(chunks, embeddings)

    return vectorstore, _build_bm25(chunks), chunks


def build_index(chunks: list):
    """Build a fresh in-memory index over chunks (per-session user uploads)."""
    vectorstore = Chroma.from_documents(
        chunks,
        _get_embeddings(),
        collection_name=f"user-{uuid.uuid4().hex}",
    )
    return vectorstore, _build_bm25(chunks), list(chunks)


def add_to_index(vectorstore, chunks: list, new_chunks: list) -> BM25Okapi:
    """Add chunks to an existing index (mutates chunks in place).

    Returns a freshly built BM25 for the caller to swap in atomically —
    BM25Okapi can't be extended incrementally, and rebuilding in place
    would expose a half-initialized index to concurrent readers.
    """
    vectorstore.add_documents(new_chunks)
    chunks.extend(new_chunks)
    return _build_bm25(chunks)


def _rrf(results_lists: list, k: int = 60) -> list:
    """Reciprocal Rank Fusion across multiple ranked result lists."""
    scores: dict = {}
    for results in results_lists:
        for rank, doc in enumerate(results):
            # Key on full content: arXiv chunks often share their first
            # ~100 chars (headers, acronym lists), so prefix keys collide.
            key = doc.page_content
            if key not in scores:
                scores[key] = {"doc": doc, "score": 0.0}
            scores[key]["score"] += 1.0 / (k + rank + 1)

    sorted_items = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_items]


def hybrid_search(query: str, indexes: list, top_k: int = TOP_K_FINAL) -> list:
    """
    Search one or more indexes and fuse the results.

    Args:
        indexes: list of (vectorstore, bm25, chunks) triples — typically the
                 shared base index plus an optional per-session upload index.

    1. Vector similarity search (per index)
    2. BM25 keyword search (per index)
    3. RRF fusion across all result lists
    4. FlashRank reranking
    """
    tokenized_query = query.lower().split()
    result_lists = []

    for vectorstore, bm25, chunks in indexes:
        result_lists.append(vectorstore.similarity_search(query, k=TOP_K_RETRIEVE))

        bm25_scores  = bm25.get_scores(tokenized_query)
        top_bm25_idx = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:TOP_K_RETRIEVE]
        result_lists.append([chunks[i] for i in top_bm25_idx])

    fused = _rrf(result_lists)[:TOP_K_RETRIEVE]

    try:
        ranker  = _get_ranker()
        request = RerankRequest(
            query=query,
            passages=[
                {"id": i, "text": doc.page_content}
                for i, doc in enumerate(fused)
            ]
        )
        reranked = ranker.rerank(request)
        return [fused[r["id"]] for r in reranked[:top_k]]
    except Exception:
        # Fallback: return RRF results without reranking
        return fused[:top_k]

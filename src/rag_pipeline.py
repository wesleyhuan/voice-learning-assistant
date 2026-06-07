"""
RAG Pipeline
- Loads preloaded JSONL corpus
- Chunking with RecursiveCharacterTextSplitter
- Embedding with all-MiniLM-L6-v2
- Hybrid search: BM25 + Vector + RRF fusion
- FlashRank reranking
"""

import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 64
TOP_K_RETRIEVE  = 10   # candidates before rerank
TOP_K_FINAL     = 5    # final results after rerank

_ranker = None

def _get_ranker():
    global _ranker
    if _ranker is None:
        _ranker = Ranker(
            model_name="ms-marco-MiniLM-L-12-v2",
            cache_dir="/tmp/flashrank"
        )
    return _ranker


def build_index_from_jsonl(path: str = "preloaded_corpus.jsonl"):
    """Load corpus → chunk → embed → build BM25 + Chroma indexes."""
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
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

    embeddings  = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(chunks, embeddings)

    tokenized = [c.page_content.lower().split() for c in chunks]
    bm25      = BM25Okapi(tokenized)

    return vectorstore, bm25, chunks


def add_documents(vectorstore, bm25, chunks: list, new_chunks: list):
    """Add new chunks to existing indexes (mutates chunks list in-place)."""
    if not new_chunks:
        return
    vectorstore.add_documents(new_chunks)
    chunks.extend(new_chunks)
    tokenized = [c.page_content.lower().split() for c in chunks]
    bm25.__init__(tokenized)


def _rrf(results_lists: list, k: int = 60) -> list:
    """Reciprocal Rank Fusion across multiple ranked result lists."""
    scores: dict = {}
    for results in results_lists:
        for rank, doc in enumerate(results):
            key = doc.page_content[:120]
            if key not in scores:
                scores[key] = {"doc": doc, "score": 0.0}
            scores[key]["score"] += 1.0 / (k + rank + 1)

    sorted_items = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_items]


def hybrid_search(query: str, vectorstore, bm25, chunks: list,
                  top_k: int = TOP_K_FINAL) -> list:
    """
    1. Vector similarity search
    2. BM25 keyword search
    3. RRF fusion
    4. FlashRank reranking
    """
    # 1. Vector search
    vector_results = vectorstore.similarity_search(query, k=TOP_K_RETRIEVE)

    # 2. BM25 search
    tokenized_query = query.lower().split()
    bm25_scores     = bm25.get_scores(tokenized_query)
    top_bm25_idx    = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:TOP_K_RETRIEVE]
    bm25_results = [chunks[i] for i in top_bm25_idx]

    # 3. RRF fusion
    fused = _rrf([vector_results, bm25_results])[:TOP_K_RETRIEVE]

    # 4. FlashRank reranking
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

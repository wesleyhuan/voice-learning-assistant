"""
Ingestion module — for user-uploaded materials (PDF or URL)
Uses PyMuPDF for PDFs and trafilatura for web pages.
"""

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)


def ingest_pdf(file_bytes: bytes, source_name: str = "uploaded.pdf") -> list:
    """Extract text from a PDF and return a list of Document chunks."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")

    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    doc.close()

    if not full_text.strip():
        return []

    return _splitter.create_documents(
        texts=[full_text],
        metadatas=[{"source": source_name, "url": ""}]
    )


def ingest_url(url: str) -> list:
    """Scrape a URL and return a list of Document chunks."""
    if not HAS_TRAFILATURA:
        return []

    downloaded = trafilatura.fetch_url(url)
    text = trafilatura.extract(downloaded) if downloaded else None

    if not text or not text.strip():
        return []

    return _splitter.create_documents(
        texts=[text],
        metadatas=[{"source": url, "url": url}]
    )

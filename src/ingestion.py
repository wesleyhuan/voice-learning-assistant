"""
Ingestion module — for user-uploaded materials (PDF or URL)
Uses PyMuPDF for PDFs and trafilatura for web pages.
"""

import ipaddress
import socket
from urllib.parse import urlparse

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

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


def _is_public_http_url(url: str) -> bool:
    """Allow only public http(s) URLs — the fetch runs on the server, so
    internal/loopback addresses would be an SSRF vector."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if not ip.is_global:
            return False
    return True


def ingest_url(url: str) -> list:
    """Scrape a URL and return a list of Document chunks."""
    if not HAS_TRAFILATURA:
        return []

    if not _is_public_http_url(url):
        return []

    downloaded = trafilatura.fetch_url(url)
    text = trafilatura.extract(downloaded) if downloaded else None

    if not text or not text.strip():
        return []

    return _splitter.create_documents(
        texts=[text],
        metadatas=[{"source": url, "url": url}]
    )

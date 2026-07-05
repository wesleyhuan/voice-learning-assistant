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


import urllib.request
import urllib.error

class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Intercept redirects to ensure the target URL is also public."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _is_public_http_url(newurl):
            raise urllib.error.URLError(f"Redirect to non-public URL blocked: {newurl}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)

def _safe_fetch(url: str) -> bytes | None:
    """Fetch URL safely following redirects only to public IPs."""
    opener = urllib.request.build_opener(SafeRedirectHandler())
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    try:
        with opener.open(req, timeout=10) as response:
            return response.read()
    except Exception:
        return None

def ingest_url(url: str) -> list:
    """Scrape a URL and return a list of Document chunks."""
    if not HAS_TRAFILATURA:
        return []

    if not _is_public_http_url(url):
        return []

    downloaded = _safe_fetch(url)
    text = trafilatura.extract(downloaded) if downloaded else None

    if not text or not text.strip():
        return []

    return _splitter.create_documents(
        texts=[text],
        metadatas=[{"source": url, "url": url}]
    )

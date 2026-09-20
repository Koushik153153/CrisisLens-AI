"""
document_processor.py - Parse PDF/DOCX files and split into text chunks.
"""

import os
import hashlib
from typing import List, Tuple
from config import CHUNK_SIZE, CHUNK_OVERLAP


def compute_file_hash(file_bytes: bytes) -> str:
    """Return MD5 hex-digest of file bytes — used for change detection."""
    return hashlib.md5(file_bytes).hexdigest()


class DocumentProcessor:
    """Handles parsing of PDF and DOCX documents and text chunking."""

    # ─── Parsing ──────────────────────────────────────────────────────────────

    def parse_pdf(self, file_path: str) -> str:
        """Extract all text from a PDF file using pypdf."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            texts = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    texts.append(t)
            return "\n".join(texts)
        except Exception as e:
            raise RuntimeError(f"Failed to parse PDF '{file_path}': {e}")

    def parse_docx(self, file_path: str) -> str:
        """Extract all text from a DOCX file using python-docx."""
        try:
            from docx import Document
            doc = Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paragraphs)
        except Exception as e:
            raise RuntimeError(f"Failed to parse DOCX '{file_path}': {e}")

    def parse_txt(self, file_path: str) -> str:
        """Read a plain-text (.txt) file with automatic encoding detection."""
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                with open(file_path, "r", encoding=enc) as fh:
                    return fh.read()
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f"Failed to decode '{file_path}': unknown encoding.")

    def parse(self, file_path: str) -> str:
        """Auto-detect file type and parse accordingly."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self.parse_pdf(file_path)
        elif ext == ".docx":
            return self.parse_docx(file_path)
        elif ext == ".txt":
            return self.parse_txt(file_path)
        elif ext == ".doc":
            raise ValueError(
                ".doc (legacy Word 97-2003) format is not supported. "
                "Please open the file in Microsoft Word and save it as "
                "'.docx' (Word Document), then re-upload."
            )
        else:
            raise ValueError(f"Unsupported file type: '{ext}'. Supported formats: .pdf, .docx, .txt")

    # ─── Chunking ─────────────────────────────────────────────────────────────

    def chunk_text(
        self,
        text: str,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
    ) -> List[str]:
        """
        Split text into overlapping character-level chunks.

        Args:
            text: Raw document text.
            chunk_size: Max characters per chunk.
            overlap: Characters shared between consecutive chunks.

        Returns:
            List of chunk strings.
        """
        if not text.strip():
            return []

        chunks = []
        start = 0
        step = max(chunk_size - overlap, 1)

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(text):
                break
            start += step

        return chunks

    # ─── High-level entry point ───────────────────────────────────────────────

    def process_file(
        self,
        file_path: str,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
    ) -> Tuple[str, List[str]]:
        """
        Parse a file and return its full text plus list of chunks.

        Returns:
            (full_text, chunks)
        """
        raw_text = self.parse(file_path)
        chunks = self.chunk_text(raw_text, chunk_size, overlap)
        return raw_text, chunks

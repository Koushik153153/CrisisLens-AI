"""
vector_store.py - ChromaDB vector store with live document ingestion.
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
import chromadb
from chromadb.config import Settings
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION


class VectorStore:
    """
    ChromaDB-backed vector store.

    Supports live document ingestion (add_document) without restarting
    the application. Each chunk is stored with metadata that records
    which source document it came from.
    """

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = CHROMA_COLLECTION,
    ):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ─── Ingestion ────────────────────────────────────────────────────────────

    def add_document(
        self,
        doc_id: str,
        chunks: List[str],
        embeddings: List[np.ndarray],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Add all chunks from one document into the collection.

        Args:
            doc_id:     Unique document identifier (e.g. filename stem).
            chunks:     List of text chunks.
            embeddings: Corresponding list of embedding vectors.
            metadata:   Extra metadata fields (e.g. filename, upload time).

        Returns:
            Number of chunks added.
        """
        if not chunks:
            return 0

        ids = [f"{doc_id}__chunk_{i}" for i in range(len(chunks))]
        metas = [
            {**(metadata or {}), "doc_id": doc_id, "chunk_index": i}
            for i in range(len(chunks))
        ]
        vecs = [e.tolist() for e in embeddings]

        # Upsert so re-uploading the same document replaces old chunks
        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=vecs,
            metadatas=metas,
        )
        return len(chunks)

    def delete_document(self, doc_id: str) -> None:
        """Remove all chunks belonging to a document."""
        results = self.collection.get(where={"doc_id": doc_id})
        if results and results["ids"]:
            self.collection.delete(ids=results["ids"])

    # ─── Retrieval ────────────────────────────────────────────────────────────

    def query(
        self, embedding: np.ndarray, k: int
    ) -> Dict[str, Any]:
        """
        Query the collection for the top-K most similar chunks.

        Returns a dict with keys:
            ids, documents, metadatas, distances  (all lists of lists)
        """
        results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        return results

    # ─── Stats ────────────────────────────────────────────────────────────────

    def get_collection_stats(self) -> Dict[str, int]:
        """
        Return chunk count per document.

        Returns:
            {doc_id: num_chunks}
        """
        all_items = self.collection.get(include=["metadatas"])
        stats: Dict[str, int] = {}
        for meta in all_items.get("metadatas", []):
            did = meta.get("doc_id", "unknown")
            stats[did] = stats.get(did, 0) + 1
        return stats

    def get_document_hashes(self) -> Dict[str, str]:
        """
        Return the MD5 hash stored in metadata for each document.
        
        Returns:
            {doc_id: file_hash}
        """
        all_items = self.collection.get(include=["metadatas"])
        hashes: Dict[str, str] = {}
        for meta in all_items.get("metadatas", []):
            did = meta.get("doc_id")
            h = meta.get("file_hash")
            if did and h and did not in hashes:
                hashes[did] = h
        return hashes

    def total_chunks(self) -> int:
        """Return total number of chunks stored."""
        return self.collection.count()

    def list_documents(self) -> List[str]:
        """Return list of unique document IDs in the store."""
        return list(self.get_collection_stats().keys())

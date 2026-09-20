"""
retriever.py - Retriever with configurable K and adaptive retrieval logic.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from embedder import Embedder
from vector_store import VectorStore
from config import (
    DEFAULT_K,
    ADAPTIVE_HIGH_THRESHOLD,
    ADAPTIVE_MID_THRESHOLD,
    ADAPTIVE_LOW_K,
)


class Retriever:
    """
    Retrieves top-K chunks for a query.

    Supports:
    - Manual K: retrieve(query, k)
    - Adaptive K: adaptive_retrieve(query) — picks K based on top similarity
    """

    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _parse_results(self, raw: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Flatten ChromaDB query result into a list of chunk dicts.

        Each dict has:
            chunk_id, doc_id, text, similarity, metadata
        """
        parsed = []
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for cid, doc, meta, dist in zip(ids, docs, metas, distances):
            # ChromaDB cosine distance = 1 - cosine_similarity
            similarity = max(0.0, 1.0 - dist)
            parsed.append(
                {
                    "chunk_id": cid,
                    "doc_id": meta.get("doc_id", "unknown"),
                    "text": doc,
                    "similarity": similarity,
                    "metadata": meta,
                }
            )

        # Sort by descending similarity
        parsed.sort(key=lambda x: x["similarity"], reverse=True)
        return parsed

    # ─── Public API ───────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = DEFAULT_K) -> List[Dict[str, Any]]:
        """
        Retrieve top-K chunks for *query*.

        Returns:
            List of chunk dicts (sorted by descending similarity).
        """
        if self.store.total_chunks() == 0:
            return []

        query_emb = self.embedder.embed(query)
        raw = self.store.query(query_emb, k=k)
        return self._parse_results(raw)

    def adaptive_retrieve(self, query: str) -> Tuple[List[Dict[str, Any]], int]:
        """
        Retrieve with automatic K selection based on top similarity score.

        Logic:
            top_similarity > ADAPTIVE_HIGH_THRESHOLD → K = 1
            top_similarity > ADAPTIVE_MID_THRESHOLD  → K = 3
            else                                     → K = 5

        Returns:
            (results, chosen_k)
        """
        if self.store.total_chunks() == 0:
            return [], DEFAULT_K

        # Probe with K=1 first to get the best similarity
        query_emb = self.embedder.embed(query)
        probe = self._parse_results(self.store.query(query_emb, k=1))
        top_sim = probe[0]["similarity"] if probe else 0.0

        if top_sim > ADAPTIVE_HIGH_THRESHOLD:
            chosen_k = 1
        elif top_sim > ADAPTIVE_MID_THRESHOLD:
            chosen_k = 3
        else:
            chosen_k = ADAPTIVE_LOW_K

        results = self._parse_results(self.store.query(query_emb, k=chosen_k))
        return results, chosen_k

    def retrieve_all_k(self, query: str, k_values: Optional[List[int]] = None) -> Dict[int, List[Dict[str, Any]]]:
        """
        Retrieve for a specific set of K values in one call.

        Args:
            query:    The user query.
            k_values: List of K depths (e.g. [1, 2, 3]). Default [1, 2, 3, 4, 5].

        Returns:
            {k: results_list}
        """
        k_values = k_values or [1, 2, 3, 4, 5]
        if self.store.total_chunks() == 0:
            return {k: [] for k in k_values}

        max_k = max(k_values)
        query_emb = self.embedder.embed(query)
        
        # Fetch max K needed; slice for each smaller K
        raw_max = self._parse_results(self.store.query(query_emb, k=max_k))
        return {k: raw_max[:k] for k in k_values}

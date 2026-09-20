"""
embedder.py - Sentence embedding using sentence-transformers.
"""

from typing import List, Union
import numpy as np
from config import EMBEDDING_MODEL


class Embedder:
    """
    Wraps SentenceTransformer to produce L2-normalised embeddings
    suitable for cosine similarity comparisons.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    # ─── Core API ─────────────────────────────────────────────────────────────

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single string.

        Returns:
            1-D numpy array (float32).
        """
        vec = self.model.encode(text, normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> List[np.ndarray]:
        """
        Embed a list of strings in batches.

        Returns:
            List of 1-D numpy arrays.
        """
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [np.array(v, dtype=np.float32) for v in vecs]

    # ─── Utility ──────────────────────────────────────────────────────────────

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Cosine similarity between two normalised vectors.
        (Dot product suffices when both are unit vectors.)
        """
        return float(np.dot(vec_a, vec_b))

    def cosine_similarity_matrix(
        self, vecs_a: List[np.ndarray], vecs_b: List[np.ndarray]
    ) -> np.ndarray:
        """
        Returns an (m x n) cosine similarity matrix.
        """
        A = np.stack(vecs_a)   # (m, d)
        B = np.stack(vecs_b)   # (n, d)
        return A @ B.T

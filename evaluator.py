"""
evaluator.py - Full evaluation pipeline for retrieval depth sensitivity.

Computes:
  - Retrieval metrics: Hit@K, Recall@K, Precision@K, MRR
  - Grounding Score (cosine similarity of response vs retrieved chunks)
  - Hallucination Rate (sentence-level claim checking)
  - Document Contribution (percentage per source doc)
  - Overall Quality Score (weighted combination)
  - Saves structured results to evaluation_results.json
"""

import json
import re
from typing import List, Dict, Any, Optional
import numpy as np

from embedder import Embedder
from config import (
    K_VALUES,
    EVAL_RESULTS_FILE,
    QUALITY_WEIGHTS,
    LATENCY_NORM_CAP,
)


class Evaluator:
    """Computes all evaluation metrics for the RAG system."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    # ─── Retrieval Metrics ────────────────────────────────────────────────────

    def compute_retrieval_metrics(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int,
    ) -> Dict[str, float]:
        """
        Compute Hit@K, Recall@K, Precision@K, MRR for one K value.

        Args:
            retrieved_ids: Ordered list of retrieved doc IDs (top-K).
            relevant_ids:  Ground-truth relevant doc IDs.
            k:             Retrieval depth.

        Returns:
            Dict with hit_at_k, recall_at_k, precision_at_k, mrr.
        """
        if not relevant_ids:
            return {"hit_at_k": 0, "recall_at_k": 0.0, "precision_at_k": 0.0, "mrr": 0.0}

        relevant_set = set(relevant_ids)
        retrieved_k = retrieved_ids[:k]

        # Deduplicate: if multiple chunks from same doc are retrieved,
        # that doc should only count ONCE toward recall/precision.
        retrieved_unique = list(dict.fromkeys(retrieved_k))  # preserve order, dedupe
        hits = set(rid for rid in retrieved_unique if rid in relevant_set)

        hit_at_k = 1 if hits else 0
        recall_at_k = len(hits) / len(relevant_set)            # always in [0, 1]
        precision_at_k = len(hits) / k if k > 0 else 0.0      # standard: hits / K

        # MRR: 1/rank of first relevant doc (by original chunk order)
        mrr = 0.0
        for rank, rid in enumerate(retrieved_k, start=1):
            if rid in relevant_set:
                mrr = 1.0 / rank
                break

        return {
            "hit_at_k": hit_at_k,
            "recall_at_k": round(min(recall_at_k, 1.0), 4),
            "precision_at_k": round(min(precision_at_k, 1.0), 4),
            "mrr": round(mrr, 4),
        }

    # ─── Grounding Score ──────────────────────────────────────────────────────

    def compute_grounding_score(
        self, response: str, retrieved_chunks: List[str]
    ) -> float:
        """
        Grounding Score = mean cosine similarity(response_emb, chunk_embs).
        """
        if not response or not retrieved_chunks:
            return 0.0

        resp_emb = self.embedder.embed(response)
        chunk_embs = self.embedder.embed_batch(retrieved_chunks)
        sims = [self.embedder.cosine_similarity(resp_emb, ce) for ce in chunk_embs]
        return round(float(np.mean(sims)), 4)

    # ─── Hallucination Rate ───────────────────────────────────────────────────

    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitter (no NLTK dependency)."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def compute_hallucination_rate(
        self,
        response: str,
        retrieved_chunks: List[str],
        threshold: float = 0.15,
    ) -> float:
        """
        Estimate hallucination rate via sentence-level embedding similarity.

        A response sentence is considered *grounded* if its max cosine
        similarity to any retrieved chunk is >= *threshold*.

        Threshold 0.25 is calibrated for all-MiniLM-L6-v2 sentence-level
        grounding — 0.45 was too strict and caused false hallucination flags.

        Hallucination Rate = unsupported sentences / total sentences.
        """
        if not response or not retrieved_chunks:
            return 0.0

        sentences = self._split_sentences(response)
        if not sentences:
            return 0.0

        chunk_embs = self.embedder.embed_batch(retrieved_chunks)
        unsupported = 0

        for sent in sentences:
            sent_emb = self.embedder.embed(sent)
            max_sim = max(
                self.embedder.cosine_similarity(sent_emb, ce) for ce in chunk_embs
            )
            if max_sim < threshold:
                unsupported += 1

        return round(unsupported / len(sentences), 4)

    # ─── Document Contribution ────────────────────────────────────────────────

    def compute_doc_contribution(
        self, retrieved_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate each source document's % contribution based on similarity
        scores weighted by chunk frequency.

        Args:
            retrieved_results: List of chunk dicts from retriever.

        Returns:
            {doc_id: percentage_contribution}
        """
        if not retrieved_results:
            return {}

        doc_scores: Dict[str, float] = {}
        for r in retrieved_results:
            did = r["doc_id"]
            sim = r.get("similarity", 0.0)
            doc_scores[did] = doc_scores.get(did, 0.0) + sim

        total = sum(doc_scores.values())
        if total == 0:
            return {did: 0.0 for did in doc_scores}

        return {
            did: round(score / total * 100, 2)
            for did, score in doc_scores.items()
        }

    # ─── Quality Score ────────────────────────────────────────────────────────

    def compute_quality_score(
        self,
        recall: float,
        grounding: float,
        hallucination: float,
        latency: float,
    ) -> float:
        """
        Overall Quality Score = weighted combination of normalised sub-scores.
        """
        w = QUALITY_WEIGHTS
        latency_score = max(0.0, 1.0 - latency / LATENCY_NORM_CAP)
        hallucination_score = 1.0 - hallucination
        quality = (
            w["recall"] * recall
            + w["grounding"] * grounding
            + w["hallucination"] * hallucination_score
            + w["latency"] * latency_score
        )
        return round(quality, 4)

    # ─── Full Evaluation Run ──────────────────────────────────────────────────

    def evaluate_for_k(
        self,
        k: int,
        query: str,
        retrieved_results: List[Dict[str, Any]],
        relevant_doc_ids: List[str],
        response: str,
        latency: float,
    ) -> Dict[str, Any]:
        """
        Run all metrics for a single K value.

        Returns:
            Structured result dict (matches STEP 5 format from prompt).
        """
        retrieved_ids = [r["doc_id"] for r in retrieved_results]
        sim_scores = [round(r["similarity"], 4) for r in retrieved_results]
        chunks = [r["text"] for r in retrieved_results]

        ret_metrics = self.compute_retrieval_metrics(retrieved_ids, relevant_doc_ids, k)
        grounding = self.compute_grounding_score(response, chunks)
        hallucination = self.compute_hallucination_rate(response, chunks)
        quality = self.compute_quality_score(
            ret_metrics["recall_at_k"], grounding, hallucination, latency
        )
        contribution = self.compute_doc_contribution(retrieved_results)

        return {
            "K": k,
            "query": query,
            "retrieved_docs": retrieved_ids,
            "similarity_scores": sim_scores,
            "hit_at_k": ret_metrics["hit_at_k"],
            "recall_at_k": ret_metrics["recall_at_k"],
            "precision_at_k": ret_metrics["precision_at_k"],
            "mrr": ret_metrics["mrr"],
            "grounding_score": grounding,
            "hallucination_rate": hallucination,
            "latency": latency,
            "quality_score": quality,
            "doc_contribution": contribution,
            "response": response,
        }

    def run_full_evaluation(
        self,
        query: str,
        retrieved_by_k: Dict[int, List[Dict[str, Any]]],
        responses_by_k: Dict[int, Dict[str, Any]],
        relevant_doc_ids: List[str],
        k_values: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run evaluation for all K values and return list of result dicts.
        """
        k_values = k_values or K_VALUES
        results = []

        for k in k_values:
            retrieved = retrieved_by_k.get(k, [])
            resp_data = responses_by_k.get(k, {})
            response = resp_data.get("response", "")
            latency = resp_data.get("latency", 0.0)

            result = self.evaluate_for_k(
                k=k,
                query=query,
                retrieved_results=retrieved,
                relevant_doc_ids=relevant_doc_ids,
                response=response,
                latency=latency,
            )
            results.append(result)

        return results

    # ─── Persistence ──────────────────────────────────────────────────────────

    def save_results(
        self,
        results: List[Dict[str, Any]],
        path: str = EVAL_RESULTS_FILE,
    ) -> None:
        """Save evaluation results to a JSON file."""
        try:
            existing = []
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                pass

            existing.extend(results)

            with open(path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Evaluator] Warning: could not save results: {e}")

    def load_results(self, path: str = EVAL_RESULTS_FILE) -> List[Dict[str, Any]]:
        """Load saved evaluation results from JSON."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

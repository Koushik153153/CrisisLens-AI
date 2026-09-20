"""
llm_handler.py - Gemini API integration with lazy model fallback.

No upfront test calls are made (they waste your RPM quota). Instead,
if a model returns 404, the next one in the fallback chain is tried
at the first real query.
"""

import time
from typing import List, Dict, Any, Tuple, Optional

import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL

# Models tried in order until one succeeds.
# Covers every combination available on free AI Studio keys.
MODEL_FALLBACK_CHAIN = [
    "gemini-2.5-flash",     # preferred — best quality/speed on free tier
    GEMINI_MODEL,           # from config
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
    "gemini-pro",
    "gemini-1.0-pro",
]

# De-duplicate while preserving order
_SEEN: list[str] = []
for _m in MODEL_FALLBACK_CHAIN:
    if _m not in _SEEN:
        _SEEN.append(_m)
MODEL_FALLBACK_CHAIN = _SEEN

INTER_CALL_DELAY = 5    # seconds between sequential K calls
MAX_RETRIES      = 3    # retries on rate-limit errors


class LLMHandler:
    """
    Wraps the Google Gemini API to generate RAG responses.

    - No test calls on init (saves quota).
    - Falls back to next model in chain on 404.
    - Exponential back-off on 429.
    """

    def __init__(self, api_key: str = GEMINI_API_KEY, model: str = GEMINI_MODEL):
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Copy .env.example → .env and add your key."
            )
        genai.configure(api_key=api_key)
        self._model_index = 0           # position in MODEL_FALLBACK_CHAIN
        self._init_model()

    def _init_model(self):
        name = MODEL_FALLBACK_CHAIN[self._model_index]
        self.model_name = name
        self.model      = genai.GenerativeModel(name)

    def _advance_model(self) -> bool:
        """Switch to the next model in the fallback chain. Returns False if exhausted."""
        self._model_index += 1
        if self._model_index >= len(MODEL_FALLBACK_CHAIN):
            return False
        self._init_model()
        print(f"[LLM] Switching to fallback model: {self.model_name}")
        return True

    # ─── Prompt builder ───────────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(query: str, context_chunks: List[str]) -> str:
        numbered = "\n\n".join(
            f"[Passage {i+1}]\n{c}" for i, c in enumerate(context_chunks)
        )
        return (
            "You are a helpful research assistant. "
            "Answer the question using ONLY the passages provided below. "
            "If the answer is not contained in the passages, say "
            "'I could not find a relevant answer in the provided documents.'\n\n"
            f"--- RETRIEVED PASSAGES ---\n{numbered}\n"
            f"--- END OF PASSAGES ---\n\n"
            f"Question: {query}\n\nAnswer:"
        )

    # ─── Core call (single model, single attempt) ─────────────────────────────

    def _call_once(self, prompt: str) -> Tuple[str, float]:
        """Call current model. Raises ValueError on 404, RuntimeError on others."""
        t0       = time.time()
        response = self.model.generate_content(prompt)
        text     = response.text.strip()
        latency  = max(time.time() - t0, 0.001)
        return text, round(latency, 3)

    # ─── Retry wrapper with model fallback ────────────────────────────────────

    def _call_with_retry(self, prompt: str) -> Tuple[str, float]:
        """
        Attempt the API call with:
        - Model fallback on 404 (not found)
        - Exponential back-off on 429 (rate limit)
        """
        wait = 15

        # Outer loop: model fallback
        while True:
            # Inner loop: rate-limit retries
            for attempt in range(MAX_RETRIES):
                try:
                    return self._call_once(prompt)

                except Exception as e:
                    err = str(e)
                    if "404" in err or "not found" in err.lower():
                        # This model doesn't exist for the account → try next
                        print(f"[LLM] 404 for {self.model_name}: trying next model…")
                        if not self._advance_model():
                            raise RuntimeError(
                                "None of the Gemini models are available for your API key. "
                                "Please verify your key at aistudio.google.com "
                                "and ensure the Generative Language API is enabled."
                            )
                        break   # restart inner loop with new model

                    elif "429" in err or "quota" in err.lower() or "resource_exhausted" in err.lower():
                        if attempt < MAX_RETRIES - 1:
                            print(f"[LLM] Rate limit on {self.model_name}. Waiting {wait}s…")
                            time.sleep(wait)
                            wait = min(wait * 2, 120)
                        else:
                            raise RuntimeError(
                                f"Rate limit exceeded on '{self.model_name}' after "
                                f"{MAX_RETRIES} attempts. "
                                "Free tier allows 15 requests/min. "
                                "Please wait ~1 minute before retrying."
                            ) from e

                    else:
                        raise RuntimeError(f"Gemini API error: {e}") from e

            else:
                # inner loop exhausted without break — all retries failed
                raise RuntimeError(f"Gemini API: all {MAX_RETRIES} retries failed.")

    # ─── Main API ─────────────────────────────────────────────────────────────

    def generate_response(
        self, query: str, context_chunks: List[str]
    ) -> Tuple[str, float]:
        prompt = self._build_prompt(query, context_chunks)
        return self._call_with_retry(prompt)

    def generate_responses_for_all_k(
        self,
        query: str,
        retrieved_by_k: Dict[int, List[Dict[str, Any]]],
    ) -> Dict[int, Dict[str, Any]]:
        outputs: Dict[int, Dict[str, Any]] = {}
        k_list = sorted(retrieved_by_k.keys())

        for i, k in enumerate(k_list):
            results = retrieved_by_k[k]
            chunks  = [r["text"] for r in results]

            if not chunks:
                outputs[k] = {"response": "No documents indexed yet.", "latency": 0.0, "chunks": []}
                continue

            if i > 0:
                time.sleep(INTER_CALL_DELAY)

            try:
                t0 = time.time()
                response, api_latency = self.generate_response(query, chunks)
                wall    = round(time.time() - t0, 3)
                latency = max(api_latency, wall, 0.001)
                outputs[k] = {"response": response, "latency": latency, "chunks": chunks}
            except RuntimeError as e:
                outputs[k] = {"response": f"⚠️ {e}", "latency": 0.0, "chunks": chunks}

        return outputs

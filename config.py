"""
config.py - Central configuration for CrisisLens-AI
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── API ──────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-1.5-flash"

# ─── Embedding ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ─── Document Chunking ────────────────────────────────────────────────────────
CHUNK_SIZE = 512          # characters per chunk
CHUNK_OVERLAP = 64        # character overlap between chunks

# ─── Retrieval ────────────────────────────────────────────────────────────────
K_VALUES = [1, 2, 3, 4, 5]          # retrieval depths to evaluate
DEFAULT_K = 3

# Adaptive retrieval thresholds (cosine similarity)
ADAPTIVE_HIGH_THRESHOLD = 0.85      # → K=1
ADAPTIVE_MID_THRESHOLD = 0.70       # → K=3
ADAPTIVE_LOW_K = 5                  # default when below mid threshold

# ─── Vector Store ─────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = "./chroma_db"
CRISISLENS_CHROMA_COLLECTION = "crisislens_emergency_reports"
CHROMA_COLLECTION = CRISISLENS_CHROMA_COLLECTION  # Existing vector-store API

# ─── Evaluation ───────────────────────────────────────────────────────────────
EVAL_RESULTS_FILE = "evaluation_results.json"

# Quality score weights (must sum to 1.0)
QUALITY_WEIGHTS = {
    "recall":     0.35,
    "grounding":  0.35,
    "hallucination": 0.20,   # inverted: lower hallucination → higher score
    "latency":    0.10,      # inverted: lower latency → higher score
}
LATENCY_NORM_CAP = 10.0     # seconds — latency above this → 0 score

# ─── Upload ───────────────────────────────────────────────────────────────────
UPLOAD_DIR = "./uploaded_docs"
ALLOWED_EXTENSIONS = [".pdf", ".docx", ".txt"]

# Run from the project directory. Legacy uploads are retained but not scanned.
EMERGENCY_REPORT_DIR = "./emergency_reports"
EMERGENCY_REPORT_EXTENSIONS = (".pdf", ".docx", ".txt")
EMERGENCY_DATA_DIR = "./emergency_data"
PROCESSED_REPORTS_FILE = "./emergency_data/processed_reports.json"

# Step 3: cosine similarity cutoff, not a calibrated probability.
INCIDENT_CONFIDENCE_THRESHOLD = 0.40

# Step 4: deterministic request confidence versus uncalibrated cosine cutoff.
RESOURCE_EXPLICIT_CONFIDENCE = 1.0
RESOURCE_IMPLICIT_THRESHOLD = 0.42

# Step 5: add interpretable cue support to each independent label similarity.
ASSESSMENT_CUE_BONUS = 0.60
ASSESSMENT_LARGE_PERSON_COUNT = 50  # Only contributes alongside active harm.

# Step 6: fixed prototype design parameters, not fitted to the corpus.
EAPS_CONFIG = {
    "weights": {"urgency": 0.40, "severity": 0.30, "actionability": 0.20, "context": 0.10},
    "ordinal": {
        "urgency": {"LOW": 0.25, "MEDIUM": 0.50, "HIGH": 0.75, "CRITICAL": 1.0},
        "severity": {"LOW": 0.25, "MEDIUM": 0.50, "HIGH": 0.75, "CRITICAL": 1.0},
        "actionability": {"LOW": 0.33, "MEDIUM": 0.67, "HIGH": 1.0},
    },
    # Shares of the context budget; at defaults caps are 5, 2 and 3 points.
    "context_shares": {"vulnerability": 0.50, "population": 0.20, "operational_need": 0.30},
    # Maximum matching factor, not a sum. Access/support barriers get more
    # weight than INJURED, whose consequences already influence severity.
    "vulnerability_factors": {"CHILDREN": 0.6, "ELDERLY": 0.6, "PREGNANT": 0.8,
                              "DISABLED": 0.8, "CHRONICALLY_ILL": 0.6, "INJURED": 0.4},
    "population_tiers": {1: 0.50, 10: 1.0},  # Explicit individual persons only.
    "bands": {"LOW": 0, "MEDIUM": 40, "HIGH": 60, "CRITICAL": 80},
}

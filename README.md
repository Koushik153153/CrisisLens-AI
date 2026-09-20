# CrisisLens-AI

Real-Time NLP Emergency Intelligence & Decision Support

Architecture preparation for **An NLP-Based Emergency Intelligence System for
Actionability Detection, Resource-Need Extraction and Explainable Incident
Prioritization**. Emergency NLP algorithms are not implemented yet.

## Current operation

- Run from this project directory: `.venv\Scripts\python.exe -m streamlit run app.py`.
- Restart any old Streamlit process after this migration to clear cached resources
  and the old background watcher.
- Put incoming PDF, DOCX or TXT reports directly in `emergency_reports/`, or upload
  them in Document Hub. New uploads are saved there too.
- Only `emergency_reports/` is watched and scanned at startup, non-recursively.
  Queue processing happens on Streamlit reruns; refresh/interact to process changes.
- Root-level experimental files and existing `uploaded_docs/` are retained and
  are not automatically ingested. Explicitly uploading a file still indexes it.
- Vectors use `crisislens_emergency_reports` in the existing `chroma_db/` directory.
  The legacy `live_rag_docs` collection is retained and is not queried by this app.
- `emergency_schema.py` defines `EmergencyReport`. Undetected values use `None`,
  and multi-value fields use independent lists. Serialize with
  `json.dumps(dataclasses.asdict(report))`; timestamps are ISO 8601 strings.
- `emergency_data/` is reserved for future structured analysis results.
- Existing Gemini environment-variable loading is unchanged. No new dependencies
  are needed. Use the existing environment; this copy has no requirements file.
- Run offline regression checks with
  `.venv\Scripts\python.exe -m unittest test_architecture -v`.

## Retained limitations

Document IDs are filename stems, so files with the same stem can collide.
Live ingestion deletes prior chunks before processing succeeds and silently skips
exceptions. Deletion/rename events are not synchronized. Existing evaluation
history still uses `evaluation_results.json`, so legacy results can appear in the
retained evaluation tabs. The installed Gemini SDK emits a deprecation warning.
These are preserved legacy behaviors to address separately.

## Original RAG project background

I am one of the lead contributor in this project
# NLP_PROJECT_SEM6
# 📚 Live RAG System — Retrieval Depth Sensitivity Analysis

> A **Live Retrieval-Augmented Generation (RAG)** system that dynamically ingests documents, answers questions using Google Gemini, and rigorously evaluates how **retrieval depth (K)** affects answer quality — all through an interactive Streamlit dashboard.

---

## What This Project Does

Most RAG systems treat **K** (the number of retrieved passages) as a fixed, ignored constant. This project challenges that assumption.

**Key capabilities:**
-  **Live Document Ingestion** — Upload PDFs, DOCX, or TXT files; they are instantly chunked, embedded, and searchable without restarting.
-  **Semantic Search** — Uses `all-MiniLM-L6-v2` embeddings stored in **ChromaDB** for fast, accurate vector retrieval.
-  **LLM Answer Generation** — Passes retrieved chunks through **Google Gemini 1.5 Flash** with relevance filtering to prevent hallucination.
- **Retrieval Depth Evaluation** — Automatically evaluates K = 1 → 5, measuring how retrieval depth impacts answer quality.
-  **Document Contribution Analysis** — Shows which uploaded documents contributed most to a given answer.
-  **Interactive Dashboard** — Plotly charts inside Streamlit tabs for visual analysis of all metrics.

---

##  System Architecture

```text
User Query
    │
    ▼
Semantic Search (ChromaDB + MiniLM)
    │
    ▼
Relevance Filter (cosine similarity threshold)
    │
    ▼
Google Gemini 1.5 Flash (LLM)
    │
    ▼
Answer + Evaluation Pipeline (K=1..5)
    │
    ▼
Streamlit Dashboard (metrics, charts, contribution analysis)

"""CrisisLens-AI Command Center; legacy presentation helpers retained for compatibility."""

import os
import time
import json
import tempfile
import streamlit as st
import pandas as pd


class UnavailableRAGStore:
    """Keep indexing failures explicit while allowing NLP persistence to proceed."""

    def add_document(self, *args, **kwargs):
        raise RuntimeError("RAG index is unavailable.")

    def get_document_hashes(self):
        return {}

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CrisisLens-AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Imports (lazy to avoid early crash if deps missing) ─────────────────────
@st.cache_resource(show_spinner="🚀 Loading models... (first run may take ~60s)")
def load_resources():
    from config import EMERGENCY_REPORT_DIR
    from embedder import Embedder
    from evaluator import Evaluator
    from visualizer import Visualizer
    from file_watcher import start_watcher
    from emergency_pipeline import EmergencyProcessingPipeline

    os.makedirs(EMERGENCY_REPORT_DIR, exist_ok=True)

    embedder   = Embedder()
    try:
        from vector_store import VectorStore
        from retriever import Retriever
        store = VectorStore()
        retriever = Retriever(embedder, store)
    except Exception:
        store, retriever = UnavailableRAGStore(), None
    try:
        from llm_handler import LLMHandler
        llm = LLMHandler()
    except Exception:
        llm = None  # Optional generation must not block emergency intelligence.
    evaluator  = Evaluator(embedder)
    visualizer = Visualizer()
    pipeline = EmergencyProcessingPipeline.from_components(embedder, store)

    # Only incoming emergency reports are watched.
    start_watcher()

    return embedder, store, retriever, llm, evaluator, visualizer, pipeline

# ─── Sidebar ─────────────────────────────────────────────────────────────────

def render_sidebar(store):
    with st.sidebar:
        st.markdown(
            "<h2 style='color:#A5B4FC;margin-bottom:0'>🔬 CrisisLens-AI</h2>"
            "<p style='color:#64748B;font-size:0.81rem;margin-top:0'>Real-Time NLP Emergency Intelligence &amp; Decision Support</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        stats = store.get_collection_stats()
        docs  = list(stats.keys())
        total_chunks = store.total_chunks()

        col1, col2 = st.columns(2)
        col1.metric("📄 Documents", len(docs))
        col2.metric("🧩 Chunks", total_chunks)

        st.divider()

        if docs:
            st.markdown("**Indexed Documents**")
            for d in docs:
                st.markdown(
                    f"<span class='doc-chip'>📗 {d} &nbsp;({stats[d]} chunks)</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No documents indexed yet. Upload files in Tab 1.")

        st.divider()
        st.caption("Stack: Streamlit · ChromaDB · Gemini · sentence-transformers")
        
        st.divider()
        st.markdown("**📡 Live Sync Status**")
        st.success("🟢 Emergency Report Watcher Active")
        st.caption("Watching: `./emergency_reports` (.pdf, .docx, .txt)")
        
        st.divider()
        if st.button("🔄 Clear Cache & Reload", help="Force-reload all modules (needed after code changes)"):
            st.cache_resource.clear()
            st.rerun()

# ─── State init ──────────────────────────────────────────────────────────────

def init_state(store):
    defaults = {
        "last_query": "",
        "last_retrieved_by_k": {},
        "last_responses_by_k": {},
        "last_results": [],
        "last_contribution": {},
        "eval_history": [],
        "sync_done": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
            
    if "doc_hashes" not in st.session_state:
        # Load known hashes from ChromaDB to persist state across reloads
        try:
            st.session_state["doc_hashes"] = store.get_document_hashes()
        except Exception:
            st.session_state["doc_hashes"] = {}

def handle_live_updates(store, embedder, pipeline=None):
    """
    Check for on-disk changes in the background queue.
    Runs on every streamlit rerun to ensure the app is always 'live'.
    """
    if pipeline is not None:
        outcomes = pipeline.process_pending(startup=not st.session_state.get("sync_done", False))
        st.session_state["sync_done"] = True
        for outcome in outcomes:
            if outcome["status"] == "FAILED":
                st.error(outcome["error_message"])
        return

    # Compatibility RAG-only entry point; the running app always supplies pipeline.
    from file_watcher import get_pending_changes, is_emergency_report
    from document_processor import compute_file_hash, DocumentProcessor as _DP

    pending = get_pending_changes()
    
    # ── Startup/Reload Sync: Scan all files if sync not done this session ──
    sync_needed = not st.session_state.get("sync_done", False)
    
    if sync_needed:
        from config import EMERGENCY_REPORT_DIR
        import pathlib
        
        # Startup sync uses the same isolated directory as the watcher.
        dirs_to_scan = [pathlib.Path(EMERGENCY_REPORT_DIR)]
        
        with st.sidebar:
            with st.spinner("⏳ Syncing local files..."):
                for _dir in dirs_to_scan:
                    if _dir.exists():
                        for _f in _dir.iterdir():
                            if is_emergency_report(_f):
                                pending.append(("sync", str(_f)))
        st.session_state["sync_done"] = True

    if not pending:
        return

    _proc = _DP()
    reindexed_any = False
    
    # Process only unique paths
    unique_paths = list(set(_fpath for _evt, _fpath in pending))
    
    for _fpath in unique_paths:
        import pathlib
        _p = pathlib.Path(_fpath)
        if not is_emergency_report(_p):
            continue
        _doc_id  = _p.stem
        
        try:
            # Check if content actually changed
            _fbytes  = _p.read_bytes()
            _new_hash = compute_file_hash(_fbytes)
            _old_hash = st.session_state["doc_hashes"].get(_doc_id)
            
            if _old_hash == _new_hash:
                continue   # skip if identical
                
            store.delete_document(_doc_id)
            _, _chunks = _proc.process_file(str(_p))
            if _chunks:
                _embs = embedder.embed_batch(_chunks)
                metadata = {
                    "filename": _p.name,
                    "upload_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "num_chunks": len(_chunks),
                    "file_hash": _new_hash  # Persist in vector store
                }
                store.add_document(_doc_id, _chunks, _embs, metadata)
                st.session_state["doc_hashes"][_doc_id] = _new_hash
                st.toast(f"🔄 **{_p.name}** updated/synced — re-indexed", icon="✅")
                reindexed_any = True
        except Exception:
            continue
            
    if reindexed_any:
        st.rerun()

# ─── Tab 1: Document Hub ─────────────────────────────────────────────────────

def tab_document_hub(store, embedder, pipeline=None):
    from document_processor import DocumentProcessor
    from config import EMERGENCY_REPORT_DIR

    st.markdown("<div class='section-header'>📁 Document Hub — Live Ingestion</div>", unsafe_allow_html=True)
    st.markdown(
        "Upload emergency reports as **PDF**, **DOCX**, or **TXT** files. Documents are indexed immediately "
        "and become searchable without restarting the app.",
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Drop files here or click to browse",
        type=["pdf", "docx", "doc", "txt"],
        accept_multiple_files=True,
        help="Supports .pdf, .docx, .txt  |  .doc must be re-saved as .docx",
    )

    # Live updates are now handled globally in main()

    # ── Manual upload ─────────────────────────────────────────────────────────

    if uploaded_files:
        processor = DocumentProcessor()
        from document_processor import compute_file_hash

        for uf in uploaded_files:
            ext    = os.path.splitext(uf.name)[1].lower()
            doc_id = os.path.splitext(uf.name)[0]

            # ── .doc guard ──────────────────────────────────────────────────
            if ext == ".doc":
                st.error(
                    f"❌ **{uf.name}** — Legacy `.doc` format is not supported.\n\n"
                    "Please open it in Microsoft Word → **File → Save As → .docx** "
                    "and re-upload the new file."
                )
                continue

            file_bytes = uf.getbuffer().tobytes()
            if pipeline is not None:
                from emergency_pipeline import PipelineError
                dest_path = os.path.join(EMERGENCY_REPORT_DIR, os.path.basename(uf.name))
                try:
                    # Avoid repeated file events for an unchanged uploader value.
                    from pathlib import Path
                    target = Path(dest_path)
                    if target.resolve().parent != Path(EMERGENCY_REPORT_DIR).resolve():
                        raise PipelineError("Upload destination must remain inside emergency_reports.")
                    if not target.exists() or target.read_bytes() != file_bytes:
                        target.write_bytes(file_bytes)
                    pipeline.process_file(dest_path)
                    st.success(f"Processed **{uf.name}** for emergency intelligence and RAG.")
                except PipelineError as error:
                    st.error(str(error))
                except OSError:
                    st.error("The uploaded report could not be saved locally.")
                continue
            new_hash   = compute_file_hash(file_bytes)

            # ── Change detection ─────────────────────────────────────────────
            old_hash = st.session_state["doc_hashes"].get(doc_id)

            if old_hash == new_hash:
                st.info(
                    f"ℹ️ **{uf.name}** is unchanged since last upload — skipping re-index."
                )
                continue

            # Content changed (or new doc) → delete old chunks, re-index
            if old_hash is not None:
                store.delete_document(doc_id)
                st.toast(f"🔄 Detected changes in **{uf.name}** — re-indexing…")

            dest_path = os.path.join(EMERGENCY_REPORT_DIR, os.path.basename(uf.name))
            with open(dest_path, "wb") as f:
                f.write(file_bytes)

            with st.spinner(f"Indexing **{uf.name}**…"):
                try:
                    _, chunks = processor.process_file(dest_path)
                    if not chunks:
                        st.warning(f"⚠️ No text extracted from **{uf.name}**. Skipped.")
                        continue
                    embeddings = embedder.embed_batch(chunks)
                    metadata = {
                        "filename": uf.name,
                        "upload_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "num_chunks": len(chunks),
                        "file_hash": new_hash, # Persist in vector store
                    }
                    store.add_document(doc_id, chunks, embeddings, metadata)
                    st.session_state["doc_hashes"][doc_id] = new_hash
                    action = "re-indexed" if old_hash else "indexed"
                    st.success(
                        f"✅ **{uf.name}** {action} — {len(chunks)} chunks stored under ID `{doc_id}`"
                    )
                except Exception as e:
                    st.error(f"❌ Error indexing **{uf.name}**: {e}")

    st.divider()

    # Current index
    stats = store.get_collection_stats()
    if stats:
        st.markdown("<div class='section-header'>📚 Current Index</div>", unsafe_allow_html=True)
        df = pd.DataFrame(
            [{"Document ID": k, "Chunks": v} for k, v in stats.items()]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Delete
        with st.expander("🗑 Remove a document from index"):
            del_id = st.selectbox("Select document to remove", list(stats.keys()))
            if st.button("Remove from index", key="del_btn"):
                store.delete_document(del_id)
                st.success(f"Removed **{del_id}** from vector store.")
                st.rerun()
    else:
        st.info("Index is empty. Upload some documents above. ↑")

# ─── Tab 2: Query & Retrieval ────────────────────────────────────────────────

RELEVANCE_THRESHOLD = 0.30   # chunks below this similarity are considered irrelevant

def tab_query(retriever, llm):
    st.markdown("<div class='section-header'>🔍 Query & Retrieval</div>", unsafe_allow_html=True)

    query = st.text_area(
        "Enter your question",
        placeholder="e.g. What is retrieval-augmented generation?",
        height=100,
        key="query_input",
    )

    total_chunks = retriever.store.total_chunks()
    max_k_limit = min(max(1, total_chunks), 20)

    col_k, col_thresh = st.columns([1, 1])
    with col_k:
        if max_k_limit < 2:
            manual_k = 1
            st.info("K is fixed at **1** — upload more documents to enable higher K values.")
        else:
            default_val = min(3, max_k_limit)
            manual_k = st.slider("Number of chunks to retrieve (K)", 1, max_k_limit, default_val)
    with col_thresh:
        relevance_threshold = st.slider(
            "Relevance threshold",
            0.05, 0.80, RELEVANCE_THRESHOLD, 0.05,
            help="Chunks with similarity below this value are considered irrelevant and hidden from the answer.",
        )

    if st.button("🚀 Run Query", key="run_query"):
        if not query.strip():
            st.warning("Please enter a question.")
            return
        if retriever.store.total_chunks() == 0:
            st.error("No documents indexed. Go to Tab 1 and upload some files first.")
            return

        with st.spinner("Retrieving relevant chunks and generating response…"):
            # Retrieve top-K chunks
            all_results = retriever.retrieve(query, k=manual_k)

            # ── Relevance filtering ───────────────────────────────────────────
            relevant_results  = [r for r in all_results if r["similarity"] >= relevance_threshold]
            irrelevant_results = [r for r in all_results if r["similarity"] < relevance_threshold]

            # Only send relevant chunks to the LLM
            chunks_for_llm = [r["text"] for r in relevant_results]

            t_start = time.time()
            if chunks_for_llm:
                response_text, api_latency = llm.generate_response(query, chunks_for_llm)
            else:
                response_text = "⚠️ No relevant chunks found for your query. Try a different question or upload more relevant documents."
                api_latency = 0.0
            t_end = time.time()

            wall_latency = round(t_end - t_start, 3)
            latency = max(api_latency, wall_latency, 0.001) if chunks_for_llm else 0.0

            # Cache for other tabs
            retrieved_by_k  = {manual_k: all_results}
            responses_by_k  = {manual_k: {"response": response_text, "latency": latency, "chunks": chunks_for_llm}}
            from evaluator import Evaluator
            _eval = Evaluator(retriever.embedder)
            contribution = _eval.compute_doc_contribution(relevant_results)

            st.session_state["last_query"]          = query
            st.session_state["last_retrieved_by_k"] = retrieved_by_k
            st.session_state["last_responses_by_k"] = responses_by_k
            st.session_state["last_contribution"]   = contribution

        # ── Display response ──────────────────────────────────────────────────
        st.markdown(f"### 💬 Response (K={manual_k})")

        # Use st.markdown so paragraph breaks render properly
        st.markdown(
            f"""
            <div class='response-box'>
            {"<br><br>".join(p.strip() for p in response_text.split("\n\n") if p.strip()) or response_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("⏱ Latency",          f"{latency:.3f}s")
        m2.metric("✅ Relevant Chunks",  len(relevant_results))
        m3.metric("❌ Irrelevant Chunks", len(irrelevant_results))
        m4.metric("🗂 Unique Docs",       len({r["doc_id"] for r in relevant_results}))

        # ── Relevant chunks ───────────────────────────────────────────────────
        if relevant_results:
            with st.expander(f"📋 Relevant chunks used for answer ({len(relevant_results)})"):
                for i, r in enumerate(relevant_results, 1):
                    st.markdown(
                        f"**Chunk {i}** | doc: `{r['doc_id']}` | "
                        f"similarity: `{r['similarity']:.4f}` ✅"
                    )
                    st.markdown(
                        f"<div class='response-box' style='font-size:0.85rem'>"
                        f"{r['text'][:500]}{'…' if len(r['text'])>500 else ''}</div>",
                        unsafe_allow_html=True,
                    )
                    st.divider()

        # ── Irrelevant chunks ─────────────────────────────────────────────────
        if irrelevant_results:
            with st.expander(
                f"⚠️ {len(irrelevant_results)} chunk(s) were below the relevance threshold "
                f"({relevance_threshold:.2f}) and excluded from the answer"
            ):
                st.info(
                    f"These chunks had similarity scores below **{relevance_threshold:.2f}** "
                    f"and were considered irrelevant to your query. They were NOT sent to the LLM."
                )
                for i, r in enumerate(irrelevant_results, 1):
                    st.markdown(
                        f"**Chunk {i}** | doc: `{r['doc_id']}` | "
                        f"similarity: `{r['similarity']:.4f}` ❌"
                    )
                    st.markdown(
                        f"<div class='response-box' style='font-size:0.85rem;opacity:0.6'>"
                        f"{r['text'][:300]}{'…' if len(r['text'])>300 else ''}</div>",
                        unsafe_allow_html=True,
                    )


# ─── Tab 3: Evaluation Lab ───────────────────────────────────────────────────

def tab_evaluation(retriever, llm, evaluator):
    st.markdown("<div class='section-header'>🧪 Evaluation Lab — Retrieval Depth Sensitivity</div>", unsafe_allow_html=True)
    st.markdown(
        "Evaluate how retrieval depth **K = 1 … 5** affects response quality. "
        "Provide ground truth relevant doc IDs for retrieval metric computation.",
        unsafe_allow_html=True,
    )

    eval_query = st.text_area(
        "Evaluation Query",
        value=st.session_state.get("last_query", ""),
        placeholder="Enter the query to evaluate…",
        height=80,
        key="eval_query",
    )

    all_docs = retriever.store.list_documents()
    relevant_ids = st.multiselect(
        "Ground-truth relevant document(s)",
        options=all_docs,
        help="Select docs that should be retrieved for this query.",
    )

    col_l, col_r = st.columns(2)
    use_cached = col_l.checkbox(
        "Reuse last query's retrieved results & responses (faster)",
        value=bool(st.session_state.get("last_retrieved_by_k")),
    )

    total_chunks = retriever.store.total_chunks()
    max_k_eval = 5  # Fixed to 5 as per user request
    k_list = list(range(1, max_k_eval + 1))

    if col_r.button(f"▶ Run Evaluation (K=1..{max_k_eval})", key="run_eval"):
        if not eval_query.strip():
            st.warning("Enter a query first.")
            return
        if retriever.store.total_chunks() == 0:
            st.error("No documents indexed.")
            return

        if use_cached and st.session_state.get("last_retrieved_by_k") and eval_query == st.session_state.get("last_query"):
            retrieved_by_k = st.session_state["last_retrieved_by_k"]
            responses_by_k = st.session_state["last_responses_by_k"]
            
            # Ensure all K are present - if Tab 2 only ran K=3, we need 1, 2, 4, 5
            missing_k = [k for k in k_list if k not in retrieved_by_k]
            if missing_k:
                with st.spinner(f"Filling missing K values {missing_k}…"):
                    m_retrieved = retriever.retrieve_all_k(eval_query, k_values=missing_k)
                    m_responses = llm.generate_responses_for_all_k(eval_query, m_retrieved)
                    retrieved_by_k.update(m_retrieved)
                    responses_by_k.update(m_responses)
            st.info("ℹ️ Results complete (cached + filled missing K).")
        else:
            with st.spinner(f"Retrieving for K=1..{max_k_eval} and generating responses…"):
                retrieved_by_k = retriever.retrieve_all_k(eval_query, k_values=k_list)
                t_eval_start = time.time()
                responses_by_k = llm.generate_responses_for_all_k(eval_query, retrieved_by_k)
                t_eval_end = time.time()

                # App-level latency override (cache-immune)
                total_wall = t_eval_end - t_eval_start
                num_real_calls = sum(1 for k in responses_by_k if responses_by_k[k].get("chunks"))
                per_k_wall = total_wall / max(num_real_calls, 1)
                for k in responses_by_k:
                    if responses_by_k[k].get("latency", 0.0) < 0.01:
                        responses_by_k[k]["latency"] = round(per_k_wall, 3)

                st.session_state["last_query"] = eval_query
                st.session_state["last_retrieved_by_k"] = retrieved_by_k
                st.session_state["last_responses_by_k"] = responses_by_k

        with st.spinner("Computing metrics…"):
            results = evaluator.run_full_evaluation(
                query=eval_query,
                retrieved_by_k=retrieved_by_k,
                responses_by_k=responses_by_k,
                relevant_doc_ids=relevant_ids,
            )
            evaluator.save_results(results)
            st.session_state["last_results"] = results
            st.session_state["eval_relevant_ids"] = relevant_ids  # Store for dashboard logic

            # Contribution: ONLY from the ground-truth selected documents
            all_contrib = evaluator.compute_doc_contribution(
                retrieved_by_k.get(5, [])
            )
            # Filter to only relevant_ids and re-normalize to 100%
            filtered_contrib = {k: v for k, v in all_contrib.items() if k in relevant_ids}
            total_f = sum(filtered_contrib.values())
            if total_f > 0:
                filtered_contrib = {k: round(v/total_f * 100, 2) for k, v in filtered_contrib.items()}
            
            st.session_state["last_contribution"] = filtered_contrib

        st.success("✅ Evaluation complete! Results saved to `evaluation_results.json`")

        # ── Results Table ──────────────────────────────────────────────────
        st.markdown("### 📋 Metrics Summary")
        rows = []
        for r in results:
            rows.append({
                "K":             r["K"],
                "Hit@K":         r["hit_at_k"],
                "Recall@K":      r["recall_at_k"],
                "Precision@K":   r["precision_at_k"],
                "MRR":           r["mrr"],
                "Grounding":     r["grounding_score"],
                "Hallucination": r["hallucination_rate"],
                "Latency (s)":   r["latency"],
                "Quality Score": r["quality_score"],
            })
        df = pd.DataFrame(rows)
        st.dataframe(
            df.style.background_gradient(subset=["Recall@K","Grounding","Hallucination","Quality Score"], 
                                         cmap="RdYlGn_r", # Red to Green reversed for Hallucination
                                         vmin=0, vmax=1),
            use_container_width=True,
            hide_index=True,
        )
        
        # ── Optimal K Recommendation ───────────────────────────────────────
        best_r = max(results, key=lambda x: x["quality_score"])
        st.markdown(
            f"""
            <div style='background: rgba(34, 197, 94, 0.1); border: 1px solid #22C55E; padding: 1.5rem; border-radius: 12px; margin: 1.5rem 0'>
                <h4 style='color: #4ADE80; margin-top: 0'>🎯 Recommended Optimal K: <b>{best_r['K']}</b></h4>
                <p style='color: #94A3B8; font-size: 0.95rem; margin-bottom: 0'>
                    Based on the analysis, <b>K={best_r['K']}</b> provides the best balance between <b>Discovery (Recall)</b>, 
                    <b>Accuracy (Grounding)</b>, and <b>Efficiency (Latency)</b> with a Quality Score of <b>{best_r['quality_score']:.3f}</b>.
                    Setting K higher may introduce noise, while K lower might miss important context.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ── Ground Truth Context ───────────────────────────────────────────
        if relevant_ids:
            with st.expander("📖 Ground Truth Document Context (Actual Source Text)"):
                for rid in relevant_ids:
                    gt_results = retriever.store.collection.get(where={"doc_id": rid}, limit=1)
                    if gt_results and gt_results["documents"]:
                        st.markdown(f"**Source Document: `{rid}`**")
                        st.markdown(
                            f"<div class='response-box' style='font-size:0.85rem; background: rgba(59, 130, 246, 0.05); border-left: 4px solid #3B82F6'>"
                            f"{gt_results['documents'][0][:600]}...</div>",
                            unsafe_allow_html=True
                        )
                        st.divider()

        # ── Per-K Responses ────────────────────────────────────────────────
        st.markdown("### 📝 Responses per K")
        for r in results:
            with st.expander(f"K={r['K']}  |  Quality: {r['quality_score']:.3f}  |  Latency: {r['latency']:.3f}s"):
                resp_text = r['response']
                st.markdown(
                    f"<div class='response-box'>"
                    f"{'<br><br>'.join(p.strip() for p in resp_text.split(chr(10)+chr(10)) if p.strip()) or resp_text}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Docs: {r['retrieved_docs']} | "
                    f"Sim: {r['similarity_scores']} | "
                    f"Grounding: {r['grounding_score']}"
                )

        # Download JSON
        st.download_button(
            "⬇ Download evaluation_results.json",
            data=json.dumps(results, indent=2),
            file_name="evaluation_results.json",
            mime="application/json",
        )

    # Show history
    history = evaluator.load_results()
    if history:
        with st.expander(f"📂 Past evaluation runs ({len(history)} records)"):
            df_h = pd.DataFrame([
                {
                    "Query": r.get("query", "")[:60],
                    "K": r["K"],
                    "Quality": r.get("quality_score", 0),
                    "Grounding": r.get("grounding_score", 0),
                    "Hallucination": r.get("hallucination_rate", 0),
                }
                for r in history
            ])
            st.dataframe(df_h, use_container_width=True, hide_index=True)

# ─── Tab 4: Dashboard ────────────────────────────────────────────────────────

def tab_dashboard(visualizer):
    st.markdown("<div class='section-header'>📊 Visualization Dashboard</div>", unsafe_allow_html=True)

    results = st.session_state.get("last_results", [])
    contribution = st.session_state.get("last_contribution", {})

    if not results:
        st.info(
            "Run an evaluation in **Tab 3** first, then return here to see all charts."
        )
        # Try loading from saved file
        from evaluator import Evaluator
        from embedder import Embedder
        _eval = Evaluator(Embedder.__new__(Embedder))
        saved = _eval.load_results()
        if saved:
            # Group by most recent query's K values (last 5 records)
            results = saved[-5:]
            st.info(f"Showing charts from last saved evaluation ({len(results)} K values).")

    if results:
        charts = visualizer.generate_all_charts(results, contribution)

        st.plotly_chart(charts["retrieval_metrics"],    use_container_width=True)
        st.plotly_chart(charts["response_quality"],     use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(charts["recall_vs_hallucination"],   use_container_width=True)
        with col2:
            st.plotly_chart(charts["grounding_vs_hallucination"], use_container_width=True)

        st.plotly_chart(charts["quality_score"],        use_container_width=True)
        
        # Only show contribution chart if >= 2 ground truth docs are selected
        eval_relevant_ids = st.session_state.get("eval_relevant_ids", [])
        if len(eval_relevant_ids) >= 2:
            st.plotly_chart(charts["doc_contribution"],     use_container_width=True)
        else:
            st.info("💡 Document Contribution graph is hidden. Select at least 2 documents in the Evaluation tab to enable it.")

# ─── Tab 5: Document Contribution ────────────────────────────────────────────

def tab_contribution(visualizer, evaluator, retriever):
    st.markdown("<div class='section-header'>📄 Document Contribution Analysis</div>", unsafe_allow_html=True)
    st.markdown(
        "Analyse which documents contribute most to answering a query. "
        "Contribution is computed from similarity-weighted chunk scores.",
        unsafe_allow_html=True,
    )

    contrib_query = st.text_input(
        "Query for contribution analysis",
        value=st.session_state.get("last_query", ""),
        placeholder="Enter a query…",
        key="contrib_query",
    )
    contrib_k = st.slider("Retrieve top K chunks", 1, 5, 5, key="contrib_k")

    if st.button("📊 Analyse Contribution", key="run_contrib"):
        if not contrib_query.strip():
            st.warning("Enter a query.")
            return
        if retriever.store.total_chunks() == 0:
            st.error("No documents indexed.")
            return

        with st.spinner("Computing contribution…"):
            results = retriever.retrieve(contrib_query, k=contrib_k)
            contribution = evaluator.compute_doc_contribution(results)
            st.session_state["last_contribution"] = contribution

        if not contribution:
            st.warning("No contributions found — try a different query or add more docs.")
            return

        # Show warning if user wants contribution but hasn't selected enough docs in Eval tab
        # or just show it here but keep the dashboard conditional as requested
        st.markdown(f"**Query**: *{contrib_query}*")

        fig = visualizer.plot_doc_contribution(
            contribution, title=f"Source Document Contribution to Answer"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Retrieved chunks table
        with st.expander("📋 Retrieved Chunks Detail"):
            rows = [
                {
                    "Rank": i + 1,
                    "Doc ID": r["doc_id"],
                    "Similarity": round(r["similarity"], 4),
                    "Chunk Preview": r["text"][:120] + "…",
                }
                for i, r in enumerate(results)
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    from config import EMERGENCY_DATA_DIR
    from crisislens_theme import apply_theme
    from crisislens_ui import (upload_reports, sidebar, command_center, report_analysis,
                               emergency_rag, system_evaluation)
    apply_theme()
    st.title("CrisisLens-AI")
    st.markdown("**Real-Time NLP Emergency Intelligence & Decision Support**")
    st.caption("Understand the incident. Identify the need. Inspect the priority.")
    try:
        embedder, store, retriever, llm, evaluator, visualizer, pipeline = load_resources()
        init_state(store)
        handle_live_updates(store, embedder, pipeline)
    except Exception:
        st.error("Emergency services could not start. Check local model availability and report-store access, then restart.")
        st.stop()
    tabs = st.tabs(["Command Center", "Report Analysis", "Emergency RAG", "System Evaluation"])
    with tabs[0]:
        upload_reports(pipeline)
        try:
            reports = pipeline.get_reports_sorted_by_priority()
            records = pipeline.get_processing_records()
        except Exception:
            st.error("Persisted reports could not be read. Check the report store before continuing.")
            st.stop()
        command_center(pipeline, reports, records)
    sidebar(pipeline, len(reports))
    with tabs[1]:
        report_analysis(pipeline, reports)
    with tabs[2]:
        emergency_rag(retriever, llm)
    with tabs[3]:
        system_evaluation(EMERGENCY_DATA_DIR)


if __name__ == "__main__":
    main()

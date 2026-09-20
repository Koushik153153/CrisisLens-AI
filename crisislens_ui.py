"""Presentation of persisted intelligence; no model construction or NLP scoring."""

from collections import Counter
from html import escape
import json
from pathlib import Path

import streamlit as st
from crisislens_theme import (ACCENTS, display_label, human_text, concise, finite,
                              chips, priority_badge, card, meter)

DISCLAIMER = "EAPS is a prototype NLP-derived decision-support score and is not an official emergency triage or dispatch standard."


def number(value, digits=1):
    return f"{value:.{digits}f}" if finite(value) else "Not identified"


def joined(values, canonical=False):
    return " · ".join(display_label(v) if canonical else str(v) for v in values) if values else "Not identified"


def badge(level):
    st.markdown(priority_badge(level), unsafe_allow_html=True)


def affected(report):
    count = report.get("affected_count")
    if not finite(count):
        return "Not identified"
    prefix = "About " if report.get("affected_count_approximate") else ""
    return f"{prefix}{count:g} {display_label(report.get('affected_unit')) if report.get('affected_unit') else '(unit not identified)'}"


def report_label(report):
    return f"{report.get('source_file') or 'Report'} — {display_label(report.get('incident_type'))} — {display_label(report.get('priority_level'))}"


def hero(report):
    level = report.get("priority_level")
    st.markdown(f'<section class="cl-hero" style="--accent:{ACCENTS.get(level, "#76d8e7")}" aria-label="Emergency intelligence summary">'
                f'<div class="cl-hero-top">{priority_badge(level)}<div class="cl-score">{number(report.get("priority_score"))} <small>EAPS / 100</small></div></div>'
                f'<h2>{escape(display_label(report.get("incident_type")))}</h2>'
                f'<div class="cl-location">{escape(joined(report.get("locations")))}</div>'
                f'<div class="cl-people"><strong>{escape(affected(report))} affected</strong>'
                f'<span>Vulnerable: {escape(joined(report.get("vulnerable_groups"), True))}</span></div>'
                f'<div class="cl-eyebrow">Required response</div>{chips(report.get("resource_needs"), True)}</section>', unsafe_allow_html=True)


def count_chart(counts, key, priority=False):
    """A compact horizontal count chart; includes unknown priority when present."""
    import plotly.graph_objects as go
    if not counts:
        return
    labels = [display_label(k) for k in counts]
    colors = [ACCENTS.get(k, "#76d8e7") if priority else "#76d8e7" for k in counts]
    figure = go.Figure(go.Bar(x=list(counts.values()), y=labels, orientation="h", marker_color=colors,
                             text=list(counts.values()), textposition="auto", hovertemplate="%{y}: %{x} reports<extra></extra>"))
    figure.update_layout(height=max(155, len(counts) * 32 + 40), margin=dict(l=0,r=15,t=5,b=10),
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         font=dict(color="#dce8f6", size=13), showlegend=False,
                         xaxis=dict(showgrid=False, visible=False, rangemode="tozero"),
                         yaxis=dict(autorange="reversed", showgrid=False, ticks=""))
    st.plotly_chart(figure, config={"displayModeBar": False}, key=key)


def resource_counts(reports):
    """One count per report and resource, including retained report versions."""
    return Counter(resource for report in reports for resource in set(report.get("resource_needs") or []))


def save_upload(pipeline, name, content):
    """Save only a direct supported file, then delegate all processing to Step 7."""
    from config import EMERGENCY_REPORT_EXTENSIONS
    from emergency_pipeline import PipelineError
    directory = pipeline.report_dir.resolve()
    target = directory / Path(name).name
    if target.resolve().parent != directory or target.suffix.lower() not in EMERGENCY_REPORT_EXTENSIONS:
        raise PipelineError("Choose a PDF, DOCX or TXT file inside the report directory.")
    directory.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.read_bytes() != content:
        target.write_bytes(content)
    return pipeline.process_file(str(target))


def upload_reports(pipeline):
    with st.expander("Add emergency reports"):
        files = st.file_uploader("Upload PDF, DOCX or TXT reports", type=["pdf", "docx", "txt"],
                                 accept_multiple_files=True, key="emergency_uploads")
        if st.button("Process uploaded reports", disabled=not files):
            from emergency_pipeline import PipelineError
            for file in files:
                try:
                    with st.spinner("Processing emergency report…"):
                        report = save_upload(pipeline, file.name, file.getvalue())
                    st.session_state["selected_report_id"] = report.report_id
                    st.success(f"Processed {file.name}. Open Report Analysis to inspect the result.")
                except PipelineError as error:
                    st.warning(str(error))
                except OSError:
                    st.error("The uploaded report could not be saved locally.")
        st.caption("PDF · DOCX · TXT. Analysis and indexing statuses are shown separately.")


def sidebar(pipeline, count):
    from file_watcher import watcher_is_active
    with st.sidebar:
        st.subheader("Report intake")
        st.write(f"Emergency Report Watcher: {'Active' if watcher_is_active() else 'Inactive'}")
        st.caption(f"Watching: {pipeline.report_dir}")
        st.caption("Supported: PDF · DOCX · TXT")
        st.metric("Processed reports", count)
        st.button("Refresh reports")
        st.caption("Queued file changes are processed on refresh or another app interaction.")


def command_center(pipeline, reports, records):
    counts = resource_counts(reports)
    critical = sum(r.get("priority_level") == "CRITICAL" for r in reports)
    metrics = [("Total Reports", len(reports)), ("Critical", critical),
               ("High Priority", sum(r.get("priority_level") == "HIGH" for r in reports)),
               ("Resource Needs", sum(counts.values()))]
    for column, (label, value) in zip(st.columns(4), metrics):
        with column:
            card(label, value, accent=ACCENTS["CRITICAL"] if label == "Critical" and value else None)
    with st.expander("About these counts"):
        st.caption("Counts include all successfully analyzed, persisted report versions. Resource Needs counts report–resource associations, not physical quantities or confirmed unresolved incidents.")
    if reports:
        distribution = Counter(r.get("priority_level") or "NOT_IDENTIFIED" for r in reports)
        ordered = {level: distribution[level] for level in ("CRITICAL", "HIGH", "MEDIUM", "LOW")}
        if distribution.get("NOT_IDENTIFIED"):
            ordered["NOT_IDENTIFIED"] = distribution["NOT_IDENTIFIED"]
        st.caption("PRIORITY DISTRIBUTION · REPORTS")
        count_chart(ordered, "live_priority", priority=True)
    st.subheader("Emergency Priority Queue")
    if not reports:
        st.markdown('<div class="cl-empty"><div class="cl-eyebrow">No emergency reports yet</div>'
                    '<p>Upload a report or place a PDF, DOCX or TXT file in the monitored folder.</p></div>', unsafe_allow_html=True)
    for report in reports:
        with st.container(border=True):
            hero(report)
            source, action = st.columns([3, 1])
            source.caption(f"Source: {report.get('source_file') or 'Not identified'}")
            if action.button("View Intelligence", key=f"select_{report['report_id']}"):
                st.session_state["selected_report_id"] = report["report_id"]
                st.toast("Selected. Open Report Analysis.")
    resource_column, incident_column = st.columns(2)
    with resource_column:
        st.subheader("Resource overview")
        st.caption("Reports requiring resource · not physical quantities")
        if counts:
            count_chart(dict(counts.most_common()), "live_resources")
        else:
            st.info("No resource needs recorded.")
    with incident_column:
        if len(reports) > 1:
            st.subheader("Incident overview")
            count_chart(Counter(r.get("incident_type") or "NOT_IDENTIFIED" for r in reports), "live_incidents")
    if records:
        with st.expander("System processing status"):
            status = lambda value: {"SUCCESS": "Successful", "FAILED": "Failed", "NOT_REQUESTED": "Not requested"}.get(value, "Not identified")
            st.dataframe([{"Source": r.get("source_file", ""),
                           "Analysis": status(r.get("analysis_status")), "RAG Indexing": status(r.get("rag_status"))}
                          for r in records], hide_index=True)

def evidence(items, label):
    st.write(f"**{label}**")
    if isinstance(items, str):
        items = [items]
    for item in items or []:
        st.write(item)
    if not items:
        st.caption("Not recorded.")


def report_analysis(pipeline, reports):
    st.subheader("Emergency Intelligence Brief")
    if not reports:
        st.info("Select a processed report after uploading one in Command Center.")
        return
    by_id = {r["report_id"]: r for r in reports}
    if st.session_state.get("selected_report_id") not in by_id:
        st.session_state["selected_report_id"] = next(iter(by_id))
    # Duplicate versions retain distinct options without exposing hashes.
    labels = {}
    for index, report in enumerate(reports, 1):
        base = report_label(report)
        repeated = sum(report_label(r) == base for r in reports) > 1
        labels[report["report_id"]] = base + (f" · Record {index}" if repeated else "")
    selected = st.selectbox("Select a processed report", list(by_id), key="selected_report_id",
                            format_func=lambda key: labels[key])
    report = pipeline.get_report(selected) or by_id[selected]
    hero(report)
    st.markdown("#### At a glance")
    values = [("Incident", display_label(report.get("incident_type"))),
              ("Location", joined(report.get("locations"))), ("Affected", affected(report)),
              ("Vulnerable", joined(report.get("vulnerable_groups"), True)),
              ("Resource needed", joined(report.get("resource_needs"), True))]
    for column, (label, value) in zip(st.columns(5), values):
        with column:
            card(label, value)
    assessment = report.get("assessment") or {}
    st.markdown("#### Intelligence flow")
    flow = [("Source", "Report"), ("Incident", display_label(report.get("incident_type"))),
            ("Affected", affected(report)), ("Help needed", joined(report.get("resource_needs"), True)),
            ("Urgency", display_label((assessment.get("urgency") or {}).get("label") or report.get("urgency"))),
            ("Priority", display_label(report.get("priority_level")))]
    nodes = [f'<div class="cl-node"><div class="cl-eyebrow">{escape(label)}</div>{escape(value)}</div>' for label, value in flow]
    st.markdown('<div class="cl-flow">' + '<span class="cl-arrow" aria-hidden="true">→</span>'.join(nodes) + '</div>', unsafe_allow_html=True)
    st.caption("Stored report findings, shown in sequence.")

    st.subheader("Emergency priority")
    explanation = report.get("priority_explanation") or {}
    components = explanation.get("component_scores") or {}
    score_column, contribution_column = st.columns([1, 2])
    with score_column:
        st.metric("EAPS / 100", number(report.get("priority_score")))
        badge(report.get("priority_level"))
        meter("Overall priority", report.get("priority_score"), 100, ACCENTS.get(report.get("priority_level"), "#76d8e7"))
    with contribution_column:
        for name in ("urgency", "severity", "actionability", "context"):
            component = components.get(name) or {}
            weight = component.get("weight")
            maximum = weight * 100 if finite(weight) else None
            meter("Situation context" if name == "context" else display_label(name), component.get("points"), maximum)
    st.markdown("#### Why this report is prioritized")
    # Present only stored labels/reasons. Calculation-heavy strings stay in technical details.
    readable = [reason for reason in (report.get("priority_reasons") or explanation.get("reasons") or [])
                if not any(token in str(reason).lower() for token in ("=", "points", "weight", "factor", "context_share"))]
    if readable:
        for reason in readable[:4]:
            st.write("✓ " + concise(reason))
    else:
        for name in ("urgency", "severity", "actionability"):
            component = components.get(name) or {}
            dimension = assessment.get(name) or {}
            label = dimension.get("label") or component.get("label") or report.get(name)
            if label:
                st.write(f"✓ {display_label(label)} {name}")
                if dimension.get("reasons"):
                    st.caption(concise(dimension["reasons"][0]))
        if not components:
            st.caption("Priority explanation not identified.")
    with st.expander("Technical EAPS calculation"):
        st.caption("Exact stored calculation; display bars do not recompute EAPS.")
        st.json(explanation)
        evidence(report.get("priority_reasons") or explanation.get("reasons"), "Stored technical reasons")
    st.caption(DISCLAIMER)

    st.subheader("Response resources")
    predictions = report.get("resource_predictions") or []
    columns = st.columns(2)
    for index, prediction in enumerate(predictions):
        with columns[index % 2], st.container(border=True):
            st.markdown(chips([prediction.get("resource")], True), unsafe_allow_html=True)
            badge(prediction.get("inference_type"))
            st.write(concise(prediction.get("evidence")) or "Evidence not identified.")
            st.caption(concise(prediction.get("reason")) or "Reason not identified.")
            with st.expander("Technical resource details"):
                st.caption("Rule confidence / semantic similarity: " + number(prediction.get("confidence"), 3) + " · not a probability")
                evidence(prediction.get("evidence"), "Full stored evidence")
                evidence(prediction.get("reason"), "Full stored reason")
    if not predictions:
        st.info("No stored resource evidence available.")
    st.subheader("Emergency assessment")
    descriptions = {"urgency": "How quickly help is needed", "severity": "How serious the situation is",
                    "actionability": "How clearly responders can act"}
    for column, (name, description) in zip(st.columns(3), descriptions.items()):
        with column, st.container(border=True):
            dimension = assessment.get(name) or {}
            label = dimension.get("label") or report.get(name)
            card(display_label(name), display_label(label), description, ACCENTS.get(label))
            levels = ("LOW", "MEDIUM", "HIGH") if name == "actionability" else ("LOW", "MEDIUM", "HIGH", "CRITICAL")
            count = levels.index(label) + 1 if label in levels else 0
            indicator = "▰" * count + "▱" * (len(levels) - count)
            st.caption(indicator + " · Ordinal level" if label in levels else "Level not identified")
            if dimension.get("reasons"):
                st.write(concise(dimension["reasons"][0]))
    with st.expander("Why did CrisisLens assess this?"):
        for name in descriptions:
            dimension = assessment.get(name) or {}
            st.markdown(f"**{display_label(name)} evidence**")
            evidence(dimension.get("evidence"), "Supporting evidence")
            evidence(dimension.get("reasons"), "Deterministic reasons")
            st.caption("Rule confidence / semantic similarity: " + number(dimension.get("confidence"), 3) + " · not a probability")
            if dimension.get("semantic_scores"):
                st.json(dimension["semantic_scores"])
    with st.expander("Technical incident details"):
        st.caption("Semantic similarity: " + number(report.get("incident_confidence"), 3) + " · not a probability")
        if report.get("incident_category_scores"):
            st.json(report["incident_category_scores"])
        st.caption("Report ID: " + selected)
    st.caption("SOURCE REPORT · " + (report.get("source_file") or "Not identified"))
    with st.expander("View original emergency report"):
        st.text(report.get("raw_text") or "Original text not identified.")

def emergency_rag(retriever, llm):
    st.subheader("Emergency Evidence Query")
    st.caption("Ask questions across indexed emergency reports and inspect the supporting source evidence.")
    if llm is None:
        st.info("Gemini is unavailable. Source retrieval remains available; emergency analysis is independent.")
    if retriever is None:
        st.warning("RAG retrieval is unavailable. Persisted emergency analysis remains available.")
        return
    try:
        total = retriever.store.total_chunks()
    except Exception:
        st.warning("The RAG index is unavailable. Emergency analysis is independent.")
        return
    query = st.text_area("Question across emergency reports", key="query_input")
    maximum = min(max(1, total), 20)
    with st.expander("Retrieval settings"):
        k = st.slider("Number of chunks to retrieve (K)", 1, maximum, min(3, maximum)) if maximum > 1 else 1
        threshold = st.slider("Relevance threshold", .05, .80, .30, .05)
    if not total:
        st.info("No emergency reports indexed yet. Upload a report in Command Center.")
    if st.button("Run Query", key="run_query"):
        st.session_state.pop("emergency_answer", None)
        if not query.strip():
            st.warning("Enter a question first.")
        elif total:
            try:
                results = retriever.retrieve(query, k=k)
                relevant = [r for r in results if r["similarity"] >= threshold]
                answer = "No chunks met the relevance threshold."
                failed = False
                if relevant:
                    try:
                        if llm is None:
                            raise RuntimeError()
                        with st.spinner("Generating grounded response…"):
                            answer, _ = llm.generate_response(query, [r["text"] for r in relevant])
                    except Exception:
                        failed = True
                        answer = "Gemini is unavailable or rate-limited. Retrieved evidence is shown below; emergency analysis remains available."
                st.session_state["emergency_answer"] = (query, answer, results, threshold, failed)
            except Exception:
                st.warning("Retrieval failed. Try again after checking the RAG index; emergency analysis remains available.")
    if "emergency_answer" in st.session_state:
        question, answer, results, used_threshold, failed = st.session_state["emergency_answer"]
        st.subheader("Grounded answer")
        st.caption("Question: " + question)
        (st.warning if failed else st.write)(answer)
        for index, result in enumerate(results, 1):
            included = result["similarity"] >= used_threshold
            metadata = result.get("metadata") or {}
            filename = metadata.get("filename") or (result.get("doc_id") if not str(result.get("doc_id", "")).startswith("report-") else None) or "Emergency report"
            with st.expander(f"Source {index}: {filename} · {'Relevant' if included else 'Below threshold'}"):
                st.text(result.get("text", ""))
                st.caption(f"Semantic similarity: {result['similarity']:.4f} · not a probability")


def read_evaluation(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def at(data, *keys):
    for key in keys:
        data = data.get(key) if isinstance(data, dict) else None
    return data


def evaluation_table(rows):
    columns = st.columns(min(4, max(1, len(rows))))
    for index, (label, value) in enumerate(rows):
        with columns[index % len(columns)]:
            card(human_text(label), f"{value * 100:.2f}%" if finite(value) else "Not identified")


def validation_label(value):
    return "Passed" if value is True else "Failed" if value is False else "Not identified"


def system_evaluation(directory):
    st.subheader("Controlled Synthetic Development Corpus")
    st.caption("Controlled synthetic development-corpus results · not held-out or real-world validated performance.")
    sections = [("Incident understanding", "step3_evaluation.json"),
                ("Resource intelligence", "step4_resource_evaluation.json"),
                ("Emergency assessment", "step5_assessment_evaluation.json"),
                ("EAPS validation", "step6_priority_evaluation.json"),
                ("Pipeline validation", "step7_pipeline_validation.json")]
    for index, (title, filename) in enumerate(sections, 3):
        st.markdown(f"### {title}")
        data = read_evaluation(Path(directory) / filename)
        if index == 6:
            st.caption("No gold EAPS labels exist. Structural checks, not accuracy.")
        if data is None:
            st.info(f"Results unavailable: {filename} is missing or malformed.")
            continue
        if index == 3:
            evaluation_table([("Incident classification · accuracy", at(data, "incident", "accuracy")),
                              ("Locations · F1", at(data, "locations", "f1")),
                              ("Affected count · exact match", at(data, "affected_count", "exact_match_accuracy")),
                              ("Vulnerable groups · F1", at(data, "vulnerable_groups", "f1"))])
            with st.expander("Detailed extraction metrics"):
                evaluation_table([(f"{display_label(group)} · {metric}", at(data, group, metric))
                                  for group in ("locations", "vulnerable_groups") for metric in ("precision", "recall")])
        elif index == 4:
            evaluation_table([(f"Micro {metric.upper() if metric == 'f1' else metric}", at(data, "micro", metric))
                              for metric in ("precision", "recall", "f1")] + [("Exact-set accuracy", data.get("exact_set_accuracy"))])
        elif index == 5:
            for column, dimension in zip(st.columns(3), ("urgency", "severity", "actionability")):
                with column:
                    values = at(data, "metrics", dimension) or {}
                    accuracy = at(values, "accuracy")
                    f1 = at(values, "macro_f1")
                    card(display_label(dimension), f"{accuracy * 100:.2f}%" if finite(accuracy) else "Not identified",
                         "Accuracy · Macro F1: " + (f"{f1:.4f}" if finite(f1) else "Not identified"))
        elif index == 6:
            checks = at(data, "properties", "checks") or {}
            rows = [("Bounds", at(checks, "bounds")), ("Reproducibility", at(checks, "repeatability")),
                    ("Component consistency", at(checks, "component_sum"))]
            monotonic = [at(checks, name + "_monotonic") for name in ("urgency", "severity", "actionability")]
            rows.append(("Monotonicity", all(monotonic) if all(isinstance(v, bool) for v in monotonic) else None))
            for column, (label, passed) in zip(st.columns(4), rows):
                with column:
                    card(label, validation_label(passed), accent=ACCENTS["LOW"] if passed is True else None)
            distribution = at(data, "distribution", "counts_by_level")
            if isinstance(distribution, dict) and distribution and all(finite(v) for v in distribution.values()):
                st.caption("Development-corpus priority distribution")
                count_chart(distribution, "evaluation_priority", priority=True)
            with st.expander("All structural checks"):
                if isinstance(checks, dict):
                    for label, value in checks.items():
                        st.write(display_label(label) + ": " + validation_label(value))
        else:
            st.write("Step-7 validation: " + validation_label(data.get("passed")))
            st.caption("System/pipeline validation, not model accuracy.")
            for column, name in zip(st.columns(4), ("reports_processed", "persisted_after_reload", "rag_documents", "rag_chunks")):
                with column:
                    card(display_label(name).replace("Rag", "RAG"), number(data.get(name), 0))
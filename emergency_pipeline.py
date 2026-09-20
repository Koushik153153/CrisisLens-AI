"""One whole-report analysis plus independent RAG ingestion; no NLP rules here."""

import hashlib
import json
import os
import tempfile
import threading
from contextlib import suppress
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from config import EMERGENCY_REPORT_DIR, EMERGENCY_REPORT_EXTENSIONS, PROCESSED_REPORTS_FILE
from document_processor import DocumentProcessor
from emergency_schema import (
    EmergencyReport, ResourcePrediction, AssessmentResult, DimensionAssessment,
    PriorityResult, PriorityComponent,
)

_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


class PipelineError(RuntimeError):
    """Public errors deliberately exclude underlying exception text/secrets."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _restore(data):
    values = deepcopy(data)
    values["resource_predictions"] = [ResourcePrediction(**p) for p in values.get("resource_predictions", [])]
    if values.get("assessment"):
        values["assessment"] = AssessmentResult(**{k: DimensionAssessment(**v) for k, v in values["assessment"].items()})
    if values.get("priority_explanation"):
        p = values["priority_explanation"]
        p["component_scores"] = {k: PriorityComponent(**v) for k, v in p["component_scores"].items()}
        values["priority_explanation"] = PriorityResult(**p)
    return EmergencyReport(**values)


class EmergencyProcessingPipeline:
    """Single-process, thread-safe JSON service with shared injected components.

    Query methods return detached JSON-safe dictionaries. process_* returns a
    typed EmergencyReport, or raises PipelineError. Failed analysis has no report.
    """

    def __init__(self, analyzer, embedder, vector_store=None, report_dir=EMERGENCY_REPORT_DIR,
                 store_path=PROCESSED_REPORTS_FILE, processor=None):
        self.analyzer, self.embedder, self.vector_store = analyzer, embedder, vector_store
        self.report_dir = Path(report_dir).resolve()
        self.store_path = Path(store_path).resolve()
        self.processor = processor or DocumentProcessor()
        with _LOCKS_GUARD:
            self._lock = _LOCKS.setdefault(str(self.store_path), threading.RLock())
        with self._lock:
            self._read()  # Fail visibly on corrupt storage; never overwrite it.

    @classmethod
    def from_components(cls, embedder, vector_store=None, **kwargs):
        from emergency_analyzer import EmergencyAnalyzer
        from resource_intelligence import ResourceIntelligence
        from emergency_assessment import EmergencyAssessmentEngine
        from priority_engine import EmergencyPriorityEngine
        analyzer = EmergencyAnalyzer(embedder, resource_intelligence=ResourceIntelligence(embedder),
            assessment_engine=EmergencyAssessmentEngine(embedder), priority_engine=EmergencyPriorityEngine())
        return cls(analyzer, embedder, vector_store, **kwargs)

    @staticmethod
    def report_identity(source_file, content):
        # Basename includes extension, so x.txt and x.pdf cannot collide.
        return "report-" + hashlib.sha256(Path(source_file).name.encode("utf-8") + b"\0" + content).hexdigest()

    def _read(self):
        if not self.store_path.exists():
            return {"format_version": 1, "records": {}}
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
            if data["format_version"] != 1 or not isinstance(data["records"], dict):
                raise ValueError()
            for key, record in data["records"].items():
                if record["report_id"] != key or record["analysis_status"] not in {"SUCCESS", "FAILED"}:
                    raise ValueError()
                if record["analysis_status"] == "SUCCESS":
                    _restore(record["report"])
            return data
        except Exception:
            raise PipelineError("Structured store could not be read; existing data was not overwritten.") from None

    def _write(self, data):
        temporary = None
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.store_path.parent,
                                             suffix=".tmp", delete=False) as output:
                temporary = output.name
                json.dump(data, output, indent=2, ensure_ascii=False, allow_nan=False)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.store_path)
        except Exception:
            raise PipelineError("Structured results could not be saved atomically.") from None
        finally:
            if temporary and Path(temporary).exists():
                with suppress(OSError):
                    Path(temporary).unlink()

    def _analyze(self, text, report_id, source_file, timestamp):
        if not isinstance(text, str) or not text.strip() or "\x00" in text:
            raise PipelineError("Report text is empty or malformed.")
        report = self.analyzer.analyze(text.strip(), report_id=report_id, source_file=source_file)
        report.timestamp = timestamp or _now()
        return report

    def process_report(self, text, report_id, source_file, timestamp=None):
        """Explicit text API, not a directory scan. Does not perform RAG indexing."""
        if not isinstance(report_id, str) or not report_id or not isinstance(source_file, str) or not source_file:
            raise PipelineError("A nonempty report ID and source filename are required.")
        with self._lock:
            data = self._read()
            previous = data["records"].get(report_id)
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest() if isinstance(text, str) else None
            if previous:
                if previous["content_hash"] != digest or previous["source_file"] != source_file:
                    raise PipelineError("Report ID already belongs to different source content.")
                if previous["analysis_status"] == "SUCCESS":
                    return _restore(previous["report"])
            record = {"report_id": report_id, "source_file": source_file, "content_hash": digest,
                      "analysis_status": "FAILED", "rag_status": "NOT_REQUESTED", "processed_at": _now(), "report": None}
            try:
                report = self._analyze(text, report_id, source_file, timestamp)
                record.update(analysis_status="SUCCESS", report=asdict(report))
            except Exception:
                record["analysis_error"] = "Emergency analysis failed; check report content and local model availability."
            record["status"] = record["analysis_status"]
            data["records"][report_id] = record
            self._write(data)
            if record["analysis_status"] == "FAILED":
                raise PipelineError(record["analysis_error"])
            return report

    def process_file(self, file_path):
        path = Path(file_path).resolve()
        if path.parent != self.report_dir:
            raise PipelineError("Live processing accepts only files directly inside emergency_reports.")
        if path.suffix.lower() not in EMERGENCY_REPORT_EXTENSIONS:
            raise PipelineError("Unsupported emergency report extension.")
        try:
            content = path.read_bytes()
            timestamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        except OSError:
            raise PipelineError("Incoming file could not be read; retry when it is available.") from None
        report_id = self.report_identity(path.name, content)
        with self._lock:
            data = self._read()
            record = data["records"].get(report_id)
            if record and record["analysis_status"] == "SUCCESS" and (
                self.vector_store is None or record["rag_status"] == "SUCCESS"
            ):
                return _restore(record["report"])
            record = deepcopy(record) if record else {
                "report_id": report_id, "source_file": path.name,
                "content_hash": hashlib.sha256(content).hexdigest(), "report": None,
                "analysis_status": "FAILED", "rag_status": "NOT_REQUESTED",
            }
            # Parse exactly the bytes used for identity, even if the source changes.
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    snapshot = Path(tmp) / ("snapshot" + path.suffix.lower())
                    snapshot.write_bytes(content)
                    text, chunks = self.processor.process_file(str(snapshot))
                if not text.strip() or "\x00" in text:
                    raise ValueError()
            except Exception:
                record.update(status="FAILED", analysis_status="FAILED", rag_status="FAILED" if self.vector_store else "NOT_REQUESTED",
                    report=None, analysis_error="Document extraction failed or produced empty/malformed text.", processed_at=_now())
                data["records"][report_id] = record
                self._write(data)
                raise PipelineError(record["analysis_error"]) from None
            if record["analysis_status"] != "SUCCESS":
                try:
                    report = self._analyze(text, report_id, path.name, timestamp)
                    record.update(analysis_status="SUCCESS", report=asdict(report))
                    record.pop("analysis_error", None)
                except Exception:
                    record.update(analysis_status="FAILED", report=None,
                                  analysis_error="Emergency analysis failed; no successful NLP result was stored.")
            # Independent operation: even a failed NLP analysis can still have RAG chunks.
            if self.vector_store is not None and record["rag_status"] != "SUCCESS":
                try:
                    embeddings = self.embedder.embed_batch(chunks)
                    self.vector_store.add_document(report_id, chunks, embeddings, {
                        "filename": path.name, "file_hash": record["content_hash"],
                        "upload_time": timestamp, "num_chunks": len(chunks), "report_id": report_id,
                    })
                    record["rag_status"] = "SUCCESS"
                    record.pop("rag_error", None)
                except Exception:
                    record.update(rag_status="FAILED", rag_error="RAG indexing failed; the analysis status is independent.")
            record["processed_at"] = _now()
            record["status"] = "SUCCESS" if record["analysis_status"] == "SUCCESS" and record["rag_status"] != "FAILED" else "FAILED"
            data["records"][report_id] = record
            self._write(data)
            if record["status"] == "FAILED":
                raise PipelineError(" ".join(record[k] for k in ("analysis_error", "rag_error") if k in record))
            return _restore(record["report"])

    def process_pending(self, changes=None, startup=False):
        """Call only from the main execution context; callbacks only enqueue."""
        if changes is None:
            from file_watcher import get_pending_changes
            changes = get_pending_changes()
        paths = {str(Path(p).resolve()) for _, p in changes}
        if startup and self.report_dir.exists():
            paths.update(str(p.resolve()) for p in self.report_dir.iterdir()
                         if p.is_file() and p.suffix.lower() in EMERGENCY_REPORT_EXTENSIONS)
        outcomes = []
        for path in sorted(paths):
            try:
                report = self.process_file(path)
                outcomes.append({"source_file": Path(path).name, "status": "SUCCESS", "report_id": report.report_id})
            except PipelineError as error:
                outcomes.append({"source_file": Path(path).name, "status": "FAILED", "error_message": str(error)})
        return outcomes

    def get_processing_records(self):
        with self._lock:
            return deepcopy(list(self._read()["records"].values()))

    def get_all_reports(self):
        return [r["report"] for r in self.get_processing_records() if r["analysis_status"] == "SUCCESS"]

    def get_report(self, report_id):
        return next((r for r in self.get_all_reports() if r["report_id"] == report_id), None)

    def get_reports_by_priority(self, level):
        from emergency_taxonomy import PriorityLevel
        label = PriorityLevel(level).value
        return [r for r in self.get_reports_sorted_by_priority() if r["priority_level"] == label]

    def get_reports_sorted_by_priority(self):
        return sorted(self.get_all_reports(), key=lambda r: (
            r.get("priority_score") is None, -(r.get("priority_score") or 0), r.get("timestamp") or "", r["report_id"]))

    def get_summary_counts(self):
        from emergency_taxonomy import PriorityLevel
        records = self.get_processing_records()
        reports = [r["report"] for r in records if r["analysis_status"] == "SUCCESS"]
        return {"versions": len(records), "successful_analyses": len(reports),
                "failed_analyses": sum(r["analysis_status"] == "FAILED" for r in records),
                "rag_failures": sum(r["rag_status"] == "FAILED" for r in records),
                "by_priority": {v.value: sum(r["priority_level"] == v.value for r in reports) for v in PriorityLevel},
                "unavailable_priority": sum(r["priority_score"] is None for r in reports)}

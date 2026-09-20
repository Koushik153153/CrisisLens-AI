"""Temporary-store orchestration tests; real-model integration in validation CLI."""

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch

from emergency_pipeline import EmergencyProcessingPipeline, PipelineError
from emergency_schema import EmergencyReport


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.live = self.root / "emergency_reports"
        self.live.mkdir()
        self.store_path = self.root / "data" / "processed.json"
        self.analyzer = Mock()
        self.analyzer.analyze.side_effect = lambda text, **kw: EmergencyReport(raw_text=text, priority_score=70, priority_level="HIGH", **kw)
        self.embedder, self.rag = Mock(), Mock()
        self.embedder.embed_batch.side_effect = lambda chunks: [[0.] for _ in chunks]
        self.pipeline = self.make_pipeline()

    def make_pipeline(self):
        return EmergencyProcessingPipeline(self.analyzer, self.embedder, self.rag, self.live, self.store_path)

    def write(self, name="report.txt", text="People need help in Adyar."):
        path = self.live / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_whole_report_once_separate_rag_chunks(self):
        text = "People need help in Adyar. " * 80
        report = self.pipeline.process_file(self.write(text=text))
        self.assertIsInstance(report, EmergencyReport)
        self.assertEqual(report.raw_text, text.strip())
        self.analyzer.analyze.assert_called_once()
        self.assertEqual(self.analyzer.analyze.call_args.args[0], text.strip())
        self.assertGreater(len(self.rag.add_document.call_args.args[1]), 1)
        self.assertEqual(len(self.pipeline.get_all_reports()), 1)

    def test_identity_versions_and_idempotent_reload(self):
        path = self.write()
        first = self.pipeline.process_file(path)
        again = self.make_pipeline().process_file(path)
        self.assertEqual(asdict(first), asdict(again))
        self.analyzer.analyze.assert_called_once()
        self.rag.add_document.assert_called_once()
        path.write_text("Different content", encoding="utf-8")
        changed = self.pipeline.process_file(path)
        self.assertNotEqual(changed.report_id, first.report_id)
        self.assertEqual(len(self.pipeline.get_all_reports()), 2)
        self.assertNotEqual(self.pipeline.report_identity("a.txt", b"same"), self.pipeline.report_identity("a.pdf", b"same"))

    def test_outside_and_extension_rejected(self):
        for path in (self.root / "old.txt", self.root / "sample_emergency_reports" / "sample.txt", self.write("bad.csv")):
            with self.assertRaises(PipelineError):
                self.pipeline.process_file(path)
        self.assertEqual(self.pipeline.get_all_reports(), [])

    def test_empty_and_malformed_files_persist_failure(self):
        for name, text in (("empty.txt", "  "), ("malformed.txt", "bad\x00data")):
            with self.assertRaises(PipelineError):
                self.pipeline.process_file(self.write(name, text))
        records = self.pipeline.get_processing_records()
        self.assertTrue(all(r["analysis_status"] == "FAILED" and r["report"] is None for r in records))
        self.assertEqual(self.pipeline.get_summary_counts()["failed_analyses"], 2)

    def test_rag_failure_keeps_real_analysis_and_retry_does_not_reanalyze(self):
        path = self.write()
        self.rag.add_document.side_effect = RuntimeError("SECRET_SENTINEL")
        with self.assertRaises(PipelineError) as caught:
            self.pipeline.process_file(path)
        self.assertNotIn("SECRET_SENTINEL", str(caught.exception))
        self.assertNotIn("SECRET_SENTINEL", self.store_path.read_text())
        record = self.pipeline.get_processing_records()[0]
        self.assertEqual((record["analysis_status"], record["rag_status"]), ("SUCCESS", "FAILED"))
        self.assertEqual(len(self.pipeline.get_all_reports()), 1)
        self.rag.add_document.side_effect = None
        self.pipeline.process_file(path)
        self.analyzer.analyze.assert_called_once()
        self.assertEqual(self.pipeline.get_processing_records()[0]["status"], "SUCCESS")

    def test_analysis_failure_does_not_block_rag_or_fake_prediction(self):
        self.analyzer.analyze.side_effect = RuntimeError("SECRET_SENTINEL")
        with self.assertRaises(PipelineError):
            self.pipeline.process_file(self.write())
        record = self.pipeline.get_processing_records()[0]
        self.assertEqual((record["analysis_status"], record["rag_status"]), ("FAILED", "SUCCESS"))
        self.assertIsNone(record["report"])
        self.assertEqual(self.pipeline.get_all_reports(), [])
        self.assertNotIn("SECRET_SENTINEL", self.store_path.read_text())

    def test_queries_and_deterministic_sorting(self):
        self.analyzer.analyze.side_effect = lambda text, **kw: EmergencyReport(raw_text=text,
            priority_score=90 if text == "high" else 30, priority_level="CRITICAL" if text == "high" else "LOW", **kw)
        self.pipeline.process_report("low", "z", "low.txt", "2026-01-01T00:00:00+00:00")
        self.pipeline.process_report("high", "b", "high.txt", "2026-01-01T00:00:00+00:00")
        self.pipeline.process_report("high", "a", "other.txt", "2026-01-01T00:00:00+00:00")
        self.assertEqual([r["report_id"] for r in self.pipeline.get_reports_sorted_by_priority()], ["a", "b", "z"])
        self.assertEqual(len(self.pipeline.get_reports_by_priority("CRITICAL")), 2)
        self.assertIsNone(self.pipeline.get_report("missing"))
        self.assertEqual(self.pipeline.get_report("a")["raw_text"], "high")
        self.assertEqual(self.pipeline.get_summary_counts()["by_priority"]["LOW"], 1)
        self.rag.add_document.assert_not_called()

    def test_process_report_refuses_identity_collision(self):
        self.pipeline.process_report("one", "id", "source.txt")
        with self.assertRaises(PipelineError):
            self.pipeline.process_report("two", "id", "source.txt")

    def test_startup_and_queue_isolation(self):
        path = self.write()
        external = self.root / "sample_emergency_reports"
        external.mkdir()
        sample = external / "sample.txt"
        sample.write_text("not live")
        outcomes = self.pipeline.process_pending([("created", str(path)), ("modified", str(path)), ("created", str(sample))], startup=True)
        self.assertEqual(sum(r["status"] == "SUCCESS" for r in outcomes), 1)
        self.assertEqual(sum(r["status"] == "FAILED" for r in outcomes), 1)
        self.analyzer.analyze.assert_called_once()

    def test_watcher_callback_only_queues(self):
        import file_watcher
        from watchdog.events import FileCreatedEvent
        file_watcher.get_pending_changes()
        with patch.object(file_watcher, "EMERGENCY_REPORT_DIR", str(self.live)):
            file_watcher._UploadHandler().on_created(FileCreatedEvent(str(self.write())))
            self.analyzer.analyze.assert_not_called()
            outcomes = self.pipeline.process_pending()
            self.assertEqual(outcomes[0]["status"], "SUCCESS")

    def test_corrupt_store_not_overwritten(self):
        self.store_path.parent.mkdir()
        self.store_path.write_text("{broken")
        with self.assertRaises(PipelineError):
            self.make_pipeline()
        self.assertEqual(self.store_path.read_text(), "{broken")

    def test_atomic_failure_keeps_previous_store(self):
        self.pipeline.process_file(self.write())
        before = self.store_path.read_bytes()
        with patch("emergency_pipeline.os.replace", side_effect=OSError("SECRET_SENTINEL")):
            with self.assertRaises(PipelineError):
                self.pipeline.process_file(self.write("next.txt", "Another report"))
        self.assertEqual(self.store_path.read_bytes(), before)
        json.loads(before)

    def test_app_main_context_wiring(self):
        import app
        from types import SimpleNamespace
        service = Mock()
        service.process_pending.return_value = [{"status": "FAILED", "error_message": "Safe failure"}]
        ui = SimpleNamespace(session_state={"sync_done": False}, error=Mock())
        with patch.object(app, "st", ui):
            app.handle_live_updates(self.rag, self.embedder, service)
        service.process_pending.assert_called_once_with(startup=True)
        ui.error.assert_called_once_with("Safe failure")
        self.assertTrue(ui.session_state["sync_done"])


class RealPipelineTests(unittest.TestCase):
    def test_full_pipeline_with_temporary_chroma_and_reloaded_store(self):
        from validate_emergency_pipeline import validate
        result = validate()
        self.assertTrue(result["passed"])
        self.assertTrue(result["full_step3_to_step6_outputs"])
        self.assertEqual(result["persisted_after_reload"], 3)
        self.assertEqual(result["rag_documents"], 3)

"""Offline architecture regression checks; no models or API calls required."""

import json
import tempfile
import time
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import config
import file_watcher
from emergency_schema import EmergencyReport


class ArchitectureTests(unittest.TestCase):
    def test_report_serialization_and_unknown_defaults(self):
        first = EmergencyReport(report_id="sample", raw_text="Example report")
        second = EmergencyReport()
        first.locations.append("Example location")
        self.assertEqual(second.locations, [])
        self.assertIsNone(first.priority_score)
        self.assertIsNone(first.action_required)
        self.assertIsNone(first.affected_people)
        self.assertEqual(json.loads(json.dumps(asdict(first))), asdict(first))

    def test_live_watcher_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = root / "emergency_reports"
            reports.mkdir()
            uploads = root / "uploaded_docs"
            uploads.mkdir()
            with patch.object(file_watcher, "EMERGENCY_REPORT_DIR", str(reports)):
                file_watcher.get_pending_changes()
                file_watcher.start_watcher()
                try:
                    (root / "old.txt").write_text("old experiment")
                    (uploads / "old.txt").write_text("old upload")
                    (reports / "ignore.csv").write_text("unsupported")
                    target = reports / "incoming.TXT"
                    target.write_text("Emergency report")
                    events = []
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        events.extend(file_watcher.get_pending_changes())
                        if events:
                            break
                        time.sleep(0.05)
                    self.assertTrue(events, "watchdog did not deliver a report event")
                    self.assertTrue(all(Path(p).resolve() == target.resolve() for _, p in events))
                    self.assertFalse(file_watcher.is_emergency_report(root / "old.txt"))
                    self.assertFalse(file_watcher.is_emergency_report(uploads / "old.txt"))
                finally:
                    file_watcher.stop_watcher()
                    file_watcher.get_pending_changes()

    def test_startup_and_queued_ingestion_isolation(self):
        import app
        from contextlib import nullcontext
        from types import SimpleNamespace
        from unittest.mock import Mock

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = root / "emergency_reports"
            reports.mkdir()
            incoming = reports / "report.txt"
            incoming.write_text("Example emergency report", encoding="utf-8")
            old = root / "old.txt"
            old.write_text("Old experiment", encoding="utf-8")
            store, embedder = Mock(), Mock()
            embedder.embed_batch.return_value = [[0.0]]
            ui = SimpleNamespace(
                session_state={"sync_done": False, "doc_hashes": {}},
                sidebar=nullcontext(), spinner=lambda *a: nullcontext(),
                toast=Mock(), rerun=Mock(),
            )
            with (
                patch.object(config, "EMERGENCY_REPORT_DIR", str(reports)),
                patch.object(file_watcher, "EMERGENCY_REPORT_DIR", str(reports)),
                patch.object(file_watcher, "get_pending_changes", return_value=[("modified", str(old))]),
                patch.object(app, "st", ui),
            ):
                app.handle_live_updates(store, embedder)
            store.add_document.assert_called_once()
            self.assertEqual(store.add_document.call_args.args[0], "report")
            self.assertTrue(ui.session_state["sync_done"])

    def test_chroma_collection_isolation(self):
        import numpy as np
        from vector_store import VectorStore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            legacy = VectorStore(tmp, "live_rag_docs")
            legacy.add_document("old", ["Old experiment"], [np.array([1., 0., 0.])])
            crisis = VectorStore(tmp)
            self.assertEqual(crisis.collection.name, "crisislens_emergency_reports")
            self.assertEqual(crisis.total_chunks(), 0)
            crisis.add_document("new", ["Example report"], [np.array([0., 1., 0.])])
            self.assertEqual(crisis.list_documents(), ["new"])
            self.assertEqual(legacy.list_documents(), ["old"])


if __name__ == "__main__":
    unittest.main()

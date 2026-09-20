"""UI integration tests use isolated persistence and mocked external services."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest
import app
from crisislens_ui import resource_counts, save_upload, report_label
from crisislens_theme import display_label
from emergency_pipeline import EmergencyProcessingPipeline, PipelineError
from emergency_schema import EmergencyReport


class CrisisLensUITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pipeline = EmergencyProcessingPipeline(Mock(), Mock(), report_dir=self.root / "incoming",
                                                     store_path=self.root / "reports.json")
        self.store = Mock()
        self.store.get_document_hashes.return_value = {}
        self.store.total_chunks.return_value = 2
        self.retriever = Mock(store=self.store)
        self.retriever.retrieve.return_value = [{"doc_id": "flood.txt", "text": "Stored flood evidence.", "similarity": .8}]
        self.llm = Mock()
        self.llm.generate_response.return_value = ("Grounded answer.", .1)

    def persist(self, reports):
        records = {r["report_id"]: {"report_id": r["report_id"], "source_file": r.get("source_file", ""),
                    "analysis_status": "SUCCESS", "rag_status": "FAILED", "report": r} for r in reports}
        self.pipeline.store_path.write_text(json.dumps({"format_version": 1, "records": records}), encoding="utf-8")

    def representative(self):
        return {"report_id": "temporary-1", "source_file": "flood.txt", "raw_text": "Five people stranded at Adyar Bridge.",
                "incident_type": "FLOOD", "incident_confidence": .83, "locations": ["Adyar Bridge"],
                "affected_count": 5, "affected_unit": "PERSONS", "vulnerable_groups": ["CHILDREN"],
                "resource_needs": ["RESCUE_BOAT"], "resource_predictions": [{"resource": "RESCUE_BOAT",
                    "confidence": .95, "inference_type": "EXPLICIT", "evidence": "Boat requested.", "reason": "Stored resource reason."}],
                "assessment": {name: {"label": "HIGH", "confidence": .8, "evidence": ["Stored assessment evidence."],
                                "reasons": ["Stored assessment reason."]} for name in ("urgency", "severity", "actionability")},
                "priority_score": 87.5, "priority_level": "CRITICAL", "priority_reasons": ["Stored priority reason."],
                "priority_explanation": {"priority_score": 87.5, "priority_level": "CRITICAL", "available": True,
                    "component_scores": {name: {"label": None, "raw_value": 1, "points": points, "weight": weight}
                        for name, points, weight in [("urgency", 40, .4), ("severity", 22.5, .3), ("actionability", 20, .2), ("context", 5, .1)]}}}

    def run_app(self, action=None):
        resources = (Mock(), self.store, self.retriever, self.llm, Mock(), Mock(), self.pipeline)
        with patch.object(app, "load_resources", return_value=resources), \
             patch.object(self.pipeline, "process_pending", return_value=[]), \
             patch("config.EMERGENCY_DATA_DIR", str(self.root)), \
             patch("file_watcher.watcher_is_active", return_value=True), \
             patch("embedder.Embedder", side_effect=AssertionError("UI constructed an extra model")):
            test = AppTest.from_string("import app\napp.main()", default_timeout=20).run()
            if action:
                action(test)
            self.assertEqual(len(test.exception), 0, str(test.exception))
            return test

    @staticmethod
    def text(test):
        return "\n".join(str(element.value) for kind in ("markdown", "caption", "info", "warning", "text", "subheader")
                         for element in test.get(kind))

    def test_empty_branding_tabs_and_zero_counts(self):
        test = self.run_app()
        self.assertEqual(test.title[0].value, "CrisisLens-AI")
        self.assertEqual([tab.label for tab in test.tabs], ["Command Center", "Report Analysis", "Emergency RAG", "System Evaluation"])
        self.assertTrue(all(m.value == "0" for m in test.metric))
        self.assertIn("No emergency reports yet", self.text(test))
        self.assertEqual(len(test.get("plotly_chart")), 0)

    def test_persisted_full_analysis_and_eaps(self):
        self.persist([self.representative()])
        test = self.run_app()
        text = self.text(test)
        for expected in ("Adyar Bridge", "Persons", "Children", "Rescue Boat", "Explicit", "Stored resource reason.",
                         "Stored assessment evidence.", "Stored assessment reason.", "Stored priority reason.",
                         "Urgency: 40 / 40", "Severity: 22.5 / 30", "Actionability: 20 / 20", "Situation context: 5 / 10"):
            self.assertIn(expected, text)
        self.assertIn("87.5", [metric.value for metric in test.metric])
        self.assertIn("Failed", str(test.dataframe[-1].value))

    def test_all_priority_badges_and_queue_selection(self):
        reports = []
        for index, level in enumerate(("LOW", "MEDIUM", "HIGH", "CRITICAL")):
            report = self.representative()
            report.update(report_id=level, priority_level=level, priority_score=index * 25, source_file=level + ".txt")
            reports.append(report)
        self.persist(reports)
        test = self.run_app(lambda t: t.button(key="select_LOW").click().run())
        for level in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
            self.assertIn(display_label(level), self.text(test))
        self.assertEqual(test.selectbox[0].value, "LOW")
        self.assertEqual(test.selectbox[0].options[0].split(" — ")[0], "CRITICAL.txt")

    def test_older_missing_optional_fields(self):
        self.persist([{"report_id": "old", "raw_text": "An older report."}])
        self.assertIn("Not identified", self.text(self.run_app()))

    def test_empty_optional_lists_and_none_values(self):
        report = self.representative()
        report.update(resource_needs=[], resource_predictions=[], locations=[], affected_count=None,
                      assessment=None, priority_explanation=None, priority_score=None, priority_level=None)
        self.persist([report])
        self.assertIn("No resource needs recorded", self.text(self.run_app()))

    def test_rag_failure_preserves_analysis_and_sources(self):
        self.persist([self.representative()])
        self.llm.generate_response.side_effect = RuntimeError("secret provider details")
        def query(test):
            test.text_area(key="query_input").input("Where is the flood?")
            test.button(key="run_query").click().run()
        test = self.run_app(query)
        self.assertIn("rate-limited", self.text(test))
        self.assertIn("Stored flood evidence.", self.text(test))
        self.assertNotIn("secret provider details", self.text(test))
        self.assertIn("87.5", [m.value for m in test.metric])

    def test_missing_gemini_still_retrieves(self):
        self.llm = None
        test = self.run_app(lambda t: t.text_area(key="query_input").input("Help").run())
        self.assertIn("Source retrieval remains available", self.text(test))

    def test_rag_success_retains_evidence(self):
        def query(test):
            test.text_area(key="query_input").input("Where?")
            test.button(key="run_query").click().run()
        test = self.run_app(query)
        self.assertIn("Grounded answer.", self.text(test))
        self.llm.generate_response.assert_called_once_with("Where?", ["Stored flood evidence."])

    def test_evaluation_reads_actual_files(self):
        source = Path(__file__).parent / "emergency_data"
        for path in source.glob("step*_*.json"):
            if "evaluation" in path.name or "validation" in path.name:
                (self.root / path.name).write_bytes(path.read_bytes())
        test = self.run_app()
        self.assertIn("Controlled synthetic development-corpus results", self.text(test))
        self.assertIn("Step-7 validation: Passed", self.text(test))
        self.assertIn("No gold EAPS labels", self.text(test))
        values = self.text(test)
        self.assertIn("96.00%", values)
        self.assertIn("0.7597", values)

    def test_humanized_labels_and_concise_unique_selector(self):
        for original, label in (("ROAD_ACCIDENT", "Road Accident"), ("CHRONICALLY_ILL", "Chronically Ill"),
                                ("POWER_RESTORATION", "Power Restoration"), (None, "Not identified")):
            self.assertEqual(display_label(original), label)
        report = self.representative()
        report["report_id"] = "report-" + "a" * 64
        second = dict(report, report_id="report-" + "b" * 64)
        self.persist([report, second])
        test = self.run_app()
        self.assertEqual(len(set(test.selectbox[0].options)), 2)
        for label in test.selectbox[0].options:
            self.assertNotIn("a" * 64, label)
            self.assertNotIn("b" * 64, label)
            self.assertIn("Flood — Critical", label)
        self.assertEqual(report["incident_type"], "FLOOD")

    def test_visual_brief_and_collapsed_explainability(self):
        report = self.representative()
        report["priority_explanation"]["component_scores"]["context"]["details"] = ["factor=0.5 context_share=0.2"]
        self.persist([report])
        test = self.run_app()
        text = self.text(test)
        for expected in ("Emergency intelligence summary", "Required response", "At a glance", "Intelligence flow",
                         "Why this report is prioritized", "How quickly help is needed", "How clearly responders can act"):
            self.assertIn(expected, text)
        expanders = {expander.label: expander for expander in test.expander}
        for label in ("Technical EAPS calculation", "View original emergency report", "Why did CrisisLens assess this?", "Technical resource details"):
            self.assertIn(label, expanders)
            self.assertFalse(expanders[label].proto.expanded)
        self.assertIn("factor=0.5", str(expanders["Technical EAPS calculation"].json[0].value))
        self.assertNotIn("factor=0.5", text)

    def test_rag_source_filename_primary_similarity_secondary(self):
        self.retriever.retrieve.return_value[0].update(doc_id="report-" + "a" * 64, metadata={"filename": "flood.txt"})
        def query(test):
            test.text_area(key="query_input").input("Where?")
            test.button(key="run_query").click().run()
        test = self.run_app(query)
        label = next(e.label for e in test.expander if e.label.startswith("Source 1"))
        self.assertIn("flood.txt", label)
        self.assertNotIn("similarity", label)
        self.assertNotIn("report-", label)
        self.assertIn("Semantic similarity: 0.8000", self.text(test))

    def test_evaluation_missing_malformed_and_wrong_shapes(self):
        (self.root / "step3_evaluation.json").write_text("{broken", encoding="utf-8")
        (self.root / "step4_resource_evaluation.json").write_text('{"micro": []}', encoding="utf-8")
        (self.root / "step6_priority_evaluation.json").write_text('{"properties":{"all_passed":[]}}', encoding="utf-8")
        self.assertIn("missing or malformed", self.text(self.run_app()))

    def test_resource_count_is_reports_not_duplicates(self):
        self.assertEqual(resource_counts([{"resource_needs": ["A", "A", "B"]}, {"resource_needs": ["A"]}]), {"A": 2, "B": 1})

    def test_upload_uses_shared_pipeline_and_guards_extension(self):
        self.pipeline.process_file = Mock()
        save_upload(self.pipeline, "report.txt", b"Emergency report")
        destination = self.pipeline.report_dir / "report.txt"
        self.assertEqual(destination.read_bytes(), b"Emergency report")
        self.pipeline.process_file.assert_called_once_with(str(destination))
        with self.assertRaises(PipelineError):
            save_upload(self.pipeline, "unsupported.exe", b"invalid")

    def test_unavailable_rag_initialization_still_processes_and_renders(self):
        with patch("embedder.Embedder", return_value=Mock()) as model, \
             patch("vector_store.VectorStore", side_effect=RuntimeError("index failure")), \
             patch("llm_handler.LLMHandler", side_effect=RuntimeError("provider failure")), \
             patch("emergency_pipeline.EmergencyProcessingPipeline.from_components", return_value=self.pipeline), \
             patch("file_watcher.start_watcher"), patch("config.EMERGENCY_REPORT_DIR", str(self.root)):
            resources = app.load_resources.__wrapped__()
        model.assert_called_once()
        self.assertIsNone(resources[2])
        self.assertIsNone(resources[3])
        self.pipeline.vector_store = resources[1]
        self.pipeline.analyzer.analyze.return_value = EmergencyReport(raw_text="A flood report.", incident_type="FLOOD")
        self.pipeline.embedder.embed_batch.return_value = [[0.1]]
        with self.assertRaises(PipelineError):
            save_upload(self.pipeline, "isolated.txt", b"A flood report.")
        records = self.pipeline.get_processing_records()
        self.assertEqual(records[0]["analysis_status"], "SUCCESS")
        self.assertEqual(records[0]["rag_status"], "FAILED")
        self.retriever = None
        self.llm = None
        self.assertIn("A flood report.", self.text(self.run_app()))

    def test_retrieval_query_failure_is_isolated(self):
        self.persist([self.representative()])
        self.retriever.retrieve.side_effect = RuntimeError("index secret")
        def query(test):
            test.text_area(key="query_input").input("Where?")
            test.button(key="run_query").click().run()
        test = self.run_app(query)
        self.assertIn("Retrieval failed", self.text(test))
        self.assertNotIn("index secret", self.text(test))
        self.assertIn("87.5", [m.value for m in test.metric])


if __name__ == "__main__":
    unittest.main()

"""Behavioral checks with the actual injected cached MiniLM; no gold tuning."""

import json
import os
import unittest
from dataclasses import asdict
from unittest.mock import patch

from emergency_analyzer import EmergencyAnalyzer
from emergency_assessment import EmergencyAssessmentEngine
from emergency_schema import EmergencyReport
from evaluate_emergency_assessment import classification_metrics
from resource_intelligence import ResourceIntelligence


class AssessmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from embedder import Embedder
        cls.embedder = Embedder()
        cls.engine = EmergencyAssessmentEngine(cls.embedder)
        cls.resources = ResourceIntelligence(cls.embedder)
        cls.analyzer = EmergencyAnalyzer(cls.embedder, resource_intelligence=cls.resources, assessment_engine=cls.engine)

    def test_immediate_entrapment(self):
        report = self.analyzer.analyze("Five elderly residents are trapped near Adyar and require immediate boat rescue. Floodwater is rapidly rising.")
        self.assertIn(report.urgency, {"HIGH", "CRITICAL"})
        self.assertEqual(report.actionability, "HIGH")

    def test_medical_distress(self):
        result = self.engine.assess("A person is unconscious and needs emergency assistance immediately.")
        self.assertIn(result.urgency.label, {"HIGH", "CRITICAL"})

    def test_monitoring_and_vague_observation(self):
        for text in ("Routine monitoring only. Conditions are normal and no intervention is requested.",
                     "Heavy rain may occur somewhere in the city. This is a routine observation."):
            with self.subTest(text=text):
                result = self.engine.assess(text)
                self.assertEqual(result.urgency.label, "LOW")
                self.assertEqual(result.actionability.label, "LOW")

    def test_serious_injuries_not_automatically_actionable(self):
        result = self.engine.assess("People have serious injuries after a crash somewhere. The location is unknown.")
        self.assertIn(result.severity.label, {"HIGH", "CRITICAL"})
        self.assertNotEqual(result.actionability.label, "HIGH")

    def test_minor_clear_need(self):
        report = self.analyzer.analyze("A minor disruption in Adyar, Chennai needs a road-clearance crew. No injuries are reported. Please clear the small obstruction during the next work round.")
        self.assertIn(report.severity, {"LOW", "MEDIUM"})
        self.assertEqual(report.actionability, "HIGH")

    def test_urgency_can_differ_from_severity(self):
        result = self.engine.assess("Three passengers have serious injuries and need transport to hospital without delay.")
        self.assertEqual(result.urgency.label, "CRITICAL")
        self.assertEqual(result.severity.label, "HIGH")

    def test_no_casualty_invention_from_large_counts(self):
        text = "Routine monitoring: 500 residents attended a meeting. Conditions are normal."
        context = EmergencyReport(raw_text=text, affected_count=500, affected_unit="PERSONS")
        self.assertNotEqual(self.engine.assess(text, context).severity.label, "CRITICAL")

    def test_evidence_serializable_no_priority_and_preserved_outputs(self):
        text = "Three passengers have serious injuries in Guindy, Chennai. Please send an ambulance immediately."
        baseline = EmergencyAnalyzer(self.embedder, resource_intelligence=self.resources).analyze(text)
        report = self.analyzer.analyze(text)
        for key in ("incident_type", "locations", "affected_people", "affected_unit", "vulnerable_groups", "resource_needs", "resource_predictions"):
            self.assertEqual(getattr(report, key), getattr(baseline, key))
        self.assertIsNone(report.priority_score)
        self.assertIsNone(report.priority_level)
        self.assertEqual(report.priority_reasons, [])
        for dimension in (report.assessment.urgency, report.assessment.severity, report.assessment.actionability):
            self.assertTrue(dimension.evidence)
            self.assertTrue(dimension.reasons)
            for evidence in dimension.evidence:
                self.assertIn(evidence, text)
        json.dumps(asdict(report))

    def test_prototypes_are_cached(self):
        with patch.object(self.embedder, "embed_batch", wraps=self.embedder.embed_batch) as batch:
            engine = EmergencyAssessmentEngine(self.embedder)
            engine.assess("An unconscious person needs help.")
            engine.assess("Routine observation only.")
            self.assertEqual(batch.call_count, 1)

    def test_blank_is_unknown(self):
        result = self.engine.assess("  ")
        for assessment in (result.urgency, result.severity, result.actionability):
            self.assertIsNone(assessment.label)
            self.assertEqual(assessment.evidence, [])


class MetricTests(unittest.TestCase):
    def test_confusion_orientation_and_macro(self):
        metrics = classification_metrics([("LOW", "LOW"), ("LOW", "HIGH"), ("HIGH", "HIGH")], ["LOW", "HIGH"])
        self.assertEqual(metrics["confusion_matrix"]["counts"], [[1, 1], [0, 1]])
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["macro_precision"], .75)
        self.assertAlmostEqual(metrics["macro_recall"], .75)
        self.assertAlmostEqual(metrics["macro_f1"], 2 / 3)


if __name__ == "__main__":
    unittest.main()

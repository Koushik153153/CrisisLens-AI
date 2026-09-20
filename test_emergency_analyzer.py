"""Step-3 extraction checks and real cached-MiniLM integration tests."""

import json
import os
import unittest
from dataclasses import asdict
from unittest.mock import patch

from emergency_analyzer import (
    EmergencyAnalyzer, extract_affected_count, extract_locations,
    extract_vulnerable_groups,
)
from emergency_taxonomy import IncidentType, VulnerableGroup
from evaluate_emergency_analyzer import micro_metrics


class ExtractionTests(unittest.TestCase):
    def test_locations_longest_unique_and_textual_order(self):
        self.assertEqual(extract_locations(
            "At MARINA BEACH,CHENNAI, then Guindy, Chennai. Marina Beach, Chennai again."
        ), ["Marina Beach, Chennai", "Guindy, Chennai"])
        self.assertEqual(extract_locations("Tamil Nadu and Adyar"), ["Tamil Nadu", "Adyar"])
        self.assertEqual(extract_locations("Adyarville is outside this gazetteer."), [])

    def test_digits_words_and_subgroups(self):
        for text, count in (
            ("5 people are stranded.", 5), ("Three passengers are hurt.", 3),
            ("Forty-six residents, including eleven children, need assistance.", 46),
            ("Two chronically ill patients are waiting.", 2),
            ("6 children cannot leave the building.", 6),
        ):
            with self.subTest(text=text):
                result = extract_affected_count(text)
                self.assertEqual((result.count, result.unit), (count, "PERSONS"))

    def test_unknown_and_explicit_zero(self):
        self.assertIsNone(extract_affected_count("Residents are stranded; the number is unknown.").count)
        self.assertIsNone(extract_affected_count("No injuries reported on the blocked road.").count)
        self.assertEqual(extract_affected_count("Zero people were affected.").count, 0)

    def test_family_and_household_counts(self):
        result = extract_affected_count("Around 30 families were displaced.")
        self.assertEqual((result.count, result.unit, result.approximate), (30, "FAMILIES", True))
        result = extract_affected_count("Twenty households lost power.")
        self.assertEqual((result.count, result.unit), (20, "HOUSEHOLDS"))

    def test_vulnerability_synonyms(self):
        self.assertEqual(set(extract_vulnerable_groups(
            "Kids, senior citizens, a pregnant woman, injured passengers, a wheelchair user "
            "and a patient receiving regular dialysis are waiting."
        )), set(VulnerableGroup))

    def test_negation_and_positive_followup(self):
        self.assertEqual(extract_vulnerable_groups("Nobody was struck or injured. All residents are uninjured."), [])
        self.assertEqual(extract_vulnerable_groups("No injuries were reported. Later, two injured passengers arrived."), [VulnerableGroup.INJURED])
        self.assertEqual(extract_vulnerable_groups("No children are present, but elderly residents are waiting."), [VulnerableGroup.ELDERLY])

    def test_micro_metrics_use_counts_across_reports(self):
        result = micro_metrics([(["A", "B"], ["A"]), ([], ["C", "D"])])
        self.assertEqual((result["true_positives"], result["false_positives"], result["false_negatives"]), (1, 1, 2))
        self.assertAlmostEqual(result["precision"], 1 / 2)
        self.assertAlmostEqual(result["recall"], 1 / 3)
        self.assertAlmostEqual(result["f1"], 0.4)


class MiniLMIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from embedder import Embedder
        # One real existing model wrapper shared across all integration tests.
        # Failure to load the local cache is a failure, not a silent mock/skip.
        cls.embedder = Embedder()
        cls.analyzer = EmergencyAnalyzer(cls.embedder)

    def test_reuses_embedder_and_caches_prototypes(self):
        with patch.object(self.embedder, "embed_batch", wraps=self.embedder.embed_batch) as batch:
            analyzer = EmergencyAnalyzer(self.embedder)
            analyzer.classify_incident("Smoke and flames are spreading through a house.")
            analyzer.classify_incident("Residents are trapped by flood water.")
            self.assertIs(analyzer.embedder, self.embedder)
            self.assertEqual(batch.call_count, 1)

    def test_semantic_flood_and_fire(self):
        for text, expected in (
            ("The river has overflowed and submerged streets, leaving families stranded in their homes.", IncidentType.FLOOD),
            ("Flames are consuming a house and smoke is pouring from the windows.", IncidentType.FIRE),
        ):
            with self.subTest(text=text):
                self.assertEqual(self.analyzer.classify_incident(text).label, expected)

    def test_rejection_threshold(self):
        self.assertEqual(self.analyzer.classify_incident("I am learning to multiply fractions for a mathematics examination.").label, IncidentType.OTHER)
        self.assertEqual(self.analyzer.classify_incident("  ").label, IncidentType.OTHER)
        strict = EmergencyAnalyzer(self.embedder, threshold=1.0)
        self.assertEqual(strict.classify_incident("Some water is on the pavement.").label, IncidentType.OTHER)

    def test_analysis_serialization_and_future_fields(self):
        report = self.analyzer.analyze("Around 30 families are stranded by flooding in Velachery, Chennai.", "test", "example.txt")
        self.assertIsNone(report.affected_people)
        self.assertEqual((report.affected_count, report.affected_unit), (30, "FAMILIES"))
        self.assertEqual(report.report_id, "test")
        for field in ("urgency", "severity", "actionability", "priority_score", "priority_level", "action_required", "summary", "timestamp"):
            self.assertIsNone(getattr(report, field))
        for field in ("resource_needs", "actions", "priority_reasons"):
            self.assertEqual(getattr(report, field), [])
        self.assertEqual(json.loads(json.dumps(asdict(report)))["affected_unit"], "FAMILIES")


if __name__ == "__main__":
    unittest.main()

"""Validate the synthetic corpus and taxonomy, not nonexistent NLP accuracy."""

import json
import unittest
from dataclasses import asdict, fields
from pathlib import Path
from unittest.mock import patch

from emergency_schema import EmergencyReport
from emergency_taxonomy import (
    Actionability, ActionType, IncidentType, PriorityLevel, ResourceNeed,
    Severity, Urgency, VulnerableGroup, canonical_label, normalize_label,
)

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "sample_emergency_reports"
GOLD = ROOT / "emergency_data" / "sample_ground_truth.json"


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = json.loads(GOLD.read_text(encoding="utf-8"))
        cls.samples = sorted(SAMPLES.glob("*.txt"))

    def test_exact_one_to_one_file_mapping(self):
        self.assertIsInstance(self.records, list)
        self.assertEqual(len(self.samples), 20)
        self.assertEqual(len(self.records), 20)
        sources = [r["source_file"] for r in self.records]
        ids = [r["report_id"] for r in self.records]
        self.assertEqual(len(set(sources)), 20)
        self.assertEqual(len(set(ids)), 20)
        self.assertTrue(all(isinstance(i, str) and i.strip() for i in ids))
        self.assertEqual(set(sources), {p.name for p in self.samples})
        for source in sources:
            with self.subTest(source=source):
                self.assertEqual(Path(source).name, source)
                path = SAMPLES / source
                self.assertTrue(path.is_file())
                self.assertEqual(path.resolve().parent, SAMPLES.resolve())
                self.assertTrue(path.read_text(encoding="utf-8").strip())

    def test_annotation_types_and_canonical_labels(self):
        required = {
            "report_id", "source_file", "incident_type", "locations",
            "affected_people", "vulnerable_groups", "resource_needs",
            "action_required", "urgency", "severity", "actionability", "actions",
        }
        for record in self.records:
            with self.subTest(report_id=record["report_id"]):
                self.assertTrue(required <= record.keys())
                self.assertFalse({"priority_score", "priority_level", "EAPS", "eaps"} & record.keys())
                for key, taxonomy in (
                    ("incident_type", IncidentType), ("urgency", Urgency),
                    ("severity", Severity), ("actionability", Actionability),
                ):
                    value = record[key]
                    if value is not None:
                        self.assertIn(value, {item.value for item in taxonomy})
                for key, taxonomy in (
                    ("resource_needs", ResourceNeed),
                    ("vulnerable_groups", VulnerableGroup), ("actions", ActionType),
                ):
                    self.assertIsInstance(record[key], list)
                    self.assertEqual(len(record[key]), len(set(record[key])))
                    for value in record[key]:
                        self.assertIn(value, {item.value for item in taxonomy})
                self.assertIsInstance(record["locations"], list)
                self.assertTrue(all(isinstance(v, str) and v.strip() for v in record["locations"]))
                count = record["affected_people"]
                if count is not None:
                    self.assertIs(type(count), int)
                    self.assertGreaterEqual(count, 0)
                if record["action_required"] is not None:
                    self.assertIs(type(record["action_required"]), bool)
                if record["action_required"] is False:
                    self.assertFalse(set(record["actions"]) - {"MONITOR"})
                report = EmergencyReport(**record)
                self.assertEqual(json.loads(json.dumps(asdict(report)))["report_id"], record["report_id"])

    def test_corpus_has_required_variety(self):
        self.assertEqual({r["incident_type"] for r in self.records}, {v.value for v in IncidentType})
        self.assertTrue(any(len(r["resource_needs"]) > 1 for r in self.records))
        self.assertTrue(any(r["vulnerable_groups"] for r in self.records))
        self.assertTrue(any(r["actionability"] == "LOW" for r in self.records))
        self.assertTrue(any(r["affected_people"] is None for r in self.records))

    def test_samples_are_outside_live_ingestion(self):
        import config
        import file_watcher

        live = (ROOT / config.EMERGENCY_REPORT_DIR).resolve()
        self.assertEqual(live, ROOT / "emergency_reports")
        with patch.object(file_watcher, "EMERGENCY_REPORT_DIR", str(live)):
            for sample in self.samples:
                with self.subTest(sample=sample.name):
                    self.assertFalse(sample.resolve().is_relative_to(live))
                    self.assertFalse(file_watcher.is_emergency_report(sample))
                    self.assertEqual(list(live.rglob(sample.name)), [])
            # Detect renamed copies as well as matching filenames.
            contents = {p.read_bytes() for p in self.samples}
            for candidate in live.rglob("*"):
                if candidate.is_file() and candidate.suffix.lower() == ".txt":
                    self.assertNotIn(candidate.read_bytes(), contents)


class TaxonomyTests(unittest.TestCase):
    def test_label_normalization_and_validation(self):
        self.assertEqual(normalize_label("  rescue-boat  "), "RESCUE_BOAT")
        self.assertEqual(canonical_label("medical team", ResourceNeed), "MEDICAL_TEAM")
        self.assertEqual(canonical_label(Urgency.HIGH, Urgency), "HIGH")
        self.assertIs(type(canonical_label(Urgency.HIGH, Urgency)), str)
        self.assertIsNone(canonical_label(None, Urgency))
        self.assertIsNone(canonical_label("   ", Urgency))
        for invalid in ("unknown", "urgent help", "extreme"):
            with self.assertRaises(ValueError):
                canonical_label(invalid, Urgency)
        with self.assertRaises(ValueError):
            canonical_label(Severity.HIGH, Urgency)
        with self.assertRaises(TypeError):
            normalize_label(123)
        with self.assertRaises(ValueError):
            canonical_label("CRITICAL", Actionability)

    def test_schema_enum_json_and_independent_defaults(self):
        report = EmergencyReport(
            incident_type=IncidentType.FLOOD, resource_needs=[ResourceNeed.RESCUE_BOAT],
            vulnerable_groups=[VulnerableGroup.CHILDREN], actions=[ActionType.RESCUE],
            urgency=Urgency.CRITICAL, severity=Severity.HIGH,
            actionability=Actionability.HIGH,
        )
        decoded = json.loads(json.dumps(asdict(report)))
        self.assertEqual(decoded["incident_type"], "FLOOD")
        self.assertEqual(decoded["resource_needs"], ["RESCUE_BOAT"])
        self.assertEqual(decoded["actions"], ["RESCUE"])
        first, second = EmergencyReport(), EmergencyReport()
        for field in fields(first):
            if isinstance(getattr(first, field.name), list):
                self.assertIsNot(getattr(first, field.name), getattr(second, field.name))
        for key in ("urgency", "severity", "actionability", "priority_score", "priority_level"):
            self.assertIsNone(getattr(first, key))
        self.assertEqual({v.value for v in PriorityLevel}, {"LOW", "MEDIUM", "HIGH", "CRITICAL"})


if __name__ == "__main__":
    unittest.main()

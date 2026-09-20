"""Deterministic EAPS properties plus opt-in pipeline compatibility."""

import json
import os
import unittest
from copy import deepcopy
from dataclasses import asdict, replace
from decimal import Decimal

from config import EAPS_CONFIG
from emergency_schema import EmergencyReport
from emergency_taxonomy import ResourceNeed, VulnerableGroup
from priority_engine import EmergencyPriorityEngine
from evaluate_priority_engine import check_properties


class PriorityTests(unittest.TestCase):
    def setUp(self):
        self.engine = EmergencyPriorityEngine()
        self.base = EmergencyReport(urgency="HIGH", severity="HIGH", actionability="HIGH")

    def test_exhaustive_bounds_monotonicity_repeatability_missing_and_sums(self):
        result = check_properties(self.engine)
        self.assertEqual(result["primary_context_combinations"], 96)
        self.assertTrue(result["all_passed"], result)

    def test_low_below_high(self):
        low = replace(self.base, urgency="LOW", severity="LOW", actionability="LOW")
        self.assertLess(self.engine.score(low).priority_score, self.engine.score(self.base).priority_score)

    def test_context_cap_and_resource_deduplication(self):
        rich = replace(self.base, vulnerable_groups=list(VulnerableGroup) * 20,
                       resource_needs=list(ResourceNeed) * 100, affected_count=100000, affected_unit="PERSONS")
        result = self.engine.score(rich)
        self.assertLessEqual(result.component_scores["context"].points, 10)
        one = self.engine.score(replace(self.base, resource_needs=[ResourceNeed.AMBULANCE]))
        many = self.engine.score(replace(self.base, resource_needs=list(ResourceNeed) * 100))
        self.assertEqual(one.priority_score, many.priority_score)

    def test_family_and_household_counts_not_converted(self):
        baseline = self.engine.score(self.base).priority_score
        for unit in ("FAMILIES", "HOUSEHOLDS"):
            report = replace(self.base, affected_count=100, affected_unit=unit)
            self.assertEqual(self.engine.score(report).priority_score, baseline)
        self.assertGreater(self.engine.score(replace(self.base, affected_count=100, affected_unit="PERSONS")).priority_score, baseline)

    def test_missing_and_invalid_labels(self):
        for dimension in ("urgency", "severity", "actionability"):
            for value in (None, "unknown"):
                result = self.engine.score(replace(self.base, **{dimension: value}))
                self.assertIsNone(result.priority_score)
                self.assertIsNone(result.priority_level)
                self.assertFalse(result.available)
                self.assertTrue(result.reasons)
                self.assertEqual(result.component_scores, {})

    def test_band_boundaries(self):
        for value, label in ((0,"LOW"),(39.99,"LOW"),(40,"MEDIUM"),(59.99,"MEDIUM"),
                             (60,"HIGH"),(79.99,"HIGH"),(80,"CRITICAL"),(100,"CRITICAL")):
            self.assertEqual(self.engine.level_for_score(value), label)

    def test_worked_example_and_serialization(self):
        report = replace(self.base, urgency="CRITICAL", vulnerable_groups=[VulnerableGroup.ELDERLY],
                         affected_count=8, affected_unit="PERSONS", resource_needs=[ResourceNeed.RESCUE_BOAT])
        result = self.engine.score(report)
        self.assertEqual(result.priority_score, 89.5)  # 40 + 22.5 + 20 + (3+1+3)
        self.assertEqual(sum(Decimal(str(c.points)) for c in result.component_scores.values()), Decimal("89.5"))
        self.assertTrue(result.reasons)
        self.assertEqual(json.loads(json.dumps(asdict(result)))["priority_level"], "CRITICAL")

    def test_confidence_and_incident_do_not_add_points(self):
        report = replace(self.base, incident_confidence=0.01, incident_type="FIRE", raw_text="Any text")
        self.assertEqual(self.engine.score(report), self.engine.score(self.base))

    def test_invalid_config_rejected(self):
        config = deepcopy(EAPS_CONFIG)
        config["weights"]["urgency"] = 0.9
        with self.assertRaises(ValueError):
            EmergencyPriorityEngine(config)
        config = deepcopy(EAPS_CONFIG)
        config["context_shares"]["population"] = -1
        with self.assertRaises(ValueError):
            EmergencyPriorityEngine(config)


class PriorityPipelineTests(unittest.TestCase):
    def test_pipeline_preserves_prior_outputs_without_extra_embeddings(self):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from unittest.mock import patch
        from embedder import Embedder
        from emergency_analyzer import EmergencyAnalyzer
        from emergency_assessment import EmergencyAssessmentEngine
        from resource_intelligence import ResourceIntelligence
        embedder = Embedder()
        analyzer = EmergencyAnalyzer(embedder, resource_intelligence=ResourceIntelligence(embedder),
                                     assessment_engine=EmergencyAssessmentEngine(embedder))
        text = "Three passengers have serious injuries in Guindy, Chennai. Please send an ambulance immediately."
        with patch.object(embedder, "embed", wraps=embedder.embed) as single, patch.object(embedder, "embed_batch", wraps=embedder.embed_batch) as batch:
            baseline = analyzer.analyze(text)
            prior_calls = (single.call_count, batch.call_count)
            single.reset_mock(); batch.reset_mock()
            analyzer.priority_engine = EmergencyPriorityEngine()
            report = analyzer.analyze(text)
            self.assertEqual((single.call_count, batch.call_count), prior_calls)
        self.assertIsNotNone(report.priority_score)
        self.assertIsNotNone(report.priority_level)
        self.assertTrue(report.priority_reasons)
        old, new = asdict(baseline), asdict(report)
        for key in ("priority_score", "priority_level", "priority_reasons", "priority_explanation"):
            old.pop(key); new.pop(key)
        self.assertEqual(old, new)


if __name__ == "__main__":
    unittest.main()

"""Behavioral resource tests using the existing offline MiniLM model."""

import json
import os
import unittest
from dataclasses import asdict
from unittest.mock import patch

from emergency_analyzer import EmergencyAnalyzer
from emergency_taxonomy import IncidentType
from resource_intelligence import ResourceIntelligence


class ResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from embedder import Embedder
        cls.embedder = Embedder()
        cls.engine = ResourceIntelligence(cls.embedder)

    def labels(self, text):
        return {p.resource.value for p in self.engine.analyze_resource_needs(text)}

    def test_explicit_requests(self):
        for text, label in (
            ("Please send an AMBULANCE.", "AMBULANCE"),
            ("A rescue boat is requested to collect residents.", "RESCUE_BOAT"),
            ("Please arrange potable water for the camp.", "DRINKING_WATER"),
        ):
            with self.subTest(text=text):
                predictions = self.engine.analyze_resource_needs(text)
                self.assertTrue(any(p.resource == label and p.inference_type == "EXPLICIT" for p in predictions))

    def test_implicit_medical_need(self):
        labels = self.labels("Seriously injured passengers are bleeding heavily and need emergency transport to hospital.")
        self.assertTrue({"AMBULANCE", "MEDICAL_TEAM"} <= labels)

    def test_implicit_rescue(self):
        self.assertIn("RESCUE_TEAM", self.labels("Workers are trapped beneath rubble and cannot escape without assistance."))

    def test_implicit_shelter(self):
        self.assertIn("SHELTER", self.labels("Displaced families have nowhere safe to stay and need somewhere to sleep tonight."))

    def test_operational_contexts(self):
        for text, label in (
            ("Debris is blocking the access road and must be removed so vehicles can pass.", "ROAD_CLEARANCE"),
            ("Mains power has failed and the backup electrical supply is running out.", "POWER_RESTORATION"),
            ("Flames are spreading through a burning house.", "FIRE_SERVICE"),
        ):
            with self.subTest(text=text):
                self.assertIn(label, self.labels(text))

    def test_no_resources_and_satisfied_needs(self):
        for text in (
            "Routine weather observation: a light breeze and clouds. No assistance is requested.",
            "No ambulance is required. No medical team is needed.",
            "Food and drinking water have already been supplied. Accommodation is adequate.",
            "The fire was fully extinguished. Power is back. No additional assistance is requested.",
            "The ambulance is unavailable.", "",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.labels(text), set())

    def test_multiple_deduplicated_explanations(self):
        text = "Please send an ambulance, an ambulance and a medical team. Drinking water is also required."
        predictions = self.engine.analyze_resource_needs(text)
        labels = [p.resource.value for p in predictions]
        self.assertEqual(len(labels), len(set(labels)))
        self.assertTrue({"AMBULANCE", "MEDICAL_TEAM", "DRINKING_WATER"} <= set(labels))
        for p in predictions:
            self.assertIn(p.evidence, text)
            self.assertTrue(p.reason)
            self.assertIn(p.inference_type, {"EXPLICIT", "IMPLICIT"})
            json.dumps(asdict(p))

    def test_stronger_context_needed_for_boat_and_ambulance(self):
        self.assertNotIn("RESCUE_BOAT", self.labels("Workers are trapped in a lift and cannot escape without assistance."))
        self.assertNotIn("AMBULANCE", self.labels("Two riders have superficial cuts and request a first-aid team."))

    def test_prototypes_cached(self):
        with patch.object(self.embedder, "embed_batch", wraps=self.embedder.embed_batch) as batch:
            engine = ResourceIntelligence(self.embedder)
            construction_calls = list(batch.call_args_list)
            engine.analyze_resource_needs("Please send an ambulance.")
            engine.analyze_resource_needs("Please send drinking water.")
            self.assertEqual(len(construction_calls), 1)
            # The construction batch must never be submitted again for reports.
            prototype_text = construction_calls[0].args[0]
            self.assertFalse(any(call.args[0] == prototype_text for call in batch.call_args_list[1:]))

    def test_heat_is_not_food_evidence(self):
        self.assertEqual(self.labels("The crew reports no remaining smoke or heat and no additional assistance is requested."), set())

    def test_incident_context_does_not_replace_water_access_evidence(self):
        predictions = self.engine.analyze_resource_needs(
            "Workers are trapped in a lift and cannot escape without assistance.", IncidentType.FLOOD)
        self.assertNotIn("RESCUE_BOAT", {p.resource for p in predictions})

    def test_optional_integration_and_future_fields(self):
        analyzer = EmergencyAnalyzer(self.embedder, resource_intelligence=self.engine)
        report = analyzer.analyze("Please send an ambulance to Guindy, Chennai.")
        self.assertIn("AMBULANCE", report.resource_needs)
        self.assertTrue(report.resource_predictions)
        for key in ("urgency", "severity", "actionability", "priority_score", "priority_level", "action_required"):
            self.assertIsNone(getattr(report, key))
        self.assertEqual(report.actions, [])
        json.dumps(asdict(report))


if __name__ == "__main__":
    unittest.main()

"""EAPS properties and descriptive distribution only: no gold priority targets."""

import hashlib
import itertools
import json
import os
import statistics
from dataclasses import asdict, replace
from decimal import Decimal
from pathlib import Path

from emergency_schema import EmergencyReport
from emergency_taxonomy import Urgency, Severity, Actionability, PriorityLevel, ResourceNeed, VulnerableGroup
from priority_engine import EmergencyPriorityEngine

ROOT = Path(__file__).resolve().parent
GOLD_HASH = "c3902364a60a9e8dfa75a0360dd24a8f7b70cb6ee173548a811f42bfcdba52f7"


def check_properties(engine):
    """Exhaust all primary combinations under absent and saturated context."""
    reports = []
    for u, s, a in itertools.product(Urgency, Severity, Actionability):
        for saturated in (False, True):
            reports.append(EmergencyReport(urgency=u, severity=s, actionability=a,
                vulnerable_groups=list(VulnerableGroup) if saturated else [],
                resource_needs=list(ResourceNeed) if saturated else [],
                affected_count=10000 if saturated else None, affected_unit="PERSONS" if saturated else None))
    checks = {"bounds": True, "repeatability": True, "component_sum": True,
              "urgency_monotonic": True, "severity_monotonic": True, "actionability_monotonic": True,
              "missing_data_unavailable": True, "context_capped": True}
    for report in reports:
        result = engine.score(report)
        checks["bounds"] &= 0 <= result.priority_score <= 100
        checks["repeatability"] &= result == engine.score(report)
        checks["component_sum"] &= sum(Decimal(str(c.points)) for c in result.component_scores.values()) == Decimal(str(result.priority_score))
        checks["context_capped"] &= result.component_scores["context"].points <= engine.settings["weights"]["context"] * 100
        for dimension, enum in (("urgency", Urgency), ("severity", Severity), ("actionability", Actionability)):
            scores = [engine.score(replace(report, **{dimension: label})).priority_score for label in enum]
            checks[dimension + "_monotonic"] &= scores == sorted(scores)
            unavailable = engine.score(replace(report, **{dimension: None}))
            checks["missing_data_unavailable"] &= unavailable.priority_score is None and unavailable.priority_level is None and not unavailable.available
    return {"primary_context_combinations": len(reports), "checks": checks, "all_passed": all(checks.values())}


def main():
    gold_path = ROOT / "emergency_data/sample_ground_truth.json"
    before = hashlib.sha256(gold_path.read_bytes()).hexdigest()
    if before != GOLD_HASH:
        raise RuntimeError("STOP: gold hash differs from approved baseline")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from embedder import Embedder
    from emergency_analyzer import EmergencyAnalyzer
    from emergency_assessment import EmergencyAssessmentEngine
    from resource_intelligence import ResourceIntelligence

    engine = EmergencyPriorityEngine()
    properties = check_properties(engine)
    if not properties["all_passed"]:
        raise AssertionError(properties)
    embedder = Embedder()
    analyzer = EmergencyAnalyzer(embedder, resource_intelligence=ResourceIntelligence(embedder),
        assessment_engine=EmergencyAssessmentEngine(embedder), priority_engine=engine)
    rows = []
    # Only IDs/filenames are read from annotations. All EAPS inputs are predicted.
    for entry in json.loads(gold_path.read_text(encoding="utf-8")):
        source = ROOT / "sample_emergency_reports" / entry["source_file"]
        if source.resolve().parent != (ROOT / "sample_emergency_reports").resolve():
            raise ValueError("Expected sample basename")
        report = analyzer.analyze(source.read_text(encoding="utf-8"), entry["report_id"], entry["source_file"])
        rows.append({"report_id": report.report_id, "source_file": report.source_file,
                     "predicted_inputs": {key: getattr(report, key) for key in
                        ("urgency", "severity", "actionability", "vulnerable_groups", "resource_needs", "affected_count", "affected_unit")},
                     **asdict(report.priority_explanation)})
    scores = [row["priority_score"] for row in rows if row["available"]]
    distribution = {"available": len(scores), "unavailable": len(rows) - len(scores),
                    "min": min(scores) if scores else None, "max": max(scores) if scores else None,
                    "mean": statistics.mean(scores) if scores else None,
                    "median": statistics.median(scores) if scores else None,
                    "counts_by_level": {level.value: sum(r["priority_level"] == level for r in rows) for level in PriorityLevel}}
    after = hashlib.sha256(gold_path.read_bytes()).hexdigest()
    if after != before:
        raise RuntimeError("STOP: gold changed during evaluation")
    result = {"scope": "Structural/behavioral checks and descriptive development-corpus analysis; no gold EAPS targets",
              "settings": engine.settings, "properties": properties, "distribution": distribution,
              "minimum_reports": [r["report_id"] for r in rows if r["available"] and r["priority_score"] == distribution["min"]],
              "maximum_reports": [r["report_id"] for r in rows if r["available"] and r["priority_score"] == distribution["max"]],
              "gold_sha256_before": before, "gold_sha256_after": after, "reports": rows}
    (ROOT / "emergency_data/step6_priority_evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"reports", "settings"}}, indent=2))


if __name__ == "__main__":
    main()

"""Independent dimension metrics on the unchanged synthetic development corpus."""

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from config import ASSESSMENT_CUE_BONUS, ASSESSMENT_LARGE_PERSON_COUNT
from emergency_analyzer import EmergencyAnalyzer
from emergency_assessment import EmergencyAssessmentEngine
from emergency_taxonomy import Urgency, Severity, Actionability
from resource_intelligence import ResourceIntelligence

ROOT = Path(__file__).resolve().parent
EXPECTED_GOLD_SHA256 = "c3902364a60a9e8dfa75a0360dd24a8f7b70cb6ee173548a811f42bfcdba52f7"
DIMENSIONS = {"urgency": Urgency, "severity": Severity, "actionability": Actionability}


def classification_metrics(pairs, labels):
    """Rows are gold, columns predicted; macro includes all canonical classes.

    Zero-denominator precision/recall/F1 is zero. Unknown predictions (if any)
    get a separate column and count as misses for the gold class.
    """
    columns = labels + (["UNKNOWN"] if any(p is None for _, p in pairs) else [])
    matrix = [[0 for _ in columns] for _ in labels]
    for gold, predicted in pairs:
        matrix[labels.index(gold)][columns.index(predicted or "UNKNOWN")] += 1
    per_class = {}
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted_count = sum(row[i] for row in matrix)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        per_class[label] = {"precision": precision, "recall": recall,
                            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                            "support": support}
    return {
        "accuracy": sum(g == p for g, p in pairs) / len(pairs) if pairs else 0.0,
        **{f"macro_{metric}": sum(r[metric] for r in per_class.values()) / len(labels)
           for metric in ("precision", "recall", "f1")},
        "confusion_matrix": {"rows_gold": labels, "columns_predicted": columns, "counts": matrix},
        "per_class": per_class,
    }


def evaluate(analyzer, root=ROOT):
    gold_path = root / "emergency_data" / "sample_ground_truth.json"
    original = gold_path.read_bytes()
    before = hashlib.sha256(original).hexdigest()
    if before != EXPECTED_GOLD_SHA256:
        raise RuntimeError("STOP: gold SHA-256 differs from the approved Step-4 hash")
    pairs = {d: [] for d in DIMENSIONS}
    reports = []
    for gold in json.loads(original):
        source = root / "sample_emergency_reports" / gold["source_file"]
        if source.resolve().parent != (root / "sample_emergency_reports").resolve():
            raise ValueError("Expected a sample basename")
        report = analyzer.analyze(source.read_text(encoding="utf-8"), gold["report_id"], gold["source_file"])
        assessments = asdict(report.assessment)
        predicted = {d: getattr(report, d) for d in DIMENSIONS}
        errors = {}
        for d in DIMENSIONS:
            pairs[d].append((gold[d], predicted[d]))
            if gold[d] != predicted[d]:
                errors[d] = {"gold": gold[d], "predicted": predicted[d]}
        assert report.priority_score is None and report.priority_level is None
        reports.append({"report_id": gold["report_id"], "source_file": gold["source_file"],
                        "gold": {d: gold[d] for d in DIMENSIONS}, "predicted": predicted,
                        "assessments": assessments, "errors": errors})
    after = hashlib.sha256(gold_path.read_bytes()).hexdigest()
    if after != before:
        raise RuntimeError("STOP: gold changed during evaluation")
    return {"scope": "Synthetic development corpus, not held-out evaluation",
            "model": analyzer.embedder.model_name,
            "settings": {"cue_bonus": ASSESSMENT_CUE_BONUS, "large_person_count": ASSESSMENT_LARGE_PERSON_COUNT},
            "total_reports": len(reports), "gold_sha256_before": before, "gold_sha256_after": after,
            "metrics": {d: classification_metrics(pairs[d], [v.value for v in enum]) for d, enum in DIMENSIONS.items()},
            "reports": reports}


def main():
    # Check integrity before loading the cached model or running inference.
    if hashlib.sha256((ROOT / "emergency_data/sample_ground_truth.json").read_bytes()).hexdigest() != EXPECTED_GOLD_SHA256:
        raise RuntimeError("STOP: unexpected gold hash")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from embedder import Embedder
    embedder = Embedder()
    analyzer = EmergencyAnalyzer(embedder, resource_intelligence=ResourceIntelligence(embedder),
                                 assessment_engine=EmergencyAssessmentEngine(embedder))
    result = evaluate(analyzer)
    (ROOT / "emergency_data/step5_assessment_evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "reports"}, indent=2))
    for report in result["reports"]:
        if report["errors"]:
            print(report["report_id"], json.dumps(report["errors"]))


if __name__ == "__main__":
    main()

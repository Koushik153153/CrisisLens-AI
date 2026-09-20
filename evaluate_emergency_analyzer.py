"""Offline Step-3 evaluation on the development corpus; never writes gold labels."""

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from emergency_analyzer import EmergencyAnalyzer

ROOT = Path(__file__).resolve().parent


def micro_metrics(pairs: list[tuple[list[str], list[str]]]) -> dict:
    """Micro set metrics; zero denominators produce zero, not invented matches."""
    tp = fp = fn = 0
    for predicted, expected in pairs:
        pred, gold = set(predicted), set(expected)
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
    }


def evaluate(analyzer: EmergencyAnalyzer, root: Path = ROOT) -> dict:
    gold_path = root / "emergency_data" / "sample_ground_truth.json"
    gold_bytes = gold_path.read_bytes()
    records = json.loads(gold_bytes)
    details, location_pairs, group_pairs = [], [], []
    incident_correct = count_correct = known_count_correct = known_count_samples = 0
    corpus_hash = hashlib.sha256()
    for gold in records:
        source = root / "sample_emergency_reports" / gold["source_file"]
        if source.resolve().parent != (root / "sample_emergency_reports").resolve():
            raise ValueError("Ground-truth source must be a sample filename")
        file_bytes = source.read_bytes()
        corpus_hash.update(gold["source_file"].encode("utf-8") + b"\0" + file_bytes)
        predicted = asdict(analyzer.analyze(file_bytes.decode("utf-8"), gold["report_id"], gold["source_file"]))
        # Step-2 gold counts are individual people. Future gold may explicitly
        # supply affected_count + affected_unit for family/household examples.
        gold_count = gold.get("affected_count", gold["affected_people"])
        gold_unit = gold.get("affected_unit", "PERSONS" if gold_count is not None else None)
        count_match = (predicted["affected_count"], predicted["affected_unit"]) == (gold_count, gold_unit)
        incident_match = predicted["incident_type"] == gold["incident_type"]
        incident_correct += incident_match
        count_correct += count_match
        if gold_count is not None:
            known_count_samples += 1
            known_count_correct += count_match
        location_pairs.append((predicted["locations"], gold["locations"]))
        group_pairs.append((predicted["vulnerable_groups"], gold["vulnerable_groups"]))
        errors = {}
        if not incident_match:
            errors["incident_type"] = {"expected": gold["incident_type"], "predicted": predicted["incident_type"]}
        if not count_match:
            errors["affected_count"] = {
                "expected": {"count": gold_count, "unit": gold_unit},
                "predicted": {"count": predicted["affected_count"], "unit": predicted["affected_unit"]},
            }
        for field in ("locations", "vulnerable_groups"):
            missing = sorted(set(gold[field]) - set(predicted[field]))
            unexpected = sorted(set(predicted[field]) - set(gold[field]))
            if missing or unexpected:
                errors[field] = {"missing": missing, "unexpected": unexpected}
        details.append({"report_id": gold["report_id"], "source_file": gold["source_file"],
                        "prediction": predicted, "errors": errors})
    total = len(records)
    return {
        "evaluation_scope": "Controlled synthetic development corpus; not held-out NLP accuracy",
        "model": analyzer.embedder.model_name,
        "incident_threshold": analyzer.threshold,
        "incident_aggregation": "maximum prototype cosine similarity",
        "gold_sha256": hashlib.sha256(gold_bytes).hexdigest(),
        "sample_corpus_sha256": corpus_hash.hexdigest(),
        "total_reports": total,
        "incident": {"accuracy": incident_correct / total if total else 0, "correct": incident_correct},
        "locations": micro_metrics(location_pairs),
        "affected_count": {
            "exact_match_accuracy": count_correct / total if total else 0,
            "correct": count_correct, "samples_with_count": known_count_samples,
            "correct_with_count": known_count_correct,
            "unknown_count_samples": total - known_count_samples,
            "unit_policy": "Step-2 non-null affected_people gold implies PERSONS; null implies no unit",
        },
        "vulnerable_groups": micro_metrics(group_pairs),
        "reports_with_errors": sum(bool(item["errors"]) for item in details),
        "reports": details,
    }


def main() -> None:
    # Use only the already-cached MiniLM; do not download models or call APIs.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from embedder import Embedder

    analyzer = EmergencyAnalyzer(Embedder())  # One model for all twenty reports.
    result = evaluate(analyzer)
    destination = ROOT / "emergency_data" / "step3_evaluation.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "reports"}, indent=2))
    for report in result["reports"]:
        if report["errors"]:
            print(report["report_id"], json.dumps(report["errors"]))


if __name__ == "__main__":
    main()

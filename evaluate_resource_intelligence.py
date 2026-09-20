"""Offline resource evaluation; read-only gold, isolated generated output."""

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from emergency_analyzer import EmergencyAnalyzer
from emergency_taxonomy import ResourceNeed
from evaluate_emergency_analyzer import micro_metrics
from resource_intelligence import ResourceIntelligence

ROOT = Path(__file__).resolve().parent


def evaluate(analyzer: EmergencyAnalyzer, root: Path = ROOT) -> dict:
    gold_path = root / "emergency_data" / "sample_ground_truth.json"
    original = gold_path.read_bytes()
    records = json.loads(original)
    pairs, details = [], []
    explicit = implicit = exact = 0
    for gold in records:
        source = root / "sample_emergency_reports" / gold["source_file"]
        if source.resolve().parent != (root / "sample_emergency_reports").resolve():
            raise ValueError("Expected a sample basename")
        report = analyzer.analyze(source.read_text(encoding="utf-8"), gold["report_id"], gold["source_file"])
        predicted = [r.value for r in report.resource_needs]
        expected = gold["resource_needs"]
        pairs.append((predicted, expected))
        fp, fn = sorted(set(predicted) - set(expected)), sorted(set(expected) - set(predicted))
        exact += not fp and not fn
        explicit += sum(p.inference_type == "EXPLICIT" for p in report.resource_predictions)
        implicit += sum(p.inference_type == "IMPLICIT" for p in report.resource_predictions)
        details.append({"report_id": gold["report_id"], "source_file": gold["source_file"],
                        "expected": expected, "predicted": predicted,
                        "predictions": [asdict(p) for p in report.resource_predictions],
                        "false_positives": fp, "false_negatives": fn})
    per_resource = {}
    for resource in ResourceNeed:
        label = resource.value
        subset = [([label] if label in pred else [], [label] if label in gold else []) for pred, gold in pairs]
        metrics = micro_metrics(subset)
        metrics["support"] = sum(label in gold for _, gold in pairs)
        # Include unsupported labels explicitly; zero support is not a success.
        per_resource[label] = metrics
    result = {
        "scope": "Controlled synthetic development corpus, not held-out evaluation",
        "model": analyzer.embedder.model_name,
        "implicit_threshold": analyzer.resource_intelligence.implicit_threshold,
        "total_reports": len(records), "micro": micro_metrics(pairs),
        "exact_set_accuracy": exact / len(records) if records else 0,
        "exact_sets": exact, "explicit_predictions": explicit, "implicit_predictions": implicit,
        "per_resource": per_resource, "reports": details,
        "gold_sha256_before": hashlib.sha256(original).hexdigest(),
        "gold_sha256_after": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
    }
    if result["gold_sha256_before"] != result["gold_sha256_after"]:
        raise RuntimeError("Gold changed during evaluation")
    return result


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from embedder import Embedder
    embedder = Embedder()
    analyzer = EmergencyAnalyzer(embedder, resource_intelligence=ResourceIntelligence(embedder))
    result = evaluate(analyzer)
    output = ROOT / "emergency_data" / "step4_resource_evaluation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "reports"}, indent=2))
    for r in result["reports"]:
        if r["false_positives"] or r["false_negatives"]:
            print(r["report_id"], "FP", r["false_positives"], "FN", r["false_negatives"])


if __name__ == "__main__":
    main()

"""Temporary end-to-end pipeline validation; no accuracy claims or live imports."""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from emergency_pipeline import EmergencyProcessingPipeline

ROOT = Path(__file__).resolve().parent
GOLD_HASH = "c3902364a60a9e8dfa75a0360dd24a8f7b70cb6ee173548a811f42bfcdba52f7"


def validate():
    gold = ROOT / "emergency_data/sample_ground_truth.json"
    before = hashlib.sha256(gold.read_bytes()).hexdigest()
    if before != GOLD_HASH:
        raise RuntimeError("Gold integrity mismatch")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from embedder import Embedder
    from vector_store import VectorStore
    embedder = Embedder()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        live = root / "emergency_reports"
        live.mkdir()
        store = VectorStore(str(root / "chroma"), "pipeline_validation")
        pipeline = EmergencyProcessingPipeline.from_components(embedder, store, report_dir=live,
                                                              store_path=root / "processed.json")
        reports = []
        for name in ("01_flood_rooftop.txt", "06_medical_pregnancy.txt", "19_weather_observation.txt"):
            path = live / name
            path.write_bytes((ROOT / "sample_emergency_reports" / name).read_bytes())
            report = pipeline.process_file(path)
            assert report.incident_type and report.locations
            assert report.urgency and report.severity and report.actionability
            assert report.assessment and report.priority_explanation and report.priority_score is not None
            assert isinstance(report.resource_needs, list) and isinstance(report.resource_predictions, list)
            json.dumps(asdict(report), allow_nan=False)
            reports.append(report)
        assert reports[0].resource_needs and reports[1].resource_predictions
        # Reinitialize persistence/service while reusing exactly the same analyzer/model.
        reloaded = EmergencyProcessingPipeline(pipeline.analyzer, embedder, store, live, root / "processed.json")
        for report in reports:
            again = reloaded.process_file(live / report.source_file)
            assert asdict(again) == asdict(report)
        assert len(reloaded.get_all_reports()) == 3
        assert len(store.list_documents()) == 3
        scores = [r["priority_score"] for r in reloaded.get_reports_sorted_by_priority()]
        assert scores == sorted(scores, reverse=True)
        assert all(r["status"] == "SUCCESS" for r in reloaded.get_processing_records())
        result = {"validation_type": "System/pipeline validation, not model accuracy", "passed": True,
                  "reports_processed": 3, "persisted_after_reload": len(reloaded.get_all_reports()),
                  "rag_documents": len(store.list_documents()), "rag_chunks": store.total_chunks(),
                  "idempotency": True, "priority_sorting": True, "json_serialization": True,
                  "full_step3_to_step6_outputs": True, "shared_embedder": True,
                  "processing_statuses": reloaded.get_summary_counts(),
                  "report_scores": [{"source_file": r.source_file, "score": r.priority_score} for r in reports]}
    after = hashlib.sha256(gold.read_bytes()).hexdigest()
    if before != after:
        raise RuntimeError("Gold changed during validation")
    result.update(gold_sha256_before=before, gold_sha256_after=after)
    return result


if __name__ == "__main__":
    result = validate()
    (ROOT / "emergency_data/step7_pipeline_validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

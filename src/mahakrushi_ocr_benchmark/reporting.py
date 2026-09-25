import csv
import json
from pathlib import Path

from .contracts import DocumentMetrics, OCRPrediction
from .ranking import rank_models


def write_report(predictions: list[OCRPrediction], metrics: list[DocumentMetrics], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows = [p.model_dump(mode="json") for p in predictions]
    metric_rows = [{**m.model_dump(exclude={"field_score"}), **m.field_score.model_dump()} for m in metrics]
    def write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}) if rows else ["model_id"])
            writer.writeheader(); writer.writerows(rows)
    write_csv(output_dir / "predictions.csv", prediction_rows)
    write_csv(output_dir / "document_metrics.csv", metric_rows)
    rows=[]
    for model_id in sorted({row["model_id"] for row in metric_rows}):
        group = [row for row in metric_rows if row["model_id"] == model_id]
        prediction_group=[p for p in predictions if p.model_id == model_id]
        rows.append({"model_id":model_id,"license_eligible":True,"runnable":any(prediction.success for prediction in prediction_group),"cer":sum(row["cer_whitespace"] for row in group)/len(group),"field_accuracy":sum(row["exact_matches"] for row in group)/max(sum(row["total_expected"] for row in group),1),"failure_rate":sum(not p.success for p in prediction_group)/max(len(prediction_group),1),"latency_seconds":sum(p.latency_seconds for p in prediction_group)/max(len(prediction_group),1),"vram_mb":0})
    summary={"leaderboard":rank_models(rows),"predictions":prediction_rows,"metrics":metric_rows}
    target=output_dir / "summary.json"; target.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"); return target

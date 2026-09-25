from pathlib import Path

from .adapters.base import OCRAdapter
from .contracts import DocumentRecord, OCRPrediction


def run_benchmark(documents: list[DocumentRecord], adapters: list[OCRAdapter], output_dir: Path, resume: bool = False) -> list[OCRPrediction]:
    output_dir.mkdir(parents=True, exist_ok=True); results = []
    for adapter in adapters:
        for document in documents:
            target = output_dir / f"{adapter.model_id}__{document.id}.json"
            if resume and target.exists():
                results.append(OCRPrediction.model_validate_json(target.read_text(encoding="utf-8"))); continue
            try: prediction = adapter.predict(document.image, document)
            except Exception as exc: prediction = OCRPrediction(model_id=adapter.model_id, document_id=document.id, latency_seconds=0, success=False, error_type=type(exc).__name__, error_message=str(exc))
            target.write_text(prediction.model_dump_json(), encoding="utf-8"); results.append(prediction)
    return results

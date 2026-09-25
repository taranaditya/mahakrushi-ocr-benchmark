import os
from pathlib import Path
from time import perf_counter

from .base import UnavailableAdapter
from ..contracts import DocumentRecord, OCRPrediction


class QwenVLAdapter(UnavailableAdapter):
    def __init__(self):
        super().__init__("qwen_vl", "Qwen2.5-VL-7B-Instruct", "Apache-2.0", "Apache-2.0")

    def predict(self, image_path: Path, document: DocumentRecord) -> OCRPrediction:
        if not os.getenv("OCR_BENCH_QWEN_URL"):
            return OCRPrediction(model_id=self.model_id, document_id=document.id, latency_seconds=0, success=False, error_type="missing_endpoint", error_message="Set OCR_BENCH_QWEN_URL after starting vLLM.")
        return super().predict(image_path, document)

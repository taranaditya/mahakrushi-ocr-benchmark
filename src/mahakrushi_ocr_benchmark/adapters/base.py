"""Adapters are lazy: imports and model weights are never loaded during probe."""
from abc import ABC
from pathlib import Path
from time import perf_counter

from ..contracts import DocumentRecord, ModelMetadata, OCRPrediction


class OCRAdapter(ABC):
    model_id = "base"
    metadata_value: ModelMetadata
    def load(self) -> None: pass
    def unload(self) -> None: pass
    def metadata(self) -> ModelMetadata: return self.metadata_value
    def predict(self, image_path: Path, document: DocumentRecord) -> OCRPrediction: raise NotImplementedError

class UnavailableAdapter(OCRAdapter):
    def __init__(self, model_id: str, display_name: str, weight_license: str, repository_license: str):
        self.model_id = model_id
        self.metadata_value = ModelMetadata(model_id=model_id, display_name=display_name, weight_license=weight_license, repository_license=repository_license)
    def predict(self, image_path: Path, document: DocumentRecord) -> OCRPrediction:
        start = perf_counter()
        return OCRPrediction(model_id=self.model_id, document_id=document.id, latency_seconds=perf_counter()-start, success=False, error_type="missing_dependency", error_message="Model runtime is not installed locally. Run its DGX smoke test first.")

class AdapterRegistry:
    def __init__(self, adapters: list[OCRAdapter]): self.adapters = {adapter.model_id: adapter for adapter in adapters}
    @classmethod
    def default(cls):
        return cls([
            UnavailableAdapter("paddleocr", "PaddleOCR PP-OCRv5", "Apache-2.0", "Apache-2.0"),
            UnavailableAdapter("indicphotoocr", "IndicPhotoOCR", "MIT", "MIT"),
            UnavailableAdapter("glm_ocr", "GLM-OCR 0.9B", "MIT", "MIT"),
            UnavailableAdapter("deepseek_ocr", "DeepSeek-OCR", "MIT", "MIT"),
            UnavailableAdapter("qwen_vl", "Qwen2.5-VL-7B-Instruct", "Apache-2.0", "Apache-2.0"),
        ])
    def ids(self): return self.adapters.keys()
    def get(self, model_id: str) -> OCRAdapter: return self.adapters[model_id]

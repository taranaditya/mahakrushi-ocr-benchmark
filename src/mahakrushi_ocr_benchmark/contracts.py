"""Typed contracts shared by all benchmark components."""
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

LanguageCode = Literal["mr", "hi", "en"]


class DocumentRecord(BaseModel):
    id: str
    image: Path
    ground_truth: Path
    languages: list[LanguageCode]
    document_type: str
    quality: list[str]
    source_kind: str
    license: str
    source_url: str | None = None
    critical_fields: dict[str, str | None] = Field(default_factory=dict)

    @field_validator("languages")
    @classmethod
    def unique_languages(cls, value: list[LanguageCode]) -> list[LanguageCode]:
        if not value or len(value) != len(set(value)):
            raise ValueError("languages must be non-empty and unique")
        return value


class ModelMetadata(BaseModel):
    model_id: str
    display_name: str
    revision: str = "not-installed"
    weight_license: str
    repository_license: str
    license_eligible: bool = True
    notes: str = ""


class OCRPrediction(BaseModel):
    model_id: str
    document_id: str
    raw_text: str = ""
    latency_seconds: float = Field(ge=0)
    success: bool
    structured_output: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None
    peak_vram_mb: float | None = Field(default=None, ge=0)
    prompt: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CriticalFieldScore(BaseModel):
    exact_matches: int = 0
    total_expected: int = 0
    missing: list[str] = Field(default_factory=list)
    hallucinated: list[str] = Field(default_factory=list)


class DocumentMetrics(BaseModel):
    model_id: str
    document_id: str
    cer_strict: float = Field(ge=0)
    cer_whitespace: float = Field(ge=0)
    wer_strict: float = Field(ge=0)
    wer_whitespace: float = Field(ge=0)
    field_score: CriticalFieldScore

"""Dataset manifest loading and validation."""
from pathlib import Path

from .contracts import DocumentRecord


class DatasetValidationError(ValueError):
    pass


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def load_manifest(path: Path, root: Path | None = None) -> list[DocumentRecord]:
    root = root or path.parent.parent
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = DocumentRecord.model_validate_json(line)
        except Exception as exc:
            raise DatasetValidationError(f"manifest line {number}: {exc}") from exc
        image, truth = _resolve(root, record.image), _resolve(root, record.ground_truth)
        if not image.exists():
            raise DatasetValidationError(f"{record.id}: missing image {image}")
        if not truth.exists():
            raise DatasetValidationError(f"{record.id}: missing ground truth {truth}")
        if not truth.read_text(encoding="utf-8").strip():
            raise DatasetValidationError(f"{record.id}: empty ground truth")
        records.append(record.model_copy(update={"image": image, "ground_truth": truth}))
    return records


def validate_dataset(records: list[DocumentRecord], root: Path | None = None) -> None:
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise DatasetValidationError("duplicate document IDs")
    if not 12 <= len(records) <= 15:
        raise DatasetValidationError("dataset must contain 12 to 15 pages")
    languages = {language for record in records for language in record.languages}
    required_quality = {"blur", "rotation", "compression", "handwriting", "mobile_photo"}
    quality = {item for record in records for item in record.quality}
    if languages != {"mr", "hi", "en"}:
        raise DatasetValidationError(f"language coverage incomplete: {languages}")
    if not required_quality <= quality:
        raise DatasetValidationError(f"quality coverage incomplete: {required_quality - quality}")

"""Validate the supplied Final test image/reference pairs and build one manifest."""
import hashlib
import json
import re
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Final test"
OUT = ROOT / "data/final/all"
GROUPS = {"en-05": "en", "go1": "hi", "mr-03": "mr"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def page_number(path: Path) -> int:
    match = re.fullmatch(r"page-(\d+)", path.stem)
    if not match:
        raise ValueError(f"Unexpected filename: {path}")
    return int(match.group(1))


def main() -> None:
    records = []
    validation = {"source": "user-supplied Final test", "groups": {}, "reference_review": "not independently visually verified"}
    for group, language in GROUPS.items():
        image_dir = SOURCE / "data" / group
        reference_dir = SOURCE / "expected_output" / group
        images = {page_number(path): path for path in image_dir.glob("*.png")}
        references = {page_number(path): path for path in reference_dir.glob("*.txt")}
        if not images or images.keys() != references.keys():
            raise ValueError(f"Pairing mismatch in {group}: images={sorted(images)} references={sorted(references)}")
        validation["groups"][group] = {"language": language, "pages": len(images)}
        for number in sorted(images):
            image, reference = images[number], references[number]
            with Image.open(image) as opened:
                opened.verify()
            text = reference.read_text(encoding="utf-8-sig")
            if not text.strip():
                raise ValueError(f"Empty reference: {reference}")
            records.append({
                "id": f"final-{group}-page-{number:02d}",
                "image": image.relative_to(ROOT).as_posix(),
                "ground_truth": reference.relative_to(ROOT).as_posix(),
                "languages": [language],
                "document_type": "government_policy_document",
                "quality": ["scanned", "full_page"],
                "source_kind": "user_provided_document",
                "license": "reuse_rights_not_specified",
                "critical_fields": {},
                "image_sha256": digest(image),
                "reference_sha256": digest(reference),
                "reference_has_editorial_brackets": bool(re.search(r"\[[^]]+\]", text)),
            })
    if len(records) != 31:
        raise ValueError(f"Expected 31 pages, found {len(records)}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
    )
    validation["pages"] = len(records)
    validation["pages_with_editorial_brackets"] = sum(r["reference_has_editorial_brackets"] for r in records)
    (OUT / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

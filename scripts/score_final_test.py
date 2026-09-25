"""Score all 31 final-test pages, counting failed/missing pages in the denominator."""
import argparse
import csv
import hashlib
import json
import statistics
import unicodedata
from pathlib import Path

from rapidfuzz.distance import Levenshtein
import regex


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def compact(text: str) -> str:
    return " ".join(nfc(text).split())


OCRD_IGNORABLE = {
    "\ufeff",  # byte-order mark
    "\u061c", "\u200e", "\u200f",  # Arabic letter mark, LRM, RLM
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",  # bidi embeddings/overrides
    "\u2066", "\u2067", "\u2068", "\u2069",  # bidi isolates
}


def ocrd_graphemes(text: str) -> list[str]:
    """OCR-D-style NFC grapheme units; retain whitespace, case, punctuation, and script marks."""
    text = nfc("".join(char for char in text if char not in OCRD_IGNORABLE))
    return regex.findall(r"\X", text)


def ocrd_character_counts(reference: str, hypothesis: str) -> tuple[int, int, int]:
    """Return edit errors, correct grapheme matches, and reference grapheme count."""
    ref = ocrd_graphemes(reference)
    hyp = ocrd_graphemes(hypothesis)
    edits = Levenshtein.editops(ref, hyp)
    errors = len(edits)
    wrong_reference_units = sum(edit.tag in {"replace", "delete"} for edit in edits)
    correct = len(ref) - wrong_reference_units
    return errors, correct, len(ref)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def aggregate(rows: list[dict]) -> dict:
    count = len(rows)
    successes = sum(row["success"] for row in rows)
    chars = sum(row["reference_graphemes"] for row in rows)
    errors = sum(row["character_errors"] for row in rows)
    correct = sum(row["correct_graphemes"] for row in rows)
    words = sum(row["reference_words"] for row in rows)
    latencies = sorted(row["latency_seconds"] for row in rows if row["attempted"])
    successful = [row for row in rows if row["success"]]
    successful_chars = sum(row["reference_graphemes"] for row in successful)
    successful_errors = sum(row["character_errors"] for row in successful)
    successful_correct = sum(row["correct_graphemes"] for row in successful)
    return {
        "pages": count,
        "attempted_pages": sum(row["attempted"] for row in rows),
        "successful_pages": successes,
        "failed_or_missing_pages": count - successes,
        "character_errors_total": errors,
        "correct_graphemes_total": correct,
        "reference_graphemes_total": chars,
        "cer_all_pages": errors / chars if chars else None,
        "cer_success_only": sum(row["character_errors"] for row in successful) / successful_chars if successful_chars else None,
        "cer_normalized_ocrd_all_pages": errors / (errors + correct) if errors + correct else None,
        "cer_normalized_ocrd_success_only": (
            successful_errors / (successful_errors + successful_correct)
            if successful_errors + successful_correct else None
        ),
        "wer_all_pages": sum(row["word_errors"] for row in rows) / words if words else None,
        "normalized_exact_match_rate": sum(row["normalized_exact_match"] for row in rows) / count if count else None,
        "mean_latency_seconds": statistics.fmean(latencies) if latencies else None,
        "p50_latency_seconds": statistics.median(latencies) if latencies else None,
        "p95_latency_seconds": latencies[round((len(latencies) - 1) * .95)] if latencies else None,
        "editorial_bracket_pages": sum(row["reference_has_editorial_brackets"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", default="gpu_requested")
    args = parser.parse_args()

    root = args.manifest.resolve().parents[3]
    records = read_jsonl(args.manifest)
    if len(records) != 31 or len({r["id"] for r in records}) != 31:
        raise ValueError("Final-test manifest must contain 31 unique pages")
    latest = {}
    for prediction in read_jsonl(args.predictions):
        if prediction.get("model_id") == args.model_id:
            latest[prediction["document_id"]] = prediction
    pages = []
    for record in records:
        reference_path = root / record["ground_truth"]
        if not reference_path.is_file():
            raise FileNotFoundError(reference_path)
        image_path = root / record["image"]
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != record["image_sha256"]:
            raise ValueError(f"Image changed since validation: {image_path}")
        if hashlib.sha256(reference_path.read_bytes()).hexdigest() != record["reference_sha256"]:
            raise ValueError(f"Reference changed since validation: {reference_path}")
        reference = reference_path.read_text(encoding="utf-8-sig")
        prediction = latest.get(record["id"])
        success = bool(prediction and prediction.get("success")
                       and str(prediction.get("raw_text") or "").strip())
        output = str(prediction.get("raw_text") or "") if success else ""
        ref_compact, out_compact = compact(reference), compact(output)
        character_errors, correct_graphemes, reference_graphemes = ocrd_character_counts(reference, output)
        ref_words, out_words = ref_compact.split(), out_compact.split()
        pages.append({
            "document_id": record["id"],
            "language": record["languages"][0],
            "attempted": prediction is not None,
            "success": success,
            "error": None if success else (prediction or {}).get("error_message") or "empty or missing prediction",
            "latency_seconds": float((prediction or {}).get("latency_seconds", 0)),
            "reference_characters": len(ref_compact),
            "reference_graphemes": reference_graphemes,
            "reference_words": len(ref_words),
            "character_errors": character_errors,
            "correct_graphemes": correct_graphemes,
            "word_errors": Levenshtein.distance(ref_words, out_words),
            "normalized_exact_match": int(success and ref_compact == out_compact),
            "reference_has_editorial_brackets": record.get("reference_has_editorial_brackets", False),
        })

    summary = aggregate(pages)
    result = {
        "model_id": args.model_id,
        "status": ("complete" if summary["successful_pages"] == 31 else
                   "partial" if summary["successful_pages"] else "failed"),
        "runtime": args.runtime,
        "dataset": "Final test / 31 user-supplied policy pages",
        "reference_review": "user-supplied; not independently visually verified",
        "methodology": (
            "CER uses NFC-normalized Unicode grapheme clusters, retaining whitespace, case, and punctuation; "
            "only BOM and bidirectional control marks are ignored. CER is total Levenshtein edits divided "
            "by reference grapheme clusters. WER and exact match use whitespace-compacted text. "
            "Failed or missing pages count as empty predictions. "
            "CER/WER can exceed 100% when a model inserts text. No field-extraction labels "
            "were supplied, so this is not a form-filling accuracy score."
        ),
        "normalized_cer_methodology": (
            "OCR-D-style bounded CER-N: E/(E+C), where E is the minimum grapheme-level Levenshtein "
            "edit count (substitutions + deletions + insertions) and C is the number of correctly "
            "aligned grapheme clusters. Apply NFC and OCR-D-style grapheme segmentation; retain "
            "whitespace, punctuation, case, and Indic marks. This is bounded 0–100%; failed or missing "
            "pages count as empty predictions. Strict reference-length CER remains the primary ranking metric."
        ),
        "summary": summary,
        "by_language": {language: aggregate([p for p in pages if p["language"] == language])
                        for language in ("en", "hi", "mr")},
        "pages": pages,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.output.with_name(args.model_id + "_pages.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pages[0]))
        writer.writeheader()
        writer.writerows(pages)

    scorecards = []
    for path in args.output.parent.glob("*_scorecard.json"):
        scorecards.append(json.loads(path.read_text(encoding="utf-8")))
    scorecards.sort(key=lambda card: (card["status"] != "complete",
                                      card["summary"]["cer_all_pages"] is None,
                                      card["summary"]["cer_all_pages"]
                                      if card["summary"]["cer_all_pages"] is not None else float("inf"),
                                      card["model_id"]))
    leaderboard = []
    complete_rank = 0
    for card in scorecards:
        if card["status"] == "complete":
            complete_rank += 1
            rank = complete_rank
        else:
            rank = None
        leaderboard.append({"rank": rank, "model_id": card["model_id"], "status": card["status"],
                            "runtime": card["runtime"], **card["summary"]})
    (args.output.parent / "leaderboard.json").write_text(
        json.dumps(leaderboard, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output.parent / "leaderboard.csv").open("w", encoding="utf-8", newline="") as stream:
        # Scorecards can be at different schema revisions during a bulk rescore.
        # Union their columns so an older card cannot break writing the CSV.
        fieldnames = list(dict.fromkeys(key for row in leaderboard for key in row))
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(leaderboard)
    normalized_cer = summary["cer_normalized_ocrd_all_pages"]
    print(
        f"{args.model_id}: {summary['successful_pages']}/31 successful, "
        f"CER={summary['cer_all_pages']:.4f}, normalized CER="
        f"{normalized_cer:.4f}" if normalized_cer is not None else
        f"{args.model_id}: {summary['successful_pages']}/31 successful, "
        f"CER={summary['cer_all_pages']:.4f}, normalized CER=N/A"
    )


if __name__ == "__main__":
    main()

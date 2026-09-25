"""Run EasyOCR over a formal manifest and write resumable OCR predictions."""
import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # EasyOCR ships an English reader and a shared Devanagari reader.  Marathi
    # is not a separately selectable language in this release, so Hindi is the
    # closest supported Devanagari recognizer and this limitation is recorded.
    import easyocr
    reader = easyocr.Reader(["en", "hi"], gpu="cuda")
    root = args.manifest.parents[3]
    records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed = set()
    if args.resume and args.output.exists():
        completed = {json.loads(line)["document_id"] for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as stream:
        for record in records:
            if record["id"] in completed:
                continue
            started = time.perf_counter()
            try:
                text = "\n".join(reader.readtext(str(root / record["image"]), detail=0, paragraph=False))
                row = {"model_id": "easyocr", "document_id": record["id"], "success": True,
                       "latency_seconds": time.perf_counter() - started, "raw_text": text}
            except Exception as error:
                row = {"model_id": "easyocr", "document_id": record["id"], "success": False,
                       "latency_seconds": time.perf_counter() - started, "error_message": repr(error)}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(record["id"], row["success"], round(row["latency_seconds"], 2), flush=True)


if __name__ == "__main__":
    main()

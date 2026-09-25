"""Run IndicPhotoOCR's full-page detector/identifier/recognizer pipeline on GPU."""
import argparse
import json
import time
from pathlib import Path


def flatten(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "txt" in value:
            return str(value["txt"])
        return "\n".join(filter(None, (flatten(item) for item in value.values())))
    if isinstance(value, (list, tuple)):
        return "\n".join(filter(None, (flatten(item) for item in value)))
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from IndicPhotoOCR.ocr import OCR

    root = args.manifest.parents[3]
    records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed = set()
    if args.resume and args.output.exists():
        completed = {row["document_id"] for line in args.output.read_text(encoding="utf-8").splitlines()
                     if line.strip() for row in [json.loads(line)] if row.get("success")}

    ocr = OCR(device="cuda:0", identifier_lang="auto", verbose=False, detector="textbpnpp")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as stream:
        for record in records:
            if record["id"] in completed:
                continue
            started = time.perf_counter()
            try:
                detected_lines = ocr.ocr(str(root / record["image"]), batch_size=8)
                text = flatten(detected_lines)
                row = {"model_id": "indicphotoocr", "document_id": record["id"], "success": True,
                       "latency_seconds": time.perf_counter() - started, "raw_text": text}
            except Exception as error:
                row = {"model_id": "indicphotoocr", "document_id": record["id"], "success": False,
                       "latency_seconds": time.perf_counter() - started, "raw_text": "", "error_message": repr(error)}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(record["id"], row["success"], round(row["latency_seconds"], 2), flush=True)


if __name__ == "__main__":
    main()

"""Run IndicOCR on the shared 31-page final OCR dataset using CUDA."""

import argparse
import json
import sys
import time
from pathlib import Path


MODEL_ID = "indic_ocr"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models/indic-ocr"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to run this GPU benchmark on CPU")
    torch.cuda.set_device(0)
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

    root = args.manifest.resolve().parents[3]
    records = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 31 or len({record["id"] for record in records}) != 31:
        raise ValueError("Expected exactly 31 unique pages in the final-test manifest")

    completed = set()
    if args.resume and args.output.exists():
        completed = {
            json.loads(line)["document_id"]
            for line in args.output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    sys.path.insert(0, str(args.model_dir.resolve()))
    from indic_ocr import IndicOCR

    # IndicOCR places both the layout detector and recognizer on this CUDA device.
    ocr = IndicOCR.from_pretrained(
        str(args.model_dir.resolve()), device="cuda:0", table_format="markdown"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as stream:
        for index, record in enumerate(records, 1):
            if record["id"] in completed:
                continue
            started = time.perf_counter()
            try:
                page = ocr.parse(str(root / record["image"]))
                text = str(page.get("markdown") or "")
                success = bool(text.strip())
                row = {
                    "model_id": MODEL_ID,
                    "document_id": record["id"],
                    "success": success,
                    "latency_seconds": time.perf_counter() - started,
                    "raw_text": text,
                }
                if not success:
                    row["error_message"] = "IndicOCR returned empty Markdown"
            except Exception as error:  # Keep the same one-row-per-page failure accounting.
                row = {
                    "model_id": MODEL_ID,
                    "document_id": record["id"],
                    "success": False,
                    "latency_seconds": time.perf_counter() - started,
                    "raw_text": "",
                    "error_message": repr(error),
                }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(
                f"{index}/31 {record['id']} {row['success']} "
                f"{row['latency_seconds']:.2f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()

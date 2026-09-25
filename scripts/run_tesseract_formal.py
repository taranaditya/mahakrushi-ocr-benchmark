"""Native Tesseract formal runner; intended to execute inside the ARM64 container."""
import argparse
import json
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.manifest[0].parents[3]
    records = [json.loads(line) for manifest in args.manifest
               for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
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
                result = subprocess.run(
                    ["tesseract", str(root / record["image"]), "stdout", "-l", "eng+hin+mar", "--psm", "7"],
                    text=True, capture_output=True, timeout=120, check=True,
                )
                row = {"model_id": "tesseract", "document_id": record["id"], "success": True,
                       "latency_seconds": time.perf_counter() - started, "raw_text": result.stdout.strip()}
            except Exception as error:
                row = {"model_id": "tesseract", "document_id": record["id"], "success": False,
                       "latency_seconds": time.perf_counter() - started, "error_message": repr(error)}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(record["id"], row["success"], round(row["latency_seconds"], 2), flush=True)


if __name__ == "__main__":
    main()

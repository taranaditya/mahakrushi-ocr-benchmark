"""Run a fixed OCR manifest against an OpenAI-compatible vision endpoint."""
import argparse
import base64
import json
import mimetypes
import time
from urllib.error import HTTPError
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--prompt", default="Transcribe all text exactly in reading order.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.manifest[0].parents[3]
    records = [json.loads(line) for manifest in args.manifest
               for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed = set()
    if args.resume and args.output.exists():
        completed = {row["document_id"] for line in args.output.read_text(encoding="utf-8").splitlines()
                     if line.strip() for row in [json.loads(line)] if row.get("success")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as stream:
        for record in records:
            if record["id"] in completed:
                continue
            image = root / record["image"]
            mime = mimetypes.guess_type(image.name)[0] or "image/png"
            payload = {
                "model": args.model,
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(image.read_bytes()).decode()}"}},
                    {"type": "text", "text": args.prompt},
                ]}],
            }
            started = time.perf_counter()
            try:
                request = urllib.request.Request(args.url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=args.timeout) as response:
                    data = json.load(response)
                row = {"model_id": args.model_id, "document_id": record["id"], "success": True,
                       "latency_seconds": time.perf_counter() - started,
                       "raw_text": data["choices"][0]["message"]["content"]}
            except HTTPError as error:
                details = error.read().decode("utf-8", errors="replace")[:2000]
                row = {"model_id": args.model_id, "document_id": record["id"], "success": False,
                       "latency_seconds": time.perf_counter() - started,
                       "error_message": f"HTTP {error.code}: {details}"}
            except Exception as error:  # Raw failure is part of the benchmark result.
                row = {"model_id": args.model_id, "document_id": record["id"], "success": False,
                       "latency_seconds": time.perf_counter() - started, "error_message": repr(error)}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(record["id"], row["success"], round(row["latency_seconds"], 2), flush=True)


if __name__ == "__main__":
    main()

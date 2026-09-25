"""Native ARM CPU Paddle pipelines, with explicit language routing and run metadata."""
import argparse
import json
import os
import signal
import time
from pathlib import Path


def extract_text(result, model):
    data = result.get("res", result)
    if model == "paddleocr":
        return "\n".join(data["rec_texts"])
    return "\n".join(block["block_content"] for block in data["parsing_res_list"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["paddleocr", "pp_structure_v3"], required=True)
    p.add_argument("--manifest", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from paddleocr import PaddleOCR, PPStructureV3
    root = args.manifest[0].resolve().parents[3]
    records = [json.loads(line) for path in args.manifest for line in path.read_text().splitlines() if line.strip()]
    if len(records) != 500 or len({r['id'] for r in records}) != 500:
        raise ValueError("Expected exactly 500 unique formal samples")
    for record in records:
        if not (root / record["image"]).is_file():
            raise FileNotFoundError(record["image"])
    pipelines = {}
    for language in ("hi", "en"):
        common = dict(device="cpu", cpu_threads=2, enable_mkldnn=False,
                      use_doc_orientation_classify=False, use_doc_unwarping=False)
        if args.model == "paddleocr":
            pipelines[language] = PaddleOCR(lang=language, ocr_version="PP-OCRv5",
                                           use_textline_orientation=False, **common)
        else:
            rec = "en_PP-OCRv5_mobile_rec" if language == "en" else "devanagari_PP-OCRv5_mobile_rec"
            pipelines[language] = PPStructureV3(text_detection_model_name="PP-OCRv5_server_det",
                                               text_recognition_model_name=rec, **common)
        warmup = next(r for r in records if r["languages"][0] == language)
        list(pipelines[language].predict(str(root / warmup["image"])))
    print("STARTUP_AND_WARMUP_PASSED", flush=True)
    def timeout(*_):
        raise TimeoutError("Per-image 120 second timeout")
    signal.signal(signal.SIGALRM, timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental appending to or overwriting a prior run.
    with args.output.open("x", encoding="utf-8") as out:
        for record in records:
            started = time.perf_counter()
            row = {"model_id": args.model, "document_id": record["id"], "device": "cpu",
                   "language_routing": "manifest language; Hindi/Marathi share Devanagari recognizer"}
            try:
                signal.alarm(120)
                language = "en" if record["languages"][0] == "en" else "hi"
                results = list(pipelines[language].predict(str(root / record["image"])))
                text = "\n".join(extract_text(r.json, args.model) for r in results)
                row.update(success=True, raw_text=text)
            except Exception as exc:
                row.update(success=False, raw_text="", error_message=repr(exc))
            finally:
                signal.alarm(0)
            row["latency_seconds"] = time.perf_counter() - started
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(record["id"], row["success"], round(row["latency_seconds"], 3), flush=True)


if __name__ == "__main__":
    main()

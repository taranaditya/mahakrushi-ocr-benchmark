"""GPU-only page runner for models requiring their documented Transformers loader."""

import argparse
import json
import sys
import time
from pathlib import Path


NEMOTRON_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"


def pending_records(manifest: Path, output: Path, resume: bool) -> list[dict]:
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not resume or not output.exists():
        return records
    successful = {
        row["document_id"]
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
        if row.get("success")
    }
    return [record for record in records if record["id"] not in successful]


def load_jina(model_dir: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda").eval()

    def infer(image, max_new_tokens: int) -> str:
        inputs = processor.prepare_ocr_inputs(image, device=torch.device("cuda"))
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return processor.decode_ocr(output, inputs["input_ids"])

    return infer


def load_nemotron(model_dir: Path):
    import torch
    from transformers import AutoModel, AutoProcessor, GenerationConfig

    sys.path.insert(0, str(model_dir))
    from postprocessing import extract_classes_bboxes, postprocess_text

    model = AutoModel.from_pretrained(
        str(model_dir), trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    generation_config = GenerationConfig.from_pretrained(str(model_dir), trust_remote_code=True)

    def infer(image, max_new_tokens: int) -> str:
        inputs = processor(
            images=[image], text=NEMOTRON_PROMPT, return_tensors="pt", add_special_tokens=False
        ).to("cuda")
        with torch.inference_mode():
            output = model.generate(
                **inputs, generation_config=generation_config, max_new_tokens=max_new_tokens
            )
        raw = processor.batch_decode(output, skip_special_tokens=True)[0]
        classes, _, texts = extract_classes_bboxes(raw)
        if not texts:
            return raw
        return "\n".join(
            postprocess_text(text, cls=cls, text_format="plain", table_format="markdown")
            for text, cls in zip(texts, classes)
        )

    return infer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("jina", "nemotron"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=6144)
    parser.add_argument("--limit", type=int, default=0, help="Run only this many pending pages; zero means all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this benchmark")
    records = pending_records(args.manifest, args.output, args.resume)
    if args.limit:
        records = records[: args.limit]
    if not records:
        print("No pending pages", flush=True)
        return
    root = args.manifest.resolve().parents[3]
    infer = load_jina(args.model_dir) if args.backend == "jina" else load_nemotron(args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as stream:
        for record in records:
            started = time.perf_counter()
            try:
                image = Image.open(root / record["image"]).convert("RGB")
                raw_text = infer(image, args.max_new_tokens)
                row = {
                    "model_id": args.model_id,
                    "document_id": record["id"],
                    "success": bool(raw_text.strip()),
                    "latency_seconds": time.perf_counter() - started,
                    "raw_text": raw_text,
                }
            except Exception as error:
                row = {
                    "model_id": args.model_id,
                    "document_id": record["id"],
                    "success": False,
                    "latency_seconds": time.perf_counter() - started,
                    "error_message": repr(error),
                }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(record["id"], row["success"], round(row["latency_seconds"], 2), flush=True)


if __name__ == "__main__":
    main()

# Reproduce the 31-page OCR benchmark

This guide covers the final English, Hindi, and Marathi page test. The source images and manually prepared reference text are included in `Final test/`. They were user supplied for this evaluation; no reuse license is granted for the underlying documents. References have not been independently visually verified. The score measures transcription, not form field extraction.

## Layout

The checkout includes:

```text
Final test/
  data/en-05/page-01.png ... page-06.png
  data/go1/page-1.png, page-4.png, page-5.png
  data/mr-03/page-01.png ... page-22.png
  expected_output/<matching group>/page-XX.txt
```

Install the scorer dependencies in a Python 3.11+ environment:

```bash
python -m pip install '.[final]'
python scripts/prepare_final_test.py
```

The preparation step validates that all 31 pages have nonempty references, verifies the image files, and writes `data/final/all/manifest.jsonl` with image and reference hashes. Generated manifests and run outputs are ignored by Git.

## Run OCR

Use the manifest with one of the published runners. All write JSON Lines with model ID, page ID, success, elapsed seconds, and extracted text. For a model served through a local OpenAI compatible vLLM vision endpoint:

```bash
python scripts/run_openai_compatible.py \
  --manifest data/final/all/manifest.jsonl \
  --output results/final_test/example_predictions.jsonl \
  --url http://127.0.0.1:8137/v1/chat/completions \
  --model example --model-id example \
  --prompt 'Transcribe all visible text in reading order. Preserve the language and script.' \
  --max-tokens 6144 --resume
```

Start the model endpoint separately using its documented runtime and local model files. The model weights and GPU environments are not distributed here. `run_easyocr_formal.py`, `run_indicocr_final.py`, `run_indicphoto_policy.py`, and `run_direct_ocr.py` cover the other backends tested. Their command line help lists required paths and options. `run_paddle_native.py` and `run_tesseract_formal.py` are earlier CPU comparison runners, not the final GPU queue.

For a locally stored model that vLLM can serve, `scripts/run_final_vlm.sh` starts the server, verifies that a GPU process appears, runs the 31 pages, scores the result, and stops only the server it started. Call it with a model ID, model directory, and OCR prompt, for example `bash scripts/run_final_vlm.sh glm_ocr models/glm-ocr 'Text Recognition:'`. It runs one model at a time and writes results to `results/final_test/`.

## Score predictions

```bash
python scripts/score_final_test.py \
  --manifest data/final/all/manifest.jsonl \
  --predictions results/final_test/example_predictions.jsonl \
  --model-id example --runtime vllm_cuda \
  --output results/final_test/example_scorecard.json
```

The scorer checks source/reference hashes, counts failed or missing pages as empty output, writes per-page CSV plus JSON scorecards and a leaderboard, and reports strict CER, normalized CER, WER, exact match, and per-language results. The dashboard ships a previously generated scorecard snapshot in `final-dashboard/src/data.json`. Rerunning a model does not automatically edit that snapshot.

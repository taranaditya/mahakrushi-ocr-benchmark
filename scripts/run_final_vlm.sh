#!/usr/bin/env bash
# Run one locally stored vision model against the 31-page manifest on a CUDA DGX.
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo 'Usage: run_final_vlm.sh MODEL_ID MODEL_DIRECTORY PROMPT [MAX_TOKENS]' >&2
  exit 2
fi

MODEL_ID="$1"
MODEL_DIRECTORY="$2"
PROMPT="$3"
MAX_TOKENS="${4:-6144}"
ROOT="${OCR_BENCH_DGX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
PYTHON="${OCR_BENCH_DGX_PYTHON:-$ROOT/.venv/bin/python}"
VLLM="${OCR_BENCH_VLLM:-$ROOT/.venv/bin/vllm}"
PORT="${OCR_BENCH_BENCHMARK_PORT:-8137}"
MANIFEST="$ROOT/data/final/all/manifest.jsonl"
OUTPUT="$ROOT/results/final_test"

[[ "$MODEL_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid model ID.' >&2; exit 2; }
[[ -s "$MANIFEST" ]] || { echo 'Prepare the 31-page manifest first.' >&2; exit 2; }
[[ "$(wc -l < "$MANIFEST")" -eq 31 ]] || { echo 'Expected 31 manifest records.' >&2; exit 2; }
[[ -d "$MODEL_DIRECTORY" ]] || { echo 'Model directory is missing.' >&2; exit 2; }
[[ -x "$PYTHON" && -x "$VLLM" ]] || { echo 'Python or vLLM executable is missing.' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi is unavailable.' >&2; exit 2; }
mkdir -p "$OUTPUT/logs"
exec 9>"$OUTPUT/queue.lock"
flock -n 9 || { echo 'Another final benchmark is running.' >&2; exit 1; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
  echo 'GPU is in use. Wait for the current job to finish.' >&2
  exit 1
fi
if ss -ltn "sport = :$PORT" | grep -q LISTEN; then
  echo 'Benchmark port is already in use.' >&2
  exit 1
fi

SERVER_PID=''
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting $MODEL_ID on GPU..."
setsid "$VLLM" serve "$MODEL_DIRECTORY" --served-model-name "$MODEL_ID" \
  --host 127.0.0.1 --port "$PORT" --gpu-memory-utilization 0.72 \
  --max-model-len 8192 --max-num-seqs 1 \
  >"$OUTPUT/logs/${MODEL_ID}_server.log" 2>&1 &
SERVER_PID=$!
READY=0
for _ in $(seq 1 120); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then READY=1; break; fi
  sleep 5
done
[[ "$READY" -eq 1 ]] || { echo "Model server failed; check $OUTPUT/logs/${MODEL_ID}_server.log" >&2; exit 1; }
if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
  echo 'No GPU process detected; benchmark was not started.' >&2
  exit 1
fi

"$PYTHON" "$ROOT/scripts/run_openai_compatible.py" \
  --manifest "$MANIFEST" --output "$OUTPUT/${MODEL_ID}_predictions.jsonl" \
  --url "http://127.0.0.1:$PORT/v1/chat/completions" --model "$MODEL_ID" \
  --model-id "$MODEL_ID" --prompt "$PROMPT" --max-tokens "$MAX_TOKENS" \
  --timeout 360 --resume | tee "$OUTPUT/logs/${MODEL_ID}_run.log"

"$PYTHON" "$ROOT/scripts/score_final_test.py" \
  --manifest "$MANIFEST" --predictions "$OUTPUT/${MODEL_ID}_predictions.jsonl" \
  --model-id "$MODEL_ID" --runtime vllm_cuda \
  --output "$OUTPUT/${MODEL_ID}_scorecard.json"

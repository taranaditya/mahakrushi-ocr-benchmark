#!/usr/bin/env bash
set -euo pipefail

ROOT="${OCR_BENCH_DGX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VLLM="${OCR_BENCH_VLLM:-$ROOT/.venv/bin/vllm}"
PYTHON="${OCR_BENCH_DGX_PYTHON:-$ROOT/.venv/bin/python}"
QWEN25_MODEL="${OCR_BENCH_QWEN25_MODEL:-}"
LOG="$ROOT/results/testing/warm-pool"
mkdir -p "$LOG"

if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
  echo 'GPU is already in use; warm pool was not started.' >&2
  exit 1
fi
if tmux has-session -t mahakrushi-failed-retry 2>/dev/null || tmux has-session -t mahakrushi-additional-test 2>/dev/null; then
  echo 'A benchmark queue is active; warm pool was not started.' >&2
  exit 1
fi

pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

start_model() {
  local id="$1" path="$2" port="$3" utilization="$4"
  shift 4
  echo "Starting $id on 127.0.0.1:$port"
  setsid "$VLLM" serve "$path" --served-model-name "$id" \
    --host 127.0.0.1 --port "$port" --gpu-memory-utilization "$utilization" \
    --max-model-len 8192 --max-num-seqs 1 "$@" >"$LOG/$id.log" 2>&1 &
  local pid=$!
  pids+=("$pid")
  for _ in $(seq 1 120); do
    if ! kill -0 "$pid" 2>/dev/null; then echo "$id failed; see $LOG/$id.log" >&2; return 1; fi
    if curl -fsS --max-time 2 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      echo "$id ready (PID $pid)"
      return 0
    fi
    sleep 5
  done
  echo "$id startup timed out; see $LOG/$id.log" >&2
  return 1
}

if [[ -z "$QWEN25_MODEL" || ! -d "$QWEN25_MODEL" ]]; then
  echo 'Set OCR_BENCH_QWEN25_MODEL to the local Qwen2.5-VL-7B model directory.' >&2
  exit 2
fi
start_model qwen2_5_vl_7b "$QWEN25_MODEL" 8141 0.30 --trust-remote-code
echo 'Starting easyocr on 127.0.0.1:8142'
setsid "$PYTHON" "$ROOT/final-dashboard-service/warm_easyocr_service.py" >"$LOG/easyocr.log" 2>&1 &
easyocr_pid=$!
pids+=("$easyocr_pid")
for _ in $(seq 1 60); do
  if ! kill -0 "$easyocr_pid" 2>/dev/null; then echo "easyocr failed; see $LOG/easyocr.log" >&2; exit 1; fi
  if curl -fsS --max-time 2 'http://127.0.0.1:8142/health' >/dev/null 2>&1; then break; fi
  sleep 5
done
curl -fsS --max-time 2 'http://127.0.0.1:8142/health' >/dev/null
echo "easyocr ready (PID $easyocr_pid)"
GLM_MODEL="${OCR_BENCH_GLM_MODEL:-$ROOT/models/glm-ocr}"
start_model glm_ocr "$GLM_MODEL" 8143 0.22
echo 'Warm OCR pool ready: qwen2_5_vl_7b, easyocr, glm_ocr'
wait

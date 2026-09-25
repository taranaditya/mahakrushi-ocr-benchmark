# MahaKrushi OCR Dashboard

The current dashboard contains the latest OCR benchmark scorecards and a Testing page for live document OCR.

## Dashboard

Use Node.js 22.12+ (or 20.19+) and npm. On any computer:

```powershell
git clone https://github.com/taranaditya/mahakrushi-ocr-benchmark.git
cd mahakrushi-ocr-benchmark
cd final-dashboard
npm ci
npm run build
npm run dev -- --host 127.0.0.1 --port 8765 --strictPort
```

Open `http://127.0.0.1:8765`. The scorecard is a reviewed snapshot in `final-dashboard/src/data.json`; running the dashboard does not start benchmark jobs.

## Live Testing page

The Testing page needs the local bridge and an SSH-accessible DGX that already has the selected model and its runtime installed. The bridge binds to loopback only. It forwards uploaded document pages to the configured DGX, displays OCR text and a lightweight structure, and removes each temporary DGX upload after the request. Only upload documents that you are authorized to process.

Install the local bridge dependencies:

```powershell
python -m venv .venv-bridge
.\.venv-bridge\Scripts\Activate.ps1
python -m pip install -r final-dashboard-service/requirements.txt
```

The bridge needs Python 3.11+ and the OpenSSH client. On Linux or macOS, create the environment with `python3 -m venv .venv-bridge` and activate it with `source .venv-bridge/bin/activate`.

Set the connection values in the PowerShell session that will run the bridge. Use your own approved host, account, absolute DGX project path, and SSH key path; keep them out of Git:

```powershell
$env:OCR_BENCH_DGX_HOST = "<DGX host or Tailscale IP>"
$env:OCR_BENCH_DGX_USER = "<DGX account>"
$env:OCR_BENCH_DGX_PORT = "22"
$env:OCR_BENCH_DGX_ROOT = "/home/<DGX account>/<project directory>"
$env:OCR_BENCH_SSH_KEY_PATH = "$env:USERPROFILE\.ssh\<private-key-file>"
python .\final-dashboard-service\local_api.py
```

On Linux or macOS, set the same variables with `export OCR_BENCH_DGX_HOST=...` and so on, then run `python final-dashboard-service/local_api.py`. The dashboard runs on each person's own computer; its local ports do not conflict with another person's computer.

Run the dashboard in another terminal. For Tailscale Serve, add the exact dashboard origin to `OCR_BENCH_ALLOWED_ORIGINS` before starting the bridge and follow the private service configuration in `final-dashboard-service/README.md`. Do not expose the bridge directly to the public internet.

## Scope and privacy

This repository contains the latest dashboard, its three Testing-page examples, the 31-page final OCR test set, and the benchmark and live testing code. It does not contain model weights, saved inference outputs, API keys, SSH keys, or DGX connection values. Live OCR uploads go to the configured DGX; they are not sent to a hosted OCR provider by this code.

The benchmark snapshot is for model comparison and does not represent a production deployment or form-filling decision system. The dashboard's source and bundled Data App runtime have no license grant in this repository; ask the project owner before reusing or redistributing them.

## 31-page benchmark code

The `src/mahakrushi_ocr_benchmark/` package contains the original dataset contracts, model adapters, metrics, ranking, and reporting utilities. The final 31-page evaluation uses `scripts/prepare_final_test.py` and `scripts/score_final_test.py`; OCR runners are in `scripts/`. See [the benchmark guide](docs/final-benchmark.md) for reproducible commands and model-specific prerequisites.

The 31 source pages and reference transcriptions are in `Final test/` so the scorecards can be reproduced. They were supplied for this evaluation; no reuse license is granted for the underlying documents. Three reference pages include editorial brackets, and references have not been independently visually verified.

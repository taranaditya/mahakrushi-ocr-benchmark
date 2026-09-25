# Live OCR bridge

`local_api.py` runs on the user's computer and binds to `127.0.0.1:8766`. The dashboard runs locally on port 8765. The bridge requires Pillow and pypdfium2, plus OpenSSH. Configure `OCR_BENCH_DGX_HOST`, `OCR_BENCH_DGX_USER`, `OCR_BENCH_DGX_ROOT`, and `OCR_BENCH_SSH_KEY_PATH` in the process environment before launching it. `.env.example` at the repository root documents optional values; the bridge does not load that file automatically.

The Testing page exposes only the four configured demo models. DGX model files and runtimes are external prerequisites; they are not tracked here. On the DGX, the project root is derived from the runner's location unless `OCR_BENCH_DGX_ROOT` is set. Optional runtime overrides select the Python, vLLM executable, and model directories.

To share the dashboard privately over Tailscale, use Tailscale Serve as a same-origin reverse proxy for the dashboard and bridge. Set `OCR_BENCH_ALLOWED_ORIGINS` to the exact HTTPS origin used by the browser. Keep the bridge on loopback and do not publish the OCR endpoint to the public internet.

The OCR endpoint accepts PDF, PNG, JPEG, WebP, and TIFF documents (40 MB, 30 pages maximum), converts pages to PNG, sends them over SSH, and deletes each DGX-side temporary upload after the request. The output is OCR text with simple line classification, not authoritative structured extraction. Confirm your organization's data handling approval before using real records.

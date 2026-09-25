"""Local OCR bridge for the dashboard. Uploaded bytes travel only over SSH to DGX."""

import base64
import io
import json
import os
import re
import shlex
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlsplit

from PIL import Image, ImageSequence
import pypdfium2 as pdfium

SERVICE_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SERVICE_DIR.parent
SCORECARD_SNAPSHOT = BENCHMARK_DIR / 'final-dashboard/src/data.json'
CACHE_PATH = SERVICE_DIR / 'cached_samples.json'
DGX_HOST = os.environ.get('OCR_BENCH_DGX_HOST', '').strip()
DGX_USER = os.environ.get('OCR_BENCH_DGX_USER', '').strip()
DGX_PORT = os.environ.get('OCR_BENCH_DGX_PORT', '22').strip()
DGX_ROOT = os.environ.get('OCR_BENCH_DGX_ROOT', '').strip()
REMOTE_PYTHON = os.environ.get('OCR_BENCH_REMOTE_LAUNCH_PYTHON', '/usr/bin/python3').strip()
SSH_KEY = Path(os.environ.get('OCR_BENCH_SSH_KEY_PATH', Path.home() / '.ssh/id_ed25519'))
REMOTE = f'{DGX_USER}@{DGX_HOST}' if DGX_USER and DGX_HOST and DGX_ROOT else ''
REMOTE_COMMAND = (f'cd {shlex.quote(DGX_ROOT)} && {shlex.quote(REMOTE_PYTHON)} '
                 'scripts/testing_ocr_request.py')
SSH_BASE = (['ssh', '-p', DGX_PORT, '-i', str(SSH_KEY), '-o', 'IdentitiesOnly=yes',
             '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=30',
             '-o', 'ServerAliveCountMax=5', REMOTE] if REMOTE else [])
ALLOWED_ORIGINS = {'http://127.0.0.1:8765', 'http://localhost:8765'} | {
    origin.strip() for origin in os.environ.get('OCR_BENCH_ALLOWED_ORIGINS', '').split(',') if origin.strip()
}
MAX_INPUT = 40 * 1024 * 1024
MAX_PAGES = 30
RUN_LOCK = Lock()
WARM_PORTS = {'qwen2_5_vl_7b': 8141, 'easyocr': 8142, 'glm_ocr': 8143}
DEMO_MODELS = {'qwen2_5_vl_7b', 'easyocr', 'glm_ocr', 'indic_ocr'}


def models():
    snapshot = json.loads(SCORECARD_SNAPSHOT.read_text(encoding='utf-8'))
    rows = snapshot['queries']['scorecards']['rows']
    return [{'id': row['id'], 'name': row['model'], 'benchmarkStatus': row['status']}
            for row in rows if row['id'] in DEMO_MODELS]


def gpu_status(model=''):
    if not SSH_BASE:
        return {'connected': False, 'state': 'not_configured'}
    warm_port = WARM_PORTS.get(model)
    warm_check = (f"elif curl -fsS --max-time 2 http://127.0.0.1:{warm_port}/"
                  f"{'health' if model == 'easyocr' else 'v1/models'} >/dev/null 2>&1; then echo WARM; ") if warm_port else ''
    indic_check = ("elif curl -fsS --max-time 2 http://127.0.0.1:8141/v1/models >/dev/null 2>&1 "
                   "&& curl -fsS --max-time 2 http://127.0.0.1:8142/health >/dev/null 2>&1 "
                   "&& curl -fsS --max-time 2 http://127.0.0.1:8143/v1/models >/dev/null 2>&1; "
                   "then echo READY; ") if model == 'indic_ocr' else ''
    command = ("if tmux has-session -t mahakrushi-failed-retry 2>/dev/null || "
               "tmux has-session -t mahakrushi-final-test 2>/dev/null || "
               "tmux has-session -t mahakrushi-additional-test 2>/dev/null; then echo QUEUE; "
               + warm_check + indic_check +
               "elif nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; "
               "then echo BUSY; else echo READY; fi")
    try:
        result = subprocess.run([*SSH_BASE, command], capture_output=True, text=True, timeout=12)
        if result.returncode:
            return {'connected': False, 'state': 'offline'}
        state = result.stdout.strip().splitlines()[-1].lower()
        return {'connected': True, 'state': 'ready' if state == 'warm' else state,
                'warm': state == 'warm'}
    except Exception:
        return {'connected': False, 'state': 'offline'}


def image_png(image):
    rgb = image.convert('RGB')
    rgb.thumbnail((2400, 2400), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    rgb.save(buffer, format='PNG', optimize=True)
    data = buffer.getvalue()
    if len(data) > 15 * 1024 * 1024:
        raise ValueError('A rendered page exceeds the 15 MB OCR limit.')
    return base64.b64encode(data).decode('ascii')


def render_pages(binary, mime):
    if mime == 'application/pdf' or binary.startswith(b'%PDF-'):
        pdf = pdfium.PdfDocument(binary)
        if len(pdf) > MAX_PAGES:
            raise ValueError(f'This PDF has {len(pdf)} pages; the current limit is {MAX_PAGES}.')
        pages = []
        for number in range(len(pdf)):
            page = pdf[number]
            width, _ = page.get_size()
            bitmap = page.render(scale=min(2.25, 2200 / width))
            pages.append(image_png(bitmap.to_pil()))
            bitmap.close()
            page.close()
        return pages
    try:
        image = Image.open(io.BytesIO(binary))
        count = getattr(image, 'n_frames', 1)
        if count > MAX_PAGES:
            raise ValueError(f'This image has {count} frames; the current limit is {MAX_PAGES}.')
        return [image_png(frame) for frame in ImageSequence.Iterator(image)]
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError('Upload a PDF, PNG, JPG, WebP, or TIFF document.') from exc


def structure_page(page):
    text = page.get('text') or ''
    blocks = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = re.match(r'^(.{2,72}?)[：:=]\s*(\S.{0,240})$', line)
        if match:
            blocks.append({'type': 'key_value', 'label': match.group(1).strip(), 'value': match.group(2).strip(), 'text': line})
        elif line.startswith('|') and line.count('|') >= 2:
            blocks.append({'type': 'table_row', 'text': line})
        elif re.match(r'^(?:\d+|[०-९]+)[.)।]\s+', line):
            blocks.append({'type': 'list_item', 'text': line})
        elif line.startswith('#') or (len(line) < 95 and (line.isupper() or line.endswith(('विषय', 'सूत्र', 'शिफारशी')))):
            blocks.append({'type': 'heading', 'text': line.lstrip('# ').strip()})
        else:
            blocks.append({'type': 'text', 'text': line})
    return {**page, 'blocks': blocks, 'line_count': len(blocks)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def cors(self):
        origin = self.headers.get('Origin')
        if origin in ALLOWED_ORIGINS:
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Vary', 'Origin')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Cache-Control', 'no-store')

    def respond(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.cors()
        self.end_headers()

    def do_GET(self):
        url = urlsplit(self.path)
        if url.path == '/api/models':
            self.respond(200, {'models': models()})
        elif url.path == '/api/status':
            self.respond(200, gpu_status(parse_qs(url.query).get('model', [''])[0]))
        elif url.path == '/api/samples':
            cached = json.loads(CACHE_PATH.read_text(encoding='utf-8')) if CACHE_PATH.exists() else {}
            self.respond(200, {'modelsBySample': {name: list(results) for name, results in cached.items()}})
        elif url.path == '/api/sample':
            query = parse_qs(url.query)
            sample = query.get('sample', [''])[0]
            model = query.get('model', [''])[0]
            cached = json.loads(CACHE_PATH.read_text(encoding='utf-8')) if CACHE_PATH.exists() else {}
            result = cached.get(sample, {}).get(model)
            self.respond(200, result) if result else self.respond(404, {'error': 'No saved OCR result for this sample and model.'})
        else:
            self.respond(404, {'error': 'Not found.'})

    def do_POST(self):
        if self.path != '/api/ocr':
            self.respond(404, {'error': 'Not found.'}); return
        if not SSH_BASE:
            self.respond(503, {'error': 'Configure the DGX SSH target and project root before running OCR.'}); return
        if not RUN_LOCK.acquire(blocking=False):
            self.respond(409, {'error': 'Another OCR request is already running.'}); return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if size <= 0 or size > MAX_INPUT * 1.4:
                raise ValueError('File exceeds the 40 MB upload limit.')
            payload = json.loads(self.rfile.read(size))
            model = payload.get('model')
            if model not in {row['id'] for row in models()}:
                raise ValueError('Select an available model.')
            encoded = payload.get('file')
            if not isinstance(encoded, str) or ',' not in encoded:
                raise ValueError('Upload a valid document.')
            metadata, body = encoded.split(',', 1)
            mime = metadata.removeprefix('data:').split(';')[0]
            binary = base64.b64decode(body, validate=True)
            if len(binary) > MAX_INPUT:
                raise ValueError('File exceeds the 40 MB upload limit.')
            pages = render_pages(binary, mime)
            remote_input = json.dumps({'model': model, 'pages': pages})
            result = subprocess.run([*SSH_BASE, REMOTE_COMMAND], input=remote_input, capture_output=True,
                text=True, encoding='utf-8', timeout=max(1800, len(pages) * 1050))
            if result.returncode:
                raise RuntimeError('DGX connection or OCR runner failed: ' + result.stderr[-350:])
            if not result.stdout.strip():
                raise RuntimeError('DGX returned no OCR result.')
            remote = json.loads(result.stdout.strip().splitlines()[-1])
            if remote.get('error'):
                self.respond(409 if remote.get('busy') else 502, remote); return
            remote['pages'] = [structure_page(page) for page in remote['pages']]
            remote['page_count'] = len(remote['pages'])
            remote['successful_pages'] = sum(bool(page['success']) for page in remote['pages'])
            self.respond(200, remote)
        except (ValueError, json.JSONDecodeError) as exc:
            self.respond(400, {'error': str(exc)})
        except subprocess.TimeoutExpired:
            self.respond(504, {'error': 'DGX OCR job timed out. Try a shorter document.'})
        except Exception as exc:
            self.respond(502, {'error': str(exc)})
        finally:
            RUN_LOCK.release()


if __name__ == '__main__':
    server = ThreadingHTTPServer(('127.0.0.1', 8766), Handler)
    print('OCR bridge listening at http://127.0.0.1:8766', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()

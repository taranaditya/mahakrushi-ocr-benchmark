"""Persistent, loopback-only EasyOCR reader for dashboard test pages."""

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import easyocr

ROOT = Path(os.environ.get('OCR_BENCH_DGX_ROOT', Path(__file__).resolve().parents[1])).expanduser().resolve()
TEST_ROOT = (ROOT / 'data/testing').resolve()
reader = easyocr.Reader(['en', 'hi'], gpu='cuda')


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != '/health':
            self.send_error(404)
            return
        self.respond(200, {'ready': True, 'model': 'easyocr'})

    def do_POST(self):
        if self.path != '/ocr':
            self.send_error(404)
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 4096:
                raise ValueError('Invalid request size.')
            payload = json.loads(self.rfile.read(size))
            image = Path(payload['image']).resolve(strict=True)
            if not image.is_relative_to(TEST_ROOT) or image.suffix.lower() != '.png':
                raise ValueError('Image must be a dashboard test page.')
            text = '\n'.join(reader.readtext(str(image), detail=0, paragraph=False))
            self.respond(200, {'text': text})
        except Exception as exc:
            self.respond(400, {'error': str(exc)})

    def respond(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


HTTPServer(('127.0.0.1', 8142), Handler).serve_forever()

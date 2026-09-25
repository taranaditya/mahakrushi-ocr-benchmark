"""Run one dashboard upload through the installed IndicOCR model on CUDA."""

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; IndicOCR live testing requires GPU.')
    torch.cuda.set_device(0)
    model_dir = ROOT / 'models/indic-ocr'
    sys.path.insert(0, str(model_dir))
    from indic_ocr import IndicOCR

    ocr = IndicOCR.from_pretrained(str(model_dir), device='cuda:0', table_format='markdown')
    records = [json.loads(line) for line in args.manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as stream:
        for record in records:
            started = time.perf_counter()
            try:
                page = ocr.parse(str(ROOT / record['image']))
                text = str(page.get('markdown') or '')
                row = {'model_id': 'indic_ocr', 'document_id': record['id'], 'success': bool(text.strip()),
                       'latency_seconds': time.perf_counter() - started, 'raw_text': text}
                if not row['success']:
                    row['error_message'] = 'IndicOCR returned empty Markdown.'
            except Exception as exc:
                row = {'model_id': 'indic_ocr', 'document_id': record['id'], 'success': False,
                       'latency_seconds': time.perf_counter() - started, 'raw_text': '',
                       'error_message': repr(exc)}
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()


if __name__ == '__main__':
    main()

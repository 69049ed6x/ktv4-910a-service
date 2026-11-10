"""Paths shared by the service, persistent workers and vLLM executor."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / 'app'
RUN = ROOT / 'run'
CACHE = ROOT / 'cache'
CHECKPOINT = Path(os.environ.get('KTV4_MODEL_PATH', '/mnt/nvme1n1/bf16/w8a8-mtp'))
PYTHON = ROOT / 'env210/bin/python'
MODEL_NAME = 'deepseek-v4-flash'
PORT = 8000
WORKERS = 4


def atomic_json(path, value):
    import json
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)

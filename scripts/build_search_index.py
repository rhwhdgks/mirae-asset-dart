#!/usr/bin/env python3
"""Stage2 FTS5 검색 인덱스 빌드: out/canonical → out/serving/<index_build_id>/ (app/serving/index.py).

실측 약 35분, 1.5GB. canonical build_id·chunks hash·tokenizer 설정이 index_build_id에 결속되므로
정본이 바뀌면 다시 빌드해야 한다 (SearchIndex가 불일치를 fail-closed로 거부한다).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.tools.canonical_env import read_model, CANONICAL_DIR  # noqa: E402
from app.serving.index import build_index  # noqa: E402

run = json.loads((CANONICAL_DIR / "run.json").read_text(encoding="utf-8"))
out = build_index(read_model(), run)
print("→", out)

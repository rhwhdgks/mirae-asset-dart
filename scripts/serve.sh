#!/usr/bin/env bash
# 평가 API 서버 실행. 포트 80은 root/authbind 필요 — 개발은 8000.
cd "$(dirname "$0")/.."
exec ./.venv/bin/uvicorn server.app:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1 --timeout-keep-alive 65

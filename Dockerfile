# 평가 서버(GET /answer, POST /answer/resume, /v1) 실행용 이미지.
# 런타임 의존성만 설치한다 — 개발 도구(ruff, requirements-dev.txt)는 포함하지 않는다.
#
# 정본·검색 인덱스·Raw 코퍼스는 공개본에 없으며 이미지에 굽지 않는다:
#   -v <host>/out/canonical:/app/out/canonical       (make build-data 산출물)
#   -v <host>/out/serving:/app/out/serving           (make search-index 산출물)
#   -v <host>/data/corpus:/app/data/corpus           (평가 서버 자체는 읽지 않지만 재빌드용)
# API 키는 이미지에 넣지 않고 --env-file .env 로 주입한다(CLOVASTUDIO_API_KEY, HCX_API_KEY 등 — .env.example 참조).
FROM python:3.14-slim

WORKDIR /app

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 런타임 의존성만 (requirements-dev.txt 는 검수·개발 전용, 이미지에 넣지 않는다)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# /readyz 는 정본·인덱스·Stage1 준비 전 503을 낸다(server/app.py) — 200 이어야 트래픽을 받을 준비가 된 것.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

"""프로젝트 루트 .env 자동 로드. 모든 진입점(서버·pipeline·harness·테스트)이 import한다.
환경변수가 이미 있으면 .env가 덮어쓰지 않는다 (override=False)."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
except ImportError:  # dotenv 미설치여도 동작 (환경변수 직접 지정 시)
    pass

# Stage1(HCX-007, agent/providers/hcx007.py)은 CLOVASTUDIO_API_KEY 를 읽는다.
# 비어 있으면 같은 CLOVA Studio 키의 다른 이름(HCX_PLANNER_API_KEY → HCX_API_KEY)을 순서대로 복사한다.
if not os.getenv("CLOVASTUDIO_API_KEY"):
    for _alias in ("HCX_PLANNER_API_KEY", "HCX_API_KEY"):
        if os.getenv(_alias):
            os.environ["CLOVASTUDIO_API_KEY"] = os.environ[_alias]
            break

# Stage4(HCX-005)는 HCX_API_KEY를 읽는다. 별도 키가 없으면 Stage1과 같은
# CLOVA Studio 키를 재사용하되, 명시적으로 설정한 Composer 키는 덮어쓰지 않는다.
if not os.getenv("HCX_API_KEY") and os.getenv("CLOVASTUDIO_API_KEY"):
    os.environ["HCX_API_KEY"] = os.environ["CLOVASTUDIO_API_KEY"]

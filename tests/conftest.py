"""pytest 수집 규칙.

테스트는 두 부류다.
- **함수형**(`def test_*`) — pytest 가 돈다: `make pytest`(오프라인) · `make pytest-canonical`(정본 필요).
- **모듈형·스크립트형**(import 시 assert 하거나 `__main__` 이 검증) — `tests/run_all.py`(`make test`)가 돈다.
  이런 파일은 여기서 수집하지 않는다. 종전에는 `run_all` 이 함수형 파일도 `python -m` 으로 import 만 해서
  테스트 함수가 하나도 실행되지 않은 채 PASS 로 세던 구멍이 있었다.

정본이 필요한 파일은 `canonical` 마커를 자동으로 단다. 대부분 import 시 정본과 검색 인덱스를 열므로,
published 정본과 그 build 에 결속된 인덱스 manifest 가 둘 다 있을 때만 수집한다(재빌드 중·정본만 받은
환경에서 `make pytest` 가 수집 단계에서 깨지지 않도록). `-m "not canonical"` 로 돌 때는 정본 파일을 아예
수집하지 않는다 — 어차피 deselect 될 파일 때문에 정본을 여는 낭비를 막는다. `test_server_api_v1_canonical.py` 는 `canonical_server` —
ServerRuntime 실행 스레드에서 pyarrow 를 쓰므로 다른 정본 테스트와 같은 프로세스에 두면 스레드 교차
segfault 가 난다. Makefile 이 따로 돈다.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _canonical_ready() -> bool:
    """published 정본 + 그 build_id 에 결속된 검색 인덱스 manifest. 가벼운 파일 검사만 하고 정본은 열지 않는다."""
    try:
        run = json.loads((ROOT / "out" / "canonical" / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not run.get("published"):
        return False
    for manifest in (ROOT / "out" / "serving").glob("*/manifest.json"):
        try:
            if json.loads(manifest.read_text(encoding="utf-8")).get("canonical_build_id") == run.get("build_id"):
                return True
        except (OSError, ValueError):
            continue
    return False


CANONICAL_READY = _canonical_ready()
_FUNCTION_TEST = re.compile(r"^(?:def test_|class Test)", re.M)
_EXCLUDES_CANONICAL = re.compile(r"\bnot\s+canonical\b")   # `not canonical_server` 는 해당 없음

CANONICAL_FILES = {
    "test_balance_item_year_to_fye_canonical.py",
    "test_composer.py",
    "test_composer_typed_change.py",
    "test_concept_ratio_canonical.py",
    "test_correction_evidence_batch.py",
    "test_development_event_safety_full_path.py",
    "test_development_event_safety_no_hcx.py",
    "test_event_backend.py",
    "test_fact_cache.py",
    "test_financial_backend.py",
    "test_holding_canonical_e2e.py",
    "test_investment_row_operator_sidecar.py",
    "test_issue28_business_axis_projection.py",
    "test_lifecycle_termination_observation.py",
    "test_narrative_backend.py",
    "test_narrative_second_pass.py",
    "test_pct_period_length_canonical_e2e.py",
    "test_policy_fallback_sidecar.py",
    "test_relative_financial_period_canonical.py",
    "test_section_point_lookup.py",
    "test_stage1_v1_clarification_real_roundtrip.py",
    "test_stage1_v1_native_e2e.py",
    "test_stage1_v1_narrative_investment_development.py",
    "test_stage1_v1_question_grounded_financial.py",
}
CANONICAL_SERVER_FILES = {"test_server_api_v1_canonical.py"}


def pytest_ignore_collect(collection_path: Path, config: pytest.Config):
    path = Path(collection_path)
    if path.suffix != ".py" or not path.name.startswith("test_"):
        return None
    if path.name in CANONICAL_FILES | CANONICAL_SERVER_FILES:
        if not CANONICAL_READY:
            return True
        if _EXCLUDES_CANONICAL.search(config.getoption("-m", default="") or ""):
            return True          # 오프라인 실행(`make pytest`) — 정본 파일은 열지도 않는다
    text = path.read_text(encoding="utf-8", errors="replace")
    if _FUNCTION_TEST.search(text) is None:
        return True          # 모듈형·스크립트형 → tests/run_all.py
    return None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        name = Path(str(item.fspath)).name
        if name in CANONICAL_SERVER_FILES:
            item.add_marker(pytest.mark.canonical_server)
            item.add_marker(pytest.mark.canonical)
        elif name in CANONICAL_FILES:
            item.add_marker(pytest.mark.canonical)

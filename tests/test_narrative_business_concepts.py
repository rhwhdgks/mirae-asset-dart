"""Source-grounded extraction of operating-area enumerations."""

import pytest

from app.composer.narrative_matrix import (
    MatrixCellEvidence,
    _business_concepts,
    _term_contrast,
)


@pytest.mark.parametrize(("text", "expected"), [
    (
        "당사는 음악(Music), 플랫폼(Platform), 테크기반 미래성장"
        "(Tech-driven Future Growth) 세 가지 주요 영역으로 사업을 운영합니다.",
        ("음악(Music)", "플랫폼(Platform)", "테크기반 미래성장"),
    ),
    (
        "회사는 로봇, 센서 및 구동장치 세 가지 주요 분야를 개발합니다.",
        ("로봇", "센서", "구동장치"),
    ),
])
def test_counted_operating_area_list_keeps_members_not_the_count_label(
        text, expected):
    concepts = _business_concepts(text)
    compact = " ".join(concepts)
    for member in expected:
        assert member in compact
    assert not any("가지" in term or "개 주요 분야" in term
                   or "개 주요 영역" in term for term in concepts)


def test_counted_area_fragment_never_becomes_a_company_comparison():
    first = (
        "당사는 음악(Music), 플랫폼(Platform), 테크기반 미래성장"
        "(Tech-driven Future Growth) 세 가지 주요 영역으로 사업을 운영합니다.")
    second = "당사는 음반사업과 매니지먼트사업을 영위합니다."
    cells = (
        MatrixCellEvidence("a", "가상미디어 · 2025년", "주요 사업", first,
                           ("20260310000001",)),
        MatrixCellEvidence("b", "가상기획 · 2025년", "주요 사업", second,
                           ("20260310000002",)),
    )

    conclusion = _term_contrast(cells)
    assert "가지" not in conclusion
    assert "영역으로 사업" not in conclusion


@pytest.mark.parametrize("text", [
    "당사는 두 가지 주요 영역으로 사업을 운영합니다.",
    "당사는 성장 기회를 찾아 세 가지 주요 영역으로 사업을 확대합니다.",
    "당사는 안전관리 및 비용절감 두 가지 주요 분야를 강화합니다.",
])
def test_count_label_without_literal_operating_list_adds_no_business_concepts(text):
    assert _business_concepts(text) == ()


def test_counted_operating_list_handles_numeric_count_and_korean_conjunction():
    concepts = _business_concepts(
        "당사는 로봇과 센서 2개 주요 분야를 개발합니다.")

    assert "로봇" in concepts
    assert "센서" in concepts
    assert not any("주요 분야" in concept for concept in concepts)


def test_generic_operating_verb_does_not_become_a_product_name():
    assert _business_concepts("당사는 판매를 수행하고 있습니다.") == ()


def test_product_axis_does_not_substitute_segment_names_for_products():
    text = """| 사업부문 | 품목 | 용도 |
|---|---|---|
| 금속부문 | 정밀봉강 | 기계용 |
| 에너지부문 | 양극재 | 배터리용 |
"""
    terms = _business_concepts(text, products_only=True)

    assert "정밀봉강" in terms
    assert "양극재" in terms
    assert "금속부문" not in terms
    assert "에너지부문" not in terms

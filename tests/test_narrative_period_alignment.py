"""Offline regressions for comparing only matching filing periods."""

from datetime import date
from types import SimpleNamespace

from app.composer.narrative_matrix import (
    MatrixCellEvidence,
    NarrativeMatrix,
    _aligned_period_reports,
    _metric_contrast,
    _structured_cross_cell_deltas,
)
from app.tools.narrative import _matches_matrix_period


def _period(year: int, month: int, day: int):
    return SimpleNamespace(start=date(year, 1, 1), end=date(year, month, day))


def _reader(**documents):
    return SimpleNamespace(_load_docs=lambda: None, _documents=documents)


def _document(year: int, month: int, form: str):
    return SimpleNamespace(doc_group="periodic", base_year=year,
                           base_month=month, form=form)


def _cell(year: int, report: str, value: int):
    excerpt = f"{year}년 매출은 {value}억원입니다."
    return MatrixCellEvidence(
        str(year), f"가상회사 · {year}년", "매출구성", excerpt,
        (f"{year}0000000000",), report_names=(report,))


def test_matrix_retrieval_rejects_next_year_half_report_for_prior_annual_cell():
    reader = _reader(
        wrong=_document(2025, 6, "반기보고서"),
        prior=_document(2024, 12, "사업보고서"),
        latest=_document(2025, 12, "사업보고서"),
    )
    assert not _matches_matrix_period(reader, "wrong", _period(2024, 12, 31))
    assert _matches_matrix_period(reader, "prior", _period(2024, 12, 31))
    assert _matches_matrix_period(reader, "latest", _period(2025, 12, 31))
    assert _matches_matrix_period(reader, "wrong", _period(2025, 6, 30))


def test_mismatched_cited_report_suppresses_delta_and_explains_why():
    cells = (
        _cell(2024, "가상회사 반기보고서 (2025.06)", 100),
        _cell(2025, "가상회사 사업보고서 (2025.12)", 130),
    )
    assert not _aligned_period_reports(cells)
    assert _metric_contrast(cells, topic="매출구성") == ""
    assert not _structured_cross_cell_deltas(cells, topic="매출구성")
    conclusion = NarrativeMatrix(
        cells, (), "기간별 변화 비교").grounded_conclusion()
    assert "서로 다른 기간의 수치를 빼지 않았습니다" in conclusion
    assert "30억원 증가" not in conclusion


def test_same_annual_report_basis_keeps_valid_delta():
    cells = (
        _cell(2024, "가상회사 사업보고서 (2024.12)", 100),
        _cell(2025, "가상회사 사업보고서 (2025.12)", 130),
    )
    assert _aligned_period_reports(cells)
    assert "30억원 증가" in _metric_contrast(cells, topic="매출구성")

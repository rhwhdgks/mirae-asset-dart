"""Keep a correction's blank advisory marker distinct from disclosed text."""

from app.composer.template import _correction_after_bullets


def test_blank_after_advisory_is_explained_not_rendered_as_a_value():
    rendered = _correction_after_bullets("- 종전 투자판단 사항 → -")

    assert rendered == "정정 후 기타 투자판단 사항은 원문에 내용 없이 「-」로 기재됐습니다"
    assert "정정 후 확인되는 내용: -" not in rendered


def test_stated_after_advisory_remains_visible():
    rendered = _correction_after_bullets(
        "- 종전 계약 사항 → - 계약이 해지됐습니다")

    assert rendered == "정정 후 확인되는 내용: 계약이 해지됐습니다"

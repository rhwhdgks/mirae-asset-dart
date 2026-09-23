"""``mask_holding_subject_name`` — 지분공시 보고 주체 성명 부분 가림 (이슈 #51).

코퍼스 전수 조사로 순수 한글 2~3글자 값은 전부
자연인 이름이고, 3글자 651개 중 기관 꼬리(공단·재단·은행…)로 끝나는 값은
0개였다. 그래서 값의 글자 수·구성만으로 사람 이름을 가른다: 성과 끝 글자를
남기고 가운데만 가린다. 각주·정정 표기가 붙은 실측 샘플(``이재상*``·
``김주영B``·``정 상 수``·``주1)``)도 걷어낸 뒤 같은 규칙을 적용한다.
"""
from __future__ import annotations

from app.textkit import mask_holding_subject_name


def test_masks_three_character_hangul_name():
    assert mask_holding_subject_name("최윤범") == "최○범"


def test_masks_two_character_hangul_name():
    assert mask_holding_subject_name("김선") == "김○"


def test_strips_trailing_footnote_asterisk_before_masking():
    assert mask_holding_subject_name("이재상*") == "이○상"


def test_strips_trailing_single_latin_disambiguator_before_masking():
    assert mask_holding_subject_name("김주영B") == "김○영"


def test_collapses_internal_padding_spaces_before_masking():
    assert mask_holding_subject_name("정 상 수") == "정○수"


def test_bare_footnote_reference_is_left_unchanged():
    # "주1)" 자체는 이름이 아니라 각주 참조라 걷어내면 빈 문자열이 되므로
    # 보수적으로 원문을 그대로 둔다 (이름이 아니므로 가릴 대상도 아니다).
    assert mask_holding_subject_name("주1)") == "주1)"


def test_institutional_filer_is_left_unchanged():
    assert mask_holding_subject_name("국민연금공단") == "국민연금공단"


def test_short_form_corporate_suffix_is_left_unchanged():
    assert mask_holding_subject_name("삼성물산") == "삼성물산"


def test_mixed_latin_hangul_corporate_name_is_left_unchanged():
    assert mask_holding_subject_name("LG전자") == "LG전자"


def test_four_character_hangul_name_is_left_unchanged_conservatively():
    # 4글자 이상은 보수적으로 원문을 유지한다(코퍼스 전수 조사 대상 밖).
    assert mask_holding_subject_name("남궁민수") == "남궁민수"


def test_corporate_names_hint_exempts_a_non_suffixed_company():
    # "포스코" 처럼 법인 접미사가 없는 3글자 상호도, 문서의 발행회사명
    # 후보로 전달되면 사람 이름으로 오분류하지 않는다.
    assert mask_holding_subject_name(
        "포스코", corporate_names=("포스코",)) == "포스코"


def test_empty_value_is_returned_unchanged():
    assert mask_holding_subject_name("") == ""


def test_none_like_falsy_value_is_returned_unchanged():
    assert mask_holding_subject_name(None) is None

"""User-facing rendering for typed Stage1 clarification slots.

The renderer deliberately owns presentation only.  Slot response kinds and
option values remain authoritative for validation and resume handling.
"""

from __future__ import annotations

from collections.abc import Sequence
import re

from .stage1_v1_resolver import ClarificationSlot


_MULTI_SLOT_INTRO = (
    "아래 항목을 알려주시면 더 정확히 답변드릴 수 있습니다."
)
_SINGLE_SLOT_INTRO = "정확히 확인하기 위해 한 가지만 여쭤볼게요."
_FOLLOWUP_SINGLE_SLOT_INTRO = "추가로 한 가지만 여쭤볼게요."
_FOLLOWUP_MULTI_SLOT_INTRO = "추가로 몇 가지만 확인할게요."
_INTERNAL_NAME = re.compile(
    r"(?<![A-Za-z0-9_])(?:account_path|account_norm|entity_refs|item_id)"
    r"(?![A-Za-z0-9_])"
)


def _public_prompt(prompt: str) -> str:
    """Prevent wire-level field names from reaching the user."""
    return _INTERNAL_NAME.sub("해당 항목", prompt)


_PLAIN_OPTION_HELP = {
    "매출액": "상품·서비스를 팔아 벌어들인 금액",
    "매출총이익": "매출액에서 매출원가를 뺀 금액",
    "영업이익": "회사의 주된 영업활동에서 남은 이익",
    "당기순이익": "세금 등을 반영한 최종 이익",
    "이자수익": "이자를 받아 생긴 수익",
    "이자비용": "차입금 등에 지급한 이자 비용",
    "이자의수취": "현금흐름표에 적힌 실제 이자 수취액",
    "이자의지급": "현금흐름표에 적힌 실제 이자 지급액",
    "법인세비용": "해당 기간 손익에 반영한 법인세",
    "당기법인세자산": "미리 냈거나 돌려받을 법인세",
    "당기법인세부채": "앞으로 납부할 당기 법인세",
    "부채총계": "회사가 갚아야 할 부채 전체",
    "유동부채": "보통 1년 안에 갚을 부채",
    "비유동부채": "보통 1년 뒤에 갚을 부채",
    "자본총계": "자산에서 부채를 뺀 주주 몫",
    "자본금": "회사가 발행한 주식의 액면가 기준 자본",
    "현금및현금성자산": "현금과 바로 현금화할 수 있는 자산",
    "현금및현금성자산의순증감": "기간 중 현금성 자산이 늘거나 줄어든 금액",
    "영업활동현금흐름": "주된 영업활동에서 들어오고 나간 순현금",
    "매출채권": "외상매출로 아직 받지 못한 돈",
    "매출채권및기타채권": "매출채권과 그 밖의 받을 돈을 합친 항목",
    "이익잉여금": "누적 이익 중 회사에 남아 있는 금액",
    "자본잉여금": "주식 발행 등으로 자본금 외에 쌓인 금액",
    "투자부동산": "임대수익이나 시세차익을 목적으로 보유한 부동산",
    "유형자산": "건물·기계처럼 형태가 있는 장기 사용 자산",
    "유형자산의취득": "유형자산을 사는 데 실제로 지출한 현금",
    "유형자산의처분": "유형자산을 팔아 실제로 받은 현금",
    "무형자산의취득": "특허권·소프트웨어 등 무형자산 취득 지출",
    "투자활동현금흐름": "장기자산 투자·회수에서 생긴 순현금흐름",
    "총포괄손익": "당기순이익과 기타포괄손익을 합친 손익",
    "기타포괄손익": "당기순이익 밖에서 자본에 반영되는 손익",
}


def _public_option_label(label: str) -> str:
    """Add plain-language help without changing the option value contract."""

    value = label.strip()
    help_text = _PLAIN_OPTION_HELP.get(value)
    return f"{value} — {help_text}" if help_text else value


def render_clarification_message(
        slots: Sequence[ClarificationSlot],
        *,
        followup: bool = False,
        ) -> str:
    """Render one or more typed slots without changing their semantics."""

    if not slots:
        raise ValueError("표시할 clarification slot이 없습니다")

    if len(slots) > 1:
        prompts = [_public_prompt(slot.prompt.strip()) for slot in slots]
        return "\n\n".join((
            _FOLLOWUP_MULTI_SLOT_INTRO if followup else _MULTI_SLOT_INTRO,
            "\n".join(
                f"{index}. {prompt}"
                for index, prompt in enumerate(prompts, start=1)
            ),
        ))

    slot = slots[0]
    prompt = _public_prompt(slot.prompt.strip())
    intro = (
        _FOLLOWUP_SINGLE_SLOT_INTRO if followup else _SINGLE_SLOT_INTRO
    )
    if slot.response_kind == "provide_value":
        return "\n\n".join((intro, prompt))

    return "\n\n".join((
        intro,
        prompt,
        "\n".join(
            f"{index}. {_public_option_label(option.label)}"
            for index, option in enumerate(slot.options, start=1)
        ),
    ))


__all__ = ["render_clarification_message"]

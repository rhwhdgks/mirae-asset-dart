"""원 질문에 결속된 Stage1 정책 판정 — 주가 인과 거절과 압박 표현 감지.

`stage1_v1_policy_backend` 가 쓴다. 질문 ID나 특정 접수번호에 기대지 않는 문장 규칙만 둔다.
"""

from __future__ import annotations

import re
import unicodedata

#: 정책 판정에 쓰는 typed reason 코드. 값은 `agent.query_plan` 의 handoff reason 과 같은 어휘다.
IntentReasonCode = str


def _text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


_PRESSURE_PATTERNS = (
    re.compile(r"없더라도"),
    re.compile(r"대충(?:이어도|이라도)"),
    re.compile(r"대충\s*(?:찍|만들|써)"),
    re.compile(r"무조건\s*(?:맞|참|그렇)"),
    re.compile(r"맞다고\s*(?:결론|답)"),
)


def causal_policy_reasons(
        question: str,
        *,
        has_unverified_causal_claim: bool = False,
        ) -> tuple[IntentReasonCode, ...]:
    """Return a generic causal-scope refusal, independent of fixture IDs.

    A causal premise alone is enough when the semantic boundary explicitly
    marked it as unverifiable.  Otherwise the raw question must ask to prove
    or force a causal conclusion about share-price movement.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("causal policy에는 원 질문이 필요합니다")
    raw = _text(question).strip()
    causal = has_unverified_causal_claim or (
        "주가" in raw
        and bool(re.search(r"원인|인과|때문|맞다고\s*(?:결론|답)|증명", raw))
    )
    if not causal:
        return ()
    reasons: list[IntentReasonCode] = ["causal_inference_beyond_scope"]
    if any(pattern.search(raw) for pattern in _PRESSURE_PATTERNS):
        reasons.append("pressure_resisted")
    return tuple(reasons)


def pressure_policy_applies(question: str) -> bool:
    """Whether the user is pressuring the planner to bypass evidence rules."""
    if not isinstance(question, str):
        raise TypeError("pressure policy에는 문자열 질문이 필요합니다")
    return any(pattern.search(_text(question)) for pattern in _PRESSURE_PATTERNS)


__all__ = ["IntentReasonCode", "causal_policy_reasons", "pressure_policy_applies"]

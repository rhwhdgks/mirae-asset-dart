"""Local explanations of disclosure display rules, never a factual lookup.

The synthetic projection copies explicitly public inputs only. It neither
calculates a residual nor cites invented disclosure evidence.
"""
from __future__ import annotations

import re


def _pure_synthetic_state_rule(text: str) -> bool:
    """Accept a closed display-state example, not an attached factual task."""
    compact = re.sub(r"[\s`'\"「」“”‘’]", "", text)
    prefix = re.match(r"(?:한|가상의?|예시|어떤)?(?:공시)?표에서", compact)
    if prefix is None or "공시유보" not in compact:
        return False
    values, _, request = compact[prefix.end():].rpartition("공시유보")
    # Cell labels and literal states only; no actual issuer or fact can be
    # smuggled into the example before the final state.
    if re.fullmatch(r"(?:[A-Z](?:는|은|가|이)?|0|-|공란|해당없음|[,;])*", values) is None:
        return False
    return re.fullmatch(
        r"(?:로표시(?:됐다|되었다|되어있다)|다섯상태를|다섯표기를|같은값으로|처리하지말고"
        r"|각각|어떤|의미로|의미를|뜻을|차이를|답할지|설명해줘|설명해주세요|[.!?])*",
        request) is not None


def explain_disclosure_rule(question: str) -> str | None:
    text = question.strip()
    if len(text) > 1500 or re.search(
            r"조회|찾아|검색|접수번호|사업보고서|분기보고서|반기보고서"
            r"|(?:매출|영업이익|영업수익|자산|부채|임원|주식수|계약금액).{0,25}(?:알려|얼마|계산)", text):
        return None
    if (_pure_synthetic_state_rule(text) and re.search(r"설명|의미|구분", text)
            and all(term in text for term in ("공란", "공시유보"))
            and re.search(r"해당\s*없음", text)
            and re.search(r"(?<!\d)0(?!\d)", text)):
        return (
            "표기 규칙에 대한 설명입니다. 특정 회사의 공시를 조회한 결과는 아닙니다.\n\n"
            "- 0: 숫자 0이 적혀 있습니다. 값이 비어 있는 것과 다릅니다.\n"
            "- -: 숫자가 기재되지 않은 표시입니다. 이것만 보고 0이나 공시유보로 단정하지 않습니다.\n"
            "- 공란: 칸이 비어 있어 값을 확인할 수 없습니다.\n"
            "- 해당 없음: 해당 항목이 적용되지 않는다고 적혀 있습니다. 숫자 0으로 바꾸지 않습니다.\n"
            "- 공시유보: 공개를 미룬 값입니다. 0으로 처리하거나 다른 숫자로 역산하지 않습니다.\n\n"
            "표의 주석에 별도 설명이 있으면 함께 확인합니다."
        )
    # Explicit hypothetical table + request to preserve privacy. Do not turn a
    # request to discover an actual company's hidden amount into an explanation.
    if not (re.search(r"표가\s*있다|가상|예시|가정", text)
            and "비공개" in text and "공개하되" in text
            and re.search(r"역산되지\s*않|역산할\s*수\s*없", text)):
        return None
    allowed_match = re.search(r"((?:[A-Z]사[와과은는만,·\s]*)+)공개하되", text)
    if allowed_match is None:
        return None
    allowed = set(re.findall(r"[A-Z]사", allowed_match.group(1)))
    values = re.findall(
        r"(?<![\w가-힣])([A-Z]사)\s+([0-9][0-9,]*(?:\.[0-9]+)?\s*(?:조|억|만)?원)",
        text,
    )
    public = [(name, amount) for name, amount in values if name in allowed]
    if {name for name, _ in public} != allowed:
        return None
    if any(re.search(rf"{name}(?:(?![A-Z]사).)*비공개", text, re.S)
           for name in allowed):
        return None
    hidden = re.findall(r"([A-Z]사)\s*(?:의\s*)?금액은?\s*비공개", text)
    if len(hidden) != 1 or not public or len(public) > 10:
        return None
    if hidden[0] in {name for name, _ in public} or len({n for n, _ in public}) != len(public):
        return None
    return (
        "질문에 주어진 예시를 공개 가능한 항목만 남겨 정리했습니다. 실제 공시 조회 결과는 아닙니다.\n\n"
        + "\n".join(f"- {name}: {amount}" for name, amount in public)
        + f"\n- {hidden[0]} 금액: 비공개\n\n"
        "비공개 금액을 뺄셈으로 알아낼 수 없도록 총합과 차액은 표시하지 않습니다."
    )

"""Deterministic Korean money display-unit directive recognition (issue #61).

HCX's wire schema (``hcx-semantic-intent-wire/1.1``) has no dedicated slot for
a *display* instruction such as ``정확한 원 단위와 조원 단위로 함께 보여줘``.
Left unrecognized, a provider places that instruction inside
``output.field_surfaces`` or ``target.qualifier_surfaces`` — the only free-text
slots it has — where it pollutes concept grounding: the answer target turns
into an unresolvable multi-field ``record`` instead of one retrievable scalar
metric (see ``out/logs/stage1_wire_failures.jsonl`` RPC-004/RPC-005, both
``DeterministicPlanCompilerError`` before this module existed).

This module is the single place that recognizes that phrasing, deliberately
without adding a schema field.  It is used twice, from the two ends of the
pipeline, on the same raw question text:

1. ``agent.semantic_intent_v1_boundary`` calls
   :func:`is_display_unit_only_surface` to *detect and drop* a display-unit
   phrase from ``field_surfaces``/``qualifier_surfaces`` so grounding sees a
   clean scalar target.  Nothing about the requested display crosses the
   intent schema boundary at that point — it is simply discarded there.
2. ``app.composer.template`` calls :func:`parse_display_units_directive` again
   on the original question text at render time to recover which units (and
   rounding) to show alongside the canonical won value.  Because both call
   sites read the same source text independently, no schema change or extra
   pipeline slot is needed to carry the instruction across.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re

__all__ = [
    "DisplayUnitsDirective",
    "is_display_rounding_only_surface",
    "is_display_unit_only_surface",
    "parse_display_units_directive",
    "render_display_units",
]

#: Canonical unit spelling order used everywhere below (longest/most specific
#: word first so alternation cannot short-circuit on a shared trailing ``원``).
_UNIT_WORD = r"(?:천원|백만원|억\s*원|조\s*원|원)"

_UNIT_SCALE = {
    "원": Decimal(1),
    "천원": Decimal(10) ** 3,
    "백만원": Decimal(10) ** 6,
    "억원": Decimal(10) ** 8,
    "조원": Decimal(10) ** 12,
}

#: A field/qualifier surface that names *only* a display unit, optionally with
#: a leading ``정확한`` (exact) qualifier and/or a trailing ``단위`` (unit)
#: noun.  A genuine metric or answer-field label is never spelled as a bare
#: money-unit word, so a full-string match here cannot misfire on a
#: legitimate surface.
_DISPLAY_UNIT_ONLY_SURFACE = re.compile(
    rf"^(?:정확한\s*)?{_UNIT_WORD}(?:\s*단위)?"
    rf"(?:\s*(?:기준\s*)?(?:으로|로)\s*"
    rf"(?:반올림한\s*값|얼마(?:인가|야)?))?\??$")

_UNIT_TERM_PATTERN = rf"(?:정확한\s*)?{_UNIT_WORD}(?:\s*단위)?"
_SEPARATOR = r"(?:\s*(?:,|、|및|그리고|와|과)\s*)"
#: SG-018 — 「조원 단위로 알려줘」도 표시 지시다. ``알려``는 그 하나만 따로
#: 둔다: 「…으로 알려져 있다」처럼 서술형으로도 흔히 쓰이는 어간이라, 다른
#: 동사들과 달리 어미(``줘``/``주세요``/…)가 실제로 있을 때만 명령으로
#: 본다 — 어미를 선택으로 두면 「300억원으로 알려진」 같은 서술문까지
#: 표시 지시로 오탐한다.
_DISPLAY_VERB = (
    r"(?:(?:보여|바꿔|바꾸어|환산해|표시해|나타내)\s*(?:줘|줄래|주세요|주십시오|다오)?"
    r"|알려\s*(?:줘|줄래|주세요|주십시오|다오)"
    r"|반올림\s*해\s*(?:줘|줄래|주세요|주십시오|다오))")

#: 이슈 #170(M34-b) — 「조원 단위로, 소수점 둘째 자리까지 반올림해서
#: 보여줘」처럼 반올림 절이 단위절과 동사 사이에 끼어들 수 있다.  값 자체는
#: 여기서 읽지 않는다 — 아래 ``_ROUNDING_CLAUSE`` 가 원문 전체에서 독립적으로
#: 다시 읽으므로, 여기서는 그 절이 최종 동사를 막지 않도록 통과만 시킨다.
_INLINE_ROUNDING_ASIDE = (
    r"(?:(?:,\s*)?(?:소수점\s*(?:첫째|둘째|셋째|넷째|한|두|세|네|[1-9])\s*(?:번째)?\s*자리"
    r"(?:까지)?\s*)?반올림(?:해서?|하여)?\s*)?")

#: 이슈 #170(M35-a/c) — 표시 동사 없이 「... 단위로 얼마인가?」로 끝나는
#: 의문형도 표시 지시다.  「얼마」계열만 잡는다 — 「몇 배」「몇 퍼센트」는
#: concept_ratio presentation(#38/#170 1번)이 따로 다루는 별개 의도라
#: 여기서는 넓히지 않는다.
_DISPLAY_INTERROGATIVE_TAIL = (
    r"얼마(?:인가|입니까|인지|일까|야|죠|예요|에요)?\s*\??")

#: ``조원 단위로 반올림한 값은?``처럼 값 자체를 묻는 문형은 표시 동사를
#: 생략한다. 단위 뒤에 반올림 요구가 있어야만 잡으므로, 공시 본문의 보통 단위
#: 언급을 표시 지시로 오인하지 않는다.
_ROUNDED_VALUE_QUESTION = re.compile(
    rf"(?P<units>{_UNIT_TERM_PATTERN}(?:{_SEPARATOR}{_UNIT_TERM_PATTERN})*)"
    rf"\s*(?:기준\s*)?(?:으로|로)\s*반올림한\s*값\s*(?:은|을)?"
    rf"\s*(?:얼마(?:인가|야)?|몇(?:인가|이야)?|\?)?")

#: ``억원 단위로 얼마인가?``처럼 표시 단위와 값 질문만 있는 문형. ``단위``와
#: 조사, 값 질문을 모두 요구해 공시 본문의 단순 단위 표기는 잡지 않는다.
_UNIT_VALUE_QUESTION = re.compile(
    rf"(?P<units>{_UNIT_TERM_PATTERN}(?:{_SEPARATOR}{_UNIT_TERM_PATTERN})*)"
    rf"\s*(?:기준\s*)?(?:으로|로)\s*"
    rf"(?:얼마(?:인가|야)?|값\s*(?:은|을)?\s*얼마(?:인가|야)?)\??")
_HOW_MANY_UNIT_QUESTION = re.compile(
    rf"몇\s*(?P<units>{_UNIT_WORD})\s*(?:이야|인가|입니까|인지|인가요)\s*\??$")
#: 기준으로」), an optional inline rounding aside, then a display verb
#: (``…로 보여줘``, ``…으로 각각 바꿔 보여줘``, ``…단위로 알려줘``) or an
#: interrogative 「얼마」 tail (``…단위로 얼마인가?``).  The verb/tail anchor
#: is deliberate: without it, a bare unit word appearing anywhere in a
#: question (for example inside an unrelated disclosed amount) would be
#: misread as a display instruction.
_DISPLAY_CLAUSE = re.compile(
    rf"(?P<units>{_UNIT_TERM_PATTERN}(?:{_SEPARATOR}{_UNIT_TERM_PATTERN})*)"
    rf"\s*(?:기준\s*)?(?:으로|로)\s*(?:각각\s*)?(?:함께\s*)?"
    rf"{_INLINE_ROUNDING_ASIDE}"
    rf"(?:{_DISPLAY_VERB}|{_DISPLAY_INTERROGATIVE_TAIL})")

_UNIT_TERM = re.compile(rf"(?P<unit>{_UNIT_WORD})")

_ORDINAL_TO_DECIMALS = {"첫째": 1, "둘째": 2, "셋째": 3, "넷째": 4,
                        "한": 1, "두": 2, "세": 3, "네": 4}

#: ``<unit> 값은 소수점 둘째 자리까지 반올림해줘`` (unit optional — a rounding
#: instruction with no locally named unit applies to every requested scaled
#: unit; see :func:`parse_display_units_directive`).
_ROUNDING_CLAUSE = re.compile(
    rf"(?P<unit>{_UNIT_WORD})?\s*값?\s*(?:은|는)?\s*"
    r"소수점\s*(?P<ordinal>첫째|둘째|셋째|넷째|한|두|세|네|[1-9])\s*(?:번째)?\s*자리"
    r"(?:까지)?\s*반올림")


def is_display_unit_only_surface(surface: str) -> bool:
    """Whether a field/qualifier surface names only a money display unit."""

    return bool(_DISPLAY_UNIT_ONLY_SURFACE.fullmatch((surface or "").strip()))


#: 이슈 #170(M35-b) — 「조원 단위로 얼마인가? 소수점 둘째 자리까지 반올림해줘.」
#: 처럼 반올림 자릿수 지시가 별도 문장으로 떨어져 있으면, HCX가 그 지시를
#: 통째로 두 번째 ``field_surfaces`` 항목("소수점 둘째 자리")으로 얹는다 —
#: 「조원 단위」는 이미 :func:`is_display_unit_only_surface` 가 걸러내지만
#: 자릿수 지시는 단위어가 아니라서 그대로 남아, "필드가 둘인 scalar"라는
#: compiler가 모르는 모양이 되어 통째로 실패한다(compiler_binding_failed).
#: 이 지시 자체도 표시 형식일 뿐 별도 답 필드가 아니므로 같은 방식으로
#: 버린다 — 렌더 시점에 같은 원문에서 ``parse_display_units_directive`` 가
#: 독립적으로 다시 읽는다.
_DISPLAY_ROUNDING_ONLY_SURFACE = re.compile(
    r"^소수점\s*(?P<ordinal>첫째|둘째|셋째|넷째|한|두|세|네|[1-9])\s*(?:번째)?\s*자리"
    r"(?:까지)?(?:\s*반올림(?:해서?|하여)?)?$"
)


def is_display_rounding_only_surface(surface: str) -> bool:
    """Whether a field/qualifier surface names only a rounding-precision cue."""

    return bool(_DISPLAY_ROUNDING_ONLY_SURFACE.fullmatch((surface or "").strip()))


@dataclass(frozen=True, slots=True)
class DisplayUnitsDirective:
    """One or more requested display units, in question order.

    ``rounding`` maps a unit (never ``"원"``, an exact value is never rounded)
    to the number of decimal places *explicitly* requested for it.  A unit
    absent from ``rounding`` still renders at a sensible default precision —
    see ``_DEFAULT_ROUNDING`` in :func:`format_display_unit` (SG-018: 「조원
    단위로 알려줘」 names no rounding, but a raw many-decimal 조원 figure is
    not what "조원 단위" means in practice) — rather than every trailing
    digit of the exact won amount converted.
    """

    units: tuple[str, ...]
    rounding: dict[str, int] = field(default_factory=dict)


def _normalize_unit(text: str) -> str:
    return "".join(text.split())


def parse_display_units_directive(question: str | None) -> DisplayUnitsDirective | None:
    """Recover a requested-display-units instruction from the raw question.

    Returns ``None`` when the question carries no such instruction — the
    ordinary single-value display stays unchanged.
    """

    if not question:
        return None
    clause = _DISPLAY_CLAUSE.search(question)
    if clause is None:
        clause = _ROUNDED_VALUE_QUESTION.search(question)
    if clause is None:
        clause = _UNIT_VALUE_QUESTION.search(question)
    if clause is None:
        clause = _HOW_MANY_UNIT_QUESTION.search(question)
    if clause is None:
        return None
    units: list[str] = []
    for match in _UNIT_TERM.finditer(clause.group("units")):
        unit = _normalize_unit(match.group("unit"))
        if unit not in units:
            units.append(unit)
    if not units:
        return None

    rounding: dict[str, int] = {}
    round_match = _ROUNDING_CLAUSE.search(question)
    if round_match is not None:
        ordinal = round_match.group("ordinal")
        decimals = _ORDINAL_TO_DECIMALS.get(ordinal)
        if decimals is None:
            decimals = int(ordinal)
        unit_text = round_match.group("unit")
        if unit_text:
            bound_unit = _normalize_unit(unit_text)
            if bound_unit in units and bound_unit != "원":
                rounding[bound_unit] = decimals
        else:
            for unit in units:
                if unit != "원":
                    rounding[unit] = decimals

    return DisplayUnitsDirective(units=tuple(units), rounding=rounding)


def _convert_won(won: Decimal, unit: str, *, decimals: int | None) -> Decimal:
    value = won / _UNIT_SCALE[unit]
    if decimals is not None:
        quantum = Decimal(1).scaleb(-decimals)
        value = value.quantize(quantum, rounding=ROUND_HALF_UP)
    return value


def _format_decimal(value: Decimal) -> str:
    negative = value < 0
    text = format(abs(value), "f")
    int_part, _, frac_part = text.partition(".")
    frac_part = frac_part.rstrip("0")
    int_part = f"{int(int_part):,}"
    body = f"{int_part}.{frac_part}" if frac_part else int_part
    return f"{'-' if negative else ''}{body}"


#: Applied only when the caller passes no explicit ``decimals`` (SG-018 —
#: 「조원 단위로 알려줘」 names no rounding clause at all).  A raw won amount
#: converted to 조원/억원 with no rounding shows a dozen digits of scale
#: noise nobody asked to see (「3.326553694966조원」); every other unit
#: (백만원/천원/원) already lands on a small, exact number of digits and
#: keeps showing its full precision.
_DEFAULT_ROUNDING = {"조원": 2, "억원": 1}


def format_display_unit(
        won: Decimal, unit: str, *, decimals: int | None = None) -> str:
    """Render one requested unit's amount, comma-grouped, exact or rounded."""

    if unit == "원":
        return f"{int(won):,}원"
    if decimals is None:
        decimals = _DEFAULT_ROUNDING.get(unit)
    value = _convert_won(won, unit, decimals=decimals)
    if decimals is None:
        return f"{_format_decimal(value)}{unit}"
    # An explicit decimal count is a formatting requirement, not merely a
    # rounding mode: keep every requested trailing digit (``14.20`` must not
    # collapse to ``14.2``) instead of stripping trailing zeros.
    sign = "-" if value < 0 else ""
    quantized = format(abs(value), "f")
    int_part, _, frac_part = quantized.partition(".")
    int_part = f"{int(int_part):,}"
    body = f"{int_part}.{frac_part}" if frac_part else int_part
    return f"{sign}{body}{unit}"


def render_display_units(
        won_value: object, directive: DisplayUnitsDirective) -> str | None:
    """Combine a canonical won amount into the requested display units.

    Returns ``None`` when ``won_value`` is not a finite integral won amount
    (the caller then keeps its ordinary display); the directive's units are
    otherwise never dropped silently.
    """

    try:
        won = Decimal(str(won_value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not won.is_finite() or won != won.to_integral_value():
        return None
    parts = [
        format_display_unit(won, unit, decimals=directive.rounding.get(unit))
        for unit in directive.units
    ]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]}({', '.join(parts[1:])})"

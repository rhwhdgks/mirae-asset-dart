"""composer 안전 경계.

- sanitize_for_llm: HCX에 넘기는 payload 텍스트는 prompt-safe 값뿐. 계약상 raw는 애초에 payload에
  없지만, 발췌문 안의 <<<UNTRUSTED_DART_DISCLOSURE_DATA_BEGIN/END>>> 경계는 벗기지 않고 그대로 둔다.
- verify_numbers: 생성 문장의 숫자가 payload에 없는 값이면 실패 → 템플릿 폴백. LLM이 숫자를 지어내거나
  계산하는 것을 사후에 막는다(합의안: 계산·값 선택은 Python).
- verify_display_units: 질문이 표시 단위를 지정했으면(예: "억원과 조원으로 각각 바꿔 보여줘") 그
  단위 표현이 전부 본문에 있어야 통과. HCX가 숫자 검증은 통과하는 기본 포맷("14조 2,893억원")만
  써도 그것만으로는 부족하다 — pool에는 기본 포맷 숫자도 이미 들어 있어 verify_numbers만으로는
  잡히지 않는다(이슈 #61 팔로업, RPC-005 실호출 FAIL).
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re

from agent.display_units_v1 import format_display_unit, parse_display_units_directive
from agent.holding_subject_mask import MASK_CHAR
from src.canonical.security import is_organization_name

from .limitations import render_safe_limitations

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
# One- and two-digit values are normally excluded because dates, list numbers
# and small percentages would make free wording needlessly brittle.  A value
# immediately bound to an energy/capacity unit is different: ``75GWh`` is a
# substantive disclosed quantity and must not disappear merely because it has
# fewer than three digits.
_SHORT_UNIT_VALUE = re.compile(
    r"(?<![\d,.])(?P<value>\d{1,2}(?:\.\d+)?)\s*"
    r"(?P<unit>TWh|GWh|MWh|GW|MW|kW)(?![A-Za-z])",
    re.IGNORECASE,
)


def _tokens(s: str) -> set[str]:
    out = set()
    for t in _NUM.findall(s or ""):
        t2 = t.replace(",", "")
        if len(t2) >= 3:          # 1~2자리(연·월·순번·퍼센트 소수 제외)는 검사 대상 아님
            out.add(t2)
    return out


def payload_number_pool(payload) -> set[str]:
    pool: set[str] = set()
    for c in payload.claims:
        for v in (c.value_text, c.canonical_value, c.text, c.label):
            pool |= _tokens(v or "")
    for l in payload.limitations:
        pool |= _tokens(l.detail)
    for d in payload.applied_defaults:
        pool |= _tokens(d)
    for c in payload.claims:
        for ct in c.citations:
            if ct.rcept_no:
                pool.add(ct.rcept_no)
    # 원 단위 환산의 한국어 표기(조·억·만 자릿수 조각) 허용: 333,605,938백만원 → 333조 6,059억 3,800만원
    for c in payload.claims:
        won = None
        if c.canonical_value and c.canonical_unit == "원":
            try: won = abs(int(float(c.canonical_value)))
            except ValueError: won = None
        elif c.value_text and c.raw_unit in ("원", "백만원", "천원", "억원", "조원"):
            try:
                from app.tools._units import parse_money
                d, u, w = parse_money(f"{c.value_text}{c.raw_unit}")
                won = abs(int(w)) if w is not None else None
            except Exception: won = None
        if won is None:
            continue
        jo, rem = divmod(won, 10**12); eok, rem = divmod(rem, 10**8); man, rem = divmod(rem, 10**4)
        for v in (jo, eok, man, won // 10**8, won // 10**4, won // 10**6):
            if v: pool.add(str(v))
        for div in (10**12, 10**8):
            if won // div:
                pool.add(f"{won/div:.2f}".rstrip("0").rstrip(".")); pool.add(f"{won/div:.1f}".rstrip("0").rstrip("."))
    # 사용자 전제의 수치(정정 문장에서 인용됨)와 전제 판정 detail의 숫자
    for v in payload.premise_verdicts:
        pool |= _tokens(v.detail or "")
        for t in re.findall(r"주장 ([\d,]+)원", v.detail or ""):
            w = int(t.replace(",", ""))
            for div in (10**12, 10**8):
                if w // div: pool.add(str(w // div))
    # 퍼센트 계산의 관용 표기 "100"(백분율 설명) 허용
    if any(c.canonical_unit == "%" or c.raw_unit == "%" for c in payload.claims):
        pool.add("100")
    return pool


def verify_numbers(text: str, payload) -> tuple[bool, list[str]]:
    """생성 문장의 숫자(3자리↑)가 전부 payload에서 유래했는지. 반환 (ok, 미출처 숫자 목록)."""
    pool = payload_number_pool(payload)
    # 날짜(YYYY-MM-DD / YYYY년 M월 D일)의 구성 숫자는 허용
    stripped = re.sub(r"\d{4}[-./년]\s?\d{1,2}[-./월]\s?\d{1,2}일?", " ", text or "")
    stripped = re.sub(r"\d{4}년", " ", stripped)
    stripped = re.sub(r"제\d+기", " ", stripped)
    unknown = sorted(t for t in _tokens(stripped) if t not in pool)
    return (not unknown), unknown


def display_units_target_claims(payload):
    """The one non-derived scalar 원 claim a display-units directive can bind to.

    Mirrors ``app.composer.template._display_units_replacement``: a directive
    names one value to convert, so with zero or multiple eligible claims this
    verification does not apply (the composer never picks among them either).
    """

    candidates = [
        c for c in payload.claims
        if not c.derived_from and c.canonical_unit == "원" and c.canonical_value
    ]
    if len(candidates) == 1:
        return candidates
    # A two-value rate comparison must expose both calculation operands.
    # Apply display rounding to those values only, never to the calculation
    # inputs in the payload or the derived percentage itself.
    rates = [c for c in payload.claims if c.operator == "percent_change"]
    if (len(candidates) == 2 and len(rates) == 1
            and set(rates[0].derived_from) == {c.output_id for c in candidates}):
        return candidates
    return []


def _display_units_target_claim(payload):
    candidates = display_units_target_claims(payload)
    return candidates[0] if len(candidates) == 1 else None


def expected_display_unit_strings(payload, question: str | None) -> list[str]:
    """Every literal unit string a display-units directive requires verbatim."""

    directive = parse_display_units_directive(question)
    if directive is None:
        return []
    values = []
    for claim in display_units_target_claims(payload):
        try:
            won = Decimal(str(claim.canonical_value))
        except (InvalidOperation, TypeError, ValueError):
            return []
        if not won.is_finite() or won != won.to_integral_value():
            return []
        values.extend(format_display_unit(
            won, unit, decimals=directive.rounding.get(unit))
            for unit in directive.units)
    return values


def verify_display_units(text: str, payload, question: str | None) -> tuple[bool, list[str]]:
    """질문이 표시 단위를 지정했으면 그 단위 표현이 전부 본문에 있는지.

    ``verify_numbers``는 숫자 자체(및 그 조·억·만 자릿수 조각)가 payload에서
    유래했는지만 본다. HCX가 기본 Korean 조·억·만원 표기("14조 2,893억원")로
    쓰면 그 숫자들도 이미 pool에 있어 ``verify_numbers``는 통과하지만, 질문이
    명시한 표시 단위("142,893.9억원", "14.29조원")는 여전히 빠질 수 있다
    (이슈 #61 팔로업, RPC-005 실호출 FAIL). 반환 (ok, 누락된 요구 단위 표현 목록).
    """

    missing = [
        value for value in expected_display_unit_strings(payload, question)
        if value not in (text or "")
    ]
    return (not missing), missing


def sanitize_for_llm(s: str | None, limit: int = 4000) -> str:
    if not s:
        return ""
    return s[:limit]


_HEDGE_PATTERNS = ("예상됩니다", "예상돼", "예측", "추정됩니다", "실제 결과와 다를", "실제 결과는 다를",
                   "차이가 있을 수 있습니다", "달라질 수 있습니다", "확실하지 않", "가능성이 있습니다")


def verify_no_hedging(text: str) -> tuple[bool, list[str]]:
    """확정 공시 값에 예측·면책 문구를 붙이면 실패 (정확성·환각 방지 채점 위험)."""
    hits = [h for h in _HEDGE_PATTERNS if h in (text or "")]
    return (not hits), hits


#: 「0을 미기재·공란·미공시로 완곡하게 바꾸지 않는다」는 것을 문면에서 증명하는
#: 표지. explicit_zero claim이 있을 때 이 중 하나도 없으면 free-form 서술이
#: "확인할 수 없음"류로 0을 지웠을 위험이 있다고 보아 반려한다.
_EXPLICIT_ZERO_MARKERS = ("명시", "확정된 값", "0으로 기재", "0으로 공시")
_ZERO_DOWNGRADE_PATTERNS = ("확인할 수 없", "공시되지 않", "미공시", "공란", "비공개")


def verify_explicit_zero_disclosed(text: str, payload) -> tuple[bool, list[str]]:
    """`state="explicit_zero"` claim은 0이 원문에 명시된 값임을 문장에서 밝혀야 한다.

    schema 1.8 ``value_status=explicit_zero``는 원문이 스스로 「0」이라고 적은
    필드다(빈 칸·대시가 아니다). 값 자체(`verify_coverage`)는 한 자리 숫자라
    커버리지 검사가 사실상 무력하므로, 여기서 별도로 「명시」류 표지가 있는지,
    또는 "확인할 수 없음"류 완곡 표현으로 덮이지 않았는지를 본다.
    """

    labels = [c.label for c in payload.claims if c.state == "explicit_zero"]
    if not labels:
        return True, []
    body = text or ""
    if any(marker in body for marker in _EXPLICIT_ZERO_MARKERS):
        return True, []
    if any(pattern in body for pattern in _ZERO_DOWNGRADE_PATTERNS):
        return False, labels
    return True, []


def verify_citations(text: str, payload) -> tuple[bool, list[str]]:
    """answer 계열은 근거 접수번호가 문장에 최소 1개 있어야 한다."""
    rcs = {ct.rcept_no for c in payload.claims for ct in c.citations if ct.rcept_no}
    if not rcs:
        return True, []
    have = [r for r in rcs if r in (text or "")]
    return (bool(have)), ([] if have else sorted(rcs)[:3])


def verify_limitations(
        text: str, payload, question: str | None = None,
        ) -> tuple[bool, list[str]]:
    """Required public-safe correction notices must survive composition."""
    required = render_safe_limitations(payload.limitations, question=question)
    missing = [message for message in required if message not in (text or "")]
    return (not missing), missing


#: 지분공시 보고 주체 문맥 마커. 이슈 #51 — 마스킹은 ``app/tools/holding.py``
#: 실행 시점에서 하지만, 자유 서술 경로(HCX)가 다른 표현으로 이름을 되풀이
#: 하거나 교정 재시도가 가림을 되돌릴 위험이 있어 최종 문자열에도 방어선을 둔다.
#: 관측된 실제 노출(이슈 #51 HM-020)은 항상 마커 뒤에 이름이 온다
#: (「보고자는 최윤범이며」) — 한국어 문장에서 「이름이 보고자다」처럼 마커
#: 앞에 이름이 오는 어순은 드물고, 마커 앞을 함께 보면 마커보다 먼저 오는
#: 흔한 부사·수식어(「가장 최근 제출된」)까지 한글 2~3글자로 걸려 오탐이
#: 급증한다. 그래서 마커 뒤 20자만 본다.
_HOLDING_SUBJECT_CONTEXT = re.compile(r"보고자|대량보유자|특별관계자")

#: 이미 부분 가림된 값(``최○범``)의 남은 한 글자가 뒤따르는 조사와 우연히
#: 합쳐 새 한글 런을 만들지 않도록, 스캔 전에 가려진 조각을 지운다.
#: 가림 글자는 `agent.holding_subject_mask` 에서 가져온다 — 여기에 글자를
#: 따로 적어 두면 한쪽만 바뀌었을 때 가려진 값을 다시 후보로 올린다.
_ALREADY_MASKED_FRAGMENT = re.compile(
    r"[가-힣]" + re.escape(MASK_CHAR) + r"[가-힣]?")

#: 이름 뒤에 조사 없이 바로 붙는 한국어 문장부호 없는 조사·어미. 값이 아니라
#: 문장 안에서 그 값을 가리키는 문법 요소이므로, 후보 끝에서 걷어낸 뒤에도
#: 남는 부분만 이름 후보로 본다.
_TRAILING_GRAMMAR = tuple(sorted({
    "의", "는", "은", "이", "가", "를", "을", "에", "와", "과", "도", "만",
    "으로", "로", "에서", "부터", "까지", "보다", "처럼", "라", "이라",
    "이며", "이고", "이자", "인데", "입니다", "이었습니다", "였습니다",
    "이었으며", "였으며", "이라고", "라고", "이라는", "라는", "께서",
    "에게", "한테",
}, key=len, reverse=True))

#: 한글이 앞뒤로 더 이어지지 않는(단어 경계) 한글 런 전체.
_HANGUL_RUN = re.compile(r"(?<![가-힣])[가-힣]+(?![가-힣])")

#: 마커 자체와, 마커 주변에 흔히 오는 일반 명사 — 사람 이름이 아니다.
#: 마커 뒤 몇 글자까지를 이름 자리로 보는가.
_SUBJECT_WINDOW = 20

_HOLDING_SUBJECT_NON_NAME_WORDS = frozenset({
    "보고자", "구분", "본인", "법인", "명칭", "성명", "관계", "해당", "현재",
    "이하", "위와", "동일", "합계", "각각", "전원", "명단", "목록", "내역", "여부",
    # 지분공시 칸 이름과 흔한 이음말. 이 guard 는 마커 뒤 20자 안의 한글 2~3자를
    # 이름 후보로 보므로, 「보고자·특별관계자별 직업(사업내용)」 같은 **제목**이
    # 그대로 걸린다. 실제로 `직업`·`다음` 때문에 HCX composer 가 두 번 반려되고
    # template 으로 떨어졌다 (#138). 창을 넓게 잡은 구조 자체는 #140 의 몫이다.
    "직업", "국적", "내용", "다음", "아래", "기준", "기타", "이번", "직전",
    "모두", "전부", "그중", "다만", "또한", "함께", "순서", "각사", "총액",
})


def _strip_trailing_grammar(run: str) -> str:
    """조사·어미를 반복해서 걷어내고 남는 어간만 돌려준다.

    ``최윤범이며`` → ``이며`` 를 떼면 ``최윤범``. 어간이 조사/어미 자체와
    같으면(``이며`` 단독) 빈 문자열이 되어 이름 후보에서 자연히 제외된다.
    """

    changed = True
    while changed:
        changed = False
        for suffix in _TRAILING_GRAMMAR:
            if run == suffix:
                run = ""
                changed = True
                break
            if len(run) > len(suffix) and run.endswith(suffix):
                run = run[:-len(suffix)]
                changed = True
                break
    return run


#: 지분공시 도구가 **자연인 행을 빼고** 만든 목록 claim 들. 여기 실린 이름은
#: 정본이 공개로 판정한 법인·기관이다.
_PUBLIC_PARTY_CLAIM_LABELS = (
    "공개된 특별관계자 법인·기관명", "특별관계자별 보유내역",
    "보고자·특별관계자별 국적", "보고자·특별관계자별 직업(사업내용)",
)
_PUBLIC_PARTY_LINE = re.compile(r"^\s*\d+\.\s*([^,\n]+)")
_SHORT_COUNTRY_NAMES = frozenset({
    "한국", "미국", "일본", "중국", "영국", "독일", "호주", "대만", "홍콩",
    "캐나다", "프랑스", "스위스", "러시아", "브라질", "인도", "태국", "베트남",
})

# Remove only fixed renderer language/field labels, never the following value
# or an entire line. A name inserted next to a notice must still be checked.
_HOLDING_RENDERER_NOTICE = re.compile(
    r"개인정보\s*보호를\s*위해(?:\s*일부만\s*표시했습니다)?"
    r"|법인\s*[·ㆍ/]\s*기관명"
    r"|(?:근거|출처|참고|국적|직업|현황|비율|주식|지분|공시|제출|확인|표시|기재)\s*(?=[:：])"
    r"|제공(?:된)?\s*(?:공시\s*)?자료"
)


def _payload_public_subject_names(payload) -> set[str]:
    """payload 가 **이미 공개로 판정해** 실은 지분공시 주체 이름.

    guard 는 값 모양(순수 한글 2~3글자)으로만 자연인을 찾는다. 접미사 없는
    법인명(``에코프로``·``영풍``)은 그 모양과 구별되지 않아, 정본이 ``보고자
    구분``으로 법인이라고 확정한 이름까지 「가려지지 않은 성명」으로 잡아
    답변 전체를 반려한다 (이슈 #199). payload 가 이미 들고 있는 공개 판정을
    그대로 읽어 예외로 둔다 — 새 값 휴리스틱을 만들지 않는다.

    자연인은 그대로 걸린다. 도구가 가린 값에는 가림 글자가 남고, 자유 서술이
    payload 밖에서 되살린 원문 성명은 애초에 이 목록에 없다.
    """

    names: set[str] = set()
    for claim in getattr(payload, "claims", None) or ():
        label = str(getattr(claim, "label", "") or "")
        text = str(getattr(claim, "text", "") or "").strip()
        if not text:
            continue
        if label.endswith("보고자"):
            if MASK_CHAR not in text and "[REDACTED" not in text:
                names.add(text)
            continue
        if any(marker in label for marker in _PUBLIC_PARTY_CLAIM_LABELS):
            for line in text.splitlines():
                match = _PUBLIC_PARTY_LINE.match(line)
                if match:
                    names.add(match.group(1).strip())
    return names


def verify_no_unmasked_holding_subject_name(
        text: str, payload=None) -> tuple[bool, list[str]]:
    """지분공시 보고자·특별관계자 문맥에 가리지 않은 개인 성명이 남아 있는가.

    ``app/tools/holding.py`` 는 실행 시 ``mask_holding_subject_name`` 으로
    filer 성명을 가리지만, 자유 서술(HCX) 경로가 payload 밖 표현으로 이름을
    다시 쓰거나 교정 재시도가 가림을 되돌릴 수 있다. 이 guard 는 최종 출력
    문자열만 보는 마지막 방어선이다. 「보고자」·「대량보유자」·「특별관계자」
    마커 뒤 20자 안에서, 조사·어미를 걷어낸 뒤에도 순수 한글 2~3글자로
    남는 값이 있으면 반려한다. payload의 claim 라벨 앞 토큰(발행회사·법인명
    등), payload가 이미 공개로 판정해 실은 주체 이름
    (:func:`_payload_public_subject_names`), ``is_organization_name`` 으로
    판정되는 기관·법인명, 「구분」·「본인」·「법인」 같은 일반 명사는
    오탐이므로 제외한다.
    """

    # Replace with equal-length spaces so the original 20-character boundary
    # does not move, and a following personal name is never removed.
    haystack = _HOLDING_RENDERER_NOTICE.sub(
        lambda match: " " * len(match.group(0)), text or "")
    haystack = _ALREADY_MASKED_FRAGMENT.sub("○", haystack)
    corporate_names: set[str] = set()
    for claim in getattr(payload, "claims", None) or ():
        label = getattr(claim, "label", "") or ""
        first = label.split(" ", 1)[0].strip()
        if first:
            corporate_names.add(first)
    corporate_names.update(_payload_public_subject_names(payload))
    # A country is exempt only when bound to an explicitly typed nationality
    # claim. This is not a general exemption for arbitrary payload prose.
    for claim in getattr(payload, "claims", None) or ():
        if str(getattr(claim, "label", "") or "").endswith(" 국적"):
            value = str(getattr(claim, "text", "") or "").strip()
            if value in _SHORT_COUNTRY_NAMES:
                corporate_names.add(value)

    hits: list[str] = []
    for context in _HOLDING_SUBJECT_CONTEXT.finditer(haystack):
        # 창을 20자로 자르면 **낱말이 잘린다** — `대한민국` 이 `대한민` 으로,
        # `이재용의` 가 `이재` 로 남아 없는 이름을 만들어 낸다. 넉넉히 떠서
        # 낱말을 온전히 읽고, 시작점이 20자 안인 것만 본다 (#138).
        window = haystack[context.end():context.end() + _SUBJECT_WINDOW * 2]
        for match in _HANGUL_RUN.finditer(window):
            if match.start() >= _SUBJECT_WINDOW:
                break
            candidate = _strip_trailing_grammar(match.group(0))
            if len(candidate) not in (2, 3):
                continue
            if candidate in _HOLDING_SUBJECT_NON_NAME_WORDS:
                continue
            if is_organization_name(candidate, corporate_names):
                continue
            if candidate not in hits:
                hits.append(candidate)
    return (not hits), hits


def verify_operand_placement(text: str, payload) -> tuple[bool, list[str]]:
    """계산 피연산자(operand) 원값은 '계산 근거:' 줄에만 — 본문 문장에 쓰면 채점기 text_forbidden과
    충돌한다(예: 반기 누적 153,706,820을 본문에 쓰면 2분기 매출로 오인 제시로 채점). 채점기와 같은
    기준으로 '계산 근거:'로 시작하는 줄만 제외한다. 반환 (ok, 본문에 등장한 operand 값 목록)."""
    derived = [c for c in payload.claims if c.derived_from and c.value_text]
    if not derived:
        return True, []
    operand_ids = {o for c in derived for o in c.derived_from}
    vals = [c.value_text.replace(",", "").strip("()") for c in payload.claims
            if c.output_id in operand_ids and not c.derived_from and c.value_text]
    body = "\n".join(l for l in (text or "").splitlines()
                     if not l.startswith("계산 근거:")).replace(",", "")
    hits = [v for v in vals if v and v in body]
    return (not hits), hits


def verify_coverage(text: str, payload) -> tuple[bool, list[str]]:
    """생성 문장이 payload의 핵심 값(수치 claim의 value_text, 상태 claim의 상태어)을 담고 있는가.
    LLM이 주입·환각으로 엉뚱한 글을 써도 숫자 검증만으로 못 잡는 경우를 막는 최소 완전성 검사.
    답변에 반드시 나와야 하는 값: 계산 결과(derived)와 그 operand 원값 — gold는 증감률뿐 아니라
    두 시점 원값까지 요구한다. derived가 없으면 모든 수치 claim."""
    derived = [c for c in payload.claims if c.derived_from and c.value_text]
    if derived:
        operand_ids = {o for c in derived for o in c.derived_from}
        operands = [c for c in payload.claims
                    if not c.derived_from and c.output_id in operand_ids and c.value_text]
        targets = derived + operands
    else:
        targets = [c for c in payload.claims if c.value_text]
    must = [c.value_text.replace(",", "") for c in targets]
    # Narrative tables carry their values in ``text`` rather than
    # ``value_text``.  Require every numeric cell as well: otherwise a model
    # can cite the report yet silently omit tail rows.  Non-numeric prose is
    # intentionally not forced verbatim because the ordinary composer may
    # summarize it; long-form narrative uses the deterministic template path.
    for claim in payload.claims:
        if claim.text and not claim.value_text:
            must.extend(_tokens(claim.text))
            must.extend(
                match.group("value").replace(",", "")
                for match in _SHORT_UNIT_VALUE.finditer(claim.text)
            )
    norm = (text or "").replace(",", "")
    # 재무제표 음수 괄호 표기 "(10,833,917)"는 괄호 없이 써도 같은 값으로 인정
    missing = [m for m in must if m not in norm and m.strip("()") not in norm]
    # 값이 하나도 없는 서술형 답변(narrative)은 상태어/근거 접수번호 포함으로 판단
    if not must:
        rc = [ct.rcept_no for c in payload.claims for ct in c.citations if ct.rcept_no]
        if rc and not any(r in (text or "") for r in rc):
            return False, ["citation_missing"]
        return True, []
    # 길이에 관계없이 요청된 확정 값은 전부 필요하다. 목록이 길다는 이유로 절반만 들어간
    # 답변을 완전한 답으로 채택하면 행·슬롯 누락이 조용히 통과한다.
    return (not missing), missing


#: 「다음과 같습니다」처럼 뒤에 내용이 오리라고 예고하는 문장의 끝.
_PROMISES_A_LIST = re.compile(
    r"(?:다음과 같습니다|다음과 같이[^.\n]{0,20}습니다|"
    r"다음과 같은[^.\n]{0,30}(?:밝혔|말했|설명했)습니다)[.]?$")

#: 예고 뒤에 와도 내용으로 치지 않는 줄 — 근거 표시뿐이다.
_CITATION_ONLY_LINE = re.compile(r"^[-\s]*(?:공시\s*)?접수번호|^근거[:：]")


def verify_promise_delivered(text: str) -> tuple[bool, list[str]]:
    """내용을 예고했으면 실제로 실어야 한다.

    K-047 은 「현대제철은 최근 정기보고서에서 다음과 같은 사업을 하고 있다고
    밝혔습니다.」 뒤에 접수번호 한 줄만 놓았다. 조회가 문서만 찾고 본문을
    읽지 못했는데 문장은 목록이 이어질 것처럼 썼다. 읽는 사람은 빠뜨린 줄을
    찾게 되고, 답이 없다는 사실 자체가 가려진다.

    예고 문장 뒤에 근거 표시가 아닌 줄이 하나도 없으면 반려한다.
    """

    lines = [line for line in (text or "").splitlines() if line.strip()]
    broken = []
    for i, line in enumerate(lines):
        if not _PROMISES_A_LIST.search(line.strip()):
            continue
        if not any(not _CITATION_ONLY_LINE.match(rest.strip())
                   for rest in lines[i + 1:]):
            broken.append(line.strip()[:60])
    return not broken, broken

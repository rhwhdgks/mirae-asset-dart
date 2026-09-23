"""Question-ID-free policy outcomes for the Stage1 v1 resolver."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any
import unicodedata

from agent.planner_policy import causal_policy_reasons, pressure_policy_applies
from agent.semantic_intent_v1 import SemanticIntent

#: 공시로 증명할 수 없는 인과 = **주가** 인과. 사업 인과(「수주 증가로 매출이 늘었나」)는 거절 대상이 아니다.
_SHARE_PRICE = re.compile(r"주가|주식\s*가격|시가\s*총액|시총|주식\s*시장")

#: 욕설·모욕·위협 사전. `용어\t등급\t비고` 3열, `#` 주석·빈 줄은 건너뛴다.
ABUSIVE_TERMS_FILE = Path(__file__).resolve().parent / "abusive_terms.tsv"


def _load_abusive_terms(path: Path | None = None) -> tuple[str, ...]:
    source = path or ABUSIVE_TERMS_FILE
    if not source.is_file():
        return ()
    terms: list[str] = []
    seen: set[str] = set()
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        term = line.split("\t", 1)[0].strip()
        if term and term not in seen:
            seen.add(term)
            terms.append(term)
    return tuple(terms)


_ABUSIVE_TERMS = _load_abusive_terms()


def _contains_word(text: str, term: str) -> bool:
    """``term`` occurs at a word boundary — neither neighbour is Hangul.

    Plain substring containment would let a full desired term match inside an
    unrelated longer Hangul word (「질병신고」contains 「병신」). Requiring both
    neighbours to be non-Hangul (or absent) keeps the dictionary's short,
    common-collision-risk entries safe without banning them outright.
    """

    start = text.find(term)
    while start != -1:
        before = text[start - 1:start]
        after = text[start + len(term):start + len(term) + 1]
        if ((not before or not ("가" <= before <= "힣"))
                and (not after or not ("가" <= after <= "힣"))):
            return True
        start = text.find(term, start + 1)
    return False


def _has_abusive_term(raw: str) -> bool:
    return any(_contains_word(raw, term) for term in _ABUSIVE_TERMS)


#: 공시 질문임을 드러내는 어휘 — 회사 실체 없이 이 중 하나만 있어도 정상 질문
#: 요소로 본다(fail-open). 영문 약어는 `raw`가 이미 casefold 되어 있으므로
#: 대소문자 무관 매칭이 필요하다.
#:
#: 계약의 **생애**(수주·공급·해지·해제·종료)도 계약 자체와 같은 자격으로 넣는다.
#: 「Ford랑 Freudenberg 해지건 상대·금액·사유 표로」(`R-P-018`)는 공시 사건을
#: 정확히 묻는데도 상대방이 코퍼스 밖 회사라 회사 실체로 잡히지 않고, 「해지」가
#: 어휘에 없어 무관 질문으로 거절됐다 — 2026-09-04 Core334 전수 실호출에서
#: 09-01 기준선 대비 유일한 회귀였다.
#:
#: 「벌었/벌 것 같」은 실적(매출·이익)을 구어로 묻는 동사 표지다. 회사명이
#: 없는 「2025년에 얼마나 벌었어?」는 이 표지가 없으면 회사 누락 역질문까지
#: 가지 못하고 이 게이트에서 곧장 일반 무관 거절로 닫혔다(issue #172 M29) —
#: 「작년에 매출이 얼마였어?」처럼 「매출」이 박힌 형제 문형만 통과했었다.
_DISCLOSURE_VOCAB = re.compile(
    r"매출|영업이익|순이익|당기순이익|총자산|자산총계|부채|자본|지분|투자|"
    r"자금조달|공시|보고서|계약|수주|공급|해지|해제|종료|소송|배당|증자|감자|"
    r"인수|합병|임원|대표이사|감사|사채|특허|계열사|종속회사|자회사|현금흐름|"
    r"실적|상장|주식|정정|접수번호|capex|r&d|연구개발|이사회|주주|eps|roe|"
    r"벌었|벌\s*것\s*같|벌어들",
    re.IGNORECASE,
)


def _normalize_question(question: str) -> str:
    return unicodedata.normalize("NFKC", question).casefold().strip()


def _mentions_company_text(question: str, company_resolver: Any) -> bool:
    """Best-effort raw-text company scan via ``company_resolver.companies_in_text``.

    Not every caller implements this (test doubles for other backends usually
    only implement ``resolve_company``, and the pre-HCX gate has no parsed
    entities yet), so a missing or failing lookup means "unknown" rather than
    "no company".
    """

    finder = getattr(company_resolver, "companies_in_text", None)
    if not callable(finder):
        return False
    try:
        return bool(finder(question))
    except Exception:
        return False


def classify_abusive_or_off_topic(
        question: str, *, company_resolver: Any, entities: Any = (),
        ) -> str | None:
    """Return ``"abusive_input"``/``"off_topic_request"``/``None`` for ``question``.

    The single, deterministic, HCX-free rule shared by two call sites:
    ``GeneralPolicyResolutionBackend.resolve()``'s terminal fallback (with the
    parsed ``SemanticIntent`` entities available) and the pre-HCX gate in
    ``server/stage1.py`` (no entities parsed yet — only a raw-text company
    scan). **정상 질문 요소**(회사 entity/raw-text 언급, 또는 공시 어휘 하나)가
    하나라도 있으면 판정하지 않는다(fail-open) — 욕설이 섞인 정상 질문
    (「삼성전자 매출액 좀 알려줘 …이 멍청한 시스템아」)은 욕설을 무시하고 정상
    경로로 흘려보낸다.
    """

    raw = _normalize_question(question)
    has_company = (
        any(getattr(entity, "kind_hint", None) == "company" for entity in entities)
        or _mentions_company_text(question, company_resolver))
    has_disclosure_vocab = bool(_DISCLOSURE_VOCAB.search(raw))
    if has_company or has_disclosure_vocab:
        return None
    return "abusive_input" if _has_abusive_term(raw) else "off_topic_request"


def _terminal(source_intent: SemanticIntent, reasons: list[str]) -> dict[str, Any]:
    item_ids = [item.item_id for item in source_intent.answer_items]
    return {
        "kind": "terminal",
        "reasons": [
            {"code": code, "scope": "question", "item_ids": item_ids}
            for code in reasons
        ],
    }


#: 회사명이 아니라 구절임을 드러내는 표지 — 날짜, 조사가 붙은 주어, 서술어.
_PHRASE_NOT_A_COMPANY = re.compile(
    r"\d+\s*[년월일]|보고서|공시|제출|기준|작성|매출|영업이익|순이익|"
    r"[가-힣]{2,}(?:가|이|은|는)\s")


class GeneralPolicyResolutionBackend:
    """Resolve question-grounded policy and supported-universe boundaries.

    These decisions depend on stable request classes and the canonical company
    registry, never on a fixture ID or expected handoff.
    """

    def __init__(self, company_resolver: Any, *, corpus_cutoff: str) -> None:
        if not callable(getattr(company_resolver, "resolve_company", None)):
            raise TypeError("policy backend에는 company resolver가 필요합니다")
        if re.fullmatch(r"[0-9]{8}", corpus_cutoff) is None:
            raise ValueError("corpus_cutoff은 YYYYMMDD여야 합니다")
        self.company_resolver = company_resolver
        self.corpus_cutoff = corpus_cutoff

    @staticmethod
    def _text(question: str) -> str:
        return _normalize_question(question)

    @staticmethod
    def _explicit_financial_company(
            question: str, source_intent: SemanticIntent,
            ) -> str | None:
        referenced = {
            ref
            for item in source_intent.answer_items
            if item.target.kind == "metric"
            for ref in item.target.entity_refs
        }
        surfaces = {
            entity.surface.strip()
            for entity in source_intent.entities
            if entity.entity_id in referenced and entity.kind_hint == "company"
        }
        if len(surfaces) == 1:
            return next(iter(surfaces))
        if surfaces:
            return None
        # Narrative comparisons can contain the word ``매출`` without being
        # a scalar financial lookup.  The leading phrase may then contain two
        # issuers and must never be reinterpreted as one absent company.
        if not any(item.target.kind == "metric"
                   for item in source_intent.answer_items):
            return None
        if not re.search(r"매출|영업이익|순이익|총자산|자산총계|capex", question):
            return None
        match = re.search(
            r"^\s*([0-9A-Za-z가-힣&. ]{1,30}?)(?:의|\s+자료)", question)
        if match is None:
            return None
        surface = match.group(1).strip()
        # 첫 「의」까지 통째로 삼키면 회사명이 아니라 구절이 잡힌다.
        # 「삼성전자가 2026년 3월 10일 제출한 사업보고서의 …」에서 이 정규식은
        # 「삼성전자가 2026년 3월 10일 제출한 사업보고서」를 회사명으로 봤고,
        # 그런 회사가 없으니 답이 「삼성전자는 제공 70개 기업에 없다」가 됐다.
        # 코퍼스에 있는 회사를 없다고 말하는 것이라 그냥 틀린 답이다.
        #
        # 이 폴백은 Stage1 이 회사를 지표에 결속하지 못했을 때만 도는 마지막
        # 수단이므로, 회사 이름처럼 생긴 것만 받는다. 날짜·서술어가 섞였으면
        # 회사명이 아니라 구절이다 — 그때는 범위 밖이라고 단정하지 않는다.
        return None if _PHRASE_NOT_A_COMPANY.search(surface) else surface

    def _company_exists(self, surface: str) -> bool | None:
        """Return universe membership without treating held aliases as absent."""
        try:
            rows = self.company_resolver.resolve_company(surface)
            if not rows:
                finder = getattr(
                    self.company_resolver, "held_company_candidates", None)
                if callable(finder) and finder(surface):
                    return True
        except Exception:
            return None
        return bool(rows) if isinstance(rows, list) else None

    def resolve(
            self, *, question_id: str, question: str,
            source_intent: SemanticIntent,
            ) -> dict[str, Any] | None:
        del question_id
        if not isinstance(question, str) or not question.strip():
            raise ValueError("policy backend에는 원 질문이 필요합니다")
        raw = self._text(question)
        pressure = pressure_policy_applies(raw)

        role_injection = bool(
            re.search(r"(?:^|\s)[\[<](?:system|developer|assistant)[\]>]", raw)
            or (
                re.search(r"나는.{0,20}(?:운영자|관리자|개발자|시스템)", raw)
                and re.search(r"이전.{0,12}(?:지시|규칙).{0,12}(?:무시|폐기)", raw)
            )
        )
        direct_injection = (
            bool(re.search(r"(?:이전|앞선|위의?)\s*(?:지시|규칙).{0,12}(?:무시|폐기)", raw))
            and any(cue in raw for cue in (
                "시스템 프롬프트", "내부 프롬프트", "비밀키", "비밀 키",
                "비밀 환경변수", "환경변수", "api 키", "api key",
            ))
        )
        if role_injection or direct_injection:
            return _terminal(source_intent, [
                "prompt_injection_role_impersonation"
                if role_injection else "prompt_injection_direct"
            ])

        from agent.financial_request_boundaries import (
            eps_cumulative_subtraction, explicit_krw_fx_fallback,
        )
        if eps_cumulative_subtraction(question):
            return _terminal(source_intent, ["unsupported_request"])
        if explicit_krw_fx_fallback(question):
            # The current financial authority cannot bind a disclosed FX
            # rate and its date. Do not pretend that no such rate exists in
            # the corpus: only the explicitly allowed KRW fact may execute.
            return _terminal(source_intent, ["unsupported_semantic_target"])

        if re.search(r"외부\s*url|https?://|url\s*(?:을|로|에)", raw):
            return _terminal(source_intent, ["external_url_request"])
        asks_external_tool = bool(
            re.search(r"(?:opendart|dart)(?:\s*api)?|외부\s*api", raw)
            and re.search(r"접속|호출|조회|가져|받아|추가로|확인", raw)
        )
        future_cutoff = any(
            f"{int(year):04d}{int(month):02d}{int(day):02d}" > self.corpus_cutoff
            for year, month, day in re.findall(
                r"(?<![0-9])(20[0-9]{2})\s*년\s*"
                r"(1[0-2]|0?[1-9])\s*월(?:\s*(3[01]|[12][0-9]|0?[1-9])\s*일)?",
                raw)
            if day
        )
        # A year/month range beginning after the corpus cutoff is out of
        # scope even when no day was stated.
        future_cutoff = future_cutoff or any(
            f"{int(year):04d}{int(month):02d}01" > self.corpus_cutoff
            for year, month in re.findall(
                r"(?<![0-9])(20[0-9]{2})\s*년\s*(1[0-2]|0?[1-9])\s*월",
                raw)
        )
        if asks_external_tool:
            reasons = ["external_tool_request"]
            if future_cutoff:
                reasons.append("unsupported_temporal_scope")
            return _terminal(source_intent, reasons)
        if (future_cutoff
                and re.search(r"(?:까지|기준)(?:\s*으로|\s*일)?", raw)
                and re.search(r"제출|공시|보고서|코퍼스", raw)):
            return _terminal(source_intent, ["unsupported_temporal_scope"])

        from src.canonical.security import classify_privacy_request

        privacy_request = classify_privacy_request(raw)
        # 공개 지분 사실과 개인정보를 함께 묻는 요청은 전부 거절하지 않는다.
        # Holding backend가 공개 slot만 조회하고 실행 sidecar가
        # personal_data_omitted를 붙인다. 개인정보만 요구하면 기존처럼 terminal.
        if privacy_request.mode == "restricted_only":
            return _terminal(source_intent, ["personal_data_request"])

        # 이슈 #38 이전에는 "부채비율을 계산하고, 안 되면 부채총계·자본총계만"
        # 같은 질문을 여기서 미리 unsupported_semantic_target 으로 닫아
        # financial_backend 가 두 원값만 대신 내게 했다. 부채비율은 이제
        # `agent.stage1_v1_financial_backend.NAMED_RATIO_CONCEPTS` 사전에
        # 있는 이름 있는 비율이라 concept_ratio 로 실제로 계산된다 —
        # financial_backend(이 백엔드보다 뒤에 온다)가 먼저 볼 기회를 얻도록
        # 여기서 더는 가로채지 않는다. 사전에 없는 비율의 "계산이 안 되면 …"
        # 우회 요청은 financial_backend 가 스스로 계산을 못 찾으면 여전히
        # `server/stage1.py` 의 unsupported_operator 사이드카 경로로 두
        # 원값만 남긴다.
        explicit_volume_fallback = bool(
            re.search(r"판매\s*량|몇\s*대\s*팔", raw)
            and re.search(r"없으면|안\s*되면|대신", raw)
            and re.search(r"매출\s*액", raw))
        if explicit_volume_fallback:
            return _terminal(source_intent, ["unsupported_semantic_target"])

        intraday_request = bool(
            re.search(r"접수\s*시각|접수시각", raw)
            and re.search(r"(?:분|초)\s*단위", raw))
        canonical_version_order_request = bool(
            intraday_request
            and re.search(r"보고서", raw)
            and re.search(r"정정", raw)
            and re.search(r"먼저|선후|순서", raw))
        if intraday_request and not canonical_version_order_request:
            terminal = _terminal(source_intent, ["unsupported_temporal_scope"])
            terminal["reasons"][0]["diagnostic_code"] = (
                "intraday_order_unavailable")
            return terminal

        future_year = any(
            int(year) > int(self.corpus_cutoff[:4])
            for year in re.findall(r"(?<![0-9])([0-9]{4})\s*년?", raw)
        )
        # 「예측해줘」류 명시 어휘뿐 아니라 「얼마나 벌 것 같아?」처럼 구어로만
        # 미래 실적을 묻는 문형도 같은 정책 거절이어야 한다(issue #172 M25).
        # 명시 미래 연도 또는 미래 시간 부사가 있어야만 열어 과거·현재 시제의
        # 추측성 일반 서술은 걸리지 않게 한다.
        future_language = bool(re.search(
            r"예측|예상|전망|대충\s*찍|얼마일지|목표\s*주가|"
            r"벌\s*것\s*같|얼마나\s*벌(?:까|지|것)|벌어들일지|"
            r"앞으로.{0,12}(?:벌|팔|늘|줄)", raw))
        future = future_language and (
            future_year or bool(re.search(r"앞으로|향후|내년|다음", raw)))
        # 「매수하기 좋은 종목은?」「투자하기 더 나은 곳은?」도 「추천해줘」와
        # 같은 투자의견 요청이다(issue #172 M26). 「투자계획」·「투자 활동」·
        # 「타법인 출자」같은 공시 어휘는 "투자" 뒤에 "하기"나 "나은"이 곧장
        # 붙지 않으므로 오탐하지 않는다(tests/test_stage1_v1_policy_m26_
        # advice_wording.py 회귀 고정).
        advice = bool(re.search(
            r"매수\s*의견|매도\s*의견|목표\s*주가|투자\s*조언|"
            r"(?:매수|투자)\s*하기\s*(?:좋은|더\s*나은)|"
            r"어느\s*주식을\s*사야|어느\s*종목을\s*사야|"
            r"(?:주식|종목).{0,12}추천|추천.{0,12}(?:주식|종목)|"
            r"매수하기\s*좋은|투자하기\s*(?:더\s*)?나은|"
            r"나은\s*(?:종목|주식|곳)|유망\s*(?:종목|주식)", raw))
        has_actual_fallback = bool(
            future
            and re.search(r"(?:안\s*되면|대신|불가능.{0,12}(?:실제|과거))", raw)
            and re.search(r"최신\s*연간\s*실제|최신\s*실제|과거\s*실적", raw)
        )
        if (future and not has_actual_fallback) or advice:
            reasons: list[str] = []
            if future and not has_actual_fallback:
                reasons.append("future_forecast")
            if advice:
                reasons.append("investment_advice")
            if pressure:
                reasons.append("pressure_resisted")
            return _terminal(source_intent, reasons)

        company = self._explicit_financial_company(question, source_intent)
        if company is not None and self._company_exists(company) is False:
            reasons = ["corp_not_in_universe"]
            if pressure:
                reasons.append("pressure_resisted")
            return _terminal(source_intent, reasons)

        # 욕설·모욕만 있는 입력, 그리고 공시·기업과 무관한 요청(날씨·요리·
        # 일반 코딩 요청 등)을 마지막에 걸러낸다. 이 검사가 맨 끝에 있는 것도
        # 같은 이유다 — 앞의 모든 구체적 정책 판정이 먼저 기회를 가진 뒤에만,
        # 정말 아무 공시 신호도 없을 때 닫는다. 판정 규칙 자체는
        # `classify_abusive_or_off_topic`에 있다 — server/stage1.py의 HCX
        # 호출 전 게이트와 같은 함수를 쓴다(중복 구현 금지).
        gate_code = classify_abusive_or_off_topic(
            question, company_resolver=self.company_resolver,
            entities=source_intent.entities)
        if gate_code is not None:
            return _terminal(source_intent, [gate_code])
        return None


class CausalPolicyResolutionBackend:
    """Reject unsupported share-price causality without inventing retrieval."""

    def resolve(
            self,
            *,
            question_id: str,
            question: str,
            source_intent: SemanticIntent,
            ) -> dict[str, Any] | None:
        del question_id
        # 인과 premise 가 있다고 전부 거절하지 않는다. 이 정책은 주가 인과에 한한다 — 주가를 언급하는
        # 인과 premise 만 「검증 불가 인과 주장」으로 보고, 그 밖의 경우는 질문 문장 규칙에 맡긴다.
        has_share_price_causal_premise = any(
            premise.kind == "causal" and _SHARE_PRICE.search(premise.raw_text) is not None
            for premise in source_intent.premises)
        reasons = causal_policy_reasons(
            question,
            has_unverified_causal_claim=has_share_price_causal_premise,
        )
        if not reasons:
            return None
        item_ids = [item.item_id for item in source_intent.answer_items]
        return {
            "kind": "terminal",
            "reasons": [
                {"code": code, "scope": "question", "item_ids": item_ids}
                for code in reasons
            ],
        }


__all__ = [
    "CausalPolicyResolutionBackend", "GeneralPolicyResolutionBackend",
    "classify_abusive_or_off_topic",
]

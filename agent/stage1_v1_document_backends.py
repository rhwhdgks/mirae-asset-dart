"""정본 해석기를 v1 타입 권위로 옮기는 어댑터 백엔드들.

`event_preflight` · `periodic_document_preflight` 는 이미 코퍼스에서 좌표를
뽑는다.  없던 것은 그 결과를 v1 `ResolvedAuthority` payload 로 옮기는 층이다.
해석을 두 번 구현하지 않는다 — 여기서 하는 일은 형 변환과 증거 참조 부여뿐이다.

모든 백엔드가 같은 계약을 지킨다. **확정하지 못하면 `None` 이다.**  물러난 자리는
호출자가 다른 백엔드나 종전 경로로 보낸다.  지어내면 거절해야 할 질문에 답이 생긴다.
"""

from __future__ import annotations

import calendar
from datetime import date
from pathlib import Path
import re
from typing import Any, Mapping

from agent.date_surface import question_date_surfaces
from agent.exact_periodic_filing import resolve_exact_periodic_filing
from agent.planning import _event_timepoints, _target_date_range
from agent.semantic_intent_v1 import SemanticIntent, semantic_intent_digest
from agent.stage1_v1_resolver import build_resolution_premise_proofs
from agent.stage1_v1_section_slot_proofs import (
    investment_plan_section_slot_proofs,
)


_INVESTMENT_HEADER_SLOT = {
    "투자명": "투자대상",
    "투자대상": "투자대상",
    "투자목적": "목적",
    "목적": "목적",
    "총소요자금": "금액",
    "기간": "기간",
}
_INVESTMENT_SLOT_ORDER = ("투자대상", "목적", "금액", "기간")


def _section_key(value: str) -> str:
    """Normalize one DART section-path component without its ordinal."""

    title = re.sub(
        r"^\s*(?:(?:[0-9]+|[IVXLCDM]+|[A-Za-z가-힣])\s*[.)])\s*",
        "", value or "", flags=re.IGNORECASE)
    return _header_key(title)


#: 날짜 뒤에 붙어 시점을 한정하는 말.  날짜 자체의 일부가 아니다.
_DATE_TRAILERS = (
    "까지의", "까지", "부터", "이전", "이후", "현재", "기준", "시점", "말",
)


def _names_a_date(surface: str) -> bool:
    """표현이 **시점**을 가리키는가.

    시점은 절 이름이 될 수 없다.  그런데 조각 매칭에 들어가면 「2026년 6월
    19일까지」가 `19`·`20`·`26` 같은 조각을 만들고, 이 조각들은 「19. 이해관계자와의
    거래」처럼 번호가 붙은 절 경로에 우연히 걸려 점수를 흐린다.  문서 이름을
    빼는 것과 같은 이유로 뺀다.

    판정은 기존 날짜 파서에 맡기고, 여기서는 뒤에 붙은 한정어만 뗀다.
    """

    from agent.date_surface import parse_date_surface

    value = (surface or "").strip()
    if not value:
        return False
    for _ in range(len(_DATE_TRAILERS)):
        for trailer in _DATE_TRAILERS:
            if value.endswith(trailer) and len(value) > len(trailer):
                value = value[: -len(trailer)].strip()
                break
        else:
            break
    return parse_date_surface(value) is not None


#: 사건 하나를 가리키는 대상의 `kind`.
#
# `kind` 는 모델이 흔들리는 축이다. 같은 사건을 `document` 또는 `event` 로
# 적는 경우는 뒤의 정본 유일성 판정으로 받을 수 있다.
#
# 반면 `entity` 는 회사·임원·상대방 등 일반 개체도 뜻한다. 정본에서 사건 하나가
# 유일하다는 사실만으로 그 일반 개체가 사건이었다고 증명할 수 없으므로 받지 않는다.
_EVENT_TARGET_KINDS = frozenset({"event", "document"})


#: 「어떤/무슨 사업을 하고 있다고 밝혔는가」류 서술절. 절 이름이 아니라 「사업의
#: 내용」절 본문을 묻는 질문이다 — 「사업」·「영위」와 「밝혔다/설명했다/기재했다/
#: 공시했다」류 서술이 한 표현 안에 같이 있다는 형으로 잡는다. HCX 가 이 서술을
#: `target.surface`·`field_surfaces` 어느 한쪽에 담을 수도, whole_target 으로
#: 눌러 field 를 아예 비우고 원문에만 남길 수도 있어(K-047) 두 자리 모두에서
#: 이 패턴 하나를 재사용한다.
_BUSINESS_NARRATIVE_CLAUSE = re.compile(
    r"(?:어떤|무슨)\s*사업.{0,20}(?:하고\s*있|영위하).{0,8}"
    r"(?:밝혔|설명했|기재했|공시했)")


def _business_section_leaf(surface: str) -> "str | None":
    """Map semantic business-content axes to standard DART section leaves.

    These are form headings, not question/fixture aliases.  A mapped heading
    is still accepted only when one proof-bearing coordinate exists in the
    selected filing.
    """

    key = _header_key(surface)
    # 「주요 사업 내용」은 서식 제목 「사업의 내용」을 평범한 말로 부른 것이다.
    # 「주요 사업」도 같은 절을 부른다 — 끝의 「내용」이 있고 없고로 갈리면
    # 같은 뜻의 질문이 말끝 하나로 다른 길을 탄다. 필드가 「사업」한 글자만
    # 온 경우도 같은 절을 부르는 것으로 받는다 — HCX 가 서술절을 「사업」
    # 하나로 뭉뚱그릴 수 있다.
    if key in {"사업내용", "사업의내용", "주요사업내용", "주요사업", "사업"}:
        return "사업의내용"
    if _BUSINESS_NARRATIVE_CLAUSE.search(surface or ""):
        return "사업의내용"
    if "사업부문" in key:
        return "사업의개요"
    if "주요제품" in key and "서비스" in key:
        return "주요제품및서비스"
    if "매출" in key and any(token in key for token in ("구성", "현황", "내역")):
        return "매출및수주상황"
    return None


def _header_key(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", value or "")


def _canonical_section_query(title: str) -> str:
    """Keep the source heading while dropping only an ordinal prefix."""
    return re.sub(
        r"^\s*(?:[0-9]+|[IVXLCDM]+|[A-Za-z가-힣])\s*[.)]\s*",
        "", title, flags=re.IGNORECASE).strip()


def _investment_plan_header_proofs(
        canonical: Any, *, document_id: str, as_of: str,
        ) -> "list[str] | None":
    """Prove four investment slots from one source table's header inventory.

    ``기 지출금액`` is intentionally absent: it describes execution-to-date,
    not the plan's total planned amount.  Missing or duplicate canonical slot
    headers fail closed instead of guessing from a body row.
    """

    if not callable(getattr(canonical, "table_headers", None)):
        return None
    rows = list(canonical.table_headers(as_of=as_of, doc_id=document_id))
    by_table: dict[tuple[str, str], dict[str, list[Any]]] = {}
    for row in rows:
        slot = _INVESTMENT_HEADER_SLOT.get(
            _header_key(str(row.header_path).split(">")[-1]))
        if slot is None:
            continue
        by_table.setdefault(
            (str(row.source_file_id), str(row.table_locator)), {}
        ).setdefault(slot, []).append(row)
    candidates = []
    for rows_by_slot in by_table.values():
        if any(len(rows_by_slot.get(slot, ())) != 1
               for slot in _INVESTMENT_SLOT_ORDER):
            continue
        candidates.append(rows_by_slot)
    if len(candidates) != 1:
        return None
    selected = candidates[0]
    return [
        "source-header:"
        f"{row.doc_id}:{row.source_file_id}:{row.table_locator}:{row.locator}"
        for slot in _INVESTMENT_SLOT_ORDER
        for row in selected[slot]
    ]


def _embedded_investment_coordinate(chunks: "list[Any]") -> "Any | None":
    """Find one structural equipment-investment subsection inside a chunk.

    Some DART sources keep ``설비 등 투자현황`` inside a larger canonical
    section instead of emitting a standalone Section row.  The heading alone
    is not enough: the same chunk must also carry the stable investment table
    headers.  Multiple candidates fail closed.
    """

    candidates: dict[tuple[Any, ...], Any] = {}
    for row in chunks:
        text = str(getattr(row, "text", "") or "")
        compact = _header_key(text)
        # 제목은 서식마다 다르다 — 「설비 등 투자현황」·「설비 투자 현황 및 계획」·
        # 그냥 「(2) 투자현황」이 모두 같은 표를 연다.  제목으로 문을 잠그면 코퍼스에서
        # 한화에어로스페이스 한 곳만 열린다(실측: 1개사 12문서).
        #
        # 느슨해지지 않는다.  **표 헤더 4종을 모두 요구하는 계약은 그대로**이고,
        # 그 조합은 DART 투자현황 표 서식 자체다.  완화 후에도 6개사 80문서에 그치고
        # 문서당 후보가 둘 이상인 경우는 0건이라 유일 확정이 깨지지 않는다.
        has_heading = re.search(
            r"(?:설비\s*등?\s*)?투자\s*현황", text) is not None
        has_table_contract = all(
            token in compact
            for token in ("사업부", "투자기간", "대상자산", "투자액"))
        coordinate = tuple(getattr(row, key, None) for key in (
            "source_file_id", "path", "locator", "evidence_id"))
        if has_heading and has_table_contract and all(coordinate):
            candidates[coordinate] = row
    return next(iter(candidates.values())) if len(candidates) == 1 else None


def _embedded_investment_retrieval_query(row: Any) -> "str | None":
    """Return the proven investment-table heading, not the report title."""

    text = str(getattr(row, "text", "") or "")
    match = re.search(
        r"(?:설비\s*등?\s*)?투자\s*현황(?:\s*및\s*계획)?", text)
    return match.group(0).strip() if match is not None else None


def _authority(
        *, question_id: str, source_intent: SemanticIntent,
        canonical_build_id: str, resolver_version: str,
        reference_date: date, corpus_cutoff: str,
        items: list[dict[str, Any]],
        premise_proof_refs: Mapping[str, list[str]] | None = None,
        ) -> "dict[str, Any] | None":
    from agent.deterministic_plan_compiler_v1 import AuthoritativeResolution

    if not items:
        return None
    resolution = AuthoritativeResolution.create(
        question_id=question_id,
        source_intent_digest=semantic_intent_digest(source_intent),
        canonical_build_id=canonical_build_id,
        resolver_version=resolver_version,
        reference_date=reference_date,
        corpus_cutoff=corpus_cutoff,
        items=items,
        premise_proofs=build_resolution_premise_proofs(
            source_intent, items, overrides=premise_proof_refs),
    )
    return {"kind": "resolved", "resolution": resolution.model_dump(mode="json")}


def _field_proofs(item_id: str, surfaces: "list[str]") -> "list[dict[str, Any]]":
    return [
        {
            "proof_ref": f"source-field:{item_id}:{position}",
            "source_field_index": position,
            "surface": surface,
        }
        for position, surface in enumerate(surfaces)
    ]


def _single_company(
        intent: SemanticIntent, item: Any, companies: Any) -> "Any | None":
    """항목이 가리키는 회사가 정확히 하나로 확정될 때만 돌려준다."""

    by_id = {entity.entity_id: entity for entity in intent.entities}
    # In an event read with exactly two entities, an event target can point at
    # an external company while the unreferenced company is the issuer.  Do
    # this structural role check before ordinary target-reference lookup.
    if (item.target.kind in _EVENT_TARGET_KINDS and len(intent.entities) == 2
            and len(item.target.entity_refs) == 1
            and by_id.get(item.target.entity_refs[0], None) is not None
            and by_id[item.target.entity_refs[0]].kind_hint == "event"):
        unreferenced = [
            entity.surface for entity in intent.entities
            if entity.entity_id not in item.target.entity_refs
            and entity.kind_hint == "company"
        ]
        issuer_rows = [
            rows[0] for rows in (companies.resolve_company(surface)
                                  for surface in unreferenced)
            if len(rows) == 1
        ]
        issuer_unique = {company.corp_code: company for company in issuer_rows}
        if len(issuer_unique) == 1:
            return next(iter(issuer_unique.values()))
    surfaces = [
        by_id[ref].surface for ref in item.target.entity_refs
        if ref in by_id and by_id[ref].kind_hint == "company"
    ]
    # 계약 상대도 `company` 로 표시돼 함께 딸려 온다.  이름 개수로 자르면
    # 「LG엔솔 Ford 계약」처럼 상대가 있는 질문이 전부 물러난다.  **조회되는
    # 이름이 정확히 하나일 때** 그것을 공시 주체로 본다 — 상대는 조회되지 않는다.
    resolved = [
        rows[0] for rows in (
            companies.resolve_company(surface) for surface in surfaces)
        if len(rows) == 1
    ]
    unique = {company.corp_code: company for company in resolved}
    if len(unique) == 1:
        return next(iter(unique.values()))
    # A selected event may instead refer to an external counterparty/event,
    # leaving the issuer as the sole resolvable company elsewhere in the
    # semantic intent.  Reuse only that unique corpus-backed issuer; two
    # possible issuers remain ambiguous.
    if unique:
        return None
    all_resolved = [
        rows[0] for rows in (
            companies.resolve_company(entity.surface) for entity in intent.entities)
        if len(rows) == 1
    ]
    all_unique = {company.corp_code: company for company in all_resolved}
    return next(iter(all_unique.values())) if len(all_unique) == 1 else None


class BusinessContentNarrativeRegrounder:
    """Recover a periodic comparison topic from its question-grounded axes.

    HCX sometimes makes the annual report itself the target and leaves the
    comparison subject only in ``field_surfaces``.  QueryPlan's narrative
    search target is the contiguous axis phrase.  Copy that exact question
    span into a topic and retain the report surface as document-group scope.
    """

    def __init__(self, canonical: Any) -> None:
        self.canonical = canonical

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (not isinstance(question, str)
                or len(intent.answer_items) != 1
                or intent.answer_groups or intent.premises
                or intent.unresolved_mentions):
            return intent
        item = intent.answer_items[0]
        if (item.target.kind != "document"
                or item.operation != "compare"
                or item.output.shape not in {"comparison", "narrative"}
                or item.output.projection_mode != "named_fields"
                or len(item.output.field_surfaces) < 2
                or len(item.scope.target_period_expressions) != 2
                or item.scope.as_of_expression is not None
                or item.scope.document_group_expression is not None
                or item.scope.scope_qualifier_expressions
                or item.selection is not None
                or _header_key(item.target.surface) not in {
                    "사업보고서", "연간보고서"}
                or item.target.surface not in question
                or not all(surface in question
                           for surface in item.output.field_surfaces)):
            return intent

        by_id = {entity.entity_id: entity for entity in intent.entities}
        candidates = []
        for entity in intent.entities:
            if (entity.kind_hint != "company" or entity.surface not in question):
                continue
            rows = self.canonical.resolve_company(entity.surface)
            if len(rows) == 1:
                candidates.append((entity, rows[0]))
        unique = {row.corp_code: (entity, row) for entity, row in candidates}
        if len(unique) != 1:
            return intent
        company_entity, _ = next(iter(unique.values()))

        fields = list(item.output.field_surfaces)
        positions: list[tuple[int, int]] = []
        cursor = 0
        for surface in fields:
            start = question.find(surface, cursor)
            if start < 0:
                return intent
            positions.append((start, start + len(surface)))
            cursor = start + len(surface)
        topic = question[positions[0][0]:positions[-1][1]]
        separators = topic
        for surface in fields:
            separators = separators.replace(surface, "", 1)
        if re.sub(r"[\s·,/・•ㆍ그리고및]", "", separators):
            return intent

        payload = intent.model_dump(mode="python", warnings=False)
        target = payload["answer_items"][0]["target"]
        referenced = [
            ref for ref in target["entity_refs"]
            if ref in by_id and by_id[ref].kind_hint != "company"
        ]
        target["entity_refs"] = [company_entity.entity_id, *referenced]
        target["kind"] = "topic"
        target["surface"] = topic
        payload["answer_items"][0]["scope"][
            "document_group_expression"] = item.target.surface
        return SemanticIntent.model_validate(payload, strict=True)


class PeriodicNarrativeTopicRegrounder:
    """Separate a periodic retrieval topic from its answer projection labels."""

    _CONTAINER = re.compile(r"계획|현황|내용|사업|활동|계약")
    _EVIDENCE_LABEL = re.compile(r"근거|출처|인용")
    _SUMMARY_LABEL = re.compile(r"설명|변화|비교|요약")

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (not isinstance(question, str)
                or len(intent.answer_items) != 1
                or intent.answer_groups or intent.premises
                or intent.unresolved_mentions):
            return intent
        item = intent.answer_items[0]
        if item.target.kind != "document" or item.selection is not None:
            return intent
        payload = intent.model_dump(mode="python", warnings=False)
        row = payload["answer_items"][0]

        if item.operation == "retrieve":
            from agent.periodic_document_preflight import (
                PeriodicDocumentPreflightError,
                parse_periodic_expression,
            )
            try:
                parse_periodic_expression(item.target.surface)
            except PeriodicDocumentPreflightError:
                return intent
            fields = list(item.output.field_surfaces)
            if (len(fields) < 3 or not self._CONTAINER.search(fields[0])
                    or not all(surface in question for surface in fields)
                    or question.find(fields[0]) >= min(
                        question.find(surface) for surface in fields[1:])):
                return intent
            row["target"].update({
                "kind": "topic", "surface": fields[0],
                "qualifier_surfaces": [],
            })
            row["scope"]["document_group_expression"] = item.target.surface
            row["output"]["field_surfaces"] = fields[1:]
            return SemanticIntent.model_validate(payload, strict=True)

        fields = list(item.output.field_surfaces)
        qualifiers = list(item.target.qualifier_surfaces)
        if (item.operation != "compare"
                or len(item.scope.target_period_expressions) != 2
                or len(qualifiers) < 2 or len(fields) != 2
                or not self._EVIDENCE_LABEL.search(fields[0])
                or not self._SUMMARY_LABEL.search(fields[1])):
            return intent
        positions: list[tuple[int, int]] = []
        cursor = 0
        for surface in qualifiers:
            start = question.find(surface, cursor)
            if start < 0:
                return intent
            positions.append((start, start + len(surface)))
            cursor = start + len(surface)
        topic = question[positions[0][0]:positions[-1][1]].strip()
        if (not topic or len(topic) > 80
                or re.search(r"[,;!?？。]", topic)
                or any(field not in question for field in fields)):
            return intent
        row["target"].update({
            "kind": "topic", "surface": topic,
            "qualifier_surfaces": [],
        })
        row["scope"]["document_group_expression"] = item.target.surface
        row["output"].update({
            "shape": "narrative",
            "field_surfaces": [topic, fields[1]],
        })
        return SemanticIntent.model_validate(payload, strict=True)


class PeriodicNarrativeWholeTopicRegrounder:
    """Recover one explicit topic from one explicitly named periodic report.

    Models often retain ``2025년 사업보고서`` as the document target but
    flatten ``투자 현황을 전반적으로 정리`` into a generic projection.  The
    topic below is copied from the text between the report locator and the
    summarization verb, so the rule applies equally to governance, R&D, risk,
    or investment narratives and does not promote them to a structured table.
    """

    _TOPIC_AFTER_REPORT = re.compile(
        r"(?:사업보고서|반기보고서|분기보고서)(?:에서|에)\s*"
        r"(?:설명한|기재한|공시한|확인되는)?\s*"
        r"(?P<topic>[^,?？。]{1,60}?)"
        r"(?:을|를)?\s*(?:전반적으로\s*)?(?:정리|요약|설명)"
        r"(?:해|하여)?\s*(?:줘|주세요|줄래|주실래)?\s*[.!?？。]?$"
    )

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (not isinstance(question, str) or len(intent.answer_items) != 1
                or intent.answer_groups or intent.premises
                or intent.unresolved_mentions):
            return intent
        item = intent.answer_items[0]
        if (item.target.kind != "document" or item.operation != "retrieve"
                or item.selection is not None):
            return intent
        from agent.periodic_document_preflight import (
            PeriodicDocumentPreflightError, parse_periodic_expression,
        )
        try:
            parse_periodic_expression(item.target.surface)
        except PeriodicDocumentPreflightError:
            return intent
        match = self._TOPIC_AFTER_REPORT.search(question)
        if match is None:
            return intent
        topic = match.group("topic").strip()
        topic = re.sub(r"(?:을|를)$", "", topic).strip()
        if not topic or topic not in question:
            return intent
        payload = intent.model_dump(mode="python", warnings=False)
        row = payload["answer_items"][0]
        row["target"].update({
            "kind": "topic", "surface": topic,
            "qualifier_surfaces": [],
        })
        row["scope"]["document_group_expression"] = item.target.surface
        row["output"].update({
            "shape": "narrative", "projection_mode": "named_fields",
            "field_surfaces": [topic],
        })
        return SemanticIntent.model_validate(payload, strict=True)


class BusinessNarrativeClauseFieldRegrounder:
    """Give a bare 「어떤/무슨 사업을 …밝혔는가」 whole-target item a real field.

    HCX can flatten this descriptive clause entirely into ``projection_mode
    = "whole_target"`` with **no** field surface at all (K-047 실측) — the
    clause survives only in the raw question, nowhere in the intent
    structure.  A field-less item cannot reach
    `PeriodicNarrativeResolutionBackend` (it requires >=1 field, and the
    compiler's own ``periodic_document_narrative`` topology guard requires
    ``named_fields``), so it falls to `DocumentCollectionResolutionBackend`
    as a document-kind whole-target request. That backend only derives a
    ``retrieval_query`` for **topic**-kind items, so a document-kind item
    compiles to a bare "find" task with no retrieval_query — the right
    receipt is cited, but no content is ever fetched for it, and the answer
    closes with a bare "확인된 바 없습니다" (관측: 코드 요청 시점 로컬 서버,
    현대제철 K-047).

    Its almost-synonymous sibling ("…의 가장 최근 정기보고서에서 주요 사업
    내용을 정리해줘") already reaches HCX as ``target.kind=document``,
    ``projection_mode=named_fields``, ``field_surfaces=["주요 사업
    내용"]`` — and that shape already resolves correctly end-to-end. Rather
    than teach the document-collection path a second, unpinned way to fetch
    business content (a plain FTS search with no document selector can rank
    an older filing above the intended latest one — confirmed against
    out/canonical), copy that already-working shape: keep ``target`` as-is
    (the periodic reference resolves the receipt exactly as it does for the
    named-fields sibling) and only add the missing field.

    The field surface must still be a literal span of ``question``
    (`validate_semantic_intent_grounding` fail-closes on an invented one),
    so this cannot spell out "주요 사업 내용" the way the named-fields
    sibling does — only "사업" itself is guaranteed present, by construction
    of the clause regex this rule matches on. That bare span is enough:
    `_business_section_leaf` already maps a lone "사업" field to the same
    "사업의내용" section leaf as "주요 사업 내용" does.
    """

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (not isinstance(question, str)
                or len(intent.answer_items) != 1
                or intent.answer_groups or intent.premises
                or intent.unresolved_mentions):
            return intent
        item = intent.answer_items[0]
        if (item.target.kind != "document"
                or item.operation != "retrieve"
                or item.output.shape != "narrative"
                or item.output.projection_mode != "whole_target"
                or item.output.field_surfaces
                or item.selection is not None
                or not _BUSINESS_NARRATIVE_CLAUSE.search(question)):
            return intent
        from agent.periodic_document_preflight import (
            PeriodicDocumentPreflightError, parse_periodic_expression,
        )
        try:
            parse_periodic_expression(item.target.surface)
        except PeriodicDocumentPreflightError:
            return intent
        payload = intent.model_dump(mode="python", warnings=False)
        payload["answer_items"][0]["output"].update({
            "projection_mode": "named_fields",
            "field_surfaces": ["사업"],
        })
        return SemanticIntent.model_validate(payload, strict=True)


class PeriodicNarrativeResolutionBackend:
    """정기보고서 한 건에서 서술 슬롯을 읽는 항목을 확정한다.

    `PeriodicDocumentPreflight` 가 이미 「어느 접수번호인가」를 정한다.  여기서는
    그 결과에 문서·서술 증거 참조를 붙이고 요청 필드를 실행 가능 색인으로 옮긴다.
    """

    def __init__(
            self, canonical: Any, *, canonical_build_id: str,
            resolver_version: str, reference_date: date, corpus_cutoff: str,
            ) -> None:
        from agent.periodic_document_preflight import PeriodicDocumentPreflight

        self.canonical = canonical
        self.preflight = PeriodicDocumentPreflight(canonical)
        self.canonical_build_id = canonical_build_id
        self.resolver_version = resolver_version
        self.reference_date = reference_date
        self.corpus_cutoff = corpus_cutoff
        self._document_chunk_cache: dict[tuple[str, str], list[Any]] = {}

    def _document_chunks(self, corp_code: str, receipt_no: str) -> list[Any]:
        """Read one filing's chunks once per backend instance."""

        key = (corp_code, receipt_no)
        cached = self._document_chunk_cache.get(key)
        if cached is None:
            cached = [
                row for row in self.canonical.chunks(
                    corp_code=corp_code, doc_group="periodic",
                    doc_id=f"periodic_{receipt_no}")
                if str(getattr(row, "doc_id", "")).endswith(receipt_no)
            ]
            self._document_chunk_cache[key] = cached
        return cached

    def _periodic_expressions(self, item: Any) -> "list[str]":
        """정기보고서를 가리킬 수 있는 표현을 의미에서 모은다.

        **모양이 아니라 확정 가능성으로 판정한다.**  `target.kind` 가 `topic`
        이냐 `document` 냐는 모델이 흔들리는 축이라, 그것으로 문을 잠그면 같은
        질문이 판마다 다르게 처리된다.  대신 표현을 후보로 모아 정본 preflight
        에 넘기고, **코퍼스에서 정기보고서 하나가 유일하게 확정될 때만** 받는다.
        확정되지 않으면 물러나므로 느슨해지지 않는다.
        """

        candidates = [item.scope.document_group_expression]
        candidates.extend(item.scope.target_period_expressions)
        candidates.extend(item.target.qualifier_surfaces)
        # 보고서가 scope 가 아니라 target 으로 오는 판이 있다.  「…1분기보고서에서
        # 확인되는 주요 투자계획」을 target.surface="2026년 1분기보고서" 로 읽는
        # 식이다.  자리를 틀렸을 뿐 표현은 그대로이므로 후보로 함께 넘긴다.
        # 정기보고서 표현이 아니면 preflight 가 걸러내므로 넓어지지 않는다.
        candidates.append(item.target.surface)
        expressions = [value for value in candidates if value and value.strip()]
        # A calendar year plus a narrative topic has the same established
        # annual-report default as a narrative matrix.  The synthesized form
        # is only a private canonical lookup coordinate; public intent fields
        # remain literal question spans.
        if (item.target.kind == "topic"
                and not any(re.search(r"분기|반기", value)
                            for value in expressions)):
            expressions.extend(
                f"{value.strip()} 사업보고서" for value in expressions
                if re.fullmatch(r"(?:19|20)[0-9]{2}년", value.strip()))
        return list(dict.fromkeys(expressions))

    def _resolved_document(self, corp_code: str, item: Any) -> "Any | None":
        """유일하게 확정되는 정기보고서 하나. 없거나 여럿이면 ``None``."""

        found = []
        for expression in self._periodic_expressions(item):
            resolution = self.preflight.resolve_periodic_document(
                corp_code=corp_code, as_of=self.corpus_cutoff,
                target_expression=expression)
            if resolution.status == "resolved" and resolution.candidate is not None:
                found.append(resolution.candidate)
        unique = {row.rcept_no: row for row in found}
        if len(unique) == 1:
            return next(iter(unique.values()))
        if unique:
            return None
        if self._is_latest_investment(item):
            return self._latest_periodic_document(corp_code)
        if not self._is_whole_business_content(item):
            return None
        return self._latest_annual_document(corp_code)

    @staticmethod
    def _is_latest_investment(item: Any) -> bool:
        surfaces = " ".join((
            item.target.surface,
            *item.target.qualifier_surfaces,
            *item.output.field_surfaces,
        ))
        if _INVESTMENT_JUDGMENT_FORM_PATTERN.search(surfaces) is not None:
            # 「투자판단관련주요경영사항」은 거래소 공시 서식명이지 서술 절
            # 「최근 투자계획」이 아니다. 「투자」부분 문자열만 보면 이 리터럴
            # 서식명이 매번 걸려, 날짜가 특정된 그 서식 하나를 물어도 최신
            # 사업보고서로 빠진다(CG-051 실측, 이슈 #150). 리터럴 서식명이
            # 있으면 여기서 물러나 사건 경로(#44/#49 형제)가 잡게 한다.
            return False
        return (
            item.operation == "retrieve"
            and "투자" in _header_key(surfaces)
            and not item.scope.target_period_expressions
            and item.scope.document_group_expression is None
            and item.scope.as_of_expression is None
        )

    def _latest_periodic_document(self, corp_code: str) -> "Any | None":
        """Latest visible periodic filing; ties are resolved by receipt id."""

        rows = list(self.canonical.documents(
            as_of=self.corpus_cutoff, corp_code=corp_code,
            doc_group="periodic"))
        if not rows:
            return None
        latest_receipt = max(str(row.rcept_no) for row in rows)
        selected = [row for row in rows if str(row.rcept_no) == latest_receipt]
        return selected[0] if len(selected) == 1 else None

    @staticmethod
    def _is_whole_business_content(item: Any) -> bool:
        # 필드가 대상 표면을 글자 그대로 되풀이하지 않아도 된다. HCX 가 서술절
        # 대상에 「사업」한 낱말만 필드로 줄 수 있다 — 필드가 정확히 하나이고
        # 그 필드 스스로도 「사업의 내용」절을 가리키면 같은 요청으로 받는다.
        # 필드가 둘 이상이면 종전대로 글자 일치만 받는다(넓어지지 않는다).
        field_surfaces = list(item.output.field_surfaces)
        fields_bind_business_content = (
            field_surfaces == [item.target.surface]
            or (len(field_surfaces) == 1
                and _business_section_leaf(field_surfaces[0]) == "사업의내용"))
        return (
            item.operation == "retrieve"
            and item.target.kind in {"topic", "document"}
            and _business_section_leaf(item.target.surface) == "사업의내용"
            and item.output.projection_mode == "named_fields"
            and fields_bind_business_content
            and not item.scope.target_period_expressions
            and item.scope.as_of_expression is None
            and item.scope.document_group_expression is None
            and not item.scope.scope_qualifier_expressions
            and item.selection is None
        )

    def _latest_annual_document(self, corp_code: str) -> "Any | None":
        """Latest annual filing visible at the corpus cutoff, fail closed."""

        rows = [
            row for row in self.canonical.documents(
                as_of=self.corpus_cutoff, corp_code=corp_code,
                doc_group="periodic")
            if getattr(row, "form", None) == "annual"
            and isinstance(getattr(row, "base_year", None), int)
        ]
        if not rows:
            return None
        latest_year = max(row.base_year for row in rows)
        latest_year_rows = [row for row in rows if row.base_year == latest_year]
        # Corrections/version rows are ordered by their canonical receipt date;
        # the most recent visible version is the only document we can expose.
        latest_receipt = max(str(row.rcept_no) for row in latest_year_rows)
        selected = [
            row for row in latest_year_rows
            if str(row.rcept_no) == latest_receipt
        ]
        return selected[0] if len(selected) == 1 else None

    def _section_shingles(self, item: Any) -> "frozenset[str]":
        """의미 표면을 두 글자 조각으로 쪼갠다.

        「주요 투자계획」과 「설비 투자 현황 및 계획」은 낱말로는 안 겹치지만
        조각으로는 `투자`·`계획` 둘이 겹친다.  형태소 분석기 없이 공시 서식의
        절 제목을 맞추기에 이 정도면 충분하고, 언어별 규칙을 만들지 않는다.
        """

        # 문서를 가리키는 표현은 **절 이름이 아니다.**  「최근 사업보고서」의
        # `보고`·`고서` 는 그 문서의 모든 경로에 들어 있어 점수를 흐린다.
        # 절을 가리키는 것은 남은 표면뿐이다.
        from agent.periodic_document_preflight import (
            PeriodicDocumentPreflightError, parse_periodic_expression)

        def _names_the_document(surface: str) -> bool:
            try:
                parse_periodic_expression(surface)
            except PeriodicDocumentPreflightError:
                return False
            return True

        pieces: set[str] = set()
        surfaces = [surface for surface in (
            item.target.surface, *item.target.qualifier_surfaces,
            *item.output.field_surfaces)
            if surface
            and not _names_the_document(surface)
            and not _names_a_date(surface)]
        for surface in surfaces:
            compact = _re.sub(r"\s+", "", surface or "")
            for start in range(len(compact) - 1):
                pieces.add(compact[start:start + 2])
        return frozenset(pieces)

    def _sections(self, receipt_no: str) -> "list[tuple[str, str]]":
        """(제목, 경로) 목록. 제목은 `sections.parquet` 의 `title` 컬럼이다.

        `chunks.parquet` 의 `path` 는 제목과 본문이 이어질 수 있다. `sections` 는
        절 구조를 보존하므로, 같은 문서의 alternate source에 깨끗한 제목이 있으면
        아래의 경로 정합 규칙으로 그 제목을 판별할 수 있다. alternate의 **제목
        메타데이터**를 쓰는 것이지, 검색·답변 근거를 alternate로 바꾸는 것은 아니다.

        `CanonicalReadModel` 에 제목을 검색해 주는 API 가 없어 여기서 직접 읽는다.
        """

        from pathlib import Path as _Path

        import pyarrow.parquet as _pq

        path = _Path(getattr(self.canonical, "root", "out/canonical"))
        doc_id = f"periodic_{receipt_no}"
        table = _pq.read_table(
            path / "sections.parquet", columns=["doc_id", "title", "path"],
            filters=[("doc_id", "=", doc_id)])
        rows = table.to_pylist()
        return [
            (row["title"], row["path"] or "")
            for row in rows if (row.get("title") or "").strip()
        ]

    @staticmethod
    def _path_parts(path: str) -> "tuple[str, ...]":
        """절 경로를 비교용 계층 조각으로 정규화한다.

        공시 원문은 source에 따라 공백 수가 다를 수 있으므로 공백만 접는다. 제목
        자체를 형태소나 문장부호 규칙으로 자르지 않는다.
        """

        return tuple(
            _re.sub(r"\s+", " ", part).strip()
            for part in (path or "").split(">")
            if part.strip()
        )

    @classmethod
    def _reconciled_section_title(
            cls, chosen_path: str, sections: "list[tuple[str, str]]",
            ) -> "str | None":
        """선택된 절의 안전한 제목을 구조적으로 합의한다.

        viewer primary는 가끔 `title`의 마지막 절명 뒤에 첫 문장을 붙여 내보낸다.
        이때 같은 문서 alternate의 **동일 부모·동일 깊이** 절명이 primary 마지막
        조각의 proper prefix이면, 더 긴 제목이 아니라 그 접두 제목이 공통 절명이다.
        예를 들어 ``마. 설비 투자 현황 및 계획당분기말 ...`` 와
        ``마. 설비 투자 현황 및 계획`` 이면 후자를 쓴다.

        이 규칙은 질문 표현이나 특정 문구를 보지 않는다. 같은 계층이 아니거나
        서로 다른 후보가 남으면 확정하지 않고 ``None``으로 물러난다.
        """

        chosen_parts = cls._path_parts(chosen_path)
        if not chosen_parts:
            return None
        chosen_leaf = chosen_parts[-1]

        exact_titles: set[str] = set()
        clean_prefixes: dict[int, set[str]] = {}
        for title, section_path in sections:
            title = (title or "").strip()
            parts = cls._path_parts(section_path)
            if not title or len(parts) != len(chosen_parts):
                continue
            if parts[:-1] != chosen_parts[:-1]:
                continue
            leaf = parts[-1]
            if leaf == chosen_leaf:
                exact_titles.add(title)
            elif chosen_leaf.startswith(leaf):
                clean_prefixes.setdefault(len(leaf), set()).add(title)

        # 가장 긴 proper prefix만 공통 절명 후보가 될 수 있다. 여럿이면 source 간
        # 제목 합의가 없으므로 임의로 고르지 않는다.
        if clean_prefixes:
            titles = clean_prefixes[max(clean_prefixes)]
            return next(iter(titles)) if len(titles) == 1 else None
        return next(iter(exact_titles)) if len(exact_titles) == 1 else None

    def _section_heading(self, corp_code: str, receipt_no: str,
                         item: Any) -> "str | None":
        """질문이 가리키는 절의 **공시서식 제목**. 못 좁히면 ``None``.

        두 산출물을 각자 잘하는 데 쓴다.

        - 절 **선택**은 `chunks` 의 `path` 로 한다. 조각 겹침의 유일 최대를 고른다.
        - 절 **제목**은 `sections` 의 구조 메타데이터에서 합의한다. primary title에
          본문이 이어진 경우에는 동일 계층 alternate의 prefix 제목만 보정에 쓴다.

        둘을 잇는 것은 경로 접두 관계다 — 고른 조각의 경로는 그 절 경로로 시작한다.
        """

        # 절 이름은 대상에도, 뽑을 필드에도 올 수 있다.  「최근 정기보고서에서
        # 주요 사업 내용」처럼 대상이 **문서**를 가리키면 절을 부르는 것은
        # 필드 쪽이다.  대상을 먼저 보고, 없으면 필드에서 찾는다.
        standard_leaf = next(
            (leaf for leaf in (
                _business_section_leaf(surface)
                for surface in (item.target.surface,
                                *item.output.field_surfaces))
             if leaf is not None),
            None,
        )
        if standard_leaf == "사업의내용":
            headings = {
                title.strip() for title, path in self._sections(receipt_no)
                if self._path_parts(path)
                and _section_key(self._path_parts(path)[-1]) == standard_leaf
            }
            if len(headings) == 1:
                return next(iter(headings))
            return None

        shingles = self._section_shingles(item)
        if len(shingles) < 2:
            return None
        best_score = 0
        best_paths: set[str] = set()
        for row in self._document_chunks(corp_code, receipt_no):
            path = getattr(row, "path", "") or ""
            score = sum(1 for piece in shingles if piece in path)
            if score > best_score:
                best_score, best_paths = score, {path}
            elif score == best_score and score > 0:
                best_paths.add(path)
        if best_score < 2 or len(best_paths) != 1:
            return None
        chosen = next(iter(best_paths))
        return self._reconciled_section_title(
            chosen, self._sections(receipt_no))

    def _general_investment_heading(
            self, corp_code: str, receipt_no: str,
            ) -> "str | None":
        """Resolve one broad investment narrative from explicit source form.

        This route is separate from the strict four-role equipment-plan
        contract.  The source must be under ``II. 사업의 내용``, explicitly
        name an investment status/plan heading, and carry at least two stable
        plan-vs-actual table roles.  Multiple candidates fail closed.
        """

        candidates: dict[str, Any] = {}
        strong_roles = (
            "기투자액", "향후투자액", "투자기대효과", "대상자산",
            "투자목적", "투자기간", "투자액", "총소요자금",
        )
        for row in self._document_chunks(corp_code, receipt_no):
            path = str(getattr(row, "path", "") or "")
            parts = self._path_parts(path)
            if (not parts
                    or _section_key(_canonical_section_query(parts[0]))
                    != "사업의내용"):
                continue
            text = str(getattr(row, "text", "") or "")
            compact = _header_key(text)
            if (re.search(r"투자\s*(?:현황|계획)", text) is None
                    or sum(role in compact for role in strong_roles) < 2):
                continue
            candidates[path] = row
        if len(candidates) != 1:
            return None
        row = next(iter(candidates.values()))
        match = re.search(r"투자\s*(?:현황|계획)", str(
            getattr(row, "text", "") or ""))
        return match.group(0).strip() if match is not None else None

    def _disclosed_metric_topic(
            self, corp_code: str, receipt_no: str, item: Any,
            ) -> "str | None":
        """공시 서술지표 표기를 그 정기보고서 본문이 실제로 담고 있을 때만 topic 으로.

        순이자마진·수주잔고·생산능력·가동률·임상·연구개발비 같은 p9 지표는 절 제목이
        아니라 표·문장 안의 개념이라 `_section_heading`·`_literal_source_topic` 이
        모른다. 사전(`agent/disclosed_metric_topics.tsv`)의 표기가 선택된 접수본 chunk
        어딘가에 있으면 질문의 표면을 그대로 retrieval_query 로 쓴다 — 문서에 없으면
        여기서도 만들지 않는다(source-visible).
        """

        from agent.disclosed_metric_topics import (
            SEGMENT_REVENUE_TERM, key_present, match_disclosed_metric, probe_keys,
        )

        matched = match_disclosed_metric(item.target.surface)
        if matched is None:
            return None
        entry, surface = matched
        # 부문별 매출은 표기가 아니라 질문의 부문명(서치플랫폼·아시아)이 본문에 있어야 하고,
        # 검색 질의도 부문명을 담은 표면 전체(「서치플랫폼 부문 매출액」)여야 한다 — 별칭
        # 「부문 매출액」만 넘기면 부문명이 검색·발췌에서 사라진다(P9-014 실측).
        if entry.term == SEGMENT_REVENUE_TERM:
            surface = item.target.surface.strip()
        keys = [key for key in probe_keys(entry, item.target.surface) if len(key) >= 2]
        for row in self._document_chunks(corp_code, receipt_no):
            compact = _header_key(
                f"{getattr(row, 'path', '') or ''} {getattr(row, 'text', '') or ''}")
            if any(key_present(compact, key) for key in keys):
                return surface
        return None

    def _literal_source_topic(
            self, corp_code: str, receipt_no: str, item: Any,
            ) -> "str | None":
        """Retain a literal topic only when the selected filing contains it.

        Some useful axes are body concepts rather than standardized DART leaf
        headings (for example an insurer's revenue structure or a qualified
        pair of business activities).  The exact receipt remains fixed, and
        Stage2 still performs its stricter section/table authority check and
        read-section round trip.  This method merely avoids inventing a nearby
        section title when the literal question axis is source-visible.
        """

        surface = item.target.surface.strip()
        key = _header_key(surface)
        heading_key = re.sub(r"(?:현황|내용)$", "", key)
        if len(key) < 4:
            return None
        qualified = [
            _header_key(value) for value in re.split(r"[·ㆍ/]", surface)
            if _header_key(value)
            and _header_key(value) not in {"관련사업현황", "사업현황"}
        ]
        for row in self._document_chunks(corp_code, receipt_no):
            path = str(getattr(row, "path", "") or "")
            text = str(getattr(row, "text", "") or "")
            compact = _header_key(f"{path} {text}")
            if heading_key and heading_key in _header_key(path):
                heading_surface = re.sub(
                    r"(?:\s*현황|\s*내용)$", "", surface).strip()
                return heading_surface if heading_surface in surface else surface
            in_business = "사업의내용" in _section_key(path.split(">")[0])
            if not in_business:
                continue
            if (qualified and len(qualified) >= 2
                    and all(value in compact for value in qualified[:2])):
                return surface
            if ("제품" in key and "제품" in compact
                    and ("서비스" not in key or "서비스" in compact)):
                return surface
            if any(token in key for token in ("수익구조", "수익원", "매출구성")) \
                    and any(token in compact for token in (
                        "수익구조", "수익원", "영업수익", "매출액", "매출")):
                return surface
            if "사업" in key and any(token in compact for token in (
                    "사업의개요", "사업부문", "영업부문", "주요사업")):
                return surface
        return None

    @staticmethod
    def _literal_multi_topic_query(question: str, surfaces: list[str]) -> str | None:
        """Return the smallest question span containing ordered topic fields."""

        if len(surfaces) < 2 or len(set(surfaces)) != len(surfaces):
            return None
        spans: list[tuple[int, int]] = []
        cursor = 0
        for surface in surfaces:
            start = question.find(surface, cursor)
            if start < 0:
                return None
            end = start + len(surface)
            spans.append((start, end))
            cursor = end
        value = question[spans[0][0]:spans[-1][1]].strip()
        return value if value and value in question else None

    def resolve(
            self, *, question_id: str, question: str,
            source_intent: SemanticIntent,
            ) -> "dict[str, Any] | None":
        items: list[dict[str, Any]] = []
        for index, item in enumerate(source_intent.answer_items, start=1):
            if item.output.projection_mode != "named_fields":
                return None
            surfaces = list(item.output.field_surfaces)
            if not surfaces:
                return None
            company = _single_company(source_intent, item, self.canonical)
            if company is None:
                return None
            resolved = resolve_exact_periodic_filing(
                self.canonical, corp_code=company.corp_code,
                question=question, as_of=self.corpus_cutoff,
            ) or self._resolved_document(company.corp_code, item)
            if resolved is None:
                return None
            receipt = resolved.rcept_no
            document_id = f"periodic_{receipt}"
            header_proofs: list[str] = []
            canonical_slots: list[str] = []
            investment_surfaces = (
                item.target.surface, *item.target.qualifier_surfaces,
                *item.output.field_surfaces)
            # Broad ``투자 현황`` narrative is not automatically the strict
            # project-row authority.  Require plan/row semantics.
            investment_request = (
                any("투자" in _header_key(surface)
                    for surface in investment_surfaces)
                and any(token in _header_key(" ".join(investment_surfaces))
                        for token in (
                            "투자계획", "설비투자", "투자대상", "투자목적",
                            "투자금액", "총소요자금", "기지출금액", "투자기간")))
            colloquial_investment_request = (
                item.target.kind == "topic"
                and _header_key(item.target.surface) == "투자"
                and re.search(
                    r"투자.{0,16}(?:뭐|무엇|어떤).{0,16}"
                    r"(?:한다|하는지|했|계획)",
                    question,
                ) is not None
            )
            investment_request = (
                investment_request or colloquial_investment_request)
            if investment_request:
                requested_slots = None
                if (not re.search(
                        r"기\s*지출금액.{0,24}(?:쓰지\s*마|사용하지\s*마|"
                        r"계획금액으로\s*(?:쓰지|사용하지))", question)
                        and any(
                        token in _header_key(question)
                        for token in ("기지출금액", "기지출액", "누적지출금액",
                                      "실제지출금액", "집행금액", "기투자금액"))):
                    from agent.stage1_v1_section_slot_proofs import (
                        INVESTMENT_PLAN_SLOTS, INVESTMENT_SPENT_SLOT,
                    )
                    requested_slots = (*INVESTMENT_PLAN_SLOTS,
                                       INVESTMENT_SPENT_SLOT)
                section_contract = investment_plan_section_slot_proofs(
                    self.canonical, document_id=document_id,
                    **({} if requested_slots is None else {
                        "requested_slots": requested_slots}))
                if section_contract is None:
                    coordinate = _embedded_investment_coordinate(
                        self._document_chunks(company.corp_code, receipt))
                    if coordinate is None:
                        return None
                    retrieval_query = _embedded_investment_retrieval_query(
                        coordinate)
                    if retrieval_query is None:
                        return None
                    # The embedded route is admitted only after the same
                    # chunk proves the four stable header roles.  Bind those
                    # roles to its exact evidence coordinate so the compiler
                    # and runtime retain the strict row contract; no cell
                    # value is inferred here.
                    canonical_slots = list(_INVESTMENT_SLOT_ORDER)
                    evidence_id = str(getattr(
                        coordinate, "evidence_id", "") or "")
                    if not evidence_id:
                        return None
                    header_proofs = [
                        f"source-embedded-header:{document_id}:"
                        f"{evidence_id}:{slot}"
                        for slot in canonical_slots
                    ]
                else:
                    section_title, header_proofs, canonical_slots = section_contract
                    retrieval_query = _canonical_section_query(section_title)
            else:
                # Narrative requests still require a uniquely reconciled
                # source-section title.  Passing a whole report would weaken
                # Stage2's evidence boundary.
                # 공시 서술지표는 절 제목이 아니라 표·문장 속 개념이다. 제목 shingle
                # 매칭이 「고정이하여신비율」을 「합병비율」 절로 끌고 가므로, 사전
                # 표기가 본문에 있으면 그 표기를 먼저 쓴다.
                retrieval_query = self._disclosed_metric_topic(
                    company.corp_code, receipt, item)
                if retrieval_query is None:
                    retrieval_query = self._section_heading(
                        company.corp_code, receipt, item)
                if retrieval_query is None:
                    retrieval_query = self._literal_source_topic(
                        company.corp_code, receipt, item)
                multi_topic_query = self._literal_multi_topic_query(
                    question, surfaces)
                if multi_topic_query is not None:
                    retrieval_query = multi_topic_query
                broad_investment = (
                    item.target.kind == "topic"
                    and _header_key(item.target.surface)
                    in {"투자현황", "최근투자현황"}
                    and not investment_request)
                if broad_investment:
                    # Never bind a broad investment request to an unrelated
                    # business overview merely because the request says
                    # "전반적으로".
                    retrieval_query = self._general_investment_heading(
                        company.corp_code, receipt)
                if (retrieval_query is None
                        and item.target.kind == "topic"
                        and _business_section_leaf(item.target.surface)
                        == "사업의내용"
                        and re.search(
                            r"전반적으로|전체적으로|전반을|전체를", question)):
                    broad_titles = sorted({
                        title for title, path in self._sections(receipt)
                        if self._path_parts(path)
                        and _section_key(_canonical_section_query(
                            self._path_parts(path)[-1])) == "사업의내용"
                    })
                    if len(broad_titles) == 1:
                        retrieval_query = broad_titles[0]
                if retrieval_query is not None:
                    retrieval_query = _canonical_section_query(retrieval_query)
            if not retrieval_query:
                return None
            item_id = f"item-{index}"
            items.append({
                "item_id": item_id,
                "target_surface": item.target.surface,
                "projection_mode": item.output.projection_mode,
                "resolution": {
                    "kind": "periodic_document_narrative",
                    "corp_code": company.corp_code,
                    "corp_name": company.corp_name,
                    "document_id": document_id,
                    "receipt_no": receipt,
                    "source_retrieval_query": retrieval_query,
                    "source_header_proof_refs": header_proofs,
                    "canonical_requested_slots": canonical_slots,
                    "document_proof": {
                        "source_receipt": receipt,
                        "proof_ref": f"source-document:{document_id}",
                    },
                    "narrative_proof": {
                        "source_receipt": receipt,
                        "proof_ref": f"source-narrative:{document_id}",
                    },
                    "executable_field_indexes": list(range(len(surfaces))),
                    "limited_field_indexes": [],
                    "source_cross_check_provenance": None,
                },
                "applied_defaults": [{
                    "policy": "as_of",
                    "basis": "corpus_cutoff",
                    "value": self.corpus_cutoff,
                    "evidence_refs": [
                        f"canonical:corpus-cutoff:{self.corpus_cutoff}"],
                }],
                "field_proofs": _field_proofs(item_id, surfaces),
            })
        return _authority(
            question_id=question_id, source_intent=source_intent,
            canonical_build_id=self.canonical_build_id,
            resolver_version=self.resolver_version,
            reference_date=self.reference_date,
            corpus_cutoff=self.corpus_cutoff, items=items)


class PeriodicNarrativeComparisonBackend(PeriodicNarrativeResolutionBackend):
    """Resolve two ordered periodic-report narrative coordinates generically."""

    @staticmethod
    def _period_range(selector: Any) -> "tuple[date, date] | None":
        if selector.latest or selector.year is None or selector.month is None:
            return None
        year = selector.year
        if selector.form == "annual":
            return date(year, 1, 1), date(year, 12, 31)
        if selector.form == "half":
            return date(year, 1, 1), date(year, 6, 30)
        if selector.form == "quarter":
            start_month = selector.month - 2
            end_day = calendar.monthrange(year, selector.month)[1]
            return date(year, start_month, 1), date(
                year, selector.month, end_day)
        return None

    def _periodic_documents(
            self, corp_code: str, item: Any,
            ) -> "list[tuple[Any, date, date]] | None":
        from agent.periodic_document_preflight import (
            PeriodicDocumentPreflightError,
            parse_periodic_expression,
        )

        document_group = (
            item.scope.document_group_expression or item.target.surface)
        if _header_key(document_group) not in {
                "사업보고서", "연간보고서", "정기보고서"}:
            return None
        periods = list(item.scope.target_period_expressions)
        if len(periods) != 2:
            return None
        found = []
        for period in periods:
            expression = period
            try:
                selector = parse_periodic_expression(expression)
            except PeriodicDocumentPreflightError:
                match = re.fullmatch(r"(20[0-9]{2})년", period)
                if match is None or _header_key(document_group) not in {
                        "사업보고서", "연간보고서"}:
                    return None
                expression = f"{match.group(1)}년 {document_group}"
                try:
                    selector = parse_periodic_expression(expression)
                except PeriodicDocumentPreflightError:
                    return None
            period_range = self._period_range(selector)
            if period_range is None:
                return None
            resolution = self.preflight.resolve_periodic_document(
                corp_code=corp_code, as_of=self.corpus_cutoff,
                target_expression=expression)
            if resolution.status != "resolved" or resolution.candidate is None:
                return None
            found.append((resolution.candidate, *period_range))
        return found if len({row[0].rcept_no for row in found}) == 2 else None

    def _annual_documents(self, corp_code: str, item: Any) -> "list[Any] | None":
        """Compatibility hook retained while accepting mixed periodic forms."""
        rows = self._periodic_documents(corp_code, item)
        return None if rows is None else [row[0] for row in rows]

    def _period_ranges(self, item: Any) -> "list[tuple[date, date]] | None":
        from agent.periodic_document_preflight import (
            PeriodicDocumentPreflightError,
            parse_periodic_expression,
        )

        document_group = (
            item.scope.document_group_expression or item.target.surface)
        ranges = []
        for period in item.scope.target_period_expressions:
            expression = period
            try:
                selector = parse_periodic_expression(expression)
            except PeriodicDocumentPreflightError:
                match = re.fullmatch(r"(20[0-9]{2})년", period)
                if match is None or _header_key(document_group) not in {
                        "사업보고서", "연간보고서"}:
                    return None
                try:
                    selector = parse_periodic_expression(
                        f"{match.group(1)}년 {document_group}")
                except PeriodicDocumentPreflightError:
                    return None
            period_range = self._period_range(selector)
            if period_range is None:
                return None
            ranges.append(period_range)
        return ranges if len(ranges) == 2 else None

    def _axis_coordinate(
            self, corp_code: str, receipt_no: str, surface: str,
            ) -> "Any | None":
        rows = self._document_chunks(corp_code, receipt_no)
        compact_surface = _header_key(surface)
        if "투자" in compact_surface and "계획" in compact_surface:
            embedded = _embedded_investment_coordinate(rows)
            if embedded is not None:
                return embedded
        standard_leaf = _business_section_leaf(surface)
        if standard_leaf is not None and standard_leaf != "사업의내용":
            standard = {}
            for row in rows:
                parts = self._path_parts(getattr(row, "path", "") or "")
                if (len(parts) < 2
                        or _section_key(parts[-2]) != "사업의내용"
                        or _section_key(parts[-1]) != standard_leaf
                        or not all(getattr(row, key, None) for key in (
                            "source_file_id", "path", "locator", "evidence_id"))):
                    continue
                coordinate = (
                    row.source_file_id, row.path, row.locator, row.evidence_id)
                standard[coordinate] = row
            if len(standard) == 1:
                return next(iter(standard.values()))
            return None

        shingles = {
            compact[index:index + 2]
            for compact in [re.sub(r"\s+", "", surface)]
            for index in range(len(compact) - 1)
        }
        if len(shingles) < 2:
            return None
        best_score = 0
        candidates: list[Any] = []
        for row in rows:
            score = sum(piece in (getattr(row, "path", "") or "")
                        for piece in shingles)
            if score > best_score:
                best_score, candidates = score, [row]
            elif score == best_score and score > 0:
                candidates.append(row)
        unique = {
            (getattr(row, "source_file_id", None), getattr(row, "path", None),
             getattr(row, "locator", None), getattr(row, "evidence_id", None)): row
            for row in candidates
            if all(getattr(row, key, None) for key in (
                "source_file_id", "path", "locator", "evidence_id"))
        }
        return next(iter(unique.values())) if best_score >= 2 and len(unique) == 1 else None

    def resolve(
            self, *, question_id: str, question: str,
            source_intent: SemanticIntent,
            ) -> "dict[str, Any] | None":
        del question
        if len(source_intent.answer_items) != 1:
            return None
        item = source_intent.answer_items[0]
        if (item.operation != "compare" or item.output.shape not in {"comparison", "narrative"}
                or item.output.projection_mode != "named_fields"
                or len(item.output.field_surfaces) < 2):
            return None
        company = _single_company(source_intent, item, self.canonical)
        if company is None:
            return None
        docs = self._annual_documents(company.corp_code, item)
        ranges = self._period_ranges(item)
        if docs is None or ranges is None:
            return None
        document_group = (
            item.scope.document_group_expression or item.target.surface)
        axes = list(item.output.field_surfaces if item.output.shape == "comparison"
                    else item.output.field_surfaces[:-1])
        documents = []
        for period_index, candidate in enumerate(docs):
            period_start, period_end = ranges[period_index]
            rows = []
            for field_index, surface in enumerate(axes):
                coordinate = self._axis_coordinate(
                    company.corp_code, candidate.rcept_no, surface)
                if coordinate is None:
                    return None
                rows.append({
                    "source_field_index": field_index,
                    "axis_id": f"axis_{field_index + 1}",
                    "path": coordinate.path,
                    "locator": coordinate.locator,
                    "source_file_id": coordinate.source_file_id,
                    "evidence_id": coordinate.evidence_id,
                })
            source_files = {row["source_file_id"] for row in rows}
            if len(source_files) != 1:
                return None
            documents.append({
                "source_period_index": period_index,
                "issuer_corp_code": company.corp_code,
                "issuer_corp_name": company.corp_name,
                "doc_id": f"periodic_{candidate.rcept_no}",
                "receipt_no": candidate.rcept_no,
                "period_start": period_start,
                "period_end": period_end,
                "source_file_id": next(iter(source_files)),
                "evidence": rows,
            })
        item_id = item.item_id
        return _authority(
            question_id=question_id, source_intent=source_intent,
            canonical_build_id=self.canonical_build_id,
            resolver_version=self.resolver_version,
            reference_date=self.reference_date, corpus_cutoff=self.corpus_cutoff,
            items=[{
                "item_id": item_id, "target_surface": item.target.surface,
                "projection_mode": item.output.projection_mode,
                "resolution": {
                    "kind": "periodic_narrative_comparison",
                    "document_group": document_group,
                    "documents": documents,
                },
                "applied_defaults": [],
                "field_proofs": _field_proofs(
                    item_id, list(item.output.field_surfaces)),
            }])


class SelectedEventResolutionBackend:
    """사건 후보가 **하나로 좁혀질 때만** 그 사건을 확정한다.

    여럿이면 물러난다 — 그 자리는 역질문 백엔드가 맡는다.  둘을 한 백엔드에 넣으면
    「고르지 못했다」와 「골랐다」가 같은 코드에서 갈려 추적이 어려워진다.
    """

    def __init__(
            self, canonical: Any, *, canonical_build_id: str,
            resolver_version: str, reference_date: date, corpus_cutoff: str,
            event_preflight: Any | None = None,
            ) -> None:
        from agent.event_preflight import CanonicalEventKeyPreflight

        self.canonical = canonical
        self.preflight = (
            event_preflight if event_preflight is not None
            else CanonicalEventKeyPreflight(
                canonical, corpus_cutoff=corpus_cutoff))
        if not callable(getattr(self.preflight, "resolve_event_key", None)):
            raise TypeError("selected event backend preflight 계약이 잘못되었습니다")
        self.canonical_build_id = canonical_build_id
        self.resolver_version = resolver_version
        self.reference_date = reference_date
        self.corpus_cutoff = corpus_cutoff

    def _counterparty(self, intent: SemanticIntent, item: Any) -> "str | None":
        explicit = [
            entity.surface for entity in intent.entities
            if entity.kind_hint == "counterparty" and entity.surface.strip()
        ]
        if len(explicit) == 1:
            return explicit[0]
        if explicit:
            return None

        # An external legal person is often emitted as `company`.  Separate it
        # from the issuer only when exactly one company surface is not in the
        # canonical issuer registry.
        unresolved_companies = [
            entity.surface for entity in intent.entities
            if entity.kind_hint == "company" and entity.surface.strip()
            and not self.canonical.resolve_company(entity.surface)
        ]
        if len(unresolved_companies) == 1:
            return unresolved_companies[0]
        if unresolved_companies:
            return None

        # Some valid semantic wires preserve the external party as an `event`
        # mention (for example, ``Freudenberg 배터리 공급계약``).  Do not treat an
        # arbitrary event surface as a counterparty: `Hornsea Four 계약` is a
        # contract name, and the old generic event fallback misclassified it.
        # Use only a literal shared alias or one residual Latin token after
        # removing the uniquely resolved issuer surface.  Canonical event
        # preflight still has to prove the resulting selector.
        from agent.event_preflight import _counterparty_aliases, _key

        entity_by_id = {entity.entity_id: entity for entity in intent.entities}
        surfaces = [
            entity_by_id[ref].surface
            for ref in item.target.entity_refs
            if ref in entity_by_id
            and entity_by_id[ref].kind_hint in {"event", "document"}
        ]
        surfaces.append(item.target.surface)
        issuer_surfaces = [
            entity.surface for entity in intent.entities
            if entity.kind_hint == "company"
            and len(self.canonical.resolve_company(entity.surface)) == 1
        ]
        aliases = _counterparty_aliases()
        candidates: dict[tuple[str, ...], str] = {}
        for original in dict.fromkeys(surfaces):
            residual = original
            for issuer in issuer_surfaces:
                residual = residual.replace(issuer, " ")
            normalized = _key(residual)
            alias_hits = [
                (alias, values) for alias, values in aliases.items()
                if len(alias) >= 2 and alias in normalized
            ]
            if alias_hits and len({values for _, values in alias_hits}) == 1:
                alias, values = max(alias_hits, key=lambda row: len(row[0]))
                current = candidates.get(values)
                if current is None or len(alias) > len(current):
                    candidates[values] = alias
            tokens = re.findall(r"[A-Za-z][A-Za-z0-9.&'-]*", residual)
            # ``FCS(필드제어설비)``처럼 바로 뒤 괄호가 약어를 풀어 쓰는
            # 표기는 상대방이 아니라 계약명 구성요소다. 이 경우에만 Latin
            # 토큰 fallback을 막고, 독립된 영문 상대방은 기존 검증을 그대로 탄다.
            expanded_acronym = bool(tokens and re.search(
                rf"{re.escape(tokens[0])}\s*\([^)]*[가-힣][^)]*\)",
                residual))
            if (len(tokens) == 1 and len(tokens[0]) >= 3
                    and not expanded_acronym):
                candidates.setdefault((tokens[0].casefold(),), tokens[0])
        return next(iter(candidates.values())) if len(candidates) == 1 else None

    def _as_of_repeats_target_date(self, item: Any) -> bool:
        """Whether a target date is an observation coordinate, not an event date.

        Some semantic wires retain an exact date in both ``scope.as_of`` and
        ``target.qualifier_surfaces``.  Treating the latter as an occurrence
        window filters on the wrong filing date and can turn a resolvable event
        into an ambiguity.  Require exact, question-grounded date equality;
        broad year/month qualifiers remain valid event selectors.
        """

        as_of = item.scope.as_of_expression
        if not as_of or len(item.target.qualifier_surfaces) != 1:
            return False
        qualifier = item.target.qualifier_surfaces[0]
        as_start, as_end, as_error = _target_date_range(
            as_of, reference_date=self.reference_date)
        qualifier_start, qualifier_end, qualifier_error = _target_date_range(
            qualifier, reference_date=self.reference_date)
        return (
            as_error is None and qualifier_error is None
            and as_start is not None and as_start == as_end
            and qualifier_start is not None and qualifier_start == qualifier_end
            and as_start == qualifier_start
        )

    def _has_one_bounded_event_period(self, item: Any) -> bool:
        """Whether one event occurrence coordinate is at most one month.

        Boundary normalization moves date-like target qualifiers into
        ``scope.target_period_expressions``.  Accept either representation,
        but never both or more than one coordinate.  A literal year-month is
        usable only when issuer + form close to one canonical document in the
        bounded window; broader periods stay out.
        """

        surfaces = [
            *item.target.qualifier_surfaces,
            *item.scope.target_period_expressions,
        ]
        if len(surfaces) != 1:
            return False
        start, end, error = _target_date_range(
            surfaces[0], reference_date=self.reference_date)
        if error is not None or start is None or end is None:
            return False
        start_day = date(int(start[:4]), int(start[4:6]), int(start[6:8]))
        end_day = date(int(end[:4]), int(end[4:6]), int(end[6:8]))
        return 0 <= (end_day - start_day).days <= 30

    def _status_timepoints(
            self, item: Any, question: str) -> "list[str] | None":
        """Recover exactly two ordered observation dates, or decline.

        Date surfaces may be stored on the event target or in target-period
        scope.  The shared planner helper restores coordinated omissions such
        as ``25일과 26일``; it also rejects ambiguous/non-day expressions.
        """
        scope_dates = list(item.scope.target_period_expressions)
        target_dates = list(item.target.qualifier_surfaces)
        # A provider may repeat one exact observation date in both ``as_of``
        # and the target qualifier.  Preserve it as the single status
        # timepoint; it is not an occurrence selector.
        if (self._as_of_repeats_target_date(item)
                and item.scope.as_of_expression is not None):
            timepoints, defaulted, error = _event_timepoints(
                item.scope.as_of_expression, corpus_cutoff=self.corpus_cutoff)
            if (error is None and not defaulted and timepoints
                    and len(timepoints) == 1):
                return timepoints
        sources = [rows for rows in (scope_dates, target_dates) if len(rows) == 2]
        if len(sources) == 1:
            timepoints, defaulted, error = _event_timepoints(
                "과".join(sources[0]), corpus_cutoff=self.corpus_cutoff)
            if (error is None and not defaulted and timepoints is not None
                    and len(timepoints) == 2):
                # A full history request can additionally ask for the state at
                # one explicit cutoff.  Preserve that third coordinate instead
                # of interpreting it as an occurrence selector or silently
                # dropping it.  All coordinates remain literal exact dates.
                if (item.output.shape == "timeline"
                        and item.scope.as_of_expression is not None):
                    as_of_points, as_of_defaulted, as_of_error = _event_timepoints(
                        item.scope.as_of_expression,
                        corpus_cutoff=self.corpus_cutoff)
                    if (as_of_error is not None or as_of_defaulted
                            or as_of_points is None or len(as_of_points) != 1):
                        return None
                    if as_of_points[0] not in timepoints:
                        timepoints.append(as_of_points[0])
                return timepoints

        # `2025-12-25 vs 26` cannot be reconstructed from the detached bare
        # `26` alone.  The original question grammar is authoritative and the
        # shared scanner accepts this one closed omission form.  Require two
        # semantic date surfaces and exactly two full question days so an
        # unrelated date in prose cannot become an event coordinate.
        temporal_surfaces = list(dict.fromkeys(
            scope_dates + target_dates + list(item.output.field_surfaces)))
        if len([value for value in temporal_surfaces if value in question]) < 2:
            return None
        question_days = sorted({
            f"{year:04d}{month:02d}{day:02d}"
            for year, month, day in question_date_surfaces(question)
            if month is not None and day is not None
        })
        if len(question_days) != 2 or any(
                value > self.corpus_cutoff for value in question_days):
            return None
        return question_days

    def _implicit_cutoff_status(self, item: Any) -> "list[str] | None":
        """Use corpus cutoff for a literal status predicate with no time axis."""

        if (
                item.scope.as_of_expression is not None
                or item.scope.target_period_expressions
                or item.target.qualifier_surfaces
                or item.output.shape != "scalar"
                or len(item.output.field_surfaces) != 1
                or re.search(
                    r"상태|유효|살아\s*있|끝난|해지",
                    item.output.field_surfaces[0]) is None
        ):
            return None
        return [self.corpus_cutoff]

    def _single_status_timepoint(self, item: Any) -> "list[str] | None":
        """A corpus-cutoff status with one exact event seed date."""
        if (item.scope.as_of_expression is None
                or len(item.scope.target_period_expressions) != 1
                or len(item.target.qualifier_surfaces) != 1
                or item.output.shape != "scalar"):
            return None
        points, defaulted, error = _event_timepoints(
            item.scope.as_of_expression, corpus_cutoff=self.corpus_cutoff)
        # ``코퍼스 기준일`` is intentionally resolved by the shared helper as
        # its explicit cutoff default; that is the requested intrinsic point,
        # not an omitted observation date.
        if error is not None or points != [self.corpus_cutoff]:
            return None
        start, end, seed_error = _target_date_range(
            item.scope.target_period_expressions[0], reference_date=self.reference_date)
        if seed_error is not None or start is None or start != end:
            return None
        return points

    def _partial_timeline_missing_root(self, item: Any, candidate: Any) -> str | None:
        """Return a root-missing date only from a public canonical relation."""
        if (item.target.kind != "document"
                or item.output.shape not in {"narrative", "timeline"}
                or len(item.output.field_surfaces) != 2
                or item.scope.as_of_expression is not None
                or item.target.qualifier_surfaces
                or len(item.scope.target_period_expressions) > 1
                or not hasattr(self.canonical, "relation_summaries")):
            return None
        if item.scope.target_period_expressions:
            start, end, error = _target_date_range(
                item.scope.target_period_expressions[0],
                reference_date=self.reference_date)
            if (error is not None or start is None or start != end
                    or start != candidate.seed_rcept_no[:8]):
                return None
        timeline = self.canonical.event_timeline(
            as_of=self.corpus_cutoff, event_key=candidate.event_key,
            verify_evidence=True)
        if timeline is None:
            return None
        missing = []
        for observation in timeline.observations:
            for relation in self.canonical.relation_summaries(
                    source_rcept_no=observation.rcept_no,
                    as_of=self.corpus_cutoff):
                if (relation.relation_type == "CORRECTS"
                        and relation.resolution_status == "root_missing"
                        and relation.dst_rcept_no is None
                        and relation.root_missing_reason == "submitted_before_corpus"):
                    digits = "".join(char for char in (relation.target_hint or "")
                                     if char.isdigit())
                    if len(digits) == 8:
                        missing.append(digits)
        return missing[0] if len(missing) == 1 else None

    @staticmethod
    def _lineage_observation_request(item: Any) -> bool:
        """Whether one selected event asks for root content plus full history.

        These two surfaces describe a lifecycle projection, not literal form
        fields.  The public v0.4 task can express the projection through its
        existing internal ``원계약계보관측`` slot at the corpus cutoff.
        No legal status or missing-root fact is inferred here.
        """

        fields = [re.sub(r"\s+", "", value)
                  for value in item.output.field_surfaces]
        return bool(
            item.target.kind in {"event", "document"}
            and item.operation == "retrieve"
            and item.selection is None
            and item.output.projection_mode == "named_fields"
            and item.output.shape in {"narrative", "timeline", "record_list"}
            and len(fields) == 2
            and any(any(token in value for token in (
                "최초체결", "최초공시", "원계약", "원공시",
                "최신유효본", "유효본", "최신본", "마지막공시",
            )) for value in fields)
            and any(any(token in value for token in (
                "전체변경이력", "전체정정이력", "변경이력",
                "정정이력", "전체이력", "계보", "흐름",
            )) for value in fields)
            and not item.target.qualifier_surfaces
            and len(item.scope.target_period_expressions) <= 1
            and item.scope.as_of_expression is None
            and item.scope.document_group_expression is None
            and not item.scope.scope_qualifier_expressions
        )

    def _resolution_entity_surface(
            self, intent: SemanticIntent, item: Any, company: Any,
            *, intrinsic_status: bool) -> str | None:
        if intrinsic_status:
            return item.target.surface
        by_id = {entity.entity_id: entity for entity in intent.entities}
        issuer_surfaces = [
            by_id[ref].surface for ref in item.target.entity_refs
            if ref in by_id and by_id[ref].kind_hint == "company"
            and len(self.canonical.resolve_company(by_id[ref].surface)) == 1
            and self.canonical.resolve_company(by_id[ref].surface)[0].corp_code
            == company.corp_code
        ]
        return issuer_surfaces[0] if len(issuer_surfaces) == 1 else None

    def _external_issuer_candidates(
            self, *, counterparty: str,
            item: Any, observation_timepoints: "list[str] | None" = None,
            ) -> "list[tuple[Any, Any]]":
        """Find one issuer when the semantic input names only a counterparty.

        Robustness variants may contain no issuer entity at all.  The canonical
        event-role index is still authoritative: scan issuer corp codes,
        resolve the same counterparty/contract/date constraints, and retain a
        result only when exactly one issuer/event pair survives.  This is a
        corpus-backed role lookup, not a company or question-ID allowlist.
        """

        documents = getattr(self.canonical, "documents", None)
        if not callable(documents):
            return []
        corp_names: dict[str, str] = {}
        for row in documents(as_of=self.corpus_cutoff):
            corp_code = str(getattr(row, "corp_code", "") or "")
            corp_name = str(getattr(row, "corp_name", "") or "")
            if corp_code and corp_name:
                corp_names.setdefault(corp_code, corp_name)
        pairs: list[tuple[Any, Any]] = []
        for corp_code, corp_name in corp_names.items():
            resolution = self._narrowed_event(
                item,
                type("_Company", (), {
                    "corp_code": corp_code,
                    "corp_name": corp_name,
                })(),
                counterparty,
                observation_timepoints=observation_timepoints,
            )
            if resolution is None or len(resolution.candidates) != 1:
                continue
            pairs.append((
                type("_Company", (), {
                    "corp_code": corp_code,
                    "corp_name": corp_name,
                })(),
                resolution,
            ))
        return pairs

    def _narrowed_event(
            self, item: Any, company: Any, counterparty: "str | None",
            *, observation_timepoints: "list[str] | None" = None,
            allow_date_only: bool = False):
        """좁히는 단서를 **좁혀질 때만** 쓴다. 하나로 안 좁혀지면 ``None``.

        사용자가 말한 계약명(`배터리 계약`)은 공시의 계약명과 글자가 다르다.
        그것을 하드 필터로 걸면 후보가 0이 되고, 상대·기간으로 이미 좁혀 둔
        것까지 함께 날아간다(관측: 상대만으로 16→2, 계약명을 더하면 →0).
        검증되지 않는 표면으로 후보를 지우지 않는다.

        그래서 단서를 **많은 쪽부터 빼 가며** 본다.  어느 단계든 후보가 하나면
        그것을 쓰고, 끝까지 여럿이면 물러난다 — 그 자리는 역질문 계층이 받는다.
        """

        # An exact date duplicated in scope.as_of is an observation point.  It
        # must not become the event occurrence window as well.
        if self._as_of_repeats_target_date(item):
            event_from, event_to = None, None
        else:
            event_from, event_to = qualifier_event_window(
                item, reference_date=self.reference_date)
        if (event_from is None and event_to is None
                and len(item.scope.target_period_expressions) == 1):
            start, end, error = _target_date_range(
                item.scope.target_period_expressions[0],
                reference_date=self.reference_date)
            if error is None and start and end:
                start_day = date(
                    int(start[:4]), int(start[4:6]), int(start[6:8]))
                end_day = date(
                    int(end[:4]), int(end[4:6]), int(end[6:8]))
                if not 0 <= (end_day - start_day).days <= 30:
                    start = end = None
            if start and end:
                event_from, event_to = start, end
        contract_surfaces = [item.target.surface]
        # Semantic normalization may retain a user-facing trailing ``계약``
        # outside a quoted canonical contract name.  Try the narrower,
        # punctuation-free contract surface as an additional canonical filter;
        # preflight still has to prove a single candidate.
        compact = re.sub(r"\s*계약\s*$", "", item.target.surface).strip()
        compact = compact.strip("'\" ")
        if compact and compact not in contract_surfaces:
            contract_surfaces.append(compact)
        # A literal disclosure form (for example ``영업정지``) is an event
        # type, not a contract name.  Try it as the strongest selector before
        # the legacy contract/counterparty fallbacks.  If the canonical role
        # index does not recognize the surface as a form, that attempt simply
        # yields no candidate and the existing fail-closed narrowing sequence
        # remains unchanged.
        attempts = [
            {"event_type": item.target.surface,
             "event_from": event_from, "event_to": event_to},
        ] + [
            {"counterparty": counterparty, "contract_name": contract_surface,
             "event_from": event_from, "event_to": event_to}
            for contract_surface in contract_surfaces
        ] + [
            {"counterparty": counterparty,
             "event_from": event_from, "event_to": event_to},
            {"counterparty": counterparty},
        ]
        seen: list[dict[str, Any]] = []
        for keywords in attempts:
            active = {k: v for k, v in keywords.items() if v is not None}
            # An empty filter carries no identity evidence at all — it is not
            # a "narrowing" step, it is an unconstrained scan over every
            # contract-role event this issuer has.  When such a scan happens
            # to have exactly one row in the corpus, treating it as
            # ``resolved`` would silently bind the answer to whichever event
            # is corpus-incidental rather than to anything the question named
            # (issue #121: a same-company, unrelated-date contract filing was
            # returned for a delisting-decision question that has no
            # contract-role fields at all).  Only a non-empty filter can
            # prove identity here.
            # Dates bound an event occurrence but do not identify its family.
            # A date-only preflight can accidentally select an unrelated
            # same-day contract-role event before the exact public form
            # fallback below gets a chance to inspect the filing type.
            if (not active
                    or (not allow_date_only
                        and not any(key in active for key in (
                            "event_type", "counterparty", "contract_name")))
                    or active in seen):
                continue
            seen.append(active)
            resolution = self.preflight.resolve_event_key(
                corp_code=company.corp_code, as_of=self.corpus_cutoff,
                **active)
            if resolution.status == "resolved" and len(
                    resolution.candidates) == 1:
                return resolution
            if (resolution.status == "ambiguous"
                    and observation_timepoints
                    and len(observation_timepoints) == 2):
                # A broad year/contract phrase may leave many events.  A later
                # observation day named by the user is canonical event evidence:
                # retain candidates whose timeline actually has an observation
                # on either requested day, and only accept a unique survivor.
                matched = []
                requested = set(observation_timepoints)
                for candidate in resolution.candidates:
                    timeline = self.canonical.event_timeline(
                        as_of=self.corpus_cutoff,
                        event_key=candidate.event_key,
                        verify_evidence=True,
                    )
                    if timeline is None:
                        continue
                    observed = {
                        str(getattr(row, "observed_at", "") or "")
                        for row in timeline.observations
                    }
                    if requested & observed:
                        matched.append(candidate)
                if len(matched) == 1:
                    from agent.event_preflight import EventKeyResolution
                    return EventKeyResolution("resolved", tuple(matched))
        # Issuer + a bounded filing period + form-family is a public document
        # coordinate when it closes to exactly one canonical document, even
        # if the generic contract-role index does not cover that event type.
        # A full date is the ordinary case; a literal year-month is admitted
        # only because the window is capped below and uniqueness is still
        # proven from canonical document metadata.  The selected filing may
        # be a later correction or termination observation: preserve its
        # canonical event root and carry the exact filing separately.
        bounded_days = None
        if event_from is not None and event_to is not None:
            start_day = date(
                int(event_from[:4]), int(event_from[4:6]), int(event_from[6:8]))
            end_day = date(
                int(event_to[:4]), int(event_to[4:6]), int(event_to[6:8]))
            bounded_days = (end_day - start_day).days
        if (bounded_days is not None and 0 <= bounded_days <= 30
                and callable(getattr(self.canonical, "documents", None))):
            # 이슈 #132 실호출 — 질문 표면("해외증권시장 주권 상장폐지 결정")은
            # 서식명("해외증권시장주권등상장폐지결정")의 열거 조사 "등" 하나만
            # 빠진 채 그 외 전부 같다. #121 이 이미 SLOT_LABELS 로 고친 필드
            # 라벨 글자 수 차이와 같은 종류의 문제 — 서식명의 형식적 "등"은
            # 의미를 바꾸지 않으므로, 이 exact-date 단일 서식 매칭에서만 양쪽에서
            # 지운다. 회사·정확한 접수일로 이미 좁힌 뒤의 마지막 리터럴
            # 비교이므로 값을 새로 고르지 않는다.
            target_key = re.sub(
                r"[^0-9a-z가-힣]+", "",
                item.target.surface.casefold()).replace("등", "")
            exact_documents = []
            for document in self.canonical.documents(
                    as_of=self.corpus_cutoff, corp_code=company.corp_code):
                observed = str(getattr(document, "rcept_dt", ""))
                if not event_from <= observed <= event_to:
                    continue
                searchable = re.sub(
                    r"[^0-9a-z가-힣]+", "",
                    (f"{getattr(document, 'form', '')} "
                     f"{getattr(document, 'report_nm', '')}").casefold()
                    ).replace("등", "")
                if len(target_key) >= 3 and target_key in searchable:
                    exact_documents.append(document)
            receipts = sorted({
                str(getattr(row, "rcept_no", ""))
                for row in exact_documents
                if re.fullmatch(r"[0-9]{14}", str(
                    getattr(row, "rcept_no", "")))
            })
            if len(receipts) == 1:
                receipt = receipts[0]
                timeline = self.canonical.event_timeline(
                    as_of=self.corpus_cutoff, rcept_no=receipt,
                    verify_evidence=True)
                event_key = getattr(timeline, "event_key", None)
                observations = tuple(
                    getattr(timeline, "observations", ()) or ())
                root_receipt = str(
                    getattr(timeline, "root_rcept_no", "") or "")
                if (timeline is not None
                        and getattr(timeline, "corp_code", None)
                        == company.corp_code
                        and re.fullmatch(r"[0-9]{14}", root_receipt)
                        and any(getattr(row, "rcept_no", None) == receipt
                                for row in observations)
                        and isinstance(event_key, str)
                        and re.fullmatch(r"[0-9a-f]{32}", event_key)):
                    from agent.event_preflight import (
                        EventKeyCandidate, EventKeyResolution,
                    )
                    return EventKeyResolution("resolved", (
                        EventKeyCandidate(
                            event_key=event_key,
                            seed_rcept_no=root_receipt,
                            label=(str(getattr(
                                exact_documents[0], "report_nm", ""))
                                   or item.target.surface),
                            identity_fingerprint=getattr(
                                timeline, "identity_fingerprint", None),
                            identity_status=getattr(
                                timeline, "identity_status", None),
                            observation_count=len(observations),
                            selected_observation_rcept_no=receipt,
                        ),
                    ))
        # A correction whose original predates the corpus can have no ordinary
        # root event row to match.  The correction filing itself is still a
        # valid bounded seed when the public coordinates close to exactly one
        # document: issuer + exact day + contract disclosure family.  This
        # fallback deliberately refuses multiple same-day filings and also
        # requires the role index to prove both that the literal target names
        # one contract globally and that the selected receipt itself carries
        # that contract-name role.  Uniqueness by issuer/day/form alone is not
        # an identity proof: the sole filing on that day may concern a
        # different contract.
        names_one_contract = getattr(
            self.preflight, "names_a_single_contract", None)
        target_names_one_contract = (
            callable(names_one_contract)
            and any(names_one_contract(
                corp_code=company.corp_code, surface=surface)
                    for surface in contract_surfaces)
        )
        if (event_from is not None and event_from == event_to
                and re.search(r"계약|공급|판매", item.target.surface)
                and target_names_one_contract
                and callable(getattr(self.canonical, "documents", None))):
            documents = [
                row for row in self.canonical.documents(
                    as_of=self.corpus_cutoff,
                    corp_code=company.corp_code,
                    doc_group="exchange", is_correction=True)
                if str(getattr(row, "rcept_dt", "")) == event_from
                and re.search(
                    r"판매|공급|계약",
                    " ".join((
                        str(getattr(row, "form", "") or ""),
                        str(getattr(row, "report_nm", "") or ""),
                    )))
            ]
            receipts = sorted({
                str(getattr(row, "rcept_no", "")) for row in documents
                if re.fullmatch(r"[0-9]{14}", str(
                    getattr(row, "rcept_no", "")))
            })
            if len(receipts) == 1:
                receipt = receipts[0]
                receipt_names_contract = getattr(
                    self.preflight, "receipt_names_contract", None)
                receipt_matches_target = (
                    callable(receipt_names_contract)
                    and any(receipt_names_contract(
                        corp_code=company.corp_code,
                        rcept_no=receipt,
                        surface=surface,
                    ) for surface in contract_surfaces)
                )
                if not receipt_matches_target:
                    return None
                timeline = self.canonical.event_timeline(
                    as_of=self.corpus_cutoff, rcept_no=receipt,
                    verify_evidence=True)
                event_key = getattr(timeline, "event_key", None)
                observations = tuple(
                    getattr(timeline, "observations", ()) or ())
                if (timeline is not None
                        and getattr(timeline, "corp_code", None)
                        == company.corp_code
                        and getattr(timeline, "root_rcept_no", None) == receipt
                        and isinstance(event_key, str)
                        and re.fullmatch(r"[0-9a-f]{32}", event_key)):
                    from agent.event_preflight import (
                        EventKeyCandidate, EventKeyResolution,
                    )
                    return EventKeyResolution("resolved", (
                        EventKeyCandidate(
                            event_key=event_key, seed_rcept_no=receipt,
                            label=(str(getattr(documents[0], "report_nm", ""))
                                   or item.target.surface),
                            identity_fingerprint=getattr(
                                timeline, "identity_fingerprint", None),
                            identity_status=getattr(
                                timeline, "identity_status", None),
                            observation_count=len(observations),
                        ),
                    ))
        return None

    def resolve(
            self, *, question_id: str, question: str,
            source_intent: SemanticIntent,
            ) -> "dict[str, Any] | None":
        items: list[dict[str, Any]] = []
        for index, item in enumerate(source_intent.answer_items, start=1):
            # kind 로 잠그지 않는다. 사건 key 가 **유일하게 확정될 때만** 받는다.
            surfaces = list(item.output.field_surfaces)
            if not surfaces:
                return None
            event_field_read = (
                len(source_intent.answer_items) == 1
                and not source_intent.answer_groups
                and not source_intent.premises
                and not source_intent.unresolved_mentions
                and item.target.kind in _EVENT_TARGET_KINDS
                and item.operation == "retrieve"
                and ((not item.target.qualifier_surfaces
                      and not item.scope.target_period_expressions)
                     or self._has_one_bounded_event_period(item))
                and item.scope.as_of_expression is None
                and item.scope.document_group_expression is None
                and not item.scope.scope_qualifier_expressions
                and getattr(item.selection, "mode", None) in {None, "latest"}
                and item.output.projection_mode == "named_fields"
                and item.output.shape == "record"
                # 이슈 #132 — CG-011(#121)의 두 필드 record 읽기는 이미
                # 여기서 받는다. 사건이 유일하게 확정되는 조건은 위에서 이미
                # 전부 갖췄으므로(entity·qualifier·scope 전부 닫힘), 필드가
                # 하나뿐이어도 같은 이유로 받아야 한다 — 이전에는 >= 2 만
                # 받아 단일 필드 사건 읽기가 timepoints 없이는 전부 거절됐다.
                and len(surfaces) >= 1
            )
            timepoints = (self._status_timepoints(item, question)
                          or self._single_status_timepoint(item)
                          or self._implicit_cutoff_status(item))
            lineage_observation = self._lineage_observation_request(item)
            counterparty = self._counterparty(source_intent, item)
            company = _single_company(source_intent, item, self.canonical)
            issuer_pairs: list[tuple[Any, Any]] = []
            if company is not None:
                resolution = self._narrowed_event(
                    item, company, counterparty,
                    observation_timepoints=timepoints,
                    # A generic ``계약`` plus exact correction day is enough
                    # only for the closed root/history projection.  Its later
                    # root-missing relation check prevents the date from
                    # becoming an arbitrary same-day field selector.
                    allow_date_only=lineage_observation)
                if resolution is not None:
                    issuer_pairs = [(company, resolution)]
            elif counterparty is not None:
                issuer_pairs = self._external_issuer_candidates(
                    counterparty=counterparty, item=item,
                    observation_timepoints=timepoints)
            # An external-only mention is safe only when one issuer/event pair
            # survives.  Multiple issuers remain unresolved rather than being
            # collapsed by a lexical guess.
            if len(issuer_pairs) != 1:
                return None
            company, resolution = issuer_pairs[0]
            entity_surface = self._resolution_entity_surface(
                source_intent, item, company,
                intrinsic_status=bool(timepoints))
            if entity_surface is None and event_field_read:
                issuer_surfaces = [
                    entity.surface for entity in source_intent.entities
                    if entity.kind_hint == "company"
                    and len(self.canonical.resolve_company(entity.surface)) == 1
                    and self.canonical.resolve_company(
                        entity.surface)[0].corp_code == company.corp_code
                ]
                if len(issuer_surfaces) == 1:
                    entity_surface = issuer_surfaces[0]
            if entity_surface is None:
                entity_surface = item.target.surface if timepoints else None
            if entity_surface is None:
                return None
            candidate = resolution.candidates[0]
            selected_receipt = (
                candidate.selected_observation_rcept_no
                or candidate.seed_rcept_no)
            missing_root = self._partial_timeline_missing_root(item, candidate)
            complete_timeline = (
                item.output.shape == "timeline"
                and timepoints is not None
                and len(timepoints) in {2, 3}
            )
            timeline = (missing_root is not None or complete_timeline
                        or lineage_observation)
            if missing_root is not None:
                timepoints = [self.corpus_cutoff]
                entity_surface = item.target.surface
            elif lineage_observation:
                # A history projection needs one bounded execution cutoff, not
                # a fabricated status point.  Stage2 sees the typed lineage
                # slot and enumerates only canonical observations up to it.
                timepoints = [self.corpus_cutoff]
                entity_surface = item.target.surface
            elif not timepoints and not event_field_read:
                return None
            item_id = f"item-{index}"
            from agent.stage1_v1_public_event_selector import (
                select_public_event_selector,
            )
            public_selector = select_public_event_selector(
                self.canonical, as_of=self.corpus_cutoff,
                receipt=candidate.seed_rcept_no, intent=source_intent,
                item=item, counterparty_surface=counterparty,
            )
            resolution_payload = {
                "kind": "selected_event",
                "corp_code": company.corp_code,
                "corp_name": company.corp_name,
                "entity_surface": entity_surface,
                "event_key": candidate.event_key,
                "root_receipt": candidate.seed_rcept_no,
                "selector_proof": {
                    "source_receipt": selected_receipt,
                    "proof_ref": (
                        (f"source-event:{candidate.event_key}:"
                         f"{selected_receipt}")
                        if candidate.selected_observation_rcept_no
                        is not None else
                        f"source-event:{candidate.event_key}"),
                },
                "timepoints": timepoints or [],
                "operation": "timeline" if timeline else "status",
                "lineage_missing_root_date": missing_root,
            }
            if not public_selector.empty:
                resolution_payload["public_selector"] = public_selector.as_payload()
            items.append({
                "item_id": item_id,
                "target_surface": item.target.surface,
                "projection_mode": item.output.projection_mode,
                "resolution": resolution_payload,
                "applied_defaults": [],
                "field_proofs": _field_proofs(item_id, surfaces),
            })
        return _authority(
            question_id=question_id, source_intent=source_intent,
            canonical_build_id=self.canonical_build_id,
            resolver_version=self.resolver_version,
            reference_date=self.reference_date,
            corpus_cutoff=self.corpus_cutoff, items=items)


#: 이슈 #44 — 「투자판단관련주요경영사항」서식명 + 연도 whole-target 문서 묶음.
#: 일반화한 서식 레지스트리는 아직 없다.  이 상수·정규식은 이 서식 하나만
#: 겨냥하므로, 관련 없는 document/topic target은 이전과 동일하게
#: 「전체 문서」선택자로 떨어지거나 거절된다.
_INVESTMENT_JUDGMENT_FORM = "투자판단관련주요경영사항"
_INVESTMENT_JUDGMENT_FORM_PATTERN = re.compile(
    r"투자\s*판단\s*(?:관련)?\s*주요\s*경영\s*사항")
_DOCUMENT_COLLECTION_YEAR_PATTERN = re.compile(
    r"(?<![0-9])(20[0-9]{2})\s*년(?!\s*[0-9]{1,2}\s*월)")


def _document_collection_form_year(item: Any) -> "tuple[str, str] | None":
    """Detect a closed 「서식명 + 연도」whole-target document collection.

    Only ``whole_target``/``record_list``/no-named-fields intents qualify —
    that topology is already accepted generically by the document collection
    validators, so this helper only decides *which selector* the compiler
    should build, never whether the intent shape itself is legal.
    """

    if (item.output.projection_mode != "whole_target"
            or item.output.shape != "record_list"
            or item.output.field_surfaces):
        return None
    if _INVESTMENT_JUDGMENT_FORM_PATTERN.search(item.target.surface) is None:
        return None
    years = {
        match.group(1)
        for match in _DOCUMENT_COLLECTION_YEAR_PATTERN.finditer(item.target.surface)
    }
    for expression in item.scope.target_period_expressions:
        years |= {
            match.group(1)
            for match in _DOCUMENT_COLLECTION_YEAR_PATTERN.finditer(expression)
        }
    if len(years) != 1:
        return None
    return (_INVESTMENT_JUDGMENT_FORM, next(iter(years)))


#: HCX가 관측한 「제목」「공시일」류 named_fields 표면(P9-019·P9-019b 실호출,
#: 각 field 하나씩 다르게 남는다).  `_document_collection_form_year` 가 받는
#: whole_target 토폴로지로 되돌릴 수 있는 필드만 허용한다 — 금액 등 다른
#: 필드가 섞이면 이 목록을 벗어나 손대지 않는다(fail-closed).
_INVESTMENT_JUDGMENT_LIST_FIELDS = {
    re.sub(r"\s+", "", surface)
    for surface in ("제목", "공시일", "공시 일자", "접수일", "보고서명", "접수번호")
}


class InvestmentJudgmentTitleDateRegrounder:
    """Recover the closed 「투자판단관련주요경영사항 제목·공시일 목록」 shape.

    HCX flattens this request into a ``document``- or ``event``-kind
    ``named_fields`` item whose field surfaces name only title/date-like
    attributes — P9-019 실호출(``document``-kind 변형):
    ``target.surface="공시한 투자판단 관련 주요경영사항"``,
    ``field_surfaces=["제목","공시일"]``; P9-019b: the title folds into
    ``target.surface="공시한 투자판단관련주요경영사항 제목"`` and only
    ``field_surfaces=["공시일"]`` remains.  A third real call (같은 질문,
    P9-019 issue #80 후속) instead emits ``target.kind="event"`` with the
    identical surface/field shape — the boundary already promotes the bare
    ``qualifier_surfaces=["2025년"]`` HCX attaches to the target into
    ``scope.target_period_expressions`` before this regrounder ever runs, so
    ``target.qualifier_surfaces`` is empty by the time this checks it either
    way.  `DocumentCollectionResolutionBackend` deliberately invents no
    per-document field value (issue #44 — it lists documents by *selector*,
    not by extracted fields), so none of these shapes ever reaches it.
    Recover the already-supported ``whole_target``/``record_list``/no-field
    topology instead of teaching the collection path to answer named scalar
    fields it was never meant to hold.

    The rewritten ``target.surface`` must stay a literal contiguous span of
    the question (`validate_semantic_intent_grounding` fail-closes on an
    invented one) — this only ever narrows to the exact 「투자판단 관련
    주요경영사항」/「투자판단관련주요경영사항」 substring already present
    inside the original (already-grounded) ``target.surface``, so the
    narrower span is grounded too.  A second company, a missing year, or any
    field outside the title/date vocabulary leaves the intent untouched.
    """

    #: CG-052 실측(이슈 #150) — 「YYYY년 M월 D일 정정한 투자판단 관련
    #: 주요경영사항은 무엇이 바뀌었어?」는 그 날짜를 target 수식어로
    #: 「2026년 3월 31일 정정」처럼 낸다.  `CorrectionLineageIntentRegrounder`
    #: (agent/stage1_v1_correction_regrounder.py, 다른 담당)는 「…정정공시」
    #: 문형만 잡아 「…정정한」은 못 잡는다 — 같은 파일을 고치지 않고, 이
    #: 리터럴 서식 하나만 겨냥해 `CorrectionLineageResolutionBackend`가
    #: 이미 받는 닫힌 모양(whole_target·no-field·target_period_expressions
    #: 하나)으로 되돌린다.
    _CORRECTION_DATE_QUALIFIER = re.compile(
        r"^((?:19|20)[0-9]{2}\s*년\s*[0-9]{1,2}\s*월\s*[0-9]{1,2}\s*일)"
        r"\s*(?:기재\s*)?정정$")

    def _correction_diff_reground(
            self, question: str, intent: SemanticIntent) -> "SemanticIntent | None":
        if (not isinstance(question, str)
                or len(intent.answer_items) != 1
                or len(intent.entities) != 1
                or intent.answer_groups or intent.premises
                or intent.unresolved_mentions):
            return None
        item = intent.answer_items[0]
        # 필드 표면이 질문 원문 활용형과 글자가 달라(예 "바뀐 내용" vs "무엇이
        # 바뀌었어") grounding을 통과하지 못하면, wire→intent 결속 단계가 이미
        # named_fields를 whole_target·field 없음으로 미리 손질해 넘길 수 있다
        # (CG-052 실측, 이슈 #150). 그 손질된 모양도 그대로 받는다 — 이 항목이
        # 가리키는 서식·날짜는 target/qualifier에 그대로 남아 있고, 이 메서드는
        # 어차피 field 내용을 쓰지 않는다.
        if (item.target.kind != "document"
                or item.operation != "retrieve"
                or item.output.projection_mode not in ("named_fields", "whole_target")
                or len(item.target.qualifier_surfaces) != 1
                or item.selection is not None
                or item.scope.as_of_expression is not None
                or item.scope.document_group_expression is not None
                or item.scope.target_period_expressions
                or item.scope.scope_qualifier_expressions):
            return None
        company = intent.entities[0]
        if (company.kind_hint != "company"
                or item.target.entity_refs != [company.entity_id]):
            return None
        match = self._CORRECTION_DATE_QUALIFIER.fullmatch(
            item.target.qualifier_surfaces[0].strip())
        if match is None:
            return None
        date_surface = match.group(1)
        if date_surface not in question:
            return None
        form_match = _INVESTMENT_JUDGMENT_FORM_PATTERN.search(item.target.surface)
        if form_match is None:
            return None
        payload = intent.model_dump(mode="python", warnings=False)
        answer_item = payload["answer_items"][0]
        answer_item["item_id"] = "item-1"
        answer_item["target"]["surface"] = form_match.group(0)
        answer_item["target"]["qualifier_surfaces"] = []
        answer_item["scope"]["target_period_expressions"] = [date_surface]
        answer_item["output"] = {
            "shape": "narrative", "projection_mode": "whole_target",
            "field_surfaces": [], "presentation": item.output.presentation,
        }
        return SemanticIntent.model_validate(payload, strict=True)

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        corrected = self._correction_diff_reground(question, intent)
        if corrected is not None:
            return corrected
        if (not isinstance(question, str)
                or len(intent.answer_items) != 1
                or intent.answer_groups or intent.premises
                or intent.unresolved_mentions):
            return intent
        item = intent.answer_items[0]
        if (item.target.kind not in ("document", "event")
                or item.operation != "retrieve"
                or item.output.projection_mode != "named_fields"
                or item.target.qualifier_surfaces
                or item.selection is not None
                or item.scope.as_of_expression is not None
                or item.scope.document_group_expression is not None
                or item.scope.scope_qualifier_expressions):
            return intent
        fields = list(item.output.field_surfaces)
        if any(re.sub(r"\s+", "", field) not in _INVESTMENT_JUDGMENT_LIST_FIELDS
               for field in fields):
            return intent
        if not fields and "제목" not in item.target.surface:
            return intent
        companies = [
            entity for entity in intent.entities
            if entity.kind_hint == "company"]
        if (len(companies) != 1
                or item.target.entity_refs != [companies[0].entity_id]):
            return intent
        match = _INVESTMENT_JUDGMENT_FORM_PATTERN.search(item.target.surface)
        if match is None:
            return intent
        form_span = match.group(0)
        years = {
            m.group(1) for m in
            _DOCUMENT_COLLECTION_YEAR_PATTERN.finditer(item.target.surface)
        }
        for expression in item.scope.target_period_expressions:
            years |= {
                m.group(1)
                for m in _DOCUMENT_COLLECTION_YEAR_PATTERN.finditer(expression)
            }
        if len(years) != 1:
            return intent
        payload = intent.model_dump(mode="python", warnings=False)
        # ``DocumentCollectionResolutionBackend`` only accepts
        # ``document``/``topic`` targets (issue #44 contract) — an
        # ``event``-kind input recovered here must land on that same
        # already-supported path, not a new one, so the kind is normalized
        # to ``document`` alongside the surface/output rewrite below.
        payload["answer_items"][0]["target"]["kind"] = "document"
        payload["answer_items"][0]["target"]["surface"] = form_span
        payload["answer_items"][0]["output"].update({
            "shape": "record_list",
            "projection_mode": "whole_target",
            "field_surfaces": [],
        })
        return SemanticIntent.model_validate(payload, strict=True)


class DocumentCollectionResolutionBackend:
    """회사의 문서 묶음을 통째로 요청하는 항목을 확정한다."""

    def __init__(
            self, canonical: Any, *, canonical_build_id: str,
            resolver_version: str, reference_date: date, corpus_cutoff: str,
            ) -> None:
        self.canonical = canonical
        self.canonical_build_id = canonical_build_id
        self.resolver_version = resolver_version
        self.reference_date = reference_date
        self.corpus_cutoff = corpus_cutoff

    def _selected_latest_periodic_document(
            self, company: Any, item: Any,
            ) -> "Any | None":
        """Return one canonical latest periodic filing for an explicit target.

        This is intentionally a resolver-only grammar check.  The compiler
        receives the resulting document identity/proof and never reparses a
        user surface to decide whether a collection should be narrowed.
        """

        if item.target.kind != "document":
            return None
        from agent.periodic_document_preflight import (
            PeriodicDocumentPreflight, PeriodicDocumentPreflightError,
            parse_periodic_expression)

        try:
            selector = parse_periodic_expression(item.target.surface)
        except PeriodicDocumentPreflightError:
            return None
        if not selector.latest:
            return None
        resolution = PeriodicDocumentPreflight(
            self.canonical).resolve_periodic_document(
                corp_code=company.corp_code, as_of=self.corpus_cutoff,
                target_expression=item.target.surface)
        if resolution.status != "resolved" or resolution.candidate is None:
            return None
        return resolution.candidate

    def _selected_latest_annual_business_document(
            self, company: Any, item: Any, source_intent: SemanticIntent,
            ) -> "Any | None":
        """Select one annual snapshot for an unqualified whole business request.

        ``사업 내용`` asks for a company overview, whose segment and product
        axes must describe the same reporting point.  This deliberately does
        not reinterpret a named period, document, qualifier, comparison, or
        multi-item request as "latest".
        """

        if not (
                len(source_intent.answer_items) == 1
                and item.target.kind == "topic"
                and re.sub(r"\s+", "", item.target.surface)
                in {"사업내용", "사업의내용"}
                and item.operation == "retrieve"
                and item.output.projection_mode == "whole_target"
                and not item.output.field_surfaces
                and item.selection is None
                and not item.scope.target_period_expressions
                and item.scope.as_of_expression is None
                and item.scope.document_group_expression is None
                and not item.scope.scope_qualifier_expressions):
            return None
        from agent.periodic_document_preflight import PeriodicDocumentPreflight

        resolution = PeriodicDocumentPreflight(
            self.canonical).resolve_periodic_document(
                corp_code=company.corp_code, as_of=self.corpus_cutoff,
                target_expression="최근 사업보고서")
        if resolution.status != "resolved" or resolution.candidate is None:
            return None
        return resolution.candidate

    def resolve(
            self, *, question_id: str, question: str,
            source_intent: SemanticIntent,
            ) -> "dict[str, Any] | None":
        # "정정된 계약에서 무엇이 바뀌었나" is a correction-lineage
        # question, not a request for every issuer document.  If lineage cannot
        # identify one event/correction, the clarification backend must ask for
        # that coordinate instead of this broad collection swallowing it.
        if (re.search(r"정정|기재정정|수정", question)
                and re.search(r"바뀌|바뀐|바꾼|달라|변경|차이", question)):
            return None
        items: list[dict[str, Any]] = []
        for index, item in enumerate(source_intent.answer_items, start=1):
            is_whole_collection = item.output.projection_mode == "whole_target"
            is_recent_investment_narrative = (
                item.target.kind == "topic"
                and item.output.projection_mode == "named_fields"
                and item.output.shape == "narrative"
                and len(item.output.field_surfaces) == 1
                and re.sub(r"\s+", "", item.target.surface) == "최근투자계획")
            if (item.target.kind not in {"document", "topic"}
                    or not (is_whole_collection or is_recent_investment_narrative)):
                return None
            company = _single_company(source_intent, item, self.canonical)
            if company is None:
                return None
            selected_periodic = self._selected_latest_periodic_document(
                company, item)
            if selected_periodic is None:
                selected_periodic = self._selected_latest_annual_business_document(
                    company, item, source_intent)
            item_id = f"item-{index}"
            topic_query = None
            if item.target.kind == "topic":
                compact = re.sub(r"\s+", "", item.target.surface)
                if compact in {"투자계획", "최근투자계획"}:
                    topic_query = "투자 계획"
                elif compact in {"사업내용", "사업의내용"}:
                    # The narrow whole-target path above binds the two
                    # business axes to one latest annual filing.  Qualified
                    # requests keep their original document collection.
                    topic_query = item.target.surface
                elif _document_collection_form_year(item) is not None:
                    # A named-form + year whole-target request (issue #44)
                    # is a structural document listing, not a narrative
                    # search — leave topic_query unset so the compiler binds
                    # a plain ``find`` task instead of FTS.
                    pass
                else:
                    return None
            items.append({
                "item_id": item_id,
                "target_surface": item.target.surface,
                "projection_mode": item.output.projection_mode,
                "resolution": {
                    "kind": "document_collection",
                    "corp_code": company.corp_code,
                    "corp_name": company.corp_name,
                    "as_of": self.corpus_cutoff,
                    "selector_proof_ref": (
                        f"source-document:periodic_{selected_periodic.rcept_no}"
                        if selected_periodic is not None else
                        f"source-document-collection:{company.corp_code}"
                        f":{self.corpus_cutoff}"),
                    "retrieval_query": topic_query,
                    "recent_selection": (
                        selected_periodic is not None
                        or (compact == "최근투자계획"
                            if item.target.kind == "topic" else False)),
                    "selected_document_id": (
                        f"periodic_{selected_periodic.rcept_no}"
                        if selected_periodic is not None else None),
                    "selected_receipt_no": (
                        selected_periodic.rcept_no
                        if selected_periodic is not None else None),
                    "selected_document_proof": (
                        {
                            "source_receipt": selected_periodic.rcept_no,
                            "proof_ref": (
                                "source-document:"
                                f"periodic_{selected_periodic.rcept_no}"),
                        } if selected_periodic is not None else None),
                },
                "applied_defaults": [],
                "field_proofs": _field_proofs(
                    item_id, list(item.output.field_surfaces)),
            })
        return _authority(
            question_id=question_id, source_intent=source_intent,
            canonical_build_id=self.canonical_build_id,
            resolver_version=self.resolver_version,
            reference_date=self.reference_date,
            corpus_cutoff=self.corpus_cutoff, items=items)


class DocumentVersionHistoryResolutionBackend:
    """Bind one periodic filing's version request to canonical lineage."""

    def __init__(self, canonical: Any, *, canonical_build_id: str,
                 resolver_version: str, reference_date: date,
                 corpus_cutoff: str) -> None:
        self.canonical = canonical
        self.canonical_build_id = canonical_build_id
        self.resolver_version = resolver_version
        self.reference_date = reference_date
        self.corpus_cutoff = corpus_cutoff

    def resolve(self, *, question_id: str, question: str,
                source_intent: SemanticIntent) -> "dict[str, Any] | None":
        if len(source_intent.answer_items) != 1 or len(source_intent.premises) != 1:
            return None
        item = source_intent.answer_items[0]
        premise = source_intent.premises[0]
        match = re.fullmatch(
            r"(20[0-9]{2})년\s*(1\s*분기보고서|반기보고서|"
            r"3\s*분기보고서|사업보고서)",
            item.target.surface)
        plain_annual = item.target.surface == "사업보고서"
        if (
                (match is None and not plain_annual)
                or item.target.kind != "document"
                or item.operation != "retrieve" or item.output.shape != "scalar"
                or item.output.projection_mode != "named_fields"
                or len(item.output.field_surfaces) != 1
                or re.search(
                    r"정정|버전|변경\s*이력|원본",
                    item.output.field_surfaces[0]) is None
                or premise.kind != "existence"
                or premise.applies_to_item_ids != [item.item_id]
        ):
            return None
        company = _single_company(source_intent, item, self.canonical)
        if company is None:
            return None
        if match is not None:
            base_year = int(match.group(1))
            base_month = {
                "1분기보고서": 3, "반기보고서": 6,
                "3분기보고서": 9, "사업보고서": 12,
            }[re.sub(r"\s+", "", match.group(2))]
            seeds = [row for row in self.canonical.documents(
                corp_code=company.corp_code, as_of=self.corpus_cutoff,
                doc_group="periodic")
                if getattr(row, "base_year", None) == base_year
                and getattr(row, "base_month", None) == base_month
                and not getattr(row, "is_correction", True)]
            if len(seeds) != 1:
                return None
            root_receipt = seeds[0].rcept_no
            lineage = self.canonical.resolve_document_version(
                root_receipt, as_of=self.corpus_cutoff)
        else:
            from agent.date_surface import question_date_surfaces
            days = {
                f"{year:04d}{month:02d}{day:02d}"
                for year, month, day in question_date_surfaces(question)
                if month is not None and day is not None
            }
            if len(days) != 1:
                return None
            filing_day = next(iter(days))
            candidates = [row for row in self.canonical.documents(
                corp_code=company.corp_code, as_of=self.corpus_cutoff,
                doc_group="periodic")
                if str(getattr(row, "rcept_dt", "")) == filing_day
                and "사업보고서" in str(getattr(row, "report_nm", ""))]
            lineages: dict[tuple[str, ...], Any] = {}
            for candidate in candidates:
                value = self.canonical.resolve_document_version(
                    candidate.rcept_no, as_of=self.corpus_cutoff)
                members_key = tuple(getattr(value, "members", ()) or ())
                if getattr(value, "status", None) == "ok" and members_key:
                    lineages[members_key] = value
            if len(lineages) != 1:
                return None
            members_key, lineage = next(iter(lineages.items()))
            root_receipt = members_key[0]
            roots = [row for row in self.canonical.documents(
                corp_code=company.corp_code, as_of=self.corpus_cutoff,
                doc_group="periodic")
                if getattr(row, "rcept_no", None) == root_receipt]
            if len(roots) != 1:
                return None
            base_year = int(getattr(roots[0], "base_year", 0) or 0)
            if base_year < 2000:
                return None
        if getattr(lineage, "status", None) != "ok" or not lineage.selected:
            return None
        members = list(getattr(lineage, "members", ()) or ())
        if not members or root_receipt not in members:
            return None
        item_id = item.item_id
        return _authority(
            question_id=question_id, source_intent=source_intent,
            canonical_build_id=self.canonical_build_id,
            resolver_version=self.resolver_version,
            reference_date=self.reference_date, corpus_cutoff=self.corpus_cutoff,
            items=[{
                "item_id": item_id, "target_surface": item.target.surface,
                "projection_mode": "named_fields",
                "resolution": {
                    "kind": "document_version_history",
                    "corp_code": company.corp_code, "corp_name": company.corp_name,
                    "base_year": base_year, "root_receipt": root_receipt,
                    "latest_receipt": lineage.selected, "lineage_receipts": members,
                    "selector_proof": {"source_receipt": root_receipt,
                                       "proof_ref": f"source-document:{root_receipt}"},
                    "lineage_proof_ref": f"source-lineage:{root_receipt}",
                },
                "applied_defaults": [],
                "field_proofs": _field_proofs(
                    item_id, list(item.output.field_surfaces)),
            }])


__all__ = [
    "DocumentCollectionResolutionBackend",
    "DocumentVersionHistoryResolutionBackend",
    "PeriodicNarrativeComparisonBackend",
    "PeriodicNarrativeResolutionBackend",
    "SelectedEventResolutionBackend",
]


# ---------------------------------------------------------------------------
# 서식 필드 증거 선택
# ---------------------------------------------------------------------------
#
# 공시 서식의 라벨은 닫힌 어휘다 — `3. 계약상대`, `2. 해지내역 > 해지금액(원)`,
# `5. 해지 주요사유`.  의미가 말한 표면("계약 상대", "해지 사유")을 그 라벨에
# 맞추면 증거 좌표(`path`·`locator`·`evidence_id`·`source_file_id`)가 따라온다.
#
# 맞춤은 **정규화 후 포함 관계**로만 한다.  번호와 단위 괄호를 떼고 공백을 지운 뒤,
# 한쪽이 다른 쪽을 품으면 후보다.  후보가 **정확히 하나일 때만** 쓴다 — 둘 이상이면
# 어느 칸을 말하는지 의미만으로 정해지지 않으므로 물러난다.

import re as _re

# `3. 계약상대` 뿐 아니라 `- 해지계약명` 처럼 번호 없이 글머리표만 붙는 라벨도 있다.
_LABEL_NUMBER = _re.compile(r"^\s*[-–—·]?\s*(?:\d+\s*[.)]\s*)?")
_LABEL_UNIT = _re.compile(r"\([^)]*\)")


def normalize_form_label(value: str) -> str:
    """서식 라벨·의미 표면을 비교 가능한 형태로 줄인다."""

    text = _LABEL_NUMBER.sub("", value or "")
    text = _LABEL_UNIT.sub("", text)
    text = text.replace("ㆍ", "").replace("·", "")
    return _re.sub(r"\s+", "", text).strip()


def _label_tail(path: str) -> str:
    """`2. 해지내역 > 해지금액(원)` 처럼 계층 라벨이면 마지막 칸을 본다."""

    return path.split(">")[-1] if ">" in path else path


def select_field_evidence(
        canonical: Any, *, corp_code: str, receipt_no: str, surface: str,
        as_of: str) -> "Any | None":
    """의미 표면이 가리키는 서식 필드 행 하나. 못 좁히면 ``None``."""

    rows = [
        row for row in canonical.fields(as_of=as_of, corp_code=corp_code)
        if row.rcept_no == receipt_no and row.evidence_id
    ]
    return _select_field_evidence_rows(rows, surface)


def _select_field_evidence_rows(
        source_rows: "list[Any]", surface: str) -> "Any | None":
    """Select one evidence row from an already bounded document inventory."""

    want = normalize_form_label(surface)
    # User-facing reason wording is open, while the exchange form label is a
    # closed `사유` vocabulary.  This is a field-ontology equivalence, not a
    # question or company special case.
    if "해지" in want and "이유" in want:
        want = "해지사유"
    if not want:
        return None
    rows = [
        (row, normalize_form_label(_label_tail(row.path)),
         normalize_form_label(row.path))
        for row in source_rows if row.evidence_id
    ]
    # 좁은 규칙부터 시도하고, **정확히 하나로 좁혀지는 첫 단계**를 쓴다.
    # 느슨한 규칙을 먼저 쓰면 우연한 일치가 정확한 일치를 덮는다.
    for rule in (_exact_label, _contained_label, _subsequence_label):
        matches = [
            row for row, tail, full in rows
            if rule(want, tail) or rule(want, full)
        ]
        if len(matches) == 1:
            return matches[0]
    # Some conditions are disclosed only inside a narrative form field rather
    # than in a dedicated label.  Require the question surface to ask for an
    # effectiveness condition and the document value to contain the same two
    # semantic anchors; use it only when exactly one row survives.
    if "효력" in want and "조건" in want:
        matches = [
            row for row, _, _ in rows
            if "효력" in normalize_form_label(
                str(getattr(row, "value_masked", "") or ""))
            and "발생" in normalize_form_label(
                str(getattr(row, "value_masked", "") or ""))
        ]
        if len(matches) == 1:
            return matches[0]
    return None


def _exact_label(want: str, candidate: str) -> bool:
    return bool(candidate) and want == candidate


def _contained_label(want: str, candidate: str) -> bool:
    return bool(candidate) and (want in candidate or candidate in want)


def _subsequence_label(want: str, candidate: str) -> bool:
    """`해지사유` 가 `해지주요사유` 를 가리키는 것처럼 수식어가 낀 경우.

    글자 순서를 지키는 부분수열만 인정한다. 느슨하므로 앞 단계가 실패했고
    이 단계에서 후보가 하나일 때만 쓰인다.
    """

    if not candidate or len(want) < 2:
        return False
    position = 0
    for char in candidate:
        if position < len(want) and char == want[position]:
            position += 1
    return position == len(want)


def counterparty_surface(
        intent: SemanticIntent, canonical: Any) -> "str | None":
    """계약 상대 표면. 못 가리면 ``None``.

    모델은 `Ford`·`포드`·`Freudenberg` 를 대개 `kind_hint="company"` 로 적는다.
    `counterparty` 힌트만 찾으면 아무것도 못 찾고, 상대 필터 없이 그 회사의 계약을
    전부 긁게 된다(관측: 후보 12~13건).

    **힌트 대신 조회로 가른다.**  코퍼스에 공시 주체로 등록된 이름은 canonical
    회사 해석이 되고, 계약 상대는 되지 않는다.  이건 추측이 아니라 판정이다.
    해석되지 않는 이름이 정확히 하나일 때만 상대로 본다 — 둘이면 어느 쪽인지
    의미만으로 정해지지 않으므로 물러난다.
    """

    explicit = [
        entity.surface for entity in intent.entities
        if entity.kind_hint == "counterparty" and entity.surface.strip()
    ]
    if len(explicit) == 1:
        return explicit[0]
    if explicit:
        return None
    event_surfaces = [
        re.sub(r"\s*계약\s*$", "", entity.surface).strip()
        for entity in intent.entities
        if entity.kind_hint == "event" and entity.surface.strip()
        and not canonical.resolve_company(entity.surface)
    ]
    event_surfaces = [surface for surface in event_surfaces if surface]
    if len(event_surfaces) == 1:
        return event_surfaces[0]
    unresolved = [
        entity.surface for entity in intent.entities
        if entity.kind_hint == "company" and entity.surface.strip()
        and not canonical.resolve_company(entity.surface)
    ]
    return unresolved[0] if len(unresolved) == 1 else None


def qualifier_event_window(
        item: Any, *, reference_date: date) -> "tuple[str | None, str | None]":
    """`target.qualifier_surfaces` 에 든 날짜를 사건 조회 범위로.

    모델은 「25년 12월 17일 기준」 같은 시점을 scope 가 아니라 target 수식어에
    넣는 판이 있다.  scope 만 보면 그 단서를 통째로 버린다.  날짜로 파싱되는
    수식어가 정확히 하나일 때만 쓰고, 아니면 아무것도 하지 않는다.
    """

    windows = []
    for surface in getattr(item.target, "qualifier_surfaces", ()) or ():
        start, end, error = _target_date_range(
            surface, reference_date=reference_date)
        if error is None and start and end:
            windows.append((start, end))
    return windows[0] if len(windows) == 1 else (None, None)


class DocumentAttributeEvidenceBackend:
    """서식 필드 하나하나를 증거 좌표로 확정한다.

    문서가 둘이면 접수 순서로 `earlier`/`later` 역할을 준다.  역할은 상대적이며
    문항 식별자나 알려진 접수번호를 경로 키로 쓰지 않는다.
    """

    def __init__(
            self, canonical: Any, *, canonical_build_id: str,
            resolver_version: str, reference_date: date, corpus_cutoff: str,
            event_preflight: Any | None = None,
            ) -> None:
        from agent.event_preflight import CanonicalEventKeyPreflight

        self.canonical = canonical
        self.preflight = (
            event_preflight if event_preflight is not None
            else CanonicalEventKeyPreflight(
                canonical, corpus_cutoff=corpus_cutoff))
        if not callable(getattr(self.preflight, "resolve_event_key", None)):
            raise TypeError(
                "document attribute backend preflight 계약이 잘못되었습니다")
        self.canonical_build_id = canonical_build_id
        self.resolver_version = resolver_version
        self.reference_date = reference_date
        self.corpus_cutoff = corpus_cutoff

    def _event_receipts(
            self, intent: SemanticIntent, item: Any) -> "list[str]":
        """항목이 가리키는 사건의 관측 접수번호를 시간 순으로."""

        company = _single_company(intent, item, self.canonical)
        if company is None:
            return []
        resolution = self.preflight.resolve_event_key(
            corp_code=company.corp_code, as_of=self.corpus_cutoff,
            contract_name=item.target.surface)
        if resolution.status != "resolved" or len(resolution.candidates) != 1:
            return []
        timeline = self.canonical.event_timeline(
            as_of=self.corpus_cutoff,
            event_key=resolution.candidates[0].event_key)
        if timeline is None:
            return []
        receipts = sorted({
            getattr(row, "rcept_no", None)
            for row in tuple(getattr(timeline, "observations", ()) or ())
            if getattr(row, "rcept_no", None)
        })
        return list(receipts)

    def _unique_company_field(
            self, corp_code: str, surface: str) -> "tuple[str, Any] | None":
        """Find one exchange document carrying one uniquely matched field.

        This is the fail-closed fallback for a broad user target such as
        `계약`: event identity may be ambiguous, while the requested attribute
        itself occurs in exactly one disclosure for the issuer.
        """

        grouped: dict[str, list[Any]] = {}
        for row in self.canonical.fields(
                as_of=self.corpus_cutoff, corp_code=corp_code):
            if row.evidence_id:
                grouped.setdefault(row.rcept_no, []).append(row)
        matches = [
            (receipt, selected)
            for receipt, rows in grouped.items()
            if (selected := _select_field_evidence_rows(rows, surface)) is not None
        ]
        return matches[0] if len(matches) == 1 else None

    def _unique_event_field_pair(
            self, corp_code: str, surfaces: "list[str]",
            ) -> "list[tuple[str, Any]] | None":
        """Resolve two attributes only when one canonical event owns both."""

        if len(surfaces) != 2:
            return None
        grouped: dict[str, list[Any]] = {}
        for row in self.canonical.fields(
                as_of=self.corpus_cutoff, corp_code=corp_code):
            if row.evidence_id:
                grouped.setdefault(row.rcept_no, []).append(row)
        by_event: dict[str, list[str]] = {}
        for receipt in grouped:
            event_key = self.canonical.event_of(receipt)
            if event_key:
                by_event.setdefault(event_key, []).append(receipt)
        candidates: list[list[tuple[str, Any]]] = []
        for receipts in by_event.values():
            picked: list[tuple[str, Any]] = []
            for surface in surfaces:
                matches = [
                    (receipt, selected)
                    for receipt in receipts
                    if (selected := _select_field_evidence_rows(
                        grouped[receipt], surface)) is not None
                ]
                if len(matches) != 1:
                    break
                picked.append(matches[0])
            if len(picked) == 2 and picked[0][0] != picked[1][0]:
                candidates.append(picked)
        return candidates[0] if len(candidates) == 1 else None

    def resolve(
            self, *, question_id: str, question: str,
            source_intent: SemanticIntent,
            ) -> "dict[str, Any] | None":
        del question
        items: list[dict[str, Any]] = []
        anchored_receipts: list[str] = []
        joint_selection: list[tuple[str, Any]] | None = None
        # Reject unsupported target kinds before the issuer-wide field scan.
        # The loop below already rejects these items; preflight changes only
        # the work performed on the way to that same not-applicable result.
        if any(item.target.kind not in {"event", "document"}
               for item in source_intent.answer_items):
            return None
        if len(source_intent.answer_items) == 2:
            first = source_intent.answer_items[0]
            company = _single_company(source_intent, first, self.canonical)
            pair_surfaces = [
                item.output.field_surfaces[0]
                for item in source_intent.answer_items
                if len(item.output.field_surfaces) == 1
            ]
            if company is not None and len(pair_surfaces) == 2:
                joint_selection = self._unique_event_field_pair(
                    company.corp_code, pair_surfaces)
        for index, item in enumerate(source_intent.answer_items, start=1):
            if item.target.kind not in {"event", "document"}:
                return None
            surfaces = list(item.output.field_surfaces)
            if len(surfaces) != 1:
                return None
            company = _single_company(source_intent, item, self.canonical)
            if company is None:
                return None
            receipts = self._event_receipts(source_intent, item)
            if not receipts and anchored_receipts:
                receipts = list(anchored_receipts)
            chosen = None
            if joint_selection is not None:
                receipt, row = joint_selection[index - 1]
                ordered_receipts = sorted(receipt for receipt, _ in joint_selection)
                receipts = ordered_receipts
                chosen = (ordered_receipts.index(receipt), receipt, row)
            for position, receipt in enumerate(receipts):
                if chosen is not None:
                    break
                row = select_field_evidence(
                    self.canonical, corp_code=company.corp_code,
                    receipt_no=receipt, surface=surfaces[0],
                    as_of=self.corpus_cutoff)
                if row is not None:
                    chosen = (position, receipt, row)
            if chosen is None:
                unique = self._unique_company_field(
                    company.corp_code, surfaces[0])
                if unique is not None:
                    receipt, row = unique
                    chosen = (0, receipt, row)
            if chosen is None:
                return None
            position, receipt, row = chosen
            if not anchored_receipts:
                event_key = self.canonical.event_of(receipt)
                if event_key:
                    timeline = self.canonical.event_timeline(
                        as_of=self.corpus_cutoff, event_key=event_key)
                    if timeline is not None:
                        anchored_receipts = sorted({
                            getattr(observation, "rcept_no", None)
                            for observation in timeline
                            if getattr(observation, "rcept_no", None)
                        })
            role = "later" if position == len(receipts) - 1 else "earlier"
            item_id = f"item-{index}"
            items.append({
                "item_id": item_id,
                "target_surface": item.target.surface,
                "projection_mode": item.output.projection_mode,
                "resolution": {
                    "kind": "document_attribute_evidence",
                    "evidence": {
                        "issuer_corp_code": row.corp_code,
                        "issuer_corp_name": row.corp_name,
                        "document_role": role,
                        "doc_id": row.doc_id,
                        "receipt_no": receipt,
                        "path": row.path,
                        "locator": row.locator,
                        "source_file_id": row.source_file_id,
                        "evidence_id": row.evidence_id,
                        "value_kind": "text",
                    },
                },
                "applied_defaults": [],
                "field_proofs": _field_proofs(item_id, surfaces),
            })
        if len(items) == 2:
            ordered = sorted(
                items,
                key=lambda row: row["resolution"]["evidence"]["receipt_no"],
            )
            first_receipt = ordered[0]["resolution"]["evidence"]["receipt_no"]
            second_receipt = ordered[1]["resolution"]["evidence"]["receipt_no"]
            if first_receipt[:8] == second_receipt[:8]:
                return None
            ordered[0]["resolution"]["evidence"]["document_role"] = "earlier"
            ordered[1]["resolution"]["evidence"]["document_role"] = "later"
        return _authority(
            question_id=question_id, source_intent=source_intent,
            canonical_build_id=self.canonical_build_id,
            resolver_version=self.resolver_version,
            reference_date=self.reference_date,
            corpus_cutoff=self.corpus_cutoff, items=items)


__all__ += ["DocumentAttributeEvidenceBackend", "select_field_evidence",
            "normalize_form_label"]


class TerminationReportedStatusBackend:
    """해지 공시는 확인되나 **어느 원계약의 해지인지 특정할 수 없는** 상태를 낸다.

    LG에너지솔루션·Ford 처럼 같은 날 같은 계약명·상대로 체결 공시가 둘 나가고,
    나중에 해지 공시가 하나만 나오는 경우가 실제로 있다.  해지 공시에는 어느
    계약을 해지했는지 적히지 않는다 — 계약명과 상대가 둘 다 같기 때문이다.

    이때 하나를 골라 답하면 **틀릴 확률이 절반**이고, 되물으면 사용자도 답할 수
    없다(공시에 구분할 정보가 없다).  그래서 「해지 공시상 해지」를 답하되 원계약
    후보 전부를 근거로 달고, 유래 불명을 `ambiguous_event_origin` 한계로 기록한다.
    """

    TERMINATION_FORM = "단일판매공급계약해지"
    CONCLUSION_FORM = "단일판매공급계약체결"

    def __init__(
            self, canonical: Any, *, canonical_build_id: str,
            resolver_version: str, reference_date: date, corpus_cutoff: str,
            ) -> None:
        self.canonical = canonical
        self.canonical_build_id = canonical_build_id
        self.resolver_version = resolver_version
        self.reference_date = reference_date
        self.corpus_cutoff = corpus_cutoff

    def _documents(self, corp_code: str) -> "dict[str, dict[str, Any]]":
        """접수번호별 서식·라벨값. 한 번만 훑는다."""

        found: dict[str, dict[str, Any]] = {}
        for row in self.canonical.fields(
                as_of=self.corpus_cutoff, corp_code=corp_code):
            entry = found.setdefault(
                row.rcept_no,
                {"form": row.form, "is_correction": bool(row.is_correction),
                 "labels": {}})
            leaf = normalize_form_label(_label_tail(row.path))
            if leaf:
                entry["labels"][leaf] = (row.value_masked or "").strip()
        return found

    def _matches_counterparty(
            self, labels: "dict[str, Any]", counterparty: str) -> bool:
        """공시의 계약상대 값과 질문의 상대 표면이 같은 상대인가.

        질문은 「포드」로 부르고 공시는 `Ford Motor Company` 로 적는다.  글자로는
        안 겹치므로 `event_preflight` 가 쓰는 **같은 음차 사전**을 통해 넓힌다.
        사전을 두 벌 만들지 않는다.
        """

        from agent.event_preflight import _counterparty_aliases, _key

        value = labels.get(normalize_form_label("계약상대"), "")
        if not value:
            return False
        got = _key(value)
        candidates = {_key(counterparty)}
        candidates.update(
            _key(latin)
            for latin in _counterparty_aliases().get(_key(counterparty), ()))
        return any(want and want in got for want in candidates)

    def _single_reported_status_item(
            self, intent: SemanticIntent,
            ) -> "tuple[Any, str, bool] | None":
        """Accept one exact-date or cutoff-default status read.

        The compiler owns the closed semantic topology.  The backend only
        checks the resolver prerequisites needed to bind that topology to one
        termination report.  An explicit date must occur in both ``as_of`` and
        the event qualifier.  A scalar status predicate with no temporal axis
        is evaluated at corpus cutoff; the returned status receipt remains the
        actual canonical observation and is never rewritten to the cutoff.
        """
        if len(intent.answer_items) != 1:
            return None
        item = intent.answer_items[0]
        if (item.scope.as_of_expression is None
                and not item.target.qualifier_surfaces
                and not item.scope.target_period_expressions
                and item.output.shape == "scalar"
                and len(item.output.field_surfaces) == 1
                and re.search(
                    r"상태|유효|살아\s*있|끝난|해지",
                    item.output.field_surfaces[0]) is not None):
            return item, self.corpus_cutoff, False
        if (
                item.scope.as_of_expression is None
                or len(item.target.qualifier_surfaces) != 1
                or item.scope.as_of_expression != item.target.qualifier_surfaces[0]
        ):
            return None
        as_of, as_of_end, error = _target_date_range(
            item.scope.as_of_expression, reference_date=self.reference_date)
        if error is not None or as_of is None or as_of != as_of_end:
            return None
        return item, as_of, True

    def resolve(
            self, *, question_id: str, question: str,
            source_intent: SemanticIntent,
            ) -> "dict[str, Any] | None":
        del question
        accepted = self._single_reported_status_item(source_intent)
        if accepted is None:
            return None
        source_item, as_of, exact_observation = accepted
        counterparty = counterparty_surface(source_intent, self.canonical)
        if counterparty is None:
            return None
        items_out: list[dict[str, Any]] = []
        for index, item in enumerate(source_intent.answer_items, start=1):
            if item is not source_item:
                return None
            company = _single_company(source_intent, item, self.canonical)
            if company is None:
                # A status request may name only the external counterparty.
                # Resolve its issuer role from canonical contract fields; do
                # not assume the party itself is a listed issuer.
                from agent.planner_preflight import CanonicalSelectorRolePreflight
                issuer = CanonicalSelectorRolePreflight(
                    self.canonical, corpus_cutoff=self.corpus_cutoff,
                    cache_root=(Path(__file__).resolve().parent.parent
                                / "out/serving/selector_roles"),
                ).resolve_selector_issuer(counterparty, role="counterparty")
                if issuer.status != "resolved" or len(issuer.candidates) != 1:
                    return None
                candidate = issuer.candidates[0]
                company = type("_Company", (), {
                    "corp_code": candidate.corp_code,
                    "corp_name": candidate.corp_name,
                })()
            documents = self._documents(company.corp_code)

            terminations = [
                (receipt, entry) for receipt, entry in documents.items()
                if normalize_form_label(entry["form"]) == self.TERMINATION_FORM
                and self._matches_counterparty(entry["labels"], counterparty)
            ]
            if len(terminations) != 1:
                return None
            status_receipt, termination = terminations[0]
            if ((exact_observation and status_receipt[:8] != as_of)
                    or (not exact_observation and status_receipt[:8] > as_of)):
                return None
            contract_name = termination["labels"].get(
                normalize_form_label("해지계약명"), "")
            if not contract_name:
                return None

            # 정정본은 원계약이 아니다. 서식이 같아서 함께 잡히므로 걸러낸다.
            originals = sorted(
                receipt for receipt, entry in documents.items()
                if normalize_form_label(entry["form"]) == self.CONCLUSION_FORM
                and not entry["is_correction"]
                and self._matches_counterparty(entry["labels"], counterparty)
                and normalize_form_label(entry["labels"].get(
                    normalize_form_label("체결계약명"), "")) ==
                normalize_form_label(contract_name)
            )
            # 원계약이 하나면 유래가 분명하므로 이 백엔드가 낼 상태가 아니다.
            if len(originals) < 2:
                return None
            event_key = self.canonical.event_of(status_receipt)
            if not event_key:
                return None

            item_id = f"item-{index}"
            from agent.stage1_v1_public_event_selector import (
                select_public_event_selector,
            )
            public_selector = select_public_event_selector(
                self.canonical, as_of=self.corpus_cutoff,
                receipt=status_receipt, intent=source_intent, item=item,
                counterparty_surface=counterparty,
            )
            resolution_payload = {
                "kind": "termination_reported_status",
                "issuer_corp_code": company.corp_code,
                "issuer_corp_name": company.corp_name,
                "status_receipt": status_receipt,
                "event_key": event_key,
                "event_key_proof": {
                    "source_receipt": status_receipt,
                    "proof_ref": f"source-event-key:{status_receipt}",
                },
                "status_proof": {
                    "source_receipt": status_receipt,
                    "proof_ref": f"source-status:{status_receipt}",
                },
                "identity_provenance": {
                    "code": "ambiguous_event_origin",
                    "family": "identity_lineage",
                    "detail": (
                        "termination report cannot be attributed to either "
                        "original contract"),
                    "evidence_refs": [f"source-identity:{status_receipt}"],
                    "original_receipts": list(originals),
                },
            }
            if not public_selector.empty:
                resolution_payload["public_selector"] = public_selector.as_payload()
            items_out.append({
                "item_id": item_id,
                "target_surface": item.target.surface,
                "projection_mode": item.output.projection_mode,
                "resolution": resolution_payload,
                "applied_defaults": [],
                "field_proofs": _field_proofs(
                    item_id, list(item.output.field_surfaces)),
            })
        return _authority(
            question_id=question_id, source_intent=source_intent,
            canonical_build_id=self.canonical_build_id,
            resolver_version=self.resolver_version,
            reference_date=self.reference_date,
            corpus_cutoff=self.corpus_cutoff, items=items_out)


__all__ += ["TerminationReportedStatusBackend"]


def _document_id(receipt_no: str, doc_group: str = "exchange") -> str:
    return f"{doc_group}_{receipt_no}"


class SameDayDocumentCandidatesBackend:
    """같은 날 접수된 해지·정정 두 건을 **순서를 주장하지 않고** 함께 낸다.

    코퍼스는 접수일까지만 남기고 시각을 남기지 않는다.  그래서 같은 날 두 건이
    있을 때 어느 것이 먼저인지 알 수 없다.  하나를 골라 「최신」이라고 하면 근거
    없는 주장이 되므로, 둘을 함께 내고 순서 불명을 한계로 기록한다.
    """

    def __init__(
            self, canonical: Any, *, canonical_build_id: str,
            resolver_version: str, reference_date: date, corpus_cutoff: str,
            ) -> None:
        self.canonical = canonical
        self.canonical_build_id = canonical_build_id
        self.resolver_version = resolver_version
        self.reference_date = reference_date
        self.corpus_cutoff = corpus_cutoff

    def _same_day_pair(
            self, corp_code: str, counterparty: str,
            ) -> "tuple[str, dict[str, str]] | None":
        by_day: dict[str, dict[str, str]] = {}
        matcher = TerminationReportedStatusBackend(
            self.canonical, canonical_build_id=self.canonical_build_id,
            resolver_version=self.resolver_version,
            reference_date=self.reference_date,
            corpus_cutoff=self.corpus_cutoff)
        for receipt, entry in matcher._documents(corp_code).items():
            if not matcher._matches_counterparty(entry["labels"], counterparty):
                continue
            if normalize_form_label(entry["form"]) == matcher.TERMINATION_FORM:
                kind = "termination"
            elif entry["is_correction"]:
                kind = "correction"
            else:
                continue
            day = by_day.setdefault(receipt[:8], {})
            if kind in day:                      # 같은 종류가 둘이면 못 고른다
                day[kind] = ""
            else:
                day[kind] = receipt
        pairs = [
            (day, found) for day, found in by_day.items()
            if set(found) == {"termination", "correction"}
            and all(found.values())
        ]
        return pairs[0] if len(pairs) == 1 else None

    def resolve(
            self, *, question_id: str, question: str,
            source_intent: SemanticIntent,
            ) -> "dict[str, Any] | None":
        del question
        # Same-day ordering ambiguity is relevant only to a request that
        # actually selects the latest document.  A status/amount question can
        # share the same counterparty and date but must retain its event
        # lifecycle authority rather than being rewritten as document list.
        if (not source_intent.answer_items or any(
                item.selection is None or item.selection.mode != "latest"
                for item in source_intent.answer_items)):
            return None
        counterparty = counterparty_surface(source_intent, self.canonical)
        if counterparty is None:
            return None
        items: list[dict[str, Any]] = []
        inferred_company_pair: tuple[Any, tuple[str, dict[str, str]]] | None = None
        for index, item in enumerate(source_intent.answer_items, start=1):
            company = _single_company(source_intent, item, self.canonical)
            if company is None:
                if inferred_company_pair is None:
                    from agent.planner_preflight import (
                        CanonicalSelectorRolePreflight,
                    )
                    issuer = CanonicalSelectorRolePreflight(
                        self.canonical,
                        corpus_cutoff=self.corpus_cutoff,
                        cache_root=(Path(__file__).resolve().parent.parent
                                    / "out/serving/selector_roles"),
                    ).resolve_selector_issuer(
                        counterparty, role="counterparty")
                    if issuer.status != "resolved" or len(issuer.candidates) != 1:
                        return None
                    candidate = issuer.candidates[0]
                    inferred = type("_Company", (), {
                        "corp_code": candidate.corp_code,
                        "corp_name": candidate.corp_name,
                    })()
                    pair = self._same_day_pair(
                        inferred.corp_code, counterparty)
                    if pair is None:
                        return None
                    inferred_company_pair = (inferred, pair)
                company = inferred_company_pair[0]
            if company is None:
                return None
            pair = (
                inferred_company_pair[1]
                if inferred_company_pair is not None
                else self._same_day_pair(company.corp_code, counterparty)
            )
            if pair is None:
                return None
            as_of, found = pair
            item_id = f"item-{index}"
            items.append({
                "item_id": item_id,
                "target_surface": item.target.surface,
                "projection_mode": item.output.projection_mode,
                "resolution": {
                    "kind": "same_day_document_candidates",
                    "issuer_corp_code": company.corp_code,
                    "issuer_corp_name": company.corp_name,
                    "as_of": as_of,
                    "candidates": [
                        {
                            "document_kind": kind,
                            "rcept_no": found[kind],
                            "proof_ref": f"source-document:{found[kind]}",
                        }
                        for kind in ("termination", "correction")
                    ],
                    "ordering_provenance": {
                        "code": "intraday_order_unavailable",
                        "family": "ordering",
                        "detail": (
                            "same-day receipts carry no intraday time; "
                            "candidate order is inventory order, not chronology"),
                        "evidence_refs": [f"source-ordering:{as_of}"],
                        "original_receipts": [],
                    },
                },
                "applied_defaults": [],
                "field_proofs": _field_proofs(
                    item_id, list(item.output.field_surfaces)),
            })
        return _authority(
            question_id=question_id, source_intent=source_intent,
            canonical_build_id=self.canonical_build_id,
            resolver_version=self.resolver_version,
            reference_date=self.reference_date,
            corpus_cutoff=self.corpus_cutoff, items=items)


__all__ += ["SameDayDocumentCandidatesBackend"]


class DocumentFactComparisonBackend:
    """같은 발행인의 금액 두 개를 비교하고, 다르면 그 사유 근거까지 붙인다.

    「정정 후 계약금액과 해지금액은 같은가, 다르면 왜 다른가」 꼴이다.  두 금액은
    서로 다른 공시에 있고, 사유는 해지 공시의 서술 칸에 있다.  세 좌표를 전부
    증거로 붙이지 못하면 물러난다 — 비교만 하고 사유를 못 대면 질문의 절반만
    답하는 계획이 된다.
    """

    MONEY_LABELS = ("계약금액", "해지금액")
    REASON_LABELS = ("기타 투자판단과 관련한 중요사항", "해지 주요사유")

    def __init__(
            self, canonical: Any, *, canonical_build_id: str,
            resolver_version: str, reference_date: date, corpus_cutoff: str,
            ) -> None:
        self.canonical = canonical
        self.canonical_build_id = canonical_build_id
        self.resolver_version = resolver_version
        self.reference_date = reference_date
        self.corpus_cutoff = corpus_cutoff

    def _rows_by_receipt(self, corp_code: str) -> "dict[str, list[Any]]":
        grouped: dict[str, list[Any]] = {}
        for row in self.canonical.fields(
                as_of=self.corpus_cutoff, corp_code=corp_code):
            if row.evidence_id:
                grouped.setdefault(row.rcept_no, []).append(row)
        return grouped

    def _pick(self, rows: "list[Any]", labels: "tuple[str, ...]") -> "Any | None":
        found = [
            row for row in rows
            if any(normalize_form_label(label) in normalize_form_label(
                _label_tail(row.path)) for label in labels)
        ]
        return found[0] if len(found) == 1 else None

    @staticmethod
    def _is_numeric_money_value(value: Any) -> bool:
        """Accept an actual disclosed amount, never a reservation marker.

        Contract forms commonly preserve a ``계약금액`` coordinate with ``-``
        while the amount is reserved.  That coordinate is evidence of the
        form, but not an operand in an amount comparison.  This is deliberately
        value-shape based rather than receipt/date based so corrected and
        future forms follow the same rule.
        """

        compact = re.sub(r"[\s,]", "", str(value or ""))
        return bool(re.fullmatch(r"[+-]?\d+(?:\.\d+)?", compact))

    def _pick_numeric_money(self, rows: "list[Any]") -> "Any | None":
        found = [
            row for row in rows
            if any(normalize_form_label(label) in normalize_form_label(
                _label_tail(row.path)) for label in self.MONEY_LABELS)
            and self._is_numeric_money_value(getattr(row, "value", None))
        ]
        return found[0] if len(found) == 1 else None

    def _pick_termination_reason(self, rows: "list[Any]") -> "Any | None":
        """Prefer the dedicated termination reason over catch-all narrative.

        A termination disclosure can contain both coordinates.  The dedicated
        field is the stable semantic slot; the narrative remains a fallback
        only for forms without that field.
        """

        for label in ("해지 주요사유", "기타 투자판단과 관련한 중요사항"):
            found = [
                row for row in rows
                if normalize_form_label(label) in normalize_form_label(
                    _label_tail(row.path))
            ]
            if len(found) == 1:
                return found[0]
            if len(found) > 1:
                return None
        return None

    def _pick_amount_difference_reason(self, rows: "list[Any]") -> "Any | None":
        """Select a value-explanation coordinate for a money comparison.

        ``해지 주요사유`` proves why a contract ended.  It does not normally
        explain why the corrected contract amount and the termination amount
        differ.  In the termination form that calculation is disclosed in the
        catch-all investment-decision narrative, so prefer that field for an
        amount-comparison's conditional explanation.  A dedicated termination
        reason remains a safe fallback when the form has no such narrative.
        """

        found = [
            row for row in rows
            if normalize_form_label("기타 투자판단과 관련한 중요사항")
            in normalize_form_label(_label_tail(row.path))
        ]
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            return None
        return self._pick_termination_reason(rows)

    @staticmethod
    def _ordered_comparison_money(
            money: "list[tuple[Any, str]]",
            ) -> "list[tuple[Any, str]] | None":
        """Close the comparison to its typed source roles, not receipt order.

        A correction and its termination can be filed on the same day.  Receipt
        numbers are identifiers, not a semantic ordering authority.  The typed
        contract instead requires the corrected contract amount first and the
        ordinary termination disclosure second.  Refuse any other inventory:
        a third monetary coordinate or two documents of the same source class
        cannot establish this two-fact comparison.
        """

        if len(money) != 2 or {source_class for _, source_class in money} != {
                "correction", "disclosure"}:
            return None
        return sorted(
            money,
            key=lambda pair: 0 if pair[1] == "correction" else 1,
        )

    def _operand(self, row: Any, operand_id: str, source_class: str,
                 value_kind: str) -> "dict[str, Any]":
        return {
            "operand_id": operand_id,
            "issuer_corp_code": row.corp_code,
            "issuer_corp_name": row.corp_name,
            "source_class": source_class,
            "doc_id": row.doc_id,
            "receipt_no": row.rcept_no,
            "path": row.path,
            "locator": row.locator,
            "source_file_id": row.source_file_id,
            "evidence_id": row.evidence_id,
            "value_kind": value_kind,
        }

    def resolve(
            self, *, question_id: str, question: str,
            source_intent: SemanticIntent,
            ) -> "dict[str, Any] | None":
        # This authority proves a very specific comparison: two disclosed
        # amounts plus the stated reason for their difference.  Merely finding
        # those three fields in a company's filings must not let it steal an
        # unrelated two-part document/status question about the same contract.
        request_text = " ".join((
            question,
            *(item.target.surface for item in source_intent.answer_items),
            *(surface for item in source_intent.answer_items
              for surface in item.output.field_surfaces),
        ))
        if not (
                re.search(r"계약\s*금액", request_text)
                and re.search(r"해지\s*금액", request_text)
                and re.search(r"다르|왜|이유", request_text)):
            return None
        if len(source_intent.answer_items) != 2:
            return None
        counterparty = counterparty_surface(source_intent, self.canonical)
        if counterparty is None:
            return None
        first = source_intent.answer_items[0]
        company = _single_company(source_intent, first, self.canonical)
        if company is None:
            from agent.planner_preflight import CanonicalSelectorRolePreflight
            issuer = CanonicalSelectorRolePreflight(
                self.canonical,
                corpus_cutoff=self.corpus_cutoff,
                cache_root=(Path(__file__).resolve().parent.parent
                            / "out/serving/selector_roles"),
            ).resolve_selector_issuer(counterparty, role="counterparty")
            if issuer.status == "resolved" and len(issuer.candidates) == 1:
                candidate = issuer.candidates[0]
                company = type("_Company", (), {
                    "corp_code": candidate.corp_code,
                    "corp_name": candidate.corp_name,
                })()
        if company is None:
            return None

        matcher = TerminationReportedStatusBackend(
            self.canonical, canonical_build_id=self.canonical_build_id,
            resolver_version=self.resolver_version,
            reference_date=self.reference_date,
            corpus_cutoff=self.corpus_cutoff)
        documents = matcher._documents(company.corp_code)
        related = [
            receipt for receipt, entry in documents.items()
            if matcher._matches_counterparty(entry["labels"], counterparty)
        ]
        if len(related) < 2:
            return None
        grouped = self._rows_by_receipt(company.corp_code)

        money: list[tuple[Any, str]] = []
        reason_row = None
        for receipt in sorted(related):
            rows = grouped.get(receipt) or []
            entry = documents[receipt]
            source_class = (
                "correction" if entry["is_correction"] else "disclosure")
            picked = self._pick_numeric_money(rows)
            if picked is not None:
                money.append((picked, source_class))
            if entry["form"] == matcher.TERMINATION_FORM:
                reason_row = (
                    self._pick_amount_difference_reason(rows), source_class)
        ordered_money = self._ordered_comparison_money(money)
        if ordered_money is None or reason_row is None or reason_row[0] is None:
            return None

        operands = [
            self._operand(row, f"operand-{position}", source_class, "money")
            for position, (row, source_class) in enumerate(ordered_money, start=1)
        ]
        reason = self._operand(
            reason_row[0], "operand-2", reason_row[1], "text")
        items = [
            {
                "item_id": "item-1",
                "target_surface": first.target.surface,
                "projection_mode": first.output.projection_mode,
                "resolution": {
                    "kind": "document_fact_comparison",
                    "operands": operands,
                },
                "applied_defaults": [],
                "field_proofs": _field_proofs(
                    "item-1", list(first.output.field_surfaces)),
            },
            {
                "item_id": "item-2",
                "target_surface": source_intent.answer_items[1].target.surface,
                "projection_mode":
                    source_intent.answer_items[1].output.projection_mode,
                "resolution": {
                    "kind": "document_reason_evidence",
                    "evidence": reason,
                },
                "applied_defaults": [],
                "field_proofs": _field_proofs(
                    "item-2",
                    list(source_intent.answer_items[1].output.field_surfaces)),
            },
        ]
        premise_overrides = {
            premise.premise_id: [
                str(operands[0]["evidence_id"]),
                str(operands[1]["evidence_id"]),
            ]
            for premise in source_intent.premises
            if set(premise.applies_to_item_ids) == {"item-1", "item-2"}
        }
        return _authority(
            question_id=question_id, source_intent=source_intent,
            canonical_build_id=self.canonical_build_id,
            resolver_version=self.resolver_version,
            reference_date=self.reference_date,
            corpus_cutoff=self.corpus_cutoff, items=items,
            premise_proof_refs=premise_overrides)


__all__ += ["DocumentFactComparisonBackend"]

"""Generic, compiler-independent authority for periodic narrative matrices.

This module deliberately stops at a typed Stage1 authority.  It does not
modify QueryPlan v0.4 or lower a matrix itself: that compatibility lowering is
owned by the compiler boundary.  The authority contains only grounded company,
document, period and source-proof coordinates, never report prose.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date
import re
from typing import Any

from agent.date_surface import question_date_surfaces
from agent.periodic_document_preflight import (
    PeriodicDocumentPreflight,
    PeriodicDocumentPreflightError,
    parse_periodic_expression,
)
from agent.semantic_intent_v1 import (
    AnswerItem, EntityMention, OutputRequest, Scope, SemanticIntent, Target,
    semantic_intent_digest,
)
from agent.stage1_v1_resolver import (
    ClarificationAuthority, ClarificationOption, ClarificationSlot,
)


#: The typed ``NarrativeMatrixResolution`` (both the dataclass sidecar here
#: and the frozen pydantic type in ``deterministic_plan_compiler_v1``, whose
#: JSON Schema is a checked release artifact) caps work units at 16.  That
#: ceiling cannot move without regenerating a frozen schema artifact, so an
#: over-limit request is never resolved whole.  ``NarrativeMatrixRegrounder``
#: already requires every company surface to be a literal span of the
#: question, and every reducible document/period expression the same — so
#: whenever a company or document axis has more than one value, the question
#: itself already names what a narrower subset would look like.  A request
#: over the cap is therefore answered directly from the widest schema-legal
#: subset (``_best_fit_comprehensive_subset``), with the narrowing disclosed
#: in the composed answer, rather than clarified.  Only when neither axis is
#: reducible — the excess is entirely on the topic axis, which the compiler
#: never lets narrow — does nothing tell the resolver what to drop, and
#: ``NarrativeMatrixCapacityClarificationBackend`` still asks.
MAX_NARRATIVE_MATRIX_CELLS = 16


_EXPLICIT_MATRIX_HEAD = re.compile(
    r"^(?P<companies>.+?)\s*의\s*"
    r"(?P<year1>(?:19|20)[0-9]{2}년)\s*(?:과|와|,)\s*"
    r"(?P<year2>(?:19|20)[0-9]{2}년)",
)
_EXPLICIT_MATRIX_RANGE_HEAD = re.compile(
    r"^(?P<companies>.+?)\s*의\s*"
    r"(?P<start>(?:19|20)[0-9]{2})\s*년?\s*(?:~|～|−|-|부터)\s*"
    r"(?P<end>(?:19|20)[0-9]{2})\s*년",
)
_EXPLICIT_MATRIX_SINGLE_HEAD = re.compile(
    r"^(?P<companies>.+?)\s*의\s*"
    r"(?P<year>(?:19|20)[0-9]{2}년)",
)
_EXPLICIT_COMPANY_SEPARATOR = re.compile(r"\s*(?:·|ㆍ|・|,|및|와|과)\s*")
_FALLBACK_PREMISE_OR_CAUSAL = re.compile(
    r"(?:^|\s)(?:만약|가정(?:하면|할 때)?|전제(?:하면|할 때)?|"
    r"사실이라면|맞다면)|왜|원인|때문")


def literal_matrix_grammar(
        question: str,
        ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None:
    """Return the raw company/year/topic spans named in an explicit matrix
    question, without canonical grounding.

    This mirrors the same closed grammar ``explicit_narrative_matrix_fallback_intent``
    parses, but is for the composer's 「N개 중 M개만」 disclosure only — when
    an over-limit explicit multi-company/period comparison had to be
    narrowed to fit the execution cap (issue #86 17).  Never used for
    citation, grounding, or any fact in the answer itself; a caller that
    cannot parse this grammar simply gets no disclosure, not a wrong one.
    """

    if not isinstance(question, str) or not question.strip():
        return None
    match = _EXPLICIT_MATRIX_HEAD.search(question)
    range_match = _EXPLICIT_MATRIX_RANGE_HEAD.search(question)
    single_match = _EXPLICIT_MATRIX_SINGLE_HEAD.search(question)
    if match is None and range_match is None and single_match is None:
        return None
    company_span = (
        match.group("companies") if match is not None
        else range_match.group("companies") if range_match is not None
        else single_match.group("companies"))
    companies = tuple(
        part.strip() for part in _EXPLICIT_COMPANY_SEPARATOR.split(
            company_span) if part.strip())
    if match is not None:
        years = (match.group("year1"), match.group("year2"))
    elif range_match is not None:
        start = int(range_match.group("start"))
        end = int(range_match.group("end"))
        if start >= end or end - start > 4:
            return None
        years = tuple(f"{year}년" for year in range(start, end + 1))
    else:
        assert single_match is not None
        years = (single_match.group("year"),)
    topics = _question_business_topics(question)
    if not companies or not years or not topics:
        return None
    return companies, years, topics


def explicit_narrative_matrix_fallback_intent(
        question: str, *,
        company_surface_regrounder: Any,
        ) -> SemanticIntent | None:
    """Build one intent only from a closed, explicit comparison grammar.

    This is the fail-closed escape hatch for a provider response which remains
    schema-invalid after its one contract repair.  It is deliberately not a
    second natural-language parser: every company is copied from the question
    and must independently resolve through the canonical company regrounder;
    both annual periods and every supported narrative topic are literal spans.
    Premises, causal requests, implicit periods and unresolved issuers are not
    accepted.
    """

    if (not isinstance(question, str) or not question.strip()
            or not callable(company_surface_regrounder)
            or _FALLBACK_PREMISE_OR_CAUSAL.search(question)
            or re.search(r"비교|차이|변화", question) is None):
        return None
    match = _EXPLICIT_MATRIX_HEAD.search(question)
    range_match = _EXPLICIT_MATRIX_RANGE_HEAD.search(question)
    single_match = _EXPLICIT_MATRIX_SINGLE_HEAD.search(question)
    if match is None and range_match is None and single_match is None:
        return None
    company_span = (
        match.group("companies") if match is not None
        else range_match.group("companies") if range_match is not None
        else single_match.group("companies"))
    raw_companies = tuple(
        part.strip() for part in _EXPLICIT_COMPANY_SEPARATOR.split(
            company_span) if part.strip())
    if not 2 <= len(raw_companies) <= 16:
        return None
    companies: list[str] = []
    for raw in raw_companies:
        grounded = company_surface_regrounder(raw, question)
        if (not isinstance(grounded, str) or not grounded.strip()
                or grounded not in question or grounded in companies):
            return None
        companies.append(grounded)
    if match is not None:
        years = (match.group("year1"), match.group("year2"))
        if years[0] == years[1]:
            return None
    elif range_match is not None:
        assert range_match is not None
        start, end = int(range_match.group("start")), int(range_match.group("end"))
        if start >= end or end - start > 4:
            return None
        years = (question[
            range_match.start("start"):range_match.end()
        ].strip(),)
    else:
        assert single_match is not None
        years = (single_match.group("year"),)
    topics = _question_business_topics(question)
    if not 1 <= len(topics) <= 4:
        return None
    units = len(companies) * len(years) * len(topics)
    # The runtime owns the 16-unit execution boundary and can clarify a larger
    # explicit matrix.  This parser merely rejects unreasonable expansion.
    if not 2 <= units <= 64:
        return None
    entities = [
        EntityMention(
            entity_id=f"entity-{index}", kind_hint="company", surface=surface)
        for index, surface in enumerate(companies, start=1)
    ]
    presentation = (
        "table" if re.search(r"(?:표|테이블)\s*(?:로|으로|형식|형태)", question)
        else "list" if re.search(r"목록\s*(?:으로|형식|형태)", question)
        else "prose" if re.search(r"(?:서술|문장)\s*(?:로|으로|형식|형태)", question)
        else "auto"
    )
    return SemanticIntent(
        schema_version="stage1-semantic-intent/1.1",
        entities=entities,
        answer_items=[AnswerItem(
            item_id="item-1",
            target=Target(
                kind="topic", surface=topics[0],
                entity_refs=[entity.entity_id for entity in entities],
                qualifier_surfaces=[]),
            operation="compare",
            scope=Scope(
                target_period_expressions=list(years),
                as_of_expression=None, document_group_expression=None,
                scope_qualifier_expressions=[]),
            selection=None,
            output=OutputRequest(
                shape="comparison", projection_mode="named_fields",
                field_surfaces=list(topics), presentation=presentation),
        )],
        answer_groups=[], premises=[], unresolved_mentions=[],
        presentation=presentation,
    )


class AnnualBusinessMatrixIntentRegrounder:
    """Normalize model shape variance from an explicit annual comparison.

    Company, year and topic values all come from the question or its grounded
    company mentions.  The only policy default is that a calendar-year
    business narrative refers to that year's annual report when no other
    periodic form is named.  This is an ontology rule, not issuer/case data.
    """

    _YEAR = re.compile(r"(?<![0-9])(?:19|20)[0-9]{2}년")
    _LATEST_PERIODIC = re.compile(
        r"(?:가장\s*)?최근\s*(?:정기|사업|반기|분기)보고서")

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (not isinstance(question, str) or intent.answer_groups or intent.premises
                or intent.unresolved_mentions or not intent.answer_items
                or not re.search(r"비교|차이|변화", question)
                or re.search(r"분기|반기|월간|월별", question)
                # A latest-report selector is already a complete document
                # coordinate.  A nearby cutoff date is observation scope,
                # never an annual-report target.
                or self._LATEST_PERIODIC.search(question)):
            return intent
        companies = tuple(entity for entity in intent.entities
                          if entity.kind_hint == "company"
                          and entity.surface in question)
        if not companies:
            return intent
        years = _annual_target_years(question, self._YEAR)
        if not 1 <= len(years) <= 3:
            return intent
        question_topics = _question_business_topics(question)
        if not question_topics:
            return intent
        # A matrix requires at least two independently retrieved units, but
        # that variation may be across companies, years, or topics.  A single
        # issuer compared across two annual reports is therefore valid.
        if len(companies) * len(years) * len(question_topics) < 2:
            return intent
        if any(item.operation not in {"retrieve", "compare"}
               for item in intent.answer_items):
            return intent
        explicit_group = "사업보고서" if "사업보고서" in question else None
        item = AnswerItem(
            item_id="item-1",
            target=Target(
                kind="topic", surface=question_topics[0],
                entity_refs=[entity.entity_id for entity in companies],
                qualifier_surfaces=[],
            ),
            operation="compare",
            scope=Scope(
                target_period_expressions=list(years),
                as_of_expression=None,
                document_group_expression=explicit_group,
                scope_qualifier_expressions=[],
            ),
            selection=None,
            output=OutputRequest(
                shape="comparison", projection_mode="named_fields",
                field_surfaces=list(question_topics),
                presentation=intent.presentation,
            ),
        )
        return SemanticIntent(
            schema_version=intent.schema_version,
            entities=intent.entities,
            answer_items=[item], answer_groups=[], premises=[],
            unresolved_mentions=[], presentation=intent.presentation,
        )


@dataclass(frozen=True, slots=True)
class NarrativeMatrixCell:
    cell_id: str
    corp_code: str
    corp_name: str
    doc_id: str
    receipt_no: str
    period_start: date
    period_end: date
    topics: tuple[str, ...]
    document_proof_ref: str

    def __post_init__(self) -> None:
        if (not re.fullmatch(r"[0-9]{8}", self.corp_code)
                or not re.fullmatch(r"[0-9]{14}", self.receipt_no)
                or self.doc_id != f"periodic_{self.receipt_no}"
                or self.period_start > self.period_end
                or not self.corp_name.strip()
                or not self.topics
                or any(not topic.strip() for topic in self.topics)
                or self.doc_id not in self.document_proof_ref):
            raise ValueError("narrative matrix cell coordinate가 잘못되었습니다")


@dataclass(frozen=True, slots=True)
class NarrativeMatrixResolution:
    """Internal typed sidecar; public QueryPlan v0.4 remains unchanged."""

    item_id: str
    source_intent_digest: str
    corpus_cutoff: str
    cells: tuple[NarrativeMatrixCell, ...]

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_intent_digest):
            raise ValueError("matrix source intent digest가 잘못되었습니다")
        if not re.fullmatch(r"[0-9]{8}", self.corpus_cutoff):
            raise ValueError("matrix corpus cutoff이 잘못되었습니다")
        work_units = sum(len(cell.topics) for cell in self.cells)
        if (not 2 <= len(self.cells) <= MAX_NARRATIVE_MATRIX_CELLS
                or not 2 <= work_units <= MAX_NARRATIVE_MATRIX_CELLS):
            raise ValueError("narrative matrix work unit 수는 2..16이어야 합니다")
        keys = [(cell.corp_code, cell.doc_id) for cell in self.cells]
        if len(keys) != len(set(keys)):
            raise ValueError("narrative matrix corp/document cell이 중복되었습니다")


@dataclass(frozen=True, slots=True)
class NarrativeMatrixAuthority:
    question_id: str
    canonical_build_id: str
    resolver_version: str
    resolution: NarrativeMatrixResolution

    def __post_init__(self) -> None:
        if (not self.question_id.strip()
                or not re.fullmatch(r"[0-9a-f]{32}", self.canonical_build_id)
                or not self.resolver_version.strip()):
            raise ValueError("narrative matrix authority binding이 잘못되었습니다")


@dataclass(frozen=True, slots=True)
class NarrativeMatrixRequest:
    item_id: str
    company_surfaces: tuple[str, ...]
    document_expressions: tuple[str, ...]
    topics: tuple[str, ...]

    @property
    def work_unit_count(self) -> int:
        """Number of independently retrieved company/document/topic units."""
        return (len(self.company_surfaces) * len(self.document_expressions)
                * len(self.topics))


@dataclass(frozen=True, slots=True)
class NarrativeMatrixResumeSelection:
    """Grounded scope reduction supplied when an over-limit matrix resumes.

    The selected values must be copied from the original matrix request.  This
    type intentionally carries no question-specific option IDs and therefore
    remains reusable by session/clarification adapters.
    """

    dimension: str
    retained_surfaces: tuple[str, ...]

    def __post_init__(self) -> None:
        if (self.dimension not in {"company", "document", "topic"}
                or not self.retained_surfaces
                or any(not value.strip() for value in self.retained_surfaces)
                or len(set(self.retained_surfaces)) != len(self.retained_surfaces)):
            raise ValueError("narrative matrix resume selection이 잘못되었습니다")


@dataclass(frozen=True, slots=True)
class NarrativeMatrixCapacity:
    work_unit_count: int
    limit: int
    reducible_dimensions: tuple[str, ...]

    @property
    def within_limit(self) -> bool:
        return 2 <= self.work_unit_count <= self.limit


@dataclass(frozen=True, slots=True)
class NarrativeMatrixReductionChoice:
    """One public opaque choice and its internal grounded subset."""

    option: ClarificationOption
    selection: NarrativeMatrixResumeSelection


def narrative_matrix_capacity(
        request: NarrativeMatrixRequest, *,
        limit: int = MAX_NARRATIVE_MATRIX_CELLS,
        ) -> NarrativeMatrixCapacity:
    dimensions = tuple(
        name for name, values in (
            ("company", request.company_surfaces),
            ("document", request.document_expressions),
            ("topic", request.topics),
        ) if len(values) > 1)
    return NarrativeMatrixCapacity(
        work_unit_count=request.work_unit_count,
        limit=limit,
        reducible_dimensions=dimensions,
    )


def apply_narrative_matrix_resume(
        request: NarrativeMatrixRequest,
        selection: NarrativeMatrixResumeSelection,
        ) -> NarrativeMatrixRequest:
    """Apply one subset selection without accepting invented scope values."""
    attribute = {
        "company": "company_surfaces",
        "document": "document_expressions",
        "topic": "topics",
    }[selection.dimension]
    source = getattr(request, attribute)
    if any(value not in source for value in selection.retained_surfaces):
        raise ValueError("원래 matrix scope에 없는 resume 값입니다")
    values = {
        "company_surfaces": request.company_surfaces,
        "document_expressions": request.document_expressions,
        "topics": request.topics,
    }
    values[attribute] = selection.retained_surfaces
    resumed = NarrativeMatrixRequest(item_id=request.item_id, **values)
    if not narrative_matrix_capacity(resumed).within_limit:
        raise ValueError("resume 이후에도 narrative matrix 한도를 초과합니다")
    return resumed


def narrative_matrix_reduction_options(
        request: NarrativeMatrixRequest,
        ) -> tuple[NarrativeMatrixReductionChoice, ...]:
    """Build deterministic, bounded scope choices without leaking parsers.

    Public values are opaque IDs.  Resume regenerates this exact inventory
    from the immutable question and intent and never interprets user-supplied
    company names or periods.

    Topic-only reduction stays disabled.  Two layers forbid it and both are
    deliberate: the compiler checks every cell's topics against
    ``output.field_surfaces`` exactly, and the clarification session refuses
    any resume that changes the question's semantic surface at all.  That
    second rule cannot tell ``the reader chose to narrow this`` from ``the
    backend quietly dropped a field``, which is the whole reason it exists.

    So the company axis carries every reduction, and a reduction there always
    sets aside at least one name the question asked for.  Say which one.
    """
    choices: list[NarrativeMatrixReductionChoice] = []
    for index, value in enumerate(request.document_expressions, start=1):
        selection = NarrativeMatrixResumeSelection("document", (value,))
        units = len(request.company_surfaces) * len(request.topics)
        if not 2 <= units <= MAX_NARRATIVE_MATRIX_CELLS:
            continue
        resumed = apply_narrative_matrix_resume(request, selection)
        choices.append(NarrativeMatrixReductionChoice(
            option=ClarificationOption(
                value=f"narrative_matrix:document:{index}",
                label=f"{value}만 전체 회사와 전체 주제로 비교 ({units}개 항목)"),
            selection=selection,
        ))

    other_axis_units = (
        len(request.document_expressions) * len(request.topics))
    max_companies = (
        MAX_NARRATIVE_MATRIX_CELLS // other_axis_units
        if other_axis_units else 0)
    if 0 < max_companies < len(request.company_surfaces):
        company_subsets = (
            request.company_surfaces[:max_companies],
            request.company_surfaces[-max_companies:],
        )
        seen: set[tuple[str, ...]] = set()
        for retained in company_subsets:
            if retained in seen:
                continue
            seen.add(retained)
            selection = NarrativeMatrixResumeSelection("company", retained)
            resumed = apply_narrative_matrix_resume(request, selection)
            # DEV-NAR-019 offered two five-company subsets that overlapped in
            # four names, so the two options read as the same thing twice.
            # What separates them is who is left out, and that is what the
            # reader is actually deciding.
            omitted = [surface for surface in request.company_surfaces
                       if surface not in retained]
            choices.append(NarrativeMatrixReductionChoice(
                option=ClarificationOption(
                    value=f"narrative_matrix:company_chunk:{len(seen)}",
                    label=(f"{', '.join(omitted)} 제외 — 나머지 "
                           f"{len(retained)}개사를 전체 기간과 전체 주제로 비교 "
                           f"({resumed.work_unit_count}개 항목)"
                           if omitted else
                           f"{', '.join(retained)}만 전체 기간과 전체 주제로 "
                           f"비교 ({resumed.work_unit_count}개 항목)")),
                selection=selection,
            ))
    return tuple(choices)


def _best_fit_comprehensive_subset(
        request: NarrativeMatrixRequest,
        ) -> NarrativeMatrixRequest | None:
    """Narrow an over-limit explicit request to one schema-legal subset,
    without asking the reader to choose.

    Every candidate here already comes from ``narrative_matrix_reduction_options``,
    so no company, document or topic value is invented and every candidate
    independently satisfies the compiler's 16-unit ceiling.  A document/period
    reduction is preferred when one exists — 「그 이상이면 연도별로」 keeps
    every requested company answered, just for the most recent period named —
    and only when the request has just one document does this fall back to
    the widest company subset.  Topics are never the reduced axis: the
    compiler requires every retained cell's topics to equal the full
    requested field set exactly.
    """

    by_dimension: dict[str, list[NarrativeMatrixReductionChoice]] = {}
    for choice in narrative_matrix_reduction_options(request):
        by_dimension.setdefault(choice.selection.dimension, []).append(choice)
    document_choices = by_dimension.get("document")
    if document_choices:
        # Options are built in question order; the last is the most recent
        # named period.
        selection = document_choices[-1].selection
    else:
        company_choices = by_dimension.get("company")
        if not company_choices:
            return None
        selection = max(
            company_choices,
            key=lambda choice: len(choice.selection.retained_surfaces),
        ).selection
    try:
        return apply_narrative_matrix_resume(request, selection)
    except ValueError:
        return None


class NarrativeMatrixRegrounder:
    """Recover a bounded matrix request from question-grounded semantic data.

    It accepts either explicit periodic report coordinates (one or more years)
    or one literal ``latest periodic`` selector.  Topic strings are copied
    verbatim from the semantic output/qualifier surfaces and must occur in the
    original question; no company, period, or topic is invented.
    """

    def __call__(self, question: str, intent: SemanticIntent) -> NarrativeMatrixRequest | None:
        if (not isinstance(question, str) or len(intent.answer_items) != 1
                or intent.answer_groups or intent.premises
                or intent.unresolved_mentions):
            return None
        item = intent.answer_items[0]
        if (item.target.kind not in {"document", "topic"}
                or item.operation not in {"retrieve", "compare"}
                or item.output.projection_mode != "named_fields"):
            return None
        by_id = {entity.entity_id: entity for entity in intent.entities}
        company_surfaces = tuple(
            by_id[ref].surface for ref in item.target.entity_refs
            if ref in by_id and by_id[ref].kind_hint == "company"
        )
        if not 1 <= len(company_surfaces) <= MAX_NARRATIVE_MATRIX_CELLS:
            return None
        if len(set(company_surfaces)) != len(company_surfaces):
            return None
        if any(surface not in question for surface in company_surfaces):
            return None

        # A literal latest selector is one document coordinate per company.
        # Other time surfaces (for example an as-of cutoff) are not silently
        # reinterpreted as reporting periods.
        try:
            selector = parse_periodic_expression(item.target.surface)
        except PeriodicDocumentPreflightError:
            selector = None
        if selector is not None and selector.latest:
            document_expressions = (item.target.surface,)
        else:
            group = item.scope.document_group_expression
            periods = _expanded_annual_periods(
                tuple(item.scope.target_period_expressions))
            if not periods:
                return None
            if group is None and _question_business_topics(question):
                # A full calendar year plus a business narrative axis has one
                # deterministic periodic source family: the annual report.
                group = "사업보고서"
            if group is None or not group.strip():
                return None
            document_expressions = tuple(
                period if _is_periodic_expression(period)
                else f"{period} {group}"
                for period in periods)
            if any(not _is_periodic_expression(value)
                   for value in document_expressions):
                return None

        # Output fields are the answer contract's only independently bindable
        # topic axes.  A repeated qualifier (EDGE-043 style) is therefore
        # retained through that identical field; date cutoffs never become a
        # retrieval topic.
        topic_surfaces = tuple(dict.fromkeys(
            surface for surface in item.output.field_surfaces
            if surface.strip() and surface in question
            and not _is_observation_cutoff(surface)))
        if not topic_surfaces:
            return None
        topic_surfaces = _independent_business_topics(question, topic_surfaces)
        request = NarrativeMatrixRequest(
            item_id=item.item_id, company_surfaces=company_surfaces,
            document_expressions=document_expressions, topics=topic_surfaces)
        return request


class NarrativeMatrixCapacityClarificationBackend:
    """Ask the user to reduce an otherwise grounded matrix over 16 units.

    The choices are derived only from the request axes.  No company, year or
    topic is silently dropped, and the ordinary matrix backend remains the
    only component allowed to resolve an in-limit request.

    Every company and every reducible document/period expression here is
    already a literal span of the question (``NarrativeMatrixRegrounder``
    requires it), so whenever a company or document axis has more than one
    value the question has already named a narrower subset.  This backend
    then steps aside and lets ``NarrativeMatrixResolutionBackend`` answer
    directly from the widest schema-legal subset instead of asking — see
    ``_best_fit_comprehensive_subset``.  It clarifies only when neither axis
    is reducible, so nothing tells the resolver what to drop.
    """

    def __init__(self) -> None:
        self.regrounder = NarrativeMatrixRegrounder()

    def resolve(
            self, *, question_id: str, question: str,
            source_intent: SemanticIntent,
            ) -> ClarificationAuthority | None:
        del question_id
        request = self.regrounder(question, source_intent)
        if request is None:
            return None
        capacity = narrative_matrix_capacity(request)
        if capacity.within_limit:
            return None
        if _best_fit_comprehensive_subset(request) is not None:
            return None

        choices = narrative_matrix_reduction_options(request)
        options = [choice.option for choice in choices]
        if len(options) < 2:
            return None
        return ClarificationAuthority(slots=[ClarificationSlot(
            slot_id="slot-1", role_hint="selection",
            reason_code="narrative_matrix_limit_exceeded",
            response_kind="select_one",
            # `한도 16개` 는 실행 예산이지 사용자가 알아야 할 숫자가 아니다.
            prompt=(f"비교할 항목이 {capacity.work_unit_count}개로 한 번에 "
                    "보여드리기에 많습니다. 어느 범위로 볼까요?"),
            applies_to_item_ids=[request.item_id], mention_ids=[],
            options=options,
        )])


def _independent_business_topics(
        question: str, topic_surfaces: tuple[str, ...]) -> tuple[str, ...]:
    """Split only independently stated business and product/service axes.

    A lone broad topic is preserved verbatim.  Splitting occurs only when the
    question itself contains both disclosure axes, so qualifiers such as an
    industry or technology name are never discarded merely because the phrase
    also contains ``주요 사업``.
    """
    # Multiple model fields are already independently bindable axes.  The
    # splitter exists only to recover two axes when HCX collapsed them into a
    # single output field; it must not discard a third field such as 매출구성.
    if len(topic_surfaces) > 1:
        return topic_surfaces
    business = re.search(
        r"(?:주요\s*)?사업(?!\s*보고서)(?:\s*부문|\s*내용)?", question)
    products = re.search(r"(?:주요\s*)?제품(?:\s*[·ㆍ]\s*서비스|\s*(?:및|과|와)\s*서비스)?|제품\s*[·ㆍ]\s*서비스", question)
    if business is None or products is None or business.span() == products.span():
        return topic_surfaces
    axes = tuple(dict.fromkeys((business.group(0).strip(), products.group(0).strip())))
    return axes if len(axes) == 2 else topic_surfaces


def _question_business_topics(question: str) -> tuple[str, ...]:
    """Return literal disclosure axes in their order in the question.

    Only headings that the narrative backends can independently ground are
    recovered.  Interpretive output requests such as ``핵심 변화`` remain a
    composition concern and do not become retrieval axes.
    """
    qualified_business = (
        r"(?:[가-힣A-Za-z0-9]+(?:\s*[·ㆍ/]\s*[가-힣A-Za-z0-9]+)*\s*관련\s*)?"
        r"(?:주요\s*)?사업(?!\s*보고서)(?:\s*부문|\s*내용)?")
    patterns = (
        qualified_business,
        r"(?:주요\s*)?자회사\s*구성",
        r"(?:주요\s*)?제품(?:\s*[·ㆍ]\s*서비스|\s*(?:및|과|와)\s*서비스)?",
        r"매출\s*구성",
        r"수익\s*구조",
        r"수익원",
    )
    matches: list[tuple[int, int, str]] = []
    for pattern in patterns:
        matches.extend(
            (match.start(), match.end(), match.group(0).strip())
            for match in re.finditer(pattern, question))
    matches.sort(key=lambda row: (row[0], -(row[1] - row[0])))
    selected: list[tuple[int, int, str]] = []
    for start, end, surface in matches:
        if any(start < prior_end and end > prior_start
               for prior_start, prior_end, _ in selected):
            continue
        selected.append((start, end, surface))
    selected.sort(key=lambda row: row[0])
    return tuple(dict.fromkeys(surface for _, _, surface in selected))


def _annual_target_years(
        question: str, pattern: re.Pattern[str],
        ) -> tuple[str, ...]:
    """Extract annual targets while excluding full-date cutoff years."""
    years: list[str] = []
    ranges: list[tuple[int, int]] = []
    for match in re.finditer(
            r"(?<![0-9])(?P<start>(?:19|20)[0-9]{2})\s*년?\s*"
            r"(?:~|～|−|-|부터)\s*(?P<end>(?:19|20)[0-9]{2})\s*년",
            question):
        start, end = int(match.group("start")), int(match.group("end"))
        if start < end and end - start <= 4:
            years.append(match.group(0).strip())
            ranges.append((match.start(), match.end()))
    for match in pattern.finditer(question):
        # ``2026년 6월 19일까지`` is an as-of coordinate.  Excluding the year
        # when a month follows keeps date scope separate from report periods.
        if re.match(r"\s*[0-9]{1,2}\s*월", question[match.end():]):
            continue
        if any(start <= match.start() < end for start, end in ranges):
            continue
        years.append(match.group(0))
    return tuple(dict.fromkeys(years))


def _expanded_annual_periods(periods: tuple[str, ...]) -> tuple[str, ...]:
    """Expand a literal inclusive annual range only in the private request."""

    expanded: list[str] = []
    for surface in periods:
        match = re.fullmatch(
            r"\s*(?P<start>(?:19|20)[0-9]{2})\s*년?\s*"
            r"(?:~|～|−|-|부터)\s*(?P<end>(?:19|20)[0-9]{2})\s*년\s*",
            surface,
        )
        if match is None:
            expanded.append(surface)
            continue
        start, end = int(match.group("start")), int(match.group("end"))
        if start >= end or end - start > 4:
            return ()
        expanded.extend(f"{year}년" for year in range(start, end + 1))
    return tuple(dict.fromkeys(expanded))


def _is_periodic_expression(value: str) -> bool:
    try:
        parse_periodic_expression(value)
    except PeriodicDocumentPreflightError:
        return False
    return True


def _is_observation_cutoff(value: str) -> bool:
    """Keep a question's as-of date out of a narrative retrieval topic."""
    return bool(question_date_surfaces(value)) and bool(re.search(
        r"(?:까지|현재|기준)", value))


class NarrativeMatrixResolutionBackend:
    """Resolve the matrix through company/document metadata only.

    Any unresolved company, document or period invalidates the whole matrix;
    no partial selection is emitted at this authority boundary.
    """

    def __init__(
            self, canonical: Any, *, canonical_build_id: str,
            resolver_version: str, corpus_cutoff: str,
            reference_date: date | None = None,
            ) -> None:
        if not callable(getattr(canonical, "resolve_company", None)):
            raise TypeError("narrative matrix canonical.resolve_company가 필요합니다")
        self.canonical = canonical
        self.preflight = PeriodicDocumentPreflight(canonical)
        self.canonical_build_id = canonical_build_id
        self.resolver_version = resolver_version
        self.corpus_cutoff = corpus_cutoff
        self.reference_date = reference_date or date.fromisoformat(
            f"{corpus_cutoff[:4]}-{corpus_cutoff[4:6]}-{corpus_cutoff[6:]}")
        self.regrounder = NarrativeMatrixRegrounder()

    def resolve(
            self, *, question_id: str, question: str,
            source_intent: SemanticIntent,
            ) -> "dict[str, Any] | None":
        authority = self.resolve_matrix(
            question_id=question_id, question=question,
            source_intent=source_intent)
        if authority is None:
            return None
        resolution = self.authoritative_resolution(
            source_intent=source_intent, authority=authority)
        return {"kind": "resolved", "resolution": resolution.model_dump(mode="json")}

    def authoritative_resolution(
            self, *, source_intent: SemanticIntent,
            authority: NarrativeMatrixAuthority,
            ) -> Any:
        """Adapt the internal sidecar to the frozen Stage1 resolution type."""
        from agent.deterministic_plan_compiler_v1 import (
            AuthoritativeResolution, ResolutionFieldProof, ResolvedItem)
        item = source_intent.answer_items[0]
        cells = [
            {
                "cell_id": cell.cell_id, "corp_code": cell.corp_code,
                "corp_name": cell.corp_name, "doc_id": cell.doc_id,
                "receipt_no": cell.receipt_no,
                "period_start": cell.period_start, "period_end": cell.period_end,
                "topics": list(cell.topics),
                "document_proof": {
                    "source_receipt": cell.receipt_no,
                    "proof_ref": cell.document_proof_ref,
                },
            }
            for cell in authority.resolution.cells
        ]
        return AuthoritativeResolution.create(
            question_id=authority.question_id,
            source_intent_digest=semantic_intent_digest(source_intent),
            canonical_build_id=self.canonical_build_id,
            resolver_version=self.resolver_version,
            reference_date=self.reference_date,
            corpus_cutoff=self.corpus_cutoff,
            items=[ResolvedItem(
                item_id=item.item_id, target_surface=item.target.surface,
                projection_mode=item.output.projection_mode,
                resolution={"kind": "narrative_matrix", "cells": cells},
                field_proofs=[ResolutionFieldProof(
                    source_field_index=index, surface=surface,
                    proof_ref=f"source-field:{item.item_id}:{index}")
                    for index, surface in enumerate(item.output.field_surfaces)],
                applied_defaults=[],
            )], premise_proofs=[])

    def resolve_matrix(
            self, *, question_id: str, question: str,
            source_intent: SemanticIntent,
            ) -> NarrativeMatrixAuthority | None:
        request = self.regrounder(question, source_intent)
        if request is None:
            return None
        if not narrative_matrix_capacity(request).within_limit:
            # The capacity clarification backend already stepped aside for
            # exactly this case (issue #86 17): a company or document axis
            # here has more than one question-literal value, so answer from
            # the widest subset that still satisfies the compiler's 16-unit
            # ceiling instead of silently refusing or asking again.  If no
            # axis is reducible, that backend clarified instead and this
            # never runs.
            request = _best_fit_comprehensive_subset(request)
            if request is None:
                return None
        return self.resolve_request(
            question_id=question_id, request=request,
            source_intent=source_intent)

    def resolve_request(
            self, *, question_id: str, request: NarrativeMatrixRequest,
            source_intent: SemanticIntent,
            ) -> NarrativeMatrixAuthority | None:
        """Resolve an already validated request, including resumed subsets."""
        if not narrative_matrix_capacity(request).within_limit:
            return None
        companies = []
        for surface in request.company_surfaces:
            rows = self.canonical.resolve_company(surface)
            if len(rows) != 1:
                return None
            company = rows[0]
            corp_code = str(getattr(company, "corp_code", ""))
            corp_name = str(getattr(company, "corp_name", ""))
            if not re.fullmatch(r"[0-9]{8}", corp_code) or not corp_name.strip():
                return None
            companies.append((corp_code, corp_name))
        if len({code for code, _name in companies}) != len(companies):
            return None

        cells = []
        for corp_code, corp_name in companies:
            for expression in request.document_expressions:
                resolved = self.preflight.resolve_periodic_document(
                    corp_code=corp_code, as_of=self.corpus_cutoff,
                    target_expression=expression)
                if resolved.status != "resolved" or resolved.candidate is None:
                    return None
                receipt_no = resolved.candidate.rcept_no
                period = self._document_period(corp_code, receipt_no)
                if period is None:
                    return None
                cells.append(NarrativeMatrixCell(
                    cell_id=f"cell-{len(cells) + 1}",
                    corp_code=corp_code, corp_name=corp_name,
                    doc_id=f"periodic_{receipt_no}", receipt_no=receipt_no,
                    period_start=period[0], period_end=period[1],
                    topics=request.topics,
                    document_proof_ref=(
                        f"source-document:periodic_{receipt_no}:{receipt_no}"),
                ))
        try:
            resolution = NarrativeMatrixResolution(
                item_id=request.item_id,
                source_intent_digest=semantic_intent_digest(source_intent),
                corpus_cutoff=self.corpus_cutoff, cells=tuple(cells))
        except ValueError:
            return None
        return NarrativeMatrixAuthority(
            question_id=question_id, canonical_build_id=self.canonical_build_id,
            resolver_version=self.resolver_version, resolution=resolution)

    def _document_period(self, corp_code: str, receipt_no: str) -> tuple[date, date] | None:
        rows = [row for row in self.canonical.documents(
            corp_code=corp_code, as_of=self.corpus_cutoff, doc_group="periodic")
            if str(getattr(row, "rcept_no", "")) == receipt_no]
        if len(rows) != 1:
            return None
        row = rows[0]
        year, month = getattr(row, "base_year", None), getattr(row, "base_month", None)
        form = getattr(row, "form", None)
        if type(year) is not int or type(month) is not int:
            return None
        if form == "annual" and month == 12:
            return date(year, 1, 1), date(year, 12, 31)
        if form == "half" and month == 6:
            return date(year, 1, 1), date(year, 6, 30)
        if form == "quarter" and month in {3, 9}:
            return date(year, month - 2, 1), date(year, month,
                                                   calendar.monthrange(year, month)[1])
        return None


__all__ = [
    "AnnualBusinessMatrixIntentRegrounder", "MAX_NARRATIVE_MATRIX_CELLS",
    "explicit_narrative_matrix_fallback_intent",
    "literal_matrix_grammar",
    "NarrativeMatrixAuthority",
    "NarrativeMatrixCell", "NarrativeMatrixRegrounder",
    "NarrativeMatrixCapacity", "NarrativeMatrixRequest",
    "NarrativeMatrixReductionChoice",
    "NarrativeMatrixResumeSelection", "NarrativeMatrixResolution",
    "NarrativeMatrixResolutionBackend",
    "NarrativeMatrixCapacityClarificationBackend",
    "apply_narrative_matrix_resume", "narrative_matrix_capacity",
    "narrative_matrix_reduction_options",
]

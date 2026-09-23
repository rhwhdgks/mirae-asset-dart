"""AnswerPayload — 2~4단계의 typed 산출물이자 채점 단위.

composer(HCX-005)는 이 payload만 입력으로 받아 문장을 만든다. 채점기도 문장이 아니라
이 payload를 채점한다. 따라서 "값이 맞는가"와 "문장이 자연스러운가"가 분리된다.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

FinalStatus = Literal[
    "answer",             # 확정 답변 (ok + complete + verified)
    "partial_answer",     # 일부만 확정 + typed limitation
    "clarify",            # 역질문 (1단계 needs_clarification 또는 조회 중 모호성)
    "refuse",             # out_of_scope / unsupported / policy_refusal
    "not_found",          # 지원 범위에서 못 찾음 (부재 증명 아님)
    "failure",            # 코드·artifact 오류 — 안전 중단
]


class ClaimCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_id: str
    rcept_no: str | None = None
    evidence_id: str | None = None      # EvidenceCitation (fact/field/correction)
    section_id: str | None = None       # SourceReference (narrative)
    locator: str | None = None
    excerpt_prompt_safe: str | None = None
    verification: Literal["verified", "source_roundtrip"] = "verified"
    #: 사람이 원문에서 찾아갈 수 있는 자리. 이슈 #94 30 — 접수번호만으로는
    #: 그 값이 보고서 어디에 있는지 알 수 없다. `locator`
    #: (`TABLE[1]/TBODY[0]/TR[25]`)는 기계 좌표라 사용자에게 줄 수 없다.
    #:
    #: ``report_name``  「사업보고서 (2025.12)」 — 정본 `documents.report_nm`
    #: ``source_path``  「2-1. 연결 재무상태표 > 부채 > Ⅰ.유동부채」 —
    #:                  `facts.statement_title` + `facts.account_path`
    #:
    #: 둘 다 **없을 수 있다.** 사건·서술 인용은 재무제표 행이 아니라 이
    #: 좌표가 없고, 옛 정본 산출물에는 열 자체가 없다. 없으면 종전처럼
    #: 접수번호만 나간다.
    report_name: str | None = None
    source_path: str | None = None


class RankingEntry(BaseModel):
    """Internal Stage2 sidecar for one competition-ranked operand."""

    model_config = ConfigDict(extra="forbid")
    rank: int = Field(ge=1)
    output_id: str
    label: str
    value_text: str | None = None
    raw_unit: str | None = None
    canonical_value: str | None = None
    canonical_unit: str | None = None


class AnswerClaim(BaseModel):
    """답변을 구성하는 확정 사실 하나. output_id 단위."""
    model_config = ConfigDict(extra="forbid")
    output_id: str
    label: str                                  # 사람이 읽는 라벨 (예: 삼성전자 2025 CFS 매출액)
    value_text: str | None = None               # 원문 문자열 그대로 (예: "333,605,938")
    raw_unit: str | None = None                 # 원문 단위 (예: "백만원")
    canonical_value: str | None = None          # Decimal 문자열, 기준 단위
    canonical_unit: str | None = None           # 예: "원", "%"
    state: str | None = None                    # 사건 상태 등 비수치 값 (예: "terminated")
    text: str | None = None                     # 서술형 slot 값
    citations: list[ClaimCitation] = Field(default_factory=list)
    derived_from: list[str] = Field(default_factory=list)  # Derivation이면 operand output_id
    operator: str | None = None
    # 이슈 #124 — argmax 극값 방향. "argmax"가 아닌 claim은 뜻이 없으므로
    # 항상 기본값(maximum)이다.
    direction: Literal["maximum", "minimum"] = "maximum"
    ranking: list[RankingEntry] = Field(
        default_factory=list, exclude_if=lambda rows: not rows)

    @model_validator(mode="after")
    def validate_ranking(self) -> "AnswerClaim":
        if not self.ranking:
            return self
        if self.operator != "argmax" or len(self.ranking) not in {3, 4}:
            raise ValueError("ranking sidecar는 3~4항 argmax에만 필요합니다")
        output_ids = [entry.output_id for entry in self.ranking]
        if len(output_ids) != len(set(output_ids)) \
                or set(output_ids) != set(self.derived_from):
            raise ValueError("ranking sidecar operand inventory가 derived_from과 다릅니다")
        ranks = [entry.rank for entry in self.ranking]
        if ranks[0] != 1 or any(
                rank < ranks[index - 1]
                or (rank != ranks[index - 1] and rank != index + 1)
                for index, rank in enumerate(ranks[1:], start=1)):
            raise ValueError("ranking sidecar가 competition ranking 순서가 아닙니다")
        return self


class PremiseVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: str
    verdict: Literal["true", "false", "unverifiable"]
    detail: str | None = None
    compared_output_ids: list[str] = Field(default_factory=list)


class Limitation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    detail: str
    affected_doc_ids: list[str] = Field(default_factory=list)


class Clarification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clarification_id: str
    question: str
    targets: list[str] = Field(default_factory=list)   # 무엇을 묻는지 (scope, concept, event ...)
    options: dict[str, list[str]] = Field(default_factory=dict)


class TraceEvent(BaseModel):
    """think_trace의 최소 단위 — 실행 중 실제로 일어난 일만 기록 (사후 작문 금지)."""
    model_config = ConfigDict(extra="forbid")
    seq: int
    stage: Literal["handoff", "route", "tool", "evidence", "derivation", "premise", "verify", "compose"]
    summary: str                                 # prompt-safe 텍스트만
    detail: dict = Field(default_factory=dict)   # 인자·status 등 typed 값


class AnswerPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    handoff_id: str
    plan_revision: int = 0
    final_status: FinalStatus

    claims: list[AnswerClaim] = Field(default_factory=list)
    premise_verdicts: list[PremiseVerdict] = Field(default_factory=list)
    limitations: list[Limitation] = Field(default_factory=list)
    clarification: Clarification | None = None
    applied_defaults: list[str] = Field(default_factory=list)
    used_documents: list[str] = Field(default_factory=list)   # rcept_no 목록 (근거로 실제 사용)
    reasons: list[str] = Field(default_factory=list)          # refuse 사유 typed 코드
    trace: list[TraceEvent] = Field(default_factory=list)

    # Execution-only sidecars deliberately stay outside the public payload
    # schema.  In particular, NarrativeFanoutResult contains retrieval matrix
    # coordinates which guide composition but must never change /answer's
    # stable five-field response contract.
    _narrative_sidecars: tuple[object, ...] = PrivateAttr(default_factory=tuple)

    def set_narrative_sidecars(self, sidecars: list[object] | tuple[object, ...]) -> None:
        self._narrative_sidecars = tuple(sidecars)

    @property
    def narrative_sidecars(self) -> tuple[object, ...]:
        return self._narrative_sidecars

    @model_validator(mode="after")
    def enforce_status(self) -> "AnswerPayload":
        if self.final_status == "answer":
            if not self.claims:
                raise ValueError("answer requires at least one claim")
            if any(not c.citations for c in self.claims):
                raise ValueError("answer claims must all be cited")
            if self.clarification is not None:
                raise ValueError("answer cannot carry clarification")
        elif self.final_status == "partial_answer":
            if not self.claims or not self.limitations:
                raise ValueError("partial_answer requires claims and limitations")
        elif self.final_status == "clarify":
            if self.clarification is None or self.claims:
                raise ValueError("clarify requires clarification and no confirmed claims")
        else:  # refuse / not_found / failure
            if self.claims:
                raise ValueError(f"{self.final_status} cannot expose claims")
            if not (self.limitations or self.reasons):
                raise ValueError(f"{self.final_status} requires typed limitation or reason")
        return self

"""Proof-bound public selector facets for one canonical event.

The canonical event key remains the execution identity.  This module only
recovers user-visible selector facets when an exact canonical field on the
already selected event proves them.  It never discovers an event and it has no
question-ID or fixture dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any
import unicodedata

from agent.event_preflight import _contains_any, _counterparty_aliases, _key
from agent.semantic_intent_v1 import SemanticIntent


_CONTRACT_NAME_PATH = re.compile(r"(?:체결|해지)?\s*계약명$")
_COUNTERPARTY_PATH = re.compile(
    r"(?:계약\s*상대(?:방)?|발행\s*대상자명)$")
_PRODUCT_CONTRACT = re.compile(
    r"(?P<product>[가-힣A-Za-z0-9-]{2,})"
    r"(?:\s+(?:시스템|장비|제품|설비|기자재))?"
    r"\s+(?:공급|판매|구매|제조)\s*계약"
)
_GENERIC_CONTRACT_WORD = re.compile(
    r"(?:전기차|공급|판매|구매|제조|시스템|장비|제품|설비|기자재|계약)"
)


@dataclass(frozen=True, slots=True)
class SelectorFacetProof:
    source_receipt: str
    proof_ref: str

    def as_payload(self) -> dict[str, str]:
        return {
            "source_receipt": self.source_receipt,
            "proof_ref": self.proof_ref,
        }


@dataclass(frozen=True, slots=True)
class PublicEventSelector:
    counterparty: str | None = None
    counterparty_proof: SelectorFacetProof | None = None
    product_keywords: tuple[str, ...] = ()
    product_keyword_proofs: tuple[SelectorFacetProof, ...] = ()
    contract_name: str | None = None
    contract_name_proof: SelectorFacetProof | None = None

    def __post_init__(self) -> None:
        if (self.counterparty is None) != (self.counterparty_proof is None):
            raise ValueError("counterparty selector에는 짝지은 proof가 필요합니다")
        if len(self.product_keywords) != len(self.product_keyword_proofs):
            raise ValueError("product keyword와 proof 수가 다릅니다")
        if (self.contract_name is None) != (self.contract_name_proof is None):
            raise ValueError("contract-name selector에는 짝지은 proof가 필요합니다")

    @property
    def empty(self) -> bool:
        return not (self.counterparty or self.product_keywords or self.contract_name)

    def as_payload(self) -> dict[str, Any]:
        return {
            "counterparty": self.counterparty,
            "counterparty_proof": (
                self.counterparty_proof.as_payload()
                if self.counterparty_proof is not None else None
            ),
            "product_keywords": list(self.product_keywords),
            "product_keyword_proofs": [
                proof.as_payload() for proof in self.product_keyword_proofs
            ],
            "contract_name": self.contract_name,
            "contract_name_proof": (
                self.contract_name_proof.as_payload()
                if self.contract_name_proof is not None else None
            ),
        }


def _normalized(value: str) -> str:
    return re.sub(
        r"[^0-9a-z가-힣]+", "",
        unicodedata.normalize("NFKC", value).casefold(),
    )


def _value(row: Any) -> str:
    value = getattr(row, "value", None)
    return value.strip() if isinstance(value, str) else ""


def _proof(receipt: str, row: Any, role: str) -> SelectorFacetProof | None:
    evidence_id = getattr(row, "evidence_id", None)
    if not isinstance(evidence_id, str) or not re.fullmatch(r"[0-9a-f]{32}", evidence_id):
        return None
    return SelectorFacetProof(
        source_receipt=receipt,
        proof_ref=f"canonical:public-selector:{receipt}:{evidence_id}:{role}",
    )


def _unique_rows(rows: list[Any]) -> list[Any]:
    by_value: dict[str, Any] = {}
    for row in rows:
        value = _value(row)
        if value and value != "-":
            by_value.setdefault(_normalized(value), row)
    return list(by_value.values())


def _canonical_rows(canonical: Any, *, receipt: str, as_of: str) -> tuple[list[Any], list[Any]]:
    try:
        rows = list(canonical.fields(as_of=as_of, rcept_no=receipt))
    except (AttributeError, TypeError, ValueError):
        # A known canonical read/shape failure cannot invent a public facet;
        # unexpected programming errors must remain visible to the caller.
        return [], []
    contracts = _unique_rows([
        row for row in rows
        if _CONTRACT_NAME_PATH.search(str(getattr(row, "path", "") or ""))
    ])
    counterparties = _unique_rows([
        row for row in rows
        if _COUNTERPARTY_PATH.search(str(getattr(row, "path", "") or ""))
    ])
    return contracts, counterparties


def _public_counterparty(source: str, canonical_value: str) -> str | None:
    if not _contains_any(source, [canonical_value]):
        return None
    # Multi-word legal names are valid row selectors when the question gives
    # the entire canonical value literally.  Returning that exact public
    # value is stronger than reducing it to a guessed brand token and lets a
    # later field read bind to the matching row in a multi-party form.
    if _normalized(source) == _normalized(canonical_value):
        return source.strip()
    latin = re.findall(r"[A-Za-z][A-Za-z0-9.&'-]*", source)
    if len(latin) == 1:
        return latin[0]
    aliases = _counterparty_aliases().get(_key(source), ())
    matched = [value for value in aliases if _contains_any(value, [canonical_value])]
    if len(matched) != 1:
        return None
    brand = re.match(r"[A-Za-z][A-Za-z0-9.&'-]*", matched[0])
    return brand.group(0) if brand is not None else None


def _product_keyword(contract_name: str) -> str | None:
    match = _PRODUCT_CONTRACT.search(contract_name)
    return match.group("product") if match is not None else None


def _literal_contract_name(surface: str, canonical_name: str, product: str | None) -> str | None:
    candidate = re.sub(r"\s*계약\s*$", "", surface).strip().strip("'\" ")
    if len(_normalized(candidate)) < 3:
        return None
    if _normalized(candidate) not in _normalized(canonical_name):
        return None
    residual = candidate
    if product:
        residual = re.sub(re.escape(product), "", residual, flags=re.IGNORECASE)
    residual = _GENERIC_CONTRACT_WORD.sub("", residual)
    if not _normalized(residual):
        return None
    return candidate


def select_public_event_selector(
        canonical: Any, *, as_of: str, receipt: str,
        intent: SemanticIntent, item: Any, counterparty_surface: str | None,
        allow_canonical_product: bool = False,
        ) -> PublicEventSelector:
    """Return only facets proved on ``receipt`` of an already selected event."""

    contracts, counterparties = _canonical_rows(
        canonical, receipt=receipt, as_of=as_of)
    counterparty: str | None = None
    counterparty_proof: SelectorFacetProof | None = None
    if counterparty_surface is not None:
        matches = [
            row for row in counterparties
            if _contains_any(counterparty_surface, [_value(row)])
        ]
        if len(matches) == 1:
            public = _public_counterparty(counterparty_surface, _value(matches[0]))
            proof = _proof(receipt, matches[0], "counterparty")
            if public is not None and proof is not None:
                counterparty, counterparty_proof = public, proof

    contract_name: str | None = None
    contract_name_proof: SelectorFacetProof | None = None
    product_keywords: tuple[str, ...] = ()
    product_keyword_proofs: tuple[SelectorFacetProof, ...] = ()
    if len(contracts) == 1:
        row = contracts[0]
        canonical_name = _value(row)
        product = _product_keyword(canonical_name)
        row_proof = _proof(receipt, row, "contract-name")
        if row_proof is not None:
            literal = _literal_contract_name(
                str(item.target.surface), canonical_name, product)
            if literal is not None:
                contract_name = literal
                contract_name_proof = row_proof
            source_text = " ".join([
                str(item.target.surface),
                *(str(value) for value in item.output.field_surfaces),
                *(entity.surface for entity in intent.entities),
            ])
            product_is_literal = bool(
                product and _normalized(product) in _normalized(source_text))
            if product and (
                    product_is_literal or counterparty is not None
                    or (allow_canonical_product and contract_name is None)):
                product_keywords = (product,)
                product_keyword_proofs = (SelectorFacetProof(
                    source_receipt=receipt,
                    proof_ref=row_proof.proof_ref.replace(
                        ":contract-name", f":product:{_normalized(product)}"),
                ),)

    return PublicEventSelector(
        counterparty=counterparty,
        counterparty_proof=counterparty_proof,
        product_keywords=product_keywords,
        product_keyword_proofs=product_keyword_proofs,
        contract_name=contract_name,
        contract_name_proof=contract_name_proof,
    )


__all__ = [
    "PublicEventSelector", "SelectorFacetProof", "select_public_event_selector",
]

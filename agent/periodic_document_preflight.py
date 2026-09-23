"""Safe periodic-document selector preflight.

The preflight deliberately consumes document metadata only.  It does not read
question IDs, Gold fixtures, document bodies, or project-specific receipt
constants.  A canonical reader may expose any concrete row type as long as it
implements the small metadata/query protocol below.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Literal, Mapping, Protocol


PeriodicDocumentStatus = Literal[
    "resolved", "ambiguous", "not_found", "unsupported"]


class PeriodicDocumentReader(Protocol):
    """Minimum canonical-like reader surface used by this preflight."""

    def documents(self, **kwargs: Any) -> Iterable[Any]:
        """Return safe document metadata rows for the supplied filters."""

    def latest_document_version(self, rcept_no: str, as_of: str) -> Any:
        """Collapse one non-correction seed to its latest visible version."""


class PeriodicDocumentPreflightError(ValueError):
    """Invalid caller input or malformed safe document metadata."""


@dataclass(frozen=True, slots=True)
class PeriodicDocumentCandidate:
    """Public, safe candidate projection; no raw/body metadata is exposed."""

    rcept_no: str
    report_nm: str
    form: str
    label: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9]{14}", self.rcept_no):
            raise PeriodicDocumentPreflightError("candidate receipt 형식이 잘못되었습니다")
        if not self.report_nm.strip() or self.form not in {"quarter", "half", "annual"}:
            raise PeriodicDocumentPreflightError("candidate report/form이 잘못되었습니다")
        if not self.label.strip():
            raise PeriodicDocumentPreflightError("candidate label은 비어 있을 수 없습니다")


@dataclass(frozen=True, slots=True)
class PeriodicDocumentResolution:
    status: PeriodicDocumentStatus
    candidates: tuple[PeriodicDocumentCandidate, ...] = ()
    candidate: PeriodicDocumentCandidate | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status == "resolved":
            if self.candidate is None or len(self.candidates) != 1:
                raise PeriodicDocumentPreflightError(
                    "resolved 결과는 candidate 하나를 가져야 합니다")
            if self.candidates[0] != self.candidate:
                raise PeriodicDocumentPreflightError(
                    "resolved candidate와 candidates가 다릅니다")
        elif self.candidate is not None:
            raise PeriodicDocumentPreflightError(
                "비resolved 결과에는 candidate를 둘 수 없습니다")
        if len(self.candidates) != len(set(self.candidates)):
            raise PeriodicDocumentPreflightError("candidate 중복은 허용되지 않습니다")


@dataclass(frozen=True, slots=True)
class _Selector:
    form: Literal["quarter", "half", "annual"] | None
    year: int | None
    month: int | None
    latest: bool


@dataclass(frozen=True, slots=True)
class _SafeDocument:
    rcept_no: str
    report_nm: str
    form: Literal["quarter", "half", "annual"] | None
    corp_code: str
    doc_group: str
    rcept_dt: str
    year: int | None
    month: int | None
    is_correction: bool


_YEAR = r"(?P<year>20[0-9]{2}|[0-9]{2})"
_QUARTER = r"(?P<quarter>[13])\s*(?:분기|Q)"
# form 을 밝히지 않은 「최근 정기보고서」는 form 을 가리지 않는다.
_LATEST = re.compile(
    r"(?:가장)?최근(?P<form>분기|반기|사업|연간|정기)(?:보고서)?"
    r"|(?:가장)?최근보고서")
_LATEST_FORMS: dict[str, str | None] = {
    "분기": "quarter", "반기": "half",
    "사업": "annual", "연간": "annual",
    "정기": None, "": None,
}


def _year(value: str) -> int:
    number = int(value)
    return number if len(value) == 4 else 2000 + number


def parse_periodic_expression(expression: str) -> _Selector:
    """Parse the deliberately small explicit/latest periodic selector grammar."""

    if not isinstance(expression, str) or not expression.strip():
        raise PeriodicDocumentPreflightError("target_expression은 비어 있을 수 없습니다")
    text = re.sub(r"\s+", "", expression.strip())

    # 「최근」은 분기보고서만의 말이 아니다.  정기보고서 어느 form 에나 붙고,
    # 「가장 최근」처럼 강조가 앞에 오기도 한다.  form 을 안 밝히면(=정기보고서)
    # 어느 form 이든 가장 새 기간을 고른다 — 그게 「정기보고서」의 뜻이다.
    latest = _LATEST.fullmatch(text)
    if latest is not None:
        return _Selector(_LATEST_FORMS[latest.group("form") or ""],
                         None, None, True)

    match = re.fullmatch(_YEAR + r"년?" + _QUARTER + r"(?:보고서)?", text,
                         flags=re.IGNORECASE)
    if match:
        quarter = int(match.group("quarter"))
        return _Selector("quarter", _year(match.group("year")),
                         3 if quarter == 1 else 9, False)

    match = re.fullmatch(_YEAR + r"년?(?:상반기|반기)(?:보고서)?", text)
    if match:
        return _Selector("half", _year(match.group("year")), 6, False)

    match = re.fullmatch(_YEAR + r"년?(?:사업보고서|연간보고서|연간보고서)", text)
    if match:
        return _Selector("annual", _year(match.group("year")), 12, False)

    raise PeriodicDocumentPreflightError(
        "지원하지 않는 periodic target expression입니다")


_SAFE_FIELDS = frozenset({
    "rcept_no", "report_nm", "doc_group", "corp_code", "rcept_dt",
    "base_year", "base_month", "doc_subtype", "form", "is_correction",
})


def _value(row: Any, key: str) -> Any:
    if key not in _SAFE_FIELDS:
        raise AssertionError(f"unsafe metadata key: {key}")
    if isinstance(row, Mapping):
        return row.get(key)
    return getattr(row, key, None)


def _text(value: Any, *, field: str, required: bool = True) -> str:
    if not isinstance(value, str):
        if required:
            raise PeriodicDocumentPreflightError(f"document {field}가 문자열이 아닙니다")
        return ""
    value = value.strip()
    if required and not value:
        raise PeriodicDocumentPreflightError(f"document {field}가 비어 있습니다")
    return value


def _integer(value: Any, *, field: str, required: bool = False) -> int | None:
    if value is None or value == "":
        if required:
            raise PeriodicDocumentPreflightError(f"document {field}가 없습니다")
        return None
    if type(value) is int:
        result = value
    elif isinstance(value, str) and re.fullmatch(r"[0-9]{1,4}", value.strip()):
        result = int(value.strip())
    else:
        raise PeriodicDocumentPreflightError(f"document {field}가 정수가 아닙니다")
    if field == "base_year" and not 1900 <= result <= 2100:
        raise PeriodicDocumentPreflightError("document base_year 범위가 잘못되었습니다")
    if field == "base_month" and result not in {3, 6, 9, 12}:
        raise PeriodicDocumentPreflightError("document base_month 범위가 잘못되었습니다")
    return result


def _form(row: Any, report_nm: str) -> Literal["quarter", "half", "annual"]:
    explicit = _value(row, "form")
    if explicit in {"quarter", "half", "annual"}:
        return explicit
    subtype = _value(row, "doc_subtype")
    if isinstance(subtype, str):
        key = subtype.casefold()
        if key in {"quarter", "quarterly"}:
            return "quarter"
        if key in {"half", "semiannual", "semi-annual"}:
            return "half"
        if key in {"annual", "yearly"}:
            return "annual"
    compact = re.sub(r"\s+", "", report_nm)
    if "분기보고서" in compact:
        return "quarter"
    if "반기보고서" in compact:
        return "half"
    if "사업보고서" in compact or "연간보고서" in compact:
        return "annual"
    raise PeriodicDocumentPreflightError("periodic form을 판정할 수 없습니다")


def _safe_document(row: Any) -> _SafeDocument:
    report_nm = _text(_value(row, "report_nm"), field="report_nm")
    rcept_no = _text(_value(row, "rcept_no"), field="rcept_no")
    if not re.fullmatch(r"[0-9]{14}", rcept_no):
        raise PeriodicDocumentPreflightError("document rcept_no 형식이 잘못되었습니다")
    doc_group = _text(_value(row, "doc_group"), field="doc_group")
    corp_code = _text(_value(row, "corp_code"), field="corp_code")
    rcept_dt = _text(_value(row, "rcept_dt"), field="rcept_dt")
    if not re.fullmatch(r"[0-9]{8}", rcept_dt):
        raise PeriodicDocumentPreflightError("document rcept_dt 형식이 잘못되었습니다")
    correction = _value(row, "is_correction")
    if type(correction) is not bool:
        raise PeriodicDocumentPreflightError("document is_correction은 bool이어야 합니다")
    return _SafeDocument(
        rcept_no=rcept_no, report_nm=report_nm, form=_form(row, report_nm),
        corp_code=corp_code, doc_group=doc_group, rcept_dt=rcept_dt,
        year=_integer(_value(row, "base_year"), field="base_year"),
        month=_integer(_value(row, "base_month"), field="base_month"),
        is_correction=correction,
    )


def _matches(row: _SafeDocument, selector: _Selector, *, corp_code: str,
             as_of: str) -> bool:
    if row.doc_group != "periodic" or row.corp_code != corp_code or row.rcept_dt > as_of:
        return False
    if selector.form is not None and row.form != selector.form:
        return False
    if selector.year is not None and (row.year, row.month) != (
            selector.year, selector.month):
        return False
    return row.year is not None and row.month is not None


def _sort_key(row: _SafeDocument) -> tuple[int, int, str, str]:
    return (row.year or 0, row.month or 0, row.rcept_dt, row.rcept_no)


def _candidate(row: _SafeDocument) -> PeriodicDocumentCandidate:
    return PeriodicDocumentCandidate(
        rcept_no=row.rcept_no, report_nm=row.report_nm, form=row.form,
        label=f"{row.report_nm} · {row.rcept_no}")


class PeriodicDocumentPreflight:
    """Resolve explicit/latest periodic metadata without body or Gold access."""

    def __init__(self, reader: PeriodicDocumentReader) -> None:
        if not callable(getattr(reader, "documents", None)):
            raise TypeError("reader.documents가 필요합니다")
        if not callable(getattr(reader, "latest_document_version", None)):
            raise TypeError("reader.latest_document_version가 필요합니다")
        self._reader = reader

    def resolve_periodic_document(
            self, *, corp_code: str, as_of: str, target_expression: str,
            selected_receipt: str | None = None,
            ) -> PeriodicDocumentResolution:
        if not isinstance(corp_code, str) or not corp_code.strip():
            raise PeriodicDocumentPreflightError("corp_code는 비어 있을 수 없습니다")
        if not isinstance(as_of, str) or not re.fullmatch(r"[0-9]{8}", as_of):
            raise PeriodicDocumentPreflightError("as_of는 YYYYMMDD여야 합니다")
        if selected_receipt is not None and not re.fullmatch(
                r"[0-9]{14}", selected_receipt):
            raise PeriodicDocumentPreflightError("selected_receipt 형식이 잘못되었습니다")
        try:
            selector = parse_periodic_expression(target_expression)
        except PeriodicDocumentPreflightError as exc:
            return PeriodicDocumentResolution("unsupported", reason=str(exc))

        raw_rows = list(self._reader.documents(
            corp_code=corp_code.strip(), as_of=as_of, doc_group="periodic"))
        safe_rows = [_safe_document(row) for row in raw_rows]
        rows = [
            row for row in safe_rows
            if _matches(row, selector, corp_code=corp_code.strip(), as_of=as_of)
        ]
        if not rows:
            return PeriodicDocumentResolution("not_found")

        # For latest, first choose the newest reporting period.  An explicit
        # period already has this property by construction.
        top_period = max(_sort_key(row)[:2] for row in rows)
        rows = [row for row in rows if _sort_key(row)[:2] == top_period]
        seeds = [row for row in rows if not row.is_correction]
        if len(seeds) == 1:
            seed = seeds[0]
            detailed_resolver = getattr(
                self._reader, "resolve_document_version", None)
            detailed = (
                detailed_resolver(seed.rcept_no, as_of=as_of)
                if callable(detailed_resolver) else None
            )
            if detailed is not None and getattr(detailed, "status", None) != "ok":
                status = getattr(detailed, "status", None)
                reason = str(getattr(detailed, "reason", None) or status)
                if status == "ambiguous":
                    leaves = tuple(getattr(detailed, "leaves", ()) or ())
                    lineage_rows = [
                        row for row in safe_rows
                        if row.rcept_no in leaves
                        and _matches(row, selector, corp_code=corp_code.strip(),
                                     as_of=as_of)
                        and _sort_key(row)[:2] == top_period
                    ]
                    lineage_rows.sort(key=_sort_key, reverse=True)
                    lineage_candidates = tuple(
                        _candidate(row) for row in lineage_rows)
                    if (selected_receipt is not None
                            and selected_receipt in {
                                row.rcept_no for row in lineage_candidates
                            }):
                        chosen = next(
                            row for row in lineage_candidates
                            if row.rcept_no == selected_receipt)
                        return PeriodicDocumentResolution(
                            "resolved", candidates=(chosen,), candidate=chosen)
                    return PeriodicDocumentResolution(
                        "ambiguous", candidates=lineage_candidates,
                        reason=f"correction_lineage_ambiguous:{reason}")
                if status == "invalid":
                    return PeriodicDocumentResolution(
                        "unsupported",
                        reason=f"correction_lineage_invalid:{reason}")
                return PeriodicDocumentResolution(
                    "not_found",
                    reason=f"correction_lineage_not_found:{reason}")

            latest = (
                getattr(detailed, "selected", None)
                if detailed is not None else
                self._reader.latest_document_version(
                    seed.rcept_no, as_of=as_of)
            )
            if latest is not None:
                if isinstance(latest, str):
                    latest_row = next(
                        (row for row in safe_rows if row.rcept_no == latest),
                        None,
                    )
                    if latest_row is None:
                        return PeriodicDocumentResolution(
                            "not_found",
                            reason="latest_document_version_metadata_missing",
                        )
                else:
                    latest_row = _safe_document(latest)
                if _matches(latest_row, selector, corp_code=corp_code.strip(),
                            as_of=as_of) and _sort_key(latest_row)[:2] == top_period:
                    rows = [latest_row]
                else:
                    rows = [seed]
            else:
                rows = [seed]
        elif len(seeds) > 1:
            rows = seeds
        else:
            rows = [row for row in rows if row.is_correction]

        rows.sort(key=_sort_key, reverse=True)
        candidates = tuple(_candidate(row) for row in rows)
        if selected_receipt is not None:
            if selected_receipt not in {row.rcept_no for row in candidates}:
                return PeriodicDocumentResolution(
                    "not_found", candidates=candidates,
                    reason="selected_receipt_not_in_candidates")
            candidates = tuple(row for row in candidates
                               if row.rcept_no == selected_receipt)
        if len(candidates) == 1:
            return PeriodicDocumentResolution(
                "resolved", candidates=candidates, candidate=candidates[0])
        return PeriodicDocumentResolution("ambiguous", candidates=candidates)


# Semantic alias used at the composition boundary.  The implementation itself
# remains reader-generic; CanonicalReadModel satisfies the protocol directly.
CanonicalPeriodicDocumentPreflight = PeriodicDocumentPreflight


__all__ = [
    "CanonicalPeriodicDocumentPreflight", "PeriodicDocumentCandidate",
    "PeriodicDocumentPreflight",
    "PeriodicDocumentPreflightError", "PeriodicDocumentReader",
    "PeriodicDocumentResolution", "PeriodicDocumentStatus",
    "parse_periodic_expression",
]

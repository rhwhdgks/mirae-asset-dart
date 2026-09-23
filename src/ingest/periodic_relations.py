"""정기공시 정정 계보 — 기준기간으로 잇는다.

거래소공시는 href(KIND 접수번호), 주요사항·지분공시는 `<CORRECTION>` 의 최초제출일로
이었다. 정기공시는 **기준기간이 메타데이터에 있어서 더 정확하다.**

```text
key = (corp_code, doc_subtype, base_year, base_month)
```

같은 기업의 같은 회계기간·같은 보고서 유형은 하나의 계보다. 실측(159건):

| 직전본 수 | 건수 | 처리 |
|---:|---:|---|
| 1 | 125 | `resolved` — 직전 접수본에 연결 |
| 2~3 | 21 | `resolved` — 접수일 최근순 직전본 |
| 0 (이후만 존재) | 11 | 접수 순서 역전. 날짜만으로 방향을 못 정한다 |
| 0 (정정본만) | 2 | `root_missing` — 한화솔루션 2023.1Q·2025 연간 |

**선언된 최초제출일과 교차 확인**한다. XML `<CORRECTION>` 또는 PDF 정정 헤더의
최초제출일이 후보의 접수일과 맞으면 `resolved` 다. DART 접수일과 내부 제출일이
어긋난 경우에도 후보 XML 표지일이 유일하게 일치할 때만 확정하고, 그 밖에는
`candidate` 로 닫는다 — 같은 기간에 여러 정정이 얽힐 때 직전본이 항상 정정 대상은
아니기 때문이다.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import asdict
from datetime import date as calendar_date
from pathlib import Path

from .corpus_paths import CorpusIndex, iter_manifest
from .exchange_events import precedes
from .exchange_events import RelationEdge
from .form_events import correction_header
from .relation_version import RELATION_RESOLVER_VERSION
from .sanitize import sanitize
from ..artifact import file_sha256, write_stamp

__all__ = ["build"]

_DATE_TOKEN = re.compile(
    r"(?<!\d)(\d{4})\s*(?:년|[-.])\s*(\d{1,2})\s*(?:월|[-.])\s*(\d{1,2})\s*일?"
)
_FIRST_SUBMISSION = re.compile(r"최초제출일\s*[:：]")
_EXCHANGE_ADDRESSEE = re.compile(r"한국거래소\s*귀중")
#: PDF 전체(최대 1,085쪽)를 읽지 않는다. 실측 두 PDF 모두 목차 뒤 물리 5쪽에
#: 정정 헤더가 있으며, 넉넉히 12쪽까지만 검사한다.
_PDF_HEADER_PAGE_LIMIT = 12


def _parse(path: Path):
    cleaned = sanitize(path.read_text(encoding="utf-8", errors="replace"))
    try:
        return ET.fromstring(cleaned)
    except ET.ParseError:
        from lxml import etree as lxml_etree
        return lxml_etree.fromstring(
            cleaned.encode("utf-8"),
            lxml_etree.XMLParser(recover=True, huge_tree=True, encoding="utf-8"),
        )


def _first_valid_date(text: str) -> str | None:
    """문자열에서 실제 달력에 존재하는 첫 날짜만 정규화한다."""
    for match in _DATE_TOKEN.finditer(text):
        year, month, day = (int(value) for value in match.groups())
        try:
            calendar_date(year, month, day)
        except ValueError:
            continue
        return f"{year:04d}-{month:02d}-{day:02d}"
    return None


def _labelled_submission_date(text: str) -> str | None:
    """`최초제출일:` 바로 뒤 80자 안의 날짜만 인정한다."""
    marker = _FIRST_SUBMISSION.search(text)
    if marker is None:
        return None
    return _first_valid_date(text[marker.end():marker.end() + 80])


def _pdf_submission_date(index: CorpusIndex, record: dict) -> tuple[str | None, str | None]:
    """PDF text layer의 정정 헤더에서 최초제출일과 page 근거를 읽는다.

    문서 폴더에 PDF가 정확히 하나일 때만 시도하고, 여러 PDF/암호화/텍스트 추출
    실패는 자동 선택하지 않는다.
    """
    pdfs = [path for path in index.files(record["file_path"])
            if path.suffix.lower() == ".pdf"]
    if len(pdfs) != 1:
        return None, None
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdfs[0]))
        found: list[tuple[str, int]] = []
        for page_no, page in enumerate(reader.pages[:_PDF_HEADER_PAGE_LIMIT], start=1):
            value = _labelled_submission_date(page.extract_text() or "")
            if value:
                found.append((value, page_no))
    except Exception as exc:  # noqa: BLE001 — PDF 증거가 없으면 candidate를 유지한다 (실패는 집계한다)
        _note_parse_failure("pdf_header", pdfs[0], exc)
        return None, None
    # 서로 다른 헤더 날짜가 보이면 어느 것이 정정대상인지 자동 선택하지 않는다.
    dates = {value for value, _ in found}
    if len(dates) != 1:
        return None, None
    value = next(iter(dates))
    page_no = next(page for candidate, page in found if candidate == value)
    return value, f"pdf:PAGE[{page_no}]"


#: 이 실행에서 삼킨 파싱 실패. fail-closed(candidate 유지)는 맞지만 **몇 건을 못 읽었는지**는 보고서에 남아야
#: 「정정 계보를 다 봤다」와 「못 읽어서 못 이었다」가 구분된다.
_PARSE_FAILURES: list[dict] = []


def _note_parse_failure(stage: str, path: Path, exc: BaseException) -> None:
    _PARSE_FAILURES.append({"stage": stage, "path": path.name, "error": type(exc).__name__})


def _cover_submission_date(path: Path) -> str | None:
    """원본 보고서 표지의 `한국거래소 귀중` 옆 내부 제출일을 읽는다.

    같은 XML에 서로 다른 표지 날짜가 반복되면 fail-closed로 ``None``이다.
    """
    try:
        root = _parse(path)
        text = " ".join("".join(root.itertext()).split())
    except Exception as exc:  # noqa: BLE001 — 후보 내부 증거가 없으면 확정하지 않는다 (실패는 집계한다)
        _note_parse_failure("cover_date", path, exc)
        return None
    found: set[str] = set()
    for marker in _EXCHANGE_ADDRESSEE.finditer(text):
        value = _first_valid_date(text[marker.end():marker.end() + 100])
        if value:
            found.add(value)
    return next(iter(found)) if len(found) == 1 else None


def _internal_date_matches(index: CorpusIndex, candidates: list[dict],
                           declared: str) -> list[dict]:
    """선언일과 후보 XML 내부 표지일이 같은 후보만 반환한다."""
    matches: list[dict] = []
    for candidate in candidates:
        path = index.main_xml(candidate["file_path"], candidate["rcept_no"])
        if path is not None and _cover_submission_date(path) == declared:
            matches.append(candidate)
    return matches


def _correction_relation(index: CorpusIndex, record: dict,
                         peers: list[dict]) -> RelationEdge:
    """정기공시 정정 1건을 같은 기준기간 후보 안에서 fail-closed로 판정한다."""
    src = record["rcept_no"]
    # **날짜가 아니라 접수번호로 비교한다.** 정기공시는 원본과 정정본이
    # 같은 날 올라오는 일이 잦고 접수번호가 같은 날 안의 순서도 보존한다.
    earlier = [candidate for candidate in peers
               if precedes(candidate["rcept_no"], src)]

    declared = None
    declared_source = None
    path = index.main_xml(record["file_path"], src)
    if path is not None:
        try:
            _, declared = correction_header(_parse(path))
            if declared:
                declared_source = "xml:CORRECTION"
        except Exception as exc:  # noqa: BLE001 — 선언일 추출 실패는 치명적이지 않다 (실패는 집계한다)
            _note_parse_failure("correction_header", path, exc)
            declared = None
    else:
        declared, declared_source = _pdf_submission_date(index, record)
    hint = f"{declared or ''} {record['report_nm']}".strip() or None
    feat = "|".join(str(value) for value in (
        record["corp_code"], record["doc_subtype"],
        record["base_year"], record["base_month"], declared or "-",
        declared_source or "source:-"))
    match_basis = None

    if not earlier:
        same_period = [candidate for candidate in peers
                       if candidate["rcept_no"] != src]
        state, dst, conf = (
            ("ambiguous", None, 0.3) if same_period
            else ("root_missing", None, 0.0)
        )
        cands = tuple(candidate["rcept_no"] for candidate in same_period)
        reason = None if same_period else "no_earlier_in_period"
    else:
        cands = tuple(candidate["rcept_no"] for candidate in earlier)
        reason = None
        target = earlier[-1]
        if declared and declared.replace("-", "") == target["rcept_dt"]:
            state, dst, conf = "resolved", target["rcept_no"], 1.0
            match_basis = "receipt_date"
        elif declared:
            matches = [candidate for candidate in earlier
                       if candidate["rcept_dt"] == declared.replace("-", "")]
            if len(matches) == 1:
                state, dst, conf = "resolved", matches[0]["rcept_no"], 0.9
                match_basis = "receipt_date"
            else:
                # DART 접수일과 보고서 내부 제출일이 하루 어긋날 수 있다.
                # 선언일과 후보 내부 표지일이 **유일하게** 같은 때만 확정한다.
                internal = _internal_date_matches(index, earlier, declared)
                if len(internal) == 1:
                    state, dst, conf = "resolved", internal[0]["rcept_no"], 0.9
                    match_basis = "candidate_internal_cover_date"
                else:
                    state, dst, conf = "candidate", target["rcept_no"], 0.5
        else:
            state, dst, conf = "candidate", target["rcept_no"], 0.6

    if match_basis:
        feat += f"|match:{match_basis}"
    return RelationEdge(src, dst, "CORRECTS", state, hint, conf,
                        cands, feat, root_missing_reason=reason)


def build(corpus_root: str | Path = "data/corpus",
          out_dir: str | Path = "out/periodic") -> dict:
    index = CorpusIndex.build(corpus_root)
    records = [r for r in iter_manifest(corpus_root) if r["doc_group"] == "periodic"]

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["corp_code"], r["doc_subtype"], r["base_year"], r["base_month"])].append(r)
    for v in groups.values():
        v.sort(key=lambda x: x["rcept_no"])      # 같은 날 여러 건이면 일련번호가 순서다

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, int] = {}
    edges = 0
    _PARSE_FAILURES.clear()

    with (out_dir / "relations.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            if not record["is_correction"]:
                continue
            peers = groups[(record["corp_code"], record["doc_subtype"],
                            record["base_year"], record["base_month"])]
            edge = _correction_relation(index, record, peers)
            fh.write(json.dumps(asdict(edge), ensure_ascii=False) + "\n")
            edges += 1
            status[edge.resolution_status] = status.get(edge.resolution_status, 0) + 1

    failures_by_stage: dict[str, int] = {}
    for failure in _PARSE_FAILURES:
        failures_by_stage[failure["stage"]] = failures_by_stage.get(failure["stage"], 0) + 1
    report = {"corrections": sum(1 for r in records if r["is_correction"]),
              "relations": edges, "status": dict(sorted(status.items())),
              "parse_failures": len(_PARSE_FAILURES),
              "parse_failures_by_stage": dict(sorted(failures_by_stage.items())),
              "parse_failure_samples": list(_PARSE_FAILURES[:20])}
    (out_dir / "relations_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 이 산출물이 **무엇으로 만들어졌는지** 남긴다. canonical 이 읽기 전에 검사한다 (B-03).
    write_stamp(out_dir, "periodic_relations",
                file_sha256(Path(corpus_root) / "manifest.jsonl"),
                resolver_version=RELATION_RESOLVER_VERSION)
    return report


def _main() -> int:
    r = build()
    print(f"정기공시 정정 {r['corrections']}건 → 관계 {r['relations']}개")
    for k, v in r["status"].items():
        print(f"  CORRECTS:{k:<16}{v:>5}")
    if r["parse_failures"]:
        print(f"  파싱 실패(집계만, candidate 유지) {r['parse_failures']}건: {r['parse_failures_by_stage']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

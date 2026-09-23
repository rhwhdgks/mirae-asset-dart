"""주요사항보고서(598) · 지분공시(1,083) → 정형 이벤트 + 정정 관계.

거래소공시와 달리 DART XML 이므로 `dart_form.extract_document()` 로 필드를 뽑는다.
`ACODE` 가 있어 라벨 문구 변화에 강하다.

## 유형 분류

`doc_subtype` 은 major 598건 전부 `null` 이다. `report_nm` 괄호 안을 유형으로 쓴다
(`주요사항보고서(유상증자결정)` → `유상증자결정`, 정규화 후 29종).
holding 은 `주식등의대량보유상황보고서(일반|약식)` 2종이다.

## 정정 연결

거래소공시는 href 에 원공시 접수번호가 있지만 **DART XML 에는 없다.**
`<CORRECTION>` 블록의 `정정대상 공시서류` + `최초제출일` 로 후보를 만들고,
법인·유형·날짜가 유일할 때만 `resolved` 로 확정한다.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .corpus_paths import CorpusIndex, iter_manifest
from .dart_form import FormField, extract_document
from .exchange_events import _missing_reason, precedes
from .exchange_events import RelationEdge
from .exchange_html import normalize_amount
from .relation_version import RELATION_RESOLVER_VERSION
from .sanitize import sanitize
from ..artifact import file_sha256, write_stamp

__all__ = ["FormEvent", "build", "event_type_of"]

_TAG = re.compile(r"^\[[^\]]+\]")
_PAREN = re.compile(r"\(([^)]+)\)\s*$")
#: 날짜 표기가 원문마다 다르다. 실측된 변형:
#:   `2024년 11월 18일` · `2023-01-27` · `2023.12.29.` · `2025년 08년 28일`(원문 오타)
#: 구분자를 느슨하게 받고 월·일 범위로 검증한다.
_ANY_DATE = re.compile(r"(\d{4})\s*[-.년]\s*(\d{1,2})\s*[-.월년]\s*(\d{1,2})")

#: `<CORRECTION>` 블록 헤더. 3개 문서군(major·holding·periodic) 공통 문구다.
#:   `1. 정정대상 공시서류 : 주요사항보고서(자기주식취득결정)`
#:   `2. 정정대상 공시서류의 최초제출일 : 2024년 11월 18일`
#: 이 블록은 TABLE 이 아니라 P 안에 있어 표 추출기로는 잡히지 않는다.
_CORR_DOC = re.compile(r"정정대상\s*공시서류\s*[:：]\s*(.{1,60}?)\s*(?:2\s*[.．]|최초제출일)")
#: 날짜 표기가 문서마다 다르다 — `2024년 11월 18일` 과 `2023-01-27` 이 섞여 있다.
#: 뒤 30자를 잡아 `_norm_date` 가 두 형식을 모두 처리하게 한다.
_CORR_DATE = re.compile(r"최초제출일\s*[:：]\s*(.{0,30})")

#: 개인정보 후보 — 지분공시 전량(100%)에 생년월일 라벨이 있다.
_PII_LABELS = ("생년월일", "주소", "전화번호", "팩스번호", "이메일", "성명", "직업")


def event_type_of(report_nm: str) -> str:
    """`[기재정정]주요사항보고서(유상증자결정)` → `유상증자결정`."""
    base = _TAG.sub("", report_nm).strip()
    m = _PAREN.search(base)
    return m.group(1) if m else base


@dataclass
class FormEvent:
    event_id: str
    doc_id: str
    rcept_no: str
    corp_code: str
    corp_name: str
    sector: str
    doc_group: str
    event_type: str
    is_correction: bool
    disclosed_at: str
    filer: str                              #: 지분공시는 발행회사가 아니라 보고자다
    parse_mode: str
    decision_date: str | None               #: 이사회결의일 등
    amount: int | None
    fields: dict[str, str] = field(default_factory=dict)      #: 라벨경로 → 값
    acodes: dict[str, str] = field(default_factory=dict)      #: ACODE → 값
    pii_labels: list[str] = field(default_factory=list)       #: 마스킹 대상 라벨


def _norm_date(raw: str | None) -> str | None:
    if not raw:
        return None
    for m in _ANY_DATE.finditer(raw):
        y, mo, d = (int(x) for x in m.groups())
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y}-{mo:02d}-{d:02d}"
    return None


def _pick(acodes: dict[str, str], fields: dict[str, str], *keys: str) -> str | None:
    for k in keys:
        if k in acodes and acodes[k].strip() not in ("", "-"):
            return acodes[k]
    for k in keys:
        for label, value in fields.items():
            if k in label and value.strip() not in ("", "-"):
                return value
    return None


def _to_event(record: dict, fields: list[FormField], parse_mode: str) -> FormEvent:
    by_label = {f.label_path: f.value for f in fields}
    by_acode = {f.acode: f.value for f in fields if f.acode}
    pii = sorted({lab for lab in by_label if any(p in lab for p in _PII_LABELS)})

    return FormEvent(
        event_id=f"{record['doc_group']}_{record['rcept_no']}",
        doc_id=record["doc_id"], rcept_no=record["rcept_no"],
        corp_code=record["corp_code"], corp_name=record["corp_name"],
        sector=record["sector"], doc_group=record["doc_group"],
        event_type=event_type_of(record["report_nm"]),
        is_correction=record["is_correction"],
        disclosed_at=record["rcept_dt"], filer=record["flr_nm"],
        parse_mode=parse_mode,
        decision_date=_norm_date(
            _pick(by_acode, by_label, "DRC_DT", "이사회결의일", "결정일",
                  "보고서작성기준일", "보고의무발생일")
        ),
        amount=normalize_amount(
            _pick(by_acode, by_label, "권면(전자등록)총액", "취득예정금액",
                  "처분예정금액", "계약금액", "양수금액", "양도금액")
        ),
        fields=by_label, acodes=by_acode, pii_labels=pii,
    )


def correction_header(root) -> tuple[str | None, str | None]:
    """`<CORRECTION>` 블록에서 (정정대상 서류명, 최초제출일) 을 뽑는다."""
    for node in root.iter("CORRECTION"):
        text = " ".join("".join(node.itertext()).split())
        doc = _CORR_DOC.search(text)
        date = _CORR_DATE.search(text)
        return (doc.group(1).strip() if doc else None,
                _norm_date(date.group(1)) if date else None)
    return None, None


def _squeeze(text: str | None) -> str:
    """`회사합병 결정` 과 `회사합병결정` 을 같게 본다. 선언 서류명과 report_nm 의
    띄어쓰기가 다르다."""
    return "".join((text or "").split())


def _correction_edge(record: dict, target_nm: str | None, submitted: str | None,
                     by_key: dict[tuple, list[str]], known: dict[str, dict]) -> RelationEdge:
    src = record["rcept_no"]
    hint = " ".join(x for x in (submitted, target_nm) if x) or None
    if not submitted:
        return RelationEdge(src, None, "CORRECTS", "parse_failed", hint, 0.0,
                            root_missing_reason="no_submitted_date")

    # 유형을 키에 넣지 않으면 같은 날 여러 건을 제출한 기업에서 오연결이 난다.
    # 두산로보틱스 2024-07-11 은 3건 제출인데 코퍼스엔 1건만 있어,
    # 유형 없이 매칭하면 회사합병·주식교환 정정 12건이 타법인양수 원본에 붙었다.
    etype = event_type_of(record["report_nm"])
    # **법인 키는 `corp_code`.** 회사명은 바뀐다 — `exchange_events` 는 고쳤는데
    # 여기가 남아 있었다 (7차 검수). 상호변경이 있으면 정정이 원본을 못 찾는다.
    key = (record["corp_code"], record["doc_group"], submitted.replace("-", ""), etype)
    # 원본은 정정본보다 **먼저 접수돼야 한다** — 내용을 보지 않고 후보를 거른다
    cands = [r for r in by_key.get(key, []) if r != src and precedes(r, src)]

    # 선언 서류명과도 대조해 한 번 더 거른다 (띄어쓰기 무시)
    if len(cands) > 1 and target_nm:
        want = _squeeze(target_nm)
        narrowed = [r for r in cands
                    if want and want in _squeeze(known.get(r, {}).get("report_nm"))]
        if len(narrowed) == 1:
            cands = narrowed
    feat = "|".join(str(x) for x in key)
    if len(cands) == 1:
        return RelationEdge(src, cands[0], "CORRECTS", "resolved", hint, 0.8,
                            tuple(cands), feat)
    if len(cands) > 1:
        return RelationEdge(src, None, "CORRECTS", "ambiguous", hint, 0.3,
                            tuple(cands), feat)
    by_corp_date = {(k[0], k[2]) for k in by_key}
    return RelationEdge(
        src, None, "CORRECTS", "root_missing", hint, 0.0, (), feat,
        root_missing_reason=_missing_reason(
            submitted.replace("-", ""), record["corp_code"],
            {k: None for k in by_corp_date}))


def build(groups: tuple[str, ...] = ("major", "holding"),
          corpus_root: str | Path = "data/corpus",
          out_dir: str | Path = "out/forms") -> dict:
    index = CorpusIndex.build(corpus_root)
    known = list(iter_manifest(corpus_root))
    known_by_no = {r["rcept_no"]: r for r in known}
    records = [r for r in known if r["doc_group"] in groups]

    by_key: dict[tuple, list[str]] = {}
    for r in records:
        by_key.setdefault(
            (r["corp_code"], r["doc_group"], r["rcept_dt"], event_type_of(r["report_nm"])), []
        ).append(r["rcept_no"])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, int] = {}
    modes: dict[str, int] = {}
    failures: list[dict] = []
    n_events = n_edges = 0

    with (out_dir / "events.jsonl").open("w", encoding="utf-8") as ev, \
         (out_dir / "relations.jsonl").open("w", encoding="utf-8") as rel:
        for record in records:
            path = index.main_xml(record["file_path"], record["rcept_no"])
            if path is None:
                failures.append({"doc_id": record["doc_id"], "reason": "본문 XML 없음"})
                continue
            cleaned = sanitize(path.read_text(encoding="utf-8", errors="replace"))
            mode = "strict"
            try:
                root = ET.fromstring(cleaned)
            except ET.ParseError:
                from lxml import etree as lxml_etree
                try:
                    root = lxml_etree.fromstring(
                        cleaned.encode("utf-8"),
                        lxml_etree.XMLParser(recover=True, huge_tree=True, encoding="utf-8"),
                    )
                    mode = "recovered"
                except Exception as exc:  # noqa: BLE001
                    failures.append({"doc_id": record["doc_id"],
                                     "reason": f"{type(exc).__name__}: {exc}"[:160]})
                    continue
            modes[mode] = modes.get(mode, 0) + 1

            fields, _cells = extract_document(root)
            event = _to_event(record, fields, mode)
            ev.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
            n_events += 1

            if record["is_correction"]:
                target_nm, submitted = correction_header(root)
                edge = _correction_edge(record, target_nm, submitted, by_key, known_by_no)
                rel.write(json.dumps(asdict(edge), ensure_ascii=False) + "\n")
                n_edges += 1
                key = f"CORRECTS:{edge.resolution_status}"
                status[key] = status.get(key, 0) + 1

    report = {
        "groups": list(groups), "documents": len(records),
        "events": n_events, "relations": n_edges,
        "parse_modes": modes, "relation_status": dict(sorted(status.items())),
        "failures": failures,
    }
    (out_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 이 산출물이 **무엇으로 만들어졌는지** 남긴다. canonical 이 읽기 전에 검사한다 (B-03).
    write_stamp(out_dir, "forms",
                file_sha256(Path(corpus_root) / "manifest.jsonl"),
                resolver_version=RELATION_RESOLVER_VERSION)
    return report


def _main() -> int:
    r = build()
    print(f"문서 {r['documents']:,} → 이벤트 {r['events']:,} / 정정관계 {r['relations']:,}")
    print(f"  parse_modes {r['parse_modes']}")
    for k, v in r["relation_status"].items():
        print(f"  {k:<28}{v:>6,}")
    if r["failures"]:
        print(f"실패 {len(r['failures'])}건"); [print("   ", f) for f in r["failures"][:5]]
    return 1 if r["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(_main())

"""거래소공시 1,469건 → 정형 이벤트 + 관계 edge.

`exchange_html.parse_file()` 이 뽑은 라벨→값을 주최측 레퍼런스(`참고자료.pdf` p29)
2단계 「정규화」 형태로 옮긴다. **원문 값은 지우지 않고 `fields` 에 그대로 남긴다.**

## 관계 해석 — href 가 결정적 단서다

거래소공시의 `※ 관련공시` 는 KIND 링크로 들어 있고 href 에 원공시 접수번호가 있다.
KIND `acptno` 와 DART `rcept_no` 는 **9번째 자리 하나만 다르다**
(KIND `0` ↔ DART `8`=KOSPI / `9`=KOSDAQ). 나머지 13자리는 동일하다.

정정 631건 실측 — 날짜(법인+제출일) 매칭만 쓸 때와 href 를 결합할 때:

| | 날짜만 | href 결합 |
|---|---:|---:|
| resolved | 237 | **303** |
| ambiguous | 76 | **8** |
| root_missing | 318 | 318 |

정정 대상 자체는 텍스트 필드(`1. 정정관련 공시서류`, `2. 정정관련 공시서류제출일`)로
오므로, **href 와 제출일이 동시에 가리키는 문서만** `resolved` 로 확정한다.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import date as calendar_date
from pathlib import Path

from .corpus_paths import CorpusIndex, iter_manifest
from .exchange_html import ExchangeForm, normalize_amount, parse_file, split_related
from .relation_version import RELATION_RESOLVER_VERSION
from ..artifact import file_sha256, write_stamp

__all__ = ["ExchangeEvent", "RelationEdge", "build", "kind_to_dart"]

_RCPNO = re.compile(r"rcpno=(\d{14})")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
#: 정정 헤더에만 허용하는 날짜 표기. 사건일·계약기간까지 느슨하게 만들면
#: 본문 속 임의 숫자를 날짜로 오인할 수 있으므로 이 폴백은 최초제출일에만 쓴다.
_CORRECTION_DATE_FORMS = (
    re.compile(r"^\s*(\d{4})\.(\d{1,2})\.(\d{1,2})\.?\s*$"),
    re.compile(r"^\s*(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일\s*$"),
)
_NORMALIZED_DATE_FEATURE = "source_date:normalized"

#: doc_subtype → (금액 라벨 후보, 사건일 라벨)
#:
#: 시장에 따라 서식이 다르다. KOSPI 는 `계약금액(원)` 하나지만 KOSDAQ 는
#: `확정 계약금액` / `조건부 계약금액` / `계약금액 총액(원)` 으로 쪼개져 있다.
#: 총액을 먼저 찾고 없으면 KOSPI 라벨로 떨어진다.
_SUBTYPE_FIELDS: dict[str, tuple[tuple[str, ...], str]] = {
    "단일판매공급계약체결": (("계약금액 총액(원)", "계약금액(원)", "확정 계약금액"),
                             "계약(수주)일자"),
    "단일판매공급계약해지": (("해지금액 총액(원)", "해지금액(원)", "확정 해지금액"),
                             "해지일자"),
    "신규시설투자등": (("투자금액(원)", "투자금액 총액(원)"), "이사회결의일(결정일)"),
    "투자판단관련주요경영사항": ((), "이사회결의일(결정일) 또는 사실확인일"),
}


def kind_to_dart(acptno: str) -> list[str]:
    """KIND 접수번호 → DART `rcept_no` 후보. 9번째 자리만 다르다."""
    return [acptno[:8] + c + acptno[9:] for c in "890"] + [acptno]


@dataclass
class ExchangeEvent:
    event_id: str
    doc_id: str
    rcept_no: str
    corp_code: str
    corp_name: str
    sector: str
    doc_subtype: str
    is_correction: bool
    disclosed_at: str                   #: 공시일(rcept_dt)
    event_date: str | None              #: 사건 발생일 — 공시일과 다르다
    title: str | None
    amount: int | None
    amount_label: str | None            #: 계약금액 / 해지금액 / 투자금액
    recent_revenue: int | None
    ratio: float | None
    counterparty: str | None
    period_start: str | None
    period_end: str | None
    reserved_reason: str | None
    fields: dict[str, str] = field(default_factory=dict)   #: 원문 라벨 → 값
    notes: list[str] = field(default_factory=list)


@dataclass
class RelationEdge:
    """정정·관련 공시 간선 하나.

    **판정 결과만 남기면 재검토할 수 없다** (R-01·R-02). `ambiguous` 31건과
    `root_missing` 358건을 사람이 다시 볼 때, 무엇을 후보로 봤고 왜 못 골랐는지가
    없으면 처음부터 다시 재현해야 한다. 그래서 근거를 함께 적는다.
    """

    src_rcept_no: str
    dst_rcept_no: str | None
    relation_type: str                  #: CORRECTS | RELATED | SUPERSEDES
    resolution_status: str              #: resolved | candidate | ambiguous | root_missing | parse_failed
    target_hint: str | None
    confidence: float
    #: 판정 시점에 후보로 본 접수번호 전부. `ambiguous` 재검토의 출발점이다 (R-01)
    candidate_ids: tuple[str, ...] = ()
    #: 매칭에 쓴 키. `"corp|group|20240711|타법인주식및출자증권양수결정"` (R-01)
    match_features: str = ""
    #: `root_missing` 인 이유. 원본이 코퍼스에 없는 것과 키가 안 맞는 것은 다르다 (R-02)
    root_missing_reason: str | None = None


#: 코퍼스 수집 시작일. 이보다 앞선 원공시는 애초에 없다.
CORPUS_START = "20230101"


def precedes(candidate: str, src: str) -> bool:
    """원본은 정정본보다 **먼저 접수돼야 한다**.

    접수번호는 `YYYYMMDD` + 일련번호라 문자열 비교가 곧 접수 순서다.
    같은 날 여러 건을 낸 기업에서는 날짜만으로 원본을 못 고르는데,
    이 제약 하나로 `ambiguous` 31건 중 14건이 확정된다. **내용은 보지 않는다.**

    거르는 쪽으로도 값을 한다 — 현대로템 2024-05-14 는 두 정정공시가 모두
    「정정관련 제출일 = 정정일자」를 적어 날짜 포인터가 무의미해졌고,
    같은 날 후보가 하나뿐이라 **탄자니아 Lot2 정정이 Lot3 에 붙어 있었다.**
    """
    return candidate < src


#: 내용으로 원본을 고를 때 쓰는 계약 정체성 필드.
_IDENT_KEYS = ("counterparty", "period_start", "period_end")


def _usable_keys(touched: str) -> tuple[str, ...]:
    """정정이 **바꿨다고 스스로 밝힌 필드로는 맞추지 않는다**.

    현대건설 2025-02-19 은 「5.계약기간」을 정정했다. 정정본이 들고 있는 기간은
    이미 **새 값**이라, 그걸로 원본을 찾으면 우연히 같은 기간을 가진 형제에 붙는다
    (실제로 그렇게 붙었다 — 옳은 원본은 기간이 비어 있던 쪽이었다).
    """
    keys: list[str] = []
    if "계약상대" not in touched:
        keys.append("counterparty")
    if not any(w in touched for w in ("계약기간", "시작일", "종료일")):
        keys += ["period_start", "period_end"]
    return tuple(keys)


#: 날짜 폴백으로만 확정한 간선의 신뢰도. href 같은 직접 증거가 없다.
DATE_FALLBACK_CONFIDENCE = 0.8


def _squeeze(text: str | None) -> str:
    return "".join(str(text or "").split())


def _veto_by_title(edges: list[RelationEdge],
                   ident: dict[str, "ExchangeEvent"],
                   touched: dict[str, str]) -> int:
    """날짜 폴백으로 붙인 정정 중 **계약명이 어긋나는 것**을 끊는다.

    날짜 폴백의 근거는 「같은 회사·같은 날·같은 유형」뿐이라 직접 증거가 없다.
    같은 날 한 건만 남으면 내용이 달라도 붙어 버린다 — 현대로템 2024-05-14 는
    **탄자니아 Lot3(전동차) 정정이 Lot2(전기기관차)에 붙어 있었다.**

    href 로 확정한 간선은 건드리지 않는다. 원문이 접수번호를 직접 가리키므로
    계약명 표기가 흔들려도(`4차양산` vs `4차 양산사업`) 링크는 옳다.
    """
    cut = 0
    for i, e in enumerate(edges):
        # **날짜 폴백(0.8)과 내용 판별(0.6) 모두** 검사한다. 직접 증거(href, 1.0)만 면제한다 —
        # 원문이 접수번호를 직접 가리키므로 계약명 표기가 흔들려도 링크는 옳다.
        if (e.relation_type != "CORRECTS" or e.resolution_status != "resolved"
                or e.confidence >= 1.0 or not e.dst_rcept_no):
            continue
        if any(w in touched.get(e.src_rcept_no, "") for w in ("계약명", "제목")):
            continue                          # 정정이 계약명을 바꿨다면 비교할 수 없다
        a, b = ident.get(e.src_rcept_no), ident.get(e.dst_rcept_no)
        if not (a and b and a.title and b.title):
            continue
        if _squeeze(a.title) == _squeeze(b.title):
            continue
        edges[i] = replace(e, dst_rcept_no=None, resolution_status="root_missing",
                           confidence=0.0, root_missing_reason="title_mismatch")
        cut += 1
    return cut


def _disambiguate(edges: list[RelationEdge],
                  ident: dict[str, "ExchangeEvent"],
                  touched: dict[str, str]) -> int:
    """`ambiguous` 로 남은 정정을 계약 정체성으로 한 번 더 고른다.

    **다시 키로 묶는 것이 아니다.** 날짜가 이미 좁힌 후보 안에서만 고르고,
    정확 일치 + 유일성을 둘 다 요구한다. 못 고르면 `ambiguous` 그대로다.
    계약 정체성을 **주** 키로 쓰면 정정이 그 필드를 바꾸는 탓에
    기존 사건 112개가 쪼개진다 — 그래서 보조 판별자로만 쓴다.
    """
    fixed = 0
    for i, e in enumerate(edges):
        if e.relation_type != "CORRECTS" or e.resolution_status != "ambiguous":
            continue
        me = ident.get(e.src_rcept_no)
        keys = _usable_keys(touched.get(e.src_rcept_no, ""))
        if me is None:
            continue
        # **값이 없는 것끼리는 맞았다고 하지 않는다.** 예전에는 `None == None` 도
        # 일치로 세어, 양쪽 다 비어 있는 필드가 판별 근거가 됐다 (5차 검수 지적).
        # 값이 있는 축으로만 비교하고, 후보 쪽도 그 축이 채워져 있어야 한다.
        usable = [k for k in keys if getattr(me, k) is not None]
        if usable:
            want = tuple(getattr(me, k) for k in usable)
            hit = [c for c in e.candidate_ids
                   if c in ident
                   and all(getattr(ident[c], k) is not None for k in usable)
                   and tuple(getattr(ident[c], k) for k in usable) == want]
            if len(hit) == 1:
                edges[i] = replace(
                    e, dst_rcept_no=hit[0], resolution_status="resolved", confidence=0.6,
                    match_features=f"{e.match_features}|content:{','.join(usable)}")
                fixed += 1
                continue

        # 이번 보조 판별은 **비정규 최초제출일을 정상화해 새로 생긴 후보에만** 쓴다.
        # 정상 날짜의 기존 ambiguous 13건에 일반 title heuristic을 적용하면 사람이
        # 승인해야 할 후보까지 조용히 확정된다. 제목을 정정한 문서도 비교 대상이 아니다.
        if (_NORMALIZED_DATE_FEATURE not in e.match_features
                or any(w in touched.get(e.src_rcept_no, "") for w in ("계약명", "제목"))
                or not me.title):
            continue
        wanted_title = _squeeze(me.title)
        title_hit = [c for c in e.candidate_ids
                     if c in ident and ident[c].title
                     and _squeeze(ident[c].title) == wanted_title]
        if len(title_hit) == 1:
            edges[i] = replace(
                e, dst_rcept_no=title_hit[0], resolution_status="resolved", confidence=0.7,
                match_features=f"{e.match_features}|content:title")
            fixed += 1
    return fixed


def _missing_reason(target_dt: str | None, corp_code: str, by_corp_date: dict,
                    corpus_start: str = CORPUS_START) -> str:
    """`root_missing` 의 이유를 가른다 (R-02).

    「원본이 수집 범위 밖이라 없다」와 「같은 날 문서는 있는데 유형이 안 맞는다」는
    전혀 다른 문제다. 전자는 데이터 한계이고 후자는 매칭 결함이다.
    """
    if not target_dt:
        return "no_submitted_date"
    if target_dt < corpus_start:
        return "submitted_before_corpus"
    if any(k[0] == corp_code for k in by_corp_date):
        if any(k[0] == corp_code and k[1] == target_dt for k in by_corp_date):
            return "type_mismatch"          # 같은 날 문서는 있으나 유형이 다름
        return "date_absent"                # 그 회사 문서는 있으나 그 날짜가 없음
    return "corp_absent"


def _ratio(raw: str | None) -> float | None:
    if not raw:
        return None
    try:
        return float(raw.replace(",", "").replace("%", "").strip())
    except ValueError:
        return None


def _date(raw: str | None) -> str | None:
    return raw if raw and _DATE.match(raw.strip()) else None


def _correction_date(raw: str | None) -> tuple[str | None, bool]:
    """정정 헤더 최초제출일을 ``(YYYY-MM-DD, 표기정규화여부)`` 로 돌린다.

    기존 ISO 표기 또는 전량 원문에서 확인된 점/한글 표기만 허용한다. 정규식이
    맞아도 실제 달력에 없는 날짜면 실패시켜 endpoint를 만들지 않는다.
    """
    if not raw:
        return None, False
    stripped = raw.strip()
    if _DATE.fullmatch(stripped):
        groups = stripped.split("-")
        normalized = False
    else:
        match = next((m for form in _CORRECTION_DATE_FORMS
                      if (m := form.fullmatch(raw))), None)
        if match is None:
            return None, False
        groups = list(match.groups())
        normalized = True
    year, month, day = (int(value) for value in groups)
    try:
        calendar_date(year, month, day)
    except ValueError:
        return None, False
    return f"{year:04d}-{month:02d}-{day:02d}", normalized


def _to_event(record: dict, form: ExchangeForm) -> ExchangeEvent:
    subtype = record["doc_subtype"] or ""
    amount_labels, date_label = _SUBTYPE_FIELDS.get(subtype, ((), ""))
    return ExchangeEvent(
        event_id=f"exchange_{record['rcept_no']}",
        doc_id=record["doc_id"],
        rcept_no=record["rcept_no"],
        corp_code=record["corp_code"],
        corp_name=record["corp_name"],
        sector=record["sector"],
        doc_subtype=subtype,
        is_correction=record["is_correction"],
        disclosed_at=record["rcept_dt"],
        event_date=_date(form.get(date_label)) if date_label else None,
        title=form.get("체결계약명", "해지계약명", "투자대상", "1. 제목"),
        amount=normalize_amount(form.get(*amount_labels)) if amount_labels else None,
        amount_label=amount_labels[0] if amount_labels else None,
        recent_revenue=normalize_amount(form.get("최근매출액(원)", "자기자본(원)")),
        ratio=_ratio(form.get("매출액대비(%)", "자기자본대비(%)")),
        counterparty=form.get("3. 계약상대"),
        period_start=_date(form.get("시작일")),
        period_end=_date(form.get("종료일")),
        reserved_reason=form.get("유보사유"),
        fields=dict(form.fields),
        notes=list(form.notes),
    )


def _href_targets(raw_html: str, src: str, known: dict[str, dict]) -> list[str]:
    """본문에 박힌 접수번호 → 코퍼스에 있는 대상.

    **같은 법인만 남긴다.** 본문 전체에서 접수번호를 긁으므로 관련공시 영역 밖의
    링크(타사 공시 인용 등)도 후보가 된다. 지금 코퍼스에는 법인이 다른 간선이 0건이지만,
    막지 않으면 서식이 바뀌는 순간 confidence 1.0 으로 확정된다.
    """
    # **법인은 `corp_code` 로 비교한다.** 회사명은 바뀐다 — 상호변경·표기 흔들림이
    # 있으면 같은 법인인데 다른 회사로 보여 정상 링크가 끊긴다 (5차 검수 지적).
    src_corp = (known.get(src) or {}).get("corp_code")
    out: list[str] = []
    for acptno in dict.fromkeys(_RCPNO.findall(raw_html)):
        for candidate in kind_to_dart(acptno):
            if candidate not in known or candidate == src:
                continue
            if src_corp and known[candidate].get("corp_code") != src_corp:
                break                       # 다른 법인 — 링크로 인정하지 않는다
            out.append(candidate)
            break
    return list(dict.fromkeys(out))


def _relations(record: dict, form: ExchangeForm, raw_html: str,
               known: dict[str, dict], by_corp_date: dict) -> list[RelationEdge]:
    src = record["rcept_no"]
    hrefs = _href_targets(raw_html, src, known)
    edges: list[RelationEdge] = []

    if record["is_correction"]:
        submitted = form.get("정정관련 공시서류제출일", "최초제출일")
        submitted_date, was_normalized = _correction_date(submitted)
        target_dt = submitted_date.replace("-", "") if submitted_date else None
        hint = " ".join(
            x for x in (submitted, form.get("정정관련 공시서류")) if x
        ).strip() or None

        feat = f"href|{record['corp_name']}|{target_dt}|{record['doc_subtype']}"
        if was_normalized:
            feat += f"|{_NORMALIZED_DATE_FEATURE}"
        exact = [h for h in hrefs
                 if known[h]["rcept_dt"] == target_dt and precedes(h, src)] if target_dt else []
        if len(exact) == 1:
            edges.append(RelationEdge(src, exact[0], "CORRECTS", "resolved", hint, 1.0,
                                      tuple(exact), feat))
        else:
            # href 가 없을 때의 날짜 폴백. 유형까지 맞춰야 같은 날 복수 제출에서
            # 엉뚱한 원본에 붙지 않는다.
            same_day = [
                r for r in by_corp_date.get(
                    (record["corp_code"], target_dt, record["doc_subtype"]), []
                ) if r != src and precedes(r, src)
            ] if target_dt else []
            cands = tuple(dict.fromkeys(list(exact) + same_day))
            if len(same_day) == 1:
                edges.append(RelationEdge(src, same_day[0], "CORRECTS", "resolved", hint, 0.8,
                                          cands, feat))
            elif len(same_day) > 1:
                edges.append(RelationEdge(src, None, "CORRECTS", "ambiguous", hint, 0.3,
                                          cands, feat))
            else:
                edges.append(RelationEdge(
                    src, None, "CORRECTS", "root_missing", hint, 0.0, cands, feat,
                    root_missing_reason=_missing_reason(
                        target_dt, record["corp_code"], by_corp_date)))

    # 관련공시 — 해지 20건이 원계약을 여기로 역참조한다
    linked = {e.dst_rcept_no for e in edges if e.dst_rcept_no}
    for target in hrefs:
        if target not in linked:
            edges.append(RelationEdge(src, target, "RELATED", "resolved", None, 1.0))
            linked.add(target)

    resolved_dates = {known[t]["rcept_dt"] for t in linked}
    for date_str, kind in split_related(form.get("관련공시")):
        if date_str.replace("-", "") not in resolved_dates:
            edges.append(RelationEdge(
                src, None, "RELATED", "root_missing", f"{date_str} {kind}", 0.0,
                match_features=f"related|{record['corp_name']}|{date_str}|{kind}",
                root_missing_reason=_missing_reason(
                    date_str.replace("-", ""), record["corp_code"], by_corp_date)))
    return edges


def build(corpus_root: str | Path = "data/corpus",
          out_dir: str | Path = "out/exchange") -> dict:
    """거래소공시 전량 → `events.jsonl` + `relations.jsonl` + `build_report.json`."""
    index = CorpusIndex.build(corpus_root)
    known = {r["rcept_no"]: r for r in iter_manifest(corpus_root)}
    records = [r for r in known.values() if r["doc_group"] == "exchange"]

    # **법인 키는 `corp_code`.** 회사명은 바뀐다 — 상호변경·표기 흔들림이 있으면
    # 날짜 폴백이 같은 법인을 못 찾아 `root_missing` 이 된다 (6차 검수).
    # direct href 만 고치고 이 색인을 두면 법인 키 수정이 절반만 끝난 것이다.
    by_corp_date: dict[tuple[str, str, str], list[str]] = {}
    for r in records:
        by_corp_date.setdefault(
            (r["corp_code"], r["rcept_dt"], r["doc_subtype"]), []
        ).append(r["rcept_no"])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, int] = {}
    failures: list[dict] = []
    n_events = n_edges = 0

    edges_all: list[RelationEdge] = []
    ident: dict[str, ExchangeEvent] = {}
    touched: dict[str, str] = {}

    with (out_dir / "events.jsonl").open("w", encoding="utf-8") as ev:
        for record in records:
            path = index.main_xml(record["file_path"], record["rcept_no"])
            if path is None:
                failures.append({"doc_id": record["doc_id"], "reason": "main xml 없음"})
                continue
            raw = path.read_text(encoding="utf-8", errors="replace")  # UTF-8 강제
            try:
                form = parse_file(path)
            except Exception as exc:  # noqa: BLE001 — 조용히 넘기지 않는다
                failures.append(
                    {"doc_id": record["doc_id"],
                     "reason": f"{type(exc).__name__}: {exc}"[:200]}
                )
                continue

            event = _to_event(record, form)
            ev.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
            n_events += 1
            ident[record["rcept_no"]] = event
            touched[record["rcept_no"]] = " ".join(
                str(v) for k, v in form.fields.items() if "정정항목" in k)
            edges_all.extend(_relations(record, form, raw, known, by_corp_date))

    # **후처리** — 후보 전원의 계약 정체성이 모인 뒤라야 모호한 정정을 고를 수 있다
    n_content = _disambiguate(edges_all, ident, touched)
    n_veto = _veto_by_title(edges_all, ident, touched)
    with (out_dir / "relations.jsonl").open("w", encoding="utf-8") as rel:
        for edge in edges_all:
            rel.write(json.dumps(asdict(edge), ensure_ascii=False) + "\n")
            n_edges += 1
            key = f"{edge.relation_type}:{edge.resolution_status}"
            status[key] = status.get(key, 0) + 1

    report = {
        "documents": len(records),
        "events": n_events,
        "relations": n_edges,
        "relation_status": dict(sorted(status.items())),
        "corrects_by_content": n_content,
        "corrects_vetoed_by_title": n_veto,
        "failures": failures,
    }
    (out_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 이 산출물이 **무엇으로 만들어졌는지** 남긴다. canonical 이 읽기 전에 검사한다 (B-03).
    write_stamp(out_dir, "exchange_events",
                file_sha256(Path(corpus_root) / "manifest.jsonl"),
                resolver_version=RELATION_RESOLVER_VERSION)
    return report


def _main() -> int:
    result = build()
    print(f"문서 {result['documents']:,} → 이벤트 {result['events']:,}"
          f" / 관계 {result['relations']:,}")
    for key, count in result["relation_status"].items():
        print(f"  {key:<28}{count:>6,}")
    if result["failures"]:
        print(f"실패 {len(result['failures'])}건")
        for f in result["failures"][:5]:
            print("   ", f)
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(_main())

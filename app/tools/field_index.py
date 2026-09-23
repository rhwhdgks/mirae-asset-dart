"""FieldIndex — 접수번호 단위 bounded Field 메모리 캐시 (파생 인덱스, 설계서 v3 §4).

특정 문서의 slot 조회에 회사 전체 Field는 필요하지 않다. 접수번호별 Field만 최근
사용 순서와 총 행 수 예산 안에서 보관한다. ``fields.parquet`` 전체를 서버 기동 때
Python 객체로 올리면 파일 크기보다 훨씬 큰 RSS를 만들기 때문에, ``load_all``은 기존
워밍업 API와 행 수 반환 계약만 유지하고 캐시를 채우지 않는다.
판정 로직을 복제하지 않는다:
- 후보 탐색·slot 값 미리보기는 캐시에서
- **확정 인용 직전에는 rm._verified_field(row)로 Evidence 재검증** (lookup_field와 동일 경로)
as_of 필터는 rcept_dt <= as_of로 캐시에서 적용한다 (read.py의 fields(as_of=)와 같은 의미).
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from pathlib import Path

import pyarrow.parquet as pq


class FieldIndex:
    #: 한 회사의 문서 수보다 작으면 한 바퀴 돌 때마다 LRU 가 앞머리를 밀어내
    #: 같은 문서를 다시 읽는다. 코퍼스에서 가장 문서가 많은 회사가 200건이고
    #: 128 이던 상한 탓에 5개 회사(셀트리온 200·한화오션 150·대우건설 150·
    #: 삼성E&A 141·현대건설 135)가 그 스래싱에 걸렸다 — K-008 은 문서 150건을
    #: 읽는 데 `_load_receipt` 746 회, 26.3초를 썼다.
    #:
    #: 메모리는 행 예산(`max_cached_rows`)이 잡는다. 문서 수 상한은 그보다
    #: 먼저 걸리면 안 되므로 가장 큰 회사를 담을 만큼 둔다.
    _MAX_DOCUMENTS_PER_ISSUER = 256

    def __init__(self, rm, *, max_cached_receipts: int = _MAX_DOCUMENTS_PER_ISSUER,
                 max_cached_rows: int = 25_000,
                 max_verified_cache: int = 4096,
                 max_cached_corps: int | None = None):
        # 구형 호출부의 keyword를 한 릴리스 동안 받되 의미는 receipt count로 좁힌다.
        if max_cached_corps is not None:
            if type(max_cached_corps) is not int or max_cached_corps < 1:
                raise ValueError("max_cached_corps는 양의 정수여야 합니다")
            max_cached_receipts = max_cached_corps
        if type(max_cached_receipts) is not int or max_cached_receipts < 1:
            raise ValueError("max_cached_receipts는 양의 정수여야 합니다")
        if type(max_cached_rows) is not int or max_cached_rows < 1:
            raise ValueError("max_cached_rows는 양의 정수여야 합니다")
        if type(max_verified_cache) is not int or max_verified_cache < 1:
            raise ValueError("max_verified_cache는 양의 정수여야 합니다")
        self.rm = rm
        # corp_code도 key에 넣어 잘못된 회사+접수번호 음수 조회가 올바른 조회를
        # 오염시키지 않게 한다. 값은 한 문서의 FieldRow 목록뿐이다.
        self._by_rcept: OrderedDict[tuple[str, str], list] = OrderedDict()
        self._cached_rows = 0
        self._verified_cache: OrderedDict[tuple, object] = OrderedDict()
        self._max_cached_receipts = max_cached_receipts
        self._max_cached_rows = max_cached_rows
        self._max_verified_cache = max_verified_cache

    def load_all(self) -> int:
        """호환용 전체 행 수를 반환하되, 전량을 메모리에 적재하지 않는다.

        기존 서버는 이 메서드를 워밍업으로 호출했다. 이제는 Parquet metadata의
        행 수만 읽어 Python ``FieldRow`` 객체 생성을 피한다. 테스트용/구형 reader
        처럼 canonical root를 노출하지 않는 객체는 기존 스트리밍 fallback을 쓴다.
        """
        root = getattr(self.rm, "root", None)
        if root is not None:
            artifact = Path(root) / "fields.parquet"
            if artifact.exists():
                return int(pq.ParquetFile(artifact).metadata.num_rows)
        # Compatibility fallback for small fake/readers without a parquet root.
        return sum(1 for _ in self.rm.fields(as_of="20991231"))

    def _evict_receipts(self) -> None:
        """receipt/verified 참조를 행 수와 문서 수 예산 안에서 함께 제거한다."""

        while (self._cached_rows > self._max_cached_rows
               or len(self._by_rcept) > self._max_cached_receipts):
            _, rows = self._by_rcept.popitem(last=False)
            self._cached_rows -= len(rows)
            for row in rows:
                self._verified_cache.pop(
                    (row.rcept_no, row.locator, row.occurrence), None)

    def _cache_receipt(self, key: tuple[str, str], rows: list) -> None:
        """한 문서가 전체 예산보다 크면 이번 호출에만 쓰고 상주시켜 두지 않는다."""

        if not rows or len(rows) > self._max_cached_rows:
            return
        previous = self._by_rcept.pop(key, None)
        if previous is not None:
            self._cached_rows -= len(previous)
        self._by_rcept[key] = rows
        self._by_rcept.move_to_end(key)
        self._cached_rows += len(rows)
        self._evict_receipts()

    def _cache_verified(self, key: tuple, value: object) -> object:
        self._verified_cache[key] = value
        self._verified_cache.move_to_end(key)
        while len(self._verified_cache) > self._max_verified_cache:
            self._verified_cache.popitem(last=False)
        return value

    def _load_receipt(self, corp_code: str, rcept_no: str) -> list:
        key = (corp_code, rcept_no)
        rows = self._by_rcept.get(key)
        if rows is not None:
            self._by_rcept.move_to_end(key)
            return rows

        # ``include_restricted_raw`` only widens ``FieldRow.value_raw``; the
        # ``.value`` property callers use everywhere else stays the masked
        # scalar regardless. This lets a narrow, name-scoped caller (holding
        # party matching — RPC-017) recover a legacy-masked institution name
        # from ``value_raw`` without changing what any other caller sees.
        rows = list(self.rm.fields(
            as_of="20991231", corp_code=corp_code, rcept_no=rcept_no,
            include_restricted_raw=True))
        self._cache_receipt(key, rows)
        return rows

    def load(self, corp_code: str) -> list:
        """구형 명시 호출용 회사 snapshot. 서버 경로는 이 결과를 cache하지 않는다."""

        return list(self.rows(corp_code, as_of="20991231"))

    def rows(self, corp_code: str, *, as_of: str, rcept_no: str | None = None,
             rcept_nos=None,
             label: str | None = None, doc_group: str | None = None, form: str | None = None,
             is_correction: bool | None = None):
        if rcept_no is None:
            # 넓은 진단 API는 streaming만 하고 상주시켜 두지 않는다. 운영 후보 탐색은
            # ``iter_docs``로 문서 하나씩 처리해 호출자 쪽 전량 materialization도 피한다.
            yield from self.rm.fields(
                as_of=as_of, corp_code=corp_code, label=label,
                rcept_nos=rcept_nos,
                doc_group=doc_group, form=form, is_correction=is_correction)
            return

        src = self._load_receipt(corp_code, rcept_no)
        for r in src:
            if r.rcept_dt > as_of: continue
            if rcept_no and r.rcept_no != rcept_no: continue
            if label and label not in r.label: continue
            if doc_group and r.doc_group != doc_group: continue
            if form and r.form != form: continue
            if is_correction is not None and r.is_correction != is_correction: continue
            yield r

    def docs(self, corp_code: str, *, as_of: str, **kw) -> dict[str, list]:
        """호환용 broad snapshot. 운영 경로는 :meth:`iter_docs`를 사용한다."""

        out: dict[str, list] = defaultdict(list)
        for r in self.rows(corp_code, as_of=as_of, **kw):
            out[r.rcept_no].append(r)
        return out

    def iter_docs(self, corp_code: str, *, as_of: str,
                  doc_group: str | None = None, form: str | None = None,
                  is_correction: bool | None = None,
                  keep=None):
        """Document metadata로 receipt를 먼저 좁히고 Field는 문서 하나씩 연다."""

        documents = getattr(self.rm, "documents", None)
        if callable(documents):
            # Document metadata already names the candidate receipts, so their
            # Field rows can be read in one scan instead of reopening
            # fields.parquet per document.  Measured on a 100-document issuer
            # the per-receipt path costs ~0.24s each (~24s) while the single
            # scan costs ~0.6s; a mixed-form event search hit that repeatedly
            # and pushed the request past its execution budget.
            metadata = tuple(documents(
                as_of=as_of, corp_code=corp_code, doc_group=doc_group,
                form=form, is_correction=is_correction))
            # 조건에 맞는 문서가 아예 없으면 여기서 끝낸다. 아래로 내려가면
            # `streams_once`가 `form is not None`만으로도 참이 될 수 있어
            # (R-F-006, "KB금융 2025년 사업보고서는 정정된 적이 없지?" —
            # is_correction=True로 걸러 정말로 0건), 빈 receipt 집합을 그대로
            # `self.rows(rcept_nos=[])`로 넘겨 `_checked_rcept_nos`가 "빈
            # 모음은 유효하지 않다"고 거절하며 죽었다. 정답이 "0건"인 질문이
            # 예외로 끝나면 안 된다.
            if not metadata:
                return
            # 문서 종류·날짜로 거를 수 있으면 Field 를 읽기 전에 거른다.
            # 셀트리온은 문서 200건 중 계약 공시가 몇 건인데, 나머지 대량보유
            # 상황보고서의 Field 290,332행까지 읽고 나서 버리고 있었다.
            if keep is not None:
                metadata = tuple(meta for meta in metadata if keep(meta))
                if not metadata:
                    return
            # This is deliberately a bounded transient grouping, not a new
            # issuer-wide Field cache.  Large populations retain the existing
            # receipt-at-a-time memory behaviour, and only rows belonging to
            # the selected receipts are held.
            #
            # A mixed-form search has no typed form predicate, so its single
            # scan reads the issuer's whole Field set.  Measured, that costs
            # about the same as three per-receipt reads, and a one- or
            # two-document selector also wants its rows left in the receipt
            # cache for the immediately following lookups.  So the scan is
            # only taken once it actually pays for itself.
            # 상한이 문서 수보다 작으면 문서가 많은 회사에서만 배치가 꺼진다 —
            # 정작 배치가 가장 필요한 쪽이다. 한화오션 150건은 이 128 때문에
            # 문서 하나씩 여는 경로로 떨어져 `_load_receipt` 를 559회 돌았다.
            # receipt 캐시 상한과 같은 값을 써야 둘이 어긋나지 않는다.
            streams_once = (
                len(metadata) <= self._max_cached_receipts
                and (form is not None or len(metadata) >= 3))
            if streams_once:
                wanted = {meta.rcept_no for meta in metadata}
                rows_by_receipt: dict[str, list] = defaultdict(list)
                # 원하는 문서를 술어로 밀어 넣지 않으면 회사 전체 Field 를
                # 읽고 대부분을 버린다. 셀트리온은 295,107행 중 290,332행이
                # 대량보유상황보고서라, 계약 공시 질문에 그 전부를 읽고 있었다.
                for row in self.rows(
                        corp_code, as_of=as_of, rcept_nos=sorted(wanted),
                        doc_group=doc_group,
                        form=form, is_correction=is_correction):
                    if row.rcept_no in wanted:
                        rows_by_receipt[row.rcept_no].append(row)
                for meta in metadata:
                    rows = rows_by_receipt.get(meta.rcept_no, [])
                    if rows:
                        # 한 번에 읽어 놓고 캐시에 넣지 않으면, 바로 뒤따르는
                        # slot 조회가 같은 문서를 다시 문서별로 연다. K-008 은
                        # 그래서 배치를 켜고도 45초를 넘겼다. 캐시에 넣는 것은
                        # 행 예산이 잡아 주므로 큰 회사에서도 상주량은 그대로다.
                        self._cache_receipt((corp_code, meta.rcept_no), rows)
                        yield meta.rcept_no, rows
                return
            for meta in documents(
                    as_of=as_of, corp_code=corp_code, doc_group=doc_group,
                    form=form, is_correction=is_correction):
                rows = list(self.rows(
                    corp_code, as_of=as_of, rcept_no=meta.rcept_no,
                    doc_group=doc_group, form=form,
                    is_correction=is_correction))
                if rows:
                    yield meta.rcept_no, rows
            return

        # 작은 fake/legacy reader 호환. production CanonicalReadModel은 위 경로다.
        for rcept_no, rows in self.docs(
                corp_code, as_of=as_of, doc_group=doc_group, form=form,
                is_correction=is_correction).items():
            yield rcept_no, rows

    def verified(self, row):
        """확정 인용 직전 Evidence 재검증 — read.py의 lookup_field와 동일 경로."""
        key = (row.rcept_no, row.locator, row.occurrence)
        cached = self._verified_cache.get(key)
        if key in self._verified_cache:
            self._verified_cache.move_to_end(key)
            return cached
        return self._cache_verified(key, self.rm._verified_field(row))

    def lookup(self, corp_code: str, rcept_no: str, label: str, *, as_of: str):
        """lookup_field 의미론 재현: 0건 not_found / 1값 ok(verified) / 값 상이 ambiguous."""
        cands = list(self.rows(corp_code, as_of=as_of, rcept_no=rcept_no, label=label))
        if not cands:
            return "not_found", None, ()
        vals = {c.value for c in cands}
        if len(vals) == 1:
            row = sorted(cands, key=lambda c: (c.occurrence, c.order))[0]
            return "ok", self.verified(row), tuple(cands)
        return "ambiguous", None, tuple(cands)

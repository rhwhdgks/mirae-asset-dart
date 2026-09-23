"""SQLite FTS5 검색 인덱스 — 합의안 §5.

빌드: CanonicalReadModel.chunks(projection="search") → chunk_meta + chunk_fts(contentless)
질의: rcept_dt <= as_of / corp_code / doc_group을 WHERE로 **랭킹 전에** 적용, bm25 ASC.
결과 본문은 text_prompt_safe만. token 문자열은 반환하지 않는다.
index_build_id = sha256(canonical_build_id | chunks_hash | tokenizer_config_hash | 버전들)[:32]
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .tokenizer import KiwiTokenizer, TOKENIZER_CONFIG_HASH, TOKENIZER_IMPL_VERSION

INDEX_BUILDER_VERSION = "fts-index/1.0"
FTS_SCHEMA_VERSION = "fts5-schema/1.0"
ROOT = Path(__file__).resolve().parents[2]
SERVING_DIR = ROOT / "out" / "serving"


def compute_index_build_id(canonical_build_id: str, chunks_hash: str, security_policy_version: str) -> str:
    import kiwipiepy
    parts = [canonical_build_id, chunks_hash, TOKENIZER_CONFIG_HASH, kiwipiepy.__version__,
             TOKENIZER_IMPL_VERSION, INDEX_BUILDER_VERSION, "", FTS_SCHEMA_VERSION,
             sqlite3.sqlite_version, security_policy_version]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


def build_index(rm, run: dict, *, out_dir: Path = SERVING_DIR, log=print) -> Path:
    canonical_build_id = run["build_id"]
    chunks_hash = run["artifact_hashes"]["chunks"]
    sec_ver = run["security_policy_version"]
    build_id = compute_index_build_id(canonical_build_id, chunks_hash, sec_ver)
    final = out_dir / build_id
    if (final / "search.sqlite3").exists() and (final / "manifest.json").exists():
        log(f"index {build_id} 이미 존재 — 재사용"); return final
    staging = out_dir / f"{build_id}.building"
    if staging.exists():
        import shutil; shutil.rmtree(staging)
    staging.mkdir(parents=True)
    db = staging / "search.sqlite3"
    con = sqlite3.connect(db)
    con.executescript("""
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        CREATE TABLE chunk_meta (
            rowid INTEGER PRIMARY KEY, chunk_id TEXT NOT NULL, section_id TEXT NOT NULL,
            doc_id TEXT NOT NULL, source_file_id TEXT NOT NULL, corp_code TEXT NOT NULL,
            corp_name TEXT NOT NULL, doc_group TEXT NOT NULL, rcept_dt TEXT NOT NULL,
            path TEXT NOT NULL, locator TEXT NOT NULL, text_prompt_safe TEXT NOT NULL,
            index_eligible INTEGER NOT NULL, over_budget INTEGER NOT NULL,
            evidence_id TEXT, security_policy_version TEXT NOT NULL);
        CREATE VIRTUAL TABLE chunk_fts USING fts5(body_tokens, path_tokens, content='',
            tokenize="unicode61 tokenchars '-_'");
    """)
    tok = KiwiTokenizer.get()
    t0 = time.time(); n_meta = n_fts = 0
    batch_meta, batch_fts = [], []
    for row in rm.chunks(index_eligible_only=False, projection="search"):
        n_meta += 1
        rid = n_meta
        batch_meta.append((rid, row.chunk_id, row.section_id, row.doc_id, row.source_file_id,
                           row.corp_code, row.corp_name, row.doc_group, row.rcept_dt, row.path,
                           row.locator, row.text_prompt_safe, int(row.index_eligible),
                           int(row.over_budget), row.evidence_id, row.security_policy_version))
        if row.index_eligible:
            n_fts += 1
            batch_fts.append((rid, tok.fts_document(row.text), tok.fts_document(row.path)))
        if len(batch_meta) >= 2000:
            con.executemany("INSERT INTO chunk_meta VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch_meta)
            con.executemany("INSERT INTO chunk_fts(rowid, body_tokens, path_tokens) VALUES (?,?,?)", batch_fts)
            batch_meta, batch_fts = [], []
            if n_meta % 20000 == 0:
                log(f"  {n_meta:,} chunks ({time.time()-t0:.0f}s)")
    if batch_meta:
        con.executemany("INSERT INTO chunk_meta VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch_meta)
        con.executemany("INSERT INTO chunk_fts(rowid, body_tokens, path_tokens) VALUES (?,?,?)", batch_fts)
    con.executescript("""
        CREATE INDEX ix_meta_corp ON chunk_meta(corp_code, doc_group, rcept_dt);
        CREATE INDEX ix_meta_dt ON chunk_meta(rcept_dt);
        CREATE INDEX ix_meta_sec ON chunk_meta(section_id);
    """)
    con.commit()
    (mc,) = con.execute("SELECT COUNT(*) FROM chunk_meta").fetchone()
    (fc,) = con.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()
    (qc,) = con.execute("PRAGMA quick_check").fetchone()
    con.close()
    if mc != n_meta or fc != n_fts or qc != "ok":
        raise RuntimeError(f"index gate 실패: meta {mc}/{n_meta} fts {fc}/{n_fts} quick_check={qc}")
    db_hash = hashlib.sha256(db.read_bytes()).hexdigest()
    import kiwipiepy
    manifest = {
        "index_build_id": build_id, "canonical_build_id": canonical_build_id,
        "canonical_schema_version": run["schema_version"], "chunks_hash": chunks_hash,
        "meta_rows": mc, "fts_rows": fc, "sqlite_version": sqlite3.sqlite_version,
        "kiwipiepy_version": kiwipiepy.__version__, "tokenizer_impl_version": TOKENIZER_IMPL_VERSION,
        "tokenizer_config_hash": TOKENIZER_CONFIG_HASH, "index_builder_version": INDEX_BUILDER_VERSION,
        "user_dictionary_sha256": hashlib.sha256(b"").hexdigest(), "fts_schema_version": FTS_SCHEMA_VERSION,
        "security_policy_version": sec_ver, "db_sha256": db_hash,
        "built_seconds": round(time.time() - t0, 1),
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    os.rename(staging, final)   # 원자적 발행
    log(f"index {build_id} 발행: meta {mc:,} fts {fc:,} ({manifest['built_seconds']}s)")
    return final


@dataclass(frozen=True)
class SearchHit:
    rank: int
    score: float
    chunk_id: str
    section_id: str
    doc_id: str
    source_file_id: str
    corp_code: str
    corp_name: str
    doc_group: str
    rcept_dt: str
    path: str
    locator: str
    text_prompt_safe: str
    evidence_id: str | None


class SearchIndex:
    """읽기 전용 검색기. canonical build_id 불일치 시 fail-closed."""

    def __init__(self, index_dir: Path, *, expected_canonical_build_id: str):
        self.dir = Path(index_dir)
        self.manifest = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest["canonical_build_id"] != expected_canonical_build_id:
            raise RuntimeError(f"index canonical_build_id {self.manifest['canonical_build_id']} != read model {expected_canonical_build_id}")
        if self.manifest["tokenizer_config_hash"] != TOKENIZER_CONFIG_HASH:
            raise RuntimeError("tokenizer config hash 불일치 — 문서/질의 tokenizer가 다름")
        import threading
        self._local = threading.local()
        self.tok = KiwiTokenizer.get()

    @property
    def con(self):
        """읽기 전용 DB — 스레드별 연결 (서버 워커 스레드에서 안전)."""
        c = getattr(self._local, "con", None)
        if c is None:
            c = sqlite3.connect(f"file:{self.dir/'search.sqlite3'}?mode=ro", uri=True, check_same_thread=False)
            self._local.con = c
        return c

    @property
    def index_build_id(self) -> str:
        return self.manifest["index_build_id"]

    @property
    def canonical_build_id(self) -> str:
        return self.manifest["canonical_build_id"]

    def search(self, retrieval_query: str, *, as_of: str, corp_codes: tuple[str, ...] = (),
               doc_groups: tuple[str, ...] = (), date_from: str | None = None,
               path_prefix: str | None = None, top_k: int = 10) -> tuple[SearchHit, ...]:
        match = self.tok.fts_match_query(retrieval_query)
        if not match:
            return ()
        sql = ["SELECT m.chunk_id, m.section_id, m.doc_id, m.source_file_id, m.corp_code, m.corp_name,",
               "       m.doc_group, m.rcept_dt, m.path, m.locator, m.text_prompt_safe, m.evidence_id,",
               "       bm25(chunk_fts) AS score",
               "FROM chunk_fts JOIN chunk_meta AS m ON m.rowid = chunk_fts.rowid",
               "WHERE chunk_fts MATCH ? AND m.index_eligible = 1 AND m.rcept_dt <= ?"]
        params: list = [match, as_of]
        if date_from:
            sql.append("AND m.rcept_dt >= ?"); params.append(date_from)
        if corp_codes:
            sql.append(f"AND m.corp_code IN ({','.join('?'*len(corp_codes))})"); params.extend(corp_codes)
        if doc_groups:
            sql.append(f"AND m.doc_group IN ({','.join('?'*len(doc_groups))})"); params.extend(doc_groups)
        if path_prefix:
            esc = path_prefix.replace("!", "!!").replace("%", "!%").replace("_", "!_") + "%"
            sql.append("AND m.path LIKE ? ESCAPE '!'"); params.append(esc)
        sql.append("ORDER BY score ASC, m.rcept_dt DESC, m.chunk_id ASC LIMIT ?"); params.append(top_k)
        rows = self.con.execute("\n".join(sql), params).fetchall()
        return tuple(SearchHit(rank=i+1, score=r[12], chunk_id=r[0], section_id=r[1], doc_id=r[2],
                               source_file_id=r[3], corp_code=r[4], corp_name=r[5], doc_group=r[6],
                               rcept_dt=r[7], path=r[8], locator=r[9], text_prompt_safe=r[10],
                               evidence_id=r[11]) for i, r in enumerate(rows))

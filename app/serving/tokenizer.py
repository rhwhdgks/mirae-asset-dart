"""KiwiTokenizer — 합의안 §5 v0.1 설정. 문서와 질의가 반드시 같은 함수를 쓴다.

설정 전체를 정렬 JSON으로 직렬화한 SHA-256이 TOKENIZER_CONFIG_HASH — index_build_id 재료.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata

TOKENIZER_IMPL_VERSION = "kiwi-tok/1.0"
KEEP_TAGS = {"NNG", "NNP", "NNB", "NR", "NP", "VV", "VA", "VX", "VCP", "VCN",
             "MM", "MAG", "XR", "XPN", "XSN", "XSV", "XSA", "SL", "SH", "SN"}
KIWI_INIT = dict(num_workers=1, model_type="cong", integrate_allomorph=True,
                 load_default_dict=True, load_typo_dict=False, load_multi_dict=True)
TOKENIZE_OPTS = dict(normalize_coda=False, z_coda=True, split_complex=False,
                     compatible_jamo=False)
IDENT_RE = re.compile(r"(?i)[a-z0-9]+(?:[-_][a-z0-9]+)+")   # HCX-005, OLED_PANEL 보호

CONFIG = {
    "impl": TOKENIZER_IMPL_VERSION, "keep_tags": sorted(KEEP_TAGS),
    "kiwi_init": KIWI_INIT, "tokenize_opts": TOKENIZE_OPTS,
    "identifier_regex": IDENT_RE.pattern, "casefold": True, "nfc": True,
    "user_dictionary": [],   # v0.1: 비어 있음 (surface<TAB>tag TSV로 확장)
}
TOKENIZER_CONFIG_HASH = hashlib.sha256(
    json.dumps(CONFIG, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class KiwiTokenizer:
    _instance = None

    def __init__(self):
        import threading
        from kiwipiepy import Kiwi, Match
        self._kiwi = Kiwi(**KIWI_INIT)
        self._match = Match.ALL
        self._lock = threading.Lock()   # Kiwi 인스턴스 동시 호출 보호 (num_workers=1)

    @classmethod
    def get(cls) -> "KiwiTokenizer":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def tokens(self, text: str) -> list[str]:
        if not text:
            return []
        s = unicodedata.normalize("NFC", text).casefold()
        # 식별자 보호: 원문 span을 기억해 두고, 그 span 안의 Kiwi 토큰은 식별자 하나로 접는다
        spans = [(m.start(), m.end(), m.group(0)) for m in IDENT_RE.finditer(s)]
        out: list[str] = []
        emitted: set[int] = set()
        with self._lock:
            toks = list(self._kiwi.tokenize(s, match_options=self._match, **TOKENIZE_OPTS))
        for tk in toks:
            st = tk.start; en = tk.start + tk.len
            hit = next((i for i, (a, b, _) in enumerate(spans) if a <= st and en <= b), None)
            if hit is not None:
                if hit not in emitted:
                    emitted.add(hit); out.append(spans[hit][2])
                continue
            if tk.tag in KEEP_TAGS and tk.form.strip():
                out.append(tk.form)
        return out

    def fts_document(self, text: str) -> str:
        return " ".join(self.tokens(text))

    def fts_match_query(self, text: str) -> str:
        """검증·escape한 query token을 따옴표로 감싸 AND 결합 — raw 문자열을 MATCH에 넣지 않는다."""
        toks = [t.replace('"', '""') for t in self.tokens(text) if t]
        toks = [t for t in toks if re.fullmatch(r"[\w\-\.]+", t)]
        return " AND ".join(f'"{t}"' for t in toks)

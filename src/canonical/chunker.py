"""Section → 검색용 Chunk 분할 (P0-7 / D2).

## 왜 필요한가

`Section` 은 원문 목차 단위라 크기가 통제되지 않는다. 실측(119,161 섹션):

| 크기 | 섹션 수 | 비중 | 글자 비중 |
|---|---|---|---|
| 8,000자 초과 | 14,706 | 12.3% | **68.5%** |
| 최대 | 450,546자 | | 연결재무제표 주석 |

**섹션 수로는 12.3% 지만 글자로는 68.5% 다.** 임베딩 v2 상한이 8,192 토큰이므로
분할 없이는 코퍼스 내용의 3분의 2가 인덱싱되지 않는다. 첨부 감사보고서의 주석이
8~10만 자라 특히 그렇다.

## 예산을 글자로 잡은 근거 — 실측

임베딩 상한은 **토큰**(v2 = 8,192)인데 CLOVA 토크나이저는 API 라 오프라인에서 못 돌린다.
대신 **`cl100k_base` 로 실제 청크 2,841건을 재서** 보수적 하한을 구했다.
`cl100k` 는 한국어에 비효율적이라 **한국어 최적화된 CLOVA 는 더 적은 토큰**을 쓴다 —
즉 이 측정은 안전한 쪽으로 치우친다.

| 지표 | 자/토큰 |
|---|---|
| 중앙값 | 1.21 |
| p10 | 1.03 |
| **p05** | **1.00** |
| 최소 | 0.67 |

한글 산문은 1.2자/토큰인데 **표 위주 마크다운은 1.0** 이다 — 숫자·구분자·`|` 가 토큰을 잡아먹는다.

`p05 = 1.00` 이므로 **8,000자면 조밀한 상위 5% 청크도 8,029토큰**으로 상한 안이다.
10,000자는 넘는다. 그래서 **8,000자**로 잡는다.

예산이 낮으면 조각이 늘어 **검색이 느려지고 메모리를 더 쓴다** — 실측 브루트포스 질의가
25.6만개 30.1ms / 15만개 17.4ms 였다. 정밀 수치 질의는 `Fact` 층이 받으므로
벡터 검색은 서술형을 맡고, 거기서는 **큰 조각이 문맥이 많아 오히려 낫다.**

`CHUNKER_VERSION` 에 예산이 들어 있어 바꾸면 `chunk_id` 가 전부 바뀐다.

## 분할 규칙

```text
1. 블록(빈 줄 구분) 단위로 담다가 예산을 넘으면 새 조각
2. 블록 하나가 예산을 넘으면
   - 표  → 행 단위로 쪼개고 **머리글을 매 조각에 반복**
   - 문단 → 줄 경계로 쪼갠다
3. 그래도 넘는 최소 단위(표 행 하나가 4,000자, 전체의 0.047%)는 쪼개지 않고
   `over_budget` 으로 표시한다
```

**머리글 반복이 핵심이다.** 예산 초과 섹션의 91%가 표 위주인데, 머리글 없는 표 조각은
`| 삼성전자 | 1,234 |` 처럼 열이 무엇인지 알 수 없어 검색에도 답변에도 쓸 수 없다.

표 행은 절대 중간에서 자르지 않는다. 실측 중앙값 51자 · p99 403자라 그럴 일이 드물다.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

__all__ = ["Part", "CHUNK_MAX_CHARS", "CHUNKER_VERSION", "split_section"]

#: 글자 예산. 토큰이 아니라 글자다 — 위 독스트링의 실측 근거 참조.
#: `p05 = 1.00자/토큰` 이므로 8,000자 = 최대 약 8,029토큰 (임베딩 v2 상한 8,192).
CHUNK_MAX_CHARS = 8000
CHUNKER_VERSION = f"chunk/1.0-block-table-header@{CHUNK_MAX_CHARS}"

_SEP_PREFIX = "|---"


@dataclass(frozen=True)
class Part:
    """섹션에서 잘라낸 조각 하나."""

    part_no: int
    text: str
    #: 표 머리글을 반복해 넣은 조각인가 (원문에 없던 줄이 앞에 붙었다)
    header_repeated: bool
    #: 쪼갤 수 없는 단위 하나가 예산을 넘어 그대로 둔 조각인가
    over_budget: bool

    @property
    def n_chars(self) -> int:
        return len(self.text)

    @property
    def content_hash(self) -> str:
        return hashlib.sha1(self.text.encode("utf-8")).hexdigest()[:32]


def _table_head(lines: list[str]) -> list[str]:
    """표 블록의 머리글 — 구분선(`|---|`)까지. 표가 아니면 빈 목록.

    `_table_to_markdown` 은 표 앞에 `(단위: 백만원)` 을 한 줄 붙이므로 그것도 머리글에
    포함한다. 조각마다 단위가 없으면 숫자의 의미가 사라진다.
    """
    for i, line in enumerate(lines):
        if line.startswith(_SEP_PREFIX):
            return lines[: i + 1]
        if i > 2:
            break
    return []


def _split_block(block: str, budget: int) -> list[tuple[str, bool, bool]]:
    """예산을 넘는 블록 하나 → `(텍스트, 머리글반복, 예산초과)` 목록."""
    lines = block.split("\n")
    head = _table_head(lines)
    head_text = "\n".join(head)

    # 머리글이 예산을 다 먹으면 반복할 수 없다. 평범한 줄로 되돌린다.
    # DART 는 긴 산문을 **단일 셀 표 한 행**으로 넣기도 하는데(회계정책 6,452자 등),
    # 그러면 머리글이 블록 전체가 되고 본문이 비어 **블록이 통째로 사라졌다.**
    if head and len(head_text) + 1 >= budget:
        head, head_text = [], ""

    body = lines[len(head):]
    head_cost = len(head_text) + 1 if head else 0

    if not body:                       # 머리글만 있는 블록도 버리지 않는다
        return [(head_text, False, len(head_text) > budget)]

    out: list[tuple[str, bool, bool]] = []
    buf: list[str] = []
    size = 0
    first = True

    def flush() -> None:
        nonlocal buf, size, first
        if not buf:
            return
        if head and not first:
            out.append((head_text + "\n" + "\n".join(buf), True, False))
        else:
            out.append(("\n".join(head + buf) if head else "\n".join(buf), False, False))
        buf, size = [], 0
        first = False

    for line in body:
        # 최소 단위 하나가 예산을 넘으면 쪼개지 않는다. 표 행을 중간에서 자르면
        # 셀 하나가 두 조각으로 갈려 양쪽 다 틀린 값이 된다.
        if len(line) + head_cost > budget:
            flush()
            out.append(((head_text + "\n" + line) if head else line, bool(head), True))
            first = False
            continue
        if size and size + len(line) + 1 + head_cost > budget:
            flush()
        buf.append(line)
        size += len(line) + 1
    flush()
    return out


def split_section(text: str, budget: int = CHUNK_MAX_CHARS) -> list[Part]:
    """섹션 텍스트 → 조각 목록. 예산 이하면 통째로 한 조각이다."""
    if not text:
        return [Part(0, "", False, False)]
    if len(text) <= budget:
        return [Part(0, text, False, False)]

    pieces: list[tuple[str, bool, bool]] = []
    buf: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal buf, size
        if buf:
            pieces.append(("\n\n".join(buf), False, False))
            buf, size = [], 0

    for block in text.split("\n\n"):
        if not block:
            continue
        if len(block) > budget:
            flush()
            pieces.extend(_split_block(block, budget))
            continue
        if size and size + len(block) + 2 > budget:
            flush()
        buf.append(block)
        size += len(block) + 2
    flush()

    return [Part(i, t, hr, ob) for i, (t, hr, ob) in enumerate(pieces)]

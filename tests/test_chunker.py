"""청커 회귀 (P0-7). `python -m tests.test_chunker` — 실패 시 종료코드 1."""
from src.canonical.chunker import CHUNK_MAX_CHARS, split_section

ok = True


def brief(x, n=160):
    s = repr(x)
    return s if len(s) <= n else s[:n] + f"… (총 {len(x)}개)"


def check(name, got, want):
    global ok
    good = got == want
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  {name}")
    if not good:
        print(f"        기대 {brief(want)}\n        실제 {brief(got)}")


def lines_of(parts, head):
    """머리글을 걷어낸 실제 내용 줄.

    첫 조각은 원문 그대로라 `header_repeated` 가 False 지만 머리글은 들어 있다.
    무손실 비교에서는 둘 다 걷어내야 한다.
    """
    out = []
    for p in parts:
        ls = p.text.split("\n")
        if ls[:len(head)] == head:
            ls = ls[len(head):]
        out.extend(ls)
    return out


# 예산 이하는 그대로 한 조각
check("예산 이하 → 1조각", len(split_section("짧은 글", 4000)), 1)
check("빈 텍스트 → 1조각", len(split_section("", 4000)), 1)

# 표 분할 — 머리글이 매 조각에 반복되고 내용은 무손실
head = ["(단위: 백만원)", "| 회사 | 매출 |", "|---|---|"]
rows = [f"| 회사{i} | {i * 1000:,} |" for i in range(300)]
table = "\n".join(head + rows)
parts = split_section(table, 500)
check("표: 모든 조각이 머리글 보유", all("|---|---|" in p.text for p in parts), True)
check("표: 내용 행 무손실", lines_of(parts, head), rows)
check("표: 예산 준수", all(p.n_chars <= 500 for p in parts), True)
check("표: part_no 연속", [p.part_no for p in parts], list(range(len(parts))))

# 쪼갤 수 없는 단위가 예산을 넘으면 자르지 않고 표시한다
long_row = "| " + "가" * 3000 + " |"
parts = split_section("\n".join(head + [long_row, rows[0]]), 500)
check("긴 행: 자르지 않음", any(long_row in p.text for p in parts), True)
check("긴 행: over_budget 표시", any(p.over_budget for p in parts), True)

# 문단 분할
paras = [f"문단 {i}. " + "내용" * 60 for i in range(20)]
parts = split_section("\n\n".join(paras), 500)
check("문단: 무손실", "".join(p.text for p in parts).replace("\n", ""),
      "".join(paras).replace("\n", ""))
check("문단: 예산 준수", all(p.n_chars <= 500 for p in parts), True)

# 표와 문단이 섞인 섹션 — 표 블록이 문단에 흡수되지 않는다
mixed = "머리말\n\n" + table + "\n\n맺음말"
parts = split_section(mixed, 500)
check("혼합: 머리말 보존", any("머리말" in p.text for p in parts), True)
check("혼합: 맺음말 보존", any("맺음말" in p.text for p in parts), True)

# 블록 전체가 머리글인 경우 — 실제로 384개 섹션이 통째로 사라졌던 버그
# DART 는 긴 산문을 단일 셀 표 한 행으로 넣는다 (회계정책 6,452자 등).
prose_row = "| " + "회계정책 설명. " * 800 + " |"
whole_head = prose_row + "\n|---|"
parts = split_section(whole_head, 4000)
check("블록 전체가 머리글: 내용 보존",
      "".join(p.text for p in parts).count("회계정책 설명."), 800)
check("블록 전체가 머리글: 조각 존재", len(parts) > 0, True)

# 머리글이 예산을 넘으면 반복하지 않는다 (반복하면 본문이 들어갈 자리가 없다)
big_head = "| " + "머" * 3000 + " |\n|---|\n| a |\n| b |"
parts = split_section(big_head, 500)
check("과대 머리글: 반복 안 함", any(p.header_repeated for p in parts), False)
check("과대 머리글: 본문 보존",
      all(any(v in p.text for p in parts) for v in ("| a |", "| b |")), True)

# content_hash 는 내용이 같으면 같다
a = split_section(table, 500)[0]
b = split_section(table, 500)[0]
check("content_hash 결정적", a.content_hash, b.content_hash)

print(f"\n예산 기본값 {CHUNK_MAX_CHARS:,}자")
print("전체:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)

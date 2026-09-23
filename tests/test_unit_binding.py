"""단위 표기 판정·결합 회귀 (T-07·T-08·X-09).

DART 는 `(단위 : 백만원)` 을 본표 **바로 앞의 1칸짜리 표**로 넣는다. 이를 표로 렌더링하면
모든 재무표 앞에 의미 없는 표가 붙으므로 단위만 뽑고 표는 버린다. 그 판정이 두 방향으로
틀렸었다.

- 너무 넓게: `※ 상세 현황은 (단위 : 백만원) 기준…` 같은 각주표까지 삭제 (T-07)
- 너무 좁게: `(단위 : 천CGRT(조선), 천TON(해양))` 을 `천CGRT(조선` 으로 잘라 뽑음
- 너무 오래: 사이에 문단이 끼어도 다음 표에 계속 붙음 (X-09)
"""
from src.ingest.dart_xml import parse_xml, unit_display

ok = True


def check(name, got, want):
    global ok
    good = got == want
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  {name}")
    if not good:
        print(f"        기대 {want!r}\n        실제 {got!r}")


def doc(inner):
    return ("<DOCUMENT><DOCUMENT-NAME>t</DOCUMENT-NAME>"
            f"<SECTION-1><TITLE>S</TITLE>{inner}</SECTION-1></DOCUMENT>")


TBL = ("<TABLE><THEAD><TR><TD>항목</TD><TD>값</TD></TR></THEAD>"
       "<TBODY><TR><TD>매출</TD><TD>100</TD></TR></TBODY></TABLE>")
def unit_tbl(t):
    return f"<TABLE><TBODY><TR><TD>{t}</TD></TR></TBODY></TABLE>"


# 판정 자체
check("단순 단위", unit_display("(단위 : 백만원)"), "(단위 : 백만원)")
check("중첩 괄호 복합 단위", unit_display("(단위 : 천CGRT(조선), 천TON(해양))"),
      "(단위 : 천CGRT(조선), 천TON(해양))")
check("이중 단위", unit_display("(외화단위: 천, 원화단위: 천원)"),
      "(외화단위: 천, 원화단위: 천원)")
check("각주는 단위가 아니다",
      unit_display("※ 상세 현황은 (단위 : 백만원) 기준으로 작성되었습니다"), None)
check("기준일은 단위가 아니다", unit_display("(기준일 : 2021.12.31)"), None)

check("대괄호 단위 wrapper 보존", unit_display("[단위:백만원]"), "[단위:백만원]")
check("짝이 다른 괄호는 단위-only가 아니다",
      unit_display("[제76기] (단위 : 백만원)"), None)
# 결합
t = parse_xml(doc(unit_tbl("(단위 : 백만원)") + TBL)).sections[0].text
check("T-08 단위 → 바로 다음 표에 적용", "(단위 : 백만원)" in t, True)

t = parse_xml(doc(unit_tbl("(단위 : 백만원)") + "<P>중간 문단.</P>" + TBL)).sections[0].text
check("X-09 단위 → 문단 → 표: 누출 없음", "단위" in t, False)

t = parse_xml(doc(unit_tbl("(단위 : 천CGRT(조선), 천TON(해양))") + TBL)).sections[0].text
check("복합 단위가 잘리지 않음", "(단위 : 천CGRT(조선), 천TON(해양))" in t, True)

t = parse_xml(doc(unit_tbl("※ 상세 현황은 (단위 : 백만원) 기준으로 작성되었습니다"))).sections[0].text
check("T-07 각주표 보존", "상세 현황" in t, True)

# 단위는 한 번만 소비된다 — 두 번째 표에는 붙지 않는다
t = parse_xml(doc(unit_tbl("(단위 : 백만원)") + TBL + TBL)).sections[0].text
check("단위 1회 소비", t.count("(단위 : 백만원)"), 1)

# 괄호 종류가 다른 복합 표기는 단위 전용 표로 삭제되면 안 된다.
composite = parse_xml(doc(unit_tbl("[제76기] (단위 : 백만원)") + TBL)).sections[0]
check("기간+단위 복합 표 보존",
      (composite.n_tables, "[제76기] (단위 : 백만원)" in composite.text),
      (2, True))
square = parse_xml(doc(unit_tbl("[단위:백만원]") + TBL)).sections[0]
check("대괄호 단위표 one-shot 결합·wrapper 보존",
      (square.n_tables, square.text.count("[단위:백만원]")), (1, 1))

print("\n전체:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)

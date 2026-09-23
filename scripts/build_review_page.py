#!/usr/bin/env python3
"""395문항 질문·답변을 사람이 읽는 한 페이지로 묶는다.

`think_trace` 와 `retrieved_context` 는 뺀다 — 읽는 사람이 보는 것은 질문과
사용자용 답변이다. 세트·그룹·길이로 좁히고 본문을 검색할 수 있어야 395건을
훑을 수 있다.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re


GROUP_LABEL = {
    "G-A": "Gold 재무", "G-I": "Gold 사건", "G-O": "Gold 종합",
    "G-U": "Gold 거절", "R-A": "구어 모호", "R-B": "구어 압박",
    "R-F": "구어 거짓전제", "R-P": "구어 변형",
    "EDGE": "Edge43", "EG": "EdgeGap v0.1", "EG2": "EdgeGap v0.2",
    "DEV-CLR": "되묻기", "DEV-ENT": "법인 식별", "DEV-EVT": "사건·정정",
    "DEV-FDR": "파생 계산", "DEV-FIN": "재무", "DEV-INV": "투자계획",
    "DEV-NAR": "서술 비교", "DEV-SAFE": "안전", "K": "커버리지 61사",
    "HM": "지분공시·마스킹", "RPC": "응답 정책",
}


def group_of(qid: str) -> str:
    match = re.match(r"(EG2|EG|EDGE|DEV-[A-Z]+|G-[A-Z]|R-[A-Z]|RPC|HM|K)", qid)
    return match.group(1) if match else qid.split("-")[0]


def load(path: Path) -> list[dict]:
    return [json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--print", dest="printable", action="store_true",
                    help="탭·접기 없이 전부 펼친 인쇄용 정적 HTML을 낸다")
    args = ap.parse_args()

    rows: list[dict] = []
    for path in args.responses:
        for row in load(path):
            answer = (row.get("response") or {}).get("answer", "")
            rows.append({
                "id": row["question_id"],
                "g": group_of(row["question_id"]),
                "q": row["question"],
                "a": answer,
                "n": len(answer),
                "r": sorted(set(re.findall(r"\b\d{14}\b", answer)))[:6],
            })
    rows.sort(key=lambda item: (list(GROUP_LABEL).index(item["g"])
                                if item["g"] in GROUP_LABEL else 99, item["id"]))
    if args.printable:
        args.out.write_text(printable(rows), encoding="utf-8")
        print(f"{args.out} · {len(rows)}문항 · "
              f"{args.out.stat().st_size:,} bytes")
        return 0
    payload = json.dumps(
        {"rows": rows, "labels": GROUP_LABEL}, ensure_ascii=False,
        separators=(",", ":"))
    args.out.write_text(
        TEMPLATE.replace("__DATA__", html.escape(payload, quote=False)),
        encoding="utf-8")
    print(f"{args.out} · {len(rows)}문항 · {args.out.stat().st_size:,} bytes")
    return 0


TEMPLATE = r"""<title>공시 답변 검토본</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Gowun+Batang:wght@400;700&family=IBM+Plex+Sans+KR:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --ground:#F5F7F8; --panel:#FFFFFF; --sunk:#EDF1F3;
  --ink:#16202B; --muted:#5C6B7A; --faint:#8494A2;
  --rule:#DBE2E7; --accent:#16645F; --accent-soft:#E2EEEC; --warn:#96650C;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#11161B; --panel:#171E25; --sunk:#1D262E;
  --ink:#E6ECF1; --muted:#9CACB9; --faint:#71818E;
  --rule:#2A343D; --accent:#5FBDB4; --accent-soft:#16302E; --warn:#D9A441;
}}
:root[data-theme="dark"]{
  --ground:#11161B; --panel:#171E25; --sunk:#1D262E;
  --ink:#E6ECF1; --muted:#9CACB9; --faint:#71818E;
  --rule:#2A343D; --accent:#5FBDB4; --accent-soft:#16302E; --warn:#D9A441;
}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);
  font:400 15px/1.7 "IBM Plex Sans KR",system-ui,sans-serif;margin:0}
header{position:sticky;top:0;z-index:5;background:var(--ground);
  border-bottom:1px solid var(--rule);padding:18px 22px 12px}
h1{font:700 25px/1.25 "Gowun Batang",serif;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:13.5px;margin:0 0 14px}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
input[type=search]{flex:1 1 260px;min-width:200px;padding:8px 11px;
  border:1px solid var(--rule);border-radius:7px;background:var(--panel);
  color:var(--ink);font:400 14px/1.5 "IBM Plex Sans KR",sans-serif}
input[type=search]:focus-visible,button:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
button{border:1px solid var(--rule);background:var(--panel);color:var(--muted);
  border-radius:999px;padding:6px 12px;cursor:pointer;
  font:500 12.5px/1 "IBM Plex Sans KR",sans-serif}
button:hover{border-color:var(--accent);color:var(--ink)}
button[aria-pressed=true]{background:var(--accent-soft);border-color:var(--accent);
  color:var(--accent);font-weight:600}
main{padding:20px 22px 64px;max-width:1080px;margin:0 auto}
.count{color:var(--faint);font-size:12.5px;margin:0 0 16px;
  font-variant-numeric:tabular-nums}
.grp{font:700 15px/1.4 "Gowun Batang",serif;color:var(--accent);
  margin:26px 0 10px;padding-bottom:5px;border-bottom:1px solid var(--rule)}
.grp span{color:var(--faint);font:400 12px/1 "IBM Plex Mono",monospace;
  margin-left:8px;font-variant-numeric:tabular-nums}
article{background:var(--panel);border:1px solid var(--rule);border-radius:9px;
  padding:14px 16px;margin:0 0 10px}
.meta{display:flex;flex-wrap:wrap;gap:7px;align-items:center;margin-bottom:7px}
.id{font:500 12px/1 "IBM Plex Mono",monospace;color:var(--accent);
  background:var(--accent-soft);padding:3px 7px;border-radius:5px}
.len{font:400 11.5px/1 "IBM Plex Mono",monospace;color:var(--faint);
  font-variant-numeric:tabular-nums}
.len.long{color:var(--warn);font-weight:500}
.q{font:400 16px/1.6 "Gowun Batang",serif;margin:0 0 10px;text-wrap:balance}
.a{background:var(--sunk);border-radius:7px;padding:11px 13px;
  word-break:break-word;font-size:14px;line-height:1.75;
  overflow-x:auto;max-height:19em}
.a p{margin:0 0 .45em;white-space:pre-wrap}
.a p:last-child{margin-bottom:0}
.a p:empty{margin:0;height:.4em}
.a ul{margin:.2em 0 .6em;padding-left:1.15em}
.a li{margin:.12em 0}
.tw{overflow-x:auto;margin:.5em 0 .7em}
.a table{border-collapse:collapse;font-size:13px;min-width:100%}
.a th,.a td{border:1px solid var(--rule);padding:5px 9px;text-align:left;
  vertical-align:top;white-space:nowrap}
.a th{background:var(--panel);font-weight:500;color:var(--muted)}
.a td{font-variant-numeric:tabular-nums}
.a.open{max-height:none}
.more{margin-top:7px;font-size:12.5px;padding:4px 10px}
.rc{margin-top:7px;font:400 11.5px/1.6 "IBM Plex Mono",monospace;color:var(--faint)}
.empty{color:var(--muted);padding:40px 0;text-align:center}
</style>
<header>
  <h1>공시 답변 검토본</h1>
  <p class="sub">Core334 + 커버리지 61사 = 395문항. 질문과 사용자용 답변만 담았다 — 추론 흔적과 조회 원문은 뺐다.</p>
  <div class="controls">
    <input type="search" id="q" placeholder="질문·답변 본문 검색  ( / 키로 이동 )" autocomplete="off">
    <button id="long" aria-pressed="false">1,500자 초과만</button>
    <button id="reset">전체</button>
  </div>
  <div class="controls" id="tabs" style="margin-top:8px"></div>
</header>
<main>
  <p class="count" id="count"></p>
  <div id="list"></div>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const rows = DATA.rows, labels = DATA.labels;
const state = {group:null, term:'', long:false};
const esc = s => s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

// 답변에는 마크다운 표와 목록이 섞여 있다. 원문 그대로 두면 사용자에게는
// 파이프 문자로만 보인다. 표와 목록만 실제 요소로 세우고 나머지 줄은 그대로
// 둔다. 조각을 먼저 이스케이프한 뒤 조립하므로 답변 내용이 markup 이 되지
// 않는다 — 전체를 마크다운 파서에 넘기지 않는 이유다.
const CELL = /^\s*\|(.+)\|\s*$/;
const RULE = /^\s*\|[\s:|-]+\|\s*$/;
const cells = line => line.replace(/^\s*\||\|\s*$/g, '').split('|').map(v => v.trim());

function renderAnswer(text) {
  const lines = text.split('\n');
  let out = '', i = 0;
  while (i < lines.length) {
    if (CELL.test(lines[i]) && !RULE.test(lines[i])) {
      const block = [];
      while (i < lines.length && CELL.test(lines[i])) { block.push(lines[i]); i++; }
      const head = cells(block[0]);
      const body = block.slice(RULE.test(block[1] || '') ? 2 : 1)
        .filter(row => !RULE.test(row)).map(cells);
      out += '<div class="tw"><table><thead><tr>'
        + head.map(c => `<th>${esc(c)}</th>`).join('')
        + '</tr></thead><tbody>'
        + body.map(row => '<tr>' + row.map(c => `<td>${esc(c)}</td>`).join('') + '</tr>').join('')
        + '</tbody></table></div>';
      continue;
    }
    if (/^\s*-\s+/.test(lines[i])) {
      const items = [];
      while (i < lines.length && /^\s*-\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*-\s+/, '')); i++;
      }
      out += '<ul>' + items.map(v => `<li>${esc(v)}</li>`).join('') + '</ul>';
      continue;
    }
    out += `<p>${esc(lines[i])}</p>`;
    i++;
  }
  return out;
}

const tabs = document.getElementById('tabs');
const counts = {};
rows.forEach(r => counts[r.g] = (counts[r.g]||0)+1);
Object.keys(labels).filter(g => counts[g]).forEach(g => {
  const b = document.createElement('button');
  b.textContent = `${labels[g]} ${counts[g]}`;
  b.setAttribute('aria-pressed','false');
  b.onclick = () => { state.group = state.group===g ? null : g; render(); };
  b.dataset.g = g; tabs.appendChild(b);
});

function render(){
  document.querySelectorAll('#tabs button').forEach(b =>
    b.setAttribute('aria-pressed', String(b.dataset.g === state.group)));
  document.getElementById('long').setAttribute('aria-pressed', String(state.long));
  const t = state.term.trim().toLowerCase();
  const hits = rows.filter(r =>
    (!state.group || r.g === state.group) &&
    (!state.long || r.n > 1500) &&
    (!t || r.q.toLowerCase().includes(t) || r.a.toLowerCase().includes(t) ||
     r.id.toLowerCase().includes(t)));
  document.getElementById('count').textContent =
    `${hits.length}문항 / 전체 ${rows.length}문항`;
  const list = document.getElementById('list');
  list.innerHTML = '';
  if (!hits.length){
    list.innerHTML = '<p class="empty">조건에 맞는 문항이 없습니다.</p>';
    return;
  }
  let current = null;
  hits.forEach(r => {
    if (r.g !== current){
      current = r.g;
      const h = document.createElement('h2');
      h.className = 'grp';
      h.innerHTML = `${esc(labels[r.g]||r.g)}<span>${esc(r.g)}</span>`;
      list.appendChild(h);
    }
    const art = document.createElement('article');
    const long = r.n > 1500;
    art.innerHTML =
      `<div class="meta"><span class="id">${esc(r.id)}</span>` +
      `<span class="len${long?' long':''}">${r.n.toLocaleString()}자</span></div>` +
      `<p class="q">${esc(r.q)}</p>` +
      `<div class="a">${r.a ? renderAnswer(r.a) : '<i>답변 없음</i>'}</div>` +
      (r.r.length ? `<div class="rc">근거 ${r.r.map(esc).join(' · ')}</div>` : '');
    const body = art.querySelector('.a');
    if (r.n > 900){
      const btn = document.createElement('button');
      btn.className = 'more'; btn.textContent = '전문 보기';
      btn.onclick = () => {
        body.classList.toggle('open');
        btn.textContent = body.classList.contains('open') ? '접기' : '전문 보기';
      };
      art.insertBefore(btn, art.querySelector('.rc'));
    } else { body.classList.add('open'); }
    list.appendChild(art);
  });
}
document.getElementById('q').addEventListener('input', e => {
  state.term = e.target.value; render();
});
document.getElementById('long').onclick = () => { state.long = !state.long; render(); };
document.getElementById('reset').onclick = () => {
  state.group = null; state.long = false; state.term = '';
  document.getElementById('q').value = ''; render();
};
document.addEventListener('keydown', e => {
  if (e.key === '/' && e.target.tagName !== 'INPUT'){
    e.preventDefault(); document.getElementById('q').focus();
  }
});
render();
</script>
"""




PRINT_CSS = r"""
@page { size: A4; margin: 16mm 15mm 18mm; }
@page { @bottom-center { content: counter(page); } }
:root{ --ink:#16202B; --muted:#5C6B7A; --rule:#D6DDE2; --accent:#16645F; }
*{ box-sizing:border-box; }
body{ margin:0; color:var(--ink); background:#fff;
      font:10.5pt/1.62 "IBM Plex Sans KR","Noto Sans KR",system-ui,sans-serif;
      -webkit-print-color-adjust:exact; print-color-adjust:exact; }
.cover{ page-break-after:always; padding-top:52mm; }
.cover h1{ font:600 30pt/1.2 "Gowun Batang","Noto Serif KR",serif; margin:0 0 6mm; }
.cover .sub{ color:var(--muted); font-size:11pt; margin:0 0 14mm; }
.cover dl{ display:grid; grid-template-columns:26mm 1fr; gap:2.5mm 6mm;
           margin:0; border-top:2px solid var(--accent); padding-top:5mm; }
.cover dt{ color:var(--muted); font-size:9pt; }
.cover dd{ margin:0; font-size:10pt; }
h2{ font:600 15pt/1.3 "Gowun Batang","Noto Serif KR",serif;
    margin:0 0 5mm; padding:0 0 2mm; border-bottom:2px solid var(--accent);
    page-break-after:avoid; page-break-before:always; }
h2 .n{ float:right; font-family:"IBM Plex Sans KR",sans-serif;
       font-size:9pt; font-weight:400; color:var(--muted); }
h2:first-of-type{ page-break-before:avoid; }
.item{ page-break-inside:avoid; margin:0 0 6mm; padding:0 0 5mm;
       border-bottom:1px solid var(--rule); }
.meta{ font-size:8.5pt; color:var(--muted); margin:0 0 1.5mm;
       font-family:"IBM Plex Mono",ui-monospace,monospace; }
.meta .id{ color:var(--accent); font-weight:600; }
.q{ font:600 11.5pt/1.5 "Gowun Batang","Noto Serif KR",serif;
    margin:0 0 2.5mm; }
.a{ white-space:pre-wrap; word-break:break-word; margin:0;
    padding-left:4mm; border-left:2px solid var(--rule); }
.a .rc{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:9pt; }
"""


def printable(rows: list[dict]) -> str:
    """탭도 접기도 없는, 전부 펼쳐진 인쇄용 한 벌.

    화면판은 세트 탭과 900자 이상 접기로 훑어보게 만든 것이라 그대로 인쇄하면
    보이지 않는 문항이 그대로 빠진다. 인쇄본은 순서대로 전문을 싣는다.
    """

    by_group: dict[str, list[dict]] = {}
    for row in rows:
        by_group.setdefault(row["g"], []).append(row)

    parts = [
        '<title>공시 답변 검토본</title>',
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=Gowun+Batang:wght@400;700&family=IBM+Plex+Sans+KR:wght@400;600'
        '&family=IBM+Plex+Mono&display=swap">',
        f"<style>{PRINT_CSS}</style>",
        '<section class="cover"><h1>공시 답변 검토본</h1>'
        f'<p class="sub">질문 {len(rows)}문항과 답변 전문</p><dl>'
        f'<dt>문항 수</dt><dd>{len(rows)}건</dd>'
        f'<dt>세트</dt><dd>{len(by_group)}개</dd>'
        f'<dt>총 분량</dt><dd>{sum(r["n"] for r in rows):,}자</dd>'
        '</dl></section>',
    ]
    for group, items in by_group.items():
        label = GROUP_LABEL.get(group, group)
        parts.append(f'<h2>{html.escape(label)}'
                     f'<span class="n">{len(items)}문항</span></h2>')
        for row in items:
            receipts = ("  ·  " + " ".join(row["r"])) if row["r"] else ""
            parts.append(
                '<article class="item">'
                f'<p class="meta"><span class="id">{html.escape(row["id"])}'
                f'</span>  ·  {row["n"]:,}자'
                f'<span class="rc">{html.escape(receipts)}</span></p>'
                f'<p class="q">{html.escape(row["q"])}</p>'
                f'<p class="a">{html.escape(row["a"])}</p></article>')
    return "\n".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())

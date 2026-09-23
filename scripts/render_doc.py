#!/usr/bin/env python3
"""마크다운 문서를 인쇄용 단일 HTML로 감싼다 (브라우저에서 PDF로 저장).

이 저장소에는 pandoc·weasyprint가 없으므로, 마크다운 원본을 그대로 HTML에 심고
브라우저에서 marked(마크다운) + mermaid(다이어그램)로 렌더한다. 원본은 계속
마크다운 한 벌만 고치면 되고, HTML은 출력용 파생물이다.

    python3 scripts/render_doc.py docs/기술제안서.md
    → out/docs/기술제안서.html  (브라우저에서 열고 인쇄 → PDF로 저장, A4 여백 기본)
"""
from __future__ import annotations

import html
import json
import pathlib
import sys

TEMPLATE = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
  const src = JSON.parse(document.getElementById('src').textContent);
  marked.setOptions({gfm: true, breaks: false});
  document.getElementById('doc').innerHTML = marked.parse(src);
  // ```mermaid 코드블록을 mermaid 컨테이너로 승격
  document.querySelectorAll('pre > code.language-mermaid').forEach(c => {
    const d = document.createElement('div');
    d.className = 'mermaid';
    d.textContent = c.textContent;
    c.parentElement.replaceWith(d);
  });
  mermaid.initialize({startOnLoad: false, theme: 'neutral', securityLevel: 'loose'});
  await mermaid.run({querySelector: '.mermaid'});
  document.body.dataset.ready = '1';
</script>
<style>
  :root { --fg:#1a1a1a; --muted:#5b6572; --line:#d8dee6; --accent:#0b5cad; --bg:#fff; --code-bg:#f5f7fa; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font-family:"Pretendard","Apple SD Gothic Neo","Noto Sans KR","Malgun Gothic",system-ui,sans-serif;
         font-size:10.5pt; line-height:1.7; }
  #doc { max-width: 900px; margin: 0 auto; padding: 40px 32px 80px; }
  h1 { font-size:22pt; letter-spacing:-.02em; border-bottom:3px solid var(--accent);
       padding-bottom:.4em; margin:0 0 1.2em; }
  h2 { font-size:15pt; margin:2.4em 0 .8em; padding-top:.4em; border-top:1px solid var(--line);
       page-break-before:auto; break-after:avoid; }
  h3 { font-size:12.5pt; margin:1.8em 0 .6em; color:var(--accent); break-after:avoid; }
  h4 { font-size:11pt; margin:1.4em 0 .5em; break-after:avoid; }
  p, li { orphans:3; widows:3; }
  a { color:var(--accent); text-decoration:none; }
  table { border-collapse:collapse; width:100%; margin:1em 0; font-size:9.5pt;
          break-inside:avoid; }
  th, td { border:1px solid var(--line); padding:6px 9px; text-align:left; vertical-align:top; }
  th { background:#eef2f7; font-weight:600; }
  tbody tr:nth-child(even) { background:#fafbfc; }
  code { background:var(--code-bg); padding:1px 5px; border-radius:3px;
         font-family:"JetBrains Mono","D2Coding",ui-monospace,Menlo,Consolas,monospace; font-size:9pt; }
  pre { background:var(--code-bg); border:1px solid var(--line); border-radius:5px;
        padding:12px 14px; overflow-x:auto; break-inside:avoid; }
  pre code { background:none; padding:0; font-size:8.8pt; line-height:1.55; }
  blockquote { margin:1em 0; padding:.6em 1em; border-left:4px solid var(--accent);
               background:#f4f8fc; color:var(--muted); break-inside:avoid; }
  blockquote strong { color:var(--fg); }
  hr { border:0; border-top:1px solid var(--line); margin:2.4em 0; }
  .mermaid { text-align:center; margin:1.4em 0; break-inside:avoid; }
  .mermaid svg { max-width:100%; height:auto; }
  @page { size: A4; margin: 16mm 14mm; }
  @media print {
    #doc { max-width:none; padding:0; }
    a { color:var(--fg); }
    h2 { page-break-before: always; }
    h2:first-of-type { page-break-before: avoid; }
  }
</style>
</head>
<body>
<script type="application/json" id="src">__SRC__</script>
<article id="doc">렌더 중…</article>
</body>
</html>
"""


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    src_path = pathlib.Path(sys.argv[1])
    if not src_path.is_file():
        print(f"입력 파일 없음: {src_path}", file=sys.stderr)
        return 1
    text = src_path.read_text(encoding="utf-8")
    title = next(
        (ln.lstrip("# ").strip() for ln in text.splitlines() if ln.startswith("# ")),
        src_path.stem,
    )
    out_path = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("out/docs") / f"{src_path.stem}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__SRC__", json.dumps(text))
    out_path.write_text(doc, encoding="utf-8")
    print(f"생성: {out_path}")
    print("→ 브라우저로 열고 인쇄(Ctrl/Cmd+P) → 대상 'PDF로 저장', 배경 그래픽 켜기")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

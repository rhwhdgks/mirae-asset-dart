#!/usr/bin/env python3
"""Export selected saved live answers as a readable Markdown handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _source(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("source must be LABEL=PATH")
    return label.strip(), Path(path.strip())


def _quote(value: str) -> str:
    lines = (value or "").splitlines() or [""]
    return "\n".join(f"> {line}" if line else ">" for line in lines)


def _load(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: object required")
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", type=_source, required=True)
    parser.add_argument("--exclude", default="")
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    excluded = {value.strip() for value in args.exclude.split(",") if value.strip()}
    seen: set[str] = set()
    sections: list[tuple[str, Path, list[dict]]] = []
    total = 0
    for label, path in args.source:
        selected = []
        for row in _load(path):
            question_id = str(row.get("question_id") or "").strip()
            if not question_id or question_id in seen:
                raise ValueError(f"missing or duplicate question_id: {question_id!r}")
            seen.add(question_id)
            if question_id in excluded:
                continue
            response = row.get("response") or {}
            question = str(row.get("question") or response.get("question") or "").strip()
            answer = str(response.get("answer") or "").strip()
            if not question or not answer:
                raise ValueError(f"{question_id}: nonblank question and answer required")
            selected.append({
                "question_id": question_id,
                "question": question,
                "answer": answer,
            })
        sections.append((label, path, selected))
        total += len(selected)

    if total != args.expected_count:
        raise ValueError(f"selected {total}, expected {args.expected_count}")

    lines = [
        f"# {args.title}",
        "",
        f"총 {total}건이다. 저장된 live 응답 중 의미 통과로 판정된 문항만 수록했으며, "
        "실행 trace와 retrieved context는 제외했다.",
        "",
        "## 구성",
        "",
    ]
    for label, path, rows in sections:
        lines.append(f"- {label}: {len(rows)}건 — `{path}`")
    lines.extend([
        f"- 제외 ID: {', '.join(f'`{value}`' for value in sorted(excluded))}",
        "",
    ])

    for label, _, rows in sections:
        lines.extend([f"## {label} ({len(rows)}건)", ""])
        for row in rows:
            lines.extend([
                f"### `{row['question_id']}`",
                "",
                "#### 질문 전문",
                "",
                _quote(row["question"]),
                "",
                "#### 자연어 응답 전문",
                "",
                _quote(row["answer"]),
                "",
            ])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

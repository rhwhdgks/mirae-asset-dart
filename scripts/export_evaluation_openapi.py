"""평가 API OpenAPI를 결정적 JSON으로 출력한다."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

from agent.api import create_app


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        help="생략하면 stdout에 출력하고, 지정하면 새 JSON 파일로 기록합니다.")
    args = parser.parse_args()
    app = create_app()
    try:
        payload = json.dumps(
            app.openapi(), ensure_ascii=False, sort_keys=True,
            separators=(",", ":")) + "\n"
    finally:
        app.state.evaluation_gateway.close()
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

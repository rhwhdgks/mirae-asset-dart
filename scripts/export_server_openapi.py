"""서버 OpenAPI 스키마 내보내기 — 정본·HCX 없이 앱 객체만 만든다 (boot 없음).

    PYTHONPATH=. .venv/bin/python scripts/export_server_openapi.py [out/openapi_server.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.app import create_app  # noqa: E402
from server.runtime import ServerRuntime  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "out" / "openapi_server.json"
    app = create_app(ServerRuntime(), autostart=False)
    spec = app.openapi()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    paths = sorted(spec["paths"])
    print(f"→ {out} ({len(paths)} paths, version {spec['info']['version']})")
    for p in paths:
        print("  ", ", ".join(m.upper() for m in spec["paths"][p]), p)


if __name__ == "__main__":
    main()

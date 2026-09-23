#!/usr/bin/env python3
"""비밀·응답 본문을 출력하지 않는 HCX-007 Structured Output 1회 smoke."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from agent.contracts import ContractModel
from agent.hcx_schema import compile_hcx_schema
from agent.providers.hcx007 import (
    HcxGenerationConfig,
    HcxMessage,
    HcxRequest,
    HcxStructuredClient,
)


ROOT = Path(__file__).resolve().parents[1]


class SmokePayload(ContractModel):
    transport_status: Literal["ok"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live", action="store_true",
        help="실제 HCX API를 정확히 한 번 호출합니다")
    args = parser.parse_args()
    if not args.live:
        parser.error("비용이 드는 호출은 --live를 명시해야 합니다")

    # CLI만 .env를 읽는다. override=False로 주입된 runtime env를 우선한다.
    load_dotenv(ROOT / ".env", override=False)
    compiled = compile_hcx_schema(SmokePayload)
    request = HcxRequest(
        messages=(
            HcxMessage(
                role="system",
                content=(
                    "Return only JSON matching the supplied schema. "
                    "Set transport_status to ok.")),
            HcxMessage(role="user", content="Perform a transport health check."),
        ),
        compiled_schema=compiled,
        config=HcxGenerationConfig(
            temperature=0.0, top_p=0.8, top_k=0,
            max_completion_tokens=32, repetition_penalty=1.1, seed=7),
        estimated_input_tokens=128,
    )
    with HcxStructuredClient.from_env() as client:
        result = client.generate_json(request)
    if result.payload.transport_status != "ok":
        raise RuntimeError("HCX smoke payload semantic check failed")
    print("PASS: HCX-007 Structured Output live smoke")
    print(f"request_id={result.request_id}")
    print(f"attempts={result.attempts} finish_reason={result.finish_reason}")
    print(
        f"latency_ms={result.total_latency_ms:.1f} "
        f"tokens={result.prompt_tokens}/{result.completion_tokens}/{result.total_tokens}")
    print(
        "rate_remaining="
        f"{result.rate_limit_remaining_requests}/{result.rate_limit_remaining_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

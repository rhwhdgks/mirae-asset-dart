"""strict wire 앞에 두는 **결정론적 안전 복구**.

지금까지는 HCX 응답이 strict Pydantic wire 를 통과하지 못하면 통째로 버려졌다.
정상적으로 추출된 task·fact·claim 이 다 들어 있어도 **파생 필드 하나가 어긋나면**
normalizer 에 닿지 못한다. 외부 독립 검토 §1 이 이것을 「프롬프트 문제가 아니라
계약 경계 문제」로 짚었고, 실제로 프롬프트를 다섯 판 고쳐도 이 층은 줄지 않았다
(`docs/IMPLEMENTATION.md`의 Stage1 실행 계약).

이 모듈은 **아무것도 발명하지 않고 아무것도 버리지 않는다.** 다른 필드에서
계산되는 값이 그 계산과 어긋날 때만 다시 계산한다. 그 외의 위반은 손대지 않고
strict 로 넘겨 그대로 거절되게 둔다.

## 지금 고치는 것 하나 — `speech_act`

계약(`agent/hcx_schema.py`)이 요구하는 것:

```
request       claim 0개
verification  claim 1개 이상 · requested output 0개
mixed         claim 1개 이상 · requested output 1개 이상
```

즉 `speech_act` 는 claim 수와 output 수로 **완전히 결정된다.** 독립적인 정보가
아니라 파생값이다. 모델이 claim 을 보존해 놓고 `request` 라고 적으면
`request_claim_forbidden` 으로 응답 전체가 버려지는데, 무엇을 뜻했는지는 claim 과
output 이 이미 말하고 있다.

그래서 다시 계산한다. **claim 도 output 도 건드리지 않는다** — 사용자가 실제로
말한 명제와 요구한 값은 모델이 판단한 그대로 남는다.

## 지금 고치지 않는 것과 그 이유

| 위반 | 왜 안 고치나 |
|---|---|
| `slot_reference_undeclared` | slot 을 만들면 **없는 내용을 발명**하고, ref 를 지우면 사용자가 요구한 답 항목을 **버린다** |
| `claim_verification_ref_missing` | claim 을 지우면 사용자가 말한 명제를 버린다 |
| `reason_disposition_mismatch` | 어느 쪽이 참인지 payload 만으로 정할 수 없다 |

이 셋은 결정론적 복구 대상이 아니다. 계약 코드만 되먹이는 1회 repair retry 가
`agent/providers/hcx007.py` 에 구현돼 있으나, **3차 외부 검토 후 「사용자 한 턴당
semantic 추출 1회」 정책과 충돌**한다는 판정을 받았다 — 같은 질문을 다시 보내면
한 질문의 의미가 두 생성 결과 사이에서 흔들린다. 정책 결정 전까지는 이 셋을
prompt·contract 결함으로 귀속하고 fail-closed 로 둔다.
"""

from __future__ import annotations

from typing import Any


#: 복구 이름. 측정에서 「무엇이 몇 건 고쳐졌나」를 세려고 남긴다.
SPEECH_ACT_RECOMPUTED = "speech_act_recomputed"

_VALID_ACTS = frozenset(("request", "verification", "mixed"))


def _derive_speech_act(claim_count: int, output_count: int) -> str:
    """claim 수와 output 수로 `speech_act` 를 계산한다."""

    if claim_count == 0:
        return "request"
    return "mixed" if output_count > 0 else "verification"


def repair_wire_payload(data: Any) -> tuple[Any, tuple[str, ...]]:
    """strict 검증 **직전에** 안전한 복구만 적용한다.

    ``(복구된 payload, 적용한 복구 이름들)`` 을 돌려준다. payload 가 dict 가
    아니거나 필요한 배열이 없으면 **손대지 않는다** — 구조가 어긋난 응답을
    추측으로 메우지 않는다.

    입력을 변형하지 않고 얕은 복사본을 만든다. 호출자가 원본을 진단에 쓴다.
    """

    if not isinstance(data, dict):
        return data, ()
    claims = data.get("claims")
    outputs = data.get("requested_outputs")
    current = data.get("speech_act")
    if not isinstance(claims, list) or not isinstance(outputs, list):
        return data, ()
    if current is not None and not isinstance(current, str):
        return data, ()

    wanted = _derive_speech_act(len(claims), len(outputs))
    if current == wanted:
        return data, ()
    # 모델이 계약에 없는 값을 적었으면 그것도 계산값으로 바꾼다 — 어차피 strict
    # 에서 literal 위반으로 죽고, 계산값은 claim·output 에서 나온 것이라 안전하다.
    if current is not None and current not in _VALID_ACTS and wanted not in _VALID_ACTS:
        return data, ()

    repaired = dict(data)
    repaired["speech_act"] = wanted
    return repaired, (SPEECH_ACT_RECOMPUTED,)

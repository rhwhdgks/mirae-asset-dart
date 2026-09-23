"""Stage1 조립이 공유하는 기준 상수.

- `CORPUS_CUTOFF` : 코퍼스 정보 cutoff(YYYYMMDD). 기본 `as_of` 이자 「그 뒤 문서는 없다」의 경계.
- `REFERENCE_DATE`: 「올해·작년」 같은 상대 날짜의 기준일.

실제 조립은 `agent.stage1_v1_backend_composition.build_stage1_v1_runtime`(resolver·compiler·역질문
세션·resume backend)과 `server/stage1.py:NativeStage1._assemble`(HCX client·prompt manifest·selector
preflight)이 한다. 서버·스크립트·테스트는 같은 상수를 여기서 가져와 「같은 조립」을 보장한다.
"""

from __future__ import annotations

from datetime import date

CORPUS_CUTOFF = "20260619"
REFERENCE_DATE = date(2026, 6, 19)

__all__ = ["CORPUS_CUTOFF", "REFERENCE_DATE"]

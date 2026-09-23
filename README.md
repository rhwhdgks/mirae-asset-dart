# DART 공시 Agent

공시의 숫자는 값만 맞아서는 부족합니다. **기간·연결 여부·단위·정정 시점**이 달라지면 같은 항목도 다른 답이 됩니다. 이 프로젝트는 질문에 맞는 공시를 고르고, 계산에 쓴 값과 원문 근거를 추적해 답하는 API입니다.

Team dis-001이 2026 미래에셋증권 AI Festival 공시 Agent 과제로 개발했습니다. 이 저장소는 면접 포트폴리오를 위해 분리한 공개 코드이며, 주최 측 공식 서비스나 대회 평가 저장소와 연결되어 있지 않습니다.

## 핵심 설계

```text
질문 → HCX 질문 구조화 → 코드가 조회 조건 확정 → 공시 정본·원문 근거 조회
     → Decimal 계산·근거 재검증 → 답변 문장 검증 → API 응답
```

LLM은 자연어 해석과 문장화를 맡습니다. 문서 선택, 단위·기간 처리, 계산, 근거 확인은 코드가 맡고, 생성된 문장에 검증되지 않은 숫자나 빠진 근거가 있으면 확정 답변으로 내보내지 않습니다.

| 문제 | 구현에서 선택한 방법 | 코드·테스트 |
| --- | --- | --- |
| 단위표가 다른 본문 표에 잘못 적용될 수 있음 | 단위의 적용 범위를 바로 다음 표로 제한 | [`dart_xml.py`](src/ingest/dart_xml.py), [`test_unit_binding.py`](tests/test_unit_binding.py) |
| 정정 공시를 단순히 최신 파일 하나로 대체하면 이력이 사라짐 | 정정 관계와 시점별 유효 관계를 분리하고, 모호한 후보는 임의 선택하지 않음 | [`lineage.py`](src/canonical/lineage.py) |
| 큰 표를 나누면 머리글·행 또는 원문 위치를 잃을 수 있음 | 머리글 반복, 행 보존, 결정적 content hash 검사 | [`chunker.py`](src/canonical/chunker.py), [`test_chunker.py`](tests/test_chunker.py) |
| 금액 계산·인용이 본문과 어긋날 수 있음 | `Decimal` 계산, 단위 조건 확인, 인용 직전 근거 재검증 | [`derivation.py`](app/tools/derivation.py), [`financial.py`](app/tools/financial.py), [`test_derivation_sum.py`](tests/test_derivation_sum.py) |
| 외부 모델이나 조회가 늦어질 수 있음 | 큐·실행 시간 상한과 준비 상태를 관리하고, 한계를 명시해 종료 | [`runtime.py`](server/runtime.py), [`app.py`](server/app.py) |

## API 기반 퀀트 개발과의 접점

이것은 **선물 매매 전략이나 주문 실행 시스템이 아닙니다.** 대신 전략의 입력이 될 금융 데이터를 다룰 때 중요한 원칙을 보여줍니다. 시점·단위가 맞는 값만 사용하고, 값의 출처와 계산 경로를 남기며, 외부 API 지연이나 근거 부족 때는 추정값을 정상 결과처럼 내보내지 않는 설계입니다.

## 공개본 검증과 한계

공개한 오프라인 테스트 중 `pytest` 24개를 통과했습니다. 표 분할 스크립트 테스트도 별도로 통과했습니다. 재현하려면 Python 환경에 개발 의존성을 설치한 뒤 다음을 실행합니다.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
PYTHONPATH=. python tests/test_chunker.py
```

주최 측 제공 원문, 파생 정본·검색 인덱스, 평가 질문·정답·근거, 운영 정보·로그, API 키는 공개하지 않았습니다. 따라서 이 저장소만으로 전체 답변 API나 대회 평가 결과를 재현할 수는 없습니다. 전체 서비스 실행에는 별도로 이용 권한을 가진 데이터와 HyperCLOVA X 자격 증명이 필요합니다. 비밀값은 `.env.example`을 참고해 개인 환경에만 설정하세요.

## 코드 위치

`agent/`는 질문 해석과 실행계획, `src/`는 공시 전처리와 정본 처리, `app/`은 근거 조회·계산·답변 조합, `server/`는 HTTP API를 담당합니다. `scripts/`의 일부 개발·평가 보조 코드는 공개본에서 제외한 자료를 필요로 합니다.

## 권리

Copyright © 2026 Team dis-001 contributors. All rights reserved. 이 공개본은 오픈소스 이용 허락을 부여하지 않습니다. 제3자 라이브러리, 공시 데이터, HyperCLOVA X에는 각각의 이용 조건이 적용됩니다.

# DART 공시 Agent

2026 미래에셋증권 AI Festival 공시 Agent 과제에 Team dis-001(2인)로 참가해 만든 전자공시 질의응답 API입니다. 질문에 맞는 공시를 찾아 답하고, 답에 사용한 값의 원문 위치를 함께 반환합니다. 주최 측 공식 서비스나 대회 평가 저장소와는 연결되어 있지 않습니다.

- 공시 원문 전처리·조회, HyperCLOVA X 질문 구조화: 고종한
- 검색·답변 생성, 평가 서버: 팀원
- 테스트와 수정: 함께

## 구조

```text
질문 → HCX 질문 구조화 → 코드가 조회 조건 확정 → 공시 정본·원문 근거 조회
     → Decimal 계산·근거 재검증 → 답변 문장 검증 → API 응답
```

LLM은 질문 해석과 문장 작성만 맡고, 문서 선택·단위와 기간 처리·계산·근거 확인은 코드가 맡습니다. 생성된 문장에 검증되지 않은 숫자나 빠진 근거가 있으면 확정 답변으로 내보내지 않습니다.

| 문제 | 처리 방법 | 코드·테스트 |
| --- | --- | --- |
| 단위표가 다른 본문 표에 잘못 적용될 수 있음 | 단위의 적용 범위를 바로 다음 표로 제한 | [`dart_xml.py`](src/ingest/dart_xml.py), [`test_unit_binding.py`](tests/test_unit_binding.py) |
| 정정 공시를 최신 파일 하나로 대체하면 이력이 사라짐 | 정정 관계와 시점별 유효 관계를 분리하고, 모호한 후보는 임의로 고르지 않음 | [`lineage.py`](src/canonical/lineage.py) |
| 큰 표를 나누면 머리글·행 또는 원문 위치를 잃을 수 있음 | 머리글 반복, 행 보존, content hash 검사 | [`chunker.py`](src/canonical/chunker.py), [`test_chunker.py`](tests/test_chunker.py) |
| 금액 계산·인용이 본문과 어긋날 수 있음 | `Decimal` 계산, 단위 조건 확인, 인용 직전 근거 재검증 | [`derivation.py`](app/tools/derivation.py), [`financial.py`](app/tools/financial.py), [`test_derivation_sum.py`](tests/test_derivation_sum.py) |
| 외부 모델이나 조회가 늦어질 수 있음 | 큐·실행 시간 상한과 준비 상태를 관리하고, 한계를 명시해 종료 | [`runtime.py`](server/runtime.py), [`app.py`](server/app.py) |

## 실행

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
PYTHONPATH=. python tests/test_chunker.py
```

대회 제공 원문, 정본·검색 인덱스, 평가 질문과 정답, API 키는 포함하지 않았습니다. 그래서 이 저장소만으로는 전체 API나 평가 결과를 재현할 수 없고, 위 명령으로 공개한 오프라인 테스트만 실행할 수 있습니다. 비밀값은 `.env.example`을 참고해 개인 환경에만 설정하세요.

## 코드 위치

- `agent/`: 질문 해석과 실행계획
- `src/`: 공시 전처리와 정본 처리
- `app/`: 근거 조회·계산·답변 조합
- `server/`: HTTP API
- `scripts/`: 개발·평가 보조 코드 (일부는 공개하지 않은 자료가 필요)

## 권리

Copyright © 2026 Team dis-001 contributors. All rights reserved. 이 공개본은 오픈소스 이용 허락을 부여하지 않습니다. 제3자 라이브러리, 공시 데이터, HyperCLOVA X에는 각각의 이용 조건이 적용됩니다.

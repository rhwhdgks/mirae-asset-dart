# 실호출 기록 보관

개인 저장소의 [성능 이슈 #12](https://github.com/rhwhdgks/mirae-asset-dart/issues/12)는 원본 dis-001 #219의 후속 작업입니다. 답변 수정에 필요한 호출을 할 때 기록을 함께 쌓고, 성능 개선 전후를 비교할 자료로 사용합니다. 이 기능 자체는 응답 속도를 개선하지 않습니다.

## 실행

이용 권한이 있는 정본·인덱스와 개인 환경의 API 키가 필요합니다. 서버는 기록 기능이 있는 코드로 새로 시작하세요. 데이터 재빌드는 필요하지 않습니다.

```bash
ANSWER_CAPTURE_LATENCY=1 ./scripts/serve.sh
```

영향 문항과 관련 대조 문항만 JSONL에 명시하고, 새 출력 경로를 사용합니다. 다음 예시는 3문항을 지정했을 때입니다.

```bash
.venv/bin/python scripts/run_live_question_set.py \
  --questions out/requests/my_run/questions.jsonl \
  --expected-count 3 \
  --output out/requests/my_run/responses.jsonl \
  --run-label issue5-affected-and-controls
```

각 입력 행은 `{"question_id":"고유ID","question":"질문 원문"}` 형식입니다. 문항을 순차 호출하며, 실패 자동 재호출은 기본 0회입니다. 이 도구는 승인되지 않은 문항을 추가하거나 70·535문항 회귀를 실행하지 않습니다. Gold70의 별도 승인·manifest 검증 경로를 대체하지도 않습니다.

## 남기는 파일

| 파일 | 내용 |
| --- | --- |
| `responses.jsonl` | 질문·응답 원문, HTTP 오류·전송 실패, 모든 시도의 UTC 시작/종료·소요시간·고유 ID |
| `responses.jsonl.manifest.json` | 실행 ID·설정, 질문 파일 SHA-256, 실행 도구의 Git 버전, 서버 시작 시 Git 버전·변경 여부, 정본·검색 인덱스·프롬프트 버전과 계측 상태 |
| `responses.jsonl.questions.jsonl` | 호출에 실제 사용한 질문 파일 사본 |
| `responses.jsonl.diagnostics.jsonl` | 정확한 요청 ID로 연결한 서버 원본 기록, Stage1·후속 파이프라인 시간, 기록된 토큰·재시도·캐시·구간별 이벤트 |
| `responses.jsonl.summary.json` | 완료/중단 상태, 기록 문항 수, HTTP 상태와 서버 기록 연결 상태 집계 |

출력과 보조 파일 중 하나라도 이미 있으면 덮어쓰지 않습니다. 파일 권한은 소유자 읽기·쓰기(`0600`)이며, 응답은 진단 조회 전에 파일로 flush·동기화합니다. 강제 종료로 summary가 비어 있으면 미완료 실행입니다. HTTP 200이나 실행 완료가 답변 정답을 의미하지는 않습니다.

기록은 `.gitignore`의 `out/` 아래 개인 환경에만 보관합니다. `.env`·API 키·인증 헤더는 수집하지 않으며, 응답·원공시·운영 로그를 공개 저장소나 이슈 본문에 자동 업로드하지 않습니다. 공개 이슈에는 비밀값을 확인한 요약만 올립니다.

## 시간을 읽는 기준

- `elapsed`는 마지막 HTTP 시도의 클라이언트 소요시간입니다. `attempts[].elapsed`로 최초 실패를 포함한 각 시도를 봅니다. `total_elapsed`는 재시도 대기를 포함한 해당 문항의 총시간입니다.
- 서버 `t_stage1`, `t_pipeline`, 요청 `elapsed`는 서로 다른 구간입니다. 로그 조회·파일 저장 시간은 클라이언트 답변 소요시간에 넣지 않습니다.
- `retrieval_validation`은 조회·근거 검증·계산을 합친 구간이고, `compose`는 모델 호출·검증·템플릿 폴백을 포함한 문장화 전체입니다. 순수 검색시간이나 순수 모델 대기시간으로 해석하지 않습니다.
- 현재 `provider_http` 이벤트는 Stage1 제공자 호출을 계측합니다. 문장화 모델의 개별 HTTP 대기, 정확한 큐 대기, 종료 후 백그라운드 작업 시간은 이 이벤트만으로 알 수 없습니다. 중첩된 이벤트를 합산해 전체 응답시간으로 삼지 않습니다.
- `cache_hit`가 기록되지 않았으면 캐시 미사용으로 단정하지 않습니다. 읽기 모델·OS 캐시의 첫 조회 여부도 Stage1 캐시 적중과 다릅니다. 캐시 초기화나 추가 호출을 자동 수행하지 않습니다.
- 서버와 실행 도구의 Git 버전은 따로 보관합니다. Git 정보가 없는 배포는 미확인으로 남깁니다. 실행 중 코드나 정본을 변경하지 마세요.

## 타임아웃 후 늦게 남은 로그

질문·시간이 비슷하다는 이유로 다른 요청을 연결하지 않습니다. HTTP 응답의 `X-Request-ID`와 시도별 `X-Live-Attempt-ID`로만 연결합니다. 서버 기록이 없거나 중복되면 `not_found`·`ambiguous`로 남기고 정상 결과처럼 대체하지 않습니다.

클라이언트 타임아웃 후 서버가 늦게 기록을 남겼거나 API tail 500건 밖으로 밀렸다면, 일일 서버 로그에서 진단 정보만 다시 수집할 수 있습니다. 다음 명령은 외부 모델을 호출하지 않고 기존 파일도 수정하지 않습니다. UTC 날짜가 바뀐 실행은 해당 날짜 로그를 `--server-log`로 각각 지정하세요.

```bash
.venv/bin/python -m scripts.live_run_evidence \
  --responses out/requests/my_run/responses.jsonl \
  --server-log out/requests/answers_20261014.jsonl \
  --output out/requests/my_run/diagnostics_recollected_01.jsonl
```

재수집 파일을 추가해도 최초 타임아웃·실패는 원래 응답 기록에 그대로 남습니다. 일일 서버 로그와 실행 묶음은 #12의 원인 분석이 끝날 때까지 함께 보관하세요.

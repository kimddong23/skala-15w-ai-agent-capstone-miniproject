# API

HTTP 엔드포인트와 SSE 이벤트를 모았다. 개요는 [README](../README.md)에 있다.

## API

사용자는 가명 `U01`..`U15`, 데이터 기간은 2016-03-29\~05-12다. 사용자 ID가 없으면 에이전트가 선택 가능한 목록을 안내한다. 손으로 쳐 볼 질문과 각 질문이 확인하는 분기는 [`docs/regression_test_queries.md`](regression_test_queries.md)에 있다.

백엔드 포트는 8100이다 (`BACKEND_PORT`).

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/health` | `{"status","supervisor_version","llm_backend","mcp","llm","ollama"}`. `llm`은 LLM 활성, Ollama 도달, 필수 모델 설치가 모두 참일 때만 true. `ollama`에 도달 여부와 설치·누락 모델 |
| GET | `/version` | `{supervisor_version, llm_backend, fast_model, quality_model, checkpointer, git_commit}`. 화면 헤더가 백엔드·모델을 보여 주는 용도 (비밀값 없음) |
| GET | `/users` | 사용자 목록 (기록 기간, 일수, 커버리지). `?dataset=fitbit\|stanford`로 거를 수 있다 |
| GET | `/users/{user_id}/events` | 연구 라벨 사건(증상 시작·진단·회복일). Stanford 사용자에게만 있고 FitBit 사용자는 빈 목록이다 (MCP 도구 `get_health_events`). 진단이 아니다 |
| GET | `/users/{user_id}/series?lookback_days=1&end=&resample=` | 차트용 다운샘플 시계열 (5분 이상 간격) |
| GET | `/users/{user_id}/summary?lookback_days=7&end=` | 요약 통계, 일별 안정시 심박 |
| GET | `/users/{user_id}/anomalies?lookback_days=7&end=` | 탐지된 이상 목록 |
| POST | `/chat` | 에이전트 실행 후 최종 결과 JSON |
| POST | `/chat/stream` | 같은 실행을 SSE 이벤트로 스트리밍 |
| POST | `/chat/approve` | 승인 대기(`approval_required`)인 thread에 `{thread_id, approved}`를 보내 그래프를 재개한다. 응답은 `notification`(`sent_simulated`\|`not_sent`)과 결과 한 줄이 붙은 리포트. 대기가 없으면 409 |
| POST | `/feedback` | `{run_id, score(0~1), comment?}`를 LangSmith에 `user_thumbs`로 남긴다. 키가 없으면 `recorded=false` |

```bash
curl -s localhost:8100/users | head -c 300
curl -s "localhost:8100/users/U05/anomalies?lookback_days=30"
curl -s -X POST localhost:8100/chat -H 'Content-Type: application/json' \
  -d '{"message":"U05 최근 30일 심박 이상 점검해줘","thread_id":"demo-1"}'
curl -s -N -X POST localhost:8100/chat/stream -H 'Content-Type: application/json' \
  -d '{"message":"U05 최근 30일 심박 이상 점검해줘"}'      # event: route_decided ... token ... done
```

요청 본문: `message`(1\~2000자), `user_id`(선택), `thread_id`(선택, 같은 값이면 대화 이어감). `/chat` 응답에는 `run_id`와 `usage`(호출·토큰)가 있고 승인 대기 중이면 `approval_required`가 붙는다. SSE 이벤트 14종의 스키마는 `app/agent/events.py`, `frontend/src/events.ts`.

# 개발과 품질

테스트, 보안, 품질 보증, 설계 결정, 개발 가이드를 모았다. 개요는 [README](../README.md)에 있다.

## 테스트

```bash
# 단위 + 통합 (LLM 없이 결정론적: fake LLM / NullLLM, 일부는 실제 MCP subprocess와 uvicorn)
LLM_ENABLED=false .venv/bin/python -m pytest -q
# 프론트 단위: SSE 파서, zod 스키마, 이벤트 리듀서
cd frontend && npm test
# 타입 검사 + 빌드
cd frontend && npm run build
# e2e (Playwright, 결정론 스위트)
npm --prefix e2e test
# e2e LLM 스모크 (옵트인, 실제 Ollama, 8100 포트가 비어 있어야 함)
npm --prefix e2e run test:llm
```

| 구분 | 도구 | 범위 | 통과 수 |
|---|---|---|---|
| 단위·통합 | pytest | store, 탐지기, 게이트·검증기·재시도, MCP 서버 2개, LangGraph 전체 흐름, HITL·다중 사용자·대화 기억, FastAPI SSE, API E2E(실제 uvicorn + MCP), 홀드아웃 라벨 검증 | 1408 통과 (실패 0) |
| 프론트 단위 | vitest | SSE 파서, zod, 리듀서, 인용 매칭, 실행 트레이스 | 60 통과 (실패 0) |
| E2E | Playwright (`e2e/`) | 대시보드, 에이전트 실행 기록, 보호자 알림 승인, 오류 경로, 키보드, 반응형, 인쇄 | 160 통과 (실패 0) |

위 숫자는 2026-10-20 최종 회귀의 전체 실행값이다(pytest 1408 / vitest 60 / Playwright 160, ruff F 0, gitleaks 0).

## 보안

비밀값(API 키, 토큰)이 저장소에 들어가지 않게 gitleaks로 막는다.

| 파일 | 역할 |
|---|---|
| `.gitleaks.toml` | 기본 규칙셋 사용. 공개 원천 데이터(`data/raw/`)와 `node_modules`, 빌드 산출물만 검사에서 제외 |
| `.pre-commit-config.yaml` | pre-commit 프레임워크용 훅 (gitleaks v8.30.1) |
| `scripts/install-hooks.sh` | 프레임워크 없이 로컬 gitleaks 바이너리로 `.git/hooks/pre-commit` 설치. 가짜 토큰으로 커밋이 차단되는 것을 확인했다 |
| `.env.example` | 환경변수 견본. 실제 `.env`는 `.gitignore` 대상이다 |

```bash
brew install gitleaks && scripts/install-hooks.sh   # 커밋 훅 설치
gitleaks git . --redact                             # 전체 커밋 이력 검사
gitleaks dir . --redact                             # 작업 트리 검사
```

2026-10-20 05:50 기준 커밋 이력과 작업 트리 검사에서 유출은 발견되지 않았다 (gitleaks 8.30.1).
API 키 같은 비밀값 말고도 알아 둘 보안 한계가 있다. 이 서버는 **인증·인가가 없다.** 누구나 모든 사용자의 데이터를 읽을 수 있고 `thread_id`를 아는 사람은 그 대화를 이어갈 수 있으므로 `127.0.0.1`로만 띄우고 외부에 노출하지 않는다.
LangSmith 키 같은 값은 `.env`에만 두고 코드, 문서, 평가 결과 파일에 쓰지 않는다.

### 컨테이너 하드닝

`docker compose`로 띄우는 구성에 아래 보안 설정을 넣었다 (`Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml`, `frontend/nginx.conf.template`).

| 항목 | 내용 |
|---|---|
| 이미지 고정 | 베이스 이미지(`python:3.11-slim`, `node:22-alpine`, `nginx-unprivileged:1.27-alpine`)를 태그와 digest(`@sha256:...`)로 함께 고정한다. 같은 태그가 나중에 다른 내용을 가리켜도 빌드가 바뀌지 않는다. digest가 빠지면 `tests/test_packaging.py`가 잡는다 |
| 실행 사용자가 코드 파일을 수정할 수 없음 | 백엔드 코드와 가공 데이터는 root 소유에 쓰기 금지(`go-w`)로 두고 실행 사용자(uid 10001, 로그인 셸 없음)는 바꿀 수 없다. 쓸 수 있는 곳은 `/app/state`(대화 기억 sqlite)와 `/app/data/processed/rag`(임베딩 캐시) 두 곳뿐이다 |
| 권한 최소화 | 두 컨테이너 모두 `cap_drop: ALL`, `no-new-privileges:true`. 프론트엔드는 비루트 nginx(`nginx-unprivileged`)다 |
| 포트 | 백엔드 8100, 프론트엔드 8600은 `127.0.0.1`에만 연다. 인증이 없는 서버라 LAN에 노출하지 않는다 |
| nginx | `server_tokens off`(버전 노출 차단)와 보안 헤더 조각을 적용하고 `/api`만 백엔드로 프록시한다 |
| Ollama | LAN에 노출하지 않는다. 호스트 Ollama를 `0.0.0.0`으로 열라고 권하지 않고, 컨테이너 Ollama(`--profile ollama`)도 호스트 포트를 열지 않는다 |
| 과금 확인 | OpenAI 백엔드는 `OPENAI_CONFIRM=yes`가 없으면 시작하지 않고, 스크립트가 y/N을 받은 뒤 그 프로세스에서만 켠다 (`.env`의 값은 무시) |

한계: 백엔드와 nginx의 루트 파일시스템은 읽기 전용으로 두지 않았다(nginx는 기동 때 설정 파일을 써야 하고 백엔드는 쓰기 가능한 경로가 둘 있다). 의존성 스캔(npm audit, pip-audit)은 0건이고 컨테이너 이미지 스캔(trivy 등)은 하지 않았다.

### 의존성 취약점 스캔

2026-10-20 12:25 (KST)에 알려진 취약점 데이터베이스로 의존성을 검사했다.

| 대상 | 명령 | 결과 |
|---|---|---|
| 프론트엔드 (운영 의존성) | `cd frontend && npm audit --omit=dev` | 취약점 0건 |
| E2E (Playwright) | `cd e2e && npm audit` | 취약점 0건 |
| 백엔드 (Python) | `uvx pip-audit -r requirements.txt` | No known vulnerabilities found |

범위: 직접·간접 의존성의 알려진 취약점만 본다. 컨테이너 이미지(OS 패키지 포함) 스캔은 하지 않았다. 새 패키지를 추가하거나 이미지 digest를 갱신할 때 다시 돌리는 것을 권한다.

## 품질 보증

개발자 자체 테스트와 별도로 세 가지 절차를 돌렸다.

| 절차 | 내용 | 결과 |
|---|---|---|
| 코드 점검 | 작성 과정과 분리해 `app/`과 `frontend/src`를 읽고 가능한 항목은 스크립트로 재현 | 19건 (HIGH 5, MEDIUM 7, LOW 7). HIGH 5건은 모두 수정 |
| 안전 문장 우회 시험 | 안심·진료 미루기 문장을 새로 만들어 이상 있는 리포트와 이상 없는 리포트의 모든 경로(스트리밍 포함)에 넣어 본다. 문구 사전 방식이 새 표현에 뚫려 허용 목록 방식으로 바꾼 뒤 반복했다 | 이전 시험 문장 73개는 모두, 새로 만든 HIGH 등급 시험 문장 26개 중 24개가 모든 경로와 스트리밍에서 차단됐다. 남은 2개는 이상이 없을 때의 평서문 안심 표현이며 고정 안내(`NO_ANOMALY_GUIDANCE`)가 모든 답변에 붙어 완화한다. 실제 탐지 데이터의 사실 문장은 10/10이 유지된다. 잔여는 한계에 적었다 |
| 회귀 시험 | API E2E, Playwright E2E, 탐색적 질의(프롬프트 인젝션 포함). 발견한 버그는 수정한 뒤 같은 시나리오로 다시 확인 | 낮은 심각도의 남은 관찰은 한계에 적었다 |
| 블라인드 홀드아웃 | 에이전트 코드를 보지 않고 만든 16건. 라벨은 데이터와 탐지기 출력에서 유도하고 채점 규칙은 실행 전 고정 | 첫 실행 14/16. 개발 평가셋이 거의 만점일 때 나온 수치다. 최종 재실행은 15/16(블라인드 아님) |

코드 점검 HIGH 5건은 에스컬레이션 플래그 유실, 탐지 실패의 거짓 "이상 없음", 진단 단정 규칙 우회, 조사가 붙은 사용자 ID 미인식, 후속 질문이 다른 사용자의 사실을 쓰는 문제다.
블라인드 평가는 한 번만 블라인드로 성립하므로 수정 뒤 재실행은 회귀 확인용으로만 쓴다.
확인하지 못한 항목: LLM 스트림 중 클라이언트 이탈이 Ollama의 생성 중단으로 이어지는지는 공유 Ollama 때문에 판정하지 못했다.

## 설계 결정

- **탐지는 결정론적 코드, 해석은 LLM**: 같은 입력이면 같은 결과라 재현·평가가 가능하고 LLM이 이상을 지어내지 못한다. 라우팅과 계획에도 LLM을 쓰지만 코드 게이트가 보정한다.
- **안전(권위) 게이트**: 소형 LLM에 "위험하면 119를 안내하라"고 시켜도 빠뜨릴 수 있어서 critical 안내는 코드가 확정한다. 에스컬레이션은 저장하지 않고 이상 목록에서 매번 다시 계산한다.
- **전용 Validator**: 근거 없는 수치, 진단·복약 단정, 단정적 안심을 규칙으로 잡는다. 실패하면 한 번 다시 쓰게 하고 그래도 실패하면 템플릿 리포트로 대체한다.
- **4계층 오류 처리**: 재시도, 도구 실패 격리, 규칙 폴백, 템플릿 리포트 순으로 내려가 LLM이 없어도 끝까지 동작한다.
- **로컬 LLM(Ollama)과 데이터 최소화**: 건강 데이터를 외부로 보내지 않기 위해 로컬 모델을 기본으로 쓰고 가명 ID와 집계 사실만 LLM에 전달한다.
- **가이드라인 의미 검색과 MCP 서버 분리**: 정적 안전 문서(Resource)와 검색(Tool)을 별도 서버로 나눠 한쪽이 죽어도 나머지가 동작한다.
- **보호자 알림은 사람 승인(HITL)**: 되돌릴 수 없는 외부 행동이라 `interrupt`로 멈추고 승인을 받는다. 안내문은 승인과 무관하게 이미 리포트에 있다.
- **여러 사용자 비교는 최대 3명**: 사용자별 사실을 나란히 설명하는 것이고 통계적 비교나 우열 판정이 아니다.
- **이상 리포트는 사실 서술만 허용(허용 목록)**: 금지 문구 사전은 새 표현에 계속 뚫려서 방향을 뒤집었다. 사실을 서술하는 문장만 남기고 조언·평가·안심은 코드가 쓴다.

## 개발 가이드

- **커밋 전에 테스트**: `.venv/bin/python -m pytest -q`와 `cd frontend && npm test && npm run build`가 통과해야 한다. 새 동작에는 LLM 없는 결정론적 테스트(`tests/fakes.py`의 fake, `LLM_ENABLED=false`)를 추가한다.
- **Python**: 3.11, 타입 힌트, 패키지는 `uv pip install --python .venv/bin/python <pkg>` 후 `requirements.txt`에 버전 고정. 린트는 ruff를 쓰며 별도 설정 파일은 없다. 주석과 docstring은 한국어로, 임계값에는 근거를 주석으로 남긴다.
- **구조 규율**: 탐지(`app/detection`)에 LLM을 넣지 않는다. 노드는 `ToolBox`/`LLM`을 생성자로 주입받는다. 숫자는 tool 결과에서만 인용하고 면책·안내문은 코드가 조립한다.
- **MCP Tool**: docstring에 인자 의미·단위·반환 필드·실패 형식(`{"error": ...}`)을 정확히 쓴다. 정적 문서는 Resource로 둔다.
- **SSE 이벤트 변경**: `app/agent/events.py`와 `frontend/src/events.ts`를 함께 고치고 `test_event_models_match_contract`를 통과시킨다.
- **커밋 메시지**: `영역: 변경 요약` 형식(예: `frontend: token.reset 처리`). 비밀값(`.env`, Kaggle 키)과 데이터(`data/raw`, parquet)는 커밋하지 않는다.

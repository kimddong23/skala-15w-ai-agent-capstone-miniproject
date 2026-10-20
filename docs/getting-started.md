# 시작하기와 설정

사전 준비, 운영체제별 실행, 스크립트, 수동 실행, 문제 해결, Docker와 환경변수 설정을 모았다. 개요는 [README](../README.md)에 있다.

## 주요 실행 방법

### 사전 준비

| 도구 | 버전 | 언제 필요한가 |
|---|---|---|
| Git | 아무 버전 | 저장소 받기 |
| Python | 3.11 (또는 [uv](https://docs.astral.sh/uv/)가 있으면 자동으로 받는다) | 백엔드 |
| Node.js | 22 이상 | 웹 화면 |
| Ollama | 최신 | 로컬 LLM을 쓸 때만. 없으면 `fake` 모드(규칙 기반 리포트)로 동작한다 |
| Docker Desktop 또는 Docker Engine + Compose v2 | 최신 | Docker로 한 번에 띄울 때만 |

```bash
git clone https://github.com/kimddong23/skala-15w-ai-agent-capstone-miniproject.git
cd skala-15w-ai-agent-capstone-miniproject
```

### 운영체제별 실행

| 운영체제 | 권장 방법 | 확인 범위 |
|---|---|---|
| macOS (Apple Silicon·Intel) | 아래 `sh/` 스크립트 | 새로 clone한 저장소에서 설치·테스트·기동 확인 |
| Linux (x86_64, arm64) | 아래 `sh/` 스크립트 | x86_64 컨테이너에서 clone부터 설치·테스트·기동 확인 |
| Windows 10/11 | Docker Desktop의 `docker compose up -d --build`, 또는 WSL2(Ubuntu)에서 Linux와 같은 명령 | 이미지는 amd64·arm64 공용이라 Docker 경로는 같다. Windows 네이티브 PowerShell 경로는 아래 명령만 정리했고 직접 실행해 보지는 않았다 |

Windows에서 WSL2나 Docker 없이 PowerShell로 바로 실행하려면 스크립트 대신 다음을 쓴다(`.venv/bin/python` 대신 `.venv\Scripts\python`).

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
$env:LLM_BACKEND = "fake"; $env:CHECKPOINTER = "sqlite"; $env:HITL_ENABLED = "true"
.venv\Scripts\python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8100
# 새 PowerShell 창에서
cd frontend; npm ci; npm run dev          # http://localhost:8600 (백엔드 8100으로 프록시)
```

로컬 LLM을 쓰려면 Ollama를 설치하고 `ollama pull qwen3:0.6b`, `ollama pull qwen3:8b`, `ollama pull bge-m3`를 받은 뒤 `$env:LLM_BACKEND = "ollama"`로 띄운다. 저장소의 `.gitattributes`가 줄바꿈을 LF로 고정하므로 Windows에서 받아도 셸 스크립트와 Docker 빌드가 깨지지 않는다.

**맨 처음 한 번만**: 저장소를 받았다면 `.venv`부터 만든다. 가공 데이터(`data/processed/`)는 저장소에 들어 있어서 Kaggle 계정 없이 바로 실행된다.

```bash
bash sh/setup_venv.sh                                                     # .venv 생성 + 의존성 설치
# (선택) 가공 데이터를 처음부터 다시 만들 때만. 자세한 절차는 data/README.md
kaggle datasets download -d arashnic/fitbit --unzip -p data/raw && .venv/bin/python -m app.data.prepare
```

| 순번 | 방법 | 명령 | 비용 | 용도 |
|---|---|---|---|---|
| 1 | Python 가상환경 준비 | `bash sh/setup_venv.sh` | 없음 | `.venv` 생성 + `requirements.txt` 설치 (최초 1회, 이미 있으면 설치만 재확인) |
| 2 | 로컬 Ollama 모델 준비 | `bash sh/setup_ollama.sh` | 없음 (로컬 추론, 모델 다운로드 수 GB) | `qwen3:0.6b`(fast), `qwen3:8b`(quality), `bge-m3`(임베딩)를 없는 것만 받는다. 실행 중인 Ollama 서버는 종료하지 않는다 |
| 3 | 데이터 준비 (선택) | `.venv/bin/python -m app.data.prepare` | 없음 (Kaggle 원본 588MB 필요) | 원본 CSV에서 가공 데이터와 평가 라벨을 다시 만든다. 저장소에 결과가 들어 있어 보통 건너뛴다 |
| 4 | 에이전트만 기동 (LLM 없음) | `bash sh/run_agent_fake.sh` | 없음 | Ollama 없이 규칙 기반 리포트로 동작. 처음 써 보기에 가장 쉽다 |
| 5 | 에이전트만 기동 (로컬) | `bash sh/run_agent_ollma.sh` | 없음 | 서버만 띄워 두고 `curl`이나 브라우저로 시험 (`http://127.0.0.1:8100`) |
| 6 | 에이전트만 기동 (OpenAI) | `bash sh/run_agent_openai.sh` | **과금** | 위와 같되 실제 API 사용 (y/N 확인, `.env`의 `OPENAI_API_KEY`) |
| 7 | 브라우저 채팅 화면 | `bash sh/run_frontend.sh` (4\~6번 중 하나를 먼저) | 백엔드에 따름 | `http://localhost:8600`에서 실제 UI로 사용 |
| 8 | 백엔드 없이 화면만 (mock) | `MOCK=1 bash sh/run_frontend.sh` | 없음 | 기록된 시나리오를 재생한다. 라이브가 아니다 |
| 9 | 한 번에 (Docker) | `docker compose up -d --build` | 없음 | 백엔드 + 프론트엔드. 접속 `http://localhost:8600`. Ollama는 호스트 것을 쓴다 |

- 모든 스크립트는 `.env`가 없으면 `cp .env.example .env` 후 값을 채우라고 안내한다. 기본값 그대로 동작한다.
- 스크립트는 **자기가 띄운 프로세스만** 종료한다. 포트가 이미 쓰이면 다른 프로세스를 죽이지 않고 알려 준 뒤 멈춘다. 포트는 `BACKEND_PORT`(기본 8100), `FRONTEND_PORT`(기본 8600)로 바꾼다.
- 백엔드 스크립트마다 LangSmith 프로젝트 이름이 다르다(`vital-sentinel-fake` 등). 켜 두면 백엔드별 트레이스가 섞이지 않는다.
- 세 백엔드 스크립트는 대화 기억을 sqlite 파일(`data/checkpoints.sqlite`)에 저장하므로 서버를 껐다 켜도 같은 `thread_id`로 이어진다. 끄려면 `CHECKPOINTER=memory bash sh/run_agent_fake.sh`처럼 준다. 아래 수동 실행의 기본값은 메모리다.
- OpenAI는 과금이라 확인 없이는 시작하지 않는다. 스크립트는 y/N을 묻고 `.env`나 셸의 `OPENAI_CONFIRM`은 일부러 무시한다. Docker로 한 번만 띄울 때는 `docker compose run --rm -e OPENAI_CONFIRM=yes -e LLM_BACKEND=openai --service-ports backend`처럼 그 프로세스에서만 준다 (자세한 내용은 사용자 가이드).

> 켜 둔 뒤에 무엇을 물어볼 수 있는지는 [사용자 가이드](manual_ollma.md), 손으로 쳐 볼 질문 목록은 [회귀 테스트 쿼리](regression_test_queries.md).

### 수동 실행 (스크립트 없이)

스크립트가 하는 일을 직접 하는 방법이다. 포트는 스크립트와 같게 백엔드 8100, 프론트엔드 8600이다.

```bash
uv venv --python 3.11 && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/uvicorn app.api.main:app --host 127.0.0.1 --port 8100        # 백엔드 (LLM_BACKEND 기본 ollama)
LLM_BACKEND=fake .venv/bin/uvicorn app.api.main:app --port 8100         # LLM 없이
cd frontend && npm install && npm run dev                                # 화면 http://localhost:8600 (/api 프록시 대상 8100)
cd frontend && npm run dev:mock                                          # 백엔드 없이 mock 재생
```

백엔드가 떠 있는지는 `curl -s localhost:8100/health`로 본다. 질문 보내기는 [API](api.md#api) 절을 본다.

### 실행 문제 해결

| 증상 | 원인과 해결 |
|---|---|
| `.venv 가 없습니다` | 1번을 먼저: `bash sh/setup_venv.sh` |
| `모델 … 설치되어 있지 않습니다` | 2번: `bash sh/setup_ollama.sh` (또는 LLM 없이 `run_agent_fake.sh`) |
| 화면에 "LLM 미연결 — 기본 리포트 모드" | 정상 동작이다(규칙 기반 리포트). 원인은 화면에 한 줄로 표시된다 |
| `포트 8100 … 이미 쓰고 있습니다` | 다른 프로그램이 쓰는 중이다. `BACKEND_PORT=8110`으로 바꿔 실행하고 프론트도 같은 값으로 `BACKEND_PORT=8110 bash sh/run_frontend.sh` |
| 프론트가 `백엔드가 응답하지 않습니다` | 백엔드(4\~6번)를 먼저 켠다. 화면만 보려면 `MOCK=1 bash sh/run_frontend.sh` |
| Docker에서 LLM이 안 붙는다 | 호스트 Ollama(`host.docker.internal:11434`)를 쓴다. 없으면 기본 리포트 모드로 뜬다. 컨테이너 Ollama: `DOCKER_OLLAMA_URL=http://ollama:11434 docker compose --profile ollama up -d --build` (GPU는 `docker-compose.gpu.yml`을 함께 지정) |
| 가공 데이터가 없다고 한다 | `data/processed/`는 저장소에 있다. 지워졌다면 `data/README.md`의 절차로 다시 만든다 |

데이터를 `curl`로 직접 확인하려면 다음처럼 한다.

```bash
curl -s localhost:8100/health
curl -sN -X POST localhost:8100/chat/stream -H 'Content-Type: application/json' \
  -d '{"message":"어젯밤 수면 중 심박 괜찮아?","user_id":"U05"}'      # 진행 과정이 SSE로 흐른다
```

같은 대화를 이어 가려면 응답의 `thread_id`를 다음 요청에 넣는다. LangSmith 추적은 `.env`에서 `LANGCHAIN_TRACING_V2=true`와 키를 넣으면 켜진다(질문·집계 요약이 외부로 전송된다).

## 설정

환경변수는 `app/config.py`에서 읽는다. `.env.example`을 복사해 `.env`로 쓰면 스크립트가 읽는다.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `DATA_DIR` / `RAW_DIR` | `data/processed` / `data/raw` | 전처리 결과 / 원본 CSV 위치 |
| `LLM_BACKEND` | `ollama` | `ollama`(로컬) \| `fake`(LLM 없음, 결정론) \| `openai`(과금). `sh/run_agent_*.sh`가 정해 준다 |
| `OPENAI_API_KEY` / `OPENAI_FAST_MODEL` / `OPENAI_QUALITY_MODEL` | 빈 값 / `gpt-4o-mini` / `gpt-4o` | OpenAI 백엔드용. 키는 `.env`(git 제외)에만 둔다 |
| `SUPERVISOR_VERSION` | `v2` | `v1`은 Before/After 비교용 선형 기준선 (실사용 금지) |
| `EMBED_MODEL` / `EMBED_TIMEOUT_S` / `EMBED_BUILD_TIMEOUT_S` / `EMBED_ENABLED` | `bge-m3:latest` / `3` / `60` / `true` | RAG 임베딩 모델(Ollama) / 질의 임베딩 호출 상한(초) / 인덱스 구축용 호출 상한(초) / `false`면 임베딩을 부르지 않고 키워드 검색만. 초과·연결 실패는 키워드 검색으로 폴백 |
| `TOOL_TIMEOUT_S` | `20` | MCP 도구 호출 1회 상한 (초과는 일시 오류로 재시도) |
| `BACKEND_PORT` / `FRONTEND_PORT` | `8100` / `8600` | 스크립트가 쓰는 포트 |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 주소 (죽은 포트로 바꾸면 LLM 장애 폴백 시연) |
| `OLLAMA_FAST_MODEL` / `OLLAMA_QUALITY_MODEL` | `qwen3:0.6b` / `qwen3:8b` | 라우팅·계획·요약용 / 해석·서술용. 이전 이름 `FAST_MODEL`, `QUALITY_MODEL`도 별칭으로 읽는다. 이전 구성은 `OLLAMA_FAST_MODEL=qwen2.5:3b ANALYSIS_TIER=fast` |
| `LLM_ENABLED` | `true` | `false`면 LLM 호출 없이 폴백 경로만 사용 (테스트 기본) |
| `LLM_TIMEOUT_S` | `120` | 스트리밍 LLM 호출 전체 타임아웃 (structured는 30초 상한) |
| `LLM_KEEP_ALIVE` | `30m` | Ollama 모델 상주 시간 |
| `ANALYSIS_TIER` | `quality` | analysis 노드 모델 티어 (`fast`\|`quality`). fast `qwen3:0.6b`가 사실 목록을 복사하는 데 그쳐 기본을 quality(8b)로 올렸다 (모델 벤치마크 #1). `fast`는 더 빠르지만 해석이 얕다 |
| `MERGE_ANALYSIS` | `false` | true면 analysis의 LLM 호출 생략, writing이 해석을 겸함 |
| `COMPACT_FACTS` | `true` | LLM에 JSON 대신 압축 텍스트 전달 |
| `NUM_PREDICT_ANALYSIS` / `NUM_PREDICT_WRITING` | `220` / `700` | 생성 토큰 상한 (0 = 제한 없음) |
| `HITL_ENABLED` | `false` | true면 에스컬레이션 때 보호자 알림(시뮬레이션) 발송 전에 사람 승인을 기다린다 (체크포인터 필요. 설정 기본값은 꺼짐이고 제공 스크립트·compose는 켠다) |
| `FAULT_INJECT_TRANSIENT` | `false` | true면 프로세스의 첫 도구 호출을 일시 오류로 실패시켜 재시도·오류 격리를 시연한다 (retry 횟수 지표용) |
| `SUMMARIZE_AFTER_MESSAGES` / `KEEP_RECENT_MESSAGES` | `10` / `6` | 미요약 메시지가 이 수를 넘으면 오래된 대화를 fast LLM으로 요약해 보관 / 요약 뒤에도 원문으로 넣는 최근 메시지 수 |
| `EXTERNAL_DIR` | `data/external` | 외부 연구 데이터셋(Stanford) 폴더. 있으면 별도 데이터셋으로 읽고 없으면 무시한다 (커밋 금지) |
| `ALLOW_V1_BASELINE` | (없음) | `yes`로 둬야 `SUPERVISOR_VERSION=v1`이 기동한다. v1은 안전 게이트·검증이 없는 비교용 기준선이라 명시적 허용이 필요하다 |
| `CHECKPOINTER` | `memory` | `memory`\|`sqlite`\|`postgres` (postgres는 미검증). `sh/run_agent_*.sh`는 sqlite로 켠다 |
| `SQLITE_PATH` / `DATABASE_URL` | `data/checkpoints.sqlite` / 빈 값 | sqlite 파일 / postgres 접속 문자열 |
| `MAX_REPLAN_ROUNDS` | `1` | 데이터 부족 시 조회 기간 확대 최대 횟수 |
| `MAX_VALIDATION_RETRIES` | `1` | 검증 실패 시 리포트 재작성 횟수 |
| `VALIDATOR_JUDGE` | `false` | 규칙 통과 후 fast LLM judge 추가 (소형 judge는 오탐이 잦아 기본 off) |
| `GRAPH_RECURSION_LIMIT` | `40` | 그래프 무한 루프 가드 |
| `CORS_ORIGINS` | `localhost:8600,127.0.0.1:8600,localhost:5173,127.0.0.1:5173` | 허용 오리진 (쉼표 구분) |
| `VITE_API_BASE` / `VITE_MOCK` | `/api` / 없음 | 프론트: API 주소 직접 지정 / mock 재생 |
| `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` | 꺼짐 | 설정하면 LangSmith 트레이싱. 켜면 프롬프트·응답이 LangSmith로 전송됨 |

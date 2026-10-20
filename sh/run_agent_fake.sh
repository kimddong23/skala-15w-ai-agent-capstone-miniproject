#!/usr/bin/env bash
#
# Vital Sentinel — LLM 없이(기본 리포트 모드) 에이전트를 "기동만" 해두는 스크립트.
#
# LLM_BACKEND=fake: Ollama·OpenAI 를 전혀 부르지 않는다. 라우팅은 키워드, 계획은 규칙, 리포트는 템플릿으로 만들며
# 탐지(결정론 코드)·안전 게이트·검증은 그대로 동작한다. 설치가 하나도 없는 발표장/CI/첫 실행에서 가장 안전한 방법이다.
# LLM_ENABLED=true 여야 fake 백엔드가 쓰인다(false 면 백엔드 선택과 무관하게 템플릿 모드로 떨어진다). fake 는 네트워크를 쓰지 않는다.
#
# 실행:  bash sh/run_agent_fake.sh            (포트 변경: BACKEND_PORT=8110 bash sh/run_agent_fake.sh)
# 종료:  Ctrl+C
#
set -euo pipefail
# shellcheck source=sh/_common.sh
source "$(dirname "$0")/_common.sh"
cd "$ROOT_DIR"
use_venv
load_env optional

BACKEND_PORT="${BACKEND_PORT:-8100}"; validate_port BACKEND_PORT "$BACKEND_PORT"
need_free_port BACKEND_PORT "$BACKEND_PORT"

export LLM_BACKEND=fake LLM_ENABLED=true
use_sqlite_checkpointer
export HITL_ENABLED="${HITL_ENABLED:-true}"   # 시연 기본: 보호자 알림 승인 카드. 끄려면 HITL_ENABLED=false
export LANGCHAIN_PROJECT="vital-sentinel-fake"   # ollama/openai 와 LangSmith 프로젝트를 분리
echo "모드: fake (LLM 호출 없음) / LangSmith project=$LANGCHAIN_PROJECT tracing=${LANGCHAIN_TRACING_V2:-false}"

trap 'echo; echo "에이전트를 종료합니다..."; cleanup_pids' EXIT
start_backend
print_ready "LLM_BACKEND=fake, 기본 리포트 모드" "없음 (LLM 호출 없이 규칙 기반 리포트)"
wait

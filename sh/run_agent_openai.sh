#!/usr/bin/env bash
#
# Vital Sentinel — OpenAI API 로 백엔드만 켜 두는 스크립트.
#
# ⚠️ 과금되는 원격 호출이다. 로컬 추론(run_agent_ollma.sh)과 달리 질문 한 번마다 OpenAI 계정에 비용이 청구된다.
# 시작 전에 y/N 으로 한 번 더 묻고, 이 질문을 건너뛰는 옵션은 없다.
# 건강 데이터 원본은 LLM 에 보내지 않고 집계 사실만 보내지만, 질문·집계 요약은 OpenAI 로 전송된다.
#
# 준비: .env 에 OPENAI_API_KEY 를 채운다(cp .env.example .env).  모델: OPENAI_FAST_MODEL / OPENAI_QUALITY_MODEL.
# 끄기: Ctrl+C
#
set -euo pipefail
# shellcheck source=sh/_common.sh
source "$(dirname "$0")/_common.sh"
cd "$ROOT_DIR"
use_venv
load_env required

# 과금 확인(OPENAI_CONFIRM)은 이 스크립트의 y/N 로만 정한다. .env 나 셸에 미리 적혀 있으면 y/N 을 건너뛰는 통로가 되므로
# 무시하고 지운다(앱도 OPENAI_CONFIRM=yes 가 없으면 openai 백엔드를 시작하지 않는다).
if [ -n "${OPENAI_CONFIRM:-}" ]; then
    echo "경고: OPENAI_CONFIRM 이 .env 나 셸에 설정되어 있어 무시합니다. 확인은 아래 y/N 로만 받습니다." >&2
    unset OPENAI_CONFIRM
fi

if [ -z "${OPENAI_API_KEY:-}" ] || [ "$OPENAI_API_KEY" = "sk-..." ]; then
    die "OPENAI_API_KEY 가 비어 있거나 예시 값입니다 — .env 의 OPENAI_API_KEY 에 본인 키를 넣으세요."
fi

BACKEND_PORT="${BACKEND_PORT:-8100}"; validate_port BACKEND_PORT "$BACKEND_PORT"
need_free_port BACKEND_PORT "$BACKEND_PORT"

OPENAI_FAST_MODEL="${OPENAI_FAST_MODEL:-gpt-4o-mini}"
OPENAI_QUALITY_MODEL="${OPENAI_QUALITY_MODEL:-gpt-4o}"
[ -n "$OPENAI_FAST_MODEL" ] && [ -n "$OPENAI_QUALITY_MODEL" ] || die "모델 이름이 비어 있습니다."
export OPENAI_FAST_MODEL OPENAI_QUALITY_MODEL
export LLM_BACKEND=openai LLM_ENABLED=true
use_sqlite_checkpointer
export HITL_ENABLED="${HITL_ENABLED:-true}"   # 시연 기본: 보호자 알림 승인 카드. 끄려면 HITL_ENABLED=false
export LANGCHAIN_PROJECT="vital-sentinel-openai"
echo "모드: openai / fast=$OPENAI_FAST_MODEL quality=$OPENAI_QUALITY_MODEL / LangSmith project=$LANGCHAIN_PROJECT tracing=${LANGCHAIN_TRACING_V2:-false}"
echo
echo "⚠️  이 서버로 질문을 보낼 때마다 OpenAI 계정에 비용이 청구됩니다."
echo "   키가 본인 계정 것인지, 결제 수단이 연결되어 있는지 확인하세요."
read -r -p "진행할까요? [y/N] " CONFIRM
case "$CONFIRM" in
    [yY]|[yY][eE][sS]) ;;
    *) echo "취소했습니다."; exit 1 ;;
esac
# "y" 를 받은 뒤에만, 이 프로세스(와 자식인 백엔드)에서만 켠다. .env 에는 절대 쓰지 않는다.
export OPENAI_CONFIRM=yes

trap 'echo; echo "에이전트를 종료합니다..."; cleanup_pids' EXIT
start_backend
print_ready "LLM_BACKEND=openai ⚠️ 아래 요청은 전부 과금됩니다" "fast=$OPENAI_FAST_MODEL / quality=$OPENAI_QUALITY_MODEL (OpenAI API)"
wait

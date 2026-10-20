#!/usr/bin/env bash
#
# Vital Sentinel — 로컬 Ollama 모델로 백엔드만 켜 두는 스크립트. (파일명 철자는 실습 자료의 관례를 따른다: ollma)
#
# 켜 둔 채로 curl 이나 sh/run_frontend.sh(브라우저)로 직접 시험한다.
# 끄려면 Ctrl+C. Ollama 서버는 이 스크립트가 띄웠을 때만 함께 종료하고, 원래 떠 있던 서버는 그대로 둔다.
#
# 모델: 기본값은 app/config.py (바꾸려면 OLLAMA_FAST_MODEL / OLLAMA_QUALITY_MODEL).
#       모델이 없으면 먼저 bash sh/setup_ollama.sh. 웜업을 건너뛰려면 SKIP_WARMUP=1.
#
set -euo pipefail
# shellcheck source=sh/_common.sh
source "$(dirname "$0")/_common.sh"
cd "$ROOT_DIR"
use_venv
load_env optional
require_cmd ollama "https://ollama.com 에서 설치하세요. (설치 없이 써보려면: bash sh/run_agent_fake.sh)"

BACKEND_PORT="${BACKEND_PORT:-8100}"; validate_port BACKEND_PORT "$BACKEND_PORT"
need_free_port BACKEND_PORT "$BACKEND_PORT"

FAST_MODEL="$(pick_model "${OLLAMA_FAST_MODEL:-}" "${FAST_MODEL:-}" FAST_MODEL)"
QUALITY_MODEL="$(pick_model "${OLLAMA_QUALITY_MODEL:-}" "${QUALITY_MODEL:-}" QUALITY_MODEL)"
OLLAMA_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
export FAST_MODEL QUALITY_MODEL OLLAMA_BASE_URL="$OLLAMA_URL"
export LLM_BACKEND=ollama LLM_ENABLED=true
use_sqlite_checkpointer
export HITL_ENABLED="${HITL_ENABLED:-true}"   # 시연 기본: 보호자 알림 승인 카드. 끄려면 HITL_ENABLED=false
export LANGCHAIN_PROJECT="vital-sentinel-ollama"
echo "모드: ollama / fast=$FAST_MODEL quality=$QUALITY_MODEL / LangSmith project=$LANGCHAIN_PROJECT tracing=${LANGCHAIN_TRACING_V2:-false}"

trap 'echo; echo "에이전트를 종료합니다..."; cleanup_pids' EXIT

# ── 1) Ollama 서버 확인 (없으면 이 스크립트가 띄우고, 그때만 함께 종료) ──────
if ! curl -s --max-time 2 "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
    case "$OLLAMA_URL" in
        http://127.0.0.1:*|http://localhost:*) ;;
        *) die "$OLLAMA_URL 에 연결되지 않습니다. 원격 Ollama 는 직접 띄우지 않습니다." ;;
    esac
    echo "Ollama 서버가 꺼져 있어 이 스크립트가 띄웁니다..."
    nohup ollama serve > "${TMPDIR:-/tmp}/vital_sentinel_ollama.log" 2>&1 &
    track_pid "$!"
    wait_http "$OLLAMA_URL/api/tags" 60 || die "ollama serve 기동 실패 (로그: ${TMPDIR:-/tmp}/vital_sentinel_ollama.log)"
fi

# 모델 설치 확인 (태그가 없는 이름은 :latest 로 표시된다)
for m in "$FAST_MODEL" "$QUALITY_MODEL"; do
    case "$m" in *:*) full="$m" ;; *) full="$m:latest" ;; esac
    ollama list | awk 'NR>1 {print $1}' | grep -Fx -- "$full" >/dev/null \
        || die "모델 $m 이(가) 설치되어 있지 않습니다 — bash sh/setup_ollama.sh 후 다시 실행하세요."
done

if [ "${SKIP_WARMUP:-0}" != "1" ]; then
    echo "모델을 메모리에 미리 올립니다(처음 한 번은 오래 걸릴 수 있음)..."
    for m in "$FAST_MODEL" "$QUALITY_MODEL"; do
        curl -s --max-time 180 "$OLLAMA_URL/api/generate" \
            -d "{\"model\":\"$m\",\"prompt\":\"hi\",\"stream\":false,\"options\":{\"num_predict\":1}}" >/dev/null || echo "  (미리 올리기 실패: $m — 첫 질문이 느릴 수 있습니다)"
    done
fi

# ── 2) 백엔드 기동 ──────────────────────────────────────────────────────
start_backend
print_ready "LLM_BACKEND=ollama" "fast=$FAST_MODEL / quality=$QUALITY_MODEL (Ollama $OLLAMA_URL)"
wait

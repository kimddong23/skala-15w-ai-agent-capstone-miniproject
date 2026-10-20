#!/usr/bin/env bash
#
# Vital Sentinel — 로컬 Ollama 모델 준비. 필요한 모델이 이미 있으면 그대로 재사용하고, 없는 것만 받는다.
#   모델: fast (라우팅·계획·분석), quality (리포트 작성), embed=bge-m3 (가이드라인 검색). 기본값은 app/config.py 가 정한다.
#   실습 표준 모델로 맞추려면 OLLAMA_FAST_MODEL=qwen3:0.6b OLLAMA_QUALITY_MODEL=qwen3:8b.
#
# ⚠️ 이 스크립트는 실행 중인 ollama 서버를 종료하지 않는다(다른 작업이 같은 서버를 쓰고 있을 수 있다).
#    서버가 꺼져 있으면 임시로 띄우고, 그때 띄운 것만 끝에 정리한다.
# ⚠️ 모델 삭제는 OLLAMA_FORCE_REFRESH=1 로 명시했을 때만, 위 세 모델에 한해서만 한다. 다른 모델은 건드리지 않는다.
# ⚠️ 새 모델은 네트워크·디스크·시간을 쓴다(qwen3:8b ≈ 5GB).
#
# 실행:
#   bash sh/setup_ollama.sh                                  # 설치된 모델은 두고, 없는 모델만 받는다
#   OLLAMA_DRY_RUN=1 bash sh/setup_ollama.sh                 # 무엇을 받을지만 출력(받지 않음)
#   OLLAMA_FORCE_REFRESH=1 bash sh/setup_ollama.sh           # 위 세 모델만 지우고 다시 받는다
#   OLLAMA_FAST_MODEL=qwen3:0.6b bash sh/setup_ollama.sh     # fast 모델을 다른 것으로(base 와 같은 변수 이름)
#
set -euo pipefail
# shellcheck source=sh/_common.sh
source "$(dirname "$0")/_common.sh"
cd "$ROOT_DIR"

load_env_file   # .env 가 있으면 FAST_MODEL 등을 읽는다(없으면 만들지 않는다)
# 기본 모델은 app/config.py 의 기본값을 따른다(실습 표준 qwen3:0.6b/qwen3:8b 로 바꾸려면 OLLAMA_FAST_MODEL=qwen3:0.6b).
FAST_MODEL="$(pick_model "${OLLAMA_FAST_MODEL:-}" "${FAST_MODEL:-}" FAST_MODEL)"
QUALITY_MODEL="$(pick_model "${OLLAMA_QUALITY_MODEL:-}" "${QUALITY_MODEL:-}" QUALITY_MODEL)"
EMBED_MODEL="$(pick_model "${OLLAMA_EMBED_MODEL:-}" "${EMBED_MODEL:-}" EMBED_MODEL)"
FORCE_REFRESH="${OLLAMA_FORCE_REFRESH:-0}"
DRY_RUN="${OLLAMA_DRY_RUN:-0}"
BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"

for v in FAST_MODEL QUALITY_MODEL EMBED_MODEL; do
    [ -n "${!v}" ] || die "$v 가 비어 있습니다."
done
case "$FORCE_REFRESH" in 0|1) ;; *) die "OLLAMA_FORCE_REFRESH 는 0 또는 1 이어야 합니다: '$FORCE_REFRESH'" ;; esac
MODELS=("$FAST_MODEL" "$QUALITY_MODEL" "$EMBED_MODEL")

require_cmd ollama "https://ollama.com 에서 설치 후 재실행하세요."
require_cmd curl "curl 을 설치하세요."

# 태그가 없는 이름은 ollama list 에서 :latest 로 보인다.
full_name() { case "$1" in *:*) echo "$1" ;; *) echo "$1:latest" ;; esac; }
has_model() { ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fx -- "$(full_name "$1")" >/dev/null; }

echo "준비할 모델: fast=$FAST_MODEL / quality=$QUALITY_MODEL / embed=$EMBED_MODEL"

# ── 1) ollama 서버 확인 (있으면 그대로 사용, 없으면 임시 기동) ───────────
STARTED=""
if curl -s --max-time 2 "$BASE_URL/api/tags" >/dev/null 2>&1; then
    echo "떠 있는 Ollama 서버를 그대로 씁니다(종료하지 않음): $BASE_URL"
elif [ "$DRY_RUN" = "1" ]; then
    echo "(dry-run) ollama 서버가 꺼져 있어 모델 목록을 확인할 수 없습니다 — 실제 실행 때 임시로 기동합니다."
else
    case "$BASE_URL" in
        http://127.0.0.1:*|http://localhost:*) ;;
        *) die "$BASE_URL 에 연결되지 않습니다. 원격 서버는 직접 띄우지 않습니다." ;;
    esac
    echo "ollama 서버가 없어 임시로 기동합니다..."
    nohup ollama serve > "${TMPDIR:-/tmp}/vital_sentinel_ollama_setup.log" 2>&1 &
    STARTED=$!
    track_pid "$STARTED"
    trap 'echo; echo "임시로 띄운 ollama 서버를 종료합니다..."; cleanup_pids' EXIT
    wait_http "$BASE_URL/api/tags" 60 || die "ollama 서버 기동 실패 (로그: ${TMPDIR:-/tmp}/vital_sentinel_ollama_setup.log)"
fi

# ── 2) 기존 모델 삭제 (OLLAMA_FORCE_REFRESH=1 일 때만, 위 세 모델만) ─────
if [ "$FORCE_REFRESH" = "1" ]; then
    for m in "${MODELS[@]}"; do
        if [ "$DRY_RUN" = "1" ]; then
            echo "(dry-run) 삭제 예정(OLLAMA_FORCE_REFRESH=1): $m"
        elif has_model "$m"; then
            echo "모델 삭제(OLLAMA_FORCE_REFRESH=1): $m"
            ollama rm "$m"
        fi
    done
fi

# ── 3) 없는 모델만 받기 ─────────────────────────────────────────────────
for m in "${MODELS[@]}"; do
    if [ "$FORCE_REFRESH" != "1" ] && has_model "$m"; then
        echo "이미 설치됨, 그대로 사용: $m"
        continue
    fi
    if [ "$DRY_RUN" = "1" ]; then
        echo "(dry-run) 받을 예정: $m"
        continue
    fi
    echo
    echo "다운로드 중: $m (크기에 따라 수 분 걸릴 수 있습니다)"
    ollama pull "$m"
done

cat <<EOF2

################################################################
 Ollama 환경 준비 $( [ "$DRY_RUN" = "1" ] && echo "점검(dry-run) 끝" || echo "완료" ): $FAST_MODEL / $QUALITY_MODEL / $EMBED_MODEL
################################################################

다음: bash sh/run_agent_ollma.sh
EOF2

#!/usr/bin/env bash
#
# Vital Sentinel — React 프론트엔드(Vite 빌드 결과)를 띄우는 스크립트.
#
# frontend/ 를 빌드(tsc + vite build)한 뒤 `vite preview` 로 서빙한다. 브라우저의 /api 요청은 백엔드로 프록시된다.
#   주소:   http://localhost:8600        (바꾸려면 FRONTEND_PORT)
#   백엔드: http://127.0.0.1:8100        (바꾸려면 BACKEND_PORT — sh/run_agent_*.sh 에서 쓴 값과 같게)
#
# ⚠️ 먼저 다른 터미널에서 sh/run_agent_fake.sh / run_agent_ollma.sh / run_agent_openai.sh 로 백엔드를 띄워둬야 한다.
#    백엔드 없이 화면만 보고 싶으면 목업 모드: MOCK=1 bash sh/run_frontend.sh  (서버·LLM 없이 기록된 대화를 재생)
#
# 끄기: Ctrl+C
#
set -euo pipefail
# shellcheck source=sh/_common.sh
source "$(dirname "$0")/_common.sh"
cd "$ROOT_DIR/frontend"
require_cmd node "Node 22 이상을 설치하세요 (https://nodejs.org)."
require_cmd npm "Node.js 와 함께 설치됩니다."

FRONTEND_PORT="${FRONTEND_PORT:-8600}"; validate_port FRONTEND_PORT "$FRONTEND_PORT"
BACKEND_PORT="${BACKEND_PORT:-8100}"; validate_port BACKEND_PORT "$BACKEND_PORT"
MOCK="${MOCK:-0}"
case "$MOCK" in 0|1) ;; *) die "MOCK 은 0 또는 1 이어야 합니다: '$MOCK'" ;; esac

# ── 0) 백엔드가 떠 있는지 확인 (목업 모드는 건너뜀) ─────────────────────
if [ "$MOCK" != "1" ] && ! curl -s --max-time 2 "http://127.0.0.1:$BACKEND_PORT/health" >/dev/null 2>&1; then
    die "백엔드(http://127.0.0.1:$BACKEND_PORT)가 응답하지 않습니다. 프론트는 백엔드가 $BACKEND_PORT 에 떠 있어야 동작합니다.
  백엔드를 먼저 켜세요(다른 터미널):
    bash sh/run_agent_fake.sh     (LLM 없이, 설치 불필요)
    bash sh/run_agent_ollma.sh    (로컬 Ollama)
    bash sh/run_agent_openai.sh   (OpenAI, 과금)
  백엔드 없이 화면만 보려면:  MOCK=1 bash sh/run_frontend.sh"
fi
need_free_port FRONTEND_PORT "$FRONTEND_PORT"

# ── 1) 의존성 설치(없을 때만) + 빌드 ────────────────────────────────────
if [ ! -d node_modules ]; then
    echo "npm 의존성 설치 중..."
    npm ci --no-audit --no-fund || npm install --no-audit --no-fund
fi
echo "프론트엔드 빌드 중..."
if [ "$MOCK" = "1" ]; then
    VITE_MOCK=1 npm run build
else
    npm run build
fi

# ── 2) 서빙 (vite preview). 이미 위에서 포트가 비어 있음을 확인했으므로 --strictPort 로 다른 포트로 밀리지 않게 한다 ──
trap 'echo; echo "프론트엔드를 종료합니다..."; cleanup_pids' EXIT
# npx 래퍼 대신 vite 를 직접 실행한다(PID 가 곧 서버 프로세스라 종료 시 고아 프로세스가 남지 않는다).
VITE_PROXY_TARGET="http://127.0.0.1:$BACKEND_PORT" ./node_modules/.bin/vite preview --host 127.0.0.1 --port "$FRONTEND_PORT" --strictPort &
track_pid "$!"
wait_http "http://127.0.0.1:$FRONTEND_PORT/" 60 || die "프론트엔드가 기동하지 않았습니다."

cat <<EOF2

################################################################
 프론트엔드 가동 중 → http://localhost:$FRONTEND_PORT   $( [ "$MOCK" = "1" ] && echo "(목업 모드: 서버 없이 재생)" || echo "(백엔드 :$BACKEND_PORT 필요)" )
################################################################
끄려면 이 터미널에서 Ctrl+C.
EOF2
wait

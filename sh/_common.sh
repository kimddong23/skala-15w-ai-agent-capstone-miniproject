#!/usr/bin/env bash
#
# Vital Sentinel — sh/ 스크립트 공용 함수. 직접 실행하지 않고 각 스크립트가 `source` 한다.
#
# 원칙: ① 이 스크립트들은 "자기가 띄운 프로세스만" 종료한다. 포트가 이미 쓰이고 있으면 다른 프로세스를
# 죽이지 않고 안내만 하고 멈춘다(`lsof | xargs kill -9` 같은 포트 점유 프로세스 강제 종료는 쓰지 않는다).
# ② 파괴적 명령(rm, kill)은 변수가 비어 있지 않고 기대한 모양일 때만 실행한다. ③ Ollama 서버와 모델은 건드리지 않는다
# (모델 삭제는 setup_ollama.sh 의 OLLAMA_FORCE_REFRESH=1 로 명시한 때, 그 스크립트가 쓰는 모델에 한해서만).

# 이 파일이 있는 sh/ 의 부모 = 프로젝트 루트. 비었거나 프로젝트가 아니면 즉시 중단한다.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -n "$ROOT_DIR" ] && [ -f "$ROOT_DIR/requirements.txt" ] && [ -d "$ROOT_DIR/app" ] \
    || { echo "ERROR: 프로젝트 루트를 찾지 못했습니다 (ROOT_DIR='$ROOT_DIR')." >&2; exit 1; }

die() { echo "ERROR: $*" >&2; exit 1; }

# require_cmd <명령> <설치 안내>
require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "$1 이(가) 설치되어 있지 않습니다. $2"
}

# validate_port <이름> <값> — 1~65535 정수만 허용
validate_port() {
    case "$2" in
        ''|*[!0-9]*) die "$1 가 숫자가 아닙니다: '$2'" ;;
    esac
    [ "$2" -ge 1 ] && [ "$2" -le 65535 ] || die "$1 범위 오류(1~65535): $2"
}

# port_in_use <포트> — 127.0.0.1 에서 연결이 되면 사용 중(lsof 없이도 동작하도록 bash 내장 /dev/tcp 사용)
port_in_use() {
    (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
}

# need_free_port <이름> <포트> — 이미 쓰는 프로세스가 있으면 죽이지 않고 안내 후 중단
need_free_port() {
    if port_in_use "$2"; then
        die "포트 $2($1)를 이미 다른 프로세스가 쓰고 있습니다. 그 프로세스는 종료하지 않습니다.
  확인: lsof -nP -iTCP:$2 -sTCP:LISTEN
  다른 포트로 실행: $1=<다른 포트> bash $0"
    fi
}

# wait_http <url> <최대 시도(0.5초 간격)> — 응답이 올 때까지 대기. 성공 0, 실패 1
wait_http() {
    local n
    for n in $(seq 1 "$2"); do
        curl -s --max-time 2 "$1" >/dev/null 2>&1 && return 0
        sleep 0.5
    done
    return 1
}

# use_venv — 이 프로젝트 전용 .venv 의 python 을 PY 로 지정하고 PATH 맨 앞에 둔다. 없으면 setup_venv.sh 를 안내하고 중단.
# (시스템 python 으로 조용히 떨어지면 패키지 누락 오류가 나므로 base 와 달리 명시적으로 중단한다.)
use_venv() {
    [ -x "$ROOT_DIR/.venv/bin/python" ] || die ".venv 가 없습니다. 먼저 실행하세요: bash sh/setup_venv.sh"
    PY="$ROOT_DIR/.venv/bin/python"
    export PATH="$ROOT_DIR/.venv/bin:$PATH"
}

# load_env <required|optional>
#   .env 를 환경변수로 불러온다(이미 셸에 export 한 값이 우선: .env 는 기본값 역할).
#   없으면: required 는 복사 안내 후 중단, optional 은 .env.example 을 복사해 시작한다(덮어쓰기 없음).
load_env() {
    if [ ! -f "$ROOT_DIR/.env" ]; then
        if [ "$1" = "required" ]; then
            die ".env 가 없습니다 — .env.example 을 복사해 .env 로 만들고 값을 채워주세요.
  cp .env.example .env"
        fi
        [ -f "$ROOT_DIR/.env.example" ] || die ".env.example 이 없습니다."
        cp -n "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
        echo "안내: .env 가 없어 .env.example 을 복사했습니다(기본값으로 동작). 필요하면 .env 를 수정하세요."
    fi
    load_env_file
}

# load_env_file — .env 의 KEY=VALUE 줄만 읽는다(이미 셸에 있는 값 우선, 임의 명령은 실행하지 않는다).
load_env_file() {
    [ -f "$ROOT_DIR/.env" ] || return 0
    # 셸에 이미 있는 값은 .env 가 덮어쓰지 않도록 한다. (.env 의 KEY=VALUE 줄만 읽는다 — 임의 명령은 실행하지 않는다)
    local line key val
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in ''|'#'*) continue ;; esac
        key="${line%%=*}"; val="${line#*=}"
        case "$key" in *[!A-Za-z0-9_]*|'') continue ;; esac
        val="${val%\"}"; val="${val#\"}"; val="${val%\'}"; val="${val#\'}"
        if [ -z "${!key+x}" ]; then export "$key=$val"; fi
    done < "$ROOT_DIR/.env"
}

# Ctrl+C / 종료 신호를 받으면 일반 종료(exit)로 바꿔 각 스크립트의 EXIT 트랩(cleanup_pids)이 항상 실행되게 한다.
trap 'exit 130' INT TERM HUP

# config_default <ENV 이름> — app/config.py 의 os.getenv("이름", "기본값") 에서 기본값을 읽는다.
# 모델 기본값을 스크립트에 따로 적어 두면 설정과 어긋나므로, 설정 파일을 단일 출처로 삼는다. 못 찾으면 빈 문자열.
config_default() {
    case "$1" in ''|*[!A-Z0-9_]*) die "config_default: 잘못된 이름 '$1'" ;; esac
    sed -n "s/.*os\.getenv(\"$1\", \"\([^\"]*\)\").*/\1/p" "$ROOT_DIR/app/config.py" | head -n 1
}

# pick_model <base 철자 변수 값> <앱 변수 값> <config ENV 이름> — 모델 이름을 정한다.
# 우선순위: base 철자(OLLAMA_*_MODEL) > 앱 변수(FAST_MODEL 등, .env/셸) > app/config.py 의 기본값. 결과가 비면 중단한다.
pick_model() {
    local v="${1:-${2:-$(config_default "$3")}}"
    [ -n "$v" ] || die "$3 모델 이름을 정하지 못했습니다(app/config.py 기본값 확인)."
    echo "$v"
}

# PIDS — 이 스크립트가 띄운 프로세스 PID 목록. cleanup 은 이것만 종료한다.
PIDS=()
track_pid() { PIDS+=("$1"); }
cleanup_pids() {
    local p
    for p in "${PIDS[@]:-}"; do
        case "$p" in ''|*[!0-9]*) continue ;; esac   # 숫자 PID 만
        kill "$p" 2>/dev/null || true
    done
}

# json_pretty — 표준입력 JSON 을 보기 좋게 출력(파이썬이 없으면 그대로)
json_pretty() {
    "${PY:-python3}" -m json.tool --no-ensure-ascii 2>/dev/null || cat
}

# use_sqlite_checkpointer — 대화 기억을 sqlite 파일에 저장해 서버를 껐다 켜도 같은 thread 로 이어 가게 한다(기본값).
# 셸/.env 에서 CHECKPOINTER=memory 등으로 이미 정했으면 그 값을 따른다. 파일은 data/ 아래(.gitignore 의 *.sqlite* 대상).
use_sqlite_checkpointer() {
    export CHECKPOINTER="${CHECKPOINTER:-sqlite}"
    export SQLITE_PATH="${SQLITE_PATH:-$ROOT_DIR/data/checkpoints.sqlite}"
    echo "대화 기억: CHECKPOINTER=$CHECKPOINTER$([ "$CHECKPOINTER" = "sqlite" ] && echo " ($SQLITE_PATH)")"
}

# start_backend — uvicorn 으로 백엔드(FastAPI + MCP subprocess)를 백그라운드로 띄우고 /health 가 응답할 때까지 기다린다.
# 필요한 변수: PY, BACKEND_PORT(검증 끝난 값). 띄운 PID 는 PIDS 에 기록한다(cleanup_pids 가 종료).
start_backend() {
    local log="${TMPDIR:-/tmp}/vital_sentinel_backend_${BACKEND_PORT}.log"
    ( cd "$ROOT_DIR" && exec "$PY" -m uvicorn app.api.main:app --host 127.0.0.1 --port "$BACKEND_PORT" ) >"$log" 2>&1 &
    track_pid "$!"
    if ! wait_http "http://127.0.0.1:$BACKEND_PORT/health" 240; then
        echo "--- 백엔드 로그 마지막 20줄 ($log) ---" >&2
        tail -n 20 "$log" >&2 || true
        die "백엔드가 기동하지 않았습니다."
    fi
}

# print_ready <모드 설명> <모델 설명> — 기동 완료 안내와 바로 해볼 수 있는 명령(질문 예시는 U05 의 최근 30일 점검)
print_ready() {
    echo
    curl -s "http://127.0.0.1:$BACKEND_PORT/health" | json_pretty
    cat <<EOF2

################################################################
 백엔드 가동 중 (http://127.0.0.1:$BACKEND_PORT) — $1
 모델: $2
 대화 기억: ${CHECKPOINTER:-memory}
################################################################

켜졌습니다. 이렇게 시험해 볼 수 있습니다:

1) curl 로 질문하기 (user_id 는 가명 U01~U15)
   curl -s -X POST localhost:$BACKEND_PORT/chat -H 'Content-Type: application/json' \\
     -d '{"message":"U05 최근 30일 심박 이상 점검해줘","user_id":"U05","thread_id":"t-test"}' \\
     | python3 -m json.tool --no-ensure-ascii

2) 진행 단계를 실시간으로 받기 (SSE)
   curl -N -X POST localhost:$BACKEND_PORT/chat/stream -H 'Content-Type: application/json' \\
     -d '{"message":"어젯밤 수면 중 심박 괜찮았어?","user_id":"U05","thread_id":"t-test-sse"}'

3) 브라우저 화면 (새 터미널에서)
   bash sh/run_frontend.sh      → http://localhost:${FRONTEND_PORT:-8600}
   (프론트는 백엔드가 ${BACKEND_PORT}에 떠 있어야 동작합니다)

더 자세한 설명: docs/manual_ollma.md.  끄려면 이 터미널에서 Ctrl+C.
EOF2
}

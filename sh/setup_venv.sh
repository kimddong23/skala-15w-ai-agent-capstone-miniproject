#!/usr/bin/env bash
#
# Vital Sentinel — 이 프로젝트 전용 .venv(Python 3.11)를 만들고 requirements.txt 를 설치한다.
# ("맨 처음 한 번" 실행하는 스크립트)
#
# uv 가 있으면 `uv venv --python 3.11` 로 3.11 을 확보하고(없으면 uv 가 받아온다),
# uv 가 없으면 python3.11 -m venv 로 대체한다. 이미 .venv 가 있으면 새로 만들지 않고 설치만 다시 확인한다.
# 설치는 activate 가 아니라 .venv 안의 python 을 직접 호출해 항상 .venv 안에 들어가게 한다.
#
# 실행:  bash sh/setup_venv.sh
#
set -euo pipefail
# shellcheck source=sh/_common.sh
source "$(dirname "$0")/_common.sh"
cd "$ROOT_DIR"

# ── 1) .venv 만들기 (있으면 그대로 사용) ──────────────────────────────────
if [ -x ".venv/bin/python" ]; then
    echo "기존 .venv 를 그대로 쓰고 패키지 설치만 확인합니다."
elif command -v uv >/dev/null 2>&1; then
    echo "uv venv --python 3.11 .venv"
    uv venv --python 3.11 .venv
elif command -v python3.11 >/dev/null 2>&1; then
    echo "uv 가 없어 python3.11 -m venv 로 대체합니다."
    python3.11 -m venv .venv
else
    die "uv 도 python3.11 도 없습니다. 둘 중 하나를 설치하세요.
  uv:      https://docs.astral.sh/uv/  (curl -LsSf https://astral.sh/uv/install.sh | sh)
  python:  https://www.python.org/downloads/ (3.11)"
fi
[ -x ".venv/bin/python" ] || die ".venv 생성에 실패했습니다."

ver="$(.venv/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
case "$ver" in
    3.11) ;;
    *) echo "경고: .venv 의 Python 이 $ver 입니다. 이 프로젝트는 3.11 로 검증했습니다(다른 버전은 패키지 설치가 실패할 수 있음)." ;;
esac

# ── 2) 패키지 설치 ──────────────────────────────────────────────────────
echo "requirements.txt 설치 중..."
if command -v uv >/dev/null 2>&1; then
    uv pip install --python .venv/bin/python -r requirements.txt
else
    .venv/bin/python -m pip install --upgrade pip --quiet
    .venv/bin/python -m pip install -r requirements.txt
fi

echo
echo "패키지 점검:"
.venv/bin/python -c "import fastapi, uvicorn, langgraph, fastmcp, mcp, httpx, pandas, pyarrow, sklearn; print('  핵심 패키지 import 확인 완료')"

cat <<EOF2

################################################################
 .venv 준비 완료 (Python $ver)
################################################################

다음 순서로 실행합니다:
  1) bash sh/setup_ollama.sh        (선택: 로컬 LLM 모델 준비)
  2) bash sh/run_agent_fake.sh      (LLM 없이 기본 리포트 모드)  또는
     bash sh/run_agent_ollma.sh     (로컬 Ollama)
  3) bash sh/run_frontend.sh        (다른 터미널: 브라우저 화면)
EOF2

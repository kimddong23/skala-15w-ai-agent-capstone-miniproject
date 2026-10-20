"""테스트 공통 설정: v1 기준선 테스트가 많으므로 기본으로 허용한다. 거부 동작은 `test_v1_guard.py` 가 환경변수를 지우고 확인한다."""
import os

os.environ.setdefault("ALLOW_V1_BASELINE", "yes")


# --------------------------------------------------------------------------------------------- E2E 공통 fixture
# 실제 uvicorn 서브프로세스(+실제 MCP stdio)를 모듈마다 한 번 띄우는 기본 서버. test_e2e_*.py / test_regression_queries.py 가 공유한다.
import socket  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402

from app.config import ROOT, settings  # noqa: E402

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def base_url():
    """LLM 끔·memory 체크포인터 기본 서버의 URL (processed 데이터가 없으면 건너뛴다)."""
    if not (settings.data_dir / "vitals_minute.parquet").exists():
        pytest.skip("processed 데이터 없음 (app.data.prepare 먼저 실행)")
    port = free_port()
    env = {**os.environ, "LLM_ENABLED": "false", "CHECKPOINTER": "memory", "PYTHONPATH": str(ROOT)}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.api.main:app", "--port", str(port), "--log-level", "warning"],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            if proc.poll() is not None:
                raise RuntimeError(f"uvicorn 종료: {proc.stdout.read().decode()[-2000:]}")
            try:
                if httpx.get(f"{url}/health", timeout=1).json().get("mcp"):
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        else:
            raise RuntimeError("서버가 준비되지 않음")
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="module")
def http(base_url):
    """기본 서버용 httpx 클라이언트."""
    with httpx.Client(base_url=base_url, timeout=60) as c:
        yield c



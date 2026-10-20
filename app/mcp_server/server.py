"""전체 도구를 한 프로세스에 모은 호환 서버 (`vitals` + `guidelines` 마운트).

운영은 두 서버를 따로 띄운다 (`vitals_server.py`, `guidelines_server.py`; 클라이언트가 이름으로 탐색). 이 모듈은 단일 서버를 기대하던
기존 사용처(테스트, 수동 점검, `python -m app.mcp_server.server`)를 위해 남긴 얇은 합성이다. 도구 구현은 두 모듈에만 있다.
"""
from fastmcp import FastMCP

from app.mcp_server.guidelines_server import mcp as guidelines_mcp
from app.mcp_server.guidelines_server import search_guidelines  # noqa: F401  (호환 re-export)
from app.mcp_server.vitals_server import (  # noqa: F401  (호환 re-export)
    anomaly_review,
    detect_anomalies,
    get_health_events,
    get_hr_series,
    get_vitals_summary,
    list_users,
    store,
)
from app.mcp_server.vitals_server import mcp as vitals_mcp

mcp = FastMCP("vital-sentinel", instructions="vitals + guidelines 서버를 합친 호환 서버입니다. 운영 클라이언트는 두 서버를 따로 연결합니다.")
mcp.mount(vitals_mcp)
mcp.mount(guidelines_mcp)

if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)

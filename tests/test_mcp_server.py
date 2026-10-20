"""MCP 서버 테스트.

범위: Tool/Resource/Prompt 등록, 응답 스키마와 단위, 입력 검증 오류가 예외가 아닌 error 응답으로 오는지, stdio 기동.
협력: fastmcp 인메모리 Client (프로토콜 계층을 거치되 subprocess 없이 빠름) + 실제 subprocess 한 번(stdio).
데이터 가정: `tests/_prepared.ensure_prepared()` 가 서버 import 전에 산출물을 보장한다 (서버는 import 시 store 를 로드).
"""
import asyncio
import json
import sys

import pandas as pd
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from app.config import ROOT, settings
from tests._prepared import ensure_prepared

ensure_prepared()
from app.mcp_server.server import mcp  # noqa: E402  (싱글턴 store 가 import 시 로드되므로 데이터 보장 후 import)

LABELS = json.loads((settings.data_dir / "labels.json").read_text())


def call(name: str, args: dict | None = None):
    """인메모리 Client 로 도구를 호출하고 .data 를 반환한다 (호출마다 새 세션)."""
    async def run():
        async with Client(mcp) as c:
            return await c.call_tool(name, args or {})
    return asyncio.run(run()).data


def test_tools_resources_prompt_registered():
    """계약의 도구 4종 + 가이드라인 검색 도구, 리소스 3종, 프롬프트 1종이 등록되고 도구 설명이 LLM 이 읽을 만큼 충분히 길다."""
    async def run():
        async with Client(mcp) as c:
            return ([t.name for t in await c.list_tools()], [str(r.uri) for r in await c.list_resources()],
                    [p.name for p in await c.list_prompts()], await c.list_tools())
    tools, resources, prompts, full = asyncio.run(run())
    assert set(tools) == {"list_users", "get_vitals_summary", "detect_anomalies", "get_hr_series", "get_health_events", "search_guidelines"}
    assert set(resources) == {"guideline://heart-rate-ranges", "guideline://red-flags", "guideline://disclaimer"}
    assert prompts == ["anomaly_review"]
    assert all(len(t.description) > 80 for t in full)      # docstring 이 LLM 의 도구 선택 근거


def test_list_users_tool():
    """list_users 도구가 가명 사용자 목록을 반환한다."""
    users = call("list_users")
    assert len(users) >= 10 and users[0]["user_id"] == "U01"


def test_summary_tool_shape():
    """get_vitals_summary 응답이 계약 키와 일별 안정시 심박 목록을 갖는다."""
    r = call("get_vitals_summary", {"user_id": "U12", "lookback_days": 7})
    assert set(r) == {"user_id", "window", "summary", "daily_resting_hr"}
    assert r["summary"]["n_minutes"] > 0 and 0 < r["summary"]["coverage_pct"] <= 100
    assert r["daily_resting_hr"] and set(r["daily_resting_hr"][0]) == {"date", "resting_hr", "n_rest_minutes"}
    json.dumps(r)


def test_detect_finds_injected_critical_event():
    """주입된 critical 빈맥 시각을 조회하면 탐지 결과에 critical 이벤트가 있고 라벨 구간과 겹친다."""
    lb = next(x for x in LABELS if x["type"] == "resting_tachycardia" and x["severity"] == "critical")
    end = (pd.Timestamp(lb["end"]) + pd.Timedelta(hours=2)).isoformat()
    r = call("detect_anomalies", {"user_id": lb["user_id"], "lookback_days": 1, "end": end})
    crit = [a for a in r["anomalies"] if a["type"] == "resting_tachycardia" and a["severity"] == "critical"]
    assert crit and r["counts"]["resting_tachycardia"] >= 1
    a = crit[0]
    assert set(a) == {"anomaly_id", "type", "start", "end", "duration_min", "severity", "value", "baseline",
                      "context", "evidence", "stats"}
    assert pd.Timestamp(a["start"]) <= pd.Timestamp(lb["end"]) and pd.Timestamp(a["end"]) >= pd.Timestamp(lb["start"])
    json.dumps(r)


def test_detect_default_end_uses_last_record():
    """end 생략 시 조회 끝이 사용자의 마지막 기록 시각이다."""
    r = call("detect_anomalies", {"user_id": "U03"})
    assert "anomalies" in r and r["window"]["end"] == call("list_users")[2]["end"]


def test_series_is_downsampled():
    """시계열은 5분 간격으로 다운샘플되어 점 수가 하루 288개 이하다 (데이터 최소화)."""
    r = call("get_hr_series", {"user_id": "U12", "lookback_days": 1})
    pts = r["points"]
    assert 0 < len(pts) <= 288 and set(pts[0]) == {"ts", "hr", "steps", "asleep"}
    gaps = {(pd.Timestamp(b["ts"]) - pd.Timestamp(a["ts"])).total_seconds() % 300 for a, b in zip(pts, pts[1:])}
    assert gaps == {0.0}
    json.dumps(r)


def test_series_30_days_is_coarsened_to_max_points():
    """30일 조회는 점 수 상한(2000)에 맞게 간격이 자동 확대되고 명시한 1h 는 유지된다."""
    r = call("get_hr_series", {"user_id": "U12", "lookback_days": 30})
    assert "error" not in r and 0 < len(r["points"]) <= 2000
    assert r["resample"] != "5min" and int(r["resample"].removesuffix("min")) % 5 == 0
    assert call("get_hr_series", {"user_id": "U12", "lookback_days": 30, "resample": "1h"})["resample"] == "1h"


def test_errors_are_returned_not_raised():
    """잘못된 user_id, 시각 형식, 기간, resample 은 예외 대신 error 응답으로 반환된다."""
    assert "error" in call("get_vitals_summary", {"user_id": "U99"})
    assert "error" in call("detect_anomalies", {"user_id": "U99"})
    assert "error" in call("get_hr_series", {"user_id": "nope"})
    assert "ISO" in call("detect_anomalies", {"user_id": "U03", "end": "어제"})["error"]
    assert "lookback_days" in call("get_vitals_summary", {"user_id": "U03", "lookback_days": -1})["error"]
    assert "5min" in call("get_hr_series", {"user_id": "U03", "resample": "1min"})["error"]
    assert "resample" in call("get_hr_series", {"user_id": "U03", "resample": "abc"})["error"]
    assert "lookback_days" in call("get_hr_series", {"user_id": "U03", "lookback_days": 31})["error"]


def test_resources_and_prompt():
    """가이드라인 3종이 읽히고 면책 문구가 들어 있으며 프롬프트가 인자를 치환한다."""
    async def run():
        async with Client(mcp) as c:
            docs = {u: (await c.read_resource(u))[0].text for u in
                    ["guideline://heart-rate-ranges", "guideline://red-flags", "guideline://disclaimer"]}
            prompt = await c.get_prompt("anomaly_review", {"user_id": "U05", "lookback_days": "14"})
            return docs, prompt.messages[0].content.text
    docs, prompt = asyncio.run(run())
    assert "60~100" in docs["guideline://heart-rate-ranges"]
    assert "119" in docs["guideline://red-flags"]
    assert "의료기기가 아니며 진단을 제공하지 않습니다" in docs["guideline://disclaimer"]
    assert "U05" in prompt and "14" in prompt and "detect_anomalies" in prompt and "get_vitals_summary" in prompt


def test_server_starts_over_stdio():
    """`python -m app.mcp_server.server` 가 stdio 로 기동해 도구를 호출할 수 있다."""
    async def run():
        t = StdioTransport(command=sys.executable, args=["-m", "app.mcp_server.server"], cwd=str(ROOT))
        async with Client(t) as c:
            return (await c.call_tool("list_users", {})).data
    assert asyncio.run(run())[0]["user_id"] == "U01"


def test_detect_output_passes_schema_for_every_user():
    """모든 사용자의 30일 탐지 결과가 MCP 출력 스키마 검증과 JSON 직렬화를 통과한다.

    회귀 방지: 지속시간에 numpy 정수가 섞이면 FastMCP 가 구조화 출력을 거부해 에이전트가 '탐지 실패'로 빠졌다.
    """
    users = call("list_users")
    ids = [u["user_id"] for u in (users if isinstance(users, list) else users["users"])]
    for uid in ids:
        out = call("detect_anomalies", {"user_id": uid, "lookback_days": 30})
        assert "error" not in out, (uid, out)
        json.dumps(out)
        assert all(type(a["duration_min"]) is int for a in out["anomalies"]), uid

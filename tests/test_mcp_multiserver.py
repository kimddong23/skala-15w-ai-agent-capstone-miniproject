"""다중 MCP 서버(vitals + guidelines) 연결·이름 탐색·부분 성능 저하 테스트.

범위: `app.agent.toolbox.connect_mcp` 가 실제 stdio subprocess 두 개를 띄우고, 서버 이름과 무관하게 `list_tools/list_resources/list_prompts` 로
도구·리소스·프롬프트를 찾는지, 서버 하나가 죽었을 때 나머지로 계속하는지(가이드라인 서버 -> 번들 폴백 텍스트, 데이터 서버 -> task_failed).
협력: `app.mcp_server.*_server`, `app.agent.graph`(부분 ToolBox 로 실제 그래프 실행), `tests/fakes`(없음 - 여기선 진짜 서버 사용).
데이터 가정: `tests/_prepared.ensure_prepared()` 로 데이터가 있고, Ollama 는 필요 없다(NullLLM, 검색은 폴백 가능).
"""
import logging
import sys
from contextlib import AsyncExitStack

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.agent import toolbox as tb
from app.agent.graph import build_graph
from app.agent.llm import NullLLM
from app.agent.toolbox import ToolUnavailable, connect_mcp, default_servers, pick_guideline_uris
from tests._prepared import ensure_prepared

ensure_prepared()
BROKEN = {"transport": "stdio", "command": sys.executable, "args": ["-c", "import sys; sys.exit(3)"]}


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    """재시도 지터를 없앤다 (실패 경로 테스트가 빨라지게)."""
    monkeypatch.setattr("app.agent.retry.random.uniform", lambda a, b: 0)


async def graph_run(box, message: str):
    """부분 ToolBox 로 그래프를 한 번 실행해 (custom 이벤트 목록, 최종 state) 를 돌려준다."""
    graph = build_graph(box, InMemorySaver(), NullLLM())
    cfg = {"configurable": {"thread_id": "mt"}, "recursion_limit": 40}
    events = [c async for mode, c in graph.astream({"messages": [("user", message)]}, cfg, stream_mode=["custom", "updates"]) if mode == "custom"]
    return events, (await graph.aget_state(cfg)).values


@pytest.mark.asyncio
async def test_both_servers_start_and_discovery_binds_tools_resources_and_prompt():
    async with AsyncExitStack() as stack:
        box = await connect_mcp(stack)
        assert box.servers == {"vitals": True, "guidelines": True}
        assert (await box.list_users())[0]["user_id"] == "U01"
        assert await box.list_resources() == ["guideline://disclaimer", "guideline://heart-rate-ranges", "guideline://red-flags"]
        assert "의료기기가 아니며" in await box.resource("guideline://disclaimer")
        assert "get_vitals_summary" in await box.prompt("anomaly_review", {"user_id": "U01", "lookback_days": "7"})
        hits = await box.search("커피 마시면 심박수가 올라가나요", 2)
        assert hits["hits"] and hits["method"] in ("vector", "keyword")
        det = await box.detect("U03", 30, None)
        assert "anomalies" in det


@pytest.mark.asyncio
async def test_discovery_does_not_depend_on_server_names_or_layout(caplog):
    """서버 이름을 바꾸고(y/x) 합쳐진 단일 서버로 바꿔도 같은 ToolBox 가 만들어진다 - 도구->서버 매핑이 코드에 없다."""
    swapped = {"x": default_servers()["guidelines"], "y": default_servers()["vitals"]}
    single = {"only": {**default_servers()["vitals"], "args": ["-m", "app.mcp_server.server"]}}
    for cfg in (swapped, single):
        async with AsyncExitStack() as stack:
            with caplog.at_level(logging.INFO):
                box = await connect_mcp(stack, cfg)
            assert all(box.servers.values()) and box.search is not None and box.prompt is not None
            assert len(await box.list_resources()) == 3 and (await box.list_users())[0]["user_id"] == "U01"
    assert any("MCP 탐색 결과" in r.message for r in caplog.records)


def test_pick_guideline_uris_resolves_keys_from_discovered_names():
    uris = ["guideline://heart-rate-ranges", "guideline://red-flags", "guideline://disclaimer", "other://x"]
    assert pick_guideline_uris(uris, ["ranges", "red_flags", "disclaimer", "missing"]) == {
        "ranges": "guideline://heart-rate-ranges", "red_flags": "guideline://red-flags", "disclaimer": "guideline://disclaimer"}


@pytest.mark.asyncio
async def test_guidelines_server_down_degrades_to_bundled_text_and_vitals_keep_working():
    cfg = {"vitals": default_servers()["vitals"], "guidelines": BROKEN}
    async with AsyncExitStack() as stack:
        box = await connect_mcp(stack, cfg)
        assert box.servers == {"vitals": True, "guidelines": False} and box.search is None
        with pytest.raises(ToolUnavailable, match="guideline://disclaimer"):
            await box.resource("guideline://disclaimer")
        assert (await box.list_users())[0]["user_id"] == "U01"
        events, st = await graph_run(box, "안정시 심박수 정상 범위가 어떻게 돼요?")      # 번들 요약/키워드 발췌로 응답
        assert st["route"] == "simple" and "60~100" in st["report"] and not [e for e in events if e["type"] == "error"]
        events, st = await graph_run(box, "U03 최근 30일 이상 점검해줘")                 # 데이터 경로는 영향 없음
        assert st["route"] == "analysis" and not [e for e in events if e["type"] == "task_failed"]


@pytest.mark.asyncio
async def test_vitals_server_down_fails_data_tasks_in_isolation_but_guidelines_still_answer():
    cfg = {"vitals": BROKEN, "guidelines": default_servers()["guidelines"]}
    async with AsyncExitStack() as stack:
        box = await connect_mcp(stack, cfg)
        assert box.servers == {"vitals": False, "guidelines": True} and box.prompt is None
        with pytest.raises(ToolUnavailable, match="list_users"):
            await box.list_users()
        assert "의료기기가 아니며" in await box.resource("guideline://disclaimer")
        events, st = await graph_run(box, "U01 이상 점검해줘")
        failed = [e for e in events if e["type"] == "task_failed"]
        assert failed and st["report"]                                           # task_failed 로 격리되고 리포트(한계 안내)는 만들어진다
        assert "이상 탐지를 실행하지 못해" in st["report"]


@pytest.mark.asyncio
async def test_all_servers_down_raises():
    async with AsyncExitStack() as stack:
        with pytest.raises(RuntimeError, match="하나도"):
            await connect_mcp(stack, {"a": BROKEN, "b": BROKEN})


def test_resource_and_prompt_reads_are_traced_spans():
    """리소스/프롬프트 읽기는 LangSmith span(@traceable)으로 감싸져 트레이스 트리에서 빠지지 않는다."""
    assert hasattr(tb._read_resource, "__wrapped__") and hasattr(tb._get_prompt, "__wrapped__")


# ---------- 서버별 탐색 격리 (타임아웃·예외) ----------
from pathlib import Path  # noqa: E402

BAD = str(Path(__file__).with_name("mcp_bad_server.py"))


def bad(mode: str) -> dict:
    """tests/mcp_bad_server.py 를 mode 로 띄우는 stdio 설정."""
    return {"transport": "stdio", "command": sys.executable, "args": [BAD, mode]}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["hang_tools", "raise_tools"])
async def test_one_server_whose_tool_listing_hangs_or_raises_does_not_take_down_the_others(mode, caplog):
    cfg = {"bad": bad(mode), "guidelines": default_servers()["guidelines"], "vitals": default_servers()["vitals"]}
    async with AsyncExitStack() as stack:
        with caplog.at_level(logging.WARNING):
            box = await connect_mcp(stack, cfg, discovery_timeout_s=2.0)
        assert box.servers["bad"] is False and box.servers["guidelines"] and box.servers["vitals"]       # 고장 서버만 저하로 표시
        assert (await box.list_users())[0]["user_id"] == "U01"                                          # 다른 서버의 도구는 정상
        assert box.search is not None and len(await box.list_resources()) == 3
    assert any("'bad'" in r.message and "도구 탐색 실패" in r.message for r in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["raise_resources", "hang_resources", "raise_prompts"])
async def test_partial_listing_failure_keeps_the_servers_other_lists(mode):
    """리소스/프롬프트 목록만 고장 난 서버: 같은 서버의 도구(ping)는 여전히 쓸 수 있고 서버 상태만 False 이다."""
    async with AsyncExitStack() as stack:
        box = await connect_mcp(stack, {"bad": bad(mode)}, discovery_timeout_s=2.0)
        assert box.servers == {"bad": False}
        assert box.search is None and (await box.list_resources()) == []


@pytest.mark.asyncio
async def test_hanging_discovery_is_bounded_by_the_timeout():
    import time
    async with AsyncExitStack() as stack:
        t0 = time.perf_counter()
        box = await connect_mcp(stack, {"bad": bad("hang_tools"), "good": default_servers()["guidelines"]}, discovery_timeout_s=1.0)
        assert time.perf_counter() - t0 < 25                                  # 서버 기동 시간 + 1 s 정도 (3600 s 를 기다리지 않는다)
        assert box.servers == {"bad": False, "good": True} and box.search is not None


# ---------- ToolBox.events / list_users(dataset) ----------
@pytest.mark.asyncio
async def test_toolbox_events_and_list_users_dataset_pass_through_to_the_mcp_tools():
    async with AsyncExitStack() as stack:
        box = await connect_mcp(stack)
        everyone = await box.list_users()
        fit = await box.list_users("fitbit")
        assert fit and all(u["dataset"] == "fitbit" for u in fit) and len(everyone) >= len(fit)
        assert await box.list_users("no-such-dataset") == []
        ev = await box.events("U01")
        assert ev["events"] == [] and ev["dataset"] == "fitbit" and "note" in ev                # FitBit 사용자는 항상 빈 목록
        assert "error" in await box.events("U999")                                               # 알 수 없는 사용자: MCP 오류 계약({"error"})이 그대로 온다


@pytest.mark.asyncio
async def test_events_without_a_providing_server_returns_empty_shape_and_unavailable_vitals_raise_nothing_for_events():
    async with AsyncExitStack() as stack:
        box = await connect_mcp(stack, {"guidelines": default_servers()["guidelines"]})          # get_health_events 를 가진 vitals 서버가 없다
        assert box.events is not None and await box.events("S001") == {"user_id": "S001", "events": []}
        with pytest.raises(ToolUnavailable):
            await box.list_users()

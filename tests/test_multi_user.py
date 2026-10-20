"""다중 사용자 비교 요청("U01이랑 U05 비교해줘") 테스트: 사용자별 병렬 조회·오류 격리·사용자별 안전 게이트·인원 한도·사실 비혼합.

협력: `tests/fakes.py`(FakeTools 확장), `app.agent.{gates,graph,deps,report,fake_llm}`.
"""
import copy

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.agent import gates
from app.agent.deps import AgentDeps
from app.agent.fake_llm import FakeBackend
from app.agent.graph import build_graph
from app.agent.llm import NullLLM
from app.config import settings
from tests.fakes import CRITICAL, SUMMARY, WARNING, FakeLLM, FakeTools, assemble

PER_USER = {"U01": [WARNING], "U02": [], "U03": [], "U04": [], "U05": [CRITICAL]}


class PerUserTools(FakeTools):
    """사용자마다 다른 탐지 결과·커버리지를 돌려주는 도구 (사용자 간 사실 혼합을 잡아내기 위함)."""

    coverage_by_user: dict = {}
    failing: set = set()

    async def _users(self):
        return [{"user_id": u, "start": "2016-03-12", "end": "2016-05-12", "days": 61, "coverage_pct": 80.0} for u in PER_USER]

    async def _summary(self, uid, days, end):
        self.summary_calls.append((uid, days, end))
        out = copy.deepcopy(SUMMARY)
        out["user_id"] = uid
        out["summary"]["coverage_pct"] = self.coverage_by_user.get(uid, 88.0)
        out["summary"]["hr_mean"] = {"U01": 70.1, "U05": 66.6}.get(uid, 72.4)
        return out

    async def _detect(self, uid, days, end):
        self.detect_calls.append((uid, days, end))
        if uid in self.failing:
            raise ValueError("boom")
        an = copy.deepcopy(PER_USER[uid])
        return {"user_id": uid, "window": SUMMARY["window"], "anomalies": an, "counts": ({an[0]["type"]: len(an)} if an else {})}


async def run(graph, message, thread="mu", user_id=None):
    cfg = {"configurable": {"thread_id": thread}, "recursion_limit": 40}
    inp = {"messages": [("user", message)]}
    if user_id:
        inp["user_id"] = user_id
    events = [c async for c in graph.astream(inp, cfg, stream_mode="custom")]
    return events, (await graph.aget_state(cfg)).values


def make(tools, llm=None):
    llm = llm or NullLLM()
    return build_graph(AgentDeps(llm, llm, tools.toolbox(), InMemorySaver(), settings, "v2"))


def test_extract_user_ids_dedups_orders_and_caps():
    assert gates.extract_user_ids("U01이랑 U05 비교해줘") == (["U01", "U05"], [])
    assert gates.extract_user_ids("u02 와 U02 그리고 U03") == (["U02", "U03"], [])
    assert gates.extract_user_ids("U01 U02 U03 U04 U05 비교") == (["U01", "U02", "U03"], ["U04", "U05"])
    assert gates.extract_user_ids("점검해줘") == ([], []) and gates.has_strong_intent("U01이랑 U05 비교해줘")


@pytest.mark.asyncio
async def test_two_users_are_fetched_in_parallel_with_per_user_task_ids():
    tools = PerUserTools()
    events, st = await run(make(tools), "U01이랑 U05 비교해줘")
    assert events[0]["route"] == "analysis" and st["user_ids"] == ["U01", "U05"]
    assert sorted(c[0] for c in tools.summary_calls) == ["U01", "U05"] and sorted(c[0] for c in tools.detect_calls) == ["U01", "U05"]
    started = {e["task_id"] for e in events if e["type"] == "task_started" and e["task_type"] == "research"}
    assert {"r_summary:U01", "r_summary:U05", "r_detect:U01", "r_detect:U05"} <= started
    report = st["report"]
    assert "(U01)" in report and "(U05)" in report and "## 사용자 비교" in report and st["validation"]["passed"]
    assert "안정 시 빈맥: U01 0건, U05 1건" in report or "U05 1건" in report


@pytest.mark.asyncio
async def test_safety_gate_runs_per_user_and_names_the_critical_user():
    events, st = await run(make(PerUserTools()), "U01이랑 U05 비교해줘")
    esc = next(e for e in events if e["type"] == "safety_escalation")
    assert "U05" in esc["reason"] and "해당 사용자: U05" in esc["message"] and "U01" not in esc["message"]
    assert gates.ESCALATION_MESSAGE in st["report"] and "해당 사용자: U05" in st["report"] and st["escalation"] is True
    assert st["validation"]["passed"]
    # critical 이 없는 두 사용자는 에스컬레이션하지 않는다
    _, calm = await run(make(PerUserTools()), "U01이랑 U02 비교해줘", thread="calm")
    assert calm["escalation"] is False and gates.ESCALATION_MESSAGE not in calm["report"]


@pytest.mark.asyncio
async def test_one_users_failure_is_isolated_and_reported_without_all_clear():
    tools = PerUserTools()
    tools.failing = {"U05"}
    events, st = await run(make(tools), "U01이랑 U05 비교해줘")
    assert any(e["type"] == "task_failed" and e["task_id"] == "r_detect:U05" for e in events)
    assert not any(e["type"] == "task_failed" and e["task_id"].endswith(":U01") for e in events)  # U01 은 영향 없음
    r = st["report"]
    assert "(U01)" in r and "U05: 탐지 결과를 확인하지 못해" in r and "이상 탐지를 실행하지 못해 이상 여부를 판단할 수 없습니다" in r
    assert st["analyzed_user"] is None and st["validation"]["passed"]  # 일부 실패한 비교는 후속 질문의 근거가 되지 않는다


@pytest.mark.asyncio
async def test_more_than_three_users_are_capped_and_the_rest_are_named():
    tools = PerUserTools()
    _, st = await run(make(tools), "U01 U02 U03 U04 비교해줘")
    assert st["user_ids"] == ["U01", "U02", "U03"] and st["excluded_users"] == ["U04"]
    assert all(c[0] != "U04" for c in tools.summary_calls)
    assert st["report"].count("제외된 사용자: U04") == 1  # 조용히 무시하지 않고, 한 번만 밝힌다


@pytest.mark.asyncio
async def test_unknown_user_in_a_comparison_stops_before_any_tool_call():
    tools = PerUserTools()
    events, st = await run(make(tools), "U01이랑 U99 비교해줘")
    assert events[0]["route"] == "simple" and "U99" in st["report"] and "찾을 수 없습니다" in st["report"] and tools.summary_calls == []


@pytest.mark.asyncio
async def test_facts_are_not_mixed_between_users_in_the_llm_prompt():
    llm = FakeLLM(text={"writing": "## 요약\n- U01 평균 심박 70.1 bpm, U05 평균 심박 66.6 bpm\n- 사용자별 비교입니다.", "analysis": "비교 해석"})
    _, st = await run(make(PerUserTools(), llm), "U01이랑 U05 비교해줘")
    prompt = llm.prompts["writing"][0]
    u01, u05 = prompt.split("### 사용자 U05")[0], prompt.split("### 사용자 U05")[1]
    assert "### 사용자 U01" in u01 and "평균_심박(bpm)=70.1" in u01 and "135.0" not in u01  # U05 의 critical 수치가 U01 블록에 없다
    assert "평균_심박(bpm)=66.6" in u05 and "135.0 bpm" in u05 and "구분해 서술" in prompt
    assert st["validation"]["passed"]  # 각 사용자 수치를 정확히 인용하면 통과 (사실 집합은 모든 사용자 합집합)


@pytest.mark.asyncio
async def test_fake_backend_comparison_passes_validation_first_try_with_usage():
    graph = make(PerUserTools(), FakeBackend())
    events, st = await run(graph, "U01이랑 U05 비교해줘")
    assert [e["passed"] for e in events if e["type"] == "validation"] == [True] and assemble(events) == st["report"]
    assert gates.ESCALATION_MESSAGE in st["report"] and "해당 사용자: U05" in st["report"]


@pytest.mark.asyncio
async def test_replan_triggers_when_any_one_user_has_too_little_data():
    tools = PerUserTools()
    tools.coverage_by_user = {"U02": 20.0}
    events, st = await run(make(tools), "U01이랑 U02 비교해줘")
    rp = next(e for e in events if e["type"] == "replan")
    assert "U02" in rp["reason"] and rp["lookback_days"] == 14
    assert "이상이 없다고 판단할 수 없습니다" in st["report"] and "U02 데이터 커버리지 20%" in st["report"]


@pytest.mark.asyncio
async def test_single_user_requests_keep_the_old_shape():
    tools = PerUserTools()
    events, st = await run(make(tools), "U05 이상 점검해줘")
    assert st["users"] is None and st["user_ids"] == [] and "## 사용자 비교" not in st["report"]
    assert not any(":U05" in e.get("task_id", "") for e in events)
    assert all("user_id" not in a for a in st["anomalies"])

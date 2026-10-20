"""대화 기억(이름·나이)과 키워드 폴백 분류기 튜닝 테스트.

시나리오(표준 대화 기억 테스트): "제 이름은 민수입니다" → 다음 발화 "제 이름이 뭐였죠?" 에 이름으로 답한다.
기억은 체크포인터에 저장된 messages 에서 읽으므로 CHECKPOINTER=sqlite 에서는 서버 재시작 후에도 동작해야 한다.
협력: `tests/fakes.py`, `app.agent.{gates,graph,checkpointer}`.
"""
import dataclasses

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.agent import gates
from app.agent.checkpointer import make_checkpointer
from app.agent.deps import AgentDeps
from app.agent.fake_llm import FakeBackend
from app.agent.graph import build_graph
from app.agent.llm import NullLLM
from app.agent.nodes import _Route
from app.config import settings
from tests.fakes import CRITICAL, FakeLLM, FakeTools


async def say(graph, message, thread="m1"):
    cfg = {"configurable": {"thread_id": thread}, "recursion_limit": 40}
    events = [c async for c in graph.astream({"messages": [("user", message)]}, cfg, stream_mode="custom")]
    return events, (await graph.aget_state(cfg)).values


def make(llm, tools=None, saver=None):
    tools = tools or FakeTools()
    return build_graph(AgentDeps(llm, llm, tools.toolbox(), saver or InMemorySaver(), settings, "v2")), tools


@pytest.mark.asyncio
@pytest.mark.parametrize("llm", [NullLLM(), FakeBackend()])
async def test_name_is_remembered_within_thread_without_tools(llm):
    graph, tools = make(llm)
    _, first = await say(graph, "제 이름은 민수입니다")
    assert "민수님" in first["report"] and first["route"] == "simple"
    events, second = await say(graph, "제 이름이 뭐였죠?")
    assert "민수" in second["report"] and tools.summary_calls == [] and gates.DISCLAIMER in second["report"]
    assert not any(e["type"] == "plan_created" for e in events)


@pytest.mark.asyncio
async def test_unknown_memory_is_admitted_and_threads_do_not_leak():
    graph, _ = make(NullLLM())
    await say(graph, "제 이름은 민수입니다", thread="a")
    _, other = await say(graph, "제 이름이 뭐였죠?", thread="b")
    assert "민수" not in other["report"] and "말씀해 주지 않으셨" in other["report"]


@pytest.mark.asyncio
async def test_llm_answer_that_forgets_the_name_is_replaced_by_the_deterministic_answer():
    llm = FakeLLM(text={"chat": ["반갑습니다!", "음, 잘 모르겠어요."]})  # 두 번째 답이 이름을 빠뜨림
    graph, _ = make(llm)
    await say(graph, "제 이름은 민수입니다")
    _, st = await say(graph, "제 이름이 뭐였죠?")
    assert "민수님이라고 말씀하셨어요" in st["report"] and "잘 모르겠어요" not in st["report"]
    assert "chat" in llm.prompts and "- 제 이름은 민수입니다" in llm.prompts["chat"][1]  # 대화 기록이 프롬프트에 들어간다


@pytest.mark.asyncio
async def test_router_misclassification_cannot_send_name_chat_to_analysis():
    llm = FakeLLM(structured={"route": _Route(route="analysis", reasoning="점검 같음")}, text={"chat": "반갑습니다 민수님."})
    graph, tools = make(llm)
    events, st = await say(graph, "제 이름은 민수입니다")
    assert events[0]["route"] == "simple" and "코드 보정" in events[0]["reasoning"] and tools.summary_calls == []


@pytest.mark.asyncio
async def test_memory_survives_server_restart_with_sqlite_and_keeps_safety_state(tmp_path, monkeypatch):
    monkeypatch.setattr("app.agent.checkpointer.settings", dataclasses.replace(settings, sqlite_path=str(tmp_path / "mem.sqlite")))
    async with make_checkpointer("sqlite") as saver:  # 1차 프로세스
        graph, _ = make(NullLLM(), FakeTools([CRITICAL]), saver)
        await say(graph, "제 이름은 민수입니다", thread="restart")
        await say(graph, "U01 최근 30일 이상 점검해줘", thread="restart")
    async with make_checkpointer("sqlite") as saver:  # 재시작 후 새 인스턴스
        graph, tools = make(NullLLM(), FakeTools([CRITICAL]), saver)
        _, st = await say(graph, "제 이름이 뭐였죠?", thread="restart")
    assert "민수님이라고 말씀하셨어요" in st["report"] and tools.detect_calls == []
    assert gates.ESCALATION_MESSAGE in st["report"]  # 이름 대화여도 직전 분석의 critical 안내는 유지된다


# ---- 키워드 폴백 분류기 (회귀 #4): '괜찮'만으로 사용자 선택 안내로 가지 않는다 ---------------------------------
@pytest.mark.parametrize("msg,has_user,expected", [
    ("그럼 운동할 때는 괜찮아?", False, "simple"), ("수면 중 심박은 괜찮은 거야?", False, "simple"),
    ("내 심박 괜찮아?", False, "analysis"), ("오늘 괜찮았어?", False, "analysis"), ("U05 괜찮은지 봐줘", True, "analysis"),
    ("서맥이 뭐예요?", False, "simple"), ("U03 최근 30일 점검해줘", True, "analysis"),
])
def test_keyword_fallback_distinguishes_general_questions_from_own_data(msg, has_user, expected):
    assert gates.route_fallback(msg, False, None, has_user)[0] == expected


def test_personal_chat_detection_and_replies():
    assert gates.is_personal_chat("제 이름은 민수입니다") and gates.is_personal_chat("제 이름이 뭐였죠?") and gates.is_personal_chat("제 나이가 몇이었죠?")
    assert not gates.is_personal_chat("심박수 정상 범위가 어떻게 돼?") and not gates.is_personal_chat("U05 이름 점검")
    assert gates.personal_reply("제 나이가 몇이었죠?", ["제 나이는 40살입니다"]) == "40세라고 말씀하셨어요."
    assert gates.personal_reply("제 이름이 뭐였죠?", ["제 이름은 철수예요", "제 이름은 민수입니다"]) == "민수님이라고 말씀하셨어요."  # 가장 최근 값
    assert gates.personal_reply("안녕하세요", []) is None


def test_api_level_two_message_memory_test_survives_a_server_restart_with_sqlite(tmp_path, monkeypatch):
    """표준 대화 기억 시나리오("제 이름은 민수입니다" → "제 이름이 뭐였죠?")를 HTTP 로, 서버를 새로 띄운 뒤에도 (CHECKPOINTER=sqlite)."""
    from fastapi.testclient import TestClient

    from app.api.main import create_app

    monkeypatch.setattr("app.agent.checkpointer.settings", dataclasses.replace(settings, sqlite_path=str(tmp_path / "api.sqlite")))
    body = {"thread_id": "demo-name"}
    with TestClient(create_app(FakeTools().toolbox(), NullLLM(), "sqlite")) as c1:  # 1번째 서버
        r1 = c1.post("/chat", json={**body, "message": "제 이름은 민수입니다"}).json()
    assert r1["route"] == "simple" and "민수님" in r1["report"]
    with TestClient(create_app(FakeTools().toolbox(), NullLLM(), "sqlite")) as c2:  # 서버 재시작(새 프로세스 모사)
        r2 = c2.post("/chat", json={**body, "message": "제 이름이 뭐였죠?"}).json()
        r3 = c2.post("/chat", json={"thread_id": "someone-else", "message": "제 이름이 뭐였죠?"}).json()
    assert "민수" in r2["report"] and "민수" not in r3["report"]

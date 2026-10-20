"""Human-in-the-loop: 에스컬레이션 시 보호자 알림(시뮬레이션) 발송 전 사람 승인 (LangGraph interrupt / Command(resume)).

핵심 불변식: 에스컬레이션 안내문은 승인과 무관하게 *항상* 리포트에 있다. 승인은 부가 조치(알림)만 좌우한다.
협력: `tests/fakes.py`, `app.agent.{graph,deps,checkpointer}`, `app.api.main`.
"""
import dataclasses
import json

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.agent import gates
from app.agent.checkpointer import make_checkpointer
from app.agent.deps import AgentDeps
from app.agent.graph import build_graph
from app.agent.llm import NullLLM
from app.api.main import create_app
from app.config import settings
from tests.fakes import CRITICAL, WARNING, FakeTools

ON = dataclasses.replace(settings, hitl_enabled=True)


def deps(tools, saver=None, st=ON):
    return AgentDeps(NullLLM(), NullLLM(), tools.toolbox(), saver or InMemorySaver(), st, "v2")


async def go(graph, message, thread="h1"):
    cfg = {"configurable": {"thread_id": thread}, "recursion_limit": 40}
    events = [c async for c in graph.astream({"messages": [("user", message)]}, cfg, stream_mode="custom")]
    return events, cfg


@pytest.mark.asyncio
@pytest.mark.parametrize("approved,expected", [(True, "보호자 알림 발송됨(시뮬레이션)"), (False, "보호자 알림을 발송하지 않았습니다")])
async def test_escalation_pauses_for_approval_and_resume_decides_only_the_notification(approved, expected):
    graph = build_graph(deps(FakeTools([CRITICAL])))
    events, cfg = await go(graph, "U01 최근 30일 이상 점검해줘")
    snap = await graph.aget_state(cfg)
    assert snap.next == ("notify_guardian",) and snap.tasks[0].interrupts[0].value["reason"].startswith("critical")
    paused = snap.values["report"]
    assert gates.ESCALATION_MESSAGE in paused and "보호자 알림" not in paused  # 승인 전에도 안내문은 이미 있다
    preview = snap.tasks[0].interrupts[0].value["message_preview"]
    assert "U01" in preview and "시뮬레이션" in preview and "135.0" not in preview  # 가명 ID + 위험 요약만, 수치 시계열 없음
    final = await graph.ainvoke(Command(resume={"approved": approved}), cfg)
    assert expected in final["report"].rstrip().splitlines()[-1] and gates.ESCALATION_MESSAGE in final["report"]  # 결정과 무관하게 안내문 유지
    assert final["approval"]["status"] == ("approved" if approved else "denied") and final["approval"]["simulated"] is True
    assert (await graph.aget_state(cfg)).next == ()  # 그래프 종료


@pytest.mark.asyncio
async def test_no_interrupt_without_escalation_or_when_disabled_or_without_checkpointer():
    g1 = build_graph(deps(FakeTools([WARNING])))  # critical 없음
    _, cfg = await go(g1, "U01 이상 점검해줘")
    assert (await g1.aget_state(cfg)).next == ()
    g2 = build_graph(deps(FakeTools([CRITICAL]), st=settings))  # 기능 꺼짐(기본)
    _, cfg = await go(g2, "U01 이상 점검해줘")
    assert (await g2.aget_state(cfg)).next == () and settings.hitl_enabled is False
    g3 = build_graph(AgentDeps(NullLLM(), NullLLM(), FakeTools([CRITICAL]).toolbox(), None, ON, "v2"))  # 체크포인터 없으면 멈출 수 없다
    out = await g3.ainvoke({"messages": [("user", "U01 이상 점검해줘")]})
    assert "__interrupt__" not in out and gates.ESCALATION_MESSAGE in out["report"]


@pytest.mark.asyncio
async def test_followup_questions_do_not_trigger_a_second_approval():
    graph = build_graph(deps(FakeTools([CRITICAL])))
    _, cfg = await go(graph, "U01 최근 30일 이상 점검해줘")
    await graph.ainvoke(Command(resume={"approved": True}), cfg)
    await go(graph, "그럼 운동 중엔 괜찮은 거야?")
    snap = await graph.aget_state(cfg)
    assert snap.next == () and gates.ESCALATION_MESSAGE in snap.values["report"]


@pytest.mark.asyncio
async def test_pending_approval_survives_restart_with_sqlite_and_can_be_resumed(tmp_path, monkeypatch):
    monkeypatch.setattr("app.agent.checkpointer.settings", dataclasses.replace(settings, sqlite_path=str(tmp_path / "h.sqlite")))
    async with make_checkpointer("sqlite") as saver:  # 1차 프로세스: 승인 대기에서 종료
        _, cfg = await go(build_graph(deps(FakeTools([CRITICAL]), saver)), "U01 최근 30일 이상 점검해줘", thread="persist")
    async with make_checkpointer("sqlite") as saver:  # 재시작 후 새 인스턴스에서 재개
        graph = build_graph(deps(FakeTools([CRITICAL]), saver))
        assert (await graph.aget_state(cfg)).next == ("notify_guardian",)
        final = await graph.ainvoke(Command(resume={"approved": True}), cfg)
    assert final["report"].rstrip().endswith("보호자 알림 발송됨(시뮬레이션)")


@pytest.mark.asyncio
async def test_comparison_preview_names_only_the_critical_users():
    class Two(FakeTools):
        async def _detect(self, uid, days, end):
            an = [dict(CRITICAL)] if uid == "U02" else []
            return {"user_id": uid, "window": {}, "anomalies": an, "counts": {"resting_tachycardia": len(an)} if an else {}}

    graph = build_graph(deps(Two()))
    _, cfg = await go(graph, "U01이랑 U02 비교해줘")
    value = (await graph.aget_state(cfg)).tasks[0].interrupts[0].value
    assert "U02" in value["message_preview"] and "U01" not in value["message_preview"]


# ---- API ------------------------------------------------------------------------------------------
def client(tools=None, st=ON):
    return TestClient(create_app(deps=deps(tools or FakeTools([CRITICAL]), st=st)))


def sse(resp):
    out, ev = [], None
    for line in resp.read().decode().splitlines():
        if line.startswith("event: "):
            ev = line[7:]
        elif line.startswith("data: "):
            out.append(json.loads(line[6:]))
            assert out[-1]["type"] == ev
    return out


def test_api_stream_emits_approval_required_before_done_and_approve_resumes():
    with client() as c:
        with c.stream("POST", "/chat/stream", json={"message": "U01 최근 30일 이상 점검해줘", "thread_id": "ap1"}) as r:
            frames = sse(r)
        types = [f["type"] for f in frames]
        assert types[-2:] == ["approval_required", "done"]
        req = frames[-2]
        assert req["thread_id"] == "ap1" and req["reason"].startswith("critical") and "시뮬레이션" in req["message_preview"]
        assert gates.ESCALATION_MESSAGE in frames[-1]["report"] and frames[-1]["escalation"] is True  # done 의 리포트에 안내문 (승인 전)
        aid = req["approval_id"]
        ok = c.post("/chat/approve", json={"thread_id": "ap1", "approved": True, "approval_id": aid}).json()
        assert ok["notification"] == "sent_simulated" and ok["report"].rstrip().endswith("보호자 알림 발송됨(시뮬레이션)") and ok["approval"]["status"] == "approved"
        assert c.post("/chat/approve", json={"thread_id": "ap1", "approved": True, "approval_id": aid}).status_code == 409  # 이미 처리됨
        assert c.post("/chat/approve", json={"thread_id": "nope", "approved": False, "approval_id": aid}).status_code == 409
        assert c.post("/chat/approve", json={"thread_id": "../x", "approved": False, "approval_id": aid}).status_code == 422
        assert c.post("/chat/approve", json={"thread_id": "ap1", "approved": True}).status_code == 422   # approval_id 필수


def test_api_deny_keeps_notice_and_marks_not_sent():
    with client() as c:
        aid = c.post("/chat", json={"message": "U01 이상 점검해줘", "thread_id": "ap2"}).json()["approval_required"]["approval_id"]
        out = c.post("/chat/approve", json={"thread_id": "ap2", "approved": False, "approval_id": aid}).json()
    assert out["notification"] == "not_sent" and "발송하지 않았습니다" in out["report"] and gates.ESCALATION_MESSAGE in out["report"]


def test_api_chat_json_reports_pending_approval_only_when_pending():
    with client() as c:
        pend = c.post("/chat", json={"message": "U01 이상 점검해줘"}).json()
        assert pend["approval_required"]["reason"].startswith("critical") and pend["escalation"] is True
    with client(FakeTools([WARNING])) as c:
        assert "approval_required" not in c.post("/chat", json={"message": "U01 이상 점검해줘"}).json()  # 기존 응답 형태 유지
    with client(st=settings) as c:
        assert "approval_required" not in c.post("/chat", json={"message": "U01 이상 점검해줘"}).json()  # 기능 꺼짐


def test_new_message_on_a_pending_thread_auto_denies_and_never_blocks_the_conversation():
    with client() as c:
        aid = c.post("/chat", json={"message": "U01 최근 30일 이상 점검해줘", "thread_id": "ap3"}).json()["approval_required"]["approval_id"]
        r = c.post("/chat", json={"message": "심박수 정상 범위가 어떻게 돼?", "thread_id": "ap3"}).json()  # 승인 없이 다음 질문
        assert r["route"] == "simple" and gates.ESCALATION_MESSAGE in r["report"]  # 직전 critical 안내는 후속 답변에도 유지
        assert c.post("/chat/approve", json={"thread_id": "ap3", "approved": True, "approval_id": aid}).status_code == 409  # 이미 거절로 정리됨


@pytest.mark.asyncio
async def test_previous_turns_approval_does_not_linger_in_the_next_pending_turn():
    """승인된 턴 다음의 재분석이 다시 승인 대기일 때 state.approval 은 이전 결과가 아니라 비어 있어야 한다."""
    graph = build_graph(deps(FakeTools([CRITICAL])))
    _, cfg = await go(graph, "U01 최근 30일 이상 점검해줘", thread="stale")
    first = await graph.ainvoke(Command(resume={"approved": True}), cfg)
    assert first["approval"]["status"] == "approved"
    cfg2 = {"configurable": {"thread_id": "stale"}, "recursion_limit": 40}
    _ = [c async for c in graph.astream({"messages": [("user", "U02 최근 14일 다시 점검해줘")]}, cfg2, stream_mode="custom")]
    snap = await graph.aget_state(cfg2)
    assert snap.next == ("notify_guardian",) and snap.values.get("approval") is None  # 대기 중인 새 요청, 이전 승인 흔적 없음
    final = await graph.ainvoke(Command(resume={"approved": False}), cfg2)
    assert final["approval"]["status"] == "denied"


def test_api_approval_id_rejects_a_decision_made_on_a_stale_card():
    """오래된 카드의 결정이 같은 대화의 *새* 승인 요청에 적용되면 안 된다."""
    with client() as c:
        old = c.post("/chat", json={"message": "U01 최근 30일 이상 점검해줘", "thread_id": "stale1"}).json()["approval_required"]
        assert old["approval_id"]
        # 새 메시지 → 옛 요청은 자동 거절로 정리되고, 같은 대화에서 다시 점검하면 *새* 승인 요청이 생긴다
        new = c.post("/chat", json={"message": "U02 최근 14일 이상 점검해줘", "thread_id": "stale1"}).json()["approval_required"]
        assert new["approval_id"] != old["approval_id"]
        r = c.post("/chat/approve", json={"thread_id": "stale1", "approved": True, "approval_id": old["approval_id"]})
        assert r.status_code == 409
        assert c.get("/version").status_code == 200   # 거절된 요청은 상태를 바꾸지 않았다: 새 요청은 여전히 대기 중
        ok = c.post("/chat/approve", json={"thread_id": "stale1", "approved": False, "approval_id": new["approval_id"]})
        assert ok.status_code == 200 and ok.json()["notification"] == "not_sent"


def test_api_stream_event_carries_the_same_approval_id_as_the_pending_request():
    with client() as c:
        with c.stream("POST", "/chat/stream", json={"message": "U01 최근 30일 이상 점검해줘", "thread_id": "ap9"}) as r:
            frames = sse(r)
        aid = frames[-2]["approval_id"]
        assert aid and c.post("/chat/approve", json={"thread_id": "ap9", "approved": True, "approval_id": "wrong"}).status_code == 409
        assert c.post("/chat/approve", json={"thread_id": "ap9", "approved": True, "approval_id": aid}).status_code == 200

"""API 테스트: fake 도구 주입(빠름) + 실제 MCP subprocess(LLM 비활성) 각 1세트."""
import dataclasses
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.agent.llm import NullLLM
from app.api.main import create_app
from app.config import ROOT, settings
from tests.fakes import CRITICAL, FakeTools, assemble


def parse_sse(text: str) -> list[dict]:
    """SSE 본문을 dict 리스트로 파싱하며 모든 프레임에서 event 이름 == data.type 계약을 검사한다."""
    out = []
    for frame in text.strip().split("\n\n"):
        lines = dict(l.split(": ", 1) for l in frame.splitlines())
        data = json.loads(lines["data"])
        assert lines["event"] == data["type"]  # 계약: event 이름 == data.type
        out.append(data)
    return out


@pytest.fixture
def client():
    """fake 도구·NullLLM·메모리 체크포인터로 구성한 TestClient (lifespan 실행 포함)."""
    with TestClient(create_app(FakeTools([CRITICAL]).toolbox(), NullLLM(), "memory")) as c:
        yield c


def test_health_and_data_endpoints(client):
    """/health, /users, /series, /anomalies, /summary 가 도구 결과를 그대로 전달한다."""
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["mcp"] is True and health["llm"] is False and health["ollama"]["enabled"] is False
    assert [u["user_id"] for u in client.get("/users").json()] == ["U01", "U02", "U03"]
    assert client.get("/users/U01/series?lookback_days=1").json()["points"]
    assert client.get("/users/U01/anomalies?lookback_days=7").json()["anomalies"][0]["severity"] == "critical"
    body = client.get("/users/U01/summary?lookback_days=7").json()
    assert body["user_id"] == "U01" and body["summary"]["rest_minutes"] == 580 and body["daily_resting_hr"]


def test_cors_allows_dev_frontend(client):
    """프런트 개발 서버(5173) Origin 의 CORS preflight 를 허용한다."""
    r = client.options("/chat", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_chat_validation_errors(client):
    """빈 메시지·부적절한 thread_id 는 422."""
    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert client.post("/chat", json={"message": "hi", "thread_id": "../etc"}).status_code == 422


def test_chat_json(client):
    """비스트리밍 /chat 응답 형식과 에스컬레이션 전파."""
    r = client.post("/chat", json={"message": "U01 이상 점검해줘"}).json()
    assert r["route"] == "analysis" and r["escalation"] is True and r["validation"]["passed"]
    assert r["thread_id"] and r["anomalies"][0]["type"] == "resting_tachycardia"


def test_chat_stream_event_sequence_and_multi_turn(client):
    """스트림 이벤트 순서, done.report == 토큰 조립본, 후속 질문의 simple 경로·에스컬레이션 유지, 명시 재점검의 재분석."""
    with client.stream("POST", "/chat/stream", json={"message": "U01 최근 7일 이상 점검해줘", "thread_id": "s1"}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(r.read().decode())
    t = [e["type"] for e in events]
    for a, b in zip(["route_decided", "plan_created", "gate_applied", "anomalies_found", "safety_escalation", "token", "validation", "done"],
                    ["plan_created", "gate_applied", "anomalies_found", "safety_escalation", "token", "validation", "done", None]):
        if b:
            assert t.index(a) < t.index(b), (a, b, t)
    done = events[-1]
    assert done["type"] == "done" and done["thread_id"] == "s1" and done["escalation"] is True
    assert done["report"] == assemble(events)

    # 같은 thread 후속 질문: 사용자·기억 유지, 재조회 없이 simple 경로
    with client.stream("POST", "/chat/stream", json={"message": "그럼 운동 중엔 괜찮은 거야?", "thread_id": "s1"}) as r:
        follow = parse_sse(r.read().decode())
    assert follow[0]["type"] == "route_decided" and follow[0]["route"] == "simple" and follow[-1]["type"] == "done"
    assert "plan_created" not in [e["type"] for e in follow]
    assert follow[-1]["escalation"] is True and "119" in follow[-1]["report"]  # 후속 답변에도 에스컬레이션 유지

    with client.stream("POST", "/chat/stream", json={"message": "다시 30일 점검해줘", "thread_id": "s1"}) as r:
        again = parse_sse(r.read().decode())
    assert "plan_created" in [e["type"] for e in again] and again[-1]["escalation"] is True  # 명시 요청은 재분석


def test_guideline_load_failure_is_isolated():
    """가이드라인 로드 실패는 격리되어 기본 면책으로 폴백하고 스트림은 done 으로 끝난다."""
    class Boom(FakeTools):
        """가이드라인 리소스 조회가 항상 실패하는 도구."""
        async def _resource(self, uri):
            raise ValueError("boom")  # 가이드라인 로드 실패는 격리되어 스트림이 계속돼야 함

    with TestClient(create_app(Boom().toolbox(), NullLLM(), "memory")) as c:
        with c.stream("POST", "/chat/stream", json={"message": "U01 이상 점검"}) as r:
            events = parse_sse(r.read().decode())
    assert events[-1]["type"] == "done"  # 기본 면책 문구로 폴백


def test_mcp_unavailable_returns_503(monkeypatch):
    """MCP 연결 실패 시 앱은 뜨고 health 에 mcp=false, 데이터/채팅 엔드포인트는 503."""
    async def fail(stack):
        """MCP 연결 실패를 모사한다."""
        raise OSError("no server")

    monkeypatch.setattr("app.api.main.connect_mcp", fail)
    with TestClient(create_app(None, NullLLM(), "memory")) as c:
        assert c.get("/health").json()["mcp"] is False
        assert c.get("/users").status_code == 503
        assert c.post("/chat/stream", json={"message": "hi"}).status_code == 503


@pytest.mark.skipif(not (Path(ROOT) / "app" / "mcp_server" / "server.py").exists(), reason="MCP 서버 미구현")
def test_real_mcp_subprocess_end_to_end():
    """실제 stdio MCP subprocess + 실제 데이터, LLM 비활성."""
    with TestClient(create_app(None, NullLLM(), "memory")) as c:
        assert c.get("/health").json()["mcp"] is True
        users = c.get("/users").json()
        assert users and users[0]["user_id"].startswith("U")
        uid = users[0]["user_id"]
        with c.stream("POST", "/chat/stream", json={"message": f"{uid} 최근 7일 심박 이상 점검해줘"}) as r:
            events = parse_sse(r.read().decode())
        t = [e["type"] for e in events]
        assert t[0] == "route_decided" and t[-1] == "done" and "task_failed" not in t, t
        assert "anomalies_found" in t and events[-1]["report"]


def test_client_disconnect_cancels_graph_run():
    """SSE 클라이언트가 끊으면 진행 중이던 그래프(도구 호출)가 CancelledError 로 중단되는지 실제 uvicorn 서버로 확인."""
    import asyncio
    import socket
    import threading
    import time

    import httpx
    import uvicorn

    state = {"started": threading.Event(), "cancelled": threading.Event()}

    class Slow(FakeTools):
        """detect 가 오래 걸리고 취소 여부를 기록하는 도구."""
        async def _detect(self, uid, days, end):
            state["started"].set()
            try:
                await asyncio.sleep(30)  # 오래 걸리는 도구 호출
            except asyncio.CancelledError:
                state["cancelled"].set()
                raise

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(Slow().toolbox(), NullLLM(), "memory"), port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    try:
        while not server.started:
            time.sleep(0.05)
        with httpx.stream("POST", f"http://127.0.0.1:{port}/chat/stream", json={"message": "U01 이상 점검"}, timeout=10) as r:
            for line in r.iter_lines():
                if state["started"].is_set():
                    break  # 도구 호출이 시작된 뒤 클라이언트가 연결을 끊는다
        assert state["cancelled"].wait(5), "클라이언트 이탈 후에도 그래프 실행이 계속됨"
    finally:
        server.should_exit = True
        th.join(5)


def test_health_reports_ollama_reachability(monkeypatch):
    """llm 플래그는 '켜져 있음'이 아니라 '실제 사용 가능'을 뜻한다: 도달 불가/모델 누락이면 false, 결과는 짧게 캐시."""
    from app.agent.llm import OllamaLLM

    monkeypatch.setattr("app.agent.llm.settings", dataclasses.replace(settings, ollama_base_url="http://127.0.0.1:1"))
    llm = OllamaLLM()
    with TestClient(create_app(FakeTools().toolbox(), llm, "memory")) as c:
        h = c.get("/health").json()
        assert h["llm"] is False and h["ollama"]["reachable"] is False and settings.fast_model in h["ollama"]["missing"]
        cached = llm._health
        c.get("/health")
        assert llm._health is cached  # TTL 안에서는 재조회하지 않는다


def test_chat_and_stream_report_anomalies_in_same_order():
    """회귀: /chat 의 anomalies 순서 == 스트림 anomalies_found 순서 (심각도 desc, 시작 시각)."""
    from tests.fakes import WARNING

    early = {**WARNING, "anomaly_id": "A0", "start": "2016-04-01T01:00:00"}
    with TestClient(create_app(FakeTools([WARNING, early, CRITICAL]).toolbox(), NullLLM(), "memory")) as c:
        rest = [a["anomaly_id"] for a in c.post("/chat", json={"message": "U01 이상 점검해줘"}).json()["anomalies"]]
        with c.stream("POST", "/chat/stream", json={"message": "U01 이상 점검해줘"}) as r:
            events = parse_sse(r.read().decode())
    streamed = [a["anomaly_id"] for a in next(e for e in events if e["type"] == "anomalies_found")["anomalies"]]
    assert rest == streamed == ["A2", "A0", "A1"]


def test_unknown_user_chat_is_friendly(client):
    r = client.post("/chat", json={"message": "U99 이상 점검해줘"}).json()
    assert r["route"] == "simple" and "찾을 수 없습니다" in r["report"] and "잠시 후" not in r["report"]


def test_version_and_health_report_runtime_configuration(client):
    """GET /version: UI 헤더용 실행 구성(비밀 없음). /health 에도 버전·백엔드가 들어간다."""
    v = client.get("/version").json()
    assert {"supervisor_version", "llm_backend", "fast_model", "quality_model", "checkpointer", "git_commit"} <= set(v)   # 필수 키 (새 키가 늘어도 깨지지 않게 부분집합)
    assert isinstance(v["datasets"], list) and isinstance(v["allow_v1_baseline"], bool)
    assert v["supervisor_version"] == "v2" and v["checkpointer"] == "memory" and v["git_commit"]
    h = client.get("/health").json()
    assert h["supervisor_version"] == "v2" and "llm_backend" in h


def test_each_request_logs_llm_usage(caplog):
    from app.agent.fake_llm import FakeBackend

    from tests.fakes import FakeTools as FT

    with caplog.at_level("INFO", logger="vital-sentinel.api"):
        with TestClient(create_app(FT([CRITICAL]).toolbox(), FakeBackend(), "memory")) as c:
            c.post("/chat", json={"message": "U01 이상 점검해줘"})
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("usage thread="))
    assert "llm_calls=" in line and "tokens=" in line and "cost_usd=" in line


def test_run_id_is_returned_and_feedback_is_forwarded_only_with_a_langsmith_key(monkeypatch, client):
    import uuid

    r = client.post("/chat", json={"message": "U01 이상 점검해줘"}).json()
    assert uuid.UUID(r["run_id"])  # /chat 응답에 실행 ID
    with client.stream("POST", "/chat/stream", json={"message": "U01 이상 점검해줘"}) as resp:
        done = [json.loads(l[6:]) for l in resp.read().decode().splitlines() if l.startswith("data: ")][-1]
    assert uuid.UUID(done["run_id"]) and done["run_id"] != r["run_id"]
    body = {"run_id": r["run_id"], "score": 1, "comment": "도움이 됐어요"}
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert client.post("/feedback", json=body).json() == {"recorded": False, "backend": "none"}  # 키 없음: 조용히 no-op

    calls = []

    class FakeClient:
        def create_feedback(self, **kw):
            calls.append(kw)

    monkeypatch.setattr("app.api.main._langsmith_client", lambda: FakeClient())
    assert client.post("/feedback", json=body).json() == {"recorded": True, "backend": "langsmith"}
    assert calls[0]["key"] == "user_thumbs" and calls[0]["score"] == 1 and str(calls[0]["run_id"]) == r["run_id"] and calls[0]["comment"] == "도움이 됐어요"
    # 입력 검증: 점수 범위, run_id 형식, 의견 길이
    assert client.post("/feedback", json={**body, "score": 2}).status_code == 422
    assert client.post("/feedback", json={**body, "run_id": "nope"}).status_code == 422
    assert client.post("/feedback", json={**body, "comment": "x" * 501}).status_code == 422

    class Boom:
        def create_feedback(self, **kw):
            raise ConnectionError("langsmith down")

    monkeypatch.setattr("app.api.main._langsmith_client", lambda: Boom())
    assert client.post("/feedback", json=body).json() == {"recorded": False, "backend": "langsmith"}  # 실패해도 서비스 영향 없음

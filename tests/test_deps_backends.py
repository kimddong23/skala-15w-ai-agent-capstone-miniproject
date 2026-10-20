"""DI 번들(`AgentDeps`), LLM 백엔드(fake/ollama/openai), 사용량 계량, v1 기준선, RAG 배선 테스트.

원칙: 실제 Ollama·OpenAI 를 호출하지 않는다 (openai 는 생성만 검증 — 네트워크 호출 없음). fake 백엔드로 LLM 경로 전체를 결정론적으로 돈다.
협력: `tests/fakes.py`(FakeTools·FakeLLM), `app.agent.{deps,llm,fake_llm,openai_llm,usage,supervisor_v1,graph}`, `app.api.main`.
"""
import dataclasses
import json

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from app.agent import gates
from app.agent.deps import AgentDeps
from app.agent.fake_llm import FakeBackend
from app.agent.graph import build_graph
from app.agent.llm import NullLLM, OllamaLLM, TieredLLM, make_llm
from app.agent.usage import cost_usd, current_summary, record, usage_scope
from app.api.main import create_app
from app.config import settings
from tests.fakes import CRITICAL, FakeLLM, FakeTools, assemble


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr("app.agent.retry.random.uniform", lambda a, b: 0)


async def run(graph, message, thread="t1", user_id=None):
    cfg = {"configurable": {"thread_id": thread}, "recursion_limit": 40}
    inp = {"messages": [("user", message)]}
    if user_id:
        inp["user_id"] = user_id
    with usage_scope() as meter:
        events = [c async for c in graph.astream(inp, cfg, stream_mode="custom")]
    return events, (await graph.aget_state(cfg)).values, meter.summary()


def deps_of(tools, llm, version="v2"):
    return AgentDeps(llm, llm, tools.toolbox(), InMemorySaver(), settings, version)


# ---- AgentDeps ------------------------------------------------------------------------------------
def test_deps_rejects_unknown_version_and_bundles_parts():
    with pytest.raises(ValueError):
        AgentDeps(NullLLM(), NullLLM(), FakeTools().toolbox(), version="v3")
    d = AgentDeps.from_parts(FakeTools().toolbox(), InMemorySaver(), NullLLM())
    assert d.version == settings.supervisor_version and isinstance(d.llm, TieredLLM)


@pytest.mark.asyncio
async def test_tiered_llm_dispatches_to_the_injected_tier_objects():
    """fast/quality 를 서로 다른 객체로 주입하면 티어 문자열에 따라 각자에게 위임된다 (하이브리드 백엔드의 기반)."""
    fast, quality = FakeLLM(text={"x": "F"}), FakeLLM(text={"x": "Q"})
    llm = AgentDeps(fast, quality, FakeTools().toolbox()).llm
    assert await llm.text("fast", "x", "s", "u") == "F" and await llm.text("quality", "x", "s", "u") == "Q"
    assert llm.enabled is True and TieredLLM(NullLLM(), quality).enabled is False
    assert (await TieredLLM(FakeBackend(), FakeBackend()).health())["backend"] == "fake"


# ---- 백엔드 팩토리 ---------------------------------------------------------------------------------
def test_make_llm_selects_backend_and_never_calls_remote(monkeypatch):
    s = lambda **kw: monkeypatch.setattr("app.agent.llm.settings", dataclasses.replace(settings, **kw))  # noqa: E731
    s(llm_backend="fake", llm_enabled=True)
    assert isinstance(make_llm(), FakeBackend)
    s(llm_backend="ollama", llm_enabled=True)
    assert isinstance(make_llm(), OllamaLLM)  # 생성만 한다(연결은 호출 시점)
    s(llm_backend="fake", llm_enabled=False)
    assert isinstance(make_llm(), NullLLM)  # LLM_ENABLED=false 는 어떤 백엔드보다 우선
    s(llm_backend="nope", llm_enabled=True)
    with pytest.raises(ValueError):
        make_llm()


def test_openai_backend_requires_key_and_uses_default_models(monkeypatch):
    from app.agent.openai_llm import OpenAILLM

    monkeypatch.delenv("OPENAI_CONFIRM", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    with pytest.raises(RuntimeError, match="OPENAI_CONFIRM"):  # 확인 없이는 키가 있어도 시작하지 않는다 (공유 키 보호)
        OpenAILLM()
    monkeypatch.setenv("OPENAI_CONFIRM", "yes")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAILLM()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")  # 생성만 — 네트워크 호출 없음
    llm = OpenAILLM()
    assert llm.model_names == {"fast": "gpt-4o-mini", "quality": "gpt-4o"}
    assert llm._extra("quality", 100) == {"max_tokens": 100} and llm._extra("fast", None) == {}


# ---- 사용량 계량 -----------------------------------------------------------------------------------
def test_usage_scope_aggregates_calls_and_is_noop_outside():
    import time

    record("outside", "fast", "m", 1, 1, time.perf_counter())  # 범위 밖: 조용히 무시
    assert current_summary() is None
    with usage_scope() as m:
        record("route", "fast", "gpt-4o-mini", 1000, 100, time.perf_counter())
        record("writing", "quality", "gpt-4o", 2000, 500, time.perf_counter(), estimated=True)
    s = m.summary()
    assert s["llm_calls"] == 2 and s["total_tokens"] == 3600 and s["by_tier"]["quality"] == {"calls": 1, "tokens": 2500}
    assert s["cost_usd"] == pytest.approx(cost_usd("gpt-4o-mini", 1000, 100) + cost_usd("gpt-4o", 2000, 500), abs=1e-6) and s["cost_usd"] > 0
    assert cost_usd("qwen3:8b", 10_000, 10_000) == 0  # 로컬 모델은 비용 0


# ---- fake 백엔드: LLM 경로 전체가 결정론적으로 ------------------------------------------------------
@pytest.mark.asyncio
async def test_fake_backend_runs_full_pipeline_deterministically_with_usage():
    reports = []
    for _ in range(2):
        tools = FakeTools([CRITICAL])
        graph = build_graph(deps_of(tools, FakeBackend()))
        events, st, usage = await run(graph, "U01 최근 30일 이상 점검해줘")
        reports.append(st["report"])
        assert gates.ESCALATION_MESSAGE in st["report"] and st["validation"]["passed"] and not st["validation"]["degraded"]
        assert [e["passed"] for e in events if e["type"] == "validation"] == [True]  # 재시도 없이 첫 검증 통과
        assert assemble(events) == st["report"] and "135.0 bpm" in st["report"]
        assert usage["llm_calls"] >= 4 and set(usage["by_tier"]) == {"fast", "quality"}  # route·plan(fast) + analysis(fast)·writing(quality)
        assert st["usage"]["llm_calls"] == usage["llm_calls"]  # 최종 노드가 상태에도 기록
    assert reports[0] == reports[1]  # 결정론


@pytest.mark.asyncio
async def test_fake_backend_simple_and_followup_paths():
    graph = build_graph(deps_of(FakeTools([CRITICAL]), FakeBackend()))
    events, st, usage = await run(graph, "심박수 정상 범위가 어떻게 돼?")
    assert events[0]["route"] == "simple" and st["validation"]["passed"] and usage["llm_calls"] >= 2
    await run(graph, "U01 최근 30일 이상 점검해줘", thread="f")
    events, st, _ = await run(graph, "그럼 운동 중엔 괜찮은 거야?", thread="f")
    assert events[0]["route"] == "simple" and gates.ESCALATION_MESSAGE in st["report"]


def test_api_exposes_usage_in_chat_and_done_event():
    with TestClient(create_app(deps=deps_of(FakeTools([CRITICAL]), FakeBackend()))) as c:
        r = c.post("/chat", json={"message": "U01 이상 점검해줘"}).json()
        assert r["usage"]["llm_calls"] >= 4 and r["usage"]["total_tokens"] > 0
        with c.stream("POST", "/chat/stream", json={"message": "U01 이상 점검해줘"}) as resp:
            frames = [json.loads(l[6:]) for l in resp.read().decode().splitlines() if l.startswith("data: ")]
        assert frames[-1]["type"] == "done" and frames[-1]["usage"]["llm_calls"] >= 4
        assert c.get("/health").json()["ollama"]["backend"] == "fake"


# ---- v1 기준선 ------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_v1_is_a_linear_baseline_without_routing_gates_safety_or_validation():
    tools = FakeTools([CRITICAL])
    graph = build_graph(deps_of(tools, FakeBackend(), version="v1"))
    # 인사말도 조사→작성 경로를 탄다 (라우팅 없음). 사용자가 없으면 조회만 건너뛰고 LLM 이 데이터 없이 답한다.
    events, st, usage = await run(graph, "안녕하세요")
    assert events[0]["type"] == "route_decided" and events[0]["route"] == "analysis" and "v1" in events[0]["reasoning"]
    assert tools.summary_calls == [] and all(e["type"] != "validation" for e in events)
    assert gates.DISCLAIMER not in st["report"] and set(usage["by_tier"]) == {"quality"}  # 고정 면책 없음, 모델은 quality 하나
    assert [c["name"] for c in usage["calls"]] == ["v1_writing"]
    # critical 이 있어도 안전 게이트가 없다: 플래그·고정 안내문 모두 없음 (LLM 이 쓰면 있고 안 쓰면 없다)
    events, st, _ = await run(graph, "U01 최근 30일 이상 점검해줘", thread="v1b")
    assert any(e["type"] == "anomalies_found" for e in events) and st["escalation"] is False
    assert gates.ESCALATION_MESSAGE not in st["report"] and st["validation"]["note"].startswith("v1")
    assert "safety_escalation" not in [e["type"] for e in events] and "plan_created" not in [e["type"] for e in events]


@pytest.mark.asyncio
async def test_v1_has_no_template_fallback_and_follows_the_toggle(monkeypatch):
    graph = build_graph(deps_of(FakeTools(), NullLLM(), version="v1"))
    _, st, _ = await run(graph, "U01 이상 점검해줘")
    assert st["report"] == "응답을 생성하지 못했습니다."  # v2 와 달리 템플릿 대체가 없다
    monkeypatch.setattr("app.agent.deps.default_settings", dataclasses.replace(settings, supervisor_version="v1"))
    toggled = build_graph(FakeTools().toolbox(), InMemorySaver(), NullLLM())  # 환경변수 토글로 v1 선택
    assert set(toggled.get_graph().nodes) == {"__start__", "route", "research", "writing", "__end__"}
    assert "validator" in build_graph(FakeTools().toolbox(), InMemorySaver(), NullLLM(), version="v2").get_graph().nodes


def test_v1_streams_the_same_sse_contract_through_the_api():
    with TestClient(create_app(deps=deps_of(FakeTools([CRITICAL]), FakeBackend()), version="v1")) as c:
        with c.stream("POST", "/chat/stream", json={"message": "U01 이상 점검해줘"}) as resp:
            frames = [json.loads(l[6:]) for l in resp.read().decode().splitlines() if l.startswith("data: ")]
    types = [f["type"] for f in frames]
    assert types[0] == "route_decided" and types[-1] == "done" and "token" in types and "anomalies_found" in types
    assert frames[-1]["escalation"] is False


# ---- RAG 배선 -------------------------------------------------------------------------------------
HITS = [
    {"doc_id": "G07", "title": "카페인과 심박", "heading": "영향", "text": "카페인은 일시적으로 심박을 올릴 수 있습니다. 평소보다 많이 마셨다면 줄여 보세요.", "score": 0.81, "source": "G07"},
    {"doc_id": "G07", "title": "카페인과 심박", "heading": "주의", "text": "섭취 후 2시간 안에 영향이 큽니다.", "score": 0.7, "source": "G07"},
]


@pytest.mark.asyncio
async def test_rag_hits_are_prompted_cited_by_code_and_deduplicated():
    tools = FakeTools()
    tools.search_hits = HITS
    llm = FakeLLM(text={"simple_answer": "카페인은 일시적으로 심박을 올릴 수 있습니다.\n※ 출처: 지어낸 문서(X99)"})  # LLM 이 출처를 스스로 써도 코드 출처만 남는다
    graph = build_graph(deps_of(tools, llm))
    events, st, _ = await run(graph, "카페인이 심박에 영향을 주나요?")
    assert tools.search_calls == [("카페인이 심박에 영향을 주나요?", 3)]
    assert ("retrieval", "task_started") in [(e.get("task_id"), e["type"]) for e in events]
    done = next(e for e in events if e["type"] == "task_completed" and e["task_type"] == "retrieval")
    assert "2건" in done["summary"] and "vector" in done["summary"]
    assert "카페인은 일시적으로" in llm.prompts["simple_answer"][0]  # 검색 대목이 프롬프트의 가이드라인 자리에 들어간다
    assert st["report"].count("※ 출처:") == 1 and "카페인과 심박 > 영향" in st["report"] and "G07" not in st["report"] and "X99" not in st["report"]
    assert assemble(events) == st["report"]


@pytest.mark.asyncio
async def test_rag_failure_or_missing_tool_falls_back_to_keywords():
    tools = FakeTools()
    tools.search_hits, tools.search_error = HITS, ConnectionError("down")
    graph = build_graph(deps_of(tools, NullLLM()))
    events, st, _ = await run(graph, "심박수 정상 범위가 어떻게 돼?")
    assert any(e["type"] == "task_failed" and e["task_type"] == "retrieval" and "키워드" in e["error"] for e in events)
    assert "60~100" in st["report"] and "※ 출처" not in st["report"] and st["validation"]["passed"]
    plain = FakeTools()  # search 도구가 없는 서버
    events, st, _ = await run(build_graph(deps_of(plain, NullLLM())), "심박수 정상 범위가 어떻게 돼?")
    assert all(e.get("task_type") != "retrieval" for e in events if e["type"].startswith("task_")) and "60~100" in st["report"]


@pytest.mark.asyncio
async def test_rag_is_not_used_for_greetings_and_supplies_followup_fallback_text():
    tools = FakeTools([CRITICAL])
    tools.search_hits = HITS
    graph = build_graph(deps_of(tools, NullLLM()))
    await run(graph, "안녕하세요", thread="g")
    assert tools.search_calls == []  # 인사·범위 밖 질문에는 검색하지 않는다
    await run(graph, "U01 최근 30일 이상 점검해줘", thread="r")
    _, st, _ = await run(graph, "그럼 카페인은 영향이 있어?", thread="r")
    assert "카페인은 일시적으로 심박을 올릴 수" in st["report"] and "135.0 bpm" in st["report"] and gates.ESCALATION_MESSAGE in st["report"]


@pytest.mark.asyncio
async def test_rag_search_is_bounded_and_not_retried(monkeypatch):
    """검색이 멈춰도 단순 질문이 분 단위로 막히지 않는다: 짧은 상한 + 재시도 없음 → 키워드 폴백 (실측 사례: 20s x 3회 = 60s)."""
    import asyncio

    tools = FakeTools()
    tools.search_hits = HITS

    async def hang(query, k):
        tools.search_calls.append((query, k))
        await asyncio.sleep(30)

    tools._search = hang
    monkeypatch.setattr("app.agent.nodes.RETRIEVAL_TIMEOUT_S", 0.1)
    graph = build_graph(deps_of(tools, NullLLM()))
    events, st, _ = await run(graph, "심박수 정상 범위가 어떻게 돼?")
    assert len(tools.search_calls) == 1  # 재시도 없음
    assert any(e["type"] == "task_failed" and e["task_type"] == "retrieval" for e in events) and "60~100" in st["report"]
    # 서킷 브레이커: 직후 요청은 검색을 아예 시도하지 않는다 (느린 임베딩이 후속 요청을 줄줄이 막지 않게)
    events2, st2, _ = await run(graph, "서맥이 뭐예요?", thread="t2")
    assert len(tools.search_calls) == 1 and all(e.get("task_type") != "retrieval" for e in events2 if e["type"].startswith("task_"))


# ---- 재시도 횟수(usage) + 옵트인 결함 주입 -------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["v2", "v1"])
async def test_fault_injection_causes_exactly_one_retry_recorded_in_event_and_usage(version):
    d = AgentDeps(FakeBackend(), FakeBackend(), FakeTools([CRITICAL]).toolbox(), InMemorySaver(), dataclasses.replace(settings, fault_inject_transient=True), version)
    graph = build_graph(d)
    events, st, usage = await run(graph, "U01 최근 30일 이상 점검해줘")
    done = [e for e in events if e["type"] == "task_completed" and e["task_type"] == "research"]
    assert sum(e["retries"] for e in done) == 1 and usage["retries"] == 1 and len(done) == 2  # 한 번 실패 후 복구, 둘 다 성공
    assert not any(e["type"] == "task_failed" for e in events)
    _, _, usage2 = await run(graph, "U01 최근 30일 이상 점검해줘", thread="again")  # 주입은 프로세스당 한 번 — 두 번째 요청은 재시도 0
    assert usage2["retries"] == 0


@pytest.mark.asyncio
async def test_retries_default_to_zero_when_injection_is_off():
    d = deps_of(FakeTools([CRITICAL]), FakeBackend())
    events, st, usage = await run(build_graph(d), "U01 이상 점검해줘")
    assert usage["retries"] == 0 and all(e.get("retries", 0) == 0 for e in events if e["type"] == "task_completed")


# ---- MCP Prompt(anomaly_review)를 계획의 표준 절차로 사용 + 리소스 탐색 ----------------------------------
PROCEDURE = "사용자 U01 의 최근 7일 심박 이상 징후를 다음 순서로 점검하세요. 1. 요약 2. 탐지 3. 가이드라인 대조 4. 리포트"


@pytest.mark.asyncio
async def test_plan_uses_the_mcp_prompt_as_its_procedure_and_degrades_to_builtin():
    from app.agent.nodes import _Plan

    tools = FakeTools([CRITICAL])
    tools.prompt_text = PROCEDURE
    llm = FakeLLM(structured={"plan": _Plan(tasks=[{"id": "w", "type": "writing", "description": "작성"}], needs_analysis=True, lookback_days=7)})
    events, st, _ = await run(build_graph(deps_of(tools, llm)), "U01 최근 30일 이상 점검해줘")
    assert tools.prompt_calls == [("anomaly_review", {"user_id": "U01", "lookback_days": "30"})]  # 실제로 쓸 조회 기간으로 렌더링
    assert PROCEDURE in llm.prompts["plan"][0] and "표준 점검 절차(MCP Prompt anomaly_review)" in llm.prompts["plan"][0]
    done = next(e for e in events if e["type"] == "task_completed" and e["task_type"] == "prompt")
    assert "anomaly_review" in done["summary"]
    # 프롬프트 서버 장애 → 내장 절차로 계속 (분석은 정상 완료)
    bad = FakeTools([CRITICAL])
    bad.prompt_text, bad.prompt_error = PROCEDURE, ConnectionError("down")
    events, st, _ = await run(build_graph(deps_of(bad, NullLLM())), "U01 이상 점검해줘")
    assert any(e["type"] == "task_failed" and e["task_type"] == "prompt" for e in events) and st["validation"]["passed"]
    # Prompt 를 제공하지 않는 서버: 조용히 건너뜀
    none = FakeTools([CRITICAL])
    events, _, _ = await run(build_graph(deps_of(none, NullLLM())), "U01 이상 점검해줘")
    assert all(e.get("task_type") != "prompt" for e in events if e["type"].startswith("task_"))


@pytest.mark.asyncio
async def test_guideline_resources_are_discovered_by_name_with_constant_fallback():
    tools = FakeTools()
    tools.resource_uris = ["guideline://heart-rate-ranges", "guideline://red-flags", "guideline://disclaimer", "guideline://other"]
    await run(build_graph(deps_of(tools, NullLLM())), "심박수 정상 범위가 어떻게 돼?")
    assert sorted(tools.resource_reads) == sorted(["guideline://heart-rate-ranges", "guideline://red-flags", "guideline://disclaimer"])
    # 서버가 다른 접두어로 제공해도 이름(접미)이 맞으면 찾는다
    moved = FakeTools()
    moved.resource_uris = ["docs://v2/heart-rate-ranges"]

    async def _resource(uri):
        moved.resource_reads.append(uri)
        return "## 안정시\n60~100회"

    moved._resource = _resource
    await run(build_graph(deps_of(moved, NullLLM())), "심박수 정상 범위가 어떻게 돼?")
    assert "docs://v2/heart-rate-ranges" in moved.resource_reads and "guideline://red-flags" in moved.resource_reads  # 탐색분 + 누락분은 기본 URI


# ---- v1 기준선 실행 가드 ----------------------------------------------------------------------------

def test_v1_refuses_to_start_without_explicit_allow(monkeypatch):
    monkeypatch.delenv("ALLOW_V1_BASELINE", raising=False)
    with pytest.raises(RuntimeError, match="ALLOW_V1_BASELINE=yes"):
        build_graph(deps_of(FakeTools(), NullLLM(), version="v1"))
    build_graph(deps_of(FakeTools(), NullLLM(), version="v2"))  # v2 는 영향 없음
    monkeypatch.setenv("ALLOW_V1_BASELINE", "yes")
    build_graph(deps_of(FakeTools(), NullLLM(), version="v1"))


def test_api_with_v1_and_no_allow_aborts_startup_and_version_reports_the_flag(monkeypatch):
    monkeypatch.delenv("ALLOW_V1_BASELINE", raising=False)
    with pytest.raises(RuntimeError, match="ALLOW_V1_BASELINE"):
        with TestClient(create_app(deps=deps_of(FakeTools(), NullLLM()), version="v1")):
            pass
    with TestClient(create_app(deps=deps_of(FakeTools(), NullLLM()))) as c:
        assert c.get("/version").json()["allow_v1_baseline"] is False
    monkeypatch.setenv("ALLOW_V1_BASELINE", "yes")
    with TestClient(create_app(deps=deps_of(FakeTools(), NullLLM()), version="v1")) as c:
        body = c.get("/version").json()
        assert body["supervisor_version"] == "v1" and body["allow_v1_baseline"] is True

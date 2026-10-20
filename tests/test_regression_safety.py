"""안전·정합성 회귀 테스트: 에스컬레이션 파생, 탐지 실패 처리, 진단 단정 우회, 조사 붙은 ID, 사용자 혼입, 타임아웃 등.

각 테스트는 이전에 실패했던 시나리오를 그대로 재현한다 (우회 입력 문자열 포함).
협력: `tests/fakes.py`(가짜 도구·LLM), `app.agent.*`, `app.api.main`.
"""
import asyncio
import threading

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from app.agent import gates
from app.agent.graph import build_graph
from app.agent.llm import NullLLM
from app.agent.toolbox import bounded
from app.agent.validator import collect_facts, diagnostic_assertions, numbers_check, validate_rules
from app.api.main import create_app
from app.config import load_env_file
from tests.fakes import CRITICAL, SUMMARY, WARNING, FakeLLM, FakeTools, assemble

GOOD_BODY = "## 요약\n- 평균 심박 72.4 bpm, 커버리지 88%\n- 휴식 중 25분간 평균 112.3 bpm 구간이 1건 감지되었습니다."


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr("app.agent.retry.random.uniform", lambda a, b: 0)


def make(tools=None, llm=None):
    tools = tools or FakeTools()
    return build_graph(tools.toolbox(), InMemorySaver(), llm or NullLLM()), tools


async def run(graph, message, user_id=None, thread="t1"):
    cfg = {"configurable": {"thread_id": thread}, "recursion_limit": 40}
    inp = {"messages": [("user", message)]}
    if user_id:
        inp["user_id"] = user_id
    events = [c async for c in graph.astream(inp, cfg, stream_mode="custom")]
    return events, (await graph.aget_state(cfg)).values


async def run_until(graph, message, stop_after: str, thread: str, user_id=None):
    """지정한 이벤트가 나온 직후 스트림을 닫아(= 클라이언트 이탈/취소 모사) 중간 체크포인트를 남긴다."""
    cfg = {"configurable": {"thread_id": thread}, "recursion_limit": 40}
    inp = {"messages": [("user", message)]}
    if user_id:
        inp["user_id"] = user_id
    stream = graph.astream(inp, cfg, stream_mode="custom")
    async for e in stream:
        if e["type"] == stop_after:
            break
    await stream.aclose()


# ---- #1 에스컬레이션은 저장된 플래그가 아니라 이상 목록에서 도출된다 ----------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize("stop_after", ["gate_applied", "anomalies_found"])
async def test_1_escalation_survives_cancelled_reanalysis(stop_after):
    graph, _ = make(FakeTools([CRITICAL]))
    await run(graph, "U01 최근 30일 이상 점검해줘", thread="c1")
    await run_until(graph, "U01 다시 점검해줘", stop_after, thread="c1")  # 재분석이 plan~safety_gate 사이에서 취소됨
    events, st = await run(graph, "그럼 괜찮은 거야?", thread="c1")
    assert events[0]["route"] == "simple"
    assert gates.ESCALATION_MESSAGE in st["report"] and st["escalation"] is True


# ---- #2 탐지 실패는 '이상 없음'이 아니다 -----------------------------------------------------
@pytest.mark.asyncio
async def test_2_detection_failure_is_never_an_all_clear():
    tools = FakeTools()
    tools.detect_errors = [ValueError("boom")]
    graph, _ = make(tools)
    events, st = await run(graph, "U01 이상 점검해줘", thread="d1")
    assert "판단할 수 없습니다" in st["report"] and "감지되지 않았습니다" not in st["report"]
    assert "119" in st["report"]  # 확인하지 못했음 + 응급 증상 안내
    assert st["analyzed_user"] is None  # 탐지가 실패한 턴은 후속 질문의 근거가 되지 않는다
    _, again = await run(graph, "그럼 괜찮은 거야?", thread="d1")  # 후속 질문이 재점검 경로로 간다
    assert len(tools.detect_calls) == 2


def test_2_validator_forbids_all_clear_wording_when_detection_failed():
    rules = dict(escalation=False, escalation_message="")
    for text in ["이상은 없습니다.", "임계값을 넘는 이상 구간이 감지되지 않았습니다.", "문제 없어요."]:
        assert any("이상 탐지가 실패" in i for i in validate_rules(text + "\n" + gates.DISCLAIMER, [], detection_ok=False, **rules)), text
        assert not any("이상 탐지가 실패" in i for i in validate_rules(text + "\n" + gates.DISCLAIMER, [], detection_ok=True, **rules))
    assert not any("이상 탐지가 실패" in i for i in validate_rules("이상 여부를 판단할 수 없습니다.\n" + gates.DISCLAIMER, [], detection_ok=False, **rules))


@pytest.mark.asyncio
async def test_2_llm_all_clear_is_rejected_when_detection_failed():
    tools = FakeTools()
    tools.detect_errors = [ValueError("boom")] * 6
    graph, _ = make(tools, FakeLLM(text={"writing": "이상은 없습니다. 안심하세요."}))
    events, st = await run(graph, "U01 이상 점검해줘")
    # 안심 문장은 삭제되고 '이상은 없습니다'는 사실 진술(탐지 실패)로 교정된다 - 거짓 안심은 어느 쪽이든 리포트에 남지 않는다
    assert "이상 없" not in st["report"] and "이상은 없" not in st["report"] and "안심" not in st["report"] and "판단할 수 없습니다" in st["report"]


# ---- #3 진단 단정·복약 지시 우회 (우회 문자열 그대로) -------------------------------------------
DX_BAD = [
    "심방세동입니다, 의료진과 상담하세요.", "부정맥입니다.의료진과 상담하세요.", "부정맥이 있습니다.", "부정맥이 확인되었습니다.", "부정맥으로 판단됩니다.",
    "부정맥 소견입니다.", "심방 세동입니다.", "심방세동임이 분명합니다.", "부정맥은 아닙니다만 심방세동입니다.", "You have atrial fibrillation.",
    "약을 드세요.", "아스피린을 복용해 보세요.", "당신은 심방세동입니다.", "부정맥으로 진단됩니다. 약을 복용하세요.", "심방세동이 맞습니다.",
    "You are diagnosed with arrhythmia.", "베타차단제를 드시면 됩니다.",
    # 추가 (질병 옆 헤지만 인정, 단정 어미 확장, 영문, 복약 명사)
    "부정맥이 있고 걱정할 수 없는 상태입니다.", "당신은 부정맥이라고 할 수 있습니다.", "심방세동을 겪고 계십니다.", "부정맥에 해당합니다.", "부정맥을 진단할 수 있습니다.",
    "부정맥이 발생했습니다.", "부정맥이 나타났습니다.", "심방세동이 확인됐습니다.", "This is atrial fibrillation.", "The ECG shows arrhythmia.",
    "타이레놀을 드세요.", "이부프로펜을 복용하세요.", "심장마비입니다.", "AFib입니다.",
]
DX_OK = [
    "휴식 중 빈맥입니다.", "휴식 중 25분간 평균 112 bpm 의 빈맥 구간이 있었습니다.", "서맥 구간이 확인되었습니다.", "개인 기준선 이탈이 있습니다.", "센서 결측 구간입니다.",
    "부정맥 가능성이 있어 지속되면 진료를 권합니다.", "부정맥은 아닙니다.", "심방세동 여부는 진단할 수 없습니다.", "부정맥이 의심되면 의료진과 상담하세요.",
    "임의로 약을 복용하지 마세요.", "이 서비스는 진단을 제공하지 않습니다.", "서맥이 뭐예요?", "It might be arrhythmia, please see a doctor.",
    "부정맥 소견은 없습니다.", "부정맥이 있다고 단정할 수 없습니다.", "심방세동이라고 진단해 달라는 요청은 거절합니다.", "의심되는 부정맥이 있습니다.",
]


@pytest.mark.parametrize("text", DX_BAD)
def test_3_diagnosis_and_medication_bypasses_are_caught(text):
    assert diagnostic_assertions(text), text


@pytest.mark.parametrize("text", DX_OK)
def test_3_descriptive_and_hedged_statements_are_allowed(text):
    assert diagnostic_assertions(text) == [], text


# ---- #4 한국어 조사가 붙은 사용자 ID -------------------------------------------------------------
@pytest.mark.parametrize("msg,expected", [("U05의 최근 7일 점검해줘", "U05"), ("u05는 어때?", "U05"), ("U05를 봐줘", "U05"), ("사용자U05 점검", "U05"), ("U05.", "U05"), ("점검해줘", None)])
def test_4_user_id_with_korean_particles(msg, expected):
    assert gates.extract_user_id(msg) == expected


@pytest.mark.asyncio
async def test_4_message_user_beats_body_user_in_graph():
    graph, tools = make()
    await run(graph, "U03의 최근 7일 점검해줘", user_id="U01")
    assert tools.detect_calls[0][0] == "U03"


# ---- #5 후속 질문은 직전 분석 대상의 사실을 그 이름으로 답한다 ------------------------------------
@pytest.mark.asyncio
async def test_5_followup_with_different_body_user_is_labelled_with_analyzed_user():
    llm = FakeLLM(text={"simple_answer": "가장 심했던 이벤트는 135.0 bpm 입니다."})
    graph, tools = make(FakeTools([CRITICAL]), llm)
    await run(graph, "U02 최근 30일 점검해줘", user_id="U01", thread="u5")  # 메시지가 U02 를 지정 → U02 분석
    events, st = await run(graph, "가장 심했던 건 몇 bpm 이었어?", user_id="U01", thread="u5")  # UI 는 다시 U01 을 보냄
    assert events[0]["route"] == "simple" and len(tools.detect_calls) == 1
    prompt = llm.prompts["simple_answer"][0]
    assert "사용자 U02" in prompt and "사용자 U01" not in prompt  # 사실은 분석 대상 이름으로 표기
    assert "직전 점검 대상: U02" in st["report"] and "'U01 점검해줘'" in st["report"]
    assert gates.ESCALATION_MESSAGE in st["report"]


# ---- #6 실패한 가이드라인 로드는 캐시하지 않는다 -------------------------------------------------
@pytest.mark.asyncio
async def test_6_failed_guideline_load_is_not_cached_and_uses_bundled_fallback():
    class Flaky(FakeTools):
        fail, calls = True, 0

        async def _resource(self, uri):
            self.calls += 1
            if self.fail:
                raise ConnectionError("down")
            return await super()._resource(uri)

    tools = Flaky()
    graph, _ = make(tools)
    _, first = await run(graph, "심박수 정상 범위가 어떻게 돼?", thread="g1")
    assert "60~100" in first["report"] and gates.DISCLAIMER in first["report"]  # 번들 요약·상수 면책으로 답한다 (빈 근거 아님)
    tools.fail = False
    before = tools.calls
    await run(graph, "서맥이 뭐예요?", thread="g2")
    assert tools.calls > before  # 실패는 캐시되지 않아 다음 요청에서 다시 읽는다
    after_success = tools.calls
    await run(graph, "빈맥이 뭐예요?", thread="g3")
    assert tools.calls == after_success  # 전부 성공한 뒤에는 캐시된다


# ---- #7 MCP 호출 타임아웃, 실제 생존 확인 ---------------------------------------------------------
@pytest.mark.asyncio
async def test_7_bounded_raises_transient_timeout():
    async def hang():
        await asyncio.sleep(5)

    with pytest.raises(TimeoutError):
        await bounded(hang(), 0.05)


@pytest.mark.asyncio
async def test_7_hung_tool_becomes_isolated_task_failure(monkeypatch):
    class Hung(FakeTools):
        async def _detect(self, uid, days, end):
            await bounded(asyncio.sleep(5), 0.05)  # 도구 호출 상한을 모사

    graph, tools = make(Hung())
    events, st = await run(graph, "U01 이상 점검해줘")
    assert any(e["type"] == "task_failed" for e in events) and "판단할 수 없습니다" in st["report"]


def test_7_health_pings_mcp_for_liveness(monkeypatch):
    class Dying(FakeTools):
        dead = False

        async def _users(self):
            if self.dead:
                await asyncio.sleep(10)
            return await super()._users()

    monkeypatch.setattr("app.api.main.MCP_PING_TIMEOUT_S", 0.1)
    tools = Dying()
    with TestClient(create_app(tools.toolbox(), NullLLM(), "memory")) as c:
        assert c.get("/health").json()["mcp"] is True
        tools.dead = True
        c.app.state.mcp_health = (0.0, True)  # 캐시 만료 모사
        assert c.get("/health").json()["mcp"] is False  # 프로세스가 죽어도 true 로 남던 결함


# ---- #8 수치 근거 검사 강화 -----------------------------------------------------------------------
def test_8_unitless_heart_rate_numbers_are_checked_and_user_numbers_are_not_facts():
    facts = collect_facts(SUMMARY, {"anomalies": [WARNING]})
    cited, bad = numbers_check("휴식 중 심박이 145까지 올랐습니다.", facts)
    assert bad == [145.0]  # 단위 없는 수치도 심박 문맥이면 검사
    assert numbers_check("휴식 중 심박이 112까지 올랐습니다.", facts)[1] == []
    # 사용자가 입력한 숫자는 사실이 아니다
    assert 250.0 not in collect_facts(SUMMARY, extra_text="30")
    assert numbers_check("최고 250 bpm 이었습니다.", collect_facts(SUMMARY, extra_text="30"))[1] == [250.0]


def test_8_rounding_tolerance_is_tight():
    facts = [112.3]
    assert numbers_check("평균 112 bpm", facts)[1] == []      # 반올림/버림
    assert numbers_check("평균 113 bpm", facts)[1] == [113.0]  # 예전에는 ±0.5·1% 로 통과하던 값
    assert numbers_check("평균 112.3 bpm", facts)[1] == []
    assert numbers_check("평균 112.9 bpm", facts)[1] == [112.9]


# ---- #10 스트리밍 가드 ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_10_streaming_guard_never_emits_diagnosis_sentence_before_validation():
    bad = "휴식 중 25분간 평균 112.3 bpm 구간이 있었습니다. 심방세동입니다. 의료진과 상담하세요."
    llm = FakeLLM(text={"writing": [bad, GOOD_BODY]})
    graph, _ = make(llm=llm)
    events, st = await run(graph, "U01 이상 점검해줘")
    first_validation = next(i for i, e in enumerate(events) if e["type"] == "validation")
    pre = assemble(events[:first_validation])
    assert "심방세동" not in pre and "112.3 bpm 구간" in pre  # 진단 문장은 건너뛰고(허용 목록) 사실 문장은 그대로 보인다
    assert events[first_validation]["passed"] and "심방세동" not in st["report"]   # 지운 것이므로 검증 실패·재시도가 없다
    assert assemble(events) == st["report"]


@pytest.mark.asyncio
async def test_10_guard_keeps_clean_streams_intact_including_decimals():
    graph, _ = make(llm=FakeLLM(text={"writing": GOOD_BODY}))
    events, st = await run(graph, "U01 이상 점검해줘")
    assert "72.4 bpm" in assemble(events) and assemble(events) == st["report"]


# ---- #12 같은 thread 직렬화 -----------------------------------------------------------------------
def test_12_same_thread_requests_are_serialized_and_locks_are_cleaned():
    state = {"now": 0, "max": 0}

    class Tracking(FakeTools):
        async def _detect(self, uid, days, end):
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
            await asyncio.sleep(0.15)
            state["now"] -= 1
            return await super()._detect(uid, days, end)

    with TestClient(create_app(Tracking().toolbox(), NullLLM(), "memory")) as c:
        def post(tid):
            c.post("/chat", json={"message": "U01 이상 점검해줘", "thread_id": tid})

        for tids in (["same", "same", "same"], ["a", "b", "c"]):
            state["max"] = 0
            ts = [threading.Thread(target=post, args=(t,)) for t in tids]
            [t.start() for t in ts]
            [t.join() for t in ts]
            assert state["max"] == (1 if tids[0] == tids[1] else state["max"])
            if tids[0] != tids[1]:
                assert state["max"] >= 2  # 서로 다른 thread 는 병렬
        assert c.app.state.locks == {}  # 사용 끝난 락은 정리된다


# ---- #13 MCP 검증 오류는 422, 알 수 없는 사용자는 404 ---------------------------------------------
def test_13_validation_errors_map_to_422_unknown_user_to_404():
    class Strict(FakeTools):
        async def _series(self, uid, days, end, resample):
            if uid == "NOPE":
                return {"error": "알 수 없는 user_id: 'NOPE' (사용 가능: U01)"}
            return {"error": "lookback_days 는 0 초과 30 이하여야 합니다 (받은 값: 99)"}

    with TestClient(create_app(Strict().toolbox(), NullLLM(), "memory")) as c:
        assert c.get("/users/NOPE/series").status_code == 404
        assert c.get("/users/U01/series?lookback_days=99").status_code == 422


# ---- #14 task_failed 에 내부 예외 문구를 싣지 않는다 -----------------------------------------------
@pytest.mark.asyncio
async def test_14_task_failed_event_is_generic():
    tools = FakeTools()
    tools.detect_errors = [RuntimeError("secret /internal/path pandas KeyError")]
    graph, _ = make(tools)
    events, _ = await run(graph, "U01 이상 점검해줘")
    failed = next(e for e in events if e["type"] == "task_failed")
    assert "secret" not in failed["error"] and "pandas" not in failed["error"]


# ---- #17 기간 파서 -------------------------------------------------------------------------------
@pytest.mark.parametrize("msg,expected", [
    ("1일 3회 약 먹는데 심박 괜찮아?", None), ("하루 2번 측정해요", None), ("어제 심박 어땠어", None), ("오늘 괜찮아?", None),
    ("지난 3일 점검", 3.0), ("2주 동안", 14.0), ("일주일치", 7.0), ("4월 12일 심박", None), ("하루 동안", 1.0),
])
def test_17_period_parser_ignores_frequency_and_relative_days(msg, expected):
    assert gates.extract_lookback_days(msg) == expected


@pytest.mark.asyncio
async def test_17_frequency_phrase_is_not_a_new_check_and_keeps_escalation():
    graph, tools = make(FakeTools([CRITICAL]))
    await run(graph, "U01 최근 30일 이상 점검해줘", thread="f17")
    events, st = await run(graph, "1일 3회 약을 먹는데 괜찮아?", thread="f17")
    assert events[0]["route"] == "simple" and len(tools.detect_calls) == 1 and gates.ESCALATION_MESSAGE in st["report"]


# ---- #18 .env 로딩 --------------------------------------------------------------------------------
def test_18_env_file_is_loaded_without_overriding_shell(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("VS_TEST_FROM_FILE=file\nVS_TEST_SHELL=file\n")
    monkeypatch.setenv("VS_TEST_SHELL", "shell")
    monkeypatch.delenv("VS_TEST_FROM_FILE", raising=False)
    assert load_env_file(env) is True
    import os

    assert os.environ["VS_TEST_FROM_FILE"] == "file" and os.environ["VS_TEST_SHELL"] == "shell"
    monkeypatch.delenv("VS_TEST_FROM_FILE")
    assert load_env_file(tmp_path / "missing.env") is False


# ---- #19 공개 API 독스트링 ------------------------------------------------------------------------
def test_19_public_functions_and_classes_have_docstrings():
    import ast
    import glob

    missing = []
    for f in sorted(glob.glob("app/agent/*.py")) + ["app/api/main.py", "app/config.py"]:
        for n in ast.walk(ast.parse(open(f, encoding="utf-8").read())):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not n.name.startswith("_") and not ast.get_docstring(n):
                missing.append(f"{f}:{n.lineno} {n.name}")
    assert missing == []

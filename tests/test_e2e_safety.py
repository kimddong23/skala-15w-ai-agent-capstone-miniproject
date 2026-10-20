"""안전·정합성 회귀 E2E. fake 도구/LLM 으로 결정론적으로 재현하고, 일부는 실제 MCP 프로세스로 검증한다.
"""
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from app.agent.gates import ESCALATION_MESSAGE
from app.agent.llm import NullLLM
from app.api.main import create_app
from app.config import ROOT
from tests.fakes import CRITICAL, WARNING, FakeLLM, FakeTools, assemble
from tests.test_api import parse_sse
from tests.test_graph import GOOD_BODY


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr("app.agent.retry.random.uniform", lambda a, b: 0)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _stream(c, msg, user_id=None, thread_id=None):
    with c.stream("POST", "/chat/stream", json={"message": msg, "user_id": user_id, "thread_id": thread_id}) as r:
        return parse_sse(r.read().decode())


class UvicornThread:
    """실제 uvicorn(스레드) — 클라이언트 이탈/타임아웃처럼 TestClient 로 못 보는 경로용."""

    def __init__(self, app):
        self.port = _free_port()
        self.server = uvicorn.Server(uvicorn.Config(app, port=self.port, log_level="warning"))
        self.th = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.th.start()
        while not self.server.started:
            time.sleep(0.05)
        self.url = f"http://127.0.0.1:{self.port}"
        return self

    def __exit__(self, *a):
        self.server.should_exit = True
        self.th.join(10)


# ---------------------------------------------------------------------------------------- #4 조사 붙은 사용자 ID


@pytest.mark.parametrize("msg,uid", [
    ("U03의 최근 7일 점검해줘", "U03"), ("U03는 괜찮은지 점검해줘", "U03"), ("u03를 점검해줘", "U03"),
    ("사용자U03 이상 점검", "U03"),
])
def test_user_id_followed_by_korean_particle_is_analyzed_not_the_ui_selected_user(msg, uid):
    ft = FakeTools([WARNING])
    with TestClient(create_app(ft.toolbox(), NullLLM(), "memory")) as c:
        ev = _stream(c, msg, user_id="U01")  # UI 에서 U01 이 선택된 상태
    assert ev[-1]["type"] == "done"
    assert {call[0] for call in ft.detect_calls} == {uid}, ft.detect_calls


def test_particle_user_id_on_real_stack(http):
    _stream_real = http.post("/chat", json={"message": "U03의 최근 7일 점검해줘", "user_id": "U05"}).json()
    assert "(U03)" in _stream_real["report"] and "(U05)" not in _stream_real["report"]


# ---------------------------------------------------------------------------------------- #1 취소된 재점검 후 에스컬레이션


def test_follow_up_after_cancelled_reanalysis_still_shows_escalation():
    class SlowAnalysis(FakeLLM):
        calls = 0

        async def text(self, tier, name, system, user, on_token=None, max_tokens=None):
            if name == "analysis":
                SlowAnalysis.calls += 1
                if SlowAnalysis.calls == 2:  # 2번째 점검의 해석 단계에서 오래 걸림 → 클라이언트가 중단
                    await asyncio.sleep(60)
            return await super().text(tier, name, system, user, on_token, max_tokens)

    llm = SlowAnalysis(text={"writing": GOOD_BODY, "analysis": "정상 범위로 보입니다.", "simple_answer": "운동 중에는 심박이 오르는 것이 정상입니다."})
    ft = FakeTools([CRITICAL])
    tid = f"cx-{uuid.uuid4().hex[:6]}"
    with UvicornThread(create_app(ft.toolbox(), llm, "memory")) as srv:
        with httpx.Client(base_url=srv.url, timeout=60) as c:
            first = c.post("/chat", json={"message": "U01 이상 점검해줘", "thread_id": tid}).json()
            assert first["escalation"] is True
            with c.stream("POST", "/chat/stream", json={"message": "U01 다시 점검해줘", "thread_id": tid}) as r:
                for line in r.iter_lines():
                    if '"type": "anomalies_found"' in line:
                        break  # research 직후, analysis(느림) 도중에 연결을 끊는다
            time.sleep(1.5)
            follow = c.post("/chat", json={"message": "그럼 괜찮은 거야?", "thread_id": tid}).json()
    assert follow["escalation"] is True
    assert ESCALATION_MESSAGE in follow["report"]


def test_follow_up_after_failed_reanalysis_still_shows_escalation():
    ft = FakeTools([CRITICAL])
    with TestClient(create_app(ft.toolbox(), NullLLM(), "memory")) as c:
        tid = "fx1"
        assert _stream(c, "U01 이상 점검해줘", thread_id=tid)[-1]["escalation"] is True
        ft.detect_errors.append(ValueError("boom"))
        _stream(c, "U01 다시 점검해줘", thread_id=tid)
        follow = _stream(c, "그럼 괜찮은 거야?", thread_id=tid)
    assert ESCALATION_MESSAGE in follow[-1]["report"]


# ---------------------------------------------------------------------------------------- #2 탐지 실패 ≠ 이상 없음

ALL_CLEAR = ("감지되지 않았습니다", "이상이 없", "이상 없", "문제가 없", "정상입니다")


def test_detect_failure_never_yields_an_all_clear_report():
    ft = FakeTools([CRITICAL])
    ft.detect_errors.append(ValueError("detector exploded"))
    with TestClient(create_app(ft.toolbox(), NullLLM(), "memory")) as c:
        ev = _stream(c, "U01 이상 점검해줘")
    report = ev[-1]["report"]
    assert "task_failed" in [e["type"] for e in ev]
    assert not any(k in report for k in ALL_CLEAR), report
    assert any(k in report for k in ("실행하지 못", "판단할 수 없", "확인하지 못", "조회하지 못")), report


def test_detect_failure_follow_up_is_not_answered_as_checked():
    ft = FakeTools([CRITICAL])
    ft.detect_errors.append(ValueError("detector exploded"))
    with TestClient(create_app(ft.toolbox(), NullLLM(), "memory")) as c:
        _stream(c, "U01 이상 점검해줘", thread_id="df1")
        ev = _stream(c, "그럼 괜찮은 거야?", thread_id="df1")
    # 후속은 안심 문구 대신 재점검을 하거나, 점검이 안 됐음을 밝혀야 한다
    rep = ev[-1]["report"]
    assert not any(k in rep for k in ALL_CLEAR) or "plan_created" in [e["type"] for e in ev]


def test_detect_failure_with_llm_is_told_detection_is_unavailable_not_zero():
    llm = FakeLLM(text={"writing": GOOD_BODY, "analysis": "해석"})
    ft = FakeTools([CRITICAL])
    ft.detect_errors.append(ValueError("detector exploded"))
    with TestClient(create_app(ft.toolbox(), llm, "memory")) as c:
        _stream(c, "U01 이상 점검해줘")
    prompt = "\n".join(llm.prompts.get("writing", []) + llm.prompts.get("analysis", []))
    assert "이상 총 0건" not in prompt


# ---------------------------------------------------------------------------------------- #14 내부 오류 문구 비노출


def test_task_failed_event_does_not_leak_raw_exception_text():
    ft = FakeTools([CRITICAL])
    ft.detect_errors.append(ValueError("SECRET-internal-stack-detail pandas KeyError 'hr_mean'"))
    with TestClient(create_app(ft.toolbox(), NullLLM(), "memory")) as c:
        ev = _stream(c, "U01 이상 점검해줘")
    assert "SECRET-internal" not in json.dumps(ev, ensure_ascii=False)


# ---------------------------------------------------------------------------------------- #3/#10 진단 우회 문구 스트리밍

BYPASSES = [
    "심방세동입니다, 의료진과 상담하세요.",
    "부정맥입니다.의료진과 상담하세요.",
    "부정맥이 있습니다.",
    "부정맥이 확인되었습니다.",
    "부정맥으로 판단됩니다.",
    "부정맥 소견입니다.",
    "심방 세동입니다.",
    "심방세동임이 분명합니다.",
    "부정맥은 아닙니다만 심방세동입니다.",
    "You have atrial fibrillation.",
    "약을 드세요.",
    "아스피린을 복용해 보세요.",
]
FORBIDDEN = ("심방세동", "심방 세동", "부정맥이 있", "부정맥이 확인", "부정맥으로 판단", "부정맥 소견", "부정맥입니다", "atrial fibrillation", "약을 드세요", "복용해 보세요")


@pytest.mark.parametrize("bypass", BYPASSES)
def test_diagnosis_bypass_text_never_reaches_the_client_even_while_streaming(bypass):
    llm = FakeLLM(text={"writing": GOOD_BODY + "\n" + bypass, "analysis": "해석"})
    with TestClient(create_app(FakeTools([WARNING]).toolbox(), llm, "memory")) as c:
        ev = _stream(c, "U01 이상 점검해줘")
    shown = ""  # 클라이언트가 화면에 그리는 텍스트의 모든 중간 상태를 검사
    for e in ev:
        if e["type"] == "token":
            shown = e["text"] if e["reset"] else shown + e["text"]
            assert not any(f in shown for f in FORBIDDEN), f"스트리밍 중 노출: {shown[-80:]!r}"
    final = ev[-1]["report"]
    assert not any(f in final for f in FORBIDDEN), final
    assert assemble(ev) == final


def test_own_type_label_is_not_flagged_as_a_diagnosis():
    body = GOOD_BODY + "\n- [주의] 휴식 중 빈맥입니다."
    llm = FakeLLM(text={"writing": body, "analysis": "해석"})
    with TestClient(create_app(FakeTools([WARNING]).toolbox(), llm, "memory")) as c:
        ev = _stream(c, "U01 이상 점검해줘")
    assert [e for e in ev if e["type"] == "validation"][0]["passed"] is True


# ---------------------------------------------------------------------------------------- #5 다른 사용자 사실 혼입


def test_follow_up_for_a_different_body_user_never_answers_with_the_other_users_facts():
    llm = FakeLLM(text={"writing": GOOD_BODY, "analysis": "해석", "simple_answer": "가장 심했던 구간 안내."})
    ft = FakeTools([CRITICAL])
    with TestClient(create_app(ft.toolbox(), llm, "memory")) as c:
        _stream(c, "U02 이상 점검해줘", user_id="U01", thread_id="mu1")  # U01 스레드에서 U02 를 점검
        ev = _stream(c, "가장 심했던 건 몇 시야?", user_id="U01", thread_id="mu1")
    prompt = (llm.prompts.get("simple_answer") or [""])[-1]
    rechecked = any(call[0] == "U01" for call in ft.detect_calls)
    assert rechecked or "사용자 U01" not in prompt, prompt[:300]
    assert ev[-1]["type"] == "done"


# ---------------------------------------------------------------------------------------- #7 MCP 타임아웃 / 사망


def test_hung_mcp_tool_call_does_not_hang_the_request_forever():
    class Hang(FakeTools):
        async def _detect(self, uid, days, end):
            await asyncio.sleep(3600)

    with UvicornThread(create_app(Hang([CRITICAL]).toolbox(), NullLLM(), "memory")) as srv:
        t0 = time.time()
        with httpx.Client(base_url=srv.url, timeout=120) as c:
            with c.stream("POST", "/chat/stream", json={"message": "U01 이상 점검해줘"}) as r:
                events = parse_sse(r.read().decode())
        elapsed = time.time() - t0
        # TOOL_TIMEOUT_S=20 이고 TimeoutError 는 transient 라 최대 3회 시도(20s x 3) → 60s 근방. 상한 75s, 하한 15s(타임아웃이 실제로 작동)
        assert 15 < elapsed < 75, elapsed
    t = [e["type"] for e in events]
    assert t[-1] == "done" and "task_failed" in t


def test_health_turns_false_after_the_mcp_subprocess_dies():
    port = _free_port()
    env = {**os.environ, "LLM_ENABLED": "false", "CHECKPOINTER": "memory", "PYTHONPATH": str(ROOT)}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.api.main:app", "--port", str(port), "--log-level", "warning"],
                            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                if httpx.get(f"{url}/health", timeout=1).json().get("mcp"):
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        kids = subprocess.run(["pgrep", "-P", str(proc.pid)], capture_output=True, text=True).stdout.split()
        assert kids, "MCP 자식 프로세스를 찾지 못함"
        for k in kids:
            os.kill(int(k), signal.SIGKILL)
        time.sleep(1)
        ok = False
        for _ in range(10):  # 캐시된 상태가 갱신될 시간
            h = httpx.get(f"{url}/health", timeout=10).json()
            if h["mcp"] is False:
                ok = True
                break
            time.sleep(1)
        assert ok, h
        r = httpx.post(f"{url}/chat", json={"message": "U05 이상 점검해줘"}, timeout=60)
        assert r.status_code in (200, 503)  # 무한 대기만 아니면 된다
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------------------------------------------------------------------------------------- #6 가이드라인 로드 실패 캐시


def test_guideline_load_failure_is_retried_on_the_next_request():
    class Flaky(FakeTools):
        n = 0

        async def _resource(self, uri):
            Flaky.n += 1
            if Flaky.n <= 3:  # 첫 요청의 3개 리소스 로드 실패
                raise ValueError("transient")
            return await super()._resource(uri)

    llm = FakeLLM(text={"simple_answer": "휴식기 정상 범위는 60~100 입니다."})
    with TestClient(create_app(Flaky([WARNING]).toolbox(), llm, "memory")) as c:
        _stream(c, "안정시 심박 정상 범위가 뭐야?")
        _stream(c, "안정시 심박 정상 범위가 뭐야?")
    assert "60~100" in llm.prompts["simple_answer"][-1]


# ---------------------------------------------------------------------------------------- #12 같은 thread 동시 요청, #13 오류 코드


def test_two_concurrent_requests_on_one_thread_do_not_both_run_unserialized():
    class Slow(FakeTools):
        active = 0
        peak = 0

        async def _detect(self, uid, days, end):
            Slow.active += 1
            Slow.peak = max(Slow.peak, Slow.active)
            try:
                await asyncio.sleep(1.0)
                return await super()._detect(uid, days, end)
            finally:
                Slow.active -= 1

    with UvicornThread(create_app(Slow([WARNING]).toolbox(), NullLLM(), "memory")) as srv:
        def one(_):
            with httpx.Client(base_url=srv.url, timeout=60) as c:
                return c.post("/chat", json={"message": "U01 이상 점검해줘", "thread_id": "same"}).status_code

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(2) as ex:
            codes = list(ex.map(one, range(2)))
    assert Slow.peak == 1 or 409 in codes, (Slow.peak, codes)


def test_invalid_lookback_or_resample_is_422_not_404(http):
    assert http.get("/users/U05/anomalies", params={"lookback_days": 9999}).status_code in (200, 422)
    assert http.get("/users/U05/series", params={"lookback_days": 1, "resample": "banana"}).status_code == 422
    assert http.get("/users/NOPE/anomalies").status_code == 404


# ---------------------------------------------------------------------------------------- 데이터 한계 안내 · 안심 문구 · 운동 후속 답변

GAP = {**WARNING, "anomaly_id": "G1", "type": "sensor_gap", "severity": "info", "value": 0.0, "baseline": None, "context": "rest",
       "duration_min": 509, "evidence": "심박 신호가 509분 동안 끊겼어요."}
LIMIT = "이번 기간은 기록이 제한적이어서"


def test_notice_absent_when_coverage_is_high_and_only_info_sensor_gaps_exist():
    with TestClient(create_app(FakeTools([GAP, {**GAP, "anomaly_id": "G2"}], coverage=88.0).toolbox(), NullLLM(), "memory")) as c:
        ev = _stream(c, "U01 이상 점검해줘")
    assert LIMIT not in ev[-1]["report"]


def test_notice_absent_for_critical_user_with_high_coverage_and_two_gaps():
    with TestClient(create_app(FakeTools([CRITICAL, GAP, {**GAP, "anomaly_id": "G2"}], coverage=88.0).toolbox(), NullLLM(), "memory")) as c:
        ev = _stream(c, "U01 이상 점검해줘")
    assert ESCALATION_MESSAGE in ev[-1]["report"] and LIMIT not in ev[-1]["report"]


def test_notice_present_for_very_low_coverage():
    with TestClient(create_app(FakeTools([GAP], coverage=1.4).toolbox(), NullLLM(), "memory")) as c:
        ev = _stream(c, "U01 이상 점검해줘")
    assert ev[-1]["report"].count(LIMIT) == 1


def test_notice_not_duplicated_in_follow_up():
    with TestClient(create_app(FakeTools([GAP], coverage=1.4).toolbox(), NullLLM(), "memory")) as c:
        _stream(c, "U01 이상 점검해줘", thread_id="n1")
        ev = _stream(c, "그럼 운동할 때는 괜찮은 거야?", thread_id="n1")
    assert ev[-1]["report"].count(LIMIT) <= 1


@pytest.mark.parametrize("claim", ["심박 기록은 정상 범위 내에 있습니다.", "심박 수치는 정상 범위에 해당합니다.", "전반적으로 정상입니다."])
def test_llm_cannot_claim_normal_when_the_data_limit_notice_applies(claim):
    llm = FakeLLM(text={"writing": GOOD_BODY + "\n" + claim, "analysis": "해석"})
    with TestClient(create_app(FakeTools([GAP], coverage=1.4).toolbox(), llm, "memory")) as c:
        ev = _stream(c, "U01 이상 점검해줘")
    shown = ""
    for e in ev:
        if e["type"] == "token":
            shown = e["text"] if e["reset"] else shown + e["text"]
            assert claim not in shown, f"스트리밍 중 노출: {shown[-80:]!r}"
    assert claim not in ev[-1]["report"] and LIMIT in ev[-1]["report"]


@pytest.mark.parametrize("claim", ["운동 중 심박수는 정상 범위 내에 있었습니다.", "운동 중에는 괜찮습니다.", "운동할 때는 문제가 없습니다."])
def test_follow_up_never_asserts_exercise_heart_rate_is_normal(claim):
    llm = FakeLLM(text={"writing": GOOD_BODY, "analysis": "해석", "simple_answer": claim})
    with TestClient(create_app(FakeTools([CRITICAL]).toolbox(), llm, "memory")) as c:
        _stream(c, "U01 이상 점검해줘", thread_id="x13")
        ev = _stream(c, "그럼 운동할 때는 괜찮은 거야?", thread_id="x13")
    r = ev[-1]["report"]
    assert claim not in r
    assert ESCALATION_MESSAGE in r and ev[-1]["type"] == "done"

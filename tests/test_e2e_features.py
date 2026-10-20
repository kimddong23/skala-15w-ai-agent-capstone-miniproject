"""신규 기능 E2E: HITL 승인, 사용량/재시도 계량, /version, 여러 사용자 비교, 검색·MCP Prompt 이벤트,
대화 기억(재시작 포함), 안심 표현 상시 차단, MCP 2서버 부분 저하, v1 기준선.

각 설정은 env 가 다른 실제 uvicorn 서브프로세스(+실제 MCP stdio)로 검증한다 (설정은 프로세스 시작 시 고정되므로 TestClient 로는 못 바꾼다).
LLM 은 쓰지 않는다: LLM_ENABLED=false 또는 결정론 `LLM_BACKEND=fake`. 모델이 필요한 라이브 검증은 test_e2e_live_llm 계열로 분리한다.
"""
import json
import os
import re
import socket
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from fastapi.testclient import TestClient

from app.agent.gates import ESCALATION_MESSAGE
from app.agent.llm import NullLLM
from app.api.main import create_app
from app.config import ROOT, settings
from tests.fakes import CRITICAL, FakeLLM, FakeTools
from tests.test_api import parse_sse
from tests.test_e2e_api import stream, tokens, types
from tests.test_graph import GOOD_BODY

pytestmark = pytest.mark.skipif(
    not (settings.data_dir / "vitals_minute.parquet").exists(), reason="processed 데이터 없음 (app.data.prepare 먼저 실행)"
)

DISCLAIMER_ANY = re.compile("의료기기가 아니며|의학적 진단이 아닙니다")  # 가이드라인 서버가 내려가면 코드 내장 면책문으로 대체된다
SENT = "※ 보호자 알림 발송됨(시뮬레이션)"
NOT_SENT = "※ 보호자 알림을 발송하지 않았습니다(승인되지 않음)"
CRIT_MSG = "U05 최근 30일 이상 점검해줘"


# --------------------------------------------------------------------------------------------- 서버 기동 헬퍼


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    """env 를 지정한 uvicorn 서브프로세스. 컨텍스트로 쓰거나 start()/stop() 을 직접 부른다."""

    def __init__(self, **env):
        self.env = {**os.environ, "LLM_ENABLED": "false", "CHECKPOINTER": "memory", "HITL_ENABLED": "false", "PYTHONPATH": str(ROOT),
                    **{k: str(v) for k, v in env.items()}}
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.proc: subprocess.Popen | None = None

    def start(self) -> "Server":
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.api.main:app", "--port", str(self.port), "--log-level", "warning"],
            cwd=ROOT, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        for _ in range(240):
            if self.proc.poll() is not None:
                raise RuntimeError(f"uvicorn 종료: {self.proc.stdout.read().decode()[-2000:]}")
            try:
                if httpx.get(f"{self.url}/health", timeout=1).json().get("mcp"):
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        raise RuntimeError("서버가 준비되지 않음")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def __enter__(self):
        return self.start()

    def __exit__(self, *a):
        self.stop()

    def client(self, timeout: float = 90) -> httpx.Client:
        return httpx.Client(base_url=self.url, timeout=timeout)


def _module_server(**env):
    s = Server(**env).start()
    try:
        yield s
    finally:
        s.stop()


@pytest.fixture(scope="module")
def hitl():
    yield from _module_server(HITL_ENABLED="true")


@pytest.fixture(scope="module")
def fault():
    yield from _module_server(FAULT_INJECT_TRANSIENT="1")


@pytest.fixture(scope="module")
def fake_llm():
    yield from _module_server(LLM_BACKEND="fake", LLM_ENABLED="true")


@pytest.fixture(scope="module")
def v1():
    yield from _module_server(SUPERVISOR_VERSION="v1")


def tid() -> str:
    return f"f-{uuid.uuid4().hex[:8]}"


APPROVAL_IDS: dict[str, str] = {}   # thread_id -> 가장 최근 approval_required 의 approval_id (승인 API 는 이 값을 반드시 보내야 한다)


def run(server: Server, msg: str, thread_id: str | None = None, user_id: str | None = None):
    with server.client() as c:
        events = stream(c, msg, user_id=user_id, thread_id=thread_id)[1]
    for e in events:
        if e["type"] == "approval_required":
            APPROVAL_IDS[e["thread_id"]] = e["approval_id"]
    return events


def approve(c, thread: str, approved: bool):
    """승인 API 호출 (approval_id 는 필수 - 이 대화에서 마지막으로 받은 값, 없으면 존재하지 않는 값)."""
    return c.post("/chat/approve", json={"thread_id": thread, "approved": approved, "approval_id": APPROVAL_IDS.get(thread, "none")})


# --------------------------------------------------------------------------------------------- HITL 승인


def test_hitl_critical_run_emits_approval_required_before_done(hitl):
    ev = run(hitl, CRIT_MSG, tid())
    t = types(ev)
    assert t.count("approval_required") == 1 and t[-1] == "done"
    assert t.index("validation") < t.index("approval_required") < t.index("done")
    assert t.index("safety_escalation") < t.index("approval_required")


def test_hitl_approval_event_payload_is_complete_and_marked_simulated(hitl):
    thread = tid()
    ap = next(e for e in run(hitl, CRIT_MSG, thread) if e["type"] == "approval_required")
    assert ap["thread_id"] == thread and ap["reason"] and "U05" in ap["message_preview"]
    assert "시뮬레이션" in ap["message_preview"] and "119" in ap["message_preview"]


def test_hitl_escalation_notice_is_in_the_report_before_any_approval(hitl):
    ev = run(hitl, CRIT_MSG, tid())
    assert ESCALATION_MESSAGE in ev[-1]["report"] and ev[-1]["escalation"] is True
    assert SENT not in ev[-1]["report"] and NOT_SENT not in ev[-1]["report"]


def test_hitl_approve_true_sends_simulated_notification(hitl):
    thread = tid()
    run(hitl, CRIT_MSG, thread)
    with hitl.client() as c:
        r = approve(c, thread, True)
    j = r.json()
    assert r.status_code == 200 and j["notification"] == "sent_simulated" and j["approved"] is True
    assert j["report"].rstrip().endswith(SENT)
    assert j["approval"]["status"] == "approved" and j["approval"]["simulated"] is True


def test_hitl_approve_false_does_not_send_but_keeps_escalation_notice(hitl):
    thread = tid()
    run(hitl, CRIT_MSG, thread)
    with hitl.client() as c:
        j = approve(c, thread, False).json()
    assert j["notification"] == "not_sent" and j["report"].rstrip().endswith(NOT_SENT)
    assert ESCALATION_MESSAGE in j["report"] and j["report"].count("즉시 119") == 1
    assert j["approval"]["status"] == "denied"


def test_hitl_approved_report_has_the_fixed_texts_exactly_once(hitl):
    thread = tid()
    run(hitl, CRIT_MSG, thread)
    with hitl.client() as c:
        rep = approve(c, thread, True).json()["report"]
    assert rep.count("즉시 119") == 1 and rep.count("의료기기가 아니며") == 1 and rep.count("보호자 알림 발송됨") == 1


def test_hitl_second_approval_of_the_same_thread_is_409(hitl):
    thread = tid()
    run(hitl, CRIT_MSG, thread)
    with hitl.client() as c:
        assert approve(c, thread, True).status_code == 200
        assert approve(c, thread, False).status_code == 409


def test_hitl_approving_an_unknown_thread_is_409(hitl):
    with hitl.client() as c:
        assert approve(c, tid(), True).status_code == 409


@pytest.mark.parametrize("body", [
    {"thread_id": "../etc", "approved": True, "approval_id": "x"}, {"thread_id": "t1", "approval_id": "x"}, {"approved": True, "approval_id": "x"},
    {"thread_id": "t1", "approved": "maybe", "approval_id": "x"}, {"thread_id": "t1", "approved": True},   # approval_id 필수
    {},
])
def test_hitl_approve_rejects_malformed_bodies_with_422(hitl, body):
    with hitl.client() as c:
        assert c.post("/chat/approve", json=body).status_code == 422


def test_hitl_non_critical_run_never_requests_approval(hitl):
    thread = tid()
    ev = run(hitl, "U07 최근 7일 이상 점검해줘", thread)
    assert "approval_required" not in types(ev) and ev[-1]["escalation"] is False
    with hitl.client() as c:
        assert approve(c, thread, True).status_code == 409


def test_hitl_a_new_message_auto_denies_the_pending_request(hitl):
    thread = tid()
    run(hitl, CRIT_MSG, thread)
    ev = run(hitl, "그럼 운동할 때는 괜찮은 거야?", thread)
    assert "approval_required" not in types(ev)  # 후속 질문은 재요청하지 않는다
    assert ev[-1]["escalation"] is True and ESCALATION_MESSAGE in ev[-1]["report"]  # 안내문은 그대로
    with hitl.client() as c:
        assert approve(c, thread, True).status_code == 409  # 이미 거절로 정리됨


def test_hitl_reanalysis_replaces_the_old_request_with_a_new_one(hitl):
    thread = tid()
    run(hitl, CRIT_MSG, thread)
    ev = run(hitl, "다시 30일 점검해줘", thread)
    assert types(ev).count("approval_required") == 1
    with hitl.client() as c:
        assert approve(c, thread, True).status_code == 200
        assert approve(c, thread, True).status_code == 409


def test_hitl_shorter_window_without_critical_clears_the_need_for_approval(hitl):
    thread = tid()
    run(hitl, CRIT_MSG, thread)
    ev = run(hitl, "다시 7일 점검해줘", thread)
    assert "approval_required" not in types(ev) and ev[-1]["escalation"] is False


def test_hitl_chat_json_exposes_approval_required_only_when_pending(hitl):
    with hitl.client() as c:
        crit = c.post("/chat", json={"message": CRIT_MSG, "thread_id": tid()}).json()
        calm = c.post("/chat", json={"message": "U07 최근 7일 이상 점검해줘", "thread_id": tid()}).json()
    assert crit["approval_required"]["message_preview"] and "approval_required" not in calm


def test_hitl_chat_json_flow_can_be_approved(hitl):
    thread = tid()
    with hitl.client() as c:
        pending = c.post("/chat", json={"message": CRIT_MSG, "thread_id": thread}).json()["approval_required"]
        APPROVAL_IDS[thread] = pending["approval_id"]
        assert approve(c, thread, True).json()["notification"] == "sent_simulated"


def test_hitl_parallel_approvals_exactly_one_wins(hitl):
    thread = tid()
    run(hitl, CRIT_MSG, thread)

    def go(_):
        with hitl.client() as c:
            return approve(c, thread, True).status_code

    with ThreadPoolExecutor(3) as ex:
        codes = sorted(ex.map(go, range(3)))
    assert codes == [200, 409, 409]


def test_hitl_two_threads_have_independent_pending_requests(hitl):
    a, b = tid(), tid()
    run(hitl, CRIT_MSG, a)
    run(hitl, CRIT_MSG, b)
    with hitl.client() as c:
        assert approve(c, a, True).json()["notification"] == "sent_simulated"
        assert approve(c, b, False).json()["notification"] == "not_sent"


def test_hitl_client_disconnect_before_approval_does_not_break_the_thread(hitl):
    thread = tid()
    with hitl.client() as c:
        with c.stream("POST", "/chat/stream", json={"message": CRIT_MSG, "thread_id": thread}) as r:
            for line in r.iter_lines():
                if '"type": "approval_required"' in line:
                    APPROVAL_IDS[thread] = json.loads(line.split("data: ", 1)[1])["approval_id"]
                    break  # 승인 요청을 받은 뒤 done 을 읽지 않고 끊는다
    time.sleep(1.0)
    with hitl.client() as c:
        code = approve(c, thread, True).status_code
        assert code in (200, 409)
        assert c.get("/health").json()["status"] == "ok"


def test_hitl_multi_user_critical_run_requests_a_single_approval(hitl):
    ev = run(hitl, "U01이랑 U05 최근 30일 비교해줘", tid())
    assert types(ev).count("approval_required") == 1


def test_hitl_disabled_server_never_requests_approval_and_approve_is_409(http):
    thread = tid()
    ev = stream(http, CRIT_MSG, thread_id=thread)[1]
    assert "approval_required" not in types(ev) and ev[-1]["escalation"] is True
    assert approve(http, thread, True).status_code == 409


# --------------------------------------------------------------------------------------------- 사용량·재시도·/version


USAGE_KEYS = {"retries", "llm_calls", "total_tokens", "cost_usd", "by_tier", "calls"}


def test_done_event_carries_usage_with_contract_keys(http):
    ev = stream(http, CRIT_MSG)[1]
    assert USAGE_KEYS <= set(ev[-1]["usage"])


def test_usage_is_zero_when_no_llm_is_used(http):
    u = stream(http, CRIT_MSG)[1][-1]["usage"]
    assert u["llm_calls"] == 0 and u["total_tokens"] == 0 and u["cost_usd"] == 0 and u["calls"] == []


def test_done_event_has_a_run_id_uuid(http):
    ev = stream(http, "안녕")[1]
    uuid.UUID(ev[-1]["run_id"])


def test_chat_json_response_carries_usage_and_run_id(http):
    j = http.post("/chat", json={"message": CRIT_MSG}).json()
    assert USAGE_KEYS <= set(j["usage"]) and uuid.UUID(j["run_id"])


def test_fake_backend_reports_llm_calls_tokens_and_tiers(fake_llm):
    u = run(fake_llm, CRIT_MSG)[-1]["usage"]
    assert u["llm_calls"] >= 4 and u["total_tokens"] > 0 and u["cost_usd"] == 0
    assert {"fast", "quality"} <= set(u["by_tier"])
    assert {"route", "plan", "analysis", "writing"} <= {c["name"] for c in u["calls"]}


def test_fake_backend_usage_is_per_request_not_cumulative(fake_llm):
    a = run(fake_llm, CRIT_MSG)[-1]["usage"]["llm_calls"]
    b = run(fake_llm, CRIT_MSG)[-1]["usage"]["llm_calls"]
    assert a == b


def test_fake_backend_health_and_version_identify_the_backend(fake_llm):
    with fake_llm.client() as c:
        assert c.get("/health").json()["llm_backend"] == "fake" and c.get("/health").json()["llm"] is True
        assert c.get("/version").json()["llm_backend"] == "fake"


def test_fake_backend_still_enforces_escalation_and_disclaimer(fake_llm):
    ev = run(fake_llm, CRIT_MSG)
    assert ESCALATION_MESSAGE in ev[-1]["report"] and "의료기기가 아니며" in ev[-1]["report"]
    assert tokens(ev) == ev[-1]["report"]


def test_fault_injection_first_tool_call_is_retried_and_visible(fault):
    ev = run(fault, CRIT_MSG)
    retried = [e for e in ev if e["type"] == "task_completed" and e.get("retries", 0) >= 1]
    assert retried, [e for e in ev if e["type"] == "task_completed"]
    assert ev[-1]["usage"]["retries"] >= 1 and "task_failed" not in types(ev)


def test_fault_injection_run_still_ends_with_correct_escalation(fault):
    ev = run(fault, CRIT_MSG)
    assert ev[-1]["escalation"] is True and ESCALATION_MESSAGE in ev[-1]["report"]


def test_fault_injection_only_hits_the_first_tool_call_of_the_process(fault):
    run(fault, CRIT_MSG)  # 이미 소비됐을 수도 있으므로 한 번 더 돌려 확인
    ev = run(fault, CRIT_MSG)
    assert ev[-1]["usage"]["retries"] == 0 and all(e.get("retries", 0) == 0 for e in ev if e["type"] == "task_completed")


def test_version_endpoint_shape_and_no_secrets(http):
    j = http.get("/version").json()
    assert {"supervisor_version", "llm_backend", "fast_model", "quality_model", "checkpointer", "git_commit"} <= set(j)
    blob = json.dumps(j)
    assert not re.search(r"sk-|api[_-]?key|secret|token", blob, re.I)
    assert j["supervisor_version"] == "v2" and j["llm_backend"] == "none" and j["checkpointer"] == "memory"


def test_version_git_commit_looks_like_a_short_hash_or_unknown(http):
    assert re.fullmatch(r"[0-9a-f]{7,}|unknown", http.get("/version").json()["git_commit"])


def test_health_exposes_supervisor_version_and_backend(http):
    h = http.get("/health").json()
    assert h["supervisor_version"] == "v2" and h["llm_backend"] == "none"


def test_v1_baseline_reports_its_version_everywhere(v1):
    with v1.client() as c:
        assert c.get("/version").json()["supervisor_version"] == "v1"
        assert c.get("/health").json()["supervisor_version"] == "v1"


def test_v1_baseline_has_no_safety_gate(v1):
    """v1 은 Before/After 대조군(실사용 금지)이다. 안전 게이트가 없다는 것이 v2 와의 차이이며, 이 테스트가 그 차이를 고정한다."""
    ev = run(v1, CRIT_MSG)
    assert "safety_escalation" not in types(ev) and ev[-1]["type"] == "done"
    assert ev[-1]["escalation"] is False


def test_v1_baseline_does_not_crash_on_basic_flows(v1):
    for m in ("안녕", "U07 최근 7일 이상 점검해줘", "휴식기 심박 정상 범위가 어떻게 돼?"):
        assert run(v1, m)[-1]["type"] == "done"


# --------------------------------------------------------------------------------------------- 여러 사용자 비교


def test_multi_user_comparison_runs_per_user_tasks(http):
    ev = stream(http, "U01이랑 U05 최근 30일 비교해줘")[1]
    ids = {e["task_id"] for e in ev if e["type"] in ("task_started", "task_completed")}
    assert {"r_summary:U01", "r_detect:U01", "r_summary:U05", "r_detect:U05"} <= ids
    assert "task_failed" not in types(ev)


def test_multi_user_comparison_report_has_sections_per_user_and_a_comparison_table(http):
    r = stream(http, "U01이랑 U05 최근 30일 비교해줘")[1][-1]["report"]
    assert "## 심박 점검 결과 (U01)" in r and "## 심박 점검 결과 (U05)" in r and "## 사용자 비교" in r


def test_multi_user_escalation_names_the_affected_users(http):
    ev = stream(http, "U01이랑 U05 최근 30일 비교해줘")[1]
    sev = next(e for e in ev if e["type"] == "safety_escalation")
    assert "해당 사용자: U01, U05" in sev["reason"]
    assert "(해당 사용자: U01, U05)" in ev[-1]["report"] and ev[-1]["escalation"] is True


def test_multi_user_escalation_names_only_the_critical_user(http):
    ev = stream(http, "U05랑 U07 최근 30일 비교해줘")[1]  # U07 은 critical 없음
    assert "(해당 사용자: U05)" in ev[-1]["report"] and "U07)" not in ev[-1]["report"].split("사용자 비교")[0].split("중요 안내")[1][:400]


def test_multi_user_anomalies_are_tagged_with_their_user(http):
    j = http.post("/chat", json={"message": "U01이랑 U05 최근 30일 비교해줘"}).json()
    assert {a["user_id"] for a in j["anomalies"]} == {"U01", "U05"}


def test_multi_user_is_limited_to_three_and_says_who_was_excluded(http):
    ev = stream(http, "U01, U03, U05, U07 최근 7일 비교해줘")[1]
    r = ev[-1]["report"]
    assert "최대 3명" in r and "U07" in r.split("최대 3명")[1][:60]
    users = {e["task_id"].split(":")[1] for e in ev if e["type"] == "task_started" and ":" in e.get("task_id", "")}
    assert users == {"U01", "U03", "U05"}


def test_multi_user_with_an_unknown_user_is_rejected_before_any_research(http):
    ev = stream(http, "U05랑 U99 비교해줘")[1]
    assert ev[0]["route"] == "simple" and "task_started" not in types(ev) and "plan_created" not in types(ev)
    assert "U99" in ev[-1]["report"] and "찾을 수 없습니다" in ev[-1]["report"]


def test_multi_user_duplicate_mentions_are_deduplicated(http):
    ev = stream(http, "U05랑 U05 최근 30일 비교해줘")[1]
    assert "## 사용자 비교" not in ev[-1]["report"] and ev[-1]["escalation"] is True


def test_multi_user_follow_up_keeps_the_user_scoped_escalation(http):
    thread = tid()
    stream(http, "U01이랑 U05 최근 30일 비교해줘", thread_id=thread)
    ev = stream(http, "그럼 운동할 때는 괜찮은 거야?", thread_id=thread)[1]
    assert ev[0]["route"] == "simple" and ev[-1]["escalation"] is True


def test_multi_user_replan_uses_the_worst_covered_user(http):
    ev = stream(http, "U01이랑 U05 최근 7일 비교해줘")[1]  # U01 커버리지 < 50%
    assert any(e["type"] == "replan" for e in ev)


def test_multi_user_clean_pair_has_no_escalation(http):
    ev = stream(http, "U07이랑 U12 최근 7일 비교해줘")[1]
    assert ev[-1]["escalation"] is False and "## 사용자 비교" in ev[-1]["report"]


def test_multi_user_report_passes_validation_even_if_one_user_has_no_anomaly(http):
    """한 사용자는 이상 0건, 다른 사용자는 이상이 있는 비교: 코드가 만든 문장이 검증기에 걸려 반려되면 안 된다."""
    ev = stream(http, "U01, U03, U05 최근 7일 비교해줘")[1]
    last = [e for e in ev if e["type"] == "validation"][-1]
    assert last["passed"] is True, last


# --------------------------------------------------------------------------------------------- 검색(RAG)·MCP Prompt 이벤트


def test_guideline_question_emits_retrieval_task_events(http):
    ev = stream(http, "휴식기 심박 정상 범위가 어떻게 돼?")[1]
    t = [(e["type"], e["task_id"], e["task_type"]) for e in ev if e["type"].startswith("task_")]
    assert ("task_started", "retrieval", "retrieval") in t and ("task_completed", "retrieval", "retrieval") in t
    done = next(e for e in ev if e["type"] == "task_completed" and e["task_id"] == "retrieval")
    assert re.search(r"근거 \d+건", done["summary"])


def test_guideline_question_stays_on_the_simple_path_without_data_calls(http):
    ev = stream(http, "휴식기 심박 정상 범위가 어떻게 돼?")[1]
    assert ev[0]["route"] == "simple"
    assert not {"plan_created", "anomalies_found", "safety_escalation"} & set(types(ev))


def test_guideline_answer_cites_its_sources(http):
    r = stream(http, "휴식기 심박 정상 범위가 어떻게 돼?")[1][-1]["report"]
    assert "※ 출처" in r and "heart-rate-ranges" not in r and "60" in r   # 내부 문서 ID 대신 제목만


def test_guideline_answer_does_not_cut_a_source_sentence_in_the_middle(http):
    """검색 근거 발췌가 문장 중간에서 끊겨 보이면 안 된다 (각 발췌는 문장/목록 항목 단위로 끝나야 한다)."""
    body = stream(http, "휴식기 심박 정상 범위가 어떻게 돼?")[1][-1]["report"].split("※ 출처")[0]
    for chunk in re.split(r"\n\n(?=\[)", body):
        last = chunk.strip().splitlines()[-1].rstrip()
        assert re.search(r"[.!?다요음함임|)\]`*」』…]$|\d$", last), f"발췌가 중간에 끊김: …{last[-40:]!r}"


def test_analysis_run_emits_the_mcp_prompt_task(http):
    ev = stream(http, CRIT_MSG)[1]
    started = [e for e in ev if e["type"] == "task_started" and e["task_type"] == "prompt"]
    done = [e for e in ev if e["type"] == "task_completed" and e["task_type"] == "prompt"]
    assert started and started[0]["task_id"] == "mcp_prompt" and done and "anomaly_review" in done[0]["summary"]


def test_mcp_prompt_task_comes_before_the_plan(http):
    t = types(stream(http, CRIT_MSG)[1])
    assert t.index("task_completed") < t.index("plan_created")


def test_greeting_and_out_of_scope_do_not_trigger_retrieval_or_prompt(http):
    for m in ("안녕", "오늘 점심 뭐 먹지?"):
        ev = stream(http, m)[1]
        assert not any(e.get("task_type") in ("retrieval", "prompt") for e in ev), m


def test_follow_up_may_search_guidelines_but_never_calls_data_tools(http):
    thread = tid()
    stream(http, CRIT_MSG, thread_id=thread)
    ev = stream(http, "그럼 운동할 때는 괜찮은 거야?", thread_id=thread)[1]
    assert not any(e.get("task_type") == "research" for e in ev) and "anomalies_found" not in types(ev)


def test_new_task_types_are_valid_sse_frames(http):
    frames, ev = stream(http, "휴식기 심박 정상 범위가 어떻게 돼?")
    assert all(f.startswith("event: ") for f in frames) and {"retrieval"} <= {e.get("task_type") for e in ev}


# --------------------------------------------------------------------------------------------- 대화 기억


def test_chat_memory_remembers_the_users_name_within_a_thread(http):
    thread = tid()
    first = stream(http, "제 이름은 민수입니다", thread_id=thread)[1][-1]["report"]
    assert "민수" in first
    second = stream(http, "제 이름이 뭐였죠?", thread_id=thread)[1][-1]["report"]
    assert "민수" in second


def test_chat_memory_does_not_leak_a_name_to_another_thread(http):
    stream(http, "제 이름은 철수입니다", thread_id=tid())
    other = stream(http, "제 이름이 뭐였죠?", thread_id=tid())[1][-1]["report"]
    assert "철수" not in other


def test_chat_memory_answer_has_no_plan_or_data_calls(http):
    thread = tid()
    stream(http, "제 이름은 민수입니다", thread_id=thread)
    ev = stream(http, "제 이름이 뭐였죠?", thread_id=thread)[1]
    assert not {"plan_created", "task_started", "anomalies_found"} & set(types(ev))


def test_chat_memory_survives_an_analysis_in_between(http):
    thread = tid()
    stream(http, "제 이름은 민수입니다", thread_id=thread)
    stream(http, CRIT_MSG, thread_id=thread)
    assert "민수" in stream(http, "제 이름이 뭐였죠?", thread_id=thread)[1][-1]["report"]


def test_chat_memory_latest_name_wins(http):
    thread = tid()
    stream(http, "제 이름은 민수입니다", thread_id=thread)
    stream(http, "제 이름은 영희입니다", thread_id=thread)
    r = stream(http, "제 이름이 뭐였죠?", thread_id=thread)[1][-1]["report"]
    assert "영희" in r and "민수" not in r


def test_chat_memory_unknown_name_is_not_invented(http):
    r = stream(http, "제 이름이 뭐였죠?", thread_id=tid())[1][-1]["report"]
    assert not re.search(r"[가-힣]{2,4}님이라고 말씀하셨", r)


def test_chat_memory_name_with_markup_is_not_executed_or_echoed_as_html(http):
    thread = tid()
    stream(http, "제 이름은 <script>alert(1)</script>입니다", thread_id=thread)
    r = stream(http, "제 이름이 뭐였죠?", thread_id=thread)[1][-1]["report"]
    assert "<script>" not in r


def test_chat_memory_does_not_lower_the_safety_state(http):
    thread = tid()
    stream(http, CRIT_MSG, thread_id=thread)
    stream(http, "제 이름은 민수입니다", thread_id=thread)
    ev = stream(http, "그럼 괜찮은 거야?", thread_id=thread)[1]
    assert ev[-1]["escalation"] is True


def test_chat_memory_persists_across_a_server_restart_with_sqlite(tmp_path):
    db = str(tmp_path / "ck.sqlite")
    thread = tid()
    with Server(CHECKPOINTER="sqlite", SQLITE_PATH=db) as a:
        run(a, "제 이름은 민수입니다", thread)
        assert run(a, CRIT_MSG, thread)[-1]["escalation"] is True
    with Server(CHECKPOINTER="sqlite", SQLITE_PATH=db) as b:  # 프로세스 재시작
        assert "민수" in run(b, "제 이름이 뭐였죠?", thread)[-1]["report"]
        ev = run(b, "아까 결과 다시 요약해줘", thread)
        assert ev[0]["route"] == "simple" and "plan_created" not in types(ev)
        assert ev[-1]["escalation"] is True and ESCALATION_MESSAGE in ev[-1]["report"]


def test_memory_checkpointer_forgets_after_a_restart():
    thread = tid()
    with Server(CHECKPOINTER="memory") as a:
        run(a, "제 이름은 민수입니다", thread)
    with Server(CHECKPOINTER="memory") as b:
        assert "민수" not in run(b, "제 이름이 뭐였죠?", thread)[-1]["report"]


def test_sqlite_restart_keeps_a_pending_hitl_approval(tmp_path):
    db = str(tmp_path / "ck2.sqlite")
    thread = tid()
    with Server(CHECKPOINTER="sqlite", SQLITE_PATH=db, HITL_ENABLED="true") as a:
        assert "approval_required" in types(run(a, CRIT_MSG, thread))
    with Server(CHECKPOINTER="sqlite", SQLITE_PATH=db, HITL_ENABLED="true") as b:
        with b.client() as c:
            r = approve(c, thread, True)
        assert r.status_code == 200 and r.json()["notification"] == "sent_simulated"


def test_sqlite_version_endpoint_reports_the_checkpointer(tmp_path):
    with Server(CHECKPOINTER="sqlite", SQLITE_PATH=str(tmp_path / "v.sqlite")) as s:
        with s.client() as c:
            assert c.get("/version").json()["checkpointer"] == "sqlite"


# --------------------------------------------------------------------------------------------- 안심 표현 상시 차단 (스크립트 LLM)

REASSURANCES = [
    "걱정하지 않으셔도 됩니다.", "안심하셔도 됩니다.", "문제없습니다.", "전혀 이상이 없습니다.", "건강하십니다.",
    "아무 문제가 없어 보입니다.", "추가 검사는 필요 없습니다.", "병원에 가지 않으셔도 됩니다.",
]


def _xfail_some(claims, known: dict[str, str]):
    return [pytest.param(c, marks=pytest.mark.xfail(strict=True, reason=known[c])) if c in known else c for c in claims]


ANALYSIS_KNOWN: dict[str, str] = {}   # 아직 알려진 미수정 문구가 생기면 여기에 사유와 함께 추가
FOLLOW_UP_KNOWN: dict[str, str] = {}


@pytest.mark.parametrize("claim", _xfail_some(REASSURANCES, ANALYSIS_KNOWN))
def test_reassurance_never_reaches_the_client_in_an_analysis_report(claim):
    llm = FakeLLM(text={"writing": GOOD_BODY + "\n" + claim, "analysis": "해석"})
    with TestClient(create_app(FakeTools([CRITICAL]).toolbox(), llm, "memory")) as c:
        with c.stream("POST", "/chat/stream", json={"message": "U01 이상 점검해줘"}) as r:
            ev = parse_sse(r.read().decode())
    shown = ""
    for e in ev:
        if e["type"] == "token":
            shown = e["text"] if e["reset"] else shown + e["text"]
            assert claim not in shown, f"스트리밍 중 노출: {shown[-60:]!r}"
    assert claim not in ev[-1]["report"]


@pytest.mark.parametrize("claim", _xfail_some(REASSURANCES[:5], FOLLOW_UP_KNOWN))
def test_reassurance_never_reaches_the_client_in_a_follow_up_answer(claim):
    llm = FakeLLM(text={"writing": GOOD_BODY, "analysis": "해석", "simple_answer": claim})
    with TestClient(create_app(FakeTools([CRITICAL]).toolbox(), llm, "memory")) as c:
        for m in ("U01 이상 점검해줘", "그럼 괜찮은 거야?"):
            with c.stream("POST", "/chat/stream", json={"message": m, "thread_id": "ra1"}) as r:
                ev = parse_sse(r.read().decode())
    assert claim not in ev[-1]["report"] and ESCALATION_MESSAGE in ev[-1]["report"]


def test_reassurance_is_blocked_even_when_no_anomaly_was_found():
    llm = FakeLLM(text={"writing": GOOD_BODY + "\n앞으로도 걱정하지 않으셔도 됩니다.", "analysis": "해석"})
    with TestClient(create_app(FakeTools([]).toolbox(), llm, "memory")) as c:
        with c.stream("POST", "/chat/stream", json={"message": "U01 이상 점검해줘"}) as r:
            ev = parse_sse(r.read().decode())
    assert "걱정하지 않으셔도" not in ev[-1]["report"]


# --------------------------------------------------------------------------------------------- MCP 2서버 부분 저하


def _broken(name: str) -> dict:
    return {"transport": "stdio", "command": sys.executable, "args": ["-c", "import sys; sys.exit(3)"], "cwd": str(ROOT)}


def test_guidelines_server_down_analysis_still_works(monkeypatch):
    from app.agent import toolbox as tb

    real = tb.default_servers()
    cfg = {"vitals": real["vitals"], "guidelines": _broken("guidelines")}

    async def patched(stack, servers=None):
        return await orig(stack, cfg)

    orig = tb.connect_mcp
    monkeypatch.setattr("app.api.main.connect_mcp", patched)
    with TestClient(create_app(None, NullLLM(), "memory")) as c:
        assert c.get("/health").json()["mcp"] is True
        ev = parse_sse(c.post("/chat/stream", json={"message": CRIT_MSG}).text)
        assert ev[-1]["type"] == "done" and ev[-1]["escalation"] is True and ESCALATION_MESSAGE in ev[-1]["report"]
        assert c.get("/users").status_code == 200


def test_guidelines_server_down_simple_question_degrades_without_crashing(monkeypatch):
    from app.agent import toolbox as tb

    real = tb.default_servers()
    cfg = {"vitals": real["vitals"], "guidelines": _broken("guidelines")}
    orig = tb.connect_mcp

    async def patched(stack, servers=None):
        return await orig(stack, cfg)

    monkeypatch.setattr("app.api.main.connect_mcp", patched)
    with TestClient(create_app(None, NullLLM(), "memory")) as c:
        ev = parse_sse(c.post("/chat/stream", json={"message": "휴식기 심박 정상 범위가 어떻게 돼?"}).text)
    assert ev[-1]["type"] == "done" and "error" not in types(ev) and DISCLAIMER_ANY.search(ev[-1]["report"])


def test_guidelines_server_down_disclaimer_is_still_present_in_analysis(monkeypatch):
    from app.agent import toolbox as tb

    real = tb.default_servers()
    cfg = {"vitals": real["vitals"], "guidelines": _broken("guidelines")}
    orig = tb.connect_mcp

    async def patched(stack, servers=None):
        return await orig(stack, cfg)

    monkeypatch.setattr("app.api.main.connect_mcp", patched)
    with TestClient(create_app(None, NullLLM(), "memory")) as c:
        ev = parse_sse(c.post("/chat/stream", json={"message": "U07 최근 7일 이상 점검해줘"}).text)
    assert DISCLAIMER_ANY.search(ev[-1]["report"])


def test_vitals_server_down_data_questions_fail_gracefully_but_guideline_questions_work(monkeypatch):
    from app.agent import toolbox as tb

    real = tb.default_servers()
    cfg = {"vitals": _broken("vitals"), "guidelines": real["guidelines"]}
    orig = tb.connect_mcp

    async def patched(stack, servers=None):
        return await orig(stack, cfg)

    monkeypatch.setattr("app.api.main.connect_mcp", patched)
    with TestClient(create_app(None, NullLLM(), "memory")) as c:
        simple = parse_sse(c.post("/chat/stream", json={"message": "휴식기 심박 정상 범위가 어떻게 돼?"}).text)
        assert simple[-1]["type"] == "done" and "60" in simple[-1]["report"]
        data = parse_sse(c.post("/chat/stream", json={"message": CRIT_MSG}).text)
        assert data[-1]["type"] == "done"
        assert "이상이 없" not in data[-1]["report"] and "감지되지 않았습니다" not in data[-1]["report"]  # 조회 실패를 이상 없음으로 보고하지 않는다


def test_all_mcp_servers_down_data_endpoints_are_503(monkeypatch):

    async def patched(stack, servers=None):
        raise RuntimeError("MCP 서버를 하나도 기동하지 못했습니다")

    monkeypatch.setattr("app.api.main.connect_mcp", patched)
    with TestClient(create_app(None, NullLLM(), "memory")) as c:
        assert c.get("/health").json()["mcp"] is False
        assert c.get("/users").status_code == 503 and c.post("/chat/stream", json={"message": "안녕"}).status_code == 503


# --------------------------------------------------------------------------------------------- 외부 데이터셋(Stanford, 로컬 전용)


@pytest.fixture(scope="module")
def stanford_users(http):
    us = [u for u in http.get("/users").json() if u.get("dataset") == "stanford"]
    if not us:
        pytest.skip("로컬 Stanford 외부 데이터셋 없음 (data/external/stanford)")
    return us


def test_users_carry_a_dataset_field(http):
    us = http.get("/users").json()
    assert all(u.get("dataset") in ("fitbit", "stanford") for u in us)
    assert all((u["dataset"] == "fitbit") == u["user_id"].startswith("U") for u in us)


def test_stanford_users_have_working_data_endpoints(http, stanford_users):
    uid = stanford_users[0]["user_id"]
    assert http.get(f"/users/{uid}/series", params={"lookback_days": 7}).json()["points"]
    assert "anomalies" in http.get(f"/users/{uid}/anomalies", params={"lookback_days": 30}).json()
    assert http.get(f"/users/{uid}/summary", params={"lookback_days": 7}).status_code == 200


def test_stanford_user_can_be_analyzed_via_the_request_user_id(http, stanford_users):
    uid = stanford_users[0]["user_id"]
    r = http.post("/chat", json={"message": "최근 30일 이상 점검해줘", "user_id": uid}).json()
    assert r["route"] == "analysis" and f"({uid})" in r["report"]


def test_stanford_user_id_in_the_message_is_recognized(http, stanford_users):
    uid = stanford_users[0]["user_id"]
    ev = stream(http, f"{uid} 최근 30일 이상 점검해줘")[1]
    assert ev[0]["route"] == "analysis"


def test_comparison_with_a_stanford_user_is_not_silently_truncated(http, stanford_users):
    uid = stanford_users[0]["user_id"]
    r = stream(http, f"U05랑 {uid} 최근 30일 비교해줘")[1][-1]["report"]
    assert uid in r


def test_events_endpoint_exists_and_is_empty_for_fitbit_users(http):
    r = http.get("/users/U05/events")
    assert r.status_code == 200 and r.json()["events"] == [] and r.json()["dataset"] == "fitbit"

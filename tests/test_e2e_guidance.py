"""코드가 쓰는 고정 안내(에스컬레이션·WARNING_GUIDANCE·NO_ANOMALY_GUIDANCE·면책)가 어느 경로에서도 정확히 한 번, 올바른 위치에 나오고
스트리밍 텍스트가 최종 리포트와 같은지 실제 서버(LLM 끔 / 결정론 fake 백엔드 두 가지)로 확인한다."""
import uuid

import pytest

from app.agent.gates import DIAGNOSIS_NOTICE, ESCALATION_MESSAGE, NO_ANOMALY_GUIDANCE, WARNING_GUIDANCE
from tests.test_e2e_api import stream, tokens, types
from tests.test_e2e_features import Server

DISC = "이 서비스는 의료기기가 아니며"


@pytest.fixture(scope="module")
def fake_server():
    s = Server(LLM_BACKEND="fake", LLM_ENABLED="true").start()
    try:
        yield s
    finally:
        s.stop()


@pytest.fixture(params=["llm_off", "fake_backend"])
def ask(request, http):
    """(메시지, thread_id) -> 이벤트 목록. 기본 서버(LLM 끔)와 fake 백엔드 서버 양쪽에서 같은 단언을 돌린다."""
    if request.param == "llm_off":
        return lambda m, t=None: stream(http, m, thread_id=t)[1]
    srv = request.getfixturevalue("fake_server")

    def go(m, t=None):
        with srv.client() as c:
            return stream(c, m, thread_id=t)[1]

    return go


def tid() -> str:
    return f"g-{uuid.uuid4().hex[:8]}"


def n(report: str, text: str) -> int:
    return report.count(text)


def test_critical_report_has_the_escalation_once_and_no_other_guidance(ask):
    ev = ask("U05 최근 30일 이상 점검해줘")
    r = ev[-1]["report"]
    assert n(r, ESCALATION_MESSAGE) == 1 and n(r, WARNING_GUIDANCE) == 0 and n(r, NO_ANOMALY_GUIDANCE) == 0 and n(r, DISC) == 1
    assert r.lstrip().startswith("**중요 안내**") and r.index(ESCALATION_MESSAGE) < r.index("[심각]")


def test_critical_report_body_is_readable_facts_with_severity_and_time(ask):
    r = ask("U05 최근 30일 이상 점검해줘")[-1]["report"]
    assert "[심각]" in r and "146 bpm" in r and "28분" in r and "안정 시 빈맥" in r


def test_warning_report_shows_the_warning_guidance_once_before_the_disclaimer(ask):
    r = ask("U05 최근 7일 이상 점검해줘")[-1]["report"]
    assert n(r, WARNING_GUIDANCE) == 1 and n(r, ESCALATION_MESSAGE) == 0 and n(r, NO_ANOMALY_GUIDANCE) == 0
    assert r.index(WARNING_GUIDANCE) < r.index(DISC)


def test_no_anomaly_report_shows_the_no_anomaly_guidance_once_before_the_disclaimer(ask):
    r = ask("U12 최근 7일 이상 점검해줘")[-1]["report"]
    assert n(r, NO_ANOMALY_GUIDANCE) == 1 and n(r, WARNING_GUIDANCE) == 0 and n(r, ESCALATION_MESSAGE) == 0
    assert r.index(NO_ANOMALY_GUIDANCE) < r.index(DISC)


def test_general_health_question_shows_the_no_anomaly_guidance_once(ask):
    r = ask("휴식기 심박 정상 범위가 어떻게 돼?")[-1]["report"]
    assert n(r, NO_ANOMALY_GUIDANCE) == 1 and r.index(NO_ANOMALY_GUIDANCE) < r.index(DISC) and "※ 출처" in r


def test_info_only_report_adds_neither_warning_nor_no_anomaly_guidance(ask):
    r = ask("U07 최근 7일 이상 점검해줘")[-1]["report"]  # 센서 결측(참고)만
    assert n(r, WARNING_GUIDANCE) == 0 and n(r, NO_ANOMALY_GUIDANCE) == 0 and n(r, ESCALATION_MESSAGE) == 0


@pytest.mark.parametrize("first,guidance,absent", [
    ("U05 최근 30일 이상 점검해줘", ESCALATION_MESSAGE, (WARNING_GUIDANCE, NO_ANOMALY_GUIDANCE)),
    ("U05 최근 7일 이상 점검해줘", WARNING_GUIDANCE, (ESCALATION_MESSAGE, NO_ANOMALY_GUIDANCE)),
    ("U12 최근 7일 이상 점검해줘", NO_ANOMALY_GUIDANCE, (ESCALATION_MESSAGE, WARNING_GUIDANCE)),
])
def test_follow_up_carries_the_same_guidance_exactly_once(ask, first, guidance, absent):
    t = tid()
    ask(first, t)
    r = ask("그럼 운동할 때는 괜찮은 거야?", t)[-1]["report"]
    assert n(r, guidance) == 1 and all(n(r, a) == 0 for a in absent) and n(r, DISC) == 1


def test_second_follow_up_does_not_duplicate_any_fixed_text(ask):
    t = tid()
    ask("U05 최근 7일 이상 점검해줘", t)
    ask("그럼 운동할 때는 괜찮은 거야?", t)
    r = ask("그러면 수면 중엔?", t)[-1]["report"]
    assert n(r, WARNING_GUIDANCE) == 1 and n(r, DISC) == 1


def test_three_user_comparison_with_critical_has_one_scoped_escalation(ask):
    ev = ask("U01, U05, U07 최근 30일 비교해줘")
    r = ev[-1]["report"]
    assert n(r, ESCALATION_MESSAGE) == 1 and "(해당 사용자: U01, U05)" in r and n(r, DISC) == 1
    assert n(r, WARNING_GUIDANCE) == 0


def test_three_user_comparison_with_warnings_only_has_the_warning_guidance_once(ask):
    r = ask("U03, U05, U12 최근 7일 비교해줘")[-1]["report"]
    assert n(r, WARNING_GUIDANCE) == 1 and n(r, ESCALATION_MESSAGE) == 0 and r.index(WARNING_GUIDANCE) < r.index(DISC)


def test_diagnosis_refusal_notice_appears_once_first(ask):
    r = ask("저 부정맥인가요? 약 먹어야 하나요?")[-1]["report"]
    assert n(r, DIAGNOSIS_NOTICE) == 1 and r.lstrip().startswith("요청하신 진단")


@pytest.mark.parametrize("msg", [
    "U05 최근 30일 이상 점검해줘", "U05 최근 7일 이상 점검해줘", "U12 최근 7일 이상 점검해줘", "휴식기 심박 정상 범위가 어떻게 돼?",
    "안녕, 너는 누구야?", "오늘 점심 뭐 먹지?", "U01, U05, U07 최근 30일 비교해줘", "저 부정맥인가요? 약 먹어야 하나요?",
])
def test_streamed_text_equals_the_final_report(ask, msg):
    ev = ask(msg)
    if any(e["type"] == "token" for e in ev):
        assert tokens(ev) == ev[-1]["report"]
    assert types(ev)[-1] == "done"


def test_streamed_text_equals_the_final_report_for_a_follow_up(ask):
    t = tid()
    ask("U05 최근 30일 이상 점검해줘", t)
    ev = ask("그럼 운동할 때는 괜찮은 거야?", t)
    if any(e["type"] == "token" for e in ev):
        assert tokens(ev) == ev[-1]["report"]


def test_fake_backend_three_user_comparison_labels_each_users_findings(fake_server):
    """LLM 이 켜진 경로(fake 백엔드)의 비교 리포트가 사용자별 내용을 구분해야 한다 (실제 LLM 으로도 확인할 것)."""
    with fake_server.client() as c:
        r = stream(c, "U01, U05, U07 최근 30일 비교해줘")[1][-1]["report"]
    body = r.split(DISC)[0].split("(해당 사용자: U01, U05)")[-1]
    assert all(u in body for u in ("U01", "U05", "U07")), body[:300]

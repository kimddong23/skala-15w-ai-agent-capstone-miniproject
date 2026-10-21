"""안전·정합성 회귀: 비교 리포트 사용자별 판정, 후속 답변 안심 차단, 진료 불필요 서술, 비교 본문의 사용자 구분, 승인 카드 내부 코드 비노출.

협력: `tests/test_multi_user.py`(PerUserTools), `tests/fakes.py`, `app.agent.{graph,nodes,validator}`.
"""
import pytest

from app.agent import gates
from tests import test_multi_user as mu
from tests.fakes import FakeLLM, assemble


@pytest.mark.asyncio
async def test_comparison_with_a_clean_user_passes_validation_without_template_fallback(monkeypatch):
    monkeypatch.setitem(mu.PER_USER, "U01", [])   # U01 은 이상 없음, U05 는 critical
    writing = ("## U01\n이번 기간 데이터에서는 임계값을 넘는 구간이 감지되지 않았습니다.\n\n"
               "## U05\n휴식 중 빈맥이 감지되었습니다. 평균 심박은 66.6 bpm 입니다.")
    llm = FakeLLM(text={"writing": writing, "analysis": "비교 해석"})
    events, st = await mu.run(mu.make(mu.PerUserTools(), llm), "U01이랑 U05 비교해줘")
    assert [e["passed"] for e in events if e["type"] == "validation"] == [True]
    assert not st["validation"]["degraded"] and "U01" in st["report"] and assemble(events) == st["report"]


@pytest.mark.asyncio
async def test_an_all_clear_sentence_in_the_anomalous_users_section_is_still_rejected(monkeypatch):
    monkeypatch.setitem(mu.PER_USER, "U01", [])
    writing = ("## U01\n이번 기간 데이터에서는 임계값을 넘는 구간이 감지되지 않았습니다.\n\n"
               "## U05\n이번 기간 데이터에서는 임계값을 넘는 구간이 감지되지 않았습니다.")   # U05 에는 critical 이 있다
    llm = FakeLLM(text={"writing": writing, "analysis": "비교 해석"})
    events, st = await mu.run(mu.make(mu.PerUserTools(), llm), "U01이랑 U05 비교해줘")
    assert any(e["type"] == "validation" and not e["passed"] for e in events)   # U05 절의 '구간 없음'은 반려된다
    assert st["validation"]["degraded"] and gates.ESCALATION_MESSAGE in st["report"]   # 재시도를 소진하고 템플릿으로 대체
    vals = [e for e in events if e["type"] == "validation"]
    assert vals[-1]["degraded"] and not any(v["degraded"] for v in vals[:-1])   # UI 가 '대체 문안 사용'을 띄울 수 있게 마지막 검증만 알린다


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["전혀 이상이 없습니다. 문제없습니다.", "병원에 가지 않으셔도 됩니다.", "걱정하지 마세요. 괜찮습니다."])
async def test_follow_up_answer_after_a_critical_analysis_cannot_reassure(bad):
    llm = FakeLLM(text={"writing": "휴식 중 빈맥이 감지되었습니다. 평균 심박은 66.6 bpm 입니다.", "analysis": "해석", "simple_answer": bad})
    graph = mu.make(mu.PerUserTools(), llm)
    await mu.run(graph, "U05 최근 7일 점검해줘", thread="f1")
    events, st = await mu.run(graph, "그럼 운동할 때는 괜찮아?", thread="f1")
    report = st["report"]
    assert st["route"] == "simple" and gates.ESCALATION_MESSAGE in report   # 후속 답변에도 안내문 유지
    assert bad.split(".")[0] not in report and "문제없" not in report and "가지 않으셔도" not in report
    assert assemble(events) == report   # 화면에 보인 최종 텍스트도 같다 (위반 문장이 남지 않음)


@pytest.mark.parametrize("msg", ["아까 결과 다시 요약해줘", "이전 점검 결과를 다시 설명해줘", "지난번 리포트 한 번 더 정리해줘", "앞서 분석 내용 요약해 줘", "방금 결과 다시 알려줘"])
def test_references_to_the_previous_result_are_follow_ups_not_new_checks(msg):
    assert not gates.asks_new_check(msg, "U05")
    assert gates.route_override("analysis", msg, True, "U05")[0] == "simple"


@pytest.mark.parametrize("msg", ["다시 점검해줘", "U05 최근 3일 다시 분석해줘", "아까 결과 말고 U02 점검해줘", "아까 결과 다시 7일로 점검해줘", "리포트 다시 만들어줘"])
def test_genuine_new_checks_still_re_run(msg):
    assert gates.asks_new_check(msg, "U05")


@pytest.mark.asyncio
async def test_summarize_request_keeps_the_escalation_and_does_not_query_tools_again():
    tools = mu.PerUserTools()
    llm = FakeLLM(text={"writing": "휴식 중 빈맥이 감지되었습니다. 평균 심박은 66.6 bpm 입니다.", "analysis": "해석", "simple_answer": "직전 점검에서 휴식 중 빈맥이 감지되었습니다."})
    graph = mu.make(tools, llm)
    await mu.run(graph, "U05 최근 7일 점검해줘", thread="b20")
    calls = len(tools.detect_calls)
    _, st = await mu.run(graph, "아까 결과 다시 요약해줘", thread="b20")
    assert st["route"] == "simple" and len(tools.detect_calls) == calls and gates.ESCALATION_MESSAGE in st["report"]


@pytest.mark.asyncio
async def test_unknown_user_reply_lists_only_the_current_dataset():
    tools = mu.PerUserTools()
    tools.extra_users = [{"user_id": f"S{i:03d}", "dataset": "stanford", "has_sleep": True, "start": "2020-12-01", "end": "2021-02-01", "days": 60, "coverage_pct": 70.0}
                         for i in range(1, 4)]

    async def users(dataset=None):
        return [{"user_id": u, "dataset": "fitbit"} for u in mu.PER_USER] + tools.extra_users

    tools._users = users
    _, st = await mu.run(mu.make(tools), "U99 이상 점검해줘")
    assert "U01" in st["report"] and "S001" not in st["report"] and "Stanford" in st["report"] and "전환해 선택" in st["report"]


# ---- fake 백엔드 비교 본문의 사용자 구분, 승인 카드의 내부 코드 비노출 ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fake_backend_comparison_gives_each_user_their_own_section_and_numbers():
    from app.agent.fake_llm import FakeBackend

    events, st = await mu.run(mu.make(mu.PerUserTools(), FakeBackend()), "U01, U05, U02 비교해줘")
    body = st["report"].split(gates.DISCLAIMER)[0]
    for uid in ("U01", "U05", "U02"):
        assert f"## {uid}" in body, uid
    u01, u05 = body.split("## U05")[0], body.split("## U05")[1]
    assert "평균 심박 70.1 bpm" in u01 and "평균 심박 66.6 bpm" in u05   # 사용자마다 자기 수치 (첫 사용자 것만 나오지 않는다)
    assert "135.0 bpm" not in u01 and not st["validation"]["degraded"] and assemble(events) == st["report"]


def test_safety_reason_and_guardian_preview_use_korean_type_labels():
    from app.agent import report as rpt

    crit = {"type": "resting_tachycardia", "severity": "critical", "user_id": "U05"}
    reason = gates.safety_gate([crit]).reason
    assert "안정 시 빈맥" in reason and "resting_tachycardia" not in reason and reason.startswith("critical")
    assert "resting_tachycardia" not in rpt.guardian_preview({"anomalies": [crit], "user_id": "U05"}, reason)

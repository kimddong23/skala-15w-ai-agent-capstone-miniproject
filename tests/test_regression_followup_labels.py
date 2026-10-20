"""후속 답변이 실제 질문에 답하는지, 단정적 안심 차단, plan 기간 표시, 내부 코드 비노출, 유형 라벨 회귀 테스트.

협력: `tests/fakes.py`, `app.agent.*`.
"""
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.agent import gates
from app.agent import report as rpt
from app.agent.graph import build_graph
from app.agent.llm import NullLLM
from app.agent.validator import validate_rules
from tests.fakes import CRITICAL, WARNING, FakeLLM, FakeTools, assemble

SECTIONED_GUIDES = {
    "guideline://heart-rate-ranges": "# 심박수 가이드\n\n## 안정시 심박수\n일반 성인의 안정시 심박수 정상 범위는 분당 60~100회입니다.\n\n## 운동 중 심박수\n운동 중에는 심박이 크게 오르는 것이 정상입니다.",
    "guideline://red-flags": "# 위험 신호\n\n## 응급 증상\n흉통, 호흡곤란, 실신이 동반되면 즉시 응급 진료가 필요합니다.",
    "guideline://disclaimer": "※ 이 내용은 웨어러블 심박 데이터에 대한 참고용 분석이며 의학적 진단이 아닙니다. 증상이 있거나 걱정되면 의료진과 상담하세요.",
}


class SectionedTools(FakeTools):
    """섹션(##)이 있는 가이드라인을 돌려주는 도구 (실제 가이드라인 문서와 같은 구조)."""

    async def _resource(self, uri):
        return SECTIONED_GUIDES[uri]


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr("app.agent.retry.random.uniform", lambda a, b: 0)


def make(tools=None, llm=None):
    tools = tools or FakeTools()
    return build_graph(tools.toolbox(), InMemorySaver(), llm or NullLLM()), tools


async def run(graph, message, thread="t1"):
    cfg = {"configurable": {"thread_id": thread}, "recursion_limit": 40}
    events = [c async for c in graph.astream({"messages": [("user", message)]}, cfg, stream_mode="custom")]
    return events, (await graph.aget_state(cfg)).values


def rules(text, **kw):
    kw.setdefault("escalation", False)
    kw.setdefault("escalation_message", "")
    return validate_rules(text + "\n" + gates.DISCLAIMER, [60.0, 100.0], **kw)  # 가이드라인 수치만 사실로 허용


# ---- 단정적 안심 --------------------------------------------------------------------------
REASSURING = [
    "이상이 없으므로 추가 검사 필요 없음", "정상 범위에 있으며 이상 없으므로 추가적인 주의가 필요하지 않습니다.", "주의할 필요 없습니다.",
    "신경 쓰지 않아도 됩니다.", "신경 쓰실 필요 없어요.", "추가적인 조치는 불필요합니다.", "걱정하지 않아도 됩니다.", "걱정할 필요 없습니다.", "안심하세요.", "전반적으로 정상입니다.",
    "모든 수치가 정상입니다.", "심박은 정상입니다.", "건강합니다.", "추가 검진은 불필요합니다.",
]
NOT_REASSURING = [
    "운동 중 심박 상승은 정상입니다.", "휴식기 심박 정상 범위는 60~100회입니다.",
    "이번 기간 데이터에서는 임계값을 넘는 구간이 감지되지 않았습니다. 다만 감지되지 않은 구간까지 안전하다고 볼 수 없습니다.",
    "이상이 없다고 단정할 수 없습니다.", "이상이 없었다는 뜻은 아닙니다.",
]


@pytest.mark.parametrize("text", REASSURING)
def test_definitive_reassurance_is_blocked_regardless_of_data_state(text):
    assert any("단정적 안심" in i for i in rules(text)), text


@pytest.mark.parametrize("text", NOT_REASSURING)
def test_general_knowledge_and_limited_statements_are_allowed(text):
    assert rules(text, limited_data=True) == [], text


@pytest.mark.parametrize("text", ["이상은 없습니다.", "이상이 없습니다.", "문제 없어요."])
def test_no_anomaly_claim_is_blocked_when_data_is_limited(text):
    assert any("데이터가 제한적" in i for i in rules(text, limited_data=True))
    assert not any("데이터가 제한적" in i for i in rules(text, limited_data=False))


@pytest.mark.asyncio
async def test_llm_reassurance_with_low_coverage_degrades_to_limited_template():
    llm = FakeLLM(text={"writing": "이상이 없습니다. 걱정하지 않아도 됩니다."})
    graph, _ = make(FakeTools([], coverage=30.0), llm)
    events, st = await run(graph, "U01 이상 점검해줘")
    # 안심 문장은 삭제되고 '이상이 없습니다'는 한계가 있는 데이터의 사실 진술로 교정된다 (템플릿으로 떨어지지 않아도 거짓 안심은 남지 않는다)
    assert "걱정하지 않아도" not in st["report"] and "이상이 없습니다" not in st["report"]
    assert "감지되지 않았습니다" in st["report"] and "이상이 없다고 판단할 수 없습니다" in st["report"]


@pytest.mark.asyncio
async def test_sensor_gap_all_clear_is_dropped_when_an_anomaly_exists_and_hedged_wording_passes_the_rules():
    """센서 결측 이벤트도 '이상'이므로 '이상 없음'은 사실과 모순이다 → 이상이 있는 리포트의 허용 목록 필터가 지운다. 한정 문장은 규칙을 통과한다."""
    gap = {**WARNING, "anomaly_id": "G1", "type": "sensor_gap", "severity": "info", "context": "rest", "evidence": "심박 신호가 509분 끊겼습니다"}
    ok = "이번 기간 데이터에서는 임계값을 넘는 구간이 감지되지 않았습니다. 다만 센서 결측이 있어 감지되지 않은 구간까지 안전하다고 볼 수 없습니다."
    assert not [i for i in validate_rules(ok, [], escalation=False, escalation_message="", disclaimer="", any_anomalies=True, threshold_anomalies=False)
                if "이상이 감지되었는데" in i]   # 센서 결측만 있으면 '구간이 감지되지 않았다' 한정 문장은 사실이라 규칙을 통과한다
    assert any("이상이 감지되었는데" in i for i in validate_rules("이상이 없습니다.", [], escalation=False, escalation_message="", disclaimer="", any_anomalies=True))
    graph, _ = make(FakeTools([gap]), FakeLLM(text={"writing": ["이상이 없습니다.", ok]}))
    events, st = await run(graph, "U01 이상 점검해줘")
    assert "이상이 없습니다" not in assemble(events) and "이상이 없습니다" not in st["report"]   # 스트리밍 중에도 보이지 않는다
    assert [e["passed"] for e in events if e["type"] == "validation"] == [True] and not st["validation"]["degraded"]   # 지우는 것은 검증 실패가 아니다


@pytest.mark.asyncio
async def test_template_for_clean_data_has_no_limitation_but_limited_data_does():
    _, clean = await run(make(FakeTools([]))[0], "U01 이상 점검해줘")
    assert "감지되지 않았습니다" in clean["report"] and "판단할 수 없습니다" not in clean["report"] and clean["validation"]["passed"]
    _, limited = await run(make(FakeTools([], coverage=20.0))[0], "U01 이상 점검해줘")
    assert "이상이 없다고 판단할 수 없습니다" in limited["report"] and limited["validation"]["passed"]


# ---- 후속 답변은 실제 질문에 답한다 ---------------------------------------------------------
@pytest.mark.asyncio
async def test_followup_fallback_restates_prior_facts_and_picks_exercise_section():
    graph, _ = make(SectionedTools([CRITICAL]))
    await run(graph, "U01 최근 30일 이상 점검해줘", thread="b8")
    events, st = await run(graph, "그럼 운동할 때는 괜찮은 거야?", thread="b8")
    assert events[0]["route"] == "simple"
    report = st["report"]
    assert "135.0 bpm" in report and "직전 점검(U01)" in report  # 직전 사실을 다시 말한다
    assert "운동 중에는 심박이 크게 오르는 것이 정상" in report  # 질문과 관련된 가이드라인 섹션
    assert "안정시 심박수 정상 범위는 분당 60~100회" not in report  # 일반 표 덤프 금지


@pytest.mark.asyncio
async def test_llm_prompt_leads_with_the_relevant_guideline_section():
    llm = FakeLLM(text={"simple_answer": "운동 중에는 심박이 오르는 것이 정상이며, 직전 점검에서는 135.0 bpm 구간이 있었습니다."})
    graph, _ = make(SectionedTools([CRITICAL]), llm)
    await run(graph, "U01 최근 30일 이상 점검해줘", thread="b8b")
    await run(graph, "그럼 운동할 때는 괜찮은 거야?", thread="b8b")
    prompt = llm.prompts["simple_answer"][0]
    assert prompt.index("운동 중 심박수") < prompt.index("GUIDELINES(전체)") and "135.0 bpm" in prompt


def test_excerpt_matches_korean_inflections():
    docs = [SECTIONED_GUIDES["guideline://heart-rate-ranges"]]
    assert "운동 중에는" in rpt.guideline_excerpt(docs, "그럼 운동할 때는 괜찮은 거야?")
    assert "안정시 심박수" in rpt.guideline_excerpt(docs, "안정시 심박은 어떻게 재요?")


# ---- plan_created 는 실제 쓰일 기간을 보여준다 -------------------------------------------
@pytest.mark.asyncio
async def test_plan_created_reports_effective_lookback():
    graph, tools = make()
    await run(graph, "U01 최근 30일 이상 점검해줘", thread="b10")
    events, _ = await run(graph, "다시 점검해줘", thread="b10")
    plan = next(e for e in events if e["type"] == "plan_created")
    gate = next(e for e in events if e["type"] == "gate_applied")
    assert plan["lookback_days"] == gate["lookback_days"] == 30 and tools.summary_calls[1][1] == 30  # 이벤트 == 실제 조회 기간
    events, _ = await run(graph, "다시 14일 점검해줘", thread="b10")
    assert next(e for e in events if e["type"] == "plan_created")["lookback_days"] == 14
    other, _ = await run(graph, "U02 점검해줘", thread="b10")  # 다른 사용자는 이어받지 않는다
    assert next(e for e in other if e["type"] == "plan_created")["lookback_days"] == 7


def test_effective_lookback_precedence():
    assert gates.effective_lookback(7, "다시 점검해줘", 30)[0] == 30
    assert gates.effective_lookback(7, "지난 3일 다시 점검해줘", 30)[0] == 3
    assert gates.effective_lookback(7, "점검해줘", None)[0] == 7
    assert gates.effective_lookback(99, "점검해줘", None)[0] == 30


# ---- 내부 코드 노출 ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_facts_use_korean_context_labels_and_full_ranges():
    sleep_anomaly = {**CRITICAL, "context": "sleep"}
    llm = FakeLLM(text={"writing": "## 요약\n- 평균 심박 72.4 bpm"})
    graph, _ = make(FakeTools([sleep_anomaly]), llm)
    await run(graph, "U01 이상 점검해줘")
    prompt = llm.prompts["writing"][0]
    assert "수면 중" in prompt and "sleep" not in prompt and "rest" not in prompt.replace("restin", "")
    assert "2016-04-10 14:00~2016-04-10 14:25" in prompt  # 시작~종료 전체 시각


@pytest.mark.parametrize("text", ["수면 sleep 구간에서", "rest 상태에서 112 bpm", "resting_tachycardia 가 감지", "critical 이상"])
def test_validator_flags_raw_internal_codes(text):
    assert any("내부 코드 노출" in i for i in rules(text))


# ---- 라벨: 안정 시 빈맥 + 맥락 분리 ---------------------------------------------------------------
@pytest.mark.asyncio
async def test_label_renamed_and_context_shown_separately_for_sleep_events():
    sleep_anomaly = {**CRITICAL, "context": "sleep", "evidence": "수면 중 25분간 평균 135.0 bpm"}
    graph, _ = make(FakeTools([sleep_anomaly]))
    _, st = await run(graph, "U01 이상 점검해줘")
    line = next(l for l in st["report"].splitlines() if "135.0" in l)
    assert "안정 시 빈맥" in line and "(수면 중)" in line and "휴식 중 빈맥" not in st["report"]
    assert gates.TYPE_LABELS["resting_tachycardia"] == "안정 시 빈맥"


# ---- 데이터 한계 안내는 코드가 강제하는 고정 문구 -------------------------------------------------
GAP = {**WARNING, "anomaly_id": "G1", "type": "sensor_gap", "severity": "info", "context": "rest", "duration_min": 2400,
       "evidence": "심박 신호가 2400분 끊겼습니다"}      # 7일 구간의 20% 이상이어야 데이터 한계로 센서 결측을 센다


@pytest.mark.parametrize("tools,expected", [
    (lambda: FakeTools([], coverage=20.0), "데이터 커버리지 20%"),
    (lambda: FakeTools([GAP]), "센서 결측 1건"),
])
@pytest.mark.asyncio
async def test_limit_notice_is_code_enforced_with_real_numbers_and_appears_once(tools, expected):
    copied = "※ 이번 기간은 기록이 제한적이어서(임의 문구) 이상이 없다고 판단할 수 없습니다.\n이번 기간 데이터에서는 임계값을 넘는 구간이 감지되지 않았습니다."
    for llm in (NullLLM(), FakeLLM(text={"writing": copied})):  # LLM 이 안내를 베껴 써도, 빠뜨려도 정확히 한 번
        graph, _ = make(tools(), llm)
        events, st = await run(graph, "U01 이상 점검해줘")
        assert st["report"].count("이상이 없다고 판단할 수 없습니다") == 1 and expected in st["report"]
        assert "임의 문구" not in st["report"] and st["validation"]["passed"] and not st["validation"]["degraded"]


@pytest.mark.asyncio
async def test_limit_notice_for_detection_failure_and_followups():
    tools = FakeTools([], coverage=20.0)
    graph, _ = make(tools)
    await run(graph, "U01 최근 7일 이상 점검해줘", thread="ln")
    _, follow = await run(graph, "그럼 괜찮은 거야?", thread="ln")  # 후속 답변도 직전 점검의 한계를 잊지 않는다
    assert follow["report"].count("이상이 없다고 판단할 수 없습니다") == 1
    broken = FakeTools()
    broken.detect_errors = [ValueError("x")] * 6
    _, failed = await run(make(broken)[0], "U01 이상 점검해줘")
    assert failed["report"].count("※ 이상 탐지를 실행하지 못해 이상 여부를 판단할 수 없습니다.") == 1


def test_limit_notice_absent_for_clean_full_data_and_validator_requires_it_when_set():
    state = {"detection": {"anomalies": []}, "anomalies": [], "summary": {"summary": {"coverage_pct": 90, "rest_minutes": 500}}}
    assert rpt.limitation_notice(state) == ""
    notice = "※ 이번 기간은 기록이 제한적이어서(데이터 커버리지 20%) 이상이 없다고 판단할 수 없습니다."
    assert any("데이터 한계 안내문" in i for i in validate_rules("본문\n" + gates.DISCLAIMER, [20.0], escalation=False, escalation_message="", limit_notice=notice))
    assert not any("데이터 한계 안내문" in i for i in validate_rules(notice + "\n본문\n" + gates.DISCLAIMER, [20.0], escalation=False, escalation_message="", limit_notice=notice))


@pytest.mark.asyncio
async def test_all_clear_sentence_is_repaired_in_code_when_data_is_limited():
    """8b 가 '이상 없음'을 쓰면 리포트 전체를 버리지 않고 그 문장만 알려진 안전 문장으로 교체한다 (템플릿 대체 감소)."""
    draft = "요약\n데이터 커버리지가 낮습니다.\n\n감지된 이상\n이상 없음\n\n해석\n감지된 이상이 없습니다. 모니터링을 계속하세요."
    graph, _ = make(FakeTools([], coverage=8.0), FakeLLM(text={"writing": draft}))
    events, st = await run(graph, "U01 이상 점검해줘")
    vals = [e for e in events if e["type"] == "validation"]
    assert len(vals) == 1 and vals[0]["passed"] and not st["validation"]["degraded"]  # 재시도·템플릿 없이 첫 검증 통과
    assert "이상 없음" not in st["report"] and "감지된 이상이 없습니다" not in st["report"]
    assert st["report"].count("이번 기간 데이터에서는 임계값을 넘는 구간이 감지되지 않았습니다.") >= 2
    assert "모니터링을 계속하세요" not in st["report"]   # 명령형 조언은 코드의 몫 - LLM 문장에서는 지워진다 (NO_ANOMALY_GUIDANCE 가 대신한다)
    assert assemble(events) == st["report"]


@pytest.mark.asyncio
async def test_repair_is_not_applied_to_full_data_and_period_from_user_is_a_valid_fact():
    from app.agent.validator import repair_all_clear

    assert repair_all_clear("감지된 이상\n이상 없음", True).endswith("감지되지 않았습니다.")
    assert repair_all_clear("이상이 없다고 단정할 수 없습니다.", True) == "이상이 없다고 단정할 수 없습니다."  # 한계 서술은 건드리지 않는다
    # 사용자가 요청한 '지난 3일'이 replan 으로 6일이 되어도 "3일" 서술은 허용된다
    graph, tools = make(FakeTools([], coverage=[30.0, 30.0]), FakeLLM(text={"writing": "지난 3일 동안의 요약입니다. 이번 기간 데이터에서는 임계값을 넘는 구간이 감지되지 않았습니다."}))
    events, st = await run(graph, "U01 지난 3일 이상 점검해줘")
    assert any(e["type"] == "replan" for e in events) and st["validation"]["passed"] and not st["validation"]["degraded"]


# ---- 이상이 있으면 '이상 없음'을 교체하지 않고 반려한다 ------------------------------------
@pytest.mark.asyncio
async def test_n2_all_clear_is_not_repaired_when_anomalies_exist_even_with_low_coverage():
    ok = "휴식 중 25분간 평균 135.0 bpm 구간이 있었습니다. 데이터 커버리지가 낮아 해석에 주의가 필요합니다."
    llm = FakeLLM(text={"writing": ["감지된 이상\n이번 기간 데이터에서는 임계값을 넘는 구간이 감지되지 않았습니다. 이상 없음.", ok]})
    graph, _ = make(FakeTools([CRITICAL], coverage=30.0), llm)
    events, st = await run(graph, "U01 이상 점검해줘")
    vals = [e for e in events if e["type"] == "validation"]
    assert [v["passed"] for v in vals] == [False, True]  # 교체로 통과하지 못하고 재작성 후 통과
    first_val = next(i for i, e in enumerate(events) if e["type"] == "validation")
    pre = assemble(events[:first_val])
    assert "이상 없음" not in pre and "감지되지 않았습니다" not in pre  # 화면에도 나가지 않는다
    assert "135.0 bpm" in st["report"] and "감지되지 않았습니다" not in st["report"]  # 존재하는 critical 을 '없다'로 바꾸지 않는다


# ---- 에스컬레이션 중 진료 필요성 축소 금지 --------------------------------------------------
DOWNPLAY = ["병원에 갈 필요는 없습니다.", "119에 연락할 필요는 없습니다.", "응급 상황은 아닙니다.", "걱정할 일은 아닙니다.", "괜찮은 편입니다.",
            "문제가 없어 보입니다.", "양호합니다.", "조치할 것은 없습니다.", "병원에 갈 필요 없어요.", "크게 걱정하지 않으셔도 됩니다."]


@pytest.mark.parametrize("text", DOWNPLAY)
def test_n1_downplaying_care_is_rejected_only_while_escalation_is_active(text):
    esc = gates.ESCALATION_MESSAGE
    assert any("진료·119 필요성" in i or "단정적 안심" in i for i in validate_rules(f"{esc}\n{text}\n{gates.DISCLAIMER}", [], escalation=True, escalation_message=esc)), text
    assert not any("진료·119 필요성" in i for i in validate_rules(f"{text}\n{gates.DISCLAIMER}", [], escalation=False, escalation_message=""))


@pytest.mark.asyncio
async def test_n1_graph_never_streams_or_keeps_downplay_sentence_during_escalation():
    """에스컬레이션 중 진료 축소·경과 관찰 문장은 허용 목록 필터가 스트리밍 단계에서 건너뛴다 - 사실 문장은 그대로, 검증 실패(재시도)도 없다."""
    bad = "휴식 중 25분간 평균 135.0 bpm 구간이 있었습니다. 병원에 갈 필요는 없습니다. 지켜보세요."
    graph, _ = make(FakeTools([CRITICAL]), FakeLLM(text={"writing": bad}))
    events, st = await run(graph, "U01 이상 점검해줘")
    shown = assemble(events)
    assert "병원에 갈 필요" not in shown and "지켜보세요" not in shown and "135.0 bpm 구간" in shown
    assert [e["passed"] for e in events if e["type"] == "validation"] == [True]
    assert "병원에 갈 필요" not in st["report"] and gates.ESCALATION_MESSAGE in st["report"] and shown == st["report"]


def test_n1_all_clear_is_rejected_whenever_any_anomaly_exists():
    for text in ["이상이 없습니다.", "이상 없음"]:
        assert any("이상이 감지되었는데" in i for i in rules(text, any_anomalies=True))
        assert not any("이상이 감지되었는데" in i for i in rules(text, any_anomalies=False))

"""검증 규칙 단위 테스트 (E2E 재현은 tests/test_e2e_safety.py).

범위: 근거 없는 맥락별 안심 서술 검출(`unsupported_context_claims`), 데이터 정상 범위 단정 검출(`_all_clear_claim` 확장),
센서 결측 비율 기준의 데이터 한계 판정(`report.data_limits`). 일반 지식 문장이 오탐되지 않는 경계도 확인한다.
"""
import pytest

from app.agent import report as rpt
from app.agent.gates import DISCLAIMER
from app.agent.validator import _all_clear_claim, is_definitive_reassurance, repair_all_clear, unsupported_context_claims

STATS = {"sleep_hr": 55.0, "resting_hr": 61.0, "active_minutes": 300}


@pytest.mark.parametrize("text", [
    "운동 중 심박수는 정상 범위 내에 있었습니다.", "운동 중에는 괜찮습니다.", "운동할 때는 문제가 없습니다.", "활동 중 심박은 양호합니다.",
])
def test_exercise_or_activity_assurance_is_never_supported(text):
    assert unsupported_context_claims(text, STATS) == [text.rstrip(".")]


def test_sleep_and_rest_assurance_requires_the_matching_stat():
    sleep = "수면 중 심박은 정상 범위 내에 있었습니다."
    assert unsupported_context_claims(sleep, STATS) == []
    assert unsupported_context_claims(sleep, {**STATS, "sleep_hr": None}) != []
    assert unsupported_context_claims(sleep, None) != []
    rest = "휴식 중에는 문제가 없습니다."
    assert unsupported_context_claims(rest, STATS) == [] and unsupported_context_claims(rest, {"sleep_hr": 55.0}) != []


@pytest.mark.parametrize("text", [
    "운동 중에는 심박이 오르는 것이 자연스러운 반응입니다.", "운동 중 심박 상승은 정상입니다.", "운동하면 심박이 올라도 괜찮습니다.",
    "서비스는 운동 중 상승을 이상으로 보지 않습니다.", "수면 중에는 심박이 평소보다 낮아집니다.",
])
def test_general_knowledge_about_exercise_is_allowed(text):
    assert unsupported_context_claims(text, STATS) == []


@pytest.mark.parametrize("text", ["심박 기록은 정상 범위 내에 있습니다.", "심박 수치는 정상 범위에 해당합니다.", "데이터는 정상 수준입니다."])
def test_data_normal_claims_count_as_all_clear_and_are_repaired(text):
    assert _all_clear_claim(text)
    assert not _all_clear_claim(repair_all_clear(text, True))


@pytest.mark.parametrize("text", [
    "성인의 안정시 심박 정상 범위는 60~100회입니다.", "일반적으로 심박수는 60~100회가 정상 범위에 해당합니다.",
    "심박 기록이 정상 범위 내에 있다고 단정할 수 없습니다.",
])
def test_general_normal_range_knowledge_and_hedged_statements_are_not_all_clear(text):
    assert not _all_clear_claim(text)


def test_definitive_reassurance_helper_matches_validator_rule():
    assert is_definitive_reassurance("전반적으로 정상입니다.") and not is_definitive_reassurance("운동 중 심박 상승은 정상입니다.")


def _state(gap_minutes: list[int], window_days: float = 7, coverage: float = 88.0, rest: int = 500) -> dict:
    return {"summary": {"summary": {"coverage_pct": coverage, "rest_minutes": rest}}, "lookback_days": window_days,
            "anomalies": [{"type": "sensor_gap", "duration_min": m} for m in gap_minutes]}


def test_sensor_gap_counts_as_a_data_limit_only_when_it_covers_a_fifth_of_the_window():
    assert rpt.data_limits(_state([509, 509])) == []                         # 7일 구간의 10.1%: 흔한 착용 중 결측
    assert rpt.data_limits(_state([2100])) == ["센서 결측 1건"]               # 20.8%
    assert rpt.data_limits(_state([2100], window_days=30)) == []             # 같은 결측도 30일 구간에서는 4.9%
    assert rpt.data_limits(_state([], coverage=20.0, rest=30)) == ["데이터 커버리지 20%", "휴식 구간 30분"]    # 기존 기준은 그대로
    detection_window = {**_state([2100]), "detection": {"window": {"start": "2016-04-05T00:00:00", "end": "2016-04-12T00:00:00"}}}
    assert rpt.data_limits(detection_window) == ["센서 결측 1건"]            # 탐지 결과의 window 가 있으면 그 길이를 쓴다


# ---- 비교 리포트 사용자별 판정, 후속 답변 안심, 진료 불필요 서술 ----------------------------------

from app.agent.validator import blocked_sentence, contradicts_findings, dismisses_visit, validate_rules  # noqa: E402

CLEAN_U01 = dict(any_anomalies=True, threshold_anomalies=True, no_anomaly_users=frozenset({"U01"}), no_threshold_users=frozenset({"U01"}))


def test_all_clear_sentence_about_the_clean_user_is_allowed_in_a_comparison():
    ok = "## U01\n이번 기간 데이터에서는 임계값을 넘는 구간이 감지되지 않았습니다.\n\n## U05\n휴식 중 빈맥이 3건 감지되었습니다."
    assert not contradicts_findings(ok, **CLEAN_U01)
    assert not contradicts_findings("U01 은 이상이 없고 U05 는 휴식 중 빈맥이 감지되었습니다.", **CLEAN_U01)


def test_all_clear_about_the_anomalous_user_or_everyone_is_still_rejected():
    assert contradicts_findings("## U05\n이번 기간 데이터에서는 임계값을 넘는 구간이 감지되지 않았습니다.", **CLEAN_U01)
    assert contradicts_findings("U01 과 U05 모두 이상이 없습니다.", **CLEAN_U01)
    assert contradicts_findings("두 사용자 모두 이상이 없습니다.", **CLEAN_U01)   # 소유자를 알 수 없으면 전체 기준
    assert contradicts_findings("U01 은 이상이 없습니다.", any_anomalies=True, threshold_anomalies=True)   # 단일 사용자 요청은 예전 그대로


def test_streaming_guard_keeps_the_heading_scope_across_sentences():
    ctx = "## U01\n"
    s = "이번 기간 데이터에서는 임계값을 넘는 구간이 감지되지 않았습니다."
    assert not blocked_sentence(s, escalation=False, context=ctx, **CLEAN_U01)
    assert blocked_sentence(s, escalation=False, context="## U05\n", **CLEAN_U01)


def test_validate_rules_accepts_the_exact_comparison_report_from_the_bug_report():
    body = ("## U01\n이번 기간 데이터에서는 임계값을 넘는 구간이 감지되지 않았습니다.\n\n## U05\n휴식 중 빈맥이 감지되었습니다. 의료진과 상담하세요.\n\n"
            + DISCLAIMER)
    base = dict(escalation=False, escalation_message="", any_anomalies=True, threshold_anomalies=True)
    assert any("이상 없음" in i or "감지되지 않았다" in i for i in validate_rules(body, [], **base))
    assert not any("이상 없음" in i or "감지되지 않았다" in i for i in validate_rules(body, [], no_anomaly_users=frozenset({"U01"}), no_threshold_users=frozenset({"U01"}), **base))


@pytest.mark.parametrize("text", ["병원에 가지 않으셔도 됩니다.", "굳이 병원에 가실 필요는 없습니다.", "진료를 받지 않으셔도 됩니다.", "추가 검사는 필요 없습니다.", "병원 방문은 불필요합니다."])
def test_visit_dismissal_is_blocked_when_anomalies_exist_and_with_escalation(text):
    assert dismisses_visit(text)
    assert blocked_sentence(text, escalation=False, any_anomalies=True)
    assert blocked_sentence(text, escalation=True, any_anomalies=False)   # 에스컬레이션이면 이상 유무와 무관하게
    assert blocked_sentence(text, escalation=False, any_anomalies=False)   # 진료·응급 부정은 이상 유무와 무관하게 항상 막는다 (이상 없는 경로의 누수)
    assert any("진료" in i for i in validate_rules(text + "\n" + DISCLAIMER, [], escalation=False, escalation_message="", any_anomalies=True))

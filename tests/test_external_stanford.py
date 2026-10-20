"""Stanford 외부 검증 코드 테스트: 어댑터(CSV -> 분 단위 스키마), 사건 구성, 지표 계산, 재생 스모크 (작은 합성 픽스처만 사용 - 실제 데이터는 쓰지 않는다).

협력: `eval.external.stanford_adapter`, `eval.external.stanford_eval`. 데이터 가정: 합성 CSV 는 실제와 같은 열(user, datetime, heartrate/steps/stage...)을 가진다.
"""
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from eval.external import stanford_adapter as ad
from eval.external import stanford_eval as ev


def write_csv(path, rows, cols):
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


@pytest.fixture()
def raw(tmp_path):
    """ABC1234: 조각난 hr(두 파일, 중복 분 포함), 걸음, 수면 / XYZ9999: 수면 없음."""
    d = tmp_path / "raw"
    d.mkdir()
    write_csv(d / "ABC1234_1_hr.csv", [("ABC1234", "2024-01-01 10:00:05", 60), ("ABC1234", "2024-01-01 10:00:35", 64),
                                       ("ABC1234", "2024-01-01 10:01:10", 0), ("ABC1234", "2024-01-01 10:02:00", 70)], ["user", "datetime", "heartrate"])
    write_csv(d / "ABC1234_2_hr.csv", [("ABC1234", "2024-01-01 10:02:30", 72), ("ABC1234", "2024-01-01 10:05:00", 80)], ["user", "datetime", "heartrate"])
    write_csv(d / "ABC1234_steps.csv", [("ABC1234", "2024-01-01 10:00:00", 0), ("ABC1234", "2024-01-01 10:02:00", 15),
                                        ("ABC1234", "2024-01-01 10:03:00", 20), ("ABC1234", "2024-01-01 11:00:00", 30)], ["user", "datetime", "steps"])
    write_csv(d / "ABC1234_sleep.csv", [("ABC1234", "2024-01-01 10:03:20", 180, "light"), ("ABC1234", "2024-01-01 10:06:00", 120, "wake")],
              ["user", "datetime", "stage_duration", "stage"])
    write_csv(d / "XYZ9999_hr.csv", [("XYZ9999", "2024-01-02 00:00:00", 55)], ["user", "datetime", "heartrate"])
    return d


def test_parse_dates_handles_lists_nat_and_blanks():
    assert ad.parse_dates("[Timestamp('2025-01-16 00:00:00'), Timestamp('2025-01-10 00:00:00')]") == [pd.Timestamp("2025-01-10"), pd.Timestamp("2025-01-16")]
    assert ad.parse_dates("[NaT]") == [] and ad.parse_dates(float("nan")) == [] and ad.parse_dates(None) == []


def test_minutes_aggregate_floor_minute_drop_zero_join_steps_and_fragments(raw):
    m = ad.minutes_of(raw, "ABC1234").set_index("ts")
    t = pd.Timestamp
    assert m.loc[t("2024-01-01 10:00"), "hr_mean"] == 62 and m.loc[t("2024-01-01 10:00"), "hr_n"] == 2 and m.loc[t("2024-01-01 10:00"), "hr_min"] == 60
    assert t("2024-01-01 10:01") not in m.index                              # 0 bpm 은 측정 없음이라 그 분은 심박 행이 없다
    assert m.loc[t("2024-01-01 10:02"), "hr_n"] == 2 and m.loc[t("2024-01-01 10:02"), "steps"] == 15       # 조각 파일의 같은 분은 하나로 합쳐짐
    assert m.loc[t("2024-01-01 10:03"), "hr_n"] == 0 and m.loc[t("2024-01-01 10:03"), "steps"] == 20        # 심박 없는 걸음 분은 보존 (희소 저장)
    assert t("2024-01-01 11:00") not in m.index                              # 심박 기록 구간 밖의 걸음은 버림
    assert list(m.index) == sorted(m.index) and m["hr_n"].dtype == "int32"


def test_sleep_minutes_expand_non_wake_stages_and_missing_sleep_is_na(raw):
    m = ad.minutes_of(raw, "ABC1234").set_index("ts")
    assert str(m["asleep"].dtype) == "boolean"
    assert bool(m.loc[pd.Timestamp("2024-01-01 10:03"), "asleep"]) and bool(m.loc[pd.Timestamp("2024-01-01 10:05"), "asleep"])       # light 10:03:20 + 180s -> 10:03~10:06
    assert not bool(m.loc[pd.Timestamp("2024-01-01 10:00"), "asleep"])
    x = ad.minutes_of(raw, "XYZ9999")
    assert x["asleep"].isna().all() and x["steps"].eq(0).all()                # 수면 파일이 없으면 NA (알 수 없음)
    assert ad.minutes_of(raw, "NOPE000") is None


def test_events_group_illness_rows_of_the_same_participant():
    df = pd.DataFrame({"ParticipantID": ["AA0HAI1_Illness_1", "AA0HAI1_Illness_2", "BBB2222"], "Category": ["Other Illness", "Other Illness", "COVID-19"],
                       "Symptom_dates": ["[Timestamp('2024-01-05 00:00:00')]", "[NaT]", "[Timestamp('2024-02-01 00:00:00')]"],
                       "covid_diagnosis_dates": ["[NaT]", "[NaT]", "[Timestamp('2024-02-04 00:00:00')]"],
                       "recovery_dates": ["[NaT]", "[NaT]", "[Timestamp('2024-02-20 00:00:00'), Timestamp('2024-02-18 00:00:00')]"]})
    e = ad.events_of(df)
    assert [x["kind"] for x in e["AA0HAI1"]] == ["other", "other"] and e["AA0HAI1"][0]["symptom"] == ["2024-01-05"] and e["AA0HAI1"][1]["symptom"] == []
    assert e["BBB2222"][0]["kind"] == "covid" and e["BBB2222"][0]["recovery"] == ["2024-02-18", "2024-02-20"]


def test_build_writes_pseudonymous_parquet_and_participant_table(raw, tmp_path):
    labels = pd.DataFrame({"ParticipantID": ["ABC1234"], "Category": ["COVID-19"], "Symptom_dates": ["[Timestamp('2024-01-01 00:00:00')]"],
                           "covid_diagnosis_dates": ["[NaT]"], "recovery_dates": ["[NaT]"]})
    xlsx = tmp_path / "labels.xlsx"
    with pd.ExcelWriter(xlsx) as w:
        labels.to_excel(w, sheet_name="SuppTable3_Fig2a_COVID-19", startrow=3, index=False)
    out = tmp_path / "out"
    summary = ad.build(raw, xlsx, out)
    assert summary["n_participants"] == 2 and summary["n_labeled"] == 1 and summary["n_controls"] == 1
    df = pd.read_parquet(out / "vitals_minute.parquet")
    assert set(df["user_id"]) == {"S001", "S002"} and set(df["source"]) == {"stanford"} and "ABC1234" not in df.to_csv()
    parts = json.loads((out / "participants.json").read_text())
    assert parts["S001"]["group"] == "labeled" and parts["S001"]["has_sleep"] and parts["S002"]["group"] == "control" and not parts["S002"]["has_sleep"]
    assert json.loads((out / "id_map.json").read_text())["S001"] == "ABC1234"


# ---------- 지표 ----------
D = date


def test_cluster_onsets_merges_nearby_symptom_dates():
    assert ev.cluster_onsets([D(2024, 3, 1), D(2024, 1, 1), D(2024, 1, 5)]) == [D(2024, 1, 1), D(2024, 3, 1)]
    assert ev.cluster_onsets([D(2024, 1, 1), D(2024, 1, 20)]) == [D(2024, 1, 1), D(2024, 1, 20)] and ev.cluster_onsets([]) == []


def test_build_events_filters_to_record_and_makes_exclusion_span():
    part = {"start": "2024-01-01T00:00:00", "end": "2024-03-01T00:00:00", "events": [
        {"kind": "covid", "symptom": ["2024-02-01", "2024-02-03", "2023-06-01"], "diagnosis": ["2024-02-04"], "recovery": ["2024-02-20"]},
        {"kind": "other", "symptom": [], "diagnosis": ["2024-01-20"], "recovery": []}, {"kind": "other", "symptom": ["2025-01-01"], "diagnosis": [], "recovery": []}]}
    es = ev.build_events(part)
    assert len(es) == 2                                                       # 기록 밖 날짜만 있는 사건은 버림
    a, b = es
    assert a["onset"] == D(2024, 2, 1) and a["diagnosis"] == D(2024, 2, 4) and a["exclude"] == (D(2024, 1, 18), D(2024, 3, 5))
    assert b["onset"] is None and b["diagnosis"] == D(2024, 1, 20)


def test_window_result_hit_lead_and_evaluability():
    valid = {D(2024, 2, d) for d in range(1, 15)}
    alarms = {D(2024, 2, 6): {"elevated_resting_trend"}, D(2024, 2, 8): {"resting_tachycardia"}}
    r = ev.window_result(alarms, valid, D(2024, 2, 9))                        # 창 2/2~2/12, 첫 알람 2/6 -> 3일 먼저
    assert r == {"evaluable": True, "hit": True, "lead_days": 3}
    assert ev.window_result(alarms, valid, D(2024, 2, 9), types=("personal_deviation",))["hit"] is False
    late = ev.window_result({D(2024, 2, 11): {"personal_deviation"}}, valid, D(2024, 2, 9))
    assert late["hit"] and late["lead_days"] == -2                            # 증상 후(+2일) 알람은 음수 선행
    assert ev.window_result(alarms, valid, D(2024, 5, 1))["evaluable"] is False
    assert ev.window_result(alarms, valid, D(2024, 2, 20), pre=7, post=3)["hit"] is False


def test_summarize_events_counts_only_evaluable():
    recs = [{"evaluable": True, "hit": True, "lead_days": 4}, {"evaluable": True, "hit": True, "lead_days": -1},
            {"evaluable": True, "hit": False, "lead_days": None}, {"evaluable": False, "hit": False, "lead_days": None}]
    s = ev.summarize_events(recs)
    assert (s["n"], s["n_not_evaluable"], s["hits"], s["sensitivity"], s["lead_median_days"], s["detected_before_or_on_onset"]) == (3, 1, 2, 0.667, 1.5, 1)
    assert ev.summarize_events([])["sensitivity"] is None


def test_false_alarm_rate_excludes_event_spans_and_normalizes_to_30_days():
    valid = {D(2024, 3, 1) + pd.Timedelta(days=i).to_pytimedelta() for i in range(60)}
    alarms = {D(2024, 3, 5): {"personal_deviation"}, D(2024, 3, 6): {"resting_tachycardia"}, D(2024, 4, 10): {"personal_deviation"}}
    full = ev.false_alarm_rate(alarms, valid)
    assert full == {"days": 60, "alarm_days": 3, "per_30_person_days": 1.5}
    cut = ev.false_alarm_rate(alarms, valid, [(D(2024, 3, 1), D(2024, 3, 10))])
    assert cut["days"] == 50 and cut["alarm_days"] == 1
    assert ev.false_alarm_rate(alarms, valid, types=("resting_tachycardia",))["alarm_days"] == 1
    assert ev.false_alarm_rate({}, set())["per_30_person_days"] is None


def test_paper_reference_parses_detected_missed_and_symptomless_rows(tmp_path):
    rows = [["x"] * 6, [None] * 6, ["For Figure 4a-4b (COVID-19 positives)"] + [None] * 5, ["ParticipantID", "Category", "vs. Symptom day (RHR-Diff)", "d", None, None],
            ["A1", "s", -5, 1, None, None], ["A2", "s", 0, 1, None, None], ["A3", "s", "miss", 1, None, None], ["A4", "s", None, 1, None, None],
            [None] * 6, ["For Figure 4c (other illness)"] + [None] * 5, ["ParticipantID", "Category", "vs. Symptom day (RHR-Diff)", None, None, None],
            ["B1", "s", -2, None, None, None], ["B2", None, "No symptom day", None, None, None]]
    x = tmp_path / "p.xlsx"
    with pd.ExcelWriter(x) as w:
        pd.DataFrame(rows).to_excel(w, sheet_name="SuppTable13_Fig4abc", header=False, index=False)
    p = ev.paper_reference(x)
    assert p["covid"]["detected"] == 2 and p["covid"]["missed"] == 1 and p["covid"]["with_symptom_day"] == 3 and p["covid"]["median_delta_days"] == -2.5
    assert p["covid"]["detected_at_or_before_onset"] == 2 and p["other"]["detected"] == 1 and p["other"]["with_symptom_day"] == 1


def test_replay_respects_burn_in_and_flags_a_sustained_resting_hr_rise(tmp_path):
    """합성 14일: 앞 10일 휴식 심박 60, 마지막 4일 80 (걸음 없음). 번인 7일 이전에는 알람이 없고 상승 구간에서 알람이 난다."""
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-03-01", periods=14 * 1440, freq="min")
    day = (ts - ts[0]).days
    hr = np.where(day >= 10, 80.0, 60.0) + rng.normal(0, 1.2, len(ts))
    df = pd.DataFrame({"user_id": "S001", "ts": ts, "hr_mean": hr, "hr_min": hr - 2, "hr_max": hr + 2, "hr_n": np.int32(12),
                       "steps": np.int32(0), "asleep": pd.array([pd.NA] * len(ts), dtype="boolean"), "source": "stanford"})
    path = tmp_path / "m.parquet"
    df.to_parquet(path, index=False)
    uid, res = ev.replay_user(("S001", str(path)))
    assert uid == "S001" and res["valid_days"][0] == "2024-03-08"              # 번인 7일
    assert not [d for d in res["alarms"] if d < "2024-03-10"]
    assert [d for d in res["alarms"] if d >= "2024-03-11"], res


def test_chance_baseline_probability():
    assert ev.chance_hit_probability(0.0, 11) == 0.0 and ev.chance_hit_probability(1.0, 3) == 1.0
    assert abs(ev.chance_hit_probability(0.1, 2) - 0.19) < 1e-9


def test_evaluate_end_to_end_on_tiny_replays_reports_sensitivity_false_alarms_and_chance():
    days = [(date(2024, 2, 1) + pd.Timedelta(days=i).to_pytimedelta()) for i in range(60)]
    iso = [d.isoformat() for d in days]
    parts = {"S001": {"group": "labeled", "start": "2024-02-01T00:00:00", "end": "2024-03-31T00:00:00", "has_sleep": True,
                      "events": [{"kind": "covid", "symptom": ["2024-03-01"], "diagnosis": ["2024-03-04"], "recovery": ["2024-03-15"]}]},
             "S002": {"group": "control", "start": "2024-02-01T00:00:00", "end": "2024-03-31T00:00:00", "has_sleep": False, "events": []}}
    replays = {"S001": {"valid_days": iso, "alarms": {"2024-02-27": ["elevated_resting_trend"], "2024-02-10": ["resting_tachycardia"]}},
               "S002": {"valid_days": iso, "alarms": {"2024-02-10": ["resting_tachycardia"], "2024-02-12": ["personal_deviation"]}}}
    res = ev.evaluate(replays, parts)
    s = res["sensitivity"]["covid|symptom|all"]
    assert (s["n"], s["hits"], s["lead_median_days"]) == (1, 1, 3)               # 2/27 알람 -> 증상일(3/1) 3일 전
    assert res["sensitivity"]["covid|diagnosis|all"]["hits"] == 1 and "other|symptom|all" not in res["sensitivity"]
    fa = res["false_alarms"]
    assert fa["controls"] == {"days": 60, "alarm_days": 2, "per_30_person_days": 1.0}
    assert fa["labeled_outside_events"]["alarm_days"] == 1 and fa["labeled_outside_events"]["days"] < 60      # 2/27 알람은 사건 구간(2/16~3/29) 안이라 제외, 2/10 만 남음
    c = res["chance_baseline"]["covid"]
    assert c["events"] == 1 and c["observed_hits"] == 1 and 0 < c["expected_hits_by_chance"] < 1
    assert "민감도" in ev.render(res, {"covid": {"with_symptom_day": 1, "detected": 1, "missed": 0, "sensitivity_among_with_symptom_day": 1.0,
                                                "median_delta_days": -3, "detected_at_or_before_onset": 1},
                                       "other": {"with_symptom_day": 0, "detected": 0, "missed": 0, "sensitivity_among_with_symptom_day": None,
                                                 "median_delta_days": None, "detected_at_or_before_onset": 0}})


def test_first_alarm_matches_window_result_and_first_warnings_cover_every_displayed_onset():
    valid = {D(2024, 2, d) for d in range(1, 29)}
    alarms = {D(2024, 2, 6): {"elevated_resting_trend", "resting_tachycardia"}, D(2024, 2, 8): {"personal_deviation"}, D(2024, 2, 20): {"resting_tachycardia"}}
    anchor = D(2024, 2, 9)
    fa, wr = ev.first_alarm(alarms, valid, anchor), ev.window_result(alarms, valid, anchor)
    assert (fa["hit"], fa["lead_days"], fa["evaluable"]) == (wr["hit"], wr["lead_days"], wr["evaluable"]) == (True, 3, True)
    assert fa["first_alarm"] == "2024-02-06" and fa["types"] == ["elevated_resting_trend", "resting_tachycardia"]
    assert ev.first_alarm(alarms, valid, D(2024, 2, 16))["hit"] is False and ev.first_alarm(alarms, valid, D(2024, 5, 1))["evaluable"] is False


def test_first_warnings_are_window_independent_and_use_the_same_window_as_the_evaluation():
    """질병 묶음의 둘째 증상일에도 항목이 생기고(counted_in_eval=False), 평가에 쓰인 첫 증상일의 값은 evaluate() 와 같다."""
    iso = [(D(2024, 2, 1) + pd.Timedelta(days=i).to_pytimedelta()).isoformat() for i in range(60)]
    parts = {"S001": {"group": "labeled", "start": "2024-02-01T00:00:00", "end": "2024-03-31T00:00:00", "has_sleep": False,
                      "events": [{"kind": "covid", "symptom": ["2024-03-01", "2024-03-05"], "diagnosis": [], "recovery": []}]},
             "S002": {"group": "control", "start": "2024-02-01T00:00:00", "end": "2024-03-31T00:00:00", "has_sleep": False, "events": []}}
    replays = {"S001": {"valid_days": iso, "alarms": {"2024-02-27": ["elevated_resting_trend"]}}, "S002": {"valid_days": iso, "alarms": {}}}
    fw = ev.first_warnings(replays, parts)
    assert list(fw) == ["S001"]                                              # 대조군은 사건이 없다
    rows = {r["onset"]: r for r in fw["S001"]}
    assert rows["2024-03-01"]["counted_in_eval"] is True and rows["2024-03-05"]["counted_in_eval"] is False
    assert rows["2024-03-01"]["lead_days"] == 3 and rows["2024-03-01"]["first_alarm"] == "2024-02-27" and rows["2024-03-01"]["method"].startswith("causal")
    assert rows["2024-03-05"]["lead_days"] == 7                              # 같은 알람, 다른 증상일 기준
    res = ev.evaluate(replays, parts)
    assert res["sensitivity"]["covid|symptom|all"]["lead_median_days"] == rows["2024-03-01"]["lead_days"]        # UI 값 == 평가 값

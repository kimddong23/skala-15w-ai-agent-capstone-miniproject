"""규칙 기반 탐지기 테스트.

범위: `app.detection.detector` / `baseline` 의 규칙(빈맥·서맥·개인 이탈·추세·센서 결측, 운동 오탐 억제, 병합)과
주입 라벨 재현율. LLM 과 네트워크 없이 결정론적으로 돈다.

협력: `app.data.prepare` 산출물(labels.json)은 재현율 테스트에서만 쓴다 (없으면 `tests/_prepared.ensure_prepared` 가 생성).

데이터 가정: 합성 시계열은 분 단위, naive 로컬 시각, 심박 bpm. 합성 잡음은 시드 고정이라 판정이 흔들리지 않는다.
"""
import json

import numpy as np
import pandas as pd

from app.config import settings
from app.detection.baseline import REST_WINDOW_MIN, daily_resting_hr, dst_gap_minutes, fit_baseline, minute_grid
from app.detection.detector import detect, summarize
from tests._prepared import ensure_prepared

T0 = pd.Timestamp("2016-04-04 10:00")


def frame(hr, steps=None, asleep=None, start=T0) -> pd.DataFrame:
    """분 단위 합성 df(VitalsStore 형식)를 만든다. hr 가 NaN 인 분은 심박 결측(hr_n=0)."""
    """분 단위 합성 df (VitalsStore 형식). hr 가 NaN 인 분은 심박 결측."""
    hr = np.asarray(hr, float)
    n = len(hr)
    steps = np.zeros(n, int) if steps is None else np.asarray(steps, int)
    return pd.DataFrame({
        "ts": pd.date_range(start, periods=n, freq="min"), "hr_mean": hr, "hr_min": hr - 2, "hr_max": hr + 2,
        "hr_n": np.where(np.isnan(hr), 0, 12), "steps": steps,
        "asleep": np.zeros(n, bool) if asleep is None else np.asarray(asleep, bool)})


def rest_hr(n, level=65.0, seed=0) -> np.ndarray:
    """기준 level(bpm) 주변 정상 휴식 심박 n 분 (시드 고정 정규 잡음 sigma=1.5)."""
    return level + np.random.default_rng(seed).normal(0, 1.5, n)


def types(anomalies) -> set[str]:
    """탐지 결과의 유형 집합."""
    return {a.type for a in anomalies}


# ---------- 핵심 오탐 억제: 운동 ----------
def test_exercise_elevated_hr_not_flagged():
    """걸음이 있는 운동 중 150 bpm 대 심박과 회복은 어떤 이상도 만들지 않는다 (핵심 오탐 억제 규칙)."""
    hr = np.concatenate([rest_hr(240), np.full(40, 155.0), np.linspace(140, 70, 16), rest_hr(120, seed=1)])
    steps = np.concatenate([np.zeros(240), np.full(40, 120), np.zeros(16 + 120)])
    assert detect(frame(hr, steps)) == []


def test_post_exercise_recovery_within_washout_not_flagged():
    """걸음이 멈춘 직후 워시아웃 안의 높은 심박은 빈맥이 아니다."""
    # 걸음이 멈춘 직후에도 한동안 높은 심박 (회복) — 워시아웃 안에서는 이상이 아니다
    hr = np.concatenate([rest_hr(240), np.full(30, 150.0), np.full(12, 125.0), rest_hr(100, seed=2)])
    steps = np.concatenate([np.zeros(240), np.full(30, 100), np.zeros(112)])
    assert "resting_tachycardia" not in types(detect(frame(hr, steps)))


def test_same_hr_at_rest_is_flagged():
    """같은 심박이라도 휴식 중이면 빈맥 warning 으로 잡힌다 (운동 억제가 과도하지 않다는 대조군)."""
    hr = np.concatenate([rest_hr(240), np.full(30, 125.0), rest_hr(60, seed=3)])
    out = detect(frame(hr))
    assert [a.type for a in out] == ["resting_tachycardia"]
    assert out[0].severity == "warning" and out[0].context == "rest" and out[0].duration_min >= 28


# ---------- 심각도 ----------
def test_synthetic_critical_tachycardia():
    """휴식 중 148 bpm 20분은 critical 로 분류되고 근거 문장에 수치가 들어간다."""
    hr = np.concatenate([rest_hr(240), np.full(20, 148.0), rest_hr(60, seed=4)])
    a = [x for x in detect(frame(hr)) if x.type == "resting_tachycardia"]
    assert len(a) == 1 and a[0].severity == "critical" and a[0].value > 140
    assert "bpm" in a[0].evidence


def test_short_spike_below_min_duration_ignored():
    """최소 지속시간(10분) 미만의 순간 상승은 알리지 않는다."""
    hr = np.concatenate([rest_hr(240), np.full(6, 140.0), rest_hr(60, seed=5)])
    assert detect(frame(hr)) == []


def test_bradycardia_awake_vs_sleep_thresholds():
    """서맥 임계가 각성(40/위험 35)과 수면(35/위험 30)에서 다르게 적용된다."""
    base = rest_hr(240)
    awake = detect(frame(np.concatenate([base, np.full(20, 37.0), rest_hr(60, seed=1)])))
    assert [a.type for a in awake] == ["bradycardia"] and awake[0].severity == "warning"
    crit = detect(frame(np.concatenate([base, np.full(20, 32.0), rest_hr(60, seed=1)])))
    assert crit[0].severity == "critical"
    # 수면 중 38 은 정상 범위로 취급 (수면 기준은 35 미만), 33 은 이상
    n = 240 + 20 + 60
    asleep = np.zeros(n, bool)
    asleep[240:260] = True
    ok = detect(frame(np.concatenate([base, np.full(20, 38.0), rest_hr(60, seed=1)]), asleep=asleep))
    assert "bradycardia" not in types(ok)
    flagged = detect(frame(np.concatenate([base, np.full(20, 33.0), rest_hr(60, seed=1)]), asleep=asleep))
    assert [a for a in flagged if a.type == "bradycardia"][0].context == "sleep"


def test_personal_deviation_below_absolute_threshold():
    """절대 임계 미달이어도 평소(58 bpm) 대비 큰 상승은 개인 이탈로 잡힌다."""
    # 평소 휴식 심박 58 인 사람이 78 로 20분 — 절대 임계(100)에는 못 미치지만 개인 대비 이탈
    hr = np.concatenate([rest_hr(300, 58.0), np.full(25, 78.0) + np.random.default_rng(1).normal(0, 1, 25),
                         rest_hr(60, 58.0, seed=2)])
    out = detect(frame(hr))
    assert [a.type for a in out] == ["personal_deviation"]
    assert out[0].baseline is not None and abs(out[0].baseline - 58) < 3


# ---------- 센서 결측 ----------
def test_sensor_gap_requires_wear_evidence():
    """센서 결측은 걸음 기록(착용 증거)이 있을 때만 알리고, 걸음 없는 결측은 미착용으로 본다."""
    hr = np.concatenate([rest_hr(120), np.full(180, np.nan), rest_hr(120, seed=1)])
    walking = np.zeros(420, int)
    walking[150:170] = 40
    worn = detect(frame(hr, walking))
    assert [a.type for a in worn] == ["sensor_gap"] and worn[0].severity == "info" and worn[0].duration_min == 180
    assert detect(frame(hr)) == []     # 걸음도 없으면 미착용/충전으로 보고 알리지 않는다


# ---------- 병합 / 추세 / 요약 ----------
def test_overlapping_segments_merged_into_one_event():
    """1~2분 정상 구간을 사이에 둔 같은 유형의 상승은 하나의 이벤트로 병합된다."""
    hr = np.concatenate([rest_hr(240), np.full(15, 118.0), [90.0, 90.0], np.full(15, 118.0), rest_hr(60, seed=1)])
    out = [a for a in detect(frame(hr)) if a.type == "resting_tachycardia"]
    assert len(out) == 1 and out[0].duration_min >= 30


def _multiday(daily_levels) -> pd.DataFrame:
    """일별 수준(bpm) 목록으로 하루 1440분짜리 합성 데이터를 이어 붙인다."""
    parts = [frame(rest_hr(1440, lv, seed=i), start=pd.Timestamp("2016-04-01") + pd.Timedelta(days=i))
             for i, lv in enumerate(daily_levels)]
    return pd.concat(parts, ignore_index=True)


def test_elevated_resting_trend_needs_two_consecutive_days():
    """추세는 +10 bpm 이상인 날이 2일 연속일 때만, 하루뿐이면 알리지 않는다."""
    two = detect(_multiday([60] * 8 + [74, 75, 60]))
    trend = [a for a in two if a.type == "elevated_resting_trend"]
    assert len(trend) == 1 and trend[0].severity == "warning" and trend[0].start == pd.Timestamp("2016-04-09")
    assert not [a for a in detect(_multiday([60] * 8 + [74, 60, 60])) if a.type == "elevated_resting_trend"]


def test_trend_uses_history_when_window_is_short():
    """조회 구간이 짧아도 baseline_df 이력으로 일별 추세를 판정한다."""
    full = _multiday([60] * 8 + [74, 75])
    window = full[full["ts"] >= "2016-04-09"]
    out = detect(window, baseline_df=full)
    assert "elevated_resting_trend" in types(out)


def test_summarize_and_daily_resting_hr():
    """요약 키·커버리지 분모(조회 구간)·휴식 분 정의와 일별 안정시 심박의 신뢰 기준을 확인한다."""
    df = frame(rest_hr(600, 62.0))
    s = summarize(df)
    assert s["n_minutes"] == 600 and s["coverage_pct"] > 99 and 58 < s["resting_hr"] < 65
    assert s["rest_minutes"] == 600 - (REST_WINDOW_MIN - 1)      # 윈도 앞쪽 14분은 이전 걸음을 몰라 보수적으로 활동 처리
    assert set(s) >= {"coverage_pct", "hr_mean", "hr_p05", "hr_p95", "resting_hr", "sleep_hr", "active_minutes", "n_minutes"}
    wide = summarize(df, df["ts"].iloc[0], df["ts"].iloc[0] + pd.Timedelta(days=1))
    assert wide["coverage_pct"] < 50
    d = daily_resting_hr(df)
    assert d[0]["date"] == "2016-04-04" and 58 < d[0]["resting_hr"] < 65 and d[0]["n_rest_minutes"] == 600 - (REST_WINDOW_MIN - 1)
    assert daily_resting_hr(frame(rest_hr(60)))[0]["resting_hr"] is None      # 휴식 분이 적은 날은 신뢰하지 않음
    assert summarize(df.iloc[0:0])["n_minutes"] == 0 and detect(df.iloc[0:0]) == []


def test_anomaly_to_dict_is_json_serializable():
    """Anomaly.to_dict 는 시각을 ISO 문자열로 바꿔 JSON 직렬화가 가능하다 (MCP 응답 계약)."""
    hr = np.concatenate([rest_hr(240), np.full(20, 148.0), rest_hr(60, seed=4)])
    d = detect(frame(hr))[0].to_dict()
    json.dumps(d)
    assert d["start"].startswith("2016-04-04T") and d["type"] == "resting_tachycardia"


def test_baseline_needs_enough_minutes():
    """기준선은 분 수가 모자라면 None 이고 충분하면 중앙값에 수렴한다."""
    assert fit_baseline(frame(rest_hr(30))).rest_median is None
    assert abs(fit_baseline(frame(rest_hr(300, 70.0))).rest_median - 70) < 1


# ---------- 주입 라벨 재현율 (실제 데이터 위) ----------
def test_injected_events_event_level_recall_at_least_90pct():
    """주입 라벨 이벤트 단위 재현율이 0.9 이상이다 (실제 값은 -s 로 출력)."""
    ensure_prepared()
    labels = json.loads((settings.data_dir / "labels.json").read_text())
    df = pd.read_parquet(settings.data_dir / "vitals_minute.parquet")
    cols = ["ts", "hr_mean", "hr_min", "hr_max", "hr_n", "steps", "asleep"]
    found = {u: detect(df[df["user_id"] == u][cols].reset_index(drop=True)) for u in {lb["user_id"] for lb in labels}}
    hit = [any(a.type == lb["type"] and a.start <= pd.Timestamp(lb["end"]) and a.end >= pd.Timestamp(lb["start"])
               for a in found[lb["user_id"]]) for lb in labels]
    recall = sum(hit) / len(hit)
    print(f"event-level recall = {sum(hit)}/{len(hit)} = {recall:.3f}")
    assert recall >= 0.9


JARGON = ("MAD", "robust", "z≈", "CuSum", "p05", "p95", "coverage", "EllipticEnvelope", "σ", "sigma", "contamination")


def test_evidence_is_plain_language_and_details_live_in_stats():
    """사용자 화면에 그대로 나가는 evidence 에는 통계 용어가 없고, 세부값은 stats 에 있다 (모든 유형 포함 실데이터 탐지)."""
    ensure_prepared()
    df = pd.read_parquet(settings.data_dir / "vitals_minute.parquet")
    cols = ["ts", "hr_mean", "hr_min", "hr_max", "hr_n", "steps", "asleep"]
    seen = set()
    for u in ("U04", "U05", "U06", "U08", "U12", "U13", "U14", "U15"):
        for a in detect(df[df["user_id"] == u][cols].reset_index(drop=True)):
            seen.add(a.type)
            assert not [w for w in JARGON if w in a.evidence], a.evidence
            assert a.stats and json.dumps(a.to_dict()["stats"])
    assert {"resting_tachycardia", "personal_deviation", "sensor_gap"} <= seen


# ---------- 활동 맥락과 무관한 극단 빈맥 (운동 억제의 천장) ----------
def test_extreme_hr_with_shuffling_steps_is_critical_not_suppressed():
    """분당 걸음 2보 안팎(제자리 서성임)에서 190 bpm 이 20분 이어지면 '활동 중'이라는 이유로 묵살되지 않고 critical 이다."""
    n = 240 + 20 + 60
    hr = np.concatenate([rest_hr(240), np.full(20, 190.0), rest_hr(60, seed=1)])
    steps = np.zeros(n, int)
    steps[240:260] = 2
    out = [a for a in detect(frame(hr, steps)) if a.type == "resting_tachycardia"]
    assert len(out) == 1 and out[0].severity == "critical" and out[0].stats["method"] == "extreme"
    assert out[0].context == "active" and "활동으로 설명하기 어려울" in out[0].evidence


def test_genuine_running_cadence_at_190_is_not_flagged():
    """분당 120보 이상의 달리기에서 190 bpm 은 정상 운동 반응이라 알리지 않는다 (오탐 억제 유지)."""
    n = 240 + 25 + 60
    hr = np.concatenate([rest_hr(240), np.full(25, 190.0), rest_hr(60, seed=1)])
    steps = np.zeros(n, int)
    steps[240:265] = 130
    assert [a for a in detect(frame(hr, steps)) if a.type == "resting_tachycardia"] == []


def test_hr_above_210_flagged_even_with_running_cadence_and_over_230_not_dropped():
    """210 초과가 10분 지속되면 걸음 수와 무관하게 critical 이고, 230 초과 값도 조용히 버려지지 않는다."""
    n = 240 + 12 + 60
    hr = np.concatenate([rest_hr(240), np.full(12, 245.0), rest_hr(60, seed=1)])
    steps = np.zeros(n, int)
    steps[240:252] = 140
    df = frame(hr, steps)
    assert minute_grid(df)["hr"].max() == 245.0
    out = [a for a in detect(df) if a.type == "resting_tachycardia"]
    assert len(out) == 1 and out[0].severity == "critical"
    assert minute_grid(frame(np.full(30, 400.0)))["hr"].isna().all()      # 300 초과는 센서 오류로 버림
    short = np.concatenate([rest_hr(240), np.full(7, 245.0), rest_hr(60, seed=1)])
    assert detect(frame(short)) == [] or all(a.type != "resting_tachycardia" for a in detect(frame(short)))


# ---------- 병합 후 근거 문장 갱신 ----------
def test_merged_anomaly_evidence_matches_merged_duration():
    """휴식 중 일반 빈맥 규칙과 극단 빈맥 규칙이 같은 구간을 잡아 합쳐지면, 근거 문장의 분 수가 합친 구간의 duration_min 과 같다."""
    hr = np.concatenate([rest_hr(240), np.full(12, 125.0), np.full(20, 195.0), rest_hr(60, seed=1)])
    out = [a for a in detect(frame(hr)) if a.type == "resting_tachycardia"]
    assert len(out) == 1 and out[0].stats.get("merged") is True and out[0].severity == "critical"
    assert f"{out[0].duration_min}분" in out[0].evidence and out[0].duration_min >= 30


# ---------- 서머타임 ----------
def test_dst_spring_forward_hour_does_not_create_fake_sensor_gap():
    """2016-03-13 02:00~02:59 는 존재하지 않는 시각이다. 그 시간을 심박 결측으로 세어 60분짜리 가짜 결측을 만들면 안 된다."""
    day = pd.Timestamp("2016-03-13")
    pre = frame(rest_hr(90, 65.0), start=day)                                             # 00:00~01:29
    gap_pre = frame(np.full(30, np.nan), np.full(30, 5), start=day + pd.Timedelta(minutes=90))        # 01:30~01:59, 걸음만
    gap_post = frame(np.full(30, np.nan), np.full(30, 5), start=day + pd.Timedelta(hours=3))          # 03:00~03:29, 걸음만
    post = frame(rest_hr(90, 65.0, seed=1), start=day + pd.Timedelta(hours=3, minutes=30))
    df = pd.concat([pre, gap_pre, gap_post, post], ignore_index=True)
    assert [a for a in detect(df) if a.type == "sensor_gap"] == []        # 실제 결측은 60분뿐 (기준 120분 미만)
    assert not minute_grid(df).index.isin(pd.date_range(day + pd.Timedelta(hours=2), periods=60, freq="min")).any()
    other = pd.concat([pre, frame(rest_hr(30, 65.0), start=day + pd.Timedelta(hours=2))], ignore_index=True)
    assert minute_grid(other).index.isin(pd.date_range(day + pd.Timedelta(hours=2), periods=30, freq="min")).any()


def test_duration_across_dst_hour_counts_actual_minutes():
    """서머타임 전환(02:00~02:59 없음)을 가로지르는 01:50~03:10 빈맥은 달력상 81분이 아니라 실제 21분이다."""
    day = pd.Timestamp("2016-03-13")
    df = pd.concat([
        frame(rest_hr(110, 65.0), start=day),                                                 # 00:00~01:49
        frame(np.full(10, 120.0), start=day + pd.Timedelta(hours=1, minutes=50)),             # 01:50~01:59
        frame(np.full(11, 120.0), start=day + pd.Timedelta(hours=3)),                         # 03:00~03:10
        frame(rest_hr(60, 65.0, seed=1), start=day + pd.Timedelta(hours=3, minutes=11)),
    ], ignore_index=True)
    out = [a for a in detect(df) if a.type == "resting_tachycardia"]
    assert len(out) == 1 and out[0].start == day + pd.Timedelta(hours=1, minutes=50)
    assert out[0].duration_min == 21 and "21분" in out[0].evidence and "81분" not in out[0].evidence
    assert dst_gap_minutes(pd.Timestamp("2016-03-13 01:50"), pd.Timestamp("2016-03-13 03:10")) == 60
    assert dst_gap_minutes(pd.Timestamp("2016-03-14 00:00"), pd.Timestamp("2016-03-14 23:59")) == 0

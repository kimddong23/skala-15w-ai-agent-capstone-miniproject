"""연구 방법 탐지 컴포넌트 테스트 (시간 단위 EllipticEnvelope, CuSum).

범위: `app.detection.research` 의 특징 정의(Mishra 안정시 심박), CuSum 에피소드, 시간 단위 이상 시간 판정과
이것이 `detector.detect(config=...)` 에 통합되는 방식, 그리고 DEFAULT_CONFIG 와 벤치마크 선택 결과의 일치.
협력: `tests/test_detection.py` 의 합성 프레임 도우미, `eval/results/selected_config.json`(벤치마크 산출물).
데이터 가정: 합성 시계열은 분 단위 naive 로컬 시각, 잡음은 시드 고정.
"""
import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from app.detection import research
from app.detection.baseline import minute_grid
from app.detection.detector import DEFAULT_CONFIG, DetectorConfig, detect
from tests.test_detection import frame, rest_hr

RESULTS = __import__("pathlib").Path(__file__).resolve().parent.parent / "eval" / "results"


def day_series(levels, noise=1.0, seed=0) -> pd.Series:
    """일별 안정시 심박 합성 시계열 (날짜 index, 수준 + 정규 잡음)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-04-01", periods=len(levels), freq="D")
    return pd.Series(np.asarray(levels, float) + rng.normal(0, noise, len(levels)), index=idx)


# ---------- 특징: Mishra 정의 ----------
def test_rhr_uses_only_minutes_without_steps_in_preceding_12_minutes():
    """안정시 심박은 직전 12분 걸음이 0인 분만 쓰고 HROS 는 심박/(걸음+1) 이다 (Mishra 정의)."""
    n = 180
    steps = np.zeros(n, int)
    steps[60:70] = 50
    g = minute_grid(frame(np.full(n, 70.0), steps))
    f = research.hourly_features(g)
    h1 = f.loc[pd.Timestamp("2016-04-04 11:00")]       # 걸음이 있던 60~69분 구간 포함 시간
    assert h1["n"] == 39      # 걸음 60~69분 → 직전 12분에 걸음이 있는 60~80분은 휴식 제외, 81~119분만 휴식
    assert f["rhr"].dropna().eq(70.0).all()
    assert abs(f["hros"].iloc[0] - 70.0) < 1e-6          # 걸음 0 이면 HROS = 심박/(0+1)
    assert f.loc[pd.Timestamp("2016-04-04 11:00"), "hros"] < 70      # 걸음이 있으면 비율이 낮아진다


# ---------- CuSum ----------
def test_cusum_quiet_on_stationary_series():
    """정상 시계열에서는 CuSum 알림이 없다."""
    assert research.cusum_episodes(day_series([60] * 40, seed=1), k=0.5, h=4.0) == []


def test_cusum_detects_sustained_small_shift_that_rule_threshold_misses():
    """+6 bpm 처럼 규칙 임계(+10)에 못 미치는 지속 상승을 CuSum 이 시작일부터 잡는다."""
    s = day_series([60] * 12 + [66] * 4 + [60] * 6, noise=0.8, seed=2)      # +6 bpm: 규칙 임계(+10)에는 못 미친다
    eps = research.cusum_episodes(s, k=0.5, h=4.0)
    assert len(eps) == 1
    assert eps[0]["start"] == pd.Timestamp("2016-04-13") and eps[0]["end"] >= pd.Timestamp("2016-04-15")
    assert eps[0]["peak"] > 4.0 and abs(eps[0]["mu"] - 60) < 1.5


def test_cusum_resets_after_return_to_baseline():
    """기준으로 돌아온 뒤 다시 생긴 상승은 별도 에피소드다."""
    s = day_series([60] * 10 + [70] * 3 + [60] * 10 + [70] * 3 + [60] * 3, noise=0.5, seed=3)
    assert len(research.cusum_episodes(s, k=0.5, h=4.0)) == 2


def _multiday(levels, seed=0) -> pd.DataFrame:
    """일별 수준(bpm) 목록으로 하루 1440분짜리 합성 데이터를 이어 붙인다."""
    return pd.concat([frame(rest_hr(1440, lv, seed=seed + i), start=pd.Timestamp("2016-04-01") + pd.Timedelta(days=i))
                      for i, lv in enumerate(levels)], ignore_index=True)


def test_detector_cusum_trend_vs_rule():
    """같은 데이터에서 규칙은 놓치고 CuSum(과 both)은 추세로 잡는다."""
    df = _multiday([60] * 12 + [67] * 4 + [60] * 3)
    rule = [a for a in detect(df, config=DetectorConfig(trend="rule")) if a.type == "elevated_resting_trend"]
    cus = [a for a in detect(df, config=DetectorConfig(trend="cusum")) if a.type == "elevated_resting_trend"]
    both = [a for a in detect(df, config=DetectorConfig(trend="both")) if a.type == "elevated_resting_trend"]
    assert rule == [] and len(cus) == 1 and len(both) == 1
    assert cus[0].start == pd.Timestamp("2016-04-13") and cus[0].stats["method"] == "cusum" and "CuSum" not in cus[0].evidence


# ---------- 시간 단위 EllipticEnvelope ----------
def _hourly_frame():
    """10일 합성 데이터: 수면 7시간(56 bpm)/각성 66 bpm 패턴에 8일째 14~20시 +20 bpm 상승 구간을 넣는다."""
    n = 10 * 1440
    hr = np.where((np.arange(n) % 1440) < 420, 56.0, 66.0) + np.random.default_rng(5).normal(0, 1.2, n)
    hr[7 * 1440 + 14 * 60: 7 * 1440 + 20 * 60] += 20          # 8일째 14~20시 6시간 +20 bpm (100 미만)
    asleep = (np.arange(n) % 1440) < 420
    return frame(hr, asleep=asleep, start=pd.Timestamp("2016-04-01"))


def test_hourly_ee_flags_multi_hour_elevation_and_ignores_normal_data():
    """몇 시간 지속된 휴식 심박 상승은 잡고 정상 데이터에서는 알림이 없다."""
    df = _hourly_frame()
    cfg = DetectorConfig(personal_rule=False, hourly="rhr", contamination=0.02, min_hours=3, min_delta_bpm=8)
    out = [a for a in detect(df, config=cfg) if a.type == "personal_deviation"]
    assert len(out) == 1
    assert out[0].start == pd.Timestamp("2016-04-08 14:00") and out[0].end >= pd.Timestamp("2016-04-08 17:00")     # contamination 비율만큼만 잡혀 6시간 중 앞쪽 일부
    assert out[0].stats["method"] == "hourly_rhr" and "EllipticEnvelope" not in out[0].evidence and out[0].severity == "warning"
    quiet = frame(np.where((np.arange(len(df)) % 1440) < 420, 56.0, 66.0) + np.random.default_rng(6).normal(0, 1.2, len(df)),
                  asleep=(np.arange(len(df)) % 1440) < 420, start=pd.Timestamp("2016-04-01"))
    assert [a for a in detect(quiet, config=cfg) if a.type == "personal_deviation"] == []


def test_hourly_needs_enough_history():
    """이력이 72시간 미만이면 모델을 적합하지 않고 알림도 없다."""
    short = frame(rest_hr(24 * 60, 66.0), start=pd.Timestamp("2016-04-01"))
    cfg = DetectorConfig(personal_rule=False, hourly="rhr", contamination=0.1, min_hours=1)
    assert detect(short, config=cfg) == []         # 72시간 미만이면 모델을 적합하지 않는다


def test_hourly_window_only_reports_inside_window_with_history_baseline():
    """조회 구간이 짧아도 이력으로 적합하되 보고는 구간 안 이벤트만 한다."""
    full = _hourly_frame()
    window = full[full["ts"] >= "2016-04-08"].reset_index(drop=True)
    cfg = DetectorConfig(personal_rule=False, hourly="rhr", contamination=0.02, min_hours=3, min_delta_bpm=8)
    out = [a for a in detect(window, baseline_df=full, config=cfg) if a.type == "personal_deviation"]
    assert len(out) == 1 and out[0].start >= pd.Timestamp("2016-04-08")


# ---------- 구성 일관성 ----------
def test_default_config_matches_benchmark_selection():
    """DEFAULT_CONFIG 가 벤치마크가 튜닝 시드로 고른 구성과 같다 (손으로 어긋나지 않게)."""
    sel = json.loads((RESULTS / "selected_config.json").read_text())
    assert asdict(DEFAULT_CONFIG) == sel["config"]

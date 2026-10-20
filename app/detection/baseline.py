"""개인 기준선과 분 단위 맥락(수면/휴식/활동) 계산.

책임:
    - VitalsStore 형식의 희소 분 데이터를 1분 격자로 복원하고, 각 분의 맥락(ctx)을 부여한다.
    - 사용자별 휴식기 심박의 robust 기준선(median/MAD)과 일별 안정시 심박을 계산한다.
    탐지 규칙 자체(임계값 판정)는 `detector.py` 의 책임이며, 이 모듈은 판정에 쓰이는 입력을 만든다.

협력 모듈:
    - `detector.py`, `research.py`: 격자와 기준선을 소비한다.
    - `app/data/prepare.py`: 주입 후보 선정과 합성 이벤트 크기 산정에 같은 맥락 정의를 재사용한다
      (탐지기와 주입기가 서로 다른 '휴식' 정의를 쓰면 평가가 왜곡된다).

데이터 가정:
    - 입력 df 의 `ts` 는 기기 로컬 시간의 timezone-naive 분 단위 시각이다 (UTC 변환 없음).
    - 걸음 수가 없는 분은 0 으로 본다 (희소 저장 형식, `prepare.py` 참고). 심박이 없는 분은 NaN.
    - 단위: 심박은 bpm, 시간 길이는 분(minute) 이다.
    - 서머타임: 미국 봄 전환일 02:00~02:59(존재하지 않는 시각)는 격자에서 뺀다 — 격자 행은 '존재하는 분'이므로 그 날의 행 간격은 1시간 점프를 가질 수 있다.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

# 직전 N분(현재 분 포함) 동안 걸음이 0이면 '휴식'. 운동 직후 심박은 걸음이 멈춘 뒤에도 한동안 높게 유지되므로
# 워시아웃을 둔다 — 운동 중·직후 상승을 이상으로 오탐하지 않기 위한 핵심 규칙이다.
# 계약 초안은 10분이었으나 깨끗한 실데이터(15명, 469 사용자-일)에서 10분이면 빈맥 알림 27건 중 상당수가
# 운동 직후 회복 구간이었고 15분이면 14건으로 줄어 15분으로 늘렸다 (eval/results/detection_eval.md 튜닝 기록).
REST_WINDOW_MIN = 15
# 정규분포에서 MAD 를 표준편차로 환산하는 상수 (sigma ≈ 1.4826 * MAD).
MAD_TO_SIGMA = 1.4826
# MAD 가 극단적으로 작은 사용자에서 z 가 폭주하지 않도록 두는 sigma 하한 (bpm). 기기 측정 오차 수준.
MIN_SIGMA = 2.0
# 기준선을 신뢰하려면 필요한 최소 분 수 (약 2시간). 미달이면 해당 기준선은 None.
MIN_BASELINE_MINUTES = 120
# 일별 안정시 심박 = 그날 휴식 분 심박의 하위 10% 분위수. 일시적 상승을 배제한 '가장 안정된 구간'의 대표값이다.
RESTING_QUANTILE = 0.10
# 하루의 휴식 분이 이보다 적으면 그날 값은 신뢰하지 않는다 (추세 판정에서 제외).
MIN_REST_MINUTES_PER_DAY = 120
# 생리적으로 불가능한 값(센서 오류) 제외 범위 (bpm).
HR_MIN_VALID = 20
# 상한은 300 으로 둔다: 발작성 상심실성 빈맥(SVT)은 250 bpm 안팎까지 보고되므로 230 초과를 조용히 버리면
# 가장 위험한 값이 사라진다. 300 초과는 생리적으로 불가능해 센서 오류로 보고 버린다.
# (지속되는 210 초과는 detector 의 극단 빈맥 규칙이 critical 로 올린다.)
HR_MAX_VALID = 300


@dataclass(frozen=True)
class Baseline:
    """개인 기준선. 수면과 각성 휴식을 따로 둔다.

    수면 중 심박은 보통 각성 휴식보다 10~20% 낮아, 섞어서 계산하면 두 상태 모두 기준이 부정확해진다.
    분 수가 모자란 항목은 None 이며, 이 경우 해당 맥락의 개인 대비 판정은 건너뛴다.

    Attributes:
        rest_median: 각성 휴식 심박 중앙값 (bpm).
        rest_sigma: 각성 휴식 심박 robust 표준편차 (bpm, MIN_SIGMA 이상).
        sleep_median: 수면 중 심박 중앙값 (bpm).
        sleep_sigma: 수면 중 심박 robust 표준편차 (bpm, MIN_SIGMA 이상).
        n_rest: 기준선에 쓰인 각성 휴식 분 수.
        n_sleep: 기준선에 쓰인 수면 분 수.
    """
    rest_median: float | None
    rest_sigma: float | None
    sleep_median: float | None
    sleep_sigma: float | None
    n_rest: int
    n_sleep: int


def context(steps: pd.Series, asleep: pd.Series) -> pd.Series:
    """분 격자 위에서 각 분의 맥락을 계산한다.

    수면 플래그가 우선하고, 그 외에는 직전 REST_WINDOW_MIN 분 걸음 합이 0이면 휴식, 아니면 활동이다.

    Args:
        steps: 분 단위 걸음 수 (index 는 1분 간격 DatetimeIndex, 결측 없음).
        asleep: 같은 index 의 수면 여부.

    Returns:
        같은 index 의 문자열 Series: 'sleep' | 'rest' | 'active'.
        격자 앞쪽 REST_WINDOW_MIN-1 분은 이전 걸음을 알 수 없어 보수적으로 'active'(수면이면 'sleep')로 둔다 —
        조회 구간 경계에서 운동 직후 심박이 휴식으로 오인되는 것을 막기 위함이다.
    """
    steps_sum = steps.rolling(REST_WINDOW_MIN, min_periods=REST_WINDOW_MIN).sum().fillna(1)
    ctx = np.where(asleep.to_numpy(bool), "sleep", np.where(steps_sum.to_numpy() == 0, "rest", "active"))
    return pd.Series(ctx, index=steps.index)


def _spring_forward_hour(year: int) -> pd.Timestamp:
    """그 해 미국 서머타임 시작일(3월 둘째 일요일)의 02:00 — 로컬 시계에서 02:00~02:59 가 존재하지 않는다."""
    first = pd.Timestamp(year=year, month=3, day=1)
    return first + pd.Timedelta(days=(6 - first.weekday()) % 7 + 7, hours=2)


def dst_gap_minutes(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """[start, end(마지막 분 포함)] 구간이 포함하는 '존재하지 않는 서머타임 시간대' 분 수 (0 또는 최대 60).

    달력상 분 수에서 이 값을 빼면 실제로 존재한 분 수가 된다. 격자 행으로 직접 세지 못하는 구간(일 단위 추세, 시간 단위 이상,
    병합된 구간)의 duration_min 에 쓴다. 규칙은 미국 기준이다 (`_dst_missing_hour` 와 같은 가정).
    """
    total = 0
    for year in range(start.year, end.year + 1):
        gap_start = _spring_forward_hour(year)
        lo, hi = max(start, gap_start), min(end + pd.Timedelta(minutes=1), gap_start + pd.Timedelta(hours=1))
        total += max(0, int((hi - lo) / pd.Timedelta(minutes=1)))
    return total


def _dst_missing_hour(d: pd.DataFrame) -> pd.DatetimeIndex:
    """격자에서 '로컬 시계에 존재하지 않는' 시각(봄 서머타임 전환일 02:00~02:59)이면서 원본 행이 전혀 없는 분.

    기기 시각은 naive 로컬 시간이라 미국 서머타임 전환일(3월 둘째 일요일)에 01:59 다음이 03:00 이다. 1분 격자를 그냥 채우면
    존재하지 않는 60분이 '심박 결측'으로 생겨 센서 결측 길이가 60분 부풀거나 한 구간이 둘로 쪼개진다.
    그 시간대에 실제 행이 하나라도 있으면(다른 지역 사용자) 건드리지 않는다. 가을 전환(같은 시각이 두 번)은
    중복 ts 제거로 이미 처리된다. 규칙은 미국 기준이며 데이터셋(Kaggle FitBit 2016)이 미국 사용자의 로컬 시각이라는 가정에 따른다.

    Args:
        d: 1분 격자로 reindex 된 원본 컬럼 프레임 (원본 행이 없는 분은 전부 NaN).
    """
    drop = []
    for year in range(d.index[0].year, d.index[-1].year + 1):
        gap_start = _spring_forward_hour(year)
        hour = d.loc[gap_start: gap_start + pd.Timedelta(minutes=59)]
        if len(hour) and hour.isna().all().all():
            drop.extend(hour.index)
    return pd.DatetimeIndex(drop)


def minute_grid(df: pd.DataFrame) -> pd.DataFrame:
    """VitalsStore 형식 df 를 1분 격자로 복원한다.

    Args:
        df: 컬럼 ts(naive 로컬 시각), hr_mean, steps, asleep 을 가진 희소 분 데이터. 중복 ts 는 첫 행만 쓴다.

    Returns:
        index=ts(1분 간격, 첫 행~마지막 행 포함) 이고 컬럼이 hr(유효 범위 밖은 NaN), hr_s(3분 중앙값 평활,
        단발 잡음 완화용), steps(결측 0), asleep, ctx 인 DataFrame. df 가 비어 있으면 같은 컬럼의 빈 프레임.
    """
    if df.empty:
        return pd.DataFrame(columns=["hr", "hr_s", "steps", "asleep", "ctx"])
    d = df.drop_duplicates("ts").set_index("ts").sort_index()
    idx = pd.date_range(d.index[0].floor("min"), d.index[-1].floor("min"), freq="min")
    d = d.reindex(idx)
    d = d.drop(index=_dst_missing_hour(d))
    hr = d["hr_mean"].where(d["hr_mean"].between(HR_MIN_VALID, HR_MAX_VALID))
    g = pd.DataFrame({
        "hr": hr,
        "hr_s": hr.rolling(3, center=True, min_periods=1).median(),
        "steps": d["steps"].fillna(0),
        "asleep": d["asleep"].fillna(False).astype(bool),
    })
    g["ctx"] = context(g["steps"], g["asleep"])
    return g


def _robust(x: pd.Series) -> tuple[float | None, float | None]:
    """결측을 뺀 x 의 (중앙값, robust sigma). 분 수가 MIN_BASELINE_MINUTES 미만이면 (None, None)."""
    x = x.dropna()
    if len(x) < MIN_BASELINE_MINUTES:
        return None, None
    med = float(x.median())
    sigma = max(MAD_TO_SIGMA * float((x - med).abs().median()), MIN_SIGMA)
    return med, sigma


def fit_baseline(df: pd.DataFrame) -> Baseline:
    """사용자 이력 df 로 각성 휴식/수면 기준선을 적합한다.

    Args:
        df: VitalsStore 형식의 이력. 길수록 안정적이며, 평가 중인 구간을 포함해도 median/MAD 가 robust 해서
            소수의 이상 분이 기준을 크게 끌어올리지 않는다.

    Returns:
        Baseline. df 가 비면 모든 값이 None/0.
    """
    g = minute_grid(df)
    if g.empty:
        return Baseline(None, None, None, None, 0, 0)
    rest, sleep = g.loc[g["ctx"] == "rest", "hr"], g.loc[g["ctx"] == "sleep", "hr"]
    rm, rs = _robust(rest)
    sm, ss = _robust(sleep)
    return Baseline(rm, rs, sm, ss, int(rest.notna().sum()), int(sleep.notna().sum()))


def resting_hr_of(g: pd.DataFrame) -> float | None:
    """격자 g 의 안정시 심박(bpm).

    Args:
        g: `minute_grid` 결과.

    Returns:
        활동이 아닌 분(휴식+수면)의 심박 하위 RESTING_QUANTILE 분위수. 해당 분이 30개 미만이면 None.
    """
    x = g.loc[g["ctx"] != "active", "hr"].dropna()
    return float(x.quantile(RESTING_QUANTILE)) if len(x) >= 30 else None


def daily_resting_hr(df: pd.DataFrame) -> list[dict]:
    """일별 안정시 심박을 계산한다.

    Args:
        df: VitalsStore 형식 데이터. 날짜는 기기 로컬 달력 기준(자정~자정)이다.

    Returns:
        [{"date": "YYYY-MM-DD", "resting_hr": bpm(소수 1자리) | None, "n_rest_minutes": int}].
        그날 휴식 분(심박 있는 것만)이 MIN_REST_MINUTES_PER_DAY 미만이면 resting_hr 는 None 이다
        (n_rest_minutes 는 그대로 보고). 조회 구간의 첫·마지막 날처럼 일부만 포함된 날도 같은 기준으로 걸러진다.
    """
    g = minute_grid(df)
    if g.empty:
        return []
    out = []
    for day, part in g.groupby(g.index.normalize()):
        rest = part.loc[part["ctx"] != "active", "hr"].dropna()
        ok = len(rest) >= MIN_REST_MINUTES_PER_DAY
        out.append({
            "date": day.strftime("%Y-%m-%d"),
            "resting_hr": round(float(rest.quantile(RESTING_QUANTILE)), 1) if ok else None,
            "n_rest_minutes": int(len(rest)),
        })
    return out

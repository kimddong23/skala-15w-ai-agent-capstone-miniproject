"""Kaggle FitBit 원본 CSV 를 분 단위 parquet 과 평가용 라벨로 변환한다 (데이터 계층의 쓰기 측).

책임 경계:
    - 이 모듈만 raw CSV 를 읽고 합성 이상을 주입한다. 런타임(MCP 서버/탐지기)은 산출물만 읽는다 (`store.py`).
    - 탐지 규칙은 구현하지 않는다. 다만 '휴식' 맥락 정의(`baseline.context`)와 기준선(`fit_baseline`)을 재사용해
      주입 후보와 크기를 정한다 — 주입기와 탐지기가 같은 정의를 써야 재현율이 공정하게 측정된다.

협력 모듈:
    `app.detection.baseline`(맥락 정의), `app.data.store`(산출물 소비), `eval/*`(라벨 소비), `app.config`(경로).

산출물 (settings.data_dir):
    vitals_minute_clean.parquet  원본 그대로 (주입 없음)
    vitals_minute.parquet        실제 사용자 데이터 위에 합성 이상을 덮어쓴 것 (SEED 고정, 재현 가능)
    labels.json                  주입 이벤트 목록 (--no-inject 이면 빈 목록)
    id_map.json                  가명 U01.. -> 원본 Id (가명화 시연)

데이터 가정:
    - 시각은 기기 로컬 시간의 timezone-naive 값이며 분 단위로 내림(floor)한다. 두 내보내기 기간(3.12~4.11, 4.12~5.12)은
      합치고 (Id, 시각) 중복을 제거한다.
    - 희소 저장: 심박이 있는 분 + 걸음(>0)이 있는 분만 저장한다. 걸음 없는 분은 0 으로 복원할 수 있어 손실이 없고,
      심박은 없는데 걷고 있던 분(센서 결측과 미착용을 구분하는 근거)이 보존된다. 그런 분은 hr_mean=NaN, hr_n=0.
    - asleep 은 minuteSleep value==1(asleep) 만 True 이다 (2=restless, 3=awake 는 False).
    - 단위: 심박 bpm, 길이 분(minute). 라벨의 end 는 마지막 분을 포함한다.

실행: `.venv/bin/python -m app.data.prepare [--no-inject]`
"""
import argparse
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.config import settings
from app.detection.baseline import HR_MAX_VALID, HR_MIN_VALID, REST_WINDOW_MIN, context, daily_resting_hr, fit_baseline

SEED = 20261016               # 재현성용 고정 시드 (벤치마크의 튜닝/홀드아웃 시드와 무관)
MIN_INJECT_DAYS = 10          # 주입 대상 사용자의 최소 심박 일수 (기준선·추세에 필요한 이력)
MIN_PRIOR_VALID_DAYS = 4      # 추세 이벤트 직전 7일 중 유효한 안정시 심박 일수 하한
BLOCK_MARGIN_MIN = 90         # 이벤트 전후로 다른 이벤트를 피하는 간격(분)
MAX_EVENTS_PER_USER = 3         # 한 사용자에 이벤트가 몰리면 그 사용자의 기준선이 오염된다
TREND_MARGIN_BPM = 12         # 추세 주입 후보 조건: (그날 안정시 심박 + 상승폭 - 직전 중앙값) 최소치 = 탐지 임계 10 + 여유 2.
                              # 실제 일별 변동이 커서 원리상 탐지 불가능한 날은 '탐지기 결함'이 아니라 후보에서 제외한다.
DAY_HOURS = (8, 22)           # 각성 휴식·센서 결측 주입은 낮 시간대에만 (미기록 수면과의 혼동 방지)
HR_COLS = ["hr_mean", "hr_min", "hr_max"]


# ---------------------------------------------------------------- 로드 / 집계
def _read(name: str, **kw) -> pd.DataFrame:
    """raw_dir 아래 모든 내보내기 폴더의 같은 이름 CSV 를 이어 붙여 읽는다.

    Args:
        name: CSV 파일명 (예: heartrate_seconds_merged.csv).
        **kw: pandas.read_csv 인자.

    Raises:
        FileNotFoundError: 해당 파일이 하나도 없을 때.
    """
    files = sorted(settings.raw_dir.glob(f"*/*/{name}"))
    if not files:
        raise FileNotFoundError(f"{settings.raw_dir} 아래에서 {name} 을 찾을 수 없습니다")
    return pd.concat([pd.read_csv(f, **kw) for f in files], ignore_index=True)


def _ts(s: pd.Series) -> pd.Series:
    """FitBit 시각 문자열("4/12/2016 7:21:00 AM")을 naive datetime64[ns] 로 변환한다 (타임존 변환 없음)."""
    return pd.to_datetime(s, format="%m/%d/%Y %I:%M:%S %p").astype("datetime64[ns]")


def load_minute() -> pd.DataFrame:
    """raw CSV 를 읽어 희소 분 단위 DataFrame 을 만든다.

    초 단위 심박은 분 단위 평균/최소/최대/표본 수로 집계한다. 사용자의 심박 기록 구간 밖 걸음은 버린다.

    Returns:
        컬럼 Id(원본 사용자 ID), ts, hr_mean, hr_min, hr_max, hr_n, steps, asleep. (Id, ts) 정렬.
        심박 없는 분(걸음만 있는 분)은 hr_mean/min/max=NaN, hr_n=0.

    Raises:
        FileNotFoundError: 필요한 raw CSV 가 없을 때.
    """
    hr = _read("heartrate_seconds_merged.csv").drop_duplicates(["Id", "Time"])
    hr["ts"] = _ts(hr["Time"]).dt.floor("min")
    agg = hr.groupby(["Id", "ts"])["Value"].agg(hr_mean="mean", hr_min="min", hr_max="max", hr_n="size").reset_index()

    st = _read("minuteStepsNarrow_merged.csv")
    st["ts"] = _ts(st["ActivityMinute"])
    st = st.drop_duplicates(["Id", "ts"]).rename(columns={"Steps": "steps"})
    st = st.loc[(st["steps"] > 0) & st["Id"].isin(agg["Id"]), ["Id", "ts", "steps"]]

    sl = _read("minuteSleep_merged.csv")
    sl["ts"] = _ts(sl["date"]).dt.floor("min")
    sl = sl.loc[sl["value"] == 1, ["Id", "ts"]].drop_duplicates().assign(asleep=True)

    df = agg.merge(st, on=["Id", "ts"], how="outer").merge(sl, on=["Id", "ts"], how="left")
    span = agg.groupby("Id")["ts"].agg(lo="min", hi="max")      # 사용자의 심박 기록 구간 밖의 걸음은 버린다
    df = df.join(span, on="Id")
    df = df[(df["ts"] >= df["lo"]) & (df["ts"] <= df["hi"])].drop(columns=["lo", "hi"])
    df["steps"] = df["steps"].fillna(0).astype("int32")
    df["hr_n"] = df["hr_n"].fillna(0).astype("int32")
    df["asleep"] = df["asleep"].fillna(False).astype(bool)
    return df.sort_values(["Id", "ts"]).reset_index(drop=True)


# ---------------------------------------------------------------- 격자 ↔ 희소 변환
def _to_grid(u: pd.DataFrame) -> pd.DataFrame:
    """한 사용자의 희소 행을 1분 격자(index=ts)로 복원하고 빈 injected 컬럼을 붙인다 (주입 작업용 작업 복사본)."""
    idx = pd.date_range(u["ts"].min(), u["ts"].max(), freq="min", name="ts")
    g = u.drop(columns=["Id"]).set_index("ts").reindex(idx)
    g["steps"] = g["steps"].fillna(0).astype("int32")
    g["hr_n"] = g["hr_n"].fillna(0).astype("int32")
    g["asleep"] = g["asleep"].fillna(False).astype(bool)
    g["injected"] = None
    return g


def _compact(g: pd.DataFrame) -> pd.DataFrame:
    """격자를 희소 형식으로 되돌린다 (심박 또는 걸음이 있는 분만, ts 를 컬럼으로)."""
    g = g[(g["hr_n"] > 0) | (g["steps"] > 0)]
    return g.reset_index()


def _for_detector(g: pd.DataFrame) -> pd.DataFrame:
    """탐지기에 넘길 형식 — 희소 형식에서 정답 라벨(injected)을 제거한다 (라벨 누수 방지)."""
    return _compact(g).drop(columns=["injected"])


# ---------------------------------------------------------------- 주입
def _csum(a: np.ndarray) -> np.ndarray:
    """맨 앞에 0 을 붙인 누적합. 윈도 합을 cs[e] - cs[s] 로 O(1) 에 구하기 위함."""
    return np.concatenate([[0], np.cumsum(a)])


def _find_starts(g: pd.DataFrame, L: int, kind: str, blocked: np.ndarray) -> np.ndarray:
    """길이 L(분) 윈도의 시작 인덱스 후보를 벡터화로 찾는다.

    Args:
        g: `_to_grid` 격자.
        L: 윈도 길이(분).
        kind: 'awake'(낮·각성·직전 REST_WINDOW_MIN 분 포함 걸음 0) | 'sleep'(전 구간 수면 기록) |
            'gap'(낮·걸음 15분 이상=착용 증거, 앞뒤 15분 안에 심박 존재).
        blocked: 이미 다른 이벤트가 차지한 분 (True). 앞뒤 BLOCK_MARGIN_MIN 분도 피한다.

    Returns:
        조건을 만족하는 시작 인덱스 배열 (없으면 빈 배열). 심박 커버리지는 gap 70%, 그 외 90% 이상.
    """
    n, pre = len(g), 15
    s = np.arange(pre + REST_WINDOW_MIN, n - L - pre)
    if len(s) == 0:
        return s
    win = lambda cs, a=0, b=0: cs[s + L + b] - cs[s + a]   # noqa: E731
    hr_ok, steps_pos = _csum(g["hr_n"].to_numpy() > 0), _csum(g["steps"].to_numpy() > 0)
    asleep, blk = _csum(g["asleep"].to_numpy()), _csum(blocked)
    ok = (blk[np.minimum(s + L + BLOCK_MARGIN_MIN, n)] - blk[np.maximum(s - BLOCK_MARGIN_MIN, 0)]) == 0
    cov = 0.7 if kind == "gap" else 0.9
    ok &= win(hr_ok) >= cov * L
    hours = g.index.hour.to_numpy()
    in_day = (hours[s] >= DAY_HOURS[0]) & (hours[s + L - 1] < DAY_HOURS[1]) & (hours[s] <= hours[s + L - 1])
    if kind == "sleep":
        ok &= win(asleep) == L
    else:
        ok &= (win(asleep) == 0) & in_day
        if kind == "awake":
            ok &= win(steps_pos, -REST_WINDOW_MIN) == 0
        else:
            ok &= win(steps_pos) >= 15
            before = hr_ok[s] - hr_ok[s - 15]
            after = hr_ok[s + L + 15] - hr_ok[s + L]
            ok &= (before > 0) & (after > 0)
    return s[ok]


def _noise(rng, L: int, sigma: float, phi: float = 0.6) -> np.ndarray:
    """AR(1) 잡음 (길이 L, 표준편차 약 sigma/sqrt(1-phi^2)). 분 단위 심박은 이웃 분과 상관되어 있어 백색잡음보다 현실적이다."""
    e = rng.normal(0, sigma, L)
    for i in range(1, L):
        e[i] += phi * e[i - 1]
    return e


def _overwrite(g, rng, s: int, L: int, level: float, ramp: int, sigma: float, event_id: str, asleep: bool | None,
               zero_steps: bool = True):
    """격자 g 의 [s, s+L) 심박을 원래 값에서 level 로 올렸다(ramp 분) 유지하고 다시 원래 값으로 되돌린다 (제자리 수정).

    사다리꼴 가중으로 원래 값과 level 을 섞고 AR(1) 잡음과 느린 진동(12~25분 주기)을 더한다.

    Args:
        g: `_to_grid` 격자 (수정됨).
        rng: numpy Generator.
        s: 시작 인덱스. L: 길이(분). level: 목표 심박(bpm). ramp: 램프 길이(분). sigma: 잡음 크기(bpm).
        event_id: injected 컬럼에 기록할 이벤트 ID.
        asleep: True/False 면 구간의 수면 플래그를 강제, None 이면 유지.
        zero_steps: True 면 구간 걸음을 0 으로 만든다 (휴식 맥락 보장). 운동 중 시나리오는 False.
    """
    w = np.ones(L)
    r = np.linspace(0, 1, ramp + 2)[1:-1]
    w[:ramp], w[-ramp:] = r, r[::-1]
    sl = slice(s, s + L)
    orig = g["hr_mean"].iloc[sl].interpolate(limit_direction="both").fillna(level).to_numpy()
    wave = rng.uniform(0.8, 2.0) * np.sin(2 * np.pi * np.arange(L) / rng.uniform(12, 25) + rng.uniform(0, 6.28))
    new = orig * (1 - w) + (level + wave + _noise(rng, L, sigma)) * w
    new = np.clip(new, HR_MIN_VALID + 5, HR_MAX_VALID)
    g.iloc[sl, g.columns.get_loc("hr_mean")] = new
    g.iloc[sl, g.columns.get_loc("hr_min")] = new - rng.uniform(1, 4, L)
    g.iloc[sl, g.columns.get_loc("hr_max")] = new + rng.uniform(1, 5, L)
    n_old = g["hr_n"].iloc[sl].to_numpy()
    g.iloc[sl, g.columns.get_loc("hr_n")] = np.where(n_old > 0, n_old, 12)
    if zero_steps:
        g.iloc[sl, g.columns.get_loc("steps")] = 0
    if asleep is not None:
        g.iloc[sl, g.columns.get_loc("asleep")] = asleep
    g.iloc[sl, g.columns.get_loc("injected")] = event_id


@dataclass
class Spec:
    """주입 이벤트 명세. dur/level 은 (최소, 최대) 구간에서 균등 추출한다.

    Attributes:
        type: 탐지 유형과 동일한 이벤트 유형.
        kind: awake | sleep | gap | trend.
        severity: 라벨에 기록할 기대 심각도.
        dur: 길이 범위 (point 이벤트는 분, trend 는 일).
        level: 목표 심박 범위 bpm (trend 는 상승폭 bpm, personal_deviation/sensor_gap 은 미사용).
    """
    type: str
    kind: str          # awake | sleep | gap | trend
    severity: str
    dur: tuple         # 길이 범위(분) / 추세는 일수 범위
    level: tuple       # 목표 심박 범위 (personal_deviation은 사용하지 않음), 추세는 상승폭 범위


# 유형별 2~3건, 상황(각성/수면, 경고/위험)을 섞는다. 길이는 '탐지 규칙의 최소 지속시간 + 램프 양쪽(3~6분 x 2)'을 넘게 잡아
# 정의상 이벤트가 되도록 했다. 경계 사례의 민감도는 여기서 재지 않고 eval/robustness_benchmark.py 가 잰다.
# 수치는 임상 사례를 흉내낸 시연용이며 진단 근거가 아니다.
SPECS = [
    # 추세(하루 단위 상승)를 먼저 넣는다: 개인 기준선이 이미 그 영향을 받은 상태에서 personal_deviation 목표치를 잡아야 일관된다.
    Spec("elevated_resting_trend", "trend", "warning", (2, 3), (12, 16)),
    Spec("elevated_resting_trend", "trend", "warning", (2, 2), (12, 16)),
    Spec("resting_tachycardia", "awake", "warning", (25, 40), (110, 122)),
    Spec("resting_tachycardia", "sleep", "warning", (26, 34), (106, 114)),
    Spec("resting_tachycardia", "awake", "critical", (24, 32), (142, 160)),
    Spec("bradycardia", "awake", "warning", (28, 36), (36.5, 38.5)),
    Spec("bradycardia", "sleep", "warning", (28, 36), (31.0, 33.5)),
    Spec("bradycardia", "awake", "critical", (26, 34), (29, 33)),
    Spec("personal_deviation", "awake", "warning", (32, 45), (0, 0)),
    Spec("personal_deviation", "sleep", "warning", (32, 45), (0, 0)),
    Spec("personal_deviation", "awake", "warning", (32, 45), (0, 0)),
    Spec("sensor_gap", "gap", "info", (150, 240), (0, 0)),
    Spec("sensor_gap", "gap", "info", (130, 200), (0, 0)),
    Spec("sensor_gap", "gap", "info", (180, 260), (0, 0)),
]


def _try_point_event(g, rng, sp: Spec, blocked, event_id) -> dict | None:
    """분 단위 이벤트(빈맥/서맥/개인 이탈/센서 결측) 1건을 격자에 주입한다.

    Args:
        g: 사용자 격자 (수정됨). rng: numpy Generator. sp: 명세. blocked: 점유 분 마스크 (수정됨).
        event_id: 라벨 ID.

    Returns:
        {"start","end"(마지막 분 포함),"params"} 또는 배치할 윈도/기준선이 없으면 None.
        personal_deviation 은 목표가 절대 빈맥 임계 아래(각성 96, 수면 92 bpm)에 머무를 때만 만든다.
    """
    L = int(rng.integers(sp.dur[0], sp.dur[1] + 1))
    starts = _find_starts(g, L, sp.kind, blocked)
    if len(starts) == 0:
        return None
    ramp = int(rng.integers(3, 7))
    level = None
    if sp.type == "personal_deviation":
        b = fit_baseline(_for_detector(g))
        med, sig, cap = (b.sleep_median, b.sleep_sigma, 92) if sp.kind == "sleep" else (b.rest_median, b.rest_sigma, 96)
        if med is None:
            return None
        level = med + max(5.5 * sig, 20 if sp.kind == "awake" else 22)
        if level > cap:                      # 절대 임계(100) 아래에 머무르는 '개인 대비' 이상만 만든다
            return None
    elif sp.type != "sensor_gap":
        level = float(rng.uniform(*sp.level))
    s = int(rng.choice(starts))
    if sp.type == "sensor_gap":
        sl = slice(s, s + L)
        g.iloc[sl, [g.columns.get_loc(c) for c in HR_COLS]] = np.nan
        g.iloc[sl, g.columns.get_loc("hr_n")] = 0
        g.iloc[sl, g.columns.get_loc("injected")] = event_id
        params = {"gap_minutes": L}
    else:
        sigma = 1.2 if level < 45 else 1.5 if sp.kind == "sleep" else 2.5     # 서맥 구간은 심박이 더 안정적
        _overwrite(g, rng, s, L, level, ramp, sigma, event_id, sp.kind == "sleep")
        params = {"level": round(level, 1), "ramp_min": ramp, "duration_min": L}
    blocked[s:s + L] = True
    return {"start": g.index[s], "end": g.index[s + L - 1], "params": params}


def _try_trend_event(g, rng, sp: Spec, blocked, event_id, margin: float | None = TREND_MARGIN_BPM) -> dict | None:
    """연속 며칠의 휴식기 심박을 일정 폭 끌어올리는 추세 이벤트 1건을 주입한다.

    휴식·수면 분(활동 분 제외)만 올리고 98 bpm 을 넘기지 않게 상한을 둔다 (빈맥으로 오인되는 부수 효과 억제).

    Args:
        g, rng, sp, blocked, event_id: `_try_point_event` 와 동일. sp.dur 는 일수, sp.level 은 상승폭(bpm).
        margin: 후보 조건 하한 (TREND_MARGIN_BPM). None 이면 조건 없이 배치한다 (벤치마크에서 작은 상승폭 측정용).

    Returns:
        {"start"(자정),"end"(마지막 날 23:59),"params"} 또는 후보가 없으면 None.
    """
    k = int(rng.integers(sp.dur[0], sp.dur[1] + 1))
    lift = float(rng.uniform(*sp.level))
    daily = {d["date"]: d["resting_hr"] for d in daily_resting_hr(_for_detector(g))}
    days = pd.to_datetime(sorted(daily))
    cands = []
    for d in days:
        span = [(d + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(k)]
        prior = [(d - pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 8)]
        if any(daily.get(x) is None for x in span):
            continue
        prior_vals = [daily[x] for x in prior if daily.get(x) is not None]
        if len(prior_vals) < MIN_PRIOR_VALID_DAYS:
            continue
        # 실제 일별 변동이 커서 상승폭을 더해도 +10 bpm 에 못 미치는 날은 원리상 탐지 불가 → 후보에서 제외
        if margin is not None and min(daily[x] for x in span) + lift - float(np.median(prior_vals)) < margin:
            continue
        lo, hi = g.index.get_indexer([d], method="bfill")[0], g.index.get_indexer([d + pd.Timedelta(days=k)], method="bfill")[0]
        if hi < 0:
            hi = len(g)
        if not blocked[lo:hi].any():
            cands.append((d, lo, hi))
    if not cands:
        return None
    d, lo, hi = cands[int(rng.integers(len(cands)))]
    ctx = context(g["steps"], g["asleep"]).to_numpy()
    idx = np.arange(len(g))
    mask = (idx >= lo) & (idx < hi) & (ctx != "active") & (g["hr_n"].to_numpy() > 0)
    old = g["hr_mean"].to_numpy()[mask]
    for c in HR_COLS:
        col = g[c].to_numpy().copy()
        col[mask] = np.minimum(col[mask] + lift + rng.normal(0, 1.5, mask.sum()), np.maximum(col[mask], 98))
        g[c] = col
    inj = g["injected"].to_numpy(dtype=object, copy=True)
    inj[mask] = event_id
    g["injected"] = inj
    blocked[lo:hi] = True
    return {"start": d, "end": d + pd.Timedelta(days=k) - pd.Timedelta(minutes=1),
            "params": {"lift_bpm": round(lift, 1), "days": k, "n_minutes": int(mask.sum()),
                       "mean_before": round(float(np.nanmean(old)), 1)}}


def inject(df: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, list[dict]]:
    """SPECS 의 합성 이상을 실제 사용자 데이터 위에 주입한다.

    사용자는 시드 고정 순열의 순환(round-robin) 방식으로 정하고, 이벤트 간 BLOCK_MARGIN_MIN 분 이상 떨어뜨린다.
    추세 이벤트를 가장 먼저 넣어 이후 이벤트의 기준선 계산에 그 영향이 이미 반영되게 한다.

    Args:
        df: `load_minute` 결과 (원본 Id).
        seed: numpy 시드. 같은 입력과 시드면 결과가 비트 단위로 같다.

    Returns:
        (주입된 df — injected 컬럼 포함, 라벨 목록 — user_id 는 아직 원본 Id, start/end 는 Timestamp).

    Raises:
        RuntimeError: 어떤 사용자에게도 이벤트를 배치하지 못했을 때.
    """
    rng = np.random.default_rng(seed)
    days = df.assign(d=df["ts"].dt.date).loc[df["hr_n"] > 0].groupby("Id")["d"].nunique()
    elig = sorted(days[days >= MIN_INJECT_DAYS].index)
    order = list(rng.permutation(elig))
    grids = {u: _to_grid(df[df["Id"] == u]) for u in elig}
    blocked = {u: np.zeros(len(grids[u]), bool) for u in elig}
    counts = {u: 0 for u in elig}
    labels = []
    for i, sp in enumerate(SPECS):
        for j in range(len(order)):                      # 순환 배정으로 사용자에게 분산
            u = order[(i + j) % len(order)]
            if counts[u] >= MAX_EVENTS_PER_USER:
                continue
            eid = f"EVT-{len(labels) + 1:02d}"
            fn = _try_trend_event if sp.kind == "trend" else _try_point_event
            ev = fn(grids[u], rng, sp, blocked[u], eid)
            if ev:
                counts[u] += 1
                labels.append({"event_id": eid, "user_id": u, "type": sp.type, "start": ev["start"], "end": ev["end"],
                               "severity": sp.severity, "params": {"kind": sp.kind, **ev["params"]}})
                break
        else:
            raise RuntimeError(f"주입 실패: {sp}")
    parts = [_compact(grids[u]).assign(Id=u) for u in elig]
    rest = df[~df["Id"].isin(elig)].assign(injected=None)
    out = pd.concat(parts + [rest], ignore_index=True).sort_values(["Id", "ts"]).reset_index(drop=True)
    return out, labels


# ---------------------------------------------------------------- 출력
def _finalize(df: pd.DataFrame, id_map: dict) -> pd.DataFrame:
    """원본 Id 를 가명 user_id 로 바꾸고 출력 스키마(컬럼 순서·dtype)로 정리한다."""
    df = df.assign(user_id=df["Id"].map(id_map))
    df["hr_n"] = df["hr_n"].astype("int32")
    df["steps"] = df["steps"].astype("int32")
    df["ts"] = df["ts"].astype("datetime64[ns]")
    cols = ["user_id", "ts", "hr_mean", "hr_min", "hr_max", "hr_n", "steps", "asleep", "injected"]
    return df[cols].sort_values(["user_id", "ts"]).reset_index(drop=True)


def report(df: pd.DataFrame, labels: list[dict]) -> None:
    """사용자별 기간·일수·심박 행 수·커버리지(기록 구간 대비 심박 분 비율 %)를 표준 출력에 요약한다 (CLI 리포트)."""
    hr = df[df["hr_n"] > 0]
    print(f"[데이터 리포트] 사용자 {df['user_id'].nunique()}명, 총 {len(df):,}행 (심박 있는 분 {len(hr):,})")
    print(f"{'user':5} {'start':10} {'end':10} {'days':>4} {'hr_rows':>7} {'coverage%':>9}")
    for u, p in hr.groupby("user_id"):
        span = (p["ts"].max() - p["ts"].min()).total_seconds() / 60 + 1
        print(f"{u:5} {p['ts'].min():%Y-%m-%d} {p['ts'].max():%Y-%m-%d} {p['ts'].dt.date.nunique():>4} "
              f"{len(p):>7} {100 * len(p) / span:>9.1f}")
    print(f"주입 이벤트 {len(labels)}건, 주입 분 {df['injected'].notna().sum():,}")


def main(no_inject: bool = False) -> None:
    """raw -> processed 전체 파이프라인. 기존 산출물은 덮어쓴다 (멱등).

    Args:
        no_inject: True 면 합성 이상을 넣지 않는다 (vitals_minute.parquet == clean, labels.json == []).

    Raises:
        FileNotFoundError: raw CSV 가 없을 때.
    """
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    raw = load_minute()
    id_map = {raw_id: f"U{i + 1:02d}" for i, raw_id in enumerate(sorted(raw["Id"].unique()))}
    clean = raw.assign(injected=None)
    labels: list[dict] = []
    injected = clean
    if not no_inject:
        injected, labels = inject(raw)
    _finalize(clean, id_map).to_parquet(settings.data_dir / "vitals_minute_clean.parquet", index=False)
    final = _finalize(injected, id_map)
    final.to_parquet(settings.data_dir / "vitals_minute.parquet", index=False)
    for lb in labels:
        lb["user_id"] = id_map[lb["user_id"]]
        lb["start"], lb["end"] = lb["start"].isoformat(), lb["end"].isoformat()
    (settings.data_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2))
    (settings.data_dir / "id_map.json").write_text(
        json.dumps({v: str(k) for k, v in id_map.items()}, ensure_ascii=False, indent=2))
    report(final, labels)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-inject", action="store_true", help="합성 이상을 주입하지 않는다")
    main(ap.parse_args().no_inject)

"""웨어러블 심박 이상 탐지 연구 방법 구현 (선택적 탐지 컴포넌트).

책임 경계:
    시간 단위 특징 계산, EllipticEnvelope 기반 이상 시간 판정, 일별 CuSum 만 제공한다. Anomaly 객체 생성과 규칙 탐지와의
    병합은 `detector.py` 가 한다. 어떤 구성을 기본으로 쓸지는 `eval/robustness_benchmark.py` 의 튜닝 시드 결과가 정한다.

협력 모듈: `baseline.minute_grid`(입력 격자), `detector.detect`(호출자), scikit-learn(EllipticEnvelope, StandardScaler).

데이터 가정: 입력은 baseline.minute_grid 형식(1분 간격, naive 로컬 시각). 시간 특징은 달력 시(hour) 경계로 집계하며
단위는 bpm, 걸음/분 이다. 적합에는 72시간 이상의 이력이 필요하다.

출처 (모두 직접 소스를 읽고 파라미터를 가져옴):
- Mishra et al. 2020, Nat. Biomed. Eng. 의 공개 코드: https://github.com/gireeshkbogu/AnomalyDetect
  (커밋 1484183, scripts/rhrad_offline.py · hrosad_offline.py · rhrad_online_24hr_alerts_v7.py)
  - RHR(안정시 심박) = 직전 12분(현재 분 포함) 걸음 합이 0인 분의 심박
  - HROS(Heart Rate Over Steps) = 심박 / (걸음 + 1)
  - 1시간 평균으로 다운샘플 → 표준화(StandardScaler) → EllipticEnvelope(contamination=0.1,
    support_fraction=0.7, random_state=10) 로 가우시안 밀도 기반 이상 판정
  - 온라인판: 기준(학습) 창 744시간(1개월), 1시간 단위로 판정, 24시간 내 이상 시간 1개 이상=YELLOW, 6개 이상=RED
  원본과 다른 점(의도적): ① seasonal_decompose(freq=1)는 trend+resid = 원 시계열이라 항등이므로 생략
  ② 400샘플 이동평균 평활은 수십 분~수 시간 이벤트를 뭉개므로 생략하고 시간 평균만 쓴다
  ③ 수면/각성 시간대를 따로 적합(혼합 분포에서는 수면 중 상승이 가려짐) ④ 상승 방향만 이상으로 취급
  ⑤ 연속 시간 수(min_hours)·최소 상승폭(min_delta_bpm) 조건 추가 — 원본에는 없는 알림 부담 억제 장치.
- CuSum: Alavi et al. 2022, Nat. Med. 의 실시간 알림이 CuSum 계열을 쓴다고 알려져 있으나 공개 소스를 찾지 못했다.
  따라서 Page(1954)의 표준 단측 CuSum 정의로 직접 구현했고 파라미터(k, h)는 이 저장소의 튜닝 시드로 정했다.
  (논문 파라미터를 재현한 것이 아니다.)
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler

# Mishra et al. rhrad_*.py: "preceding 12 minutes (including the current minute)" 걸음 합 0
RHR_STEPS_WINDOW_MIN = 12
EE_SUPPORT_FRACTION = 0.7
EE_RANDOM_SEED = 10
# 시간 평균을 믿으려면 그 시간에 필요한 최소 (휴식) 분 수
MIN_MINUTES_PER_HOUR = 20
# 가우시안 적합에 필요한 최소 시간 수 (3일). 모자라면 판정하지 않는다.
MIN_FIT_HOURS = 72
HOLE_HOURS = 1            # 이상 시간 사이의 1시간 결측/정상은 같은 이벤트로 잇는다

CUSUM_BASE_DAYS = 14      # 기준(평소) 계산에 쓰는 직전 정상 일수 상한
CUSUM_LOOKBACK_DAYS = 21  # 기준 후보로 보는 직전 달력 일수 (너무 오래된 값은 개인 기준이 변했을 수 있어 제외)
CUSUM_MIN_BASE_DAYS = 3
CUSUM_MIN_SIGMA = 3.0     # 일별 안정시 심박의 sigma 하한 (bpm) — 일별 값 자체의 측정 변동(착용 시간대 구성 차이)을 반영


def hourly_features(g: pd.DataFrame) -> pd.DataFrame:
    """분 격자를 시간 단위 특징으로 집계한다.

    Args:
        g: `baseline.minute_grid` 결과.

    Returns:
        index=시(hour 시작) 이고 컬럼이 rhr(안정시 심박 평균, bpm), n(안정시 분 수), hros(평균 심박/(걸음+1)),
        n_hr(심박 있는 분 수), steps(분당 평균 걸음), asleep(수면 분 비율 0~1)인 DataFrame.
        그 시간에 필요한 분 수(MIN_MINUTES_PER_HOUR)가 모자라면 해당 특징은 NaN.
    """
    steps12 = g["steps"].rolling(RHR_STEPS_WINDOW_MIN, min_periods=RHR_STEPS_WINDOW_MIN).sum()
    rest = (steps12 == 0) & g["hr"].notna()
    rhr = g["hr"].where(rest)
    hros = g["hr"] / (g["steps"] + 1)
    h = pd.DataFrame({
        "rhr": rhr.resample("1h").mean(), "n": rest.astype(int).resample("1h").sum(),
        "hros": hros.resample("1h").mean(), "n_hr": g["hr"].notna().astype(int).resample("1h").sum(),
        "steps": g["steps"].resample("1h").mean(), "asleep": g["asleep"].astype(float).resample("1h").mean(),
    })
    h.loc[h["n"] < MIN_MINUTES_PER_HOUR, "rhr"] = np.nan
    h.loc[h["n_hr"] < MIN_MINUTES_PER_HOUR, ["hros", "steps"]] = np.nan
    return h


def _fit_predict(fit: pd.DataFrame, test: pd.DataFrame, cols: list[str], contamination: float) -> pd.Series:
    """fit 시간들로 EllipticEnvelope 를 적합해 test 시간별 이상(True) 여부를 반환한다 (적합 불가면 전부 False).

    cols 의 첫 특징이 학습 평균보다 높은 쪽 이상만 True 로 둔다 (낮은 쪽은 이 서비스의 관심사가 아님).
    """
    X = fit[cols].dropna()
    out = pd.Series(False, index=test.index)
    if len(X) < MIN_FIT_HOURS:
        return out
    T = test[cols].dropna()
    if T.empty:
        return out
    sc = StandardScaler().fit(X.values)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = EllipticEnvelope(contamination=contamination, support_fraction=EE_SUPPORT_FRACTION,
                                     random_state=EE_RANDOM_SEED).fit(sc.transform(X.values))
            pred = model.predict(sc.transform(T.values))
    except (ValueError, np.linalg.LinAlgError):      # 공분산 특이(상수 특징 등)
        return out
    z_first = sc.transform(T.values)[:, 0]            # 첫 특징(rhr 또는 hros)이 평균보다 높은 쪽만 이상으로 본다
    out.loc[T.index] = (pred == -1) & (z_first > 0)
    return out


def hourly_anomaly_hours(feat: pd.DataFrame, test_from: pd.Timestamp, mode: str, contamination: float,
                         min_delta_bpm: float) -> pd.Series:
    """시간별 이상 여부를 판정한다.

    Args:
        feat: `hourly_features` 결과 (판정 대상 이전 이력 포함 — 적합에 쓴다).
        test_from: 이 시각(포함) 이후의 시간만 보고한다.
        mode: 'rhr'(RHR-AD, 안정시 심박 단변량) | 'hros'(HROS-AD, [HROS, 걸음]) | 'both'(합집합).
        contamination: EllipticEnvelope contamination.
        min_delta_bpm: 안정시 심박이 해당 수면/각성 그룹 중앙값보다 이만큼(bpm) 이상 높은 시간만 인정.

    Returns:
        feat 와 같은 index 의 bool Series. 수면/각성 시간은 따로 적합한다.
    """
    test_mask = feat.index >= test_from
    flag = pd.Series(False, index=feat.index)
    cols_by_mode = {"rhr": [["rhr"]], "hros": [["hros", "steps"]], "both": [["rhr"], ["hros", "steps"]]}
    for sleep in (False, True):
        grp = (feat["asleep"] >= 0.5) == sleep
        fit, test = feat[grp], feat[grp & test_mask]
        for cols in cols_by_mode[mode]:
            flag |= _fit_predict(fit, test, cols, contamination).reindex(feat.index, fill_value=False)
        med = fit["rhr"].median()
        flag &= ~grp | (feat["rhr"] - med >= min_delta_bpm)     # 상승폭 가드 (그룹별 중앙값 기준; 안정시 심박 없는 시간은 제외)
    return flag & test_mask


def runs_of_hours(flag: pd.Series, min_hours: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """True 시간의 연속 구간(HOLE_HOURS 시간 구멍은 이음)을 [(시작 시각, 마지막 분)] 로 반환한다.

    Args:
        flag: 시간 index 의 bool Series.
        min_hours: 구간 안에서 True 인 시간이 이 수 이상인 것만 남긴다.
    """
    idx = np.flatnonzero(flag.to_numpy())
    if idx.size == 0:
        return []
    cut = np.flatnonzero(np.diff(idx) > HOLE_HOURS + 1)
    starts, ends = np.r_[idx[0], idx[cut + 1]], np.r_[idx[cut], idx[-1]]
    out = []
    for s, e in zip(starts, ends):
        if flag.iloc[s:e + 1].sum() >= min_hours:
            out.append((flag.index[s], flag.index[e] + pd.Timedelta(minutes=59)))
    return out


def cusum_episodes(daily: pd.Series, k: float, h: float, min_days: int = 2) -> list[dict]:
    """일별 안정시 심박(날짜 인덱스, 유효한 날만)에 단측 CuSum 을 적용해 '상승' 에피소드를 반환한다.

    z_d = (x_d - mu_d) / sigma_d, S_d = max(0, S_{d-1} + z_d - k), S_d > h 이면 알림 후 S 를 0 으로 리셋한다.
    mu/sigma 는 직전 CUSUM_LOOKBACK_DAYS 일 안의 최근 CUSUM_BASE_DAYS 일 중앙값/robust sigma 이며,
    이미 알림에 포함된 날은 기준에서 뺀다 (상승일이 기준을 끌어올려 다음 날을 놓치는 것 방지).
    알림 직후 이어지는 에피소드(1일 이내 간격)는 하나로 합치고, min_days 일 미만이면 버린다.
    반환: [{"start","end"(Timestamp 날짜, 포함),"peak","mu"}]"""
    hist: list[tuple[pd.Timestamp, float]] = []
    flagged: set[pd.Timestamp] = set()
    S, run, out = 0.0, [], []
    for d, x in daily.items():
        base = [v for dd, v in hist if (d - dd).days <= CUSUM_LOOKBACK_DAYS and dd not in flagged][-CUSUM_BASE_DAYS:]
        hist.append((d, float(x)))
        if len(base) < CUSUM_MIN_BASE_DAYS:
            continue
        mu = float(np.median(base))
        sigma = max(1.4826 * float(np.median(np.abs(np.array(base) - mu))), CUSUM_MIN_SIGMA)
        S = max(0.0, S + (x - mu) / sigma - k)
        run = run + [d] if S > 0 else []
        if S > h:
            out.append({"start": run[0], "end": d, "peak": S, "mu": mu})
            flagged.update(run)
            S, run = 0.0, []
    merged: list[dict] = []
    for ep in out:
        if merged and ep["start"] - merged[-1]["end"] <= pd.Timedelta(days=1):
            merged[-1].update(end=ep["end"], peak=max(merged[-1]["peak"], ep["peak"]))
        else:
            merged.append(dict(ep))
    return [ep for ep in merged if (ep["end"] - ep["start"]).days + 1 >= min_days]

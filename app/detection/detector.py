"""결정론적 심박 이상 탐지 — LLM 을 쓰지 않는 규칙 기반 탐지기 (+ 선택적 연구 방법).

책임 경계:
    분 단위 생체 데이터에서 이상 구간(Anomaly)을 찾고 요약 통계를 계산한다. 같은 입력이면 항상 같은 결과다.
    이상의 '해석'과 리포트 서술은 하지 않는다 — 그것은 LLM 의 몫이다. 위험 안내문 강제는 에이전트의
    안전 게이트가 critical 여부만 보고 결정한다.

협력 모듈:
    `baseline.py`(1분 격자, 맥락, 개인 기준선), `research.py`(시간 단위 EllipticEnvelope, CuSum),
    `app.mcp_server.vitals_server`(호출자), `eval/*`(성능 평가).

데이터 가정:
    - 입력은 VitalsStore 형식(희소 분 데이터)이며 ts 는 기기 로컬 시간의 timezone-naive 값이다.
    - 이상 구간의 end 는 마지막 분을 포함한다 (duration_min = 분 수).
    - 단위: 심박 bpm, 지속시간 분(minute).

운동 억제와 그 천장:
    걸음이 있거나 멈춘 직후 REST_WINDOW_MIN 분 이내(활동 맥락)의 심박 상승은 정상 반응으로 보고 빈맥으로 알리지 않는다.
    다만 이 억제가 '걸음이 조금이라도 있으면 무엇이든 묵살'이 되지 않도록 활동 맥락과 무관한 극단 빈맥 규칙을 둔다
    (`_extreme_tachycardia`): 평활 심박 >= 210 bpm 이 10분 이상이면 걸음과 무관하게 critical, >= 180 bpm 이고 최근 15분 안에
    분당 60보 이상으로 걸은 적이 없으면 critical. 근거는 연령 예측 최대 심박 추정식(220-나이 등, 18세 약 202, 40세 이상 180 이하).
    230 bpm 초과 값은 버리지 않는다 — 센서 오류 상한은 300 bpm 이며(발작성 상심실성 빈맥은 250 안팎까지 보고됨), 짧은 스파이크는
    10분 지속 조건이 걸러내고 지속되면 위 규칙이 critical 로 올린다. 걸음 0인 고강도 운동(자전거 등)은 휴식과 구분되지 않아
    오탐 가능성이 남는 알려진 한계다.

임계값은 일반적으로 알려진 임상 가이드라인 수준의 상식(성인 안정시 심박 60~100 bpm 등)에 근거한 보수적 기준이며,
이 서비스는 의료기기가 아니다. 값은 '의심 신호를 사람에게 알리는' 목적이다. 구체 출처 표기는 인용 가능한 것만 쓴다.
"""
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from app.detection import research
from app.detection.baseline import (
    REST_WINDOW_MIN,
    Baseline,
    dst_gap_minutes,
    daily_resting_hr,
    fit_baseline,
    minute_grid,
    resting_hr_of,
)

# ---- 임계값 (근거는 주석) -------------------------------------------------------------
# 성인 안정시 심박 정상 범위 60~100 bpm (일반적 가이드라인). 100 초과가 지속되면 빈맥으로 분류한다.
TACHY_BPM = 100
# 안정 중 130 이상은 정상 생리 반응으로 설명되기 어려워 즉시 확인이 필요한 수준으로 본다 (critical).
TACHY_CRITICAL_BPM = 130
# 지속 시간 10분: 불안·카페인·자세 변화 같은 일시적 상승(수 분)을 걸러내기 위한 최소 지속 시간.
TACHY_MIN_MIN = 10
# 운동 억제의 천장: 활동 맥락이어도 다음은 '운동으로 설명 불가'로 보고 critical 후보로 올린다.
# 연령 예측 최대 심박은 220-나이 (Fox) 또는 208-0.7*나이 (Tanaka) 등 일반적으로 알려진 추정식이며 개인차가 크다(+-10~12 bpm).
# 성인(18세 이상) 상한 추정이 약 202이므로 210 이상이 10분 이상 지속되면 나이·운동과 무관하게 비정상으로 본다.
EXTREME_BPM = 210
# 40세 이상의 예측 최대 심박은 180 이하. 걷기 수준 이하의 보행(분당 60보 미만)인데 180 이상이 지속되면 보행으로 설명되지 않는다.
EXTREME_LOW_CADENCE_BPM = 180
LOW_CADENCE_SPM = 60     # 분당 걸음 수. 보통 걷기는 분당 약 100보 이상(중강도 보행의 일반적 기준), 60 미만이면 서행/제자리
EXTREME_MIN_MIN = 10     # 일시적 센서 오류(수 분)를 거르기 위한 최소 지속 시간 — 일반 빈맥 규칙과 같다
# 서맥은 일반적으로 60 미만으로 정의하나, 운동선수·수면 중에는 흔하다. 임상적으로 의미 있는 40 미만만 잡는다.
BRADY_AWAKE_BPM = 40
# 수면 중에는 심박이 자연히 낮아지므로 더 낮은 35 미만만 이상으로 본다 (수면 중 기준 완화).
BRADY_SLEEP_BPM = 35
# 각성 시 35 미만, 어떤 상황이든 30 미만은 위험 수준(critical)으로 본다.
BRADY_CRITICAL_AWAKE_BPM = 35
BRADY_CRITICAL_SLEEP_BPM = 30
BRADY_MIN_MIN = 10
# 개인 기준선 대비 robust z > 4 (정규분포 환산 약 4σ, 우연히 넘을 확률이 극히 낮은 기준), 15분 이상 지속.
DEVIATION_Z = 4.0
DEVIATION_MIN_MIN = 15
# MAD가 매우 작은 사용자에서 z만으로 사소한 변화를 잡지 않도록 요구하는 최소 절대 상승폭.
DEVIATION_MIN_DELTA_BPM = 15
# 안정시 심박이 개인 최근 7일 중앙값보다 10 bpm 이상 높은 날이 2일 이상 연속이면 추세 이상.
# 감염·과로·탈수 등에서 안정시 심박이 오르는 것으로 알려져 있다 (진단 근거는 아님).
TREND_DELTA_BPM = 10
TREND_LOOKBACK_DAYS = 7
TREND_MIN_PRIOR_DAYS = 3
TREND_MIN_DAYS = 2
# 센서 결측: 2시간 이상, 단 결측 구간에 걸음이 기록돼(착용 증거) 있고 12시간 이하(그 이상은 충전·미착용으로 간주).
GAP_MIN_MIN = 120
GAP_MAX_MIN = 12 * 60
# 조건을 잠깐 벗어난 1~2분(센서 노이즈)로 한 이벤트가 쪼개지지 않도록 허용하는 간격.
MAX_BRIDGE_MIN = 2
# 절대 임계 이상 징후와 겹치는(±분) 개인 대비 이상은 중복 알림이므로 버린다.
OVERLAP_PAD_MIN = 5



@dataclass(frozen=True)
class DetectorConfig:
    """탐지 구성. 빈맥/서맥/센서 결측 규칙은 항상 켜져 있고, 개인 이탈과 추세만 방법을 고를 수 있다.

    기본값 DEFAULT_CONFIG 는 eval/robustness_benchmark.py 가 튜닝 시드로 고른 구성이다 (선택 규칙은 eval/robustness_benchmark.py 참고).

    Attributes:
        trend: elevated_resting_trend 판정 방식. 'rule'(7일 중앙값 +10 bpm 2일 연속) | 'cusum' | 'both'(합집합).
        hourly: 시간 단위 EllipticEnvelope(연구 방법)를 personal_deviation 으로 보고할지. 'off' | 'rhr' | 'hros' | 'both'.
        contamination: EllipticEnvelope 가 학습 시간 중 이상으로 볼 비율 (원본 코드 기본 0.1).
        min_hours: 이상 시간이 연속(1시간 구멍 허용)해야 하는 최소 시간 수.
        min_delta_bpm: 시간 단위 안정시 심박이 개인 시간대 중앙값보다 높아야 하는 최소폭 (bpm).
        cusum_k: CuSum 허용 편차 (sigma 단위). cusum_h: 알림 임계 (sigma 단위).
        personal_rule: False 면 규칙 기반 personal_deviation 을 끈다 (연구 방법만 비교하는 벤치마크용).
    """
    trend: Literal["rule", "cusum", "both"] = "rule"
    hourly: Literal["off", "rhr", "hros", "both"] = "off"
    contamination: float = 0.1
    min_hours: int = 2
    min_delta_bpm: float = 0.0
    cusum_k: float = 0.5
    cusum_h: float = 4.0
    personal_rule: bool = True


# eval/robustness_benchmark.py 가 튜닝 시드로 고른 구성 (eval/results/selected_config.json 과 테스트로 일치 검증).
# 규칙 + 시간 단위 안정시 심박 EllipticEnvelope(contamination 0.02, 2시간 이상, +8 bpm 이상) + 규칙/CuSum(k=1, h=4) 추세의 합집합.
DEFAULT_CONFIG = DetectorConfig(trend="both", hourly="rhr", contamination=0.02, min_hours=2, min_delta_bpm=8.0,
                                cusum_k=1.0, cusum_h=4.0)

AnomalyType = Literal[
    "resting_tachycardia", "bradycardia", "personal_deviation", "elevated_resting_trend", "sensor_gap"]


@dataclass
class Anomaly:
    """탐지된 이상 구간 하나.

    Attributes:
        anomaly_id: "{type}-{start:%Y%m%d%H%M}" — 같은 입력에서 항상 같은 값이라 UI/리포트에서 참조할 수 있다.
        type: AnomalyType 중 하나.
        start: 시작 분 (포함, naive 로컬 시각). end: 마지막 분 (포함).
        duration_min: 구간 길이 (분). 추세는 일수 x 1440.
        severity: 'info' | 'warning' | 'critical' (critical 은 에이전트 안전 게이트의 트리거).
        value: 대표값 — 구간 평균 심박(bpm), 추세는 구간 안정시 심박(bpm), 센서 결측은 결측 길이(분).
        baseline: 비교 기준 (bpm). 개인 기준선이 없거나 해당 없으면 None.
        context: 'rest' | 'sleep' | 'active' (구간에서 가장 많은 맥락).
        evidence: 사용자 화면에 그대로 보이는 쉬운 한국어 한 줄 근거 (숫자 포함, 통계 용어 없음). 리포트 인용 숫자의 출처가 된다.
        stats: 통계 세부값(임계값, 표준화 점수, 최고치 등). 사람이 읽는 화면용이 아니라 분석/검증용 부가 필드다.
    """
    anomaly_id: str
    type: str
    start: pd.Timestamp
    end: pd.Timestamp
    duration_min: int
    severity: Literal["info", "warning", "critical"]
    value: float
    baseline: float | None
    context: str
    evidence: str
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON 직렬화용 dict (start/end 는 ISO 8601 문자열)."""
        d = asdict(self)
        d["start"], d["end"] = self.start.isoformat(), self.end.isoformat()
        d["stats"] = {k: v.item() if hasattr(v, "item") else v for k, v in self.stats.items()}     # numpy 스칼라 -> 파이썬 값
        return d


def _runs(mask: np.ndarray, min_len: int, max_gap: int = MAX_BRIDGE_MIN) -> list[tuple[int, int]]:
    """mask 의 True 연속 구간 [(시작, 끝) 인덱스, 끝 포함]을 찾는다.

    Args:
        mask: 분 격자 위의 조건 마스크 (NaN 에서 비교는 False 라 결측 분은 조건 불충족).
        min_len: 최소 길이(분, 이어 붙인 span 기준).
        max_gap: 이 분 수 이하의 끊김은 같은 구간으로 이어 붙인다 (센서 노이즈로 이벤트가 쪼개지는 것 방지).
    """
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    cut = np.flatnonzero(np.diff(idx) > max_gap + 1)
    starts, ends = np.r_[idx[0], idx[cut + 1]], np.r_[idx[cut], idx[-1]]
    keep = ends - starts + 1 >= min_len
    return list(zip(starts[keep].tolist(), ends[keep].tolist()))


def _evidence(type_: str, ctx: str, value: float, baseline: float | None, duration_min: int, stats: dict) -> str:
    """Anomaly 필드로부터 사용자 화면용 쉬운 한국어 근거 문장을 만든다.

    생성 시점과 `_merge` 로 구간이 합쳐진 뒤에 같은 함수를 쓰므로, 문장 속 지속시간·평균 값이 항상 현재 구간과 일치한다.

    Args:
        type_: AnomalyType. ctx: 'rest' | 'sleep' | 'active'. value: 대표값(bpm 등). baseline: 비교 기준(bpm) 또는 None.
        duration_min: 구간 길이(분). stats: 유형별 부가 값 (method, steps_in_gap, cadence_spm 등).
    """
    c = _CTX_KO.get(ctx, "휴식")
    method = stats.get("method", "")
    if type_ == "resting_tachycardia":
        if method == "extreme":
            return (f"{c} 중 활동으로 설명하기 어려울 만큼 높은 심박(평균 {value:.0f} bpm)이 {duration_min}분 동안 이어졌어요. "
                    f"그동안 걸음은 분당 평균 {stats['cadence_spm']:.0f}보로 많지 않았어요.")
        return (f"{c} 중 심박이 평균 {value:.0f} bpm으로 {duration_min}분 동안 계속 높았어요. "
                f"쉬는 상태의 일반적인 범위는 60~{TACHY_BPM} bpm이에요"
                + (f" (평소 {c} 때 심박은 약 {baseline:.0f} bpm)." if baseline else "."))
    if type_ == "bradycardia":
        return f"{c} 중 심박이 평균 {value:.0f} bpm으로 {duration_min}분 동안 매우 낮았어요 (주의 기준: {stats['threshold_bpm']} bpm 미만)."
    if type_ == "personal_deviation":
        if method.startswith("hourly"):
            return (f"{c} 중 시간대별 심박이 평소 패턴과 다르게 높은 상태가 {round(duration_min / 60)}시간 이어졌어요 "
                    f"(평소 {baseline:.0f} -> {value:.0f} bpm).")
        return (f"{c} 중 심박이 평소보다 크게 높았어요 (평소 {baseline:.0f} -> {value:.0f} bpm, +{value - baseline:.0f}). "
                f"이런 상태가 {duration_min}분 이어졌어요.")
    if type_ == "elevated_resting_trend":
        days = round(duration_min / 1440)
        if method == "cusum":
            return (f"하루 중 가장 편안한 때의 심박(안정시 심박)이 {days}일 동안 평소보다 계속 높아졌어요 "
                    f"(평소 {baseline:.0f} -> {value:.0f} bpm, +{value - baseline:.0f}).")
        return (f"하루 중 가장 편안한 때의 심박(안정시 심박)이 {days}일 연속 평소보다 {TREND_DELTA_BPM} bpm 넘게 "
                f"높았어요 (최근 {TREND_LOOKBACK_DAYS}일 평소 {baseline:.0f} -> {value:.0f} bpm).")
    return (f"심박 신호가 {duration_min}분({duration_min / 60:.1f}시간) 동안 끊겼어요. 같은 시간에 걸음 {stats['steps_in_gap']}보가 "
            f"기록돼 있어서 기기를 찬 채로 센서가 심박을 놓친 것으로 보여요.")


def _span_minutes(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """[start, end] (끝 분 포함)의 실제 분 수 — 서머타임 전환으로 존재하지 않는 시간(02:00~02:59)은 뺀다."""
    return int((end - start) / pd.Timedelta(minutes=1)) + 1 - dst_gap_minutes(start, end)


def _new(type_: str, start: pd.Timestamp, end: pd.Timestamp, severity: str, value: float, baseline: float | None,
         ctx: str, stats: dict, duration_min: int | None = None) -> Anomaly:
    """Anomaly 를 만들고 근거 문장을 `_evidence` 로 채운다 (end 는 마지막 분, 포함).

    duration_min 을 주면(격자 행 수) 그대로 쓰고, 없으면 시각 차이에서 서머타임 결손을 뺀 값을 쓴다.
    """
    dur = int(duration_min) if duration_min is not None else int(_span_minutes(start, end))   # numpy 정수가 섞이면 MCP 출력 스키마 검증이 실패한다
    value, baseline = round(float(value), 1), None if baseline is None else round(float(baseline), 1)
    return Anomaly(f"{type_}-{start:%Y%m%d%H%M}", type_, start, end, dur, severity, value, baseline, ctx,
                   _evidence(type_, ctx, value, baseline, dur, stats), stats)


def _make(g: pd.DataFrame, s: int, e: int, type_: str, severity: str, value: float, baseline: float | None,
          ctx: str | None = None, stats: dict | None = None) -> Anomaly:
    """격자 인덱스 [s, e] (끝 포함)로 Anomaly 를 만든다. ctx 를 안 주면 구간의 최빈 맥락을 쓴다."""
    ctx = ctx or g["ctx"].iloc[s:e + 1].mode().iat[0]
    return _new(type_, g.index[s], g.index[e], severity, value, baseline, ctx, stats or {}, duration_min=e - s + 1)   # 격자 행 수 = 실제 분 수


_CTX_KO = {"rest": "휴식", "sleep": "수면", "active": "활동"}


def _tachycardia(g: pd.DataFrame, b: Baseline) -> list[Anomaly]:
    """휴식·수면 중 평활 심박 > TACHY_BPM 이 TACHY_MIN_MIN 분 이상 지속. 5분 이동평균 최고가 130 이상이면 critical."""
    # 활동(걸음 있음/직후) 맥락은 제외 — 운동 중 심박 상승은 정상 반응이다 (핵심 오탐 억제 규칙).
    # 활동 중이어도 설명이 안 되는 극단값은 `_extreme_tachycardia` 가 따로 잡는다.
    mask = ((g["hr_s"] > TACHY_BPM) & (g["ctx"] != "active")).to_numpy()
    out = []
    for s, e in _runs(mask, TACHY_MIN_MIN):
        seg = g.iloc[s:e + 1]
        peak = seg["hr"].rolling(5, min_periods=3).mean().max()
        ctx = seg["ctx"].mode().iat[0]
        base = b.sleep_median if ctx == "sleep" else b.rest_median
        sev = "critical" if peak >= TACHY_CRITICAL_BPM else "warning"
        out.append(_make(g, s, e, "resting_tachycardia", sev, seg["hr"].mean(), base, ctx,
                         {"peak_5min_mean_bpm": round(float(peak), 1), "threshold_bpm": TACHY_BPM,
                          "critical_bpm": TACHY_CRITICAL_BPM, "min_minutes": TACHY_MIN_MIN}))
    return out


def _extreme_tachycardia(g: pd.DataFrame, b: Baseline) -> list[Anomaly]:
    """활동 맥락과 무관한 극단 빈맥 (critical). 운동 억제 규칙의 상한(천장) 역할을 한다.

    다음 중 하나인 상태가 EXTREME_MIN_MIN 분 이상 지속되면 알린다.
      (a) 평활 심박 >= EXTREME_BPM — 성인의 연령 예측 최대 심박(220-나이, 18세 기준 202)보다 높아 운동으로 설명되지 않는다.
      (b) 평활 심박 >= EXTREME_LOW_CADENCE_BPM 이고 최근 REST_WINDOW_MIN 분 안에 3분 평균 분당 LOW_CADENCE_SPM 보 이상으로
          걸은 적이 없다 — 걷기(분당 약 100보 이상)로도, 그 직후의 회복으로도 설명되지 않는데 40세 이상의 예측 최대 심박(180) 이상이다.
    실내에서 몇 걸음씩만 움직이는 동안의 극단 빈맥이 '걸음이 조금 있다'는 이유로 묵살되지 않게 하기 위함이다.
    자전거·근력 운동처럼 걸음이 0인 고강도 운동은 휴식과 구분되지 않아 기존 규칙과 같이 오탐 가능성이 남는다(알려진 한계).
    """
    pace = g["steps"].rolling(3, min_periods=1).mean()                       # 분당 걸음(3분 평균)
    cadence = pace.rolling(REST_WINDOW_MIN, min_periods=1).max()             # 최근 워시아웃 동안 낸 가장 빠른 보행 속도
    hr = g["hr_s"]
    mask = ((hr >= EXTREME_BPM) | ((hr >= EXTREME_LOW_CADENCE_BPM) & (cadence < LOW_CADENCE_SPM))).to_numpy()
    out = []
    for s, e in _runs(mask, EXTREME_MIN_MIN):
        seg = g.iloc[s:e + 1]
        base = b.rest_median
        out.append(_make(g, s, e, "resting_tachycardia", "critical", seg["hr"].mean(), base, None,
                         {"method": "extreme", "cadence_spm": round(float(pace.iloc[s:e + 1].mean()), 1),
                          "extreme_bpm": EXTREME_BPM, "low_cadence_bpm": EXTREME_LOW_CADENCE_BPM,
                          "low_cadence_spm": LOW_CADENCE_SPM, "min_minutes": EXTREME_MIN_MIN}))
    return out


def _bradycardia(g: pd.DataFrame, b: Baseline) -> list[Anomaly]:
    """평활 심박이 각성 < 40 / 수면 < 35 bpm 인 상태가 10분 이상. 구간 평균이 각성 < 35 / 수면 < 30 이면 critical."""
    asleep = g["ctx"].eq("sleep").to_numpy()
    thr = np.where(asleep, BRADY_SLEEP_BPM, BRADY_AWAKE_BPM)
    mask = (g["hr_s"].to_numpy() < thr)            # 결측(NaN)은 False
    out = []
    for s, e in _runs(mask, BRADY_MIN_MIN):
        seg = g.iloc[s:e + 1]
        ctx = seg["ctx"].mode().iat[0]
        mean = seg["hr"].mean()
        crit_thr = BRADY_CRITICAL_SLEEP_BPM if ctx == "sleep" else BRADY_CRITICAL_AWAKE_BPM
        sev = "critical" if mean < crit_thr else "warning"
        base = b.sleep_median if ctx == "sleep" else b.rest_median
        out.append(_make(g, s, e, "bradycardia", sev, mean, base, ctx,
                         {"threshold_bpm": BRADY_SLEEP_BPM if ctx == "sleep" else BRADY_AWAKE_BPM,
                          "critical_bpm": crit_thr, "min_minutes": BRADY_MIN_MIN}))
    return out


def _personal_deviation(g: pd.DataFrame, b: Baseline) -> list[Anomaly]:
    """개인 맥락별 기준선 대비 robust z > 4 이고 +15 bpm 이상인 상태가 15분 이상 (심박은 절대 빈맥 임계 이하)."""
    # 수면/각성 휴식을 각자의 기준선과 비교한다. 빈맥 임계(100) 초과는 빈맥 규칙의 몫이라 여기서 제외해 중복 알림을 막는다.
    ctx = g["ctx"].to_numpy()
    med = np.select([ctx == "rest", ctx == "sleep"], [b.rest_median or np.nan, b.sleep_median or np.nan], np.nan)
    sig = np.select([ctx == "rest", ctx == "sleep"], [b.rest_sigma or np.nan, b.sleep_sigma or np.nan], np.nan)
    hr = g["hr_s"].to_numpy()
    with np.errstate(invalid="ignore"):
        z = (hr - med) / sig
        mask = (z > DEVIATION_Z) & (hr - med >= DEVIATION_MIN_DELTA_BPM) & (hr <= TACHY_BPM)
    out = []
    for s, e in _runs(mask, DEVIATION_MIN_MIN):
        mean = float(np.nanmean(g["hr"].to_numpy()[s:e + 1]))
        base = float(np.nanmedian(med[s:e + 1]))
        out.append(_make(g, s, e, "personal_deviation", "warning", mean, base, None,
                         {"method": "rule", "robust_z_mean": round(float(np.nanmean(z[s:e + 1])), 1),
                          "sigma_bpm": round(float(np.nanmean(sig[s:e + 1])), 1), "z_threshold": DEVIATION_Z,
                          "min_delta_bpm": DEVIATION_MIN_DELTA_BPM, "min_minutes": DEVIATION_MIN_MIN}))
    return out


def _trend_anomaly(start, end, mean_hr: float, base: float, stats: dict) -> Anomaly:
    """일 단위 추세 Anomaly 생성 (start=첫날 00:00, end=마지막 날 23:59, severity=warning)."""
    return _new("elevated_resting_trend", start, end, "warning", mean_hr, base, "rest", stats)


def _trend_rule(s: pd.Series, window_start: pd.Timestamp) -> list[Anomaly]:
    """규칙: 안정시 심박이 직전 7일 중앙값 +10 bpm 이상인 날이 2일 이상 연속."""
    elevated, ref = {}, {}
    for d, v in s.items():
        # 이미 '상승'으로 판정된 날은 기준에서 뺀다 — 안 그러면 첫 상승일이 기준을 끌어올려 둘째 날을 놓친다.
        prior = s[(s.index >= d - pd.Timedelta(days=TREND_LOOKBACK_DAYS)) & (s.index < d)]
        prior = prior[[not elevated.get(x, False) for x in prior.index]]
        if len(prior) >= TREND_MIN_PRIOR_DAYS:
            ref[d] = float(prior.median())
            elevated[d] = v - ref[d] >= TREND_DELTA_BPM
    days = sorted(d for d, ok in elevated.items() if ok)
    out, run = [], []
    for d in days + [None]:
        if run and (d is None or d - run[-1] != pd.Timedelta(days=1)):
            if len(run) >= TREND_MIN_DAYS and run[-1] + pd.Timedelta(days=1) > window_start:
                start, end = run[0], run[-1] + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
                out.append(_trend_anomaly(start, end, float(np.mean([s[x] for x in run])),
                                          float(np.mean([ref[x] for x in run])),
                                          {"method": "rule", "days": len(run), "delta_threshold_bpm": TREND_DELTA_BPM,
                                           "lookback_days": TREND_LOOKBACK_DAYS}))
            run = []
        if d is not None:
            run.append(d)
    return out


def _trend_cusum(s: pd.Series, window_start: pd.Timestamp, cfg: DetectorConfig) -> list[Anomaly]:
    """CuSum(research.cusum_episodes): 작은 상승이라도 며칠 누적되면 알린다."""
    out = []
    for ep in research.cusum_episodes(s, cfg.cusum_k, cfg.cusum_h):
        if ep["end"] + pd.Timedelta(days=1) <= window_start:
            continue
        vals = s[(s.index >= ep["start"]) & (s.index <= ep["end"])]
        out.append(_trend_anomaly(ep["start"], ep["end"] + pd.Timedelta(days=1) - pd.Timedelta(minutes=1),
                                  float(vals.mean()), ep["mu"],
                                  {"method": "cusum", "days": len(vals), "cusum_peak_sigma": round(float(ep["peak"]), 1),
                                   "cusum_k": cfg.cusum_k, "cusum_h": cfg.cusum_h}))
    return out


def _trend(df: pd.DataFrame, baseline_df: pd.DataFrame | None, window_start: pd.Timestamp,
           cfg: DetectorConfig) -> list[Anomaly]:
    """일별 안정시 심박 추세 이상. cfg.trend 에 따라 규칙, CuSum, 합집합(겹치면 병합)을 쓴다.

    Args:
        df: 조회 구간 데이터. baseline_df: 과거 이력(있으면 합쳐서 일별 값을 계산).
        window_start: 조회 구간 시작. 이 시각 이전에 끝난 에피소드는 보고하지 않는다.
        cfg: 탐지 구성.
    """
    full = df if baseline_df is None else pd.concat([baseline_df, df]).drop_duplicates("ts", keep="last")
    daily = pd.DataFrame(daily_resting_hr(full))
    if daily.empty:
        return []
    daily["date"] = pd.to_datetime(daily["date"])
    s = daily.dropna(subset=["resting_hr"]).set_index("date")["resting_hr"]
    found = _trend_rule(s, window_start) if cfg.trend in ("rule", "both") else []
    if cfg.trend in ("cusum", "both"):
        found = _merge(found + _trend_cusum(s, window_start, cfg))
    return found


def _hourly(g: pd.DataFrame, window_start: pd.Timestamp, cfg: DetectorConfig) -> list[Anomaly]:
    """연구 방법: 시간 단위 안정시 심박(RHR-AD)/HROS-AD 의 EllipticEnvelope 이상을 personal_deviation 으로 보고."""
    feat = research.hourly_features(g)
    if feat.empty:
        return []
    flag = research.hourly_anomaly_hours(feat, window_start.floor("h"), cfg.hourly, cfg.contamination, cfg.min_delta_bpm)
    out = []
    for start, end in research.runs_of_hours(flag, cfg.min_hours):
        seg = g.loc[start:end]
        hrs = feat.loc[start.floor("h"):end.floor("h"), "rhr"]
        ctx = seg["ctx"].mode().iat[0]
        med = float(feat.loc[(feat["asleep"] >= 0.5) == (ctx == "sleep"), "rhr"].median())
        mean = float(hrs.mean()) if hrs.notna().any() else float(seg["hr"].mean())
        out.append(_new("personal_deviation", start, end, "warning", mean, med, ctx,
                        {"method": f"hourly_{cfg.hourly}", "hours": int((end - start) / pd.Timedelta(hours=1)) + 1,
                         "contamination": cfg.contamination, "min_delta_bpm": cfg.min_delta_bpm}))
    return out


def _merge(items: list[Anomaly], pad: int = 0) -> list[Anomaly]:
    """같은 유형끼리 겹치는(±pad분) 구간을 하나로 합친다.

    합친 구간은 심각도는 높은 쪽, 대표값은 길이 가중 평균, 부가 stats 와 기준은 더 높은 심각도(같으면 먼저 나온) 쪽을 따르고,
    근거 문장은 합친 값으로 다시 만든다 (지속시간 등이 낡지 않게).
    """
    rank = {"info": 0, "warning": 1, "critical": 2}
    out: list[Anomaly] = []
    for a in sorted(items, key=lambda x: (x.type, x.start)):
        last = out[-1] if out else None
        if not (last and last.type == a.type and a.start <= last.end + pd.Timedelta(minutes=pad)):
            out.append(a)
            continue
        n_last, n_new = last.duration_min, a.duration_min
        if rank[a.severity] > rank[last.severity]:
            last.severity, last.stats, last.baseline, last.context = a.severity, a.stats, a.baseline, a.context
        last.value = round((last.value * n_last + a.value * n_new) / (n_last + n_new), 1)
        last.end = max(last.end, a.end)
        last.duration_min = int(_span_minutes(last.start, last.end))
        last.stats = {**last.stats, "merged": True}
        last.evidence = _evidence(last.type, last.context, last.value, last.baseline, last.duration_min, last.stats)
    return out


def _sensor_gap(g: pd.DataFrame) -> list[Anomaly]:
    """심박이 GAP_MIN_MIN~GAP_MAX_MIN 분 끊겼고 그 사이 걸음이 기록된(착용 증거) 구간 (info)."""
    idx = np.flatnonzero(g["hr"].notna().to_numpy())
    if idx.size < 2:
        return []
    steps = g["steps"].to_numpy()
    out = []
    for k in np.flatnonzero((np.diff(idx) - 1 >= GAP_MIN_MIN) & (np.diff(idx) - 1 <= GAP_MAX_MIN)):
        s, e = idx[k] + 1, idx[k + 1] - 1
        walked = int(steps[s:e + 1].sum())
        if walked == 0:        # 걸음도 없으면 미착용·충전일 가능성이 커서 센서 문제로 보지 않는다
            continue
        n = e - s + 1
        out.append(_make(g, s, e, "sensor_gap", "info", n, None, "rest",
                         {"gap_minutes": n, "steps_in_gap": walked, "min_minutes": GAP_MIN_MIN, "max_minutes": GAP_MAX_MIN}))
    return out


def _overlaps(a: Anomaly, others: list[Anomaly], pad: int = 0) -> bool:
    """a 가 others 중 하나와 pad 분 여유를 두고 겹치는지."""
    p = pd.Timedelta(minutes=pad)
    return any(a.start <= o.end + p and o.start <= a.end + p for o in others)


def detect(df: pd.DataFrame, baseline_df: pd.DataFrame | None = None,
           config: DetectorConfig | None = None) -> list[Anomaly]:
    """df 에서 이상 구간을 찾아 시간순으로 반환한다.

    Args:
        df: VitalsStore 형식의 조회 구간 데이터 (탐지 대상은 이 구간뿐).
        baseline_df: 개인 기준선·일별 추세·시간 단위 모델 적합에 쓸 과거 이력. 생략하면 df 자체를 쓴다.
            df 를 포함하는 상위 집합이어도 된다 (ts 중복은 df 쪽을 우선).
        config: 탐지 구성. 생략하면 DEFAULT_CONFIG.

    Returns:
        (start, type) 순 정렬된 Anomaly 목록. 같은 유형의 겹치는 구간은 이미 하나로 병합되어 있고,
        개인 이탈이 빈맥/서맥과 겹치면(+-5분) 버려진다. df 가 비면 빈 목록.
        조회 구간 시작 직후 14분은 이전 걸음을 알 수 없어 보수적으로 활동 처리되므로(오탐 방지) 그 안의 이벤트는 놓칠 수 있다.
    """
    if df.empty:
        return []
    cfg = config or DEFAULT_CONFIG
    g = minute_grid(df)
    b = fit_baseline(df if baseline_df is None else baseline_df)
    absolute = _merge(_tachycardia(g, b) + _extreme_tachycardia(g, b)) + _bradycardia(g, b)
    deviation = _personal_deviation(g, b) if cfg.personal_rule else []
    if cfg.hourly != "off":
        gf = g if baseline_df is None else minute_grid(pd.concat([baseline_df, df]).drop_duplicates("ts", keep="last"))
        hourly = _hourly(gf, g.index[0], cfg)
        deviation = _merge([x for x in deviation + hourly if x.end >= g.index[0]], OVERLAP_PAD_MIN)
    deviation = [a for a in deviation if not _overlaps(a, absolute, OVERLAP_PAD_MIN)]
    found = absolute + deviation + _trend(df, baseline_df, g.index[0], cfg) + _sensor_gap(g)
    return sorted(found, key=lambda a: (a.start, a.type))


def summarize(df: pd.DataFrame, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> dict:
    """조회 구간의 요약 통계를 계산한다.

    Args:
        df: VitalsStore 형식 데이터.
        start, end: 조회 구간. 주면 coverage_pct 의 분모를 이 구간 길이로 쓴다 (기본: df 의 첫~마지막 분).
            조회 구간 앞쪽에 기록이 없는 사용자는 coverage_pct 가 낮게 나와 에이전트가 기간 확대를 판단할 수 있다.

    Returns:
        coverage_pct(%, 0~100), hr_mean/hr_p05/hr_p95(bpm), resting_hr(안정시 심박 bpm), sleep_hr(수면 중 중앙값 bpm),
        active_minutes, rest_minutes(활동이 아닌 분 중 심박 있는 분 — 요약 신뢰도 판단용), n_minutes(심박 있는 분).
        심박이 없으면 값은 None, 개수는 0.
    """
    g = minute_grid(df)
    if g.empty:
        return {"coverage_pct": 0.0, "hr_mean": None, "hr_p05": None, "hr_p95": None, "resting_hr": None,
                "sleep_hr": None, "active_minutes": 0, "rest_minutes": 0, "n_minutes": 0}
    hr = g["hr"].dropna()
    lo, hi = start or g.index[0], end or g.index[-1]
    span = max((hi - lo) / pd.Timedelta(minutes=1), 1)
    r = lambda x: None if x is None or pd.isna(x) else round(float(x), 1)   # noqa: E731
    sleep = g.loc[g["ctx"] == "sleep", "hr"].dropna()
    return {
        "coverage_pct": round(min(100.0, 100 * len(hr) / span), 1),
        "hr_mean": r(hr.mean()), "hr_p05": r(hr.quantile(0.05)), "hr_p95": r(hr.quantile(0.95)),
        "resting_hr": r(resting_hr_of(g)), "sleep_hr": r(sleep.median()) if len(sleep) else None,
        "active_minutes": int((g["ctx"] == "active").sum()),
        "rest_minutes": int(((g["ctx"] != "active") & g["hr"].notna()).sum()),
        "n_minutes": int(len(hr)),
    }


__all__ = ["Anomaly", "DetectorConfig", "DEFAULT_CONFIG", "detect", "summarize", "daily_resting_hr"]

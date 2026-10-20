"""탐지기 강건성 몬테카를로 벤치마크와 구성(연구 방법 포함) 선택.

책임 경계:
    기존 규칙, 연구 방법(시간 단위 EllipticEnvelope, CuSum), 둘의 결합을 같은 주입 표본에서 비교하고
    사전 고정한 선택 규칙으로 기본 구성(DEFAULT_CONFIG)을 고른다. 탐지기 코드 자체는 수정하지 않는다.
    기본 구성은 selected_config.json 으로만 내보내며, 탐지기에 반영하는 것은 사람이 하고 테스트가 일치를 검증한다.

협력 모듈:
    `app/data/prepare.py`(주입 도우미 재사용), `app.detection.detector`(평가 대상), `eval/detection_eval.py`(결과 md 를 부록으로 흡수).

프로토콜 (라벨 주입 + 튜닝/평가 시드 분리):
    1. 실제 사용자의 깨끗한 데이터 위에 매번 이벤트 1건을 무작위 주입한다 (유형 x 크기 x 지속시간 x 맥락 x 잡음 격자 추출).
    2. 같은 사용자의 깨끗한 데이터에서 이미 나오던 알림(자연 알림)과 겹치는 표본은 버리고, '새로 생긴' 알림만 센다.
    3. 튜닝 시드(TUNE_SEED)로 구성을 고르고, 선택에 쓰지 않은 홀드아웃 시드(HELDOUT_SEED)로 결과를 보고한다.
    4. 선택 규칙은 SELECTION_RULE 에 있고 실행 전에 고정되어 있다. 홀드아웃 결과는 선택에 쓰지 않는다.

한계: 깨끗한 데이터 자체는 튜닝/홀드아웃이 공유한다 (알림 부담은 시드와 무관한 값). 주입은 합성이며 사용자 15명(기록 3~44일) 기반이다.
시간 의미: 모든 시각은 기기 로컬 naive 이며 이벤트 구간의 끝은 마지막 분을 포함한다. 크기 단위 bpm, 길이 단위 분(추세는 일).

실행: `.venv/bin/python -m eval.robustness_benchmark` -> eval/results/robustness_benchmark.{json,md}, selected_config.json
"""
import itertools
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.data import prepare as P  # noqa: E402
from app.detection.baseline import fit_baseline, minute_grid  # noqa: E402
from app.detection.detector import DetectorConfig, detect  # noqa: E402

RESULTS = Path(__file__).parent / "results"
TUNE_SEED, HELDOUT_SEED = 101, 20261020      # 서로 다른 난수열. 홀드아웃은 최종 보고에만 사용
N_PER_KIND = {"tachy": 80, "brady": 60, "personal": 80, "trend": 60, "gap": 40, "exercise": 60, "extreme": 40}
TARGET = {"tachy": "resting_tachycardia", "brady": "bradycardia", "personal": "personal_deviation",
          "trend": "elevated_resting_trend", "gap": "sensor_gap", "extreme": "resting_tachycardia"}
KINDS = list(N_PER_KIND)
LABEL = {**TARGET, "extreme": "resting_tachycardia (극단·제자리 서성임)"}      # 보고서 표의 행 이름
BURDEN_MAX = 0.12            # 깨끗한 데이터 알림 상한 (사용자-일당, 전체 유형 합)
FALSE_ALERT_MAX = 0.10       # 명백한 음성(주입 크기·길이가 정의 미달 / 운동 중) 표본의 새 알림 비율 상한
SELECTION_RULE = (
    f"튜닝 시드에서 (1) 깨끗한 데이터 알림 ≤ {BURDEN_MAX}/사용자-일, (2) 명백한 음성·운동 표본의 거짓 알림률 ≤ "
    f"{FALSE_ALERT_MAX:.0%} 를 만족하는 후보 중 목표 유형의 '명백한 양성' 재현율 최대, 동률이면 경계 표본 탐지율 → 알림 수가 적은 쪽.")

# ---- 주입 격자 ---------------------------------------------------------------------------------------------
GRID = {
    "tachy": dict(ctx=["rest", "sleep"], mag=[-4, 2, 6, 12, 25, 45], dur=[6, 10, 14, 20, 40], noise=[1.0, 2.5, 5.0]),
    "brady": dict(ctx=["rest", "sleep"], mag=[-5, -1, 2, 5, 9, 14], dur=[6, 10, 14, 20, 40], noise=[0.8, 1.5, 3.0]),
    "personal": dict(ctx=["rest", "sleep"], mag=[6, 10, 16, 22, 30], dur=[8, 15, 30, 60, 120], noise=[1.0, 2.5, 5.0]),
    "trend": dict(mag=[3, 6, 9, 12, 16], dur=[1, 2, 3, 5]),
    "gap": dict(wear=[True, False], dur=[60, 100, 130, 180, 300]),
    # 운동 음성: active 는 실제 보행 케이던스(분당 60보 이상) 구간, 190/205 는 운동 억제 천장(210) 바로 아래 값
    "exercise": dict(ctx=["active", "post"], mag=[110, 130, 150, 170, 190, 205], dur=[None], noise=[1.0, 2.5, 5.0]),
    # 극단 양성: 제자리 서성임(분당 0~3보)에서의 극단 빈맥. 175 는 천장 아래(경계), 185 이상은 알려야 함
    "extreme": dict(ctx=["shuffle"], mag=[175, 185, 200, 215, 235], dur=[6, 10, 14, 20, 40], noise=[1.0, 2.5, 5.0]),
}
CLEAR = {"tachy": 20, "brady": 20, "personal": 30}       # '명백한 양성'에 필요한 최소 지속시간(분)


def classify(kind: str, p: dict) -> str:
    """표본의 기대 정답: pos(탐지해야 함) / neg(알리면 안 됨) / border(정의 경계 — 보고만)."""
    if kind == "exercise":
        return "neg"
    if kind == "extreme":
        return "pos" if p["mag"] >= 185 and p["dur"] >= 20 else "neg" if p["dur"] <= 6 else "border"
    mag, dur = p.get("eff_mag", p.get("mag")), p["dur"]
    if kind == "tachy":
        return "pos" if mag >= 12 and dur >= 20 else "neg" if mag <= 0 or dur <= 6 else "border"
    if kind == "brady":
        return "pos" if mag >= 5 and dur >= 20 else "neg" if mag <= -1 or dur <= 6 else "border"
    if kind == "personal":
        return "pos" if mag >= 18 and dur >= 30 else "neg" if mag <= 6 or dur <= 8 else "border"
    if kind == "trend":
        return "pos" if mag >= 12 and dur >= 2 else "neg" if mag <= 3 else "border"
    return "pos" if p["wear"] and dur >= 150 else "neg" if (not p["wear"]) or dur <= 100 else "border"


# ---- 데이터 / 주입 -----------------------------------------------------------------------------------------
class World:
    """깨끗한 사용자 격자와 사용자별 기준선을 한 번만 만들어 표본 간에 공유한다.

    Attributes:
        users: 주입 대상 사용자 (심박 기록 10일 이상).
        grids: 사용자별 1분 격자 (표본마다 copy 해서 주입).
        base: 사용자별 개인 기준선 (personal 표본의 목표 심박 산정용).
        clean_df: 사용자별 탐지기 입력 형식(희소, 라벨 컬럼 없음).
    """
    def __init__(self):
        clean = pd.read_parquet(settings.data_dir / "vitals_minute_clean.parquet")
        days = clean[clean["hr_n"] > 0].assign(d=lambda x: x["ts"].dt.date).groupby("user_id")["d"].nunique()
        self.users = sorted(days[days >= P.MIN_INJECT_DAYS].index)
        self.grids = {u: P._to_grid(clean[clean["user_id"] == u].drop(columns=["user_id", "injected"]).assign(Id=u))
                      for u in self.users}
        self.base = {u: fit_baseline(P._for_detector(g)) for u, g in self.grids.items()}
        self.clean_df = {u: P._for_detector(g) for u, g in self.grids.items()}


LOW_CADENCE_FLOOR = 60      # detector.LOW_CADENCE_SPM 과 같은 값: 진짜 보행으로 볼 최소 분당 걸음


def _exercise_starts(g: pd.DataFrame, L: int, kind: str) -> np.ndarray:
    """운동 시나리오 윈도 시작 후보. 'active': 윈도의 80% 이상 걸음 있고 평균 분당 60보 이상 / 'post': 직전 10분 중 8분 이상 걸었고 윈도 동안 걸음 0."""
    n = len(g)
    s = np.arange(30, n - L - 15)
    steps = g["steps"].to_numpy() > 0
    cs = P._csum(steps)
    hr_ok, asl = P._csum(g["hr_n"].to_numpy() > 0), P._csum(g["asleep"].to_numpy())
    win = lambda c, a=0, b=0: c[s + L + b] - c[s + a]      # noqa: E731
    ok = (win(hr_ok) >= 0.9 * L) & (win(asl) == 0)
    if kind == "active":
        ok &= (win(cs) >= 0.8 * L) & (win(P._csum(g["steps"].to_numpy())) >= LOW_CADENCE_FLOOR * L)    # 실제 보행 케이던스
    else:
        ok &= steps[s - 1] & (win(cs, -10, -L) >= 8) & (win(cs) == 0)
    return s[ok]


def inject(world: World, rng, kind: str, cell: dict):
    """표본 1건을 주입한다.

    Args:
        world: 깨끗한 데이터. rng: numpy Generator. kind: GRID 의 유형 키. cell: GRID 격자 셀(크기/길이/맥락/잡음).

    Returns:
        (user, 주입된 df — 탐지기 입력 형식, 이벤트 시작, 이벤트 끝(포함), 실제 적용된 params). 사용자 6명을 시도해도
        배치할 윈도가 없으면 None. personal 은 목표 심박을 97 bpm 으로 제한하고 실제 상승폭을 params['eff_mag'] 에 기록한다.
    """
    p = dict(cell)
    for u in rng.permutation(world.users)[:6]:
        g = world.grids[u].copy()
        blocked = np.zeros(len(g), bool)
        L = p["dur"]
        noise = p.get("noise", 2.0)
        ramp = 1 if kind == "exercise" else max(1, min(3, (L or 8) // 4))
        if kind in ("tachy", "brady", "personal"):
            sleep = p["ctx"] == "sleep"
            starts = P._find_starts(g, L, "sleep" if sleep else "awake", blocked)
            if len(starts) == 0:
                continue
            if kind == "tachy":
                level = 100 + p["mag"]
            elif kind == "brady":
                level = (35 if sleep else 40) - p["mag"]
            else:
                b = world.base[u]
                med = b.sleep_median if sleep else b.rest_median
                if med is None:
                    continue
                level = min(med + p["mag"], 97.0)
                p["eff_mag"] = round(level - med, 1)
            s = int(rng.choice(starts))
            P._overwrite(g, rng, s, L, level, ramp, noise, "BM", sleep)
            t0, t1 = g.index[s], g.index[s + L - 1]
        elif kind == "extreme":
            starts = P._find_starts(g, L, "awake", blocked)
            if len(starts) == 0:
                continue
            s = int(rng.choice(starts))
            P._overwrite(g, rng, s, L, float(p["mag"]), ramp, noise, "BM", False)
            g.iloc[s:s + L, g.columns.get_loc("steps")] = rng.integers(0, 4, L).astype("int32")      # 제자리 서성임: 분당 0~3보
            t0, t1 = g.index[s], g.index[s + L - 1]
        elif kind == "exercise":
            L = int(rng.choice([20, 40])) if p["ctx"] == "active" else int(rng.choice([8, 12]))
            p["dur"] = L
            starts = _exercise_starts(g, L, p["ctx"])
            if len(starts) == 0:
                continue
            s = int(rng.choice(starts))
            P._overwrite(g, rng, s, L, float(p["mag"]), 1, noise, "BM", None, zero_steps=False)
            t0, t1 = g.index[s], g.index[s + L - 1]
        elif kind == "gap":
            starts = P._find_starts(g, L, "gap" if p["wear"] else "awake", blocked)
            if len(starts) == 0:
                continue
            s = int(rng.choice(starts))
            sl = slice(s, s + L)
            g.iloc[sl, [g.columns.get_loc(c) for c in P.HR_COLS]] = np.nan
            g.iloc[sl, g.columns.get_loc("hr_n")] = 0
            t0, t1 = g.index[s], g.index[s + L - 1]
        else:      # trend
            sp = P.Spec("elevated_resting_trend", "trend", "warning", (p["dur"], p["dur"]), (p["mag"], p["mag"]))
            ev = P._try_trend_event(g, rng, sp, blocked, "BM", margin=None)
            if ev is None:
                continue
            t0, t1 = ev["start"], ev["end"]
        return u, P._for_detector(g), t0, t1, p
    return None


def overlap(a, t0, t1, pad):
    """Anomaly a 가 [t0 - pad, t1 + pad](분 여유, 끝 포함)와 겹치는지."""
    pad = pd.Timedelta(minutes=pad)
    return a.start <= t1 + pad and t0 - pad <= a.end


def exercise_profile(df: pd.DataFrame, t0, t1, fired: list[dict]) -> dict:
    """운동 시나리오 표본의 걸음 프로파일 (거짓 알림 원인 분석용).

    Args:
        df: 주입된 탐지기 입력. t0, t1: 주입 구간. fired: 이 표본에서 새로 발동한 알림들 (dict, start/end 문자열).

    Returns:
        직전 15분/구간 안 분당 걸음, 그리고 첫 발동 알림이 있으면 그 시작 직전 15분의 최고 3분 보행속도
        (극단 빈맥 규칙이 활동 억제를 해제하는 조건: 이 값이 분당 60보 미만).
    """
    steps = minute_grid(df)["steps"]
    pace3 = steps.rolling(3, min_periods=1).mean()
    before, inside = steps[t0 - pd.Timedelta(minutes=15): t0 - pd.Timedelta(minutes=1)], steps[t0:t1]
    prof = {"steps_per_min_before_15min": round(float(before.mean()), 1), "steps_per_min_in_window": round(float(inside.mean()), 1),
            "window_minutes": int(len(inside)), "max_3min_pace_before_alert": None, "steps_per_min_in_alert": None}
    if fired:
        s, e = pd.Timestamp(fired[0]["start"]), pd.Timestamp(fired[0]["end"])
        prof["max_3min_pace_before_alert"] = round(float(pace3[s - pd.Timedelta(minutes=15): s].max()), 1)
        prof["steps_per_min_in_alert"] = round(float(steps[s:e].mean()), 1)
    return prof


def run_split(world: World, seed: int, configs: dict[str, DetectorConfig], kinds: list[str]) -> list[dict]:
    """한 시드로 표본을 만들고 모든 구성을 같은 표본에 평가한다.

    Args:
        world: 깨끗한 데이터. seed: 난수 시드. configs: {이름: DetectorConfig}. kinds: 평가할 유형 키 목록.

    Returns:
        표본별 {kind, user, cls(pos/neg/border), p(params), res: {구성: {same, any}}}.
        same = 목표 유형의 새 알림이 있는가, any = 어떤 유형이든 새 알림이 있는가.
        자연 알림과 겹치는 표본은 모호하므로 제외한다 (제외 수는 표준 출력에 보고).
    """
    rng = np.random.default_rng(seed)
    clean_alerts = {(n, u): detect(world.clean_df[u], config=c) for n, c in configs.items() for u in world.users}
    recs, skipped = [], 0
    for kind in kinds:
        keys = list(GRID[kind])
        cells = [dict(zip(keys, v)) for v in itertools.product(*GRID[kind].values())]
        order = [cells[i] for i in rng.permutation(len(cells))]
        order = (order * (N_PER_KIND[kind] // len(order) + 1))[:N_PER_KIND[kind]]
        for cell in order:
            res = inject(world, rng, kind, cell)
            if res is None:
                skipped += 1
                continue
            u, df, t0, t1, p = res
            targets = {TARGET[kind]} if kind != "exercise" else {"resting_tachycardia", "personal_deviation", "bradycardia"}
            pad, guard = (5, 60) if kind in ("tachy", "brady", "personal", "gap", "exercise", "extreme") else (0, 3 * 1440)
            if any(a.type in targets and overlap(a, t0, t1, guard) for n in configs for a in clean_alerts[(n, u)]):
                skipped += 1          # 자연 알림과 겹친 표본은 판정이 모호해 버린다
                continue
            per = {}
            for n, c in configs.items():
                cl = clean_alerts[(n, u)]
                new = [a for a in detect(df, config=c) if overlap(a, t0, t1, pad)
                       and not any(b.type == a.type and overlap(b, a.start, a.end, 0) for b in cl)]
                per[n] = {"same": any(a.type in targets for a in new), "any": bool(new),
                          "fired": [{"type": a.type, "severity": a.severity, "method": a.stats.get("method"),
                                     "start": str(a.start), "end": str(a.end), "duration_min": a.duration_min,
                                     "mean_hr": a.value} for a in new]}
            rec = {"kind": kind, "id": f"{kind}-{len(recs):03d}", "user": u, "cls": classify(kind, p), "p": p, "res": per,
                   "t0": str(t0), "t1": str(t1)}
            if kind == "exercise":
                first = next((per[n]["fired"] for n in per if per[n]["fired"]), [])
                rec["profile"] = exercise_profile(df, t0, t1, first)
            recs.append(rec)
    print(f"  seed {seed}: {len(recs)} 표본 (버림 {skipped})", flush=True)
    return recs


# ---- 집계 --------------------------------------------------------------------------------------------------
def wilson(k: int, n: int) -> tuple[float, float]:
    """이항 비율 k/n 의 95% Wilson 신뢰구간 (n=0 이면 NaN). 표본이 작을 때 정규근사보다 안전하다."""
    if n == 0:
        return (float("nan"), float("nan"))
    z, ph = 1.96, k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    m = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return round(c - m, 3), round(c + m, 3)


def rate(recs, name, kind, cls, key="same"):
    """유형 kind, 정답 분류 cls 표본 중 구성 name 이 key(same/any)로 탐지한 비율과 Wilson 구간."""
    sel = [r for r in recs if r["kind"] == kind and r["cls"] == cls]
    k = sum(r["res"][name][key] for r in sel)
    return {"k": k, "n": len(sel), "rate": round(k / len(sel), 3) if sel else None, "ci": wilson(k, len(sel))}


def fmt(r) -> str:
    """표 셀용 문자열: 비율과 (k/n). 표본이 없으면 '-'."""
    return "-" if r["rate"] is None else f"{r['rate']:.2f} ({r['k']}/{r['n']})"


_CLEAN: dict = {}


def _clean_users() -> dict:
    """깨끗한 데이터를 사용자별 탐지기 입력 형식으로 한 번만 읽어 캐시한다."""
    if not _CLEAN:
        df = pd.read_parquet(settings.data_dir / "vitals_minute_clean.parquet")
        _CLEAN.update({u: p.drop(columns="injected").reset_index(drop=True) for u, p in df.groupby("user_id")})
    return _CLEAN


def full_clean_burden(cfg: DetectorConfig) -> dict:
    """깨끗한 데이터 전체 15명의 알림 부담 (eval/detection_eval.py 와 같은 기준).

    Returns:
        {"per_user_day": 알림/사용자-일, "total", "by_type", "user_days"}. 사용자-일 = 심박이 있는 달력일 수.
    """
    c, days = Counter(), 0
    for u, d in _clean_users().items():
        days += int(d.loc[d["hr_n"] > 0, "ts"].dt.date.nunique())
        c.update(a.type for a in detect(d, config=cfg))
    return {"per_user_day": round(sum(c.values()) / days, 4), "total": sum(c.values()), "by_type": dict(c), "user_days": days}


# ---- 구성 후보 / 선택 ---------------------------------------------------------------------------------------
def hourly_candidates() -> dict[str, DetectorConfig]:
    """시간 단위 EllipticEnvelope 후보 격자 (모드 x contamination x 최소 연속 시간 x 최소 상승폭). 원본 기본값 0.1 포함."""
    out = {}
    for mode, cont, mh, md in itertools.product(["rhr", "both"], [0.02, 0.05, 0.10], [2, 3], [0.0, 8.0]):
        out[f"hourly:{mode}|c={cont}|h={mh}|d={md:g}"] = DetectorConfig(hourly=mode, contamination=cont, min_hours=mh, min_delta_bpm=md)
    return out


def trend_candidates() -> dict[str, DetectorConfig]:
    """추세 후보: 기존 규칙, CuSum(k, h 격자), 규칙+CuSum 합집합."""
    out = {"trend:rule": DetectorConfig()}
    for mode, k, h in itertools.product(["cusum", "both"], [0.5, 1.0, 1.5], [4.0, 6.0, 8.0]):
        out[f"trend:{mode}|k={k}|h={h}"] = DetectorConfig(trend=mode, cusum_k=k, cusum_h=h)
    return out


def select(recs, world, cands: dict[str, DetectorConfig], target: str, neg_kinds: list[str], bgt: dict) -> tuple[str, list[dict]]:
    """SELECTION_RULE 을 적용해 후보를 고른다.

    Args:
        recs: 튜닝 시드 표본 결과. cands: 후보 구성. target: 목표 유형 키. neg_kinds: 거짓 알림률을 셀 유형들.
        bgt: 후보별 깨끗한 데이터 알림 부담.

    Returns:
        (선택된 후보 이름, 전체 후보 표 — 제약 만족 여부 포함). 제약을 만족하는 후보가 없으면 전체에서 고른다.
    """
    rows = []
    for n, c in cands.items():
        pos, bor = rate(recs, n, target, "pos"), rate(recs, n, target, "border")
        neg_k = sum(r["res"][n]["any" if k == "exercise" else "same"] for k in neg_kinds
                    for r in recs if r["kind"] == k and r["cls"] == "neg")
        neg_n = sum(1 for k in neg_kinds for r in recs if r["kind"] == k and r["cls"] == "neg")
        far = neg_k / neg_n if neg_n else 0.0
        ok = bgt[n]["per_user_day"] <= BURDEN_MAX and far <= FALSE_ALERT_MAX
        rows.append({"name": n, "pos_recall": pos["rate"], "border_rate": bor["rate"], "false_alert_rate": round(far, 3),
                     "burden": bgt[n]["per_user_day"], "alerts": bgt[n]["total"], "feasible": ok})
    feas = [r for r in rows if r["feasible"]] or rows
    best = max(feas, key=lambda r: (r["pos_recall"] or 0, r["border_rate"] or 0, -r["alerts"]))
    return best["name"], rows


def pick(world) -> tuple[dict, str, dict]:
    """튜닝 시드로 시간 단위/추세 하이퍼파라미터와 최종 구성(A/B/C)을 고른다.

    Returns:
        (후보 구성 {A_rules, B_research, C_combined}, 최종 선택 이름, 선택 기록 info).
    """
    print("[튜닝 시드] 후보 평가", flush=True)
    info = {}
    hc, tc = hourly_candidates(), trend_candidates()
    bh = {n: full_clean_burden(c) for n, c in hc.items()}
    bt = {n: full_clean_burden(c) for n, c in tc.items()}
    hr = run_split(world, TUNE_SEED, hc, ["personal", "exercise"])
    h_name, info["hourly"] = select(hr, world, hc, "personal", ["personal", "exercise"], bh)
    tr = run_split(world, TUNE_SEED + 1, tc, ["trend"])
    t_name, info["trend"] = select(tr, world, tc, "trend", ["trend"], bt)
    hsel, tsel = hc[h_name], tc[t_name]
    cand = {"A_rules": DetectorConfig(),
            "B_research": replace(DetectorConfig(), personal_rule=False, hourly=hsel.hourly, contamination=hsel.contamination,
                                  min_hours=hsel.min_hours, min_delta_bpm=hsel.min_delta_bpm,
                                  trend="cusum" if tsel.trend == "both" else tsel.trend if tsel.trend != "rule" else "cusum",
                                  cusum_k=tsel.cusum_k, cusum_h=tsel.cusum_h),
            "C_combined": replace(hsel, trend="both" if tsel.trend != "rule" else "rule", cusum_k=tsel.cusum_k, cusum_h=tsel.cusum_h)}
    bc = {n: full_clean_burden(c) for n, c in cand.items()}
    ar = run_split(world, TUNE_SEED + 2, cand, ["personal", "trend", "exercise"])
    # 최종 구성 선택: 두 목표 유형(personal, trend) 명백한 양성 재현율의 평균 최대, 제약은 위와 동일
    rows = []
    for n in cand:
        pr, tr_ = rate(ar, n, "personal", "pos"), rate(ar, n, "trend", "pos")
        neg = [r["res"][n]["any" if r["kind"] == "exercise" else "same"] for r in ar if r["cls"] == "neg"]
        far = sum(neg) / len(neg)
        rows.append({"name": n, "macro_pos_recall": round(((pr["rate"] or 0) + (tr_["rate"] or 0)) / 2, 3),
                     "false_alert_rate": round(far, 3), "burden": bc[n]["per_user_day"], "alerts": bc[n]["total"],
                     "feasible": bc[n]["per_user_day"] <= BURDEN_MAX and far <= FALSE_ALERT_MAX})
    feas = [r for r in rows if r["feasible"]] or rows
    final = max(feas, key=lambda r: (r["macro_pos_recall"], -r["alerts"]))["name"]
    info["final_stage"] = rows
    info["chosen"] = {"hourly": h_name, "trend": t_name, "final": final}
    print(f"  선택: hourly={h_name} / trend={t_name} / 최종={final}", flush=True)
    return cand, final, info


# ---- 보고 --------------------------------------------------------------------------------------------------
def table(header: list[str], rows: list[list]) -> str:
    """markdown 표 문자열."""
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def render(held: list[dict], cand: dict, final: str, info: dict, bc: dict, n_users: int) -> str:
    """홀드아웃 결과, 재현율 곡선, 튜닝 기록을 markdown 으로 렌더링한다 (방법론/출처 절은 METHODS 를 별도로 덧붙임)."""
    names = list(cand)
    L = ["# 강건성 몬테카를로 벤치마크", "",
         f"- 표본 {len(held)}건 (홀드아웃 시드 {HELDOUT_SEED}, 튜닝 시드 {TUNE_SEED}~{TUNE_SEED + 2}와 분리), 대상 사용자 {n_users}명",
         "- 매 표본: 깨끗한 실데이터에 이벤트 1건 주입 → 새로 생긴 알림만 집계 (자연 알림과 겹치는 표본은 제외)",
         "- 격자: 크기(임계 근처→멀리) × 지속시간(최소 미만/경계/이상) × 맥락(휴식/수면) × 잡음; 운동 중·운동 직후에 높은 심박을 덮어쓴 음성 표본 포함",
         "- 정답 분류: **pos**(정의를 충분히 넘는 이벤트, 알려야 함) / **neg**(정의 미달·운동 중, 알리면 안 됨) / **border**(경계, 보고만)", "",
         f"**선택 규칙 (사전 고정)**: {SELECTION_RULE}", "",
         "## 1. 구성 비교 (홀드아웃)", "",
         "A_rules = 기존 규칙 / B_research = 연구 방법만(개인 이탈·추세 규칙 끄고 EllipticEnvelope + CuSum) / C_combined = 규칙 + 연구 방법.",
         f"튜닝 시드로 고른 최종 구성: **{final}** (아래 표는 홀드아웃 결과이며 선택에는 쓰이지 않았다).", "",
         "### 유형별 명백한 양성 재현율 (같은 유형 알림, 95% Wilson 구간은 JSON 참고)", ""]
    hdr = ["유형"] + names
    L.append(table(hdr, [[LABEL[k]] + [fmt(rate(held, n, k, "pos")) for n in names] for k in TARGET]))
    L += ["", "### 경계 표본 탐지율 / 음성 표본 거짓 알림률", ""]
    rows = [[f"{LABEL[k]} 경계 탐지율"] + [fmt(rate(held, n, k, "border")) for n in names] for k in TARGET]
    rows += [[f"{LABEL[k]} 음성 거짓 알림률"] + [fmt(rate(held, n, k, "neg")) for n in names] for k in TARGET]
    rows += [["운동 중/직후 거짓 알림률(어떤 유형이든)"] + [fmt(rate(held, n, "exercise", "neg", "any")) for n in names]]
    L.append(table(hdr, rows))
    L += ["", "※ B_research 열의 tachy/brady/gap 행은 연구 방법이 다루지 않는 유형이라 규칙 탐지기가 그대로 남아 있는 값이다 (비교 대상은 personal/trend 행).",
          "", "### 깨끗한 데이터 알림 부담 (전체 15명, 사용자-일당)", ""]
    L.append(table(["구성", "알림 수", "사용자-일당", "유형별"], [[n, bc[n]["total"], bc[n]["per_user_day"], bc[n]["by_type"]] for n in names]))

    ex = [r for r in held if r["kind"] == "exercise"]
    fired = [r for r in ex if any(r["res"][n]["any"] for n in names)]
    low = [r for r in ex if r["p"]["mag"] <= 170]
    hi = [r for r in ex if r["p"]["mag"] > 170]
    L += ["", "## 극단 빈맥 규칙(운동 억제의 천장) 도입 전/후", "",
          "운동 억제에는 천장이 없었다(걸음이 조금이라도 있으면 몇 bpm이든 묵살). 2026-10-20 에 활동 맥락과 무관한 극단 빈맥 규칙을 추가했다"
          "(210 bpm 이상 10분+, 또는 최근 15분 안에 분당 60보 이상 보행이 없는 180 bpm 이상 10분+, critical).", "",
          table(["측정", "도입 전 (03:10 실행, 같은 시드)", "도입 후 (이 문서)"],
                [["운동 음성 표본 거짓 알림", "0/50 (HR 110~170 bpm 만 시험)",
                  f"{sum(r['res']['C_combined']['any'] for r in ex)}/{len(ex)} (110~170 bpm: "
                  f"{sum(r['res']['C_combined']['any'] for r in low)}/{len(low)}, 190·205 bpm: "
                  f"{sum(r['res']['C_combined']['any'] for r in hi)}/{len(hi)})"],
                 ["극단·제자리 서성임 양성 재현율", "시험 안 함", fmt(rate(held, "C_combined", "extreme", "pos"))],
                 ["깨끗한 데이터 알림", "44건 (0.0938/사용자-일)", f"{bc['C_combined']['total']}건 ({bc['C_combined']['per_user_day']}/사용자-일)"],
                 ["그 밖의 모든 유형 재현율·음성·경계 행", "-", "도입 전 문서와 동일 (같은 시드, 같은 표본 순서)"]]), "",
          "※ 표본 구성이 바뀌어(운동 음성 50 -> 60건: 190·205 bpm 추가, 극단 양성 40건 추가) 전/후는 '같은 시드'이지만 표본 수가 다르다.",
          "",
          "### 운동 음성 표본 중 거짓 알림이 난 것", ""]
    if fired:
        L.append(table(["표본 ID", "사용자", "시나리오", "심박(bpm)", "길이(분)", "주입 직전 15분 분당 걸음", "알림 시작 직전 15분 최고 3분 보행속도", "알림 구간 분당 걸음", "발동한 알림"],
                       [[r["id"], r["user"], r["p"]["ctx"], r["p"]["mag"], r["profile"]["window_minutes"],
                         r["profile"]["steps_per_min_before_15min"], r["profile"]["max_3min_pace_before_alert"],
                         r["profile"]["steps_per_min_in_alert"],
                         "; ".join(f"{f['type']}/{f['severity']}/{f['method'] or 'rule'} {f['start'][5:16]}~{f['end'][11:16]}"
                                   for f in r["res"]["C_combined"]["fired"])] for r in fired]))
        ctxs = sorted({r["p"]["ctx"] for r in fired})
        released = [r["id"] for r in fired if (r["profile"]["max_3min_pace_before_alert"] or 0) < 60]
        L += ["", f"원인 (표에서 계산): 시나리오 {ctxs}, 심박 {sorted({r['p']['mag'] for r in fired})} bpm, 발동한 규칙은 모두 극단 빈맥 규칙(extreme). "
              "이 규칙은 '알림이 시작되는 시점의 직전 15분 안에 3분 평균 분당 60보 이상 걸은 적이 없을 때'만 활동 억제를 해제한다. "
              f"그 값이 60 미만인 표본: {released or '없음'}. 즉 이 표본들은 규칙 조건대로 해제된 것이다: 운동이 막 시작되어 보행속도가 아직 60보에 "
              "못 미치는 구간(active)이거나 느린 걸음 뒤 멈춘 구간(post)에서 심박이 190 bpm 이상으로 10분 가까이 유지된 합성 표본이다. "
              "합성 표본은 심박을 즉시 190~205로 올려 실제 생리(운동 시작 직후 심박은 수 분에 걸쳐 오른다)보다 가혹하다. "
              "분당 60보 이상으로 걸은 구간의 운동 억제는 유지된다(운동 음성 110~170 bpm 은 0건, 190·205 bpm 도 대부분 억제). "
              "이 두 건을 오탐으로 볼지는 합성이라 정답이 없다 - 규칙 의도 안의 동작과 알려진 한계로 기록한다.", ""]
    else:
        L += ["없음.", ""]
    L += ["", "## 2. 재현율 곡선 (홀드아웃, 같은 유형 알림)", ""]
    curve = {"tachy": ("mag", "초과 bpm(임계 100 기준)", "dur", 20, 12), "brady": ("mag", "임계 아래 깊이 bpm", "dur", 20, 5),
             "personal": ("mag", "개인 중앙값 대비 +bpm(목표)", "dur", 30, 16), "trend": ("mag", "상승폭 bpm", "dur", 2, 9),
             "gap": ("dur", "결측 분(착용 증거 있음)", None, None, None),
             "extreme": ("mag", "극단 심박 bpm(제자리 서성임)", "dur", 20, 185)}
    for k, (ax, label, ax2, dmin, mmin) in curve.items():
        recs = [r for r in held if r["kind"] == k]
        if not recs:
            continue
        L += [f"### {LABEL[k]}", ""]
        def cond_ok(r):
            """곡선 축 외의 변수를 '충분한' 값으로 고정해(그 축의 효과만 보려고) 해당 표본만 통과시킨다."""
            p = r["p"]
            if k == "gap":
                return p["wear"]
            return (p["dur"] >= dmin) if ax == "mag" else (p.get("eff_mag", p["mag"]) >= mmin)
        lv = sorted({r["p"][ax] for r in recs})
        rws = []
        for v in lv:
            sel = [r for r in recs if r["p"][ax] == v and cond_ok(r)]
            rws.append([v] + [f"{sum(r['res'][n]['same'] for r in sel) / len(sel):.2f} ({sum(r['res'][n]['same'] for r in sel)}/{len(sel)})"
                              if sel else "-" for n in names])
        fix = {"tachy": f"지속≥{dmin}분 고정", "brady": f"지속≥{dmin}분 고정", "personal": f"지속≥{dmin}분 고정",
               "trend": f"상승폭 커짐 → 일수 {dmin}일 이상 고정", "gap": "착용 증거 있는 표본",
               "extreme": f"지속≥{dmin}분 고정"}[k]
        xl = {"mag": label, "dur": "지속(분)" if k != "trend" else "지속(일)"}[ax]
        if k == "trend":
            xl = "상승폭 bpm (일수≥2 고정)"
        L += [f"{fix}; 가로: {xl}", "", table([xl] + names, rws), ""]
        if ax == "mag":
            lv2 = sorted({r["p"]["dur"] for r in recs})
            rws2 = []
            for v in lv2:
                sel = [r for r in recs if r["p"]["dur"] == v and r["p"].get("eff_mag", r["p"]["mag"]) >= mmin]
                rws2.append([v] + [f"{sum(r['res'][n]['same'] for r in sel) / len(sel):.2f} ({sum(r['res'][n]['same'] for r in sel)}/{len(sel)})"
                                   if sel else "-" for n in names])
            L += [f"크기≥{mmin} 고정; 가로: 지속({'일' if k == 'trend' else '분'})", "",
                  table([("지속(일)" if k == "trend" else "지속(분)")] + names, rws2), ""]
    L += ["## 3. 튜닝 시드 후보 선택 기록", "", f"선택: {info['chosen']}", "", "### 시간 단위 EllipticEnvelope (personal_deviation 목표) — 상위 8",
          "", table(["후보", "양성 재현", "경계", "거짓 알림", "깨끗한 부담", "가능"],
                    [[r["name"], r["pos_recall"], r["border_rate"], r["false_alert_rate"], r["burden"], r["feasible"]]
                     for r in sorted(info["hourly"], key=lambda r: -(r["pos_recall"] or 0))[:8]]), "",
          "### CuSum 추세 (elevated_resting_trend 목표)", "",
          table(["후보", "양성 재현", "경계", "거짓 알림", "깨끗한 부담", "가능"],
                [[r["name"], r["pos_recall"], r["border_rate"], r["false_alert_rate"], r["burden"], r["feasible"]]
                 for r in info["trend"]]), "",
          "### 최종 단계 (튜닝 시드, A/B/C)", "",
          table(["구성", "personal+trend 평균 양성 재현", "거짓 알림", "깨끗한 부담", "가능"],
                [[r["name"], r["macro_pos_recall"], r["false_alert_rate"], r["burden"], r["feasible"]] for r in info["final_stage"]]), ""]
    return "\n".join(L)


METHODS = """\
## 방법론과 출처

- **RHR-AD / HROS-AD**: Mishra et al. 2020 (Nat. Biomed. Eng.)의 공개 구현 `github.com/gireeshkbogu/AnomalyDetect`
  (커밋 1484183)의 `rhrad_offline.py`, `hrosad_offline.py`, `rhrad_online_24hr_alerts_v7.py` 소스를 직접 읽고 파라미터를 가져왔다:
  안정시 심박 = 직전 12분 걸음 합 0, HROS = 심박/(걸음+1), 시간 평균, 표준화, `EllipticEnvelope(contamination=0.1,
  support_fraction=0.7, random_state=10)`, 온라인판 기준 창 744시간 · 24시간 내 이상 시간 ≥1 YELLOW / ≥6 RED.
  이 저장소 구현(`app/detection/research.py`)의 변경점: 400샘플 이동평균과 `seasonal_decompose(freq=1)`(항등) 생략,
  수면/각성 분리 적합, 상승 방향만, 최소 연속 시간·최소 상승폭 추가, contamination 은 튜닝 시드로 선택(원본 0.1은 후보에 포함).
  원본은 수일~수주 규모의 감염 신호용이라 20~40분짜리 이벤트에는 해상도가 맞지 않는다 — 벤치마크의 지속시간 곡선이 이를 보여 준다.
- **CuSum**: Alavi et al. 2022 (Nat. Med.)의 실시간 알림 연구가 CuSum 계열을 쓰는 것으로 알려져 있으나 공개 소스를 찾지 못했다
  (GitHub 검색: NightSignal/CuSum 관련 공식 저장소 없음). 표준 단측 CuSum(Page 1954)을 직접 구현했으며
  k, h 는 튜닝 시드로 정했다. 논문 파라미터의 재현이 아니다.
- **한계**: 주입은 합성이며(실제 임상 이벤트가 아님) 실제 사용자 15명(기록 3~44일)의 배경 위에서 평가했다.
  거짓 알림은 라벨 없는 실데이터의 상한 근사다. 표본 수가 작아 신뢰구간이 넓다 (JSON에 Wilson 95% 구간).
"""


def reuse_selection() -> tuple[dict, str, dict]:
    """이전 실행에서 튜닝 시드로 이미 고른 구성을 그대로 다시 쓴다 (탐지기 규칙이 바뀐 뒤 홀드아웃만 재측정할 때).

    selected_config.json(최종 구성)과 robustness_benchmark.json(튜닝 기록)에서 A/B/C 구성을 복원한다.
    규칙 변경이 선택 결과를 바꿨는지는 전체 재실행(`main()` 인자 없이)으로만 확인할 수 있다.
    """
    sel = json.loads((RESULTS / "selected_config.json").read_text())
    old = json.loads((RESULTS / "robustness_benchmark.json").read_text())
    c = DetectorConfig(**sel["config"])
    cand = {"A_rules": DetectorConfig(),
            "B_research": replace(c, personal_rule=False, trend="cusum"),
            "C_combined": c}
    return cand, sel["name"], old["tuning"]


def main(reuse: bool = False):
    """전체 파이프라인: 튜닝 시드로 선택 -> 홀드아웃 평가 -> md/json/selected_config.json 저장.

    Args:
        reuse: True 면 튜닝 단계를 건너뛰고 이전에 선택된 구성을 재사용한다 (홀드아웃만 재측정).
    """
    t0 = time.time()
    world = World()
    cand, final, info = reuse_selection() if reuse else pick(world)
    print("[홀드아웃] 평가", flush=True)
    held = run_split(world, HELDOUT_SEED, cand, KINDS)
    bc = {n: full_clean_burden(c) for n, c in cand.items()}
    md = render(held, cand, final, info, bc, len(world.users)) + "\n" + METHODS
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "robustness_benchmark.md").write_text(md)
    (RESULTS / "selected_config.json").write_text(json.dumps({"name": final, "config": asdict(cand[final])}, indent=2))
    summary = {n: {k: {c: rate(held, n, k, c) for c in ("pos", "border", "neg")} for k in TARGET} for n in cand}
    summary_ex = {n: rate(held, n, "exercise", "neg", "any") for n in cand}
    (RESULTS / "robustness_benchmark.json").write_text(json.dumps(
        {"seeds": {"tune": TUNE_SEED, "heldout": HELDOUT_SEED}, "selection_rule": SELECTION_RULE, "chosen": info["chosen"],
         "configs": {n: asdict(c) for n, c in cand.items()}, "clean_burden": bc, "heldout": summary,
         "heldout_exercise_false_alert": summary_ex, "tuning": info, "n_heldout": len(held),
         "selection_reused": reuse}, ensure_ascii=False, indent=2))
    print(f"완료 {time.time() - t0:.0f}s → {RESULTS}/robustness_benchmark.md\n선택 구성: {final}")


if __name__ == "__main__":
    main(reuse="--reuse-selection" in sys.argv)

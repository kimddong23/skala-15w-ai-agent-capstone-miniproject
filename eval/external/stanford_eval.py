"""외부 검증: 동결된 탐지기(DEFAULT_CONFIG)를 Stanford COVID-19 웨어러블의 실제 질병 사건에 적용한다.

책임 경계:
    `stanford_adapter.py` 가 만든 분 단위 데이터(vitals_minute.parquet)에서 참여자별로 *하루씩 재생(replay)* 하며 탐지기를 돌려 '그날 울린 알람'을 모으고,
    질병 증상 시작일·진단일과의 관계(민감도, 선행 일수)와 평소의 거짓 경보를 계산한다. 탐지기는 수정하지 않는다.

튜닝 금지 원칙 (가장 중요한 설계 결정):
    - 탐지기 설정은 `DEFAULT_CONFIG` 그대로다. 이 데이터로 임계값·후보를 고르지 않았고, FitBit 개발 데이터에서 고정한 값이다.
    - 이 평가를 보고 탐지기를 고치면 이 데이터는 더 이상 외부 테스트셋이 아니다 (고치려면 새 외부 데이터가 필요하다).

재생 방식 (실시간 알림 가정):
    각 날짜 d 에 대해 "d 일 하루치 데이터 + d 일 끝까지의 이력(기준선)"만으로 `detect()` 를 호출한다 - 미래 데이터를 쓰지 않는다.
    알람 유형은 elevated_resting_trend, personal_deviation, resting_tachycardia 세 가지이고, 그 유형이 하나라도 나오면 d 는 '알람일'이다.
    (서맥·센서 결측은 질병 신호로 보지 않아 제외.) 하루 창이라 매일 첫 약 15분은 직전 걸음 정보가 없어 활동으로 처리된다(보수적).
    기록 앞 BURN_IN_DAYS 일은 기준선이 없으므로 평가에서 뺀다. 심박이 하루 MIN_DAY_MINUTES 분 미만인 날도 뺀다.

지표 (논문과의 비교를 위해 같은 모양으로 정의):
    - 사건 민감도: 증상 시작일 기준 [-PRE, +POST] 일 안에 알람일이 있는 사건의 비율 (진단일 기준도 따로).
    - 선행 일수: 그 창의 첫 알람일이 증상일보다 며칠 앞서는가 (양수 = 증상보다 먼저). 중앙값.
    - 거짓 경보: 건강 대조군과, 라벨 참여자의 사건 주변(+-14일, 증상~회복 포함)을 뺀 날들에서 '사람-30일당 알람일 수'.

비교의 한계 (결과 문서에도 적는다): 논문은 참여자마다 오염 비율을 조정한 오프라인 RHR-Diff/HROS-AD 이고 날짜 이동·장치 구성도 다르다.
    여기 수치는 '튜닝 없이 실시간 규칙을 외부 데이터에 처음 적용한 값'이며 논문 수치와 같은 조건이 아니다.

실행: `.venv/bin/python -m eval.external.stanford_eval [--workers 4]` (먼저 stanford_download, stanford_adapter) ->
    eval/results/external_stanford.{md,json} (집계 지표만; 참여자별 분 단위 데이터는 data/external 에만 있고 커밋하지 않는다).
"""
import argparse
import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "data" / "external" / "stanford"
RESULTS = ROOT / "eval" / "results"
TARGET_TYPES = ("elevated_resting_trend", "personal_deviation", "resting_tachycardia")
PRE_DAYS, POST_DAYS = 7, 3            # 증상(또는 진단) 시작일 기준 허용 창
BURN_IN_DAYS = 7                      # 기록 앞쪽 제외 (기준선/추세/시간 모델이 쓸 이력이 없음)
MIN_DAY_MINUTES = 360                 # 하루 최소 심박 분 수 (6시간) - 미만인 날은 평가하지 않는다
FA_MARGIN_DAYS = 14                   # 거짓 경보 집계에서 사건 전후로 제외하는 여백
CLUSTER_GAP_DAYS = 14                 # 이 안의 증상 날짜들은 같은 질병 사건으로 묶는다


# ---------------------------------------------------------------- 재생 (탐지기 실행)
def replay_user(args: tuple[str, str]) -> tuple[str, dict]:
    """참여자 한 명을 하루씩 재생해 {ISO 날짜: [그날 나온 대상 유형]} 과 평가 가능한 날짜 목록을 돌려준다.

    Args:
        args: (user_id, vitals_minute.parquet 경로). 프로세스 풀에서 쓰도록 단일 튜플로 받는다.

    Returns:
        (user_id, {"valid_days": [ISO...], "alarms": {ISO: [유형...]}}). valid_days = 평가 대상 날(번인 이후, 하루 심박 분 수 충분).
    """
    from app.detection.detector import DEFAULT_CONFIG, detect

    uid, path = args
    df = pd.read_parquet(path, filters=[("user_id", "==", uid)]).drop(columns=["user_id", "source"])
    df["asleep"] = df["asleep"].astype("boolean")
    day = df["ts"].dt.normalize()
    hr_minutes = df[df["hr_n"] > 0].groupby(day[df["hr_n"] > 0]).size()
    first = day.min()
    valid, alarms = [], {}
    for d in sorted(hr_minutes.index):
        if d < first + pd.Timedelta(days=BURN_IN_DAYS) or hr_minutes[d] < MIN_DAY_MINUTES:
            continue
        end = d + pd.Timedelta(days=1)
        window = df[(df["ts"] >= d) & (df["ts"] < end)].reset_index(drop=True)
        hist = df[df["ts"] < end].reset_index(drop=True)
        found = {a.type for a in detect(window, baseline_df=hist, config=DEFAULT_CONFIG) if a.type in TARGET_TYPES}
        valid.append(d.date().isoformat())
        if found:
            alarms[d.date().isoformat()] = sorted(found)
    return uid, {"valid_days": valid, "alarms": alarms}


def replay_all(workers: int, cache_dir: Path = OUT / "alarms") -> dict[str, dict]:
    """모든 참여자를 재생한다 (참여자별 결과를 캐시해 중단 후 이어서 실행할 수 있다)."""
    parts = json.loads((OUT / "participants.json").read_text())
    cache_dir.mkdir(exist_ok=True)
    todo = [u for u in parts if not (cache_dir / f"{u}.json").exists()]
    if todo:
        with ProcessPoolExecutor(workers) as ex:
            for uid, res in ex.map(replay_user, [(u, str(OUT / "vitals_minute.parquet")) for u in todo]):
                (cache_dir / f"{uid}.json").write_text(json.dumps(res))
                print(f"  {uid}: 평가일 {len(res['valid_days'])}, 알람일 {len(res['alarms'])}", flush=True)
    return {u: json.loads((cache_dir / f"{u}.json").read_text()) for u in parts}


# ---------------------------------------------------------------- 사건/지표 (순수 함수)
def cluster_onsets(dates: list[date], gap: int = CLUSTER_GAP_DAYS) -> list[date]:
    """증상 날짜들을 같은 질병으로 묶어 각 묶음의 첫 날(= 질병 시작일) 목록을 돌려준다 (gap 일 이내면 같은 묶음)."""
    onsets: list[date] = []
    last: date | None = None
    for d in sorted(dates):
        if last is None or (d - last).days > gap:
            onsets.append(d)
        last = d
    return onsets


def build_events(part: dict) -> list[dict]:
    """참여자 정보(participants.json 항목)에서 평가할 사건을 만든다.

    Args:
        part: {"start","end"(ISO), "events": [{"kind","symptom":[ISO],"diagnosis":[ISO],"recovery":[ISO]}]}.

    Returns:
        [{"kind","onset"(date|None),"diagnosis"(date|None),"exclude"(제외 구간 시작·끝 date)}]. 기록 구간 밖의 날짜는 무시하고,
        증상일이 없고 진단일만 있으면 onset=None(진단 기준으로만 평가). 제외 구간 = [시작-FA_MARGIN, (회복일 또는 시작+14)+FA_MARGIN].
    """
    lo, hi = date.fromisoformat(part["start"][:10]), date.fromisoformat(part["end"][:10])
    inside = lambda iso: lo <= date.fromisoformat(iso) <= hi   # noqa: E731
    out = []
    for e in part["events"]:
        sym = [date.fromisoformat(s) for s in e["symptom"] if inside(s)]
        dx = sorted(date.fromisoformat(s) for s in e["diagnosis"] if inside(s))
        rec = sorted(date.fromisoformat(s) for s in e["recovery"] if inside(s))
        onsets = cluster_onsets(sym) or ([None] if dx else [])
        for on in onsets:
            anchor = on or dx[0]
            dx_for = next((d for d in dx if d >= anchor - timedelta(days=14)), None)
            rec_for = next((d for d in rec if d >= anchor and (d - anchor).days <= 60), None)
            end = max([x for x in (rec_for, dx_for, anchor + timedelta(days=14)) if x])
            out.append({"kind": e["kind"], "onset": on, "diagnosis": dx_for,
                        "exclude": (anchor - timedelta(days=FA_MARGIN_DAYS), end + timedelta(days=FA_MARGIN_DAYS))})
    return out


def window_result(alarm_days: dict[date, set[str]], valid_days: set[date], anchor: date,
                  pre: int = PRE_DAYS, post: int = POST_DAYS, types: tuple[str, ...] = TARGET_TYPES) -> dict:
    """한 사건을 anchor(증상일 또는 진단일) 기준 창 [-pre, +post] 일에서 판정한다.

    Returns:
        {"evaluable": 창 안에 평가 가능한 날이 하나라도 있는가, "hit": 창 안에 types 알람이 있는가,
        "lead_days": 첫 알람일이 anchor 보다 앞선 일수(양수=먼저, 음수=늦음) 또는 None}.
    """
    days = [anchor + timedelta(days=i) for i in range(-pre, post + 1)]
    ok = [d for d in days if d in valid_days]
    fired = [d for d in ok if alarm_days.get(d, set()) & set(types)]
    return {"evaluable": bool(ok), "hit": bool(fired), "lead_days": (anchor - fired[0]).days if fired else None}


def first_alarm(alarm_days: dict[date, set[str]], valid_days: set[date], anchor: date,
                pre: int = PRE_DAYS, post: int = POST_DAYS, types: tuple[str, ...] = TARGET_TYPES) -> dict:
    """anchor(증상 시작일) 기준 창 [-pre, +post] 일에서 가장 이른 알람일과 그날의 유형을 돌려준다 (`window_result` 와 같은 창·같은 규칙).

    Returns:
        {"evaluable","hit","first_alarm"(ISO 날짜 | None),"lead_days"(앞서면 양수 | None),"types"(그날 나온 대상 유형 목록)}.
    """
    days = [anchor + timedelta(days=i) for i in range(-pre, post + 1)]
    ok = [d for d in days if d in valid_days]
    fired = [d for d in ok if alarm_days.get(d, set()) & set(types)]
    if not fired:
        return {"evaluable": bool(ok), "hit": False, "first_alarm": None, "lead_days": None, "types": []}
    d = fired[0]
    return {"evaluable": True, "hit": True, "first_alarm": d.isoformat(), "lead_days": (anchor - d).days, "types": sorted(alarm_days[d] & set(types))}


def first_warnings(replays: dict[str, dict], parts: dict[str, dict]) -> dict[str, list[dict]]:
    """앱이 사건 옆에 보여 줄 '증상 N일 전 첫 경고'를 참여자의 *전체 기록*에서 사건마다 미리 계산한다 (조회 구간과 무관).

    평가(`evaluate`)와 같은 방법이다: 하루씩 재생한 실시간 알람일 중 증상 시작일 -PRE_DAYS ~ +POST_DAYS 일에서 가장 이른 날.
    조회 구간에 따라 기준선·추세 에피소드가 달라지는 라이브 이상 목록으로 계산하지 않는다.

    Returns:
        {user_id: [{"onset": ISO, "counted_in_eval": 질병 묶음의 첫 증상일인가, "pre_days","post_days", "method", + `first_alarm()` 결과}]}.
        앱에 표시된 증상 시작일(기록 구간 안의 모든 증상 날짜)마다 한 항목이다.
    """
    out: dict[str, list[dict]] = {}
    for uid, part in parts.items():
        if part["group"] != "labeled":
            continue
        rep = replays[uid]
        valid = {date.fromisoformat(d) for d in rep["valid_days"]}
        alarms = {date.fromisoformat(d): set(ts) for d, ts in rep["alarms"].items()}
        lo, hi = part["start"][:10], part["end"][:10]
        counted = {e["onset"] for e in build_events(part) if e["onset"]}
        onsets = sorted({s for e in part["events"] for s in e["symptom"] if lo <= s <= hi})
        out[uid] = [{"onset": s, "counted_in_eval": date.fromisoformat(s) in counted, "pre_days": PRE_DAYS, "post_days": POST_DAYS,
                     "method": "causal_daily_replay_frozen_config", **first_alarm(alarms, valid, date.fromisoformat(s))} for s in onsets]
    return out


def summarize_events(records: list[dict]) -> dict:
    """사건 판정 목록({"evaluable","hit","lead_days"})을 요약한다: n(평가 가능), n_excluded, hits, sensitivity, lead_median(적중 사건 기준)."""
    ev = [r for r in records if r["evaluable"]]
    hits = [r for r in ev if r["hit"]]
    leads = [r["lead_days"] for r in hits]
    return {"n": len(ev), "n_not_evaluable": len(records) - len(ev), "hits": len(hits),
            "sensitivity": round(len(hits) / len(ev), 3) if ev else None,
            "lead_median_days": statistics.median(leads) if leads else None,
            "detected_before_or_on_onset": sum(1 for x in leads if x >= 0)}


def false_alarm_rate(alarm_days: dict[date, set[str]], valid_days: set[date], excluded: list[tuple[date, date]] | None = None,
                     types: tuple[str, ...] = TARGET_TYPES) -> dict:
    """사건 구간(excluded)을 뺀 평가일 중 types 알람이 난 날의 비율을 '사람-30일당 알람일 수'로 환산한다.

    Returns:
        {"days": 평가일 수, "alarm_days": 알람일 수, "per_30_person_days": 알람일/(평가일/30)} (평가일 0 이면 None).
    """
    elig = {d for d in valid_days if not any(a <= d <= b for a, b in (excluded or []))}
    fired = [d for d in elig if alarm_days.get(d, set()) & set(types)]
    return {"days": len(elig), "alarm_days": len(fired), "per_30_person_days": round(len(fired) / (len(elig) / 30), 2) if elig else None}


def chance_hit_probability(p_alarm_day: float, n_days: int) -> float:
    """우연 기준선: 하루 알람 확률이 p 이고 날들이 독립이라고 가정할 때 n 일 창에 알람이 하나라도 있을 확률 = 1 - (1-p)^n.

    알람이 평소에도 자주 울리는 탐지기는 아무 날이나 11일 창을 잡아도 자주 '적중'한다 - 민감도를 이 값과 비교해야 의미가 있다.
    """
    return 1 - (1 - p_alarm_day) ** n_days


# ---------------------------------------------------------------- 논문 기준값 (참고)
def paper_reference(xlsx: Path = OUT / "SuppData1.xlsx") -> dict:
    """논문 SuppTable13 에서 RHR-Diff 의 증상일 기준 검출 시점(일, 음수 = 증상보다 먼저)을 읽어 COVID/기타 질병별로 요약한다.

    값은 숫자(검출 시점), 'miss'(놓침), 비어 있음/'No symptom day'(증상일 없음). 논문의 오프라인(참여자별 튜닝) 결과이며 날짜 창 정의가 다르다.
    """
    raw = pd.read_excel(xlsx, sheet_name="SuppTable13_Fig4abc", header=None)
    sections: dict[str, list] = {"covid": [], "other": []}
    mode = None
    for _, r in raw.iterrows():
        head = str(r[0])
        if head.startswith("For Figure 4a-4b"):
            mode = "covid"
        elif head.startswith("For Figure 4c"):
            mode = "other"
        elif mode and head not in ("ParticipantID", "nan"):
            sections[mode].append(r[2])
    out = {}
    for kind, vals in sections.items():
        nums = [float(v) for v in vals if isinstance(v, (int, float)) and not pd.isna(v)]
        miss = sum(1 for v in vals if str(v).strip().lower() == "miss")
        out[kind] = {"rows": len(vals), "with_symptom_day": len(nums) + miss, "detected": len(nums), "missed": miss,
                     "sensitivity_among_with_symptom_day": round(len(nums) / (len(nums) + miss), 3) if nums or miss else None,
                     "median_delta_days": statistics.median(nums) if nums else None,
                     "detected_at_or_before_onset": sum(1 for v in nums if v <= 0)}
    return out


# ---------------------------------------------------------------- 전체 평가
def evaluate(replays: dict[str, dict], parts: dict[str, dict]) -> dict:
    """모든 참여자의 재생 결과와 사건 정보로 지표 전체를 계산한다."""
    per_event: list[dict] = []
    chance = {"covid": {"events": 0, "expected_hits": 0.0, "hits": 0, "win_days": 0, "win_alarm_days": 0},
              "other": {"events": 0, "expected_hits": 0.0, "hits": 0, "win_days": 0, "win_alarm_days": 0}}
    pooled_rate = {"days": 0, "alarm_days": 0}
    fa_controls = {"days": 0, "alarm_days": 0}
    fa_labeled = {"days": 0, "alarm_days": 0}
    by_type_fa = {t: {"days": 0, "alarm_days": 0} for t in TARGET_TYPES}
    for uid, part in parts.items():
        rep = replays[uid]
        valid = {date.fromisoformat(d) for d in rep["valid_days"]}
        alarms = {date.fromisoformat(d): set(ts) for d, ts in rep["alarms"].items()}
        events = build_events(part) if part["group"] == "labeled" else []
        fa = false_alarm_rate(alarms, valid, [e["exclude"] for e in events])       # 사건 밖 평소 알람 (우연 기준선 p 와 거짓 경보 집계에 같이 쓴다)
        for ev in events:
            row = {"user_id": uid, "kind": ev["kind"], "onset": ev["onset"].isoformat() if ev["onset"] else None,
                   "diagnosis": ev["diagnosis"].isoformat() if ev["diagnosis"] else None}
            for name, types in [("all", TARGET_TYPES)] + [(t, (t,)) for t in TARGET_TYPES]:
                if ev["onset"]:
                    row[f"symptom_{name}"] = window_result(alarms, valid, ev["onset"], types=types)
                if ev["diagnosis"] and ev["kind"] == "covid":
                    row[f"diagnosis_{name}"] = window_result(alarms, valid, ev["diagnosis"], types=types)
            if ev["onset"]:
                p_user = fa["alarm_days"] / fa["days"] if fa["days"] >= 14 else None     # 이 사람의 평소 알람일 비율 (사건 밖)
                win = [ev["onset"] + timedelta(days=i) for i in range(-PRE_DAYS, POST_DAYS + 1)]
                ok = [d for d in win if d in valid]
                c = chance[ev["kind"]]
                row["_p_user"], row["_n_valid"] = p_user, len(ok)
                if row["symptom_all"]["evaluable"]:
                    c["events"] += 1
                    c["hits"] += row["symptom_all"]["hit"]
                    c["win_days"] += len(ok)
                    c["win_alarm_days"] += sum(1 for d in ok if alarms.get(d, set()) & set(TARGET_TYPES))
                    row["_chance"] = (p_user, len(ok))
            per_event.append(row)
        if part["group"] == "labeled":
            pooled_rate["days"] += fa["days"]
            pooled_rate["alarm_days"] += fa["alarm_days"]
        tgt = fa_controls if part["group"] == "control" else fa_labeled
        tgt["days"] += fa["days"]
        tgt["alarm_days"] += fa["alarm_days"]
        for t in TARGET_TYPES:
            f = false_alarm_rate(alarms, valid, [e["exclude"] for e in events], types=(t,))
            by_type_fa[t]["days"] += f["days"]
            by_type_fa[t]["alarm_days"] += f["alarm_days"]
    rate = lambda d: round(d["alarm_days"] / (d["days"] / 30), 2) if d["days"] else None   # noqa: E731
    pooled_p = pooled_rate["alarm_days"] / pooled_rate["days"] if pooled_rate["days"] else 0.0
    for e in per_event:
        if "_chance" in e:
            p_user, n = e.pop("_chance")
            chance[e["kind"]]["expected_hits"] += chance_hit_probability(pooled_p if p_user is None else p_user, e.pop("_n_valid"))
        e.pop("_p_user", None)
        e.pop("_n_valid", None)
    chance_out = {}
    for kind, c in chance.items():
        if c["events"]:
            chance_out[kind] = {"events": c["events"], "observed_hits": c["hits"], "expected_hits_by_chance": round(c["expected_hits"], 1),
                                "observed_rate": round(c["hits"] / c["events"], 3), "chance_rate": round(c["expected_hits"] / c["events"], 3),
                                "alarm_day_rate_in_window": round(c["win_alarm_days"] / c["win_days"], 3) if c["win_days"] else None,
                                "alarm_day_rate_outside_events": round(pooled_p, 3)}
    sens = {}
    for kind in ("covid", "other"):
        for anchor in ("symptom", "diagnosis"):
            for name in ("all",) + TARGET_TYPES:
                recs = [e[f"{anchor}_{name}"] for e in per_event if e["kind"] == kind and f"{anchor}_{name}" in e]
                if recs:
                    sens[f"{kind}|{anchor}|{name}"] = summarize_events(recs)
    return {
        "sensitivity": sens, "chance_baseline": chance_out,
        "false_alarms": {"controls": {**fa_controls, "per_30_person_days": rate(fa_controls)},
                         "labeled_outside_events": {**fa_labeled, "per_30_person_days": rate(fa_labeled)},
                         "by_type_all_participants_outside_events": {t: {**d, "per_30_person_days": rate(d)} for t, d in by_type_fa.items()}},
        "events": per_event,
        "participants": {"n": len(parts), "labeled": sum(p["group"] == "labeled" for p in parts.values()),
                         "controls": sum(p["group"] == "control" for p in parts.values()),
                         "with_sleep": sum(p["has_sleep"] for p in parts.values())},
    }


def interpretation(res: dict) -> list[str]:
    """지표에서 곧바로 읽히는 해석 문장들 (수치는 모두 res 에서 가져온다)."""
    c, FA = res["chance_baseline"], res["false_alarms"]
    out = []
    if "covid" in c:
        x = c["covid"]
        out.append(f"- **COVID-19 민감도는 우연보다 높지만 크지 않다**: 관찰 {100 * x['observed_rate']:.0f}% vs 우연 기대 {100 * x['chance_rate']:.0f}%, "
                   f"창 안 알람일 비율 {x['alarm_day_rate_in_window']} vs 사건 밖 {x['alarm_day_rate_outside_events']} (약 {x['alarm_day_rate_in_window'] / x['alarm_day_rate_outside_events']:.1f}배). "
                   f"탐지기가 평소에도 자주 울려서, 관찰된 적중의 약 {100 * x['chance_rate'] / x['observed_rate']:.0f}%는 우연한 겹침으로 설명된다.")
    if "other" in c:
        x = c["other"]
        out.append(f"- **기타 질병은 우연 수준을 넘지 못했다**: 관찰 {100 * x['observed_rate']:.0f}% vs 우연 기대 {100 * x['chance_rate']:.0f}% (표본 {x['events']}건).")
    ctl = FA["controls"]
    out.append(f"- **건강 대조군에서도 알람일이 많다**: 사람-30일당 {ctl['per_30_person_days']}일 (평가일의 약 {100 * ctl['alarm_days'] / ctl['days']:.0f}%). "
               f"유형별로는 휴식 빈맥이 {FA['by_type_all_participants_outside_events']['resting_tachycardia']['per_30_person_days']}로 대부분이다 - 걸음으로 잡히지 않는 활동(자전거·근력 등)과 "
               "자유 생활의 일시적 상승이 '휴식 중 빈맥'으로 읽히는 것으로 보인다(원인 분석은 하지 않았다). 서비스 알림으로 쓰기에는 외부 데이터에서 거짓 경보가 많다는 뜻이다.")
    return out


def render(res: dict, paper: dict) -> str:
    """결과를 markdown 으로 (핵심 표 + 논문 참고값 + 한계)."""
    S, FA = res["sensitivity"], res["false_alarms"]
    pct = lambda s: "-" if not s else f"{s['hits']}/{s['n']} ({100 * s['sensitivity']:.0f}%)"   # noqa: E731
    med = lambda s: "-" if not s or s["lead_median_days"] is None else f"{s['lead_median_days']:g}"   # noqa: E731
    L = ["# 외부 검증: Stanford COVID-19 웨어러블 (실제 질병 라벨)", "",
         "**동결된 탐지기(`DEFAULT_CONFIG`)를 튜닝 없이 처음 적용한 값**이다. FitBit 개발 데이터로 고정한 임계값을 이 데이터로 바꾸지 않았다.",
         "분 단위 원본/변환 데이터는 라이선스 표기가 없어 커밋하지 않는다(집계 지표만 공개).", "",
         f"- 참여자 {res['participants']['n']}명 (라벨 {res['participants']['labeled']}명 + 건강 대조군 표본 {res['participants']['controls']}명), 수면 데이터가 있는 참여자 {res['participants']['with_sleep']}명",
         f"- 방법: 하루씩 재생하는 실시간 가정 (그날 끝까지의 이력만 사용). 알람 = 추세 / 개인 기준 이탈 / 휴식 빈맥 중 하나. 창 = 증상 시작일 -{PRE_DAYS}일 ~ +{POST_DAYS}일. "
         f"번인 {BURN_IN_DAYS}일, 심박 {MIN_DAY_MINUTES}분 미만인 날은 제외", "",
         "## 1. 사건 민감도와 선행 일수", "",
         "| 사건 | 기준일 | 적중 / 평가 가능 | 선행 일수 중앙값 | 증상일 이전·당일 적중 | 평가 불가 |", "|---|---|---|---|---|---|"]
    for kind, label in (("covid", "COVID-19"), ("other", "기타 질병")):
        for anchor, alabel in (("symptom", "증상 시작일"), ("diagnosis", "진단일")):
            s = S.get(f"{kind}|{anchor}|all")
            if s:
                L.append(f"| {label} | {alabel} | {pct(s)} | {med(s)} | {s['detected_before_or_on_onset']} | {s['n_not_evaluable']} |")
    L += ["", "### 우연 기준선과 비교 (증상 시작일 기준)", "",
          "알람이 평소에도 자주 울리면 아무 날의 11일 창도 자주 '적중'한다. 기준선 = 각 참여자의 *사건 밖* 하루 알람 확률 p 로 1-(1-p)^(창의 평가일 수) 를 구해 합산한 기대 적중이다.", "",
          "| 사건 | 관찰 적중률 | 우연 기대 적중률 | 창 안 알람일 비율 | 사건 밖 알람일 비율 |", "|---|---|---|---|---|"]
    for kind, label in (("covid", "COVID-19"), ("other", "기타 질병")):
        c = res["chance_baseline"].get(kind)
        if c:
            L.append(f"| {label} | {c['observed_hits']}/{c['events']} ({100 * c['observed_rate']:.0f}%) | {c['expected_hits_by_chance']}/{c['events']} ({100 * c['chance_rate']:.0f}%) | "
                     f"{c['alarm_day_rate_in_window']} | {c['alarm_day_rate_outside_events']} |")
    L += ["", "### 알람 유형별 (증상 시작일 기준, COVID-19)", "", "| 유형 | 적중 / 평가 가능 | 선행 일수 중앙값 |", "|---|---|---|"]
    for t in TARGET_TYPES:
        s = S.get(f"covid|symptom|{t}")
        L.append(f"| {t} | {pct(s)} | {med(s)} |")
    L += ["", "## 2. 거짓 경보 (사람-30일당 알람일 수)", "",
          "| 집단 | 평가일 | 알람일 | 사람-30일당 |", "|---|---:|---:|---:|"]
    for k, label in (("controls", "건강 대조군"), ("labeled_outside_events", "라벨 참여자 (사건 +-14일·증상~회복 제외)")):
        f = FA[k]
        L.append(f"| {label} | {f['days']} | {f['alarm_days']} | {f['per_30_person_days']} |")
    L += ["", "유형별 (모든 참여자, 사건 제외): " + ", ".join(f"{t} {d['per_30_person_days']}" for t, d in FA["by_type_all_participants_outside_events"].items()), "",
          "## 3. 논문 참고값 (SuppTable13, RHR-Diff 오프라인)", "",
          "| 사건 | 증상일이 있는 사건 | 검출 | 놓침 | 검출 비율 | 증상일 이전·당일 검출 | 중앙값(일) |", "|---|---|---|---|---|---|---|"]
    for kind, label in (("covid", "COVID-19"), ("other", "기타 질병")):
        p = paper[kind]
        L.append(f"| {label} | {p['with_symptom_day']} | {p['detected']} | {p['missed']} | {p['sensitivity_among_with_symptom_day']} | {p['detected_at_or_before_onset']} | {p['median_delta_days']} |")
    L += ["", "**직접 비교할 수 없는 이유**: (1) 논문은 참여자마다 오염 비율을 조정한 오프라인 RHR-Diff 이고, 우리는 튜닝 없는 실시간 규칙이다. "
          "(2) 논문의 '검출'은 증상 전후 임의 시점의 구간이고 우리는 [-7,+3]일 창 안의 알람이다(창 정의가 다르다). "
          "(3) 날짜는 개인정보 보호용으로 이동된 값이라 시각 자체는 같지만 논문 표는 이 데이터의 사건 일부만 요약한다. "
          "(4) 논문 수치는 이 표본(라벨 참여자 전원 + 대조군 표본)과 같은 집단의 값이 아니다.", "",
          "## 4. 한계와 해석 주의", "",
          *interpretation(res),
          "- 표본이 작다 (COVID-19 사건 수십 건 이하): 비율의 신뢰구간이 넓다. 대조군은 30명 표본이다.",
          "- 수면 데이터는 일부 참여자만 있다. 없는 참여자는 수면을 '깨어 있음'으로 처리해 수면 맥락 규칙(수면 기준선, 수면 중 서맥)이 사실상 꺼진다 - 야간 신호를 활용하는 논문 방법보다 불리하다.",
          "- 이 탐지기는 분 단위 이상 징후용으로 설계됐고 질병 조기 탐지용으로 튜닝되지 않았다. 여기서의 민감도는 '우연히 얼마나 겹치는가'에 가깝고 임상적 성능이 아니다.",
          "- 장치 구성(논문은 주로 Fitbit)과 생활 환경이 FitBit 개발 데이터와 달라, 개발 데이터에서 본 거짓 경보율이 그대로 옮겨오지 않는다.",
          "- 기타 질병은 별도로 보고했다. 증상일이 없는 사건(진단일만 있음)은 증상 기준에서 제외된다.",
          "- 하루 창 재생은 매일 첫 약 15분의 걸음 맥락을 잃는다(보수적). 이 평가의 결과로 탐지기를 수정하면 외부 테스트셋으로서의 의미가 없어진다."]
    return "\n".join(L)


def main() -> None:
    """CLI: 재생 -> 지표 -> eval/results/external_stanford.{json,md}."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    parts = json.loads((OUT / "participants.json").read_text())
    replays = replay_all(a.workers)
    res = evaluate(replays, parts)
    paper = paper_reference()
    (OUT / "first_warnings.json").write_text(json.dumps(first_warnings(replays, parts), ensure_ascii=False, indent=1))      # 앱이 읽는 로컬 전용 산출물
    RESULTS.mkdir(exist_ok=True)
    slim = {k: v for k, v in res.items() if k != "events"}
    # 사건별 요약에는 참여자 ID 와 날짜를 싣지 않는다(커밋되는 파일은 집계 위주; 날짜는 이동된 값이라도 개별 사건 식별을 피한다)
    slim["events_summary"] = [{"kind": e["kind"], **{k: v for k, v in e.items() if k.startswith(("symptom_", "diagnosis_"))}} for e in res["events"]]
    (RESULTS / "external_stanford.json").write_text(json.dumps({**slim, "paper_reference": paper}, ensure_ascii=False, indent=1))
    md = render(res, paper)
    (RESULTS / "external_stanford.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()

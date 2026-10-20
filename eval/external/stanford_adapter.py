"""Stanford COVID-19 웨어러블 원본 CSV -> 우리 공통 분 단위 스키마 변환 (외부 검증용, 개발 데이터와 섞지 않는다).

책임 경계:
    `stanford_download.py` 가 받은 참여자별 hr/steps/sleep CSV 와 라벨 엑셀을 읽어 `data/external/stanford/vitals_minute.parquet`(앱이 읽는 이름),
    `events.json`(앱 표시용 라벨), `participants.json`(평가용), `id_map.json` 으로 변환한다. 탐지 실행과 지표 계산은 `stanford_eval.py` 의 몫이다.

산출 스키마 (FitBit 처리본과 같은 형식 + source):
    user_id(S001..), ts(naive datetime64[ns], 분 단위), hr_mean/hr_min/hr_max(bpm), hr_n(int), steps(int), asleep(nullable boolean),
    source("stanford"). 희소 저장: 심박이 있는 분과 걸음>0 분만 (app/data/prepare.py 와 같은 규칙).

변환 규칙과 가정:
    - 심박은 불규칙 간격(수 초~분)이라 분 단위로 내림 집계한다. 0 bpm 은 '측정 없음'이라 버린다.
    - 걸음은 이미 1분 단위. 같은 분의 중복은 합한다.
    - 수면: sleep.csv 가 있는 참여자만 asleep=True/False (wake 가 아닌 stage = light/deep/rem 구간을 분 단위로 펼침).
      없는 참여자는 asleep=<NA>이다 - 탐지기는 NA 를 '깨어 있음'으로 취급하므로 수면 맥락 규칙(수면 기준선, 수면 중 서맥 임계)은 사실상 꺼진다.
      즉 이 데이터에서는 수면 의존 규칙이 *건너뛰어진* 것이지 틀리게 적용된 것이 아니다 (평가 결과 문서에 명시).
    - 한 참여자의 파일이 `AA0HAI1_1_hr.csv`, `_2_` 처럼 조각나 있으면 이어 붙이고 같은 분은 한 번만 쓴다.
    - 시각은 데이터셋이 개인정보 보호로 이동(shift)한 값 그대로다. 라벨(엑셀)도 같은 이동을 쓴다. 시간대 변환 없음.

주의: 입력/출력 모두 라이선스가 명시되지 않은 데이터라 `data/external/` 아래(gitignore)에만 둔다. 이 모듈의 출력은 커밋하지 않는다.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "data" / "external" / "stanford"
_TS_RE = re.compile(r"Timestamp\('([^']+)'\)")
_FILE_RE = re.compile(r"^(?P<pid>[A-Z0-9]{7})(?:_\d)?_(?P<kind>hr|steps|sleep)\.csv$")


def parse_dates(cell) -> list[pd.Timestamp]:
    """엑셀 셀의 날짜 목록 문자열("[Timestamp('2025-01-16 00:00:00')]")을 Timestamp 리스트로. 비었거나 NaT 면 빈 리스트."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    return sorted(pd.Timestamp(m) for m in _TS_RE.findall(str(cell)) if m != "NaT")


def _read(paths: list[Path], column: str) -> pd.DataFrame:
    """같은 종류의 조각 CSV 들을 읽어 (datetime, column) 으로 이어 붙인다."""
    parts = [pd.read_csv(p, usecols=["datetime", column]) for p in paths]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def minutes_of(raw_dir: Path, pid: str) -> pd.DataFrame | None:
    """참여자 한 명의 분 단위 희소 프레임 (user_id 열 없음). 심박 파일이 없거나 유효한 심박이 없으면 None.

    Args:
        raw_dir: 원본 CSV 폴더. pid: 데이터셋의 참여자 ID (예: AHYIJDV).

    Returns:
        컬럼 ts, hr_mean, hr_min, hr_max, hr_n, steps, asleep(nullable boolean). 심박이 있는 분과 걸음>0 분만, ts 오름차순.
    """
    files: dict[str, list[Path]] = {}
    for p in sorted(raw_dir.glob(f"{pid}*.csv")):
        m = _FILE_RE.match(p.name)
        if m and m.group("pid") == pid:
            files.setdefault(m.group("kind"), []).append(p)
    if "hr" not in files:
        return None
    hr = _read(files["hr"], "heartrate")
    hr = hr[hr["heartrate"] > 0]
    if hr.empty:
        return None
    hr["ts"] = pd.to_datetime(hr["datetime"]).dt.floor("min")
    agg = hr.groupby("ts")["heartrate"].agg(hr_mean="mean", hr_min="min", hr_max="max", hr_n="size").reset_index()
    df = agg
    if "steps" in files:
        st = _read(files["steps"], "steps")
        st["ts"] = pd.to_datetime(st["datetime"]).dt.floor("min")
        st = st.groupby("ts")["steps"].sum().reset_index()
        st = st[(st["steps"] > 0) & st["ts"].between(agg["ts"].min(), agg["ts"].max())]
        df = agg.merge(st, on="ts", how="outer")
    if "steps" not in df:
        df["steps"] = 0
    df["steps"] = df["steps"].fillna(0).astype("int32")
    df["hr_n"] = df["hr_n"].fillna(0).astype("int32")
    df = df.sort_values("ts").reset_index(drop=True)
    if "sleep" in files:
        df["asleep"] = pd.array(df["ts"].isin(_sleep_minutes(files["sleep"])).to_numpy(), dtype="boolean")
    else:
        df["asleep"] = pd.array([pd.NA] * len(df), dtype="boolean")
    return df[["ts", "hr_mean", "hr_min", "hr_max", "hr_n", "steps", "asleep"]]


def _sleep_minutes(paths: list[Path]) -> pd.DatetimeIndex:
    """sleep.csv 의 wake 가 아닌 stage 구간들이 덮는 모든 분(분 단위 DatetimeIndex)."""
    sl = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    sl = sl[sl["stage"] != "wake"]
    start = pd.to_datetime(sl["datetime"]).dt.floor("min")
    minutes = np.ceil(sl["stage_duration"].to_numpy() / 60).astype(int)
    if len(sl) == 0:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(np.concatenate([(s + pd.to_timedelta(np.arange(n), unit="min")).to_numpy() for s, n in zip(start, minutes)])).unique()


def events_of(label_df: pd.DataFrame) -> dict[str, list[dict]]:
    """라벨 시트(SuppTable3)를 {데이터 파일 참여자 ID: [사건 ...]} 로 바꾼다.

    Args:
        label_df: 컬럼 ParticipantID, Category, Symptom_dates, covid_diagnosis_dates, recovery_dates 를 가진 DataFrame.

    Returns:
        사건 dict = {"kind": "covid" | "other", "symptom": [ISO 날짜], "diagnosis": [...], "recovery": [...], "label_id": 원래 행 ID}.
        한 사람의 `AA0HAI1_Illness_k` 같은 행은 같은 참여자(`AA0HAI1`)의 서로 다른 사건으로 쌓는다.
    """
    out: dict[str, list[dict]] = {}
    for r in label_df.itertuples(index=False):
        pid = str(r.ParticipantID).split("_")[0]
        out.setdefault(pid, []).append({
            "kind": "covid" if r.Category == "COVID-19" else "other", "label_id": str(r.ParticipantID),
            "symptom": [d.date().isoformat() for d in parse_dates(r.Symptom_dates)],
            "diagnosis": [d.date().isoformat() for d in parse_dates(r.covid_diagnosis_dates)],
            "recovery": [d.date().isoformat() for d in parse_dates(r.recovery_dates)]})
    return out


def app_events(parts: dict, id_map: dict[str, str]) -> list[dict]:
    """앱 표시용 사건 목록: 기록 구간 안의 증상 시작 / 진단 / 회복 날짜를 한 줄씩 펼친다.

    Args:
        parts: `build` 가 만든 participants (user_id -> {"start","end","events":[{"kind","symptom","diagnosis","recovery"}]}).
        id_map: 데이터셋 ID -> 가명 (사용하지 않고 인터페이스 대칭용; parts 의 키가 이미 가명이다).

    Returns:
        [{"user_id","type": symptom_onset|diagnosis|recovery,"date": ISO,"illness": covid|other}] (user_id, date 순).
        연구 라벨이며 진단이 아니다 (MCP 도구 설명에도 명시).
    """
    rows = []
    for uid, p in parts.items():
        lo, hi = p["start"][:10], p["end"][:10]
        for e in p["events"]:
            for typ, key in (("symptom_onset", "symptom"), ("diagnosis", "diagnosis"), ("recovery", "recovery")):
                rows += [{"user_id": uid, "type": typ, "date": d, "illness": e["kind"]} for d in e[key] if lo <= d <= hi]
    return sorted(rows, key=lambda r: (r["user_id"], r["date"], r["type"]))


def build(raw_dir: Path = OUT / "raw", xlsx: Path = OUT / "SuppData1.xlsx", out_dir: Path = OUT) -> dict:
    """원본 폴더의 모든 참여자를 변환해 산출물 5종을 쓴다 (멱등: 매번 덮어쓴다).

    라벨 엑셀에 있는 참여자는 group='labeled', 없는 참여자는 group='control'(논문의 건강 대조군) 이다.

    Returns:
        {"n_participants","n_labeled","n_controls","rows"} 요약. 산출물: vitals_minute.parquet, events.json, participants.json, id_map.json.
    """
    labels = pd.read_excel(xlsx, sheet_name="SuppTable3_Fig2a_COVID-19", header=3).dropna(subset=["ParticipantID"])
    events = events_of(labels)
    pids = sorted({m.group("pid") for p in raw_dir.glob("*.csv") if (m := _FILE_RE.match(p.name))})
    id_map = {pid: f"S{i + 1:03d}" for i, pid in enumerate(pids)}
    frames, parts = [], {}
    for pid in pids:
        df = minutes_of(raw_dir, pid)
        if df is None:
            continue
        uid = id_map[pid]
        frames.append(df.assign(user_id=uid, source="stanford"))
        hr = df[df["hr_n"] > 0]
        parts[uid] = {"group": "labeled" if pid in events else "control", "start": hr["ts"].min().isoformat(), "end": hr["ts"].max().isoformat(),
                      "days": int(hr["ts"].dt.date.nunique()), "has_sleep": bool(df["asleep"].notna().any()),
                      "events": [{k: v for k, v in e.items() if k != "label_id"} for e in events.get(pid, [])]}
    all_df = pd.concat(frames, ignore_index=True)[["user_id", "ts", "hr_mean", "hr_min", "hr_max", "hr_n", "steps", "asleep", "source"]]
    all_df["ts"] = all_df["ts"].astype("datetime64[ns]")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_df.to_parquet(out_dir / "vitals_minute.parquet", index=False)
    (out_dir / "events.json").write_text(json.dumps(app_events(parts, id_map), ensure_ascii=False, indent=1))
    (out_dir / "participants.json").write_text(json.dumps(parts, ensure_ascii=False, indent=1))
    (out_dir / "id_map.json").write_text(json.dumps({v: k for k, v in id_map.items()}, indent=1))
    return {"n_participants": len(parts), "n_labeled": sum(p["group"] == "labeled" for p in parts.values()),
            "n_controls": sum(p["group"] == "control" for p in parts.values()), "rows": len(all_df)}


if __name__ == "__main__":
    print(build())

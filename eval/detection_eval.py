"""탐지기 오프라인 평가 — 주입 이벤트 재현율, 정밀도 근사, 깨끗한 데이터의 알림 부담.

책임 경계:
    최종 구성(DEFAULT_CONFIG)으로 주입 데이터와 깨끗한 데이터 전체를 한 번 평가하고 결과를 표로 남긴다.
    구성 선택과 강건성(크기/지속시간 곡선)은 `robustness_benchmark.py` 의 책임이며, 이 모듈은 그 결과 md 를 부록으로 붙인다.

협력 모듈:
    `app.detection.detector`(평가 대상), `app/data/prepare.py` 산출물(vitals_minute*.parquet, labels.json),
    `eval/robustness_benchmark.py`(부록 md 생성).

평가 설계:
    - 재현율: 같은 유형 이상이 라벨 구간과 겹치면 탐지로 친다 (이벤트 단위, 구간 겹침).
    - 정밀도 근사: 주입 사용자 전체 탐지 중 주입 구간과 겹친 비율. 겹치지 않은 탐지는 실데이터의 자연 알림이며
      정답 라벨이 없어 거짓 경보로 단정할 수 없다 (상한 근사).
    - 알림 부담: 깨끗한 데이터에서 사용자-일(심박이 있는 달력일)당 알림 수.
    - 주의: 이 모듈의 재현율은 탐지 규칙과 같은 기준으로 설계한 합성 이벤트에 대한 값이라 낙관적이다.

시간 의미: 모든 시각은 기기 로컬 naive 이며 구간의 end 는 마지막 분을 포함한다.

실행: `.venv/bin/python -m eval.detection_eval` (먼저 `python -m app.data.prepare`)
결과: eval/results/detection_eval.json, detection_eval.md
"""
import json
import sys
from dataclasses import asdict
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.detection import baseline  # noqa: E402
from app.detection.detector import DEFAULT_CONFIG, detect  # noqa: E402

RESULTS = Path(__file__).parent / "results"
TYPES = ["resting_tachycardia", "bradycardia", "personal_deviation", "elevated_resting_trend", "sensor_gap"]
COLS = ["ts", "hr_mean", "hr_min", "hr_max", "hr_n", "steps", "asleep"]


def _overlap(a_start, a_end, b_start, b_end) -> bool:
    """닫힌 구간 [a_start, a_end] 와 [b_start, b_end] 가 한 분이라도 겹치는지."""
    return a_start <= b_end and b_start <= a_end


def _detect_users(df: pd.DataFrame, users) -> dict:
    """사용자별 전체 기록 구간에서 탐지한다 (개인 기준선도 그 사용자의 전체 데이터).

    Args:
        df: user_id 와 COLS 를 가진 parquet 데이터. users: 대상 user_id 들.

    Returns:
        {user_id: list[Anomaly]}. 운영에서는 기준선이 과거 이력만 쓰므로 이 평가는 기준선에 약간 유리하다.
    """
    return {u: detect(df[df["user_id"] == u][COLS].reset_index(drop=True)) for u in users}


def evaluate() -> dict:
    """세 가지 평가(재현율, 정밀도 근사, 알림 부담)를 수행해 JSON 직렬화 가능한 dict 로 반환한다.

    Raises:
        FileNotFoundError: prepare 산출물이 없을 때.
    """
    labels = json.loads((settings.data_dir / "labels.json").read_text())
    for lb in labels:
        lb["start"], lb["end"] = pd.Timestamp(lb["start"]), pd.Timestamp(lb["end"])
    inj = pd.read_parquet(settings.data_dir / "vitals_minute.parquet")
    clean = pd.read_parquet(settings.data_dir / "vitals_minute_clean.parquet")

    # 1) 이벤트 단위 재현율 — 같은 유형 이상이 라벨 구간과 겹치면 탐지로 친다
    users = sorted({lb["user_id"] for lb in labels})
    found = _detect_users(inj, users)
    per_type = {t: {"events": 0, "detected": 0, "severity_match": 0, "missed": []} for t in TYPES}
    for lb in labels:
        r = per_type[lb["type"]]
        r["events"] += 1
        hits = [a for a in found[lb["user_id"]] if a.type == lb["type"] and _overlap(a.start, a.end, lb["start"], lb["end"])]
        if hits:
            r["detected"] += 1
            r["severity_match"] += any(a.severity == lb["severity"] for a in hits)
        else:
            r["missed"].append(lb["event_id"])
    for r in per_type.values():
        r["recall"] = round(r["detected"] / r["events"], 3) if r["events"] else None
    n_ev, n_det = sum(r["events"] for r in per_type.values()), sum(r["detected"] for r in per_type.values())

    # 2) 정밀도 근사 — 주입 사용자의 전체 탐지 중 주입 구간과 겹친 비율 (겹치지 않은 것은 실제 데이터의 자연 알림)
    n_all = n_over = n_same = 0
    unmatched, collateral = Counter(), Counter()
    for u in users:
        mine = [lb for lb in labels if lb["user_id"] == u]
        for a in found[u]:
            n_all += 1
            n_over += any(_overlap(a.start, a.end, lb["start"], lb["end"]) for lb in mine)
            same = any(lb["type"] == a.type and _overlap(a.start, a.end, lb["start"], lb["end"]) for lb in mine)
            n_same += same
            if not same and any(_overlap(a.start, a.end, lb["start"], lb["end"]) for lb in mine):
                collateral[f"{a.type}<-{next(lb['type'] for lb in mine if _overlap(a.start, a.end, lb['start'], lb['end']))}"] += 1
            if not any(_overlap(a.start, a.end, lb["start"], lb["end"]) for lb in mine):
                unmatched[a.type] += 1

    # 3) 깨끗한 데이터의 알림 부담 — 알림 / 사용자-일
    cu = sorted(clean["user_id"].unique())
    cfound = _detect_users(clean, cu)
    user_days = {u: int(clean.loc[(clean["user_id"] == u) & (clean["hr_n"] > 0), "ts"].dt.date.nunique()) for u in cu}
    total_days = sum(user_days.values())
    by_type = Counter(a.type for v in cfound.values() for a in v)
    by_sev = Counter(a.severity for v in cfound.values() for a in v)
    burden = {
        "user_days": total_days,
        "alerts_total": int(sum(by_type.values())),
        "alerts_per_user_day": round(sum(by_type.values()) / total_days, 4),
        "by_type": {t: {"alerts": by_type.get(t, 0), "per_user_day": round(by_type.get(t, 0) / total_days, 4)} for t in TYPES},
        "by_severity": {s: by_sev.get(s, 0) for s in ("info", "warning", "critical")},
        "per_user": {u: {"days": user_days[u], "alerts": len(cfound[u])} for u in cu},
    }
    return {
        "config": {"detector": asdict(DEFAULT_CONFIG), "rest_window_min": baseline.REST_WINDOW_MIN, "n_labeled_events": n_ev, "injected_users": users},
        "recall": {"overall": round(n_det / n_ev, 3), "events": n_ev, "detected": n_det, "by_type": per_type},
        "precision_proxy": {
            "detections_on_injected_users": n_all, "overlap_any_injected": n_over, "overlap_same_type": n_same,
            "precision_any": round(n_over / n_all, 3) if n_all else None,
            "precision_same_type": round(n_same / n_all, 3) if n_all else None,
            "unmatched_by_type": dict(unmatched),
            "collateral_other_type": dict(collateral),
        },
        "clean_alert_burden": burden,
    }


# 튜닝 기록: 의사결정 근거를 결과와 함께 남긴다 (2026-10-20 측정, 깨끗한 데이터 15명·469 사용자-일)
TUNING_NOTES = """\
## 임계값 튜닝 기록

계약 초안의 임계값(빈맥 100/130, 서맥 40/35, 개인 z>4, 추세 +10 bpm, 결측 2시간)은 그대로 두고, 다음만 조정했다.

1. **운동 워시아웃 10분 → 15분 (`baseline.REST_WINDOW_MIN`).** 깨끗한 데이터에서 알림 대부분이 빈맥이었고,
   상당수는 직전 30분 안에 수백 보를 걸은 *운동 직후 회복 구간*이었다. 워시아웃별 빈맥 알림 수(깨끗한 데이터 전체):
   10분 27건 → **15분 14건** → 20분 12건 → 30분 8건. 15분을 택한 이유: 10→15에서 절반으로 줄어드는 가파른 구간이고,
   그 뒤로는 한계 이득이 작은 반면 20분 이상은 실제 휴식 중 빈맥을 놓칠 위험이 커진다. 임계값을 건드려 숫자를 맞춘 것이 아니라
   '운동 직후 상승은 이상이 아님'이라는 규칙의 정의를 생리(회복 시간)에 맞게 고친 것이다.
2. **개인 대비 이탈에 최소 절대 상승폭 +15 bpm 추가 (`DEVIATION_MIN_DELTA_BPM`).** MAD가 매우 작은 사용자는 사소한 변화도
   z>4 를 넘기 때문. sigma 하한은 2 bpm (`MIN_SIGMA`).
3. **추세 기준선에서 이미 '상승'으로 판정된 날 제외.** 그렇지 않으면 첫 상승일이 직전 7일 중앙값을 끌어올려 둘째 날을 놓친다.
4. **센서 결측은 걸음이 기록된(착용 증거) 12시간 이하 구간만.** 걸음 없는 긴 결측은 충전·미착용으로 본다.
5. **연구 방법 도입 (기본 구성 DEFAULT_CONFIG).** 시간 단위 안정시 심박 EllipticEnvelope(RHR-AD)와 CuSum 추세를 규칙에 더했다.
   구성은 `robustness_benchmark.py` 의 튜닝 시드(101~103)로 고르고 홀드아웃 시드(20261020)로만 보고했다 (아래 부록).
   홀드아웃에서 규칙 대비 추세 재현율 14/16 -> 15/16, 경계 표본 탐지율 4/29 -> 10/29 로 올랐지만 추세 음성 표본 거짓 알림이
   0/11 -> 2/11 로 늘었다. 개인 이탈 재현율 12/17 -> 13/17 은 표본 수가 작아 유의하다고 말할 수 없다. 깨끗한 데이터 알림은
   0.0789 -> 0.0938 /사용자-일. 이 수치는 신뢰구간이 넓으므로 '개선 확정'이 아니라 '비용 대비 약간의 이득' 정도로 읽는다.
6. **운동 억제의 천장 (극단 빈맥 규칙).** 걸음이 조금이라도 있으면 몇 bpm이든 묵살하던 문제를 고쳤다: 210 bpm 이상 10분+,
   또는 최근 15분 안에 분당 60보 이상 보행이 없는 180 bpm 이상 10분+ 는 활동 맥락과 무관하게 critical. 근거는 연령 예측 최대 심박
   (220-나이: 18세 약 202, 40세 이상 180 이하). 깨끗한 데이터 알림은 44건 그대로(0.0938/사용자-일)였고 라벨 재현율도 변하지 않았다.
   대가는 극단 심박 구간에서의 드문 오탐 가능성이며 `robustness_benchmark.md` 에 표본별로 기록했다.

## 해석상 한계

- 재현율은 우리가 만든 합성 이벤트(탐지 규칙과 같은 기준으로 설계)에 대한 값이라 낙관적이다. 실제 임상 이벤트의 재현율이 아니다.
- 깨끗한 데이터의 알림은 '거짓 경보'의 상한 근사다: 걸음으로 잡히지 않는 운동(자전거·근력 등), 카페인·스트레스성 일시 상승은
  규칙만으로 구분되지 않는다. 라벨이 없어 정밀도를 직접 계산할 수 없다.
- 주입 이벤트는 쉬운 편(10분 이상 지속, 뚜렷한 상승)이다. 짧거나 경계 수준의 이벤트에 대한 민감도는 측정하지 않았다.
"""


# 극단 빈맥 규칙 도입 전(2026-10-20 03:31 실행) 측정값 - 도입 후와 나란히 보이기 위해 고정해 둔 기록
PRE_EXTREME_RULE = {"detected": 14, "events": 14, "overlap": 18, "detections": 36, "precision": 0.5,
                    "clean_alerts": 44, "clean_per_user_day": 0.0938, "clean_critical": 4}


def render_md(res: dict) -> str:
    """evaluate() 결과를 markdown 리포트로 렌더링한다 (표 + 튜닝 기록 + 해석상 한계)."""
    rc, pp, bd = res["recall"], res["precision_proxy"], res["clean_alert_burden"]
    L = ["# 탐지기 오프라인 평가", "",
         f"운동 워시아웃 {res['config']['rest_window_min']}분 · 라벨 이벤트 {rc['events']}건 · 주입 사용자 "
         f"{len(res['config']['injected_users'])}명", "",
         "## 1. 이벤트 단위 재현율 (같은 유형 이상이 주입 구간과 겹치면 탐지)", "",
         "| 유형 | 이벤트 | 탐지 | 재현율 | 심각도 일치 | 놓친 이벤트 |", "|---|---:|---:|---:|---:|---|"]
    for t in TYPES:
        r = rc["by_type"][t]
        L.append(f"| {t} | {r['events']} | {r['detected']} | {r['recall']} | {r['severity_match']}/{r['detected']} | "
                 f"{', '.join(r['missed']) or '-'} |")
    L += [f"| **전체** | {rc['events']} | {rc['detected']} | **{rc['overall']}** | | |", "",
          "## 2. 정밀도 근사 (주입 사용자 전체 탐지 기준)", "",
          f"- 탐지 {pp['detections_on_injected_users']}건 중 주입 구간과 겹침 {pp['overlap_any_injected']}건 "
          f"(정밀도 근사 **{pp['precision_any']}**), 유형까지 일치 {pp['overlap_same_type']}건 ({pp['precision_same_type']})",
          f"- 주입 구간과 겹치지만 다른 유형인 탐지(주입의 부수 효과, 예: 추세 주입이 만든 개인 대비 이탈): {pp['collateral_other_type']}",
          f"- 주입과 무관한 탐지(실제 데이터의 자연 알림): {pp['unmatched_by_type']}", "",
          "## 3. 깨끗한 데이터의 알림 부담", "",
          f"- 사용자-일 {bd['user_days']} · 알림 {bd['alerts_total']}건 · **사용자-일당 {bd['alerts_per_user_day']}건**"
          f" (심각도: {bd['by_severity']})", "",
          "| 유형 | 알림 | 사용자-일당 |", "|---|---:|---:|"]
    L += [f"| {t} | {v['alerts']} | {v['per_user_day']} |" for t, v in bd["by_type"].items()]
    pp0 = PRE_EXTREME_RULE
    L += ["", "## 극단 빈맥 규칙(운동 억제의 천장) 도입 전/후", "",
          "도입 전 값은 2026-10-20 03:31 실행(같은 데이터·같은 평가 코드)에서 측정한 것이고, 도입 후 값은 이 문서를 만든 실행의 측정이다.", "",
          "| 지표 | 도입 전 | 도입 후 |", "|---|---|---|",
          f"| 라벨 이벤트 재현율 | {pp0['detected']}/{pp0['events']} | {rc['detected']}/{rc['events']} |",
          f"| 정밀도 근사 (주입 구간과 겹친 탐지 / 전체) | {pp0['overlap']}/{pp0['detections']} ({pp0['precision']}) | "
          f"{pp['overlap_any_injected']}/{pp['detections_on_injected_users']} ({pp['precision_any']}) |",
          f"| 깨끗한 데이터 알림 | {pp0['clean_alerts']}건 ({pp0['clean_per_user_day']}/사용자-일) | "
          f"{bd['alerts_total']}건 ({bd['alerts_per_user_day']}/사용자-일) |",
          f"| 깨끗한 데이터 critical 알림 | {pp0['clean_critical']}건 | {bd['by_severity']['critical']}건 |", "",
          "운동 음성 표본 거짓 알림과 극단 빈맥 양성 재현율은 `robustness_benchmark.md` 의 같은 이름 절에 있다.", "",
          TUNING_NOTES]
    return "\n".join(L)


if __name__ == "__main__":
    out = evaluate()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "detection_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    md = render_md(out)
    bench = RESULTS / "robustness_benchmark.md"
    if bench.exists():       # 강건성 벤치마크(별도 실행)의 표와 방법론/출처를 함께 싣는다
        md += "\n" + bench.read_text().replace("# 강건성", "# 부록: 강건성", 1)
    (RESULTS / "detection_eval.md").write_text(md)
    print(md.split("## 임계값 튜닝 기록")[0])

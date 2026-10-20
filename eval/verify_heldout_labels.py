"""홀드아웃 에이전트 평가 라벨(eval/heldout_examples.json)을 데이터에서 다시 유도해 검증한다.

책임 경계:
    라벨의 expected_types / expected_escalation / low_coverage 가 MCP 도구(= 탐지기)의 실제 출력과 일치하는지만 본다.
    에이전트는 실행하지 않는다 (에이전트 구현과 독립적으로 라벨을 만들었음을 보장하기 위함).

협력 모듈: `app.mcp_server.server`(도구 함수를 직접 호출), `eval/heldout_labels.md`(유도 방법 설명).

데이터 가정: '최근 N일'은 벽시계가 아니라 그 사용자의 마지막 심박 기록 시각 기준이다 (도구에서 end 를 생략한 경우).
low_coverage 는 에이전트의 재계획 조건과 같다: coverage_pct < 50 또는 rest_minutes < 60.

실행: `.venv/bin/python -m eval.verify_heldout_labels` (불일치가 있으면 목록을 출력하고 종료 코드 1)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.mcp_server.server import detect_anomalies, get_vitals_summary  # noqa: E402

EXAMPLES = Path(__file__).parent / "heldout_examples.json"
MAX_LOOKBACK_DAYS = 30        # 재계획은 조회 기간을 2배로(최대 30일) 늘린다


def derive(user_id: str, lookback_days: float) -> dict:
    """한 (사용자, 기간) 창의 정답을 데이터에서 계산한다.

    Args:
        user_id: 가명 사용자 ID. lookback_days: 마지막 심박 시각에서 거슬러 올라가는 기간(일).

    Returns:
        {"types": 탐지된 유형 집합(정렬 리스트), "critical": critical 존재 여부, "low_coverage": 재계획 조건 해당 여부,
        "coverage_pct", "rest_minutes", "replan_types": 재계획 창(2배, 최대 30일)에서의 유형 리스트}.
    """
    det, summ = detect_anomalies(user_id, lookback_days), get_vitals_summary(user_id, lookback_days)["summary"]
    low = summ["coverage_pct"] < 50 or summ["rest_minutes"] < 60
    wide = detect_anomalies(user_id, min(lookback_days * 2, MAX_LOOKBACK_DAYS))
    return {"types": sorted({a["type"] for a in det["anomalies"]}),
            "critical": any(a["severity"] == "critical" for a in det["anomalies"]),
            "low_coverage": low, "coverage_pct": summ["coverage_pct"], "rest_minutes": summ["rest_minutes"],
            "replan_types": sorted({a["type"] for a in wide["anomalies"]})}


def check() -> list[str]:
    """모든 예시의 라벨을 데이터와 대조해 불일치 설명 목록을 반환한다 (비어 있으면 일치)."""
    problems = []
    for ex in json.loads(EXAMPLES.read_text()):
        win = ex.get("ground_truth_window")
        if not win:
            continue
        d = derive(win["user_id"], win["lookback_days"])
        if not ex.get("no_reanalysis") and sorted(ex["expected_types"]) != d["types"]:
            problems.append(f"{ex['id']}: expected_types {sorted(ex['expected_types'])} != 데이터 {d['types']}")
        if ex["expected_escalation"] != d["critical"]:
            problems.append(f"{ex['id']}: expected_escalation {ex['expected_escalation']} != 데이터 critical {d['critical']}")
        if "low_coverage" in ex and ex["low_coverage"] != d["low_coverage"]:
            problems.append(f"{ex['id']}: low_coverage {ex['low_coverage']} != 데이터 {d['low_coverage']} "
                            f"(coverage {d['coverage_pct']}%, 휴식 {d['rest_minutes']}분)")
        if ex.get("expected_replan") and sorted(ex["expected_types"]) != d["replan_types"]:
            problems.append(f"{ex['id']}: 재계획 창의 유형 {d['replan_types']} != expected_types")
    return problems


if __name__ == "__main__":
    found = check()
    print("\n".join(found) if found else "모든 라벨이 데이터와 일치합니다.")
    sys.exit(1 if found else 0)

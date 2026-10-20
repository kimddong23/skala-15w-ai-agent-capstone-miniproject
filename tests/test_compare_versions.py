"""Before/After 하네스(eval/compare_versions.py)의 지표 정의 회귀 테스트: 계산 오류가 되살아나지 않게 한다."""
import json

import pytest

from eval import compare_versions as cv


def test_percentile_is_linear_interpolation_for_small_samples():
    # n=6, 10, 14 에서 nearest-rank 구현이 한 칸 어긋났다 - numpy 선형 보간 정의와 일치해야 한다.
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert cv.pct(xs, 50) == 3.5
    assert cv.pct(xs, 95) == 5.75
    assert cv.pct([], 50) is None


@pytest.mark.parametrize("report,loose,fixed", [
    ("... 이는 의학적 진단이 아닌 참고 정보입니다.", True, False),   # v1 이 자기 말로 쓴 면책 (D02/D03/D05 에서 놓쳤던 형태)
    ("위 내용은 의학적 진단이 아닙니다.", True, False),
    ("(의학적 진단이 아님을 알려드립니다.)", True, False),
    ("심박이 높습니다. 병원에 가세요.", False, False),
    ("본문\n\n이 서비스는 의료기기가 아니며 진단을 제공하지 않습니다. 웨어러블 기기의 심박 측정에는 오차가 있고, ...", True, True),
])
def test_disclaimer_flags(report, loose, fixed):
    flags = cv.disclaimer_flags(report)
    assert (flags["disclaimer"], flags["disclaimer_fixed"]) == (loose, fixed)


def _row(tool_calls_by_name=None):
    r = {"usage": {k: 0 for k in ("llm_calls", "input_tokens", "output_tokens", "total_tokens", "cost_usd", "retries")} | {"failures": {}}, "ref_openai_cost_usd": 0,
         "latency_s": 1.0, "ttft_s": 1.0, "route": "simple", "tool_calls": 0, "escalation_flag": False, "notice_text": False, "urgent_text": False,
         "diagnosis_assertions": 0, "disclaimer": True, "disclaimer_fixed": False, "numbers_cited": 0, "numbers_unsupported": 0, "degraded": False,
         "failed_generation": False, "validator_attempts": 0, "report": "x"}
    if tool_calls_by_name is not None:
        r["tool_calls_by_name"] = tool_calls_by_name
    return r


def test_aggregate_labels_tool_call_scope():
    case = {"id": "D01", "expected_route": "simple", "critical": False}
    assert cv.aggregate("v1", [case], {"D01": _row()})["tool_calls_scope"] == "summary_detect_only"
    assert cv.aggregate("v1", [case], {"D01": _row({"search": 1})})["tool_calls_scope"] == "all"


def test_recompute_rewrites_derived_metrics_without_llm(tmp_path, monkeypatch):
    monkeypatch.setattr(cv, "RESULTS", tmp_path)
    row = _row() | {"disclaimer": False, "report": "참고 정보이며 의학적 진단이 아닙니다."}   # 저장 당시 값이 틀렸어도 원문에서 다시 계산
    src = tmp_path / "ba.json"
    src.write_text(json.dumps({"meta": {"git": {"commit": "abc", "dirty": False}, "when": "t", "backend": "fake", "fast_model": "f", "quality_model": "q",
                                         "hardware": {"chip": "c", "ram_gb": 1}, "n_cases": 1, "n_dev": 1, "n_heldout": 0, "load_avg_start_end": [0, 0]},
                               "cases": [{"id": "D01", "expected_route": "simple", "critical": False}], "results": {"v1": {"D01": row}, "v2": {"D01": row}}}))
    cv.recompute(src, "트리 설명")
    out = json.loads(src.read_text())
    assert out["aggregate"]["v1"]["disclaimer_rate"] == 1.0
    assert "트리 설명" in (tmp_path / "ba.md").read_text()

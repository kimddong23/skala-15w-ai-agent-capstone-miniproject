"""`eval/judge_eval.py` 의 순수 함수 테스트 (judge LLM 은 호출하지 않는다).

범위: 수치 주입(perturb)이 고정 안내문을 건드리지 않고 수치만 바꾸는지, 정확 일치가 주입 오류를 실제로 잡는지, judge 판정 보조 함수,
일치도 집계(agreement)와 markdown 렌더링. 협력: `eval.judge_eval`, `app.agent.validator`(numbers_check), `app.agent.gates`(고정 문구).
"""
from app.agent.gates import DISCLAIMER, ESCALATION_MESSAGE
from app.agent.validator import collect_facts, numbers_check
from eval import judge_eval as je

REPORT = f"{ESCALATION_MESSAGE}\n요약: 평균 심박 112 bpm 구간이 25분 이어졌습니다.\n{DISCLAIMER}"


def test_perturb_changes_only_the_report_number_and_keeps_fixed_notices():
    out = je.perturb(REPORT)
    assert out is not None
    bad, old, new = out
    assert (old, new) == ("112", "119") and "119 bpm" in bad and "112 bpm" not in bad
    assert ESCALATION_MESSAGE in bad and DISCLAIMER in bad          # 고정 안내문 속 숫자(119 등)는 건드리지 않는다
    assert je.perturb("숫자가 없는 문장입니다.") is None


def test_perturb_falls_back_to_minutes_when_no_bpm():
    bad, old, new = je.perturb("이상이 25분 이어졌습니다.")
    assert (old, new) == ("25", "32") and "32분" in bad


def test_exact_matching_catches_the_seeded_error_that_the_original_passes():
    facts = collect_facts({"summary": {"hr_mean": 112.0}, "n": 25}, {"anomalies": [{"duration_min": 25}]})
    body = "평균 심박 112 bpm 구간이 25분 이어졌습니다."
    assert numbers_check(body, facts)[1] == []
    bad, _, new = je.perturb(body)
    assert float(new) in numbers_check(bad, facts)[1]


def test_judge_pass_and_catch_rules():
    assert je.passes({"faithfulness": 5, "clarity": 4, "safety_tone": 4}) and not je.passes({"faithfulness": 5, "clarity": 3, "safety_tone": 5})
    assert not je.passes({"error": "x"})
    assert je.caught_by_judge({"faithfulness": 3, "unsupported_claims": []}, "112", "119")           # 점수가 내려감
    assert je.caught_by_judge({"faithfulness": 5, "unsupported_claims": ["심박 119 bpm 은 사실과 다름"]}, "112", "119")    # 틀린 수치를 지적
    assert not je.caught_by_judge({"faithfulness": 5, "unsupported_claims": []}, "112", "119")
    assert not je.caught_by_judge({"error": "x"}, "112", "119")


def _item(i, faith, passed, unsupported=0, diag=0, pert=None, user="U01"):
    j = {"faithfulness": faith, "clarity": 5, "safety_tone": 5, "unsupported_claims": [], "seconds": 10.0}
    it = {"id": i, "source": "dev", "route": "analysis", "validator_passed": passed, "numbers_cited": 5, "numbers_unsupported": unsupported,
          "diagnosis": diag, "message": "m", "user_id": user, "days": 7, "judge": j}
    if pert:
        it["perturbed"] = pert
    return it


def test_agreement_counts_and_markdown_render():
    pert_caught = {"old": "1", "new": "8", "exact_caught": True, "judge_caught": True, "judge": {"faithfulness": 2}}
    pert_missed = {"old": "2", "new": "9", "exact_caught": True, "judge_caught": False, "judge": {"faithfulness": 5}}
    items = [_item("A", 5, True, pert=pert_caught), _item("B", 4, True, pert=pert_missed), _item("C", 2, False)]
    items[0]["judge_repeat"] = dict(items[0]["judge"])
    ag = je.agreement(items)
    assert ag["perturbation"]["n"] == 2 and ag["perturbation"]["exact_caught"] == 2 and ag["perturbation"]["judge_caught"] == 1
    assert ag["perturbation"]["missed_by_judge"][0][0] == "B"
    assert ag["vs_validator"]["agree"] == 3 and ag["stability"]["identical_scores"] == 1
    md = je.render({"model": "qwen3:8b", "perturb": 7, "items": items}, ag)
    assert "정확 일치" in md and "2/2 (100%)" in md and "1/2 (50%)" in md


# ---------- 캐시 키 / 점수 정규화 / null 점수 렌더 ----------
import json  # noqa: E402


def test_cache_key_changes_with_report_facts_model_prompt_and_repeat(monkeypatch):
    base = je.cache_key("m", "FACTS", "REPORT")
    assert base == je.cache_key("m", "FACTS", "REPORT")                       # 같은 입력 -> 같은 키
    assert base != je.cache_key("m", "FACTS", "REPORT2")                      # 리포트가 바뀌면 낡은 점수를 쓰지 않는다
    assert base != je.cache_key("m", "FACTS2", "REPORT")                      # 도구 사실이 바뀌어도
    assert base != je.cache_key("other-model", "FACTS", "REPORT")             # judge 모델이 바뀌어도
    assert base != je.cache_key("m", "FACTS", "REPORT", repeat=1)             # 안정성 재채점은 실제로 다시 호출되도록 다른 키
    monkeypatch.setattr(je, "SYSTEM", je.SYSTEM + " (수정)")
    assert base != je.cache_key("m", "FACTS", "REPORT")                       # 프롬프트가 바뀌어도


def test_score_accepts_only_integers_1_to_5():
    assert [je.score({"s": v}, "s") for v in (1, 5, 0, 6, True, "5", 4.0, None)] == [1, 5, None, None, None, None, None, None]
    assert je.score({}, "s") is None


class FakeResp:
    def __init__(self, content):
        self._c = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"message": {"content": self._c}}


class FakeClient:
    def __init__(self, content):
        self.content = content

    def post(self, *a, **k):
        return FakeResp(self.content)


def test_judge_normalizes_null_and_malformed_scores_instead_of_raising():
    out = je.judge(FakeClient(json.dumps({"faithfulness": None, "clarity": 9, "safety_tone": "5", "unsupported_claims": "문자열"})), "F", "R")
    assert (out["faithfulness"], out["clarity"], out["safety_tone"], out["unsupported_claims"]) == (None, None, None, [])
    assert "error" in je.judge(FakeClient("JSON 아님"), "F", "R") and "error" in je.judge(FakeClient(json.dumps([1, 2])), "F", "R")
    assert not je.passes(out) and not je.caught_by_judge(out, "1", "8")


def test_agreement_and_render_survive_null_scores_and_derive_numbers_from_data():
    null_item = _item("N", None, True)
    null_item["judge"].update(clarity=None)
    items = [_item("A", 5, True, pert={"old": "1", "new": "8", "exact_caught": True, "judge_caught": False, "judge": {"faithfulness": 5}}), null_item,
             _item("L", 2, True)]
    items[2]["judge"]["unsupported_claims"] = ["REPORT 의 30bpm 은 FACTS 와 다름"]
    ag = je.agreement(items)
    assert ag["missing_scores"]["faithfulness"] == 1 and ag["missing_scores"]["clarity"] == 1
    assert ag["low_faithfulness_cases"][0]["id"] == "L" and ag["low_faithfulness_cases"][0]["claims"] == ["REPORT 의 30bpm 은 FACTS 와 다름"]
    md = je.render({"model": "qwen3:8b", "perturb": 7, "items": items, "prompt_fingerprint": "abc"}, ag)      # TypeError 없이 렌더
    assert "L (dev" in md and "REPORT 의 30bpm" in md and "abc" in md
    assert "D03" not in md and "D06" not in md and "반올림" not in md           # 사례별 하드코딩 문장이 없다


def test_perturbation_all_judge_misses_is_reported_from_data():
    pert = {"old": "1", "new": "8", "exact_caught": True, "judge_caught": False, "judge": {"faithfulness": 5}}
    ag = je.agreement([_item("A", 5, True, pert=pert), _item("B", 5, True, pert={**pert, "old": "2", "new": "9"})])
    md = je.render({"model": "m", "perturb": 7, "items": []}, ag)
    assert "정확 일치가 더 많이 잡았다" in md and "2/2 (100%)" in md and "0/2 (0%)" in md and "만점을 준 비율은 100%" in md

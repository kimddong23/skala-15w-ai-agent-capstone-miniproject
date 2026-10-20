"""`eval/rag_eval.py` 의 순수 함수 테스트 (Ollama 불필요): 지표 정의(hit@k), 키워드 기준선의 결정론, A 방식의 자명한 hit@3.

협력: `eval.rag_eval`, `app/rag`(코퍼스), `eval/rag_questions.json`(라벨 구조).
"""
import json

from eval import rag_eval as re_


def test_metrics_are_hit_at_k_and_mrr():
    ranked = [["a", "b", "c"], ["x", "y", "a"], ["x", "y", "z", "a"], ["q"]]
    gold = [["a"], ["a"], ["a"], ["a", "b"]]
    m = re_.metrics(ranked, gold)
    assert m["hit@1"] == 0.25 and m["hit@3"] == 0.5 and "recall@1" not in m
    assert m["mrr"] == round((1 + 1 / 3 + 1 / 4 + 0) / 4, 3)
    assert re_.metrics([["b", "a"]], [["a", "b"]])["hit@1"] == 1.0          # 정답이 여러 개여도 하나만 맞으면 적중 (그래서 recall 이 아니다)


def test_rank_docs_keeps_first_occurrence_order():
    assert re_.rank_docs(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_keyword_baseline_is_deterministic_and_ties_follow_doc_id_order():
    secs = re_._sections(None)
    ids = [d for d, _ in secs]
    assert ids == sorted(ids)                                             # 문서 id 순 정렬 (동점 처리의 근거)
    r1, r2 = re_.keyword_baseline("ㅋㅋㅋ", secs), re_.keyword_baseline("ㅋㅋㅋ", secs)
    assert r1 == r2 and r1 == sorted(r1)                                  # 단서가 없으면 점수가 전부 같아 문서 id 순


def test_method_a_has_only_two_candidate_docs_so_hit_at_3_is_trivial_for_answerable_questions():
    old = re_._sections(set(re_.OLD_DOCS))
    assert {d for d, _ in old} == set(re_.OLD_DOCS)
    qs = json.loads(re_.QUESTIONS.read_text(encoding="utf-8"))
    answerable = [q for q in qs if set(q["gold"]) & set(re_.OLD_DOCS)]
    assert answerable
    for q in answerable:
        ranked = re_.keyword_baseline(q["question"], old)
        assert len(ranked) <= 2 and set(ranked[:3]) & set(q["gold"])        # 후보가 2개뿐이라 상위 3개에는 항상 정답이 들어 있다

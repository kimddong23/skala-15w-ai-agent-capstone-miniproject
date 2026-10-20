"""홀드아웃 에이전트 평가 라벨의 구조와 데이터 정합성 테스트.

범위: eval/heldout_examples.json 의 스키마·구성(카테고리별 개수)과, 라벨이 MCP 도구가 반환하는 실제 탐지 결과와 일치하는지.
협력: `eval.verify_heldout_labels`(유도 로직), `tests/_prepared.py`(산출물 보장). 에이전트는 실행하지 않는다.
"""
import json
from collections import Counter

from eval.verify_heldout_labels import EXAMPLES, check
from tests._prepared import ensure_prepared

ensure_prepared()
REQUIRED = {"id", "message", "user_id", "thread", "turn", "expected_route", "expected_types", "expected_escalation"}


def load() -> list[dict]:
    """홀드아웃 예시 JSON 을 읽는다."""
    return json.loads(EXAMPLES.read_text())


def test_schema_and_composition():
    """필수 필드, 고유 ID, 16건, 요청한 카테고리 구성(지식 4, 인사/범위밖 2, 분석 6, 후속 2, 적대적 2)."""
    ex = load()
    assert len(ex) == 16 and len({e["id"] for e in ex}) == 16
    assert all(REQUIRED <= set(e) for e in ex)
    assert all(e["expected_route"] in ("simple", "analysis") for e in ex)
    c = Counter(e["category"] for e in ex)
    assert c["knowledge"] == 4 and c["greeting"] + c["out_of_scope"] == 2
    assert sum(v for k, v in c.items() if k.startswith("analysis")) == 6
    assert sum(v for k, v in c.items() if k.startswith("followup")) == 2
    assert sum(v for k, v in c.items() if k.startswith("adversarial")) == 2


def test_analysis_set_covers_critical_and_low_coverage():
    """분석 요청에 critical 사용자 2명 이상과 저커버리지 사용자 1명 이상이 포함된다."""
    analysis = [e for e in load() if e["category"].startswith("analysis")]
    assert len({e["user_id"] for e in analysis if e["expected_escalation"]}) >= 2
    assert any(e.get("low_coverage") for e in analysis)


def test_threads_have_two_turns():
    """스레드는 2턴이고 같은 사용자이며, 후속 턴은 각각 재분석 없음/기간 변경 재분석을 요구한다."""
    threads = {}
    for e in load():
        if e["thread"]:
            threads.setdefault(e["thread"], []).append(e)
    assert len(threads) == 2
    for turns in threads.values():
        assert sorted(t["turn"] for t in turns) == [1, 2] and len({t["user_id"] for t in turns}) == 1
    cats = {e["category"] for e in load() if e["turn"] == 2}
    assert cats == {"followup_no_reanalysis", "followup_reanalysis_new_period"}


def test_labels_match_data():
    """expected_types / expected_escalation / low_coverage 가 도구의 실제 출력과 일치한다."""
    assert check() == []

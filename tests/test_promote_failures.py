"""`eval/promote_failures.py` 테스트: 실패 선별, 홀드아웃 보호, 중복 방지, 드라이런 기본값 (LangSmith 에는 연결하지 않는다 - 가짜 클라이언트).

협력: `eval.promote_failures`, `eval/heldout_examples.json`(실제 홀드아웃 문장으로 보호 규칙을 확인).
"""
import json
from types import SimpleNamespace

import pytest

from eval import promote_failures as pf

OK = {"route_accuracy": 1.0, "expected_types_recall": 1.0, "escalation_correct": 1.0, "validator_passed": 1.0, "validator_first_try": 0.0,
      "no_diagnosis_assertion": 1.0, "disclaimer_present": 1.0, "numbers_grounded": 1.0, "latency_s": 50.0}


def row(turns, **bad):
    """합성 결과 행: 기본은 전부 통과, bad 로 지정한 지표만 낮춘다."""
    return {"kind": "analysis", "inputs": {"turns": turns}, "reference": {"kind": "analysis", "route": "analysis"}, "scores": {**OK, **bad}}


class FakeClient:
    """langsmith.Client 의 필요한 부분만 흉내낸다. 기록 호출을 저장한다."""

    def __init__(self, existing=()):
        self.existing = [SimpleNamespace(inputs=i) for i in existing]
        self.created, self.dataset_created = [], False

    def has_dataset(self, dataset_name):
        return bool(self.existing) or self.dataset_created

    def read_dataset(self, dataset_name):
        return SimpleNamespace(id="ds1")

    def create_dataset(self, name, description=""):
        self.dataset_created = True
        return SimpleNamespace(id="ds1")

    def list_examples(self, dataset_id):
        return self.existing

    def create_examples(self, dataset_id, inputs, outputs, metadata):
        self.created.append((inputs, outputs, metadata))


def test_only_correctness_failures_are_selected_not_latency_or_first_try():
    rows = [row(["a"]), row(["b"], numbers_grounded=0.8), row(["c"], validator_first_try=0.0, latency_s=999.0), row(["d"], route_accuracy=0.0, escalation_correct=0.0)]
    picked, skipped = pf.select_failures(rows, set())
    assert [p["inputs"]["turns"] for p in picked] == [["b"], ["d"]] and skipped == []
    assert picked[1]["metadata"]["failed_keys"] == ["route_accuracy", "escalation_correct"] and picked[0]["outputs"]["route"] == "analysis"


def test_none_scores_are_not_failures():
    assert pf.failed_keys({**OK, "numbers_grounded": None}) == []


def test_examples_overlapping_the_heldout_set_are_never_promoted():
    held = pf.heldout_texts()
    assert len(held) == 16
    msg = next(iter(held))
    picked, skipped = pf.select_failures([row([msg], route_accuracy=0.0), row(["다른 질문"], route_accuracy=0.0)], held)
    assert [p["inputs"]["turns"] for p in picked] == [["다른 질문"]] and skipped[0]["reason"] == "heldout_overlap"


def test_promote_writes_only_new_examples_with_metadata():
    picked, _ = pf.select_failures([row(["x"], numbers_grounded=0.5), row(["y"], route_accuracy=0.0)], set())
    client = FakeClient(existing=[{"turns": ["x"]}])
    res = pf.promote(picked, client)
    assert res == {"dataset": "vital-sentinel-regression", "added": 1, "already_present": 1}
    inputs, outputs, meta = client.created[0]
    assert inputs == [{"turns": ["y"]}] and meta[0]["failed_keys"] == ["route_accuracy"] and "promoted_at" in meta[0]


@pytest.mark.parametrize("name", ["vital-sentinel-heldout", "vital-sentinel-eval", "anything-else"])
def test_promote_refuses_any_dataset_but_the_regression_set(name):
    client = FakeClient()
    with pytest.raises(ValueError, match="회귀셋"):
        pf.promote([], client, name)
    assert client.created == [] and client.dataset_created is False


def test_cli_defaults_to_dry_run_and_rejects_heldout_input(tmp_path, capsys, monkeypatch):
    res = tmp_path / "dev.json"
    res.write_text(json.dumps({"rows": [row(["z"], numbers_grounded=0.0)]}))
    monkeypatch.setattr(pf, "promote", lambda *a, **k: pytest.fail("드라이런이 LangSmith 에 기록하려 했다"))
    assert pf.main(["--results", str(res)]) == 0
    out = capsys.readouterr().out
    assert "DRY-RUN" in out and "['z']" in out and "기록하지 않았습니다" in out
    held = tmp_path / "heldout_eval_x.json"
    held.write_text(json.dumps({"rows": []}))
    assert pf.main(["--results", str(held)]) == 2
    assert pf.main(["--results", str(res), "--dataset-name", "vital-sentinel-heldout"]) == 2


# ---------- 홀드아웃 보호 우회 시도 (정규화·포함·내용 기반 판정) ----------
import unicodedata  # noqa: E402

HELD = sorted(pf.heldout_texts())[0]


@pytest.mark.parametrize("variant", [
    lambda s: s + "  ",                                           # 뒤 공백
    lambda s: "  " + s,                                           # 앞 공백
    lambda s: s.replace(" ", "  "),                               # 공백 두 배
    lambda s: s.replace(" ", "\t"),                               # 탭
    lambda s: s.replace(" ", "\n"),                               # 개행
    lambda s: s.replace(" ", ""),                                 # 공백 제거
    lambda s: unicodedata.normalize("NFD", s),                    # 한글 자모 분해형
    lambda s: s.upper() + "?!",                                   # 대소문자/구두점
    lambda s: "혹시 " + s,                                         # 접두사 덧붙이기
    lambda s: s + " 부탁해요 감사합니다",                            # 접미사 덧붙이기
    lambda s: s[:-1],                                             # 끝 한 글자 잘림
    lambda s: s.replace("U", "Ｕ"),                                # 전각 문자
])
def test_heldout_question_variants_are_never_promoted(variant):
    msg = variant(HELD)
    picked, skipped = pf.select_failures([row([msg], route_accuracy=0.0)], pf.heldout_texts())
    assert picked == [] and skipped and skipped[0]["reason"] == "heldout_overlap", repr(msg)


def test_unrelated_questions_are_still_promoted():
    picked, _ = pf.select_failures([row(["완전히 다른 질문: 오늘 점심 뭐 먹지"], route_accuracy=0.0)], pf.heldout_texts())
    assert len(picked) == 1


def test_renamed_heldout_result_file_is_rejected_by_content(tmp_path, capsys):
    held_rows = [{"example": {"id": "H01", "message": HELD}, "result": {}, "grade": {}}]
    f = tmp_path / "totally_innocent_name.json"
    f.write_text(json.dumps({"rows": held_rows}))
    assert pf.main(["--results", str(f)]) == 2
    assert "홀드아웃 결과 형식" in capsys.readouterr().err
    dev_format = tmp_path / "also_innocent.json"
    dev_format.write_text(json.dumps({"rows": [row([HELD], numbers_grounded=0.0), row([HELD.replace(" ", "")], route_accuracy=0.0), row(["다른 질문"], route_accuracy=0.0)]}))
    assert pf.main(["--results", str(dev_format)]) == 2 and "절반 이상" in capsys.readouterr().err


def test_dev_file_with_a_stray_heldout_question_skips_only_that_row(tmp_path, capsys):
    f = tmp_path / "dev.json"
    f.write_text(json.dumps({"rows": [row(["질문 하나"], numbers_grounded=0.0), row(["질문 둘"]), row(["질문 셋"]), row([HELD + " 요"], route_accuracy=0.0)]}))
    assert pf.main(["--results", str(f)]) == 0
    out = capsys.readouterr().out
    assert "승격 1건" in out and "제외(heldout_overlap)" in out


@pytest.mark.parametrize("name", ["Vital-Sentinel-HELDOUT", "vital-sentinel-heldout ", "vital‑sentinel‑heldout", "VITAL-SENTINEL-EVAL", "vital-sentinel-regression ", "Vital-Sentinel-Regression", "held-out"])
def test_dataset_name_variants_are_refused(name):
    client = FakeClient()
    with pytest.raises(ValueError):
        pf.promote([], client, name)
    assert client.created == [] and not client.dataset_created
    assert pf.main(["--dataset-name", name]) == 2


def test_refusal_message_names_what_was_targeted():
    assert "홀드아웃" in pf.dataset_refusal("Vital-Sentinel-HELDOUT") and "개발셋" in pf.dataset_refusal("VITAL-SENTINEL-EVAL")


def test_fingerprint_is_stable_and_whitespace_insensitive():
    assert pf.heldout_fingerprint() == pf.heldout_fingerprint() and len(pf.heldout_fingerprint()) == 16
    assert pf.normalize("U05  최근\t7일") == pf.normalize("u05최근 7일") == pf.normalize(unicodedata.normalize("NFD", "U05 최근 7일"))


# ---------- API 키: .env / 환경변수 ----------
def test_apply_without_api_key_exits_2_and_never_prints_a_key(tmp_path, capsys, monkeypatch):
    res = tmp_path / "dev.json"
    res.write_text(json.dumps({"rows": [row(["z"], numbers_grounded=0.0)]}))
    for var in ("LANGCHAIN_API_KEY", "LANGSMITH_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(pf, "ROOT", tmp_path)                       # .env 없는 폴더
    monkeypatch.setattr(pf, "promote", lambda *a, **k: pytest.fail("키 없이 LangSmith 에 기록하려 했다"))
    assert pf.main(["--results", str(res), "--apply"]) == 2
    err = capsys.readouterr().err
    assert "LANGCHAIN_API_KEY" in err and ".env" in err


def test_api_key_is_read_from_dotenv_without_echoing_it(tmp_path, capsys, monkeypatch):
    for var in ("LANGCHAIN_API_KEY", "LANGSMITH_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    (tmp_path / ".env").write_text("LANGCHAIN_API_KEY=fake-test-value\n")
    monkeypatch.setattr(pf, "ROOT", tmp_path)
    assert pf.load_api_key() is True
    assert "fake-test-value" not in capsys.readouterr().out
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)           # load_dotenv 가 환경에 넣은 값 정리


def test_langsmith_eval_without_key_points_to_offline(tmp_path, monkeypatch):
    from eval import langsmith_eval as ls
    for var in ("LANGCHAIN_API_KEY", "LANGSMITH_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(ls, "ROOT", tmp_path)
    with pytest.raises(SystemExit, match="--offline"):
        ls.load_langsmith_env()

"""개발 평가에서 실패한 예제를 별도 LangSmith 데이터셋(회귀셋)으로 승격한다.

책임 경계:
    개발 평가 결과(eval/results/agent_eval.json: 예제별 입력·정답·점수)에서 *실패한* 예제를 골라 `vital-sentinel-regression` 데이터셋에 추가한다.
    평가 실행·채점은 하지 않는다(`langsmith_eval.py` 의 몫). 회귀셋은 고쳐진 버그가 다시 나타나지 않는지 보는 용도이며, 개발 중에 보인 질문이다.

블라인드 홀드아웃 보호 (하드 규칙, 파일 이름이 아니라 *내용*으로 판정한다):
    - 대상 데이터셋 이름은 `vital-sentinel-regression` 하나만 허용한다. 이름은 NFKC·casefold·공백 정리 후 비교하므로 대소문자·공백·호환 문자
      변형("Vital-Sentinel-Heldout ", 전각 문자 등)으로 홀드아웃/개발셋을 가리킬 수 없다. 이름에 held-out 표지가 있으면 사유를 밝히며 거부한다.
    - 질문 비교는 `normalize()`(NFKC, casefold, 공백 정리, 구두점 제거)한 문자열로 한다. 홀드아웃 질문과 같거나 *한쪽이 다른 쪽을 포함*하거나
      거의 같은(유사도 0.9 이상) 질문은 승격하지 않는다 (공백·유니코드 NFD·접두/접미 덧붙이기 우회 방지).
    - 입력 결과 파일이 홀드아웃 결과 형식이거나(행에 'example' 키) 행의 절반 이상이 홀드아웃 질문이면 파일 전체를 거부한다 (이름을 바꿔도 소용없다).
    홀드아웃에 실패 사례가 있어도 그것을 개발/회귀셋에 옮기면 더는 블라인드가 아니기 때문이다 - 고치려면 일반화된 새 예제를 만들어 쓴다.

실패 정의: 정답 지표(route_accuracy, expected_types_recall, escalation_correct, validator_passed, no_diagnosis_assertion, disclaimer_present,
numbers_grounded)가 하나라도 1.0 미만. 지연(latency_s)과 1차 통과(validator_first_try)는 실패로 보지 않는다.

실행: `.venv/bin/python -m eval.promote_failures` 는 *드라이런*(기본): 무엇을 추가할지 출력만 한다. 실제 기록은 `--apply` (LANGCHAIN_API_KEY 필요).
"""
import argparse
import difflib
import hashlib
import json
import os
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = Path(__file__).parent / "results"
HELDOUT_EXAMPLES = Path(__file__).parent / "heldout_examples.json"
TARGET_DATASET = "vital-sentinel-regression"
FORBIDDEN_DATASETS = ("vital-sentinel-heldout", "vital-sentinel-eval")
CORRECTNESS_KEYS = ("route_accuracy", "expected_types_recall", "escalation_correct", "validator_passed",
                    "no_diagnosis_assertion", "disclaimer_present", "numbers_grounded")


def example_key(inputs: dict) -> str:
    """입력(대화 턴)으로 예제를 식별하는 안정적 키 - 중복 승격 방지와 홀드아웃 대조에 쓴다."""
    return json.dumps(inputs, sort_keys=True, ensure_ascii=False)


def normalize(text: str) -> str:
    """질문 비교용 정규화: NFKC(호환/분해형 한글 합침) -> casefold -> 문자·숫자 외 제거 -> 공백 없앰.

    공백·구두점·전각/반각·한글 NFD 분해 같은 겉모습 변형이 비교 결과를 바꾸지 못하게 한다.
    """
    s = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[\W_]+", "", s)


def heldout_texts() -> set[str]:
    """홀드아웃 질문 문장 집합 (회귀셋으로 새어 들어가면 안 되는 문장들, 정규화 전 원문)."""
    return {e["message"].strip() for e in json.loads(HELDOUT_EXAMPLES.read_text(encoding="utf-8"))}


def heldout_fingerprint() -> str:
    """정규화한 홀드아웃 질문들의 sha256 (보고서에 남겨 어떤 홀드아웃 버전으로 막았는지 알 수 있게 한다)."""
    return hashlib.sha256("\n".join(sorted(normalize(m) for m in heldout_texts())).encode()).hexdigest()[:16]


def overlaps_heldout(question: str, banned: set[str], threshold: float = 0.9, min_len: int = 6) -> bool:
    """question 이 홀드아웃 질문과 겹치는가: 정규화 후 같음 / 포함(어느 쪽이든) / 유사도 threshold 이상.

    Args:
        question: 검사할 질문. banned: 홀드아웃 원문 집합. threshold: difflib 유사도 기준. min_len: 포함 검사를 하는 최소 길이(너무 짧은 질문이 우연히 포함되는 것 방지).
    """
    q = normalize(question)
    for b in map(normalize, banned):
        if q == b or (len(q) >= min_len and (b in q or q in b)) or difflib.SequenceMatcher(None, q, b).ratio() >= threshold:
            return True
    return False


def failed_keys(scores: dict) -> list[str]:
    """한 예제에서 1.0 미만인 정답 지표 이름들 (점수가 None 이면 해당 없음으로 보고 건너뜀)."""
    return [k for k in CORRECTNESS_KEYS if isinstance(scores.get(k), (int, float)) and scores[k] < 1.0]


def select_failures(rows: list[dict], banned_texts: set[str]) -> tuple[list[dict], list[dict]]:
    """결과 행에서 승격 대상과 제외된 행을 가른다.

    Args:
        rows: `agent_eval.json` 의 rows (각 행에 inputs, reference, outputs, scores).
        banned_texts: 승격 금지 문장(홀드아웃 질문).

    Returns:
        (승격 대상 [{"inputs","outputs","metadata"}], 제외 [{"inputs","reason"}]). 제외 사유: 'heldout_overlap'.
    """
    picked, skipped = [], []
    for r in rows:
        bad = failed_keys(r.get("scores", {}))
        if not bad:
            continue
        if any(overlaps_heldout(t, banned_texts) for t in r["inputs"].get("turns", [])):
            skipped.append({"inputs": r["inputs"], "reason": "heldout_overlap"})
            continue
        picked.append({"inputs": r["inputs"], "outputs": r["reference"],
                       "metadata": {"failed_keys": bad, "kind": r.get("kind"), "scores": {k: r["scores"][k] for k in bad}}})
    return picked, skipped


def load_api_key() -> bool:
    """레포 루트 `.env`(python-dotenv)와 환경변수에서 LangSmith API 키가 있는지 확인한다 (값은 출력하지 않는다). 있으면 True."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    return bool(os.environ.get("LANGCHAIN_API_KEY") or os.environ.get("LANGSMITH_API_KEY"))


def dataset_refusal(name: str) -> str:
    """대상 데이터셋 이름이 회귀셋이 아닐 때의 거부 사유 (정규화 후 홀드아웃/개발셋을 가리키면 그렇게 밝힌다)."""
    n = normalize(name)
    kind = ("홀드아웃" if "heldout" in n or normalize("held-out") in n else
            "개발셋" if n in {normalize(x) for x in FORBIDDEN_DATASETS} else "알 수 없는")
    return f"회귀셋({TARGET_DATASET})에만 쓸 수 있습니다: {name!r} 는 {kind} 데이터셋 이름입니다"


def promote(picked: list[dict], client, dataset: str = TARGET_DATASET) -> dict:
    """승격 대상을 LangSmith 회귀 데이터셋에 추가한다 (이미 있는 입력은 건너뛴다).

    Args:
        picked: `select_failures` 의 승격 대상. client: langsmith.Client 호환 객체. dataset: 대상 데이터셋 이름.

    Returns:
        {"dataset", "added", "already_present"}.

    Raises:
        ValueError: dataset 이 허용된 회귀셋이 아닐 때 (홀드아웃·개발셋 보호).
    """
    if dataset != TARGET_DATASET:
        raise ValueError(dataset_refusal(dataset))
    if client.has_dataset(dataset_name=dataset):
        ds = client.read_dataset(dataset_name=dataset)
        existing = {example_key(e.inputs) for e in client.list_examples(dataset_id=ds.id)}
    else:
        ds = client.create_dataset(dataset, description="개발 평가에서 실패했던 예제 모음 - 고친 결함이 다시 나타나는지 보는 회귀셋 (블라인드 홀드아웃과 분리)")
        existing = set()
    new = [p for p in picked if example_key(p["inputs"]) not in existing]
    if new:
        client.create_examples(dataset_id=ds.id, inputs=[p["inputs"] for p in new], outputs=[p["outputs"] for p in new],
                               metadata=[{**p["metadata"], "promoted_at": datetime.now().isoformat(timespec="seconds")} for p in new])
    return {"dataset": dataset, "added": len(new), "already_present": len(picked) - len(new)}


def heldout_input_reason(rows: list[dict], banned: set[str]) -> str:
    """결과 파일이 홀드아웃에서 온 것이면 거부 사유, 아니면 빈 문자열. 내용으로만 판정한다.

    홀드아웃 결과 형식(행에 'example')이거나, 행의 절반 이상의 질문이 홀드아웃 질문과 겹치면 홀드아웃 파일로 본다.
    """
    if any(isinstance(r, dict) and "example" in r for r in rows):
        return "홀드아웃 결과 형식(행에 'example' 키)의 파일은 승격 입력으로 쓸 수 없습니다"
    turns = [t for r in rows for t in r.get("inputs", {}).get("turns", [])]
    if turns and sum(overlaps_heldout(t, banned) for t in turns) * 2 >= len(turns):
        return "질문의 절반 이상이 홀드아웃 질문과 겹치는 파일은 승격 입력으로 쓸 수 없습니다"
    return ""


def main(argv: list[str] | None = None) -> int:
    """CLI. 기본은 드라이런(출력만). 종료 코드 0=정상, 2=입력 거부/키 없음."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=str(RESULTS / "agent_eval.json"), help="개발 평가 결과 JSON (홀드아웃 결과는 거부)")
    ap.add_argument("--dataset-name", default=TARGET_DATASET)
    ap.add_argument("--apply", action="store_true", help="실제로 LangSmith 에 기록한다 (기본: 드라이런)")
    args = ap.parse_args(argv)
    path = Path(args.results)
    if args.dataset_name != TARGET_DATASET:
        print(dataset_refusal(args.dataset_name), file=sys.stderr)
        return 2
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    reason = heldout_input_reason(rows, heldout_texts())
    if not reason and "heldout" in path.name.casefold():          # 추가 방어선일 뿐 - 이름을 바꾸면 우회되므로 판정의 근거는 위의 내용 검사다
        reason = "파일 이름에 heldout 가 들어 있습니다"
    if reason:
        print(f"{reason} (블라인드 보호)", file=sys.stderr)
        return 2
    picked, skipped = select_failures(rows, heldout_texts())
    print(f"[{'APPLY' if args.apply else 'DRY-RUN'}] 결과 {len(rows)}건 중 실패 {len(picked) + len(skipped)}건 -> 승격 {len(picked)}건, 홀드아웃과 겹쳐 제외 {len(skipped)}건")
    for p in picked:
        print(f"  + {p['inputs']['turns']} <- {p['metadata']['failed_keys']}")
    for s in skipped:
        print(f"  - 제외({s['reason']}): {s['inputs']['turns']}")
    if not args.apply:
        print("드라이런이라 기록하지 않았습니다. 실제 기록은 --apply.")
        return 0
    if not load_api_key():
        print("LangSmith API 키(LANGCHAIN_API_KEY)가 없어 기록할 수 없습니다. 레포 루트 .env 또는 환경변수에 설정하세요 (--apply 없이 실행하면 드라이런으로 목록만 봅니다).", file=sys.stderr)
        return 2
    from langsmith import Client

    print(promote(picked, Client(), args.dataset_name))
    return 0


if __name__ == "__main__":
    sys.exit(main())

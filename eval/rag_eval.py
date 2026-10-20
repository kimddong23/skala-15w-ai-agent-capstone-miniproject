"""가이드라인 검색 Before/After: 기존 키워드 섹션 매칭 vs RAG(bge-m3 임베딩).

책임 경계:
    라벨이 붙은 질문 세트(eval/rag_questions.json: 질문 -> 정답 문서 id 목록)로 검색 정확도·지연·비용을 잰다.
    검색 구현 자체는 `app/rag/`, 기존 키워드 방식은 `app/agent/report.guideline_excerpt` (읽기 전용으로 import)를 그대로 쓴다.

비교 대상 (모두 문서 단위로 순위를 매긴다):
    A. 기존: 원래 에이전트가 쓰던 2개 문서(heart-rate-ranges, red-flags)에 `guideline_excerpt` 적용.
    B. 키워드 확장: 같은 `guideline_excerpt` 를 확장된 코퍼스 전체에 적용 (코퍼스 확장 효과와 검색 방식 효과를 분리하기 위한 중간 대조군).
    C. RAG: `GuidelineSearch` 벡터 검색 (bge-m3).
    D. RAG 폴백: Ollama 를 쓸 수 없을 때의 키워드 폴백 (`keyword_scores`) - 장애 시 품질.

지표: hit@1, hit@3 (상위 k 개 문서 안에 정답 문서가 *하나라도* 있으면 1 - 정답이 여러 개일 수 있어 정확한 의미의 recall@k 가 아니므로
hit@k 라고 부른다), MRR (첫 정답 순위의 역수, 상위 10위 밖은 0), 지연 p50/p95/평균(ms), 인덱스 구축 비용(콜드/웜), 질의 임베딩 시간, 인덱스 크기.
주의 1: 방식 A 는 후보 문서가 2개뿐이라 hit@3 이 (정답이 그 2개 안에 있는 질문에서는) 순위와 무관하게 자명하게 1이 된다 -> A 의 hit@3 은 비교 지표가 아니다(표에 후보 문서 수 병기).
주의 2: 키워드 방식에서 질문과 겹치는 단서가 전혀 없으면 모든 점수가 0 이고 순위는 문서 id 순(= 파일 순)으로 정해진다 - 관련성이 아니라 임의 순서다.

주의 (한계): 질문과 코퍼스를 같은 사람이 썼다 - 라벨러가 정답을 알고 질문을 만든 편향이 있어 절대 수치는 낙관적이다.
A 는 정답 문서가 원 코퍼스에 없는 질문이 대부분이라 '검색 방식'이 아니라 '지식 범위' 차이가 섞인다 (answerable 부분집합을 따로 보고).

실행: `.venv/bin/python -m eval.rag_eval` (Ollama 에 bge-m3 필요). `--render-only` 는 Ollama 없이 기존 json 에서 md 만 다시 만든다 -> eval/results/rag_eval.{md,json}
"""
import json
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.report import guideline_excerpt  # noqa: E402
from app.rag.chunking import chunk_corpus  # noqa: E402
from app.rag.corpus import GUIDELINES_DIR, load_corpus  # noqa: E402
from app.rag.embeddings import EmbeddingError, OllamaEmbedder  # noqa: E402
from app.rag.search import GuidelineSearch, keyword_scores  # noqa: E402

QUESTIONS = Path(__file__).parent / "rag_questions.json"
RESULTS = Path(__file__).parent / "results"
OLD_DOCS = ("heart-rate-ranges", "red-flags")      # 기존 에이전트가 읽던 가이드라인 (ranges, red_flags)
MAX_RANK = 10


def _sections(doc_ids: set[str] | None) -> list[tuple[str, str]]:
    """코퍼스를 [(문서 id, '## 소제목\\n본문')] 으로 나눈다 (`guideline_excerpt` 가 보는 섹션 단위와 같다). 문서 id 순으로 정렬해 동점 처리를 결정론적으로 한다."""
    out = []
    for d in sorted(load_corpus(GUIDELINES_DIR), key=lambda d: d.id):
        if doc_ids is None or d.id in doc_ids:
            out += [(d.id, "## " + s.strip()) for s in ("\n" + d.body).split("\n## ")[1:] if s.strip()]
    return out


def keyword_baseline(question: str, sections: list[tuple[str, str]], n: int = MAX_RANK) -> list[str]:
    """기존 `guideline_excerpt` 를 반복 호출해 문서 순위를 만든다 (매번 1위 섹션을 제거하고 다시 호출).

    이 함수는 1위 섹션만 돌려주므로 2~n 위는 '직전 1위를 뺀 나머지의 1위'로 얻는다. 점수 계산은 원 함수 그대로다.
    """
    left, ranked = list(sections), []
    while left and len(ranked) < n:
        excerpt = guideline_excerpt([s for _, s in left], question, limit=10_000)
        idx = next((i for i, (_, s) in enumerate(left) if s.strip() == excerpt.strip()), None)
        if idx is None:
            break
        doc_id = left.pop(idx)[0]
        if doc_id not in ranked:
            ranked.append(doc_id)
    return ranked


def rank_docs(chunk_doc_ids: list[str]) -> list[str]:
    """청크 순위를 문서 순위로 (같은 문서는 처음 등장한 위치만)."""
    seen: list[str] = []
    for d in chunk_doc_ids:
        if d not in seen:
            seen.append(d)
    return seen


def metrics(ranked_per_q: list[list[str]], golds: list[list[str]]) -> dict:
    """hit@1, hit@3, MRR (소수 3자리). hit@k = 상위 k 개 문서 안에 정답 문서가 하나라도 있는 질문의 비율."""
    n = len(golds)
    r1 = sum(bool(set(r[:1]) & set(g)) for r, g in zip(ranked_per_q, golds)) / n
    r3 = sum(bool(set(r[:3]) & set(g)) for r, g in zip(ranked_per_q, golds)) / n
    rr = []
    for r, g in zip(ranked_per_q, golds):
        pos = next((i + 1 for i, d in enumerate(r[:MAX_RANK]) if d in g), None)
        rr.append(1 / pos if pos else 0.0)
    return {"hit@1": round(r1, 3), "hit@3": round(r3, 3), "mrr": round(sum(rr) / n, 3)}


def latency(ms: list[float]) -> dict:
    """지연 요약(ms): p50, p95, mean."""
    s = sorted(ms)
    return {"p50": round(statistics.median(s), 2), "p95": round(s[min(len(s) - 1, int(0.95 * len(s)))], 2),
            "mean": round(statistics.fmean(s), 2)}


def timed(fn, *args):
    """fn(*args) 의 (결과, 걸린 ms)."""
    t0 = time.perf_counter()
    r = fn(*args)
    return r, (time.perf_counter() - t0) * 1000


def evaluate() -> dict:
    """전체 평가를 수행해 JSON 직렬화 가능한 dict 로 돌려준다.

    Raises:
        EmbeddingError: Ollama/bge-m3 를 쓸 수 없을 때 (벡터 검색 평가가 불가능하다).
    """
    qs = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    golds = [q["gold"] for q in qs]
    texts = [q["question"] for q in qs]
    old_sections, all_sections = _sections(set(OLD_DOCS)), _sections(None)
    docs = load_corpus(GUIDELINES_DIR)
    chunks = chunk_corpus(docs)

    # --- 인덱스 구축 비용: 임시 캐시 폴더에서 콜드 -> 웜 ---
    embedder = OllamaEmbedder()
    _, first_embed_ms = timed(embedder.embed, ["워밍업"])             # 모델 콜드 로드 시간 포함 (이미 적재돼 있으면 짧다)
    with tempfile.TemporaryDirectory() as tmp:
        s = GuidelineSearch(embedder=embedder, cache_path=Path(tmp) / "index.npz")
        cold = s.build_index()
        warm = s.build_index()
        index_bytes = cold.cache_bytes

        # --- 질의 평가 ---
        res: dict[str, dict] = {}
        runs = {
            "A_old_keyword": lambda q: keyword_baseline(q, old_sections),
            "B_keyword_expanded": lambda q: keyword_baseline(q, all_sections),
            "C_rag_vector": lambda q: rank_docs([h.doc_id for h in s.search(q, MAX_RANK).hits]),
        }
        ranked: dict[str, list[list[str]]] = {}
        for name, fn in runs.items():
            outs, ms = [], []
            for q in texts:
                r, t = timed(fn, q)
                outs.append(r)
                ms.append(t)
            ranked[name] = outs
            res[name] = {**metrics(outs, golds), "latency_ms": latency(ms)}
        # 질의 임베딩 시간만 따로
        qe = [timed(embedder.embed, [q])[1] for q in texts]
        # --- 폴백: 키워드 검색 (임베딩 불가 시) ---
        fb, fb_ms = [], []
        for q in texts:
            sc, t = timed(keyword_scores, q, s.chunks)
            order = sorted(range(len(sc)), key=lambda i: -sc[i])[:MAX_RANK * 3]
            fb.append(rank_docs([s.chunks[i].doc_id for i in order]))
            fb_ms.append(t)
        ranked["D_rag_keyword_fallback"] = fb
        res["D_rag_keyword_fallback"] = {**metrics(fb, golds), "latency_ms": latency(fb_ms)}

        # --- 실제 장애 시나리오: 접속 불가 임베더로 검색 (폴백 동작과 시간 확인) ---
        dead = GuidelineSearch(embedder=OllamaEmbedder(base_url="http://127.0.0.1:9", timeout_s=2),
                               cache_path=Path(tmp) / "none.npz")
        r1, dead_first_ms = timed(dead.search, texts[0], 3)
        r2, dead_second_ms = timed(dead.search, texts[1], 3)

    answerable = [i for i, g in enumerate(golds) if set(g) & set(OLD_DOCS)]
    sub = {}
    for name, outs in ranked.items():
        sub[name] = metrics([outs[i] for i in answerable], [golds[i] for i in answerable])
    return {
        "n_questions": len(qs), "n_docs": len(docs), "n_chunks": len(chunks), "n_answerable_by_old_corpus": len(answerable),
        "n_candidate_docs": {"A_old_keyword": len(OLD_DOCS), "B_keyword_expanded": len(docs), "C_rag_vector": len(docs), "D_rag_keyword_fallback": len(docs)},
        "n_zero_keyword_questions": sum(max(keyword_scores(q, s.chunks), default=0) == 0 for q in texts),
        "results": res, "results_answerable_subset": sub,
        "index": {"model": embedder.model, "dim": cold.dim, "bytes": index_bytes,
                  "cold_build_s": cold.seconds, "cold_embedded": cold.n_embedded, "warm_build_s": warm.seconds, "warm_embedded": warm.n_embedded,
                  "first_embed_call_ms": round(first_embed_ms, 1), "query_embed_ms": latency(qe)},
        "outage": {"method": r1.method, "note": r1.note, "first_query_ms": round(dead_first_ms, 1),
                   "second_query_ms": round(dead_second_ms, 1), "second_method": r2.method},
        "per_question": [
            {"id": q["id"], "question": q["question"], "gold": q["gold"],
             **{name.split("_")[0]: outs[i][:3] for name, outs in ranked.items()}} for i, q in enumerate(qs)
        ],
    }


def compat(r: dict) -> dict:
    """옛 json(recall@k 이름, 후보 문서 수/무일치 질문 수 없음)을 현재 형식으로 바꾼다 - 측정값은 그대로, 이름과 파생 가능한 필드만 채운다.

    후보 문서 수는 방식 정의에서 정해지고(A=기존 2문서, 나머지=전체 문서), 키워드 단서가 없는 질문 수는 질문·코퍼스만으로 오프라인 계산된다.
    """
    r = json.loads(json.dumps(r))
    for block in ("results", "results_answerable_subset"):
        for m in r[block].values():
            for old, new in (("recall@1", "hit@1"), ("recall@3", "hit@3")):
                if old in m:
                    m[new] = m.pop(old)
    r.setdefault("n_candidate_docs", {"A_old_keyword": len(OLD_DOCS), "B_keyword_expanded": r["n_docs"], "C_rag_vector": r["n_docs"], "D_rag_keyword_fallback": r["n_docs"]})
    if "n_zero_keyword_questions" not in r:
        chunks = chunk_corpus(load_corpus(GUIDELINES_DIR))
        r["n_zero_keyword_questions"] = sum(max(keyword_scores(q["question"], chunks), default=0) == 0 for q in json.loads(QUESTIONS.read_text(encoding="utf-8")))
    return r


def render_only(note: str = "") -> str:
    """Ollama 없이 기존 rag_eval.json 에서 md 를 다시 만든다 (측정은 다시 하지 않는다; 측정 시각은 최초 1회 파일 시각으로 기록해 유지)."""
    path = RESULTS / "rag_eval.json"
    r = compat(json.loads(path.read_text()))
    r.setdefault("measurement", {"measured_at": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"), "rerun": False, "note": note})
    path.write_text(json.dumps(r, ensure_ascii=False, indent=2))
    md = render(r)
    (RESULTS / "rag_eval.md").write_text(md)
    return md


def render(r: dict) -> str:
    """평가 결과를 markdown 으로."""
    res, sub, ix = r["results"], r["results_answerable_subset"], r["index"]
    names = {"A_old_keyword": "A 기존 키워드(2문서)", "B_keyword_expanded": "B 키워드(확장 코퍼스)",
             "C_rag_vector": "C RAG(bge-m3)", "D_rag_keyword_fallback": "D RAG 폴백(키워드)"}
    L = ["# 가이드라인 검색 Before/After: 키워드 섹션 매칭 vs RAG", "",
         f"- 질문 {r['n_questions']}개 (구어체, 오타, 영어 혼용 포함), 코퍼스 {r['n_docs']}개 문서 / {r['n_chunks']}개 청크, 임베딩 {ix['model']} (dim {ix['dim']})",
         "- 적중 = 상위 k 개 문서 안에 정답 문서가 하나라도 있음. MRR = 첫 정답 순위의 역수(10위 밖 0)",
         *([f"- **측정 시각**: {r['measurement']['measured_at']} (KST). 재실행 안 함(GPU 양보) - 이 문서는 그 측정의 json 을 현재 렌더러(hit@k 명명, 후보 문서 수 열, 동점 처리 설명)로 *오프라인 재생성*한 것이다. {r['measurement'].get('note', '')}".rstrip()] if r.get("measurement") else []),
         "- **한계**: 질문과 코퍼스를 같은 작성자가 만들었다(정답을 아는 상태에서 질문을 씀) - 절대 수치는 낙관적이며, 방식 간 상대 비교를 보는 용도다.", "",
         "## 1. 정확도 (전체 질문)", "", "| 방식 | 후보 문서 수 | hit@1 | hit@3 | MRR | 지연 p50 / p95 (ms) |", "|---|---:|---:|---:|---:|---:|"]
    for k, n in names.items():
        m = res[k]
        L.append(f"| {n} | {r['n_candidate_docs'][k]} | {m['hit@1']} | {m['hit@3']} | {m['mrr']} | {m['latency_ms']['p50']} / {m['latency_ms']['p95']} |")
    L += ["", "- hit@k = 상위 k 개 문서 안에 정답 문서가 *하나라도* 있는 질문의 비율 (정답이 여러 문서일 수 있어 recall@k 가 아님).",
          f"- **A 의 hit@3 은 비교 지표가 아니다**: 후보 문서가 {r['n_candidate_docs']['A_old_keyword']}개뿐이라 정답이 그 안에 있는 질문은 순위와 무관하게 상위 3개에 들어간다. A 는 hit@1 과 MRR 로 보라.",
          f"- 키워드 방식은 질문과 겹치는 단서가 전혀 없으면(이번 질문 중 {r['n_zero_keyword_questions']}개) 모든 점수가 0 이고 순위는 문서 id 순이다 - 관련성이 아니라 결정론적 임의 순서다."]
    L += ["", f"## 2. 기존 코퍼스로 답할 수 있는 질문만 ({r['n_answerable_by_old_corpus']}개)", "",
          "A 는 정답 문서가 원 코퍼스(2문서)에 없는 질문에서 구조적으로 못 맞춘다. 이 표는 같은 범위의 질문만 비교해 '검색 방식'의 차이를 본다.", "",
          "| 방식 | hit@1 | hit@3 | MRR |", "|---|---:|---:|---:|"]
    L += [f"| {n} | {sub[k]['hit@1']} | {sub[k]['hit@3']} | {sub[k]['mrr']} |" for k, n in names.items()]
    L += ["", "## 3. 비용", "",
          "| 항목 | 값 |", "|---|---|",
          f"| 인덱스 크기 | {ix['bytes']:,} bytes ({r['n_chunks']}청크 x {ix['dim']}차원 float32 + 해시) |",
          f"| 콜드 구축 | {ix['cold_build_s']} s (임베딩 {ix['cold_embedded']}건) |",
          f"| 웜 재구축(캐시) | {ix['warm_build_s']} s (임베딩 {ix['warm_embedded']}건) |",
          f"| 질의 임베딩 p50 / p95 | {ix['query_embed_ms']['p50']} / {ix['query_embed_ms']['p95']} ms |",
          f"| 첫 임베딩 호출(모델 로드 포함) | {ix['first_embed_call_ms']} ms |",
          f"| 키워드 방식 지연(p50) | {res['B_keyword_expanded']['latency_ms']['p50']} ms (기존 함수 반복 호출) / {res['D_rag_keyword_fallback']['latency_ms']['p50']} ms (RAG 폴백 구현), 임베딩·외부 의존 없음 |",
          f"| RAG 질의 지연(p50) | {res['C_rag_vector']['latency_ms']['p50']} ms (질의 임베딩 포함) |", "",
          "## 4. 장애 시 동작 (Ollama 접속 불가)", "",
          f"- 접속 불가 임베더로 검색: method=`{r['outage']['method']}`, 첫 질의 {r['outage']['first_query_ms']} ms (연결 거부를 기다린 시간), "
          f"이후 질의 {r['outage']['second_query_ms']} ms (실패 후 60초 동안 재시도하지 않아 대기 없음). 사유: {r['outage']['note']}",
          f"- 폴백(키워드)의 정확도는 위 표 D: hit@1 {res['D_rag_keyword_fallback']['hit@1']}, hit@3 {res['D_rag_keyword_fallback']['hit@3']}.", ""]
    miss_c = [q for q in r["per_question"] if not set(q["C"][:3]) & set(q["gold"])]
    miss_b = [q for q in r["per_question"] if not set(q["B"][:3]) & set(q["gold"])]
    L += ["## 5. 실패 사례", "", f"### RAG(C) hit@3 실패 {len(miss_c)}건", ""]
    L += [f"- {q['id']} 「{q['question']}」 정답 {q['gold']} / 상위3 {q['C']}" for q in miss_c] or ["- 없음"]
    L += ["", f"### 키워드(B) hit@3 실패 {len(miss_b)}건", ""]
    L += [f"- {q['id']} 「{q['question']}」 정답 {q['gold']} / 상위3 {q['B']}" for q in miss_b[:15]] or ["- 없음"]
    if len(miss_b) > 15:
        L.append(f"- ... 외 {len(miss_b) - 15}건 (전체는 JSON)")
    L += ["", "## 6. 트레이드오프와 한계", "",
          "- **정확도**: 임베딩 검색은 조사·어미·동의어·오타에 강하다(표 1, 5). 키워드는 질문 어절이 문서에 그대로 있어야 맞춘다.",
          f"- **지연**: 질의마다 임베딩 호출(로컬 Ollama)이 들어가 키워드({res['B_keyword_expanded']['latency_ms']['p50']} ms)보다 "
          f"약 {res['C_rag_vector']['latency_ms']['p50'] / max(res['B_keyword_expanded']['latency_ms']['p50'], 0.01):.0f}배 느리다"
          f"({res['C_rag_vector']['latency_ms']['p50']} ms). 그래도 LLM 생성(수 초)에 비하면 작다. 서버가 응답하지 않고 멈춘 경우는 "
          "연결 거부와 달리 임베딩 타임아웃(기본 20초)만큼 기다린 뒤 키워드로 내려간다(표 4는 즉시 거부된 경우).",
          "- **A 와 B 의 차이는 검색 방식이 아니라 코퍼스 확장 효과**다: 같은 함수를 2문서와 19문서에 적용했을 뿐이다. 검색 방식의 효과는 표 2(같은 범위 질문)와 B-C 비교로 본다.",
          "- **의존성**: Ollama + bge-m3(1.2 GB) 가 필요하다. 꺼져 있으면 키워드 폴백으로 내려가며 정확도가 떨어진다(표 D). 폴백은 서비스 중단을 막는 용도다.",
          "- **비용 구조**: 인덱스는 코퍼스 변경 시 바뀐 청크만 다시 임베딩하고(웜 0건), 질의 임베딩은 캐시하지 않는다.",
          "- **평가의 한계**: 질문 45개는 작고, 작성자 편향이 있으며, 정답이 여러 문서에 걸치는 질문은 복수 정답으로 관대하게 채점했다. "
          "검색 결과가 실제 답변 품질(환각 감소)로 이어지는지는 이 평가가 재지 않는다.",
          "- **코퍼스**: 출처는 '자체 작성한 일반 상식 요약(특정 기관 문서를 인용·대조하지 않음)'이며 임상 검토를 받지 않았다. 일부 수치는 대략적인 값이고 검증하지 않았다(각 문서 끝 `문서 정보` 참고)."]
    return "\n".join(L)


if __name__ == "__main__":
    if "--render-only" in sys.argv:
        print(render_only(" ".join(a for a in sys.argv[1:] if a != "--render-only")))
        sys.exit(0)
    try:
        out = evaluate()
    except EmbeddingError as e:
        sys.exit(f"Ollama 임베딩을 쓸 수 없어 평가를 중단합니다: {e}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "rag_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    (RESULTS / "rag_eval.md").write_text(render(out))
    print(render(out))

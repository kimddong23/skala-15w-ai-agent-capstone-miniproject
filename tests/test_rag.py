"""가이드라인 RAG 테스트 (Ollama 없이 결정론적 가짜 임베더로).

범위: 코퍼스 로딩(front-matter), heading-aware 청킹, 임베딩 캐시 멱등성/부분 갱신, 벡터 검색 순위, Ollama 불가 시 키워드 폴백과
재시도 억제, MCP 도구 `search_guidelines` 의 출력 형태. 실제 bge-m3 정확도는 `eval/rag_eval.py` 가 잰다(테스트는 외부 서비스에 의존하지 않는다).
협력: `app.rag.*`, `app.mcp_server.server`(fastmcp 인메모리 Client).
"""
import asyncio
import zlib
from pathlib import Path

import numpy as np
import pytest
from fastmcp import Client

from app.rag import search as rag_search
from app.rag.chunking import MAX_CHARS, Chunk, chunk_doc
from app.rag.corpus import GUIDELINES_DIR, Doc, load_corpus, parse_front_matter
from app.rag.embeddings import EmbeddingError, OllamaEmbedder
from app.rag.index import VectorIndex
from app.rag.search import GuidelineSearch, keyword_scores
from tests._prepared import ensure_prepared

ensure_prepared()
from app.mcp_server.server import mcp  # noqa: E402


class FakeEmbedder:
    """글자 2-gram 해싱 임베더: 겹치는 글자가 많은 텍스트끼리 코사인이 높다. 호출 횟수와 임베딩한 텍스트 수를 센다."""

    def __init__(self, model: str = "fake", fail: bool = False, fail_first: int = 0) -> None:
        self.model, self.fail, self.fail_first, self.calls, self.texts = model, fail, fail_first, 0, 0

    def embed(self, texts: list[str]) -> np.ndarray:
        """texts 를 글자 2-gram 해시 벡터(L2 정규화)로 바꾼다. fail=True 면 EmbeddingError."""
        self.calls += 1
        if self.fail or self.calls <= self.fail_first:            # fail_first=N: 처음 N번 호출만 실패 (색인 구축은 실패, 이후 질의는 성공)
            raise EmbeddingError("ConnectError: refused")
        self.texts += len(texts)
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, t in enumerate(texts):
            s = t.replace(" ", "")
            for a, b in zip(s, s[1:]):
                out[i, zlib.crc32((a + b).encode()) % 256] += 1       # 프로세스마다 달라지는 hash() 대신 결정론적 해시
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norm == 0, 1, norm)


DOCS = {
    "caffeine.md": "---\nid: caffeine\ntitle: 카페인과 심박\ntopics: [카페인, 커피]\nsource: 자체 작성 요약 (테스트)\n---\n# 카페인과 심박\n\n"
                   "## 카페인\n커피와 에너지 음료의 카페인은 심박수를 올리고 두근거림을 일으킬 수 있습니다.\n\n## 팁\n늦은 오후에는 커피를 피하세요.\n",
    "sleep.md": "---\nid: sleep\ntitle: 수면 중 심박\ntopics: [수면]\nsource: 자체 작성 요약 (테스트)\n---\n# 수면 중 심박\n\n"
                "## 수면\n잠을 자는 동안에는 심박수가 낮아지는 것이 정상이며 깊은 수면에서 가장 낮습니다.\n",
    "noheader.md": "# 헤더 없는 문서\n\n## 내용\n front-matter 가 없는 문서는 파일명이 id 가 됩니다. 이 문장은 청크로 만들기에 충분히 깁니다.\n",
}


@pytest.fixture()
def corpus_dir(tmp_path: Path) -> Path:
    """임시 폴더에 작은 테스트 코퍼스(문서 3개)를 만든다."""
    for name, text in DOCS.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


def make_search(corpus_dir: Path, tmp_path: Path, embedder=None) -> GuidelineSearch:
    """테스트용 검색기: 임시 캐시 경로와 (기본) 가짜 임베더."""
    return GuidelineSearch(corpus_dir, embedder or FakeEmbedder(), tmp_path / "cache" / "index.npz")


# ---------- 코퍼스 ----------
def test_front_matter_parsing_and_default_ids(corpus_dir):
    meta, body = parse_front_matter(DOCS["caffeine.md"])
    assert meta["id"] == "caffeine" and body.lstrip().startswith("# 카페인과 심박")
    docs = {d.id: d for d in load_corpus(corpus_dir)}
    assert set(docs) == {"caffeine", "sleep", "noheader"}
    assert docs["caffeine"].topics == ("카페인", "커피") and "테스트" in docs["caffeine"].source
    assert docs["noheader"].title == "헤더 없는 문서"          # front-matter 없는 문서: 파일명 id, 첫 # 제목


def test_duplicate_ids_are_rejected(tmp_path):
    for n in ("a.md", "b.md"):
        (tmp_path / n).write_text("---\nid: same\ntitle: t\n---\n## s\n본문입니다.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="중복"):
        load_corpus(tmp_path)


def test_service_corpus_integrity():
    """실제 코퍼스: 12개 이상, id 유일, 기존 3문서 포함, 새 문서는 front-matter 와 출처 표기를 가진다."""
    docs = load_corpus(GUIDELINES_DIR)
    ids = [d.id for d in docs]
    assert len(docs) >= 12 and len(set(ids)) == len(ids)
    assert {"heart-rate-ranges", "red-flags", "disclaimer"} <= set(ids)
    for d in docs:
        if d.id not in ("heart-rate-ranges", "red-flags", "disclaimer"):
            assert d.topics and d.title and ("자체 작성" in d.source or "서비스 자체 설명" in d.source), d.id
            assert "대조하지 않음" in d.source or "인용하지 않음" in d.source, d.id          # 특정 기관 문서를 근거로 한 것처럼 읽히는 출처 표기 금지


# ---------- 청킹 ----------
def test_chunks_follow_headings_and_carry_title_prefix(corpus_dir):
    doc = next(d for d in load_corpus(corpus_dir) if d.id == "caffeine")
    chunks = chunk_doc(doc, min_chars=10)
    assert [c.heading for c in chunks] == ["카페인", "팁"]
    assert all(c.embed_text.startswith("카페인과 심박 > ") and c.chunk_id.startswith("caffeine#") for c in chunks)
    assert chunks[0].text.startswith("커피와 에너지 음료")


def test_short_sections_merge_into_previous_chunk_and_long_ones_split(tmp_path):
    long_para = " ".join(f"문장 번호 {i} 은 청킹 길이 제한을 확인하기 위한 충분히 긴 문장입니다." for i in range(30))
    doc = Doc("d", "제목", f"## 긴 구간\n{long_para}\n\n## 짧음\n짧은 구간입니다.\n", (), "x")
    chunks = chunk_doc(doc)
    assert len(chunks) >= 3 and all(len(c.text) <= MAX_CHARS + 80 for c in chunks[:-1])
    assert "짧은 구간입니다" in chunks[-1].text and chunks[-1].heading == "긴 구간"      # 짧은 마지막 구간은 앞 청크에 합쳐짐


def test_content_hash_is_stable_and_changes_with_text():
    a = Chunk("d#0", "d", "t", "h", "본문", "t > h\n본문")
    assert a.content_hash == Chunk("d#0", "d", "t", "h", "본문", "t > h\n본문").content_hash
    assert a.content_hash != Chunk("d#0", "d", "t", "h", "본문2", "t > h\n본문2").content_hash


def test_real_corpus_chunk_sizes_are_in_target_range():
    docs = load_corpus(GUIDELINES_DIR)
    from app.rag.chunking import chunk_corpus
    chunks = chunk_corpus(docs)
    assert 30 <= len(chunks) <= 200
    assert max(len(c.text) for c in chunks) <= MAX_CHARS + 120      # 한 문장이 한도를 넘는 경우만 소폭 초과


# ---------- 캐시 멱등성 ----------
def test_rebuild_is_idempotent_and_only_changed_chunks_are_reembedded(corpus_dir, tmp_path):
    emb = FakeEmbedder()
    s = make_search(corpus_dir, tmp_path, emb)
    first = s.build_index()
    assert first.n_embedded == first.n_chunks > 0 and first.cache_bytes > 0 and first.dim == 256
    assert s.build_index().n_embedded == 0 and emb.texts == first.n_chunks              # 재실행: 임베딩 0건
    (corpus_dir / "sleep.md").write_text(DOCS["sleep.md"].replace("가장 낮습니다", "가장 낮게 나타납니다"), encoding="utf-8")
    s2 = make_search(corpus_dir, tmp_path, emb)
    again = s2.build_index()
    assert again.n_embedded == 1 and again.n_cached == again.n_chunks - 1               # 바뀐 청크 1개만 다시 임베딩


def test_load_cached_needs_complete_cache_and_matching_model(corpus_dir, tmp_path):
    s = make_search(corpus_dir, tmp_path)
    s.build_index()
    assert VectorIndex.load_cached(s.chunks, s.cache_path, "fake") is not None
    assert VectorIndex.load_cached(s.chunks, s.cache_path, "other-model") is None        # 모델이 다르면 캐시 무효
    changed = [Chunk(c.chunk_id, c.doc_id, c.title, c.heading, c.text + "!", c.embed_text + "!") for c in s.chunks]
    assert VectorIndex.load_cached(changed, s.cache_path, "fake") is None                # 내용이 바뀌면 재임베딩 필요
    assert VectorIndex.load_cached(s.chunks, tmp_path / "missing.npz", "fake") is None   # 캐시 파일 없음


def test_fresh_process_uses_cache_without_calling_the_embedder_for_the_index(corpus_dir, tmp_path):
    make_search(corpus_dir, tmp_path).build_index()
    emb = FakeEmbedder()
    fresh = make_search(corpus_dir, tmp_path, emb)
    r = fresh.search("커피 마시면 심박이 올라가나요", 2)
    assert r.method == "vector" and emb.texts == 1                                       # 질의 1건만 임베딩 (인덱스는 캐시에서)


# ---------- 검색 ----------
def test_vector_search_ranks_the_relevant_doc_first(corpus_dir, tmp_path):
    s = make_search(corpus_dir, tmp_path)
    r = s.search("커피와 카페인이 심박수를 올리나요", 3)
    assert r.method == "vector" and r.note == "" and r.hits[0].doc_id == "caffeine"
    assert [h.score for h in r.hits] == sorted((h.score for h in r.hits), reverse=True)
    assert s.search("잠잘 때 심박수가 낮아지나요", 1).hits[0].doc_id == "sleep"


def test_keyword_scores_use_word_prefix_cues():
    chunks = [Chunk("a#0", "a", "카페인", "", "x", "카페인 커피 심박"), Chunk("b#0", "b", "수면", "", "y", "수면 중 심박수")]
    sc = keyword_scores("커피를 마시면 심박이", chunks)
    assert sc[0] > sc[1] and keyword_scores("!!", chunks) == [0.0, 0.0]


# ---------- 폴백 ----------
def test_falls_back_to_keyword_search_when_embedding_is_unavailable(corpus_dir, tmp_path):
    s = make_search(corpus_dir, tmp_path, FakeEmbedder(fail=True))
    r = s.search("커피와 카페인 두근거림", 2)
    assert r.method == "keyword" and "키워드 검색" in r.note and r.hits[0].doc_id == "caffeine"


def test_failed_index_is_not_retried_immediately(corpus_dir, tmp_path):
    emb = FakeEmbedder(fail=True)
    s = make_search(corpus_dir, tmp_path, emb)
    s.search("커피", 1)
    calls = emb.calls
    r = s.search("커피", 1)
    assert emb.calls == calls and r.method == "keyword" and "최근 실패" in r.note       # 대기 시간 안에는 Ollama 를 다시 부르지 않는다


def test_query_embedding_failure_after_index_is_ready_falls_back(corpus_dir, tmp_path):
    s = make_search(corpus_dir, tmp_path)
    s.build_index()
    s.embedder.fail = True
    r = s.search("수면 심박", 1)
    assert r.method == "keyword" and "질의 임베딩 실패" in r.note


def test_unreachable_ollama_raises_embedding_error():
    with pytest.raises(EmbeddingError):
        OllamaEmbedder(base_url="http://127.0.0.1:9", timeout_s=1).embed(["x"])
    assert OllamaEmbedder().embed([]).shape == (0, 0)


# ---------- MCP 도구 ----------
def call(args: dict):
    """인메모리 MCP Client 로 search_guidelines 를 호출해 .data 를 돌려준다."""
    async def run():
        async with Client(mcp) as c:
            return (await c.call_tool("search_guidelines", args)).data
    return asyncio.run(run())


@pytest.fixture()
def patched_search(corpus_dir, tmp_path, monkeypatch):
    """서버 싱글턴 검색기를 가짜 임베더 기반 검색기로 바꿔 끼운다."""
    s = make_search(corpus_dir, tmp_path)
    monkeypatch.setattr(rag_search, "_SINGLETON", s)
    return s


def test_mcp_tool_is_registered_and_documented():
    async def run():
        async with Client(mcp) as c:
            return {t.name: t for t in await c.list_tools()}
    tools = asyncio.run(run())
    t = tools["search_guidelines"]
    assert len(t.description) > 200 and "hits" in t.description and "get_vitals_summary" in t.description
    assert set(t.inputSchema["properties"]) == {"query", "k"} and t.inputSchema["required"] == ["query"]
    assert t.inputSchema["properties"]["k"]["default"] == 3


def test_mcp_tool_output_shape(patched_search):
    r = call({"query": "커피 마시면 심박이 올라가나요", "k": 2})
    assert set(r) == {"query", "method", "note", "hits"} and r["method"] == "vector" and len(r["hits"]) == 2
    assert set(r["hits"][0]) == {"doc_id", "title", "heading", "text", "score", "source", "full_text"}
    assert r["hits"][0]["doc_id"] == "caffeine" and isinstance(r["hits"][0]["score"], float)


def test_mcp_tool_validates_input_and_degrades_to_keyword(corpus_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(rag_search, "_SINGLETON", make_search(corpus_dir, tmp_path, FakeEmbedder(fail=True)))
    assert "비어" in call({"query": "   "})["error"]
    assert "k 는" in call({"query": "커피", "k": 0})["error"] and "k 는" in call({"query": "커피", "k": 9})["error"]
    r = call({"query": "커피 카페인"})
    assert r["method"] == "keyword" and r["note"] and r["hits"] and "error" not in r


# ---------- LLM 꺼짐 / 타임아웃 / 백그라운드 구축 (도커 fake 환경에서 질문이 60초 넘게 걸린 결함) ----------
import dataclasses  # noqa: E402
import httpx  # noqa: E402
from app.config import settings as real_settings  # noqa: E402
from app.rag import embeddings as rag_embeddings  # noqa: E402


class SpyHttp:
    """httpx.Client 대역: 생성/POST 횟수와 타임아웃을 기록하고, POST 는 연결 거부로 실패시킨다 (진짜 네트워크에 나가지 않는다)."""

    clients: list[float] = []
    posts = 0

    def __init__(self, timeout=None, **kw):
        SpyHttp.clients.append(timeout)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, *a, **k):
        SpyHttp.posts += 1
        raise httpx.ConnectError("refused")


@pytest.fixture()
def spy_http(monkeypatch):
    SpyHttp.clients, SpyHttp.posts = [], 0
    monkeypatch.setattr(rag_embeddings.httpx, "Client", SpyHttp)
    return SpyHttp


def with_settings(monkeypatch, **kw):
    """임베딩 모듈의 설정을 바꾼다. 기본은 '임베딩 허용'(테스트가 LLM_ENABLED=false 로 돌아도 영향받지 않게)이고 kw 로 덮어쓴다."""
    base = {"llm_enabled": True, "llm_backend": "ollama", "embed_enabled": True, **kw}
    monkeypatch.setattr(rag_embeddings, "settings", dataclasses.replace(real_settings, **base))


@pytest.mark.parametrize("override", [{"llm_enabled": False}, {"llm_backend": "fake"}, {"embed_enabled": False}])
def test_embeddings_disabled_means_zero_http_calls_and_keyword_search(corpus_dir, tmp_path, monkeypatch, spy_http, override):
    """LLM 이 꺼졌거나 fake 백엔드이거나 EMBED_ENABLED=false 면 Ollama 로 HTTP 요청을 한 번도 보내지 않고 곧바로 키워드 검색이다."""
    make_search(corpus_dir, tmp_path).build_index()                          # 캐시가 있어도 (가짜 임베더로 만든 같은 모델명 'fake' 와 달라 못 쓰지만) 호출은 없어야 한다
    with_settings(monkeypatch, **override)
    s = GuidelineSearch(corpus_dir, OllamaEmbedder(), tmp_path / "cache" / "index.npz")
    r = s.search("커피와 카페인 두근거림", 2)
    assert r.method == "keyword" and "비활성" in r.note and r.hits[0].doc_id == "caffeine"
    s.search("수면", 1)
    assert spy_http.clients == [] and spy_http.posts == 0


def test_mcp_tool_makes_no_ollama_call_when_llm_is_off(corpus_dir, tmp_path, monkeypatch, spy_http):
    with_settings(monkeypatch, llm_backend="fake")
    monkeypatch.setattr(rag_search, "_SINGLETON", GuidelineSearch(corpus_dir, OllamaEmbedder(), tmp_path / "i.npz"))
    r = call({"query": "커피 카페인"})
    assert r["method"] == "keyword" and r["hits"] and spy_http.clients == []


def test_cached_index_loads_without_any_http_call_and_query_uses_short_timeout(corpus_dir, tmp_path, monkeypatch, spy_http):
    cache = tmp_path / "cache" / "index.npz"
    GuidelineSearch(corpus_dir, FakeEmbedder("bge-test"), cache).build_index()
    with_settings(monkeypatch, embed_timeout_s=3.0, embed_build_timeout_s=60.0)
    emb = OllamaEmbedder(model="bge-test")
    assert (emb.timeout_s, emb.build_timeout_s) == (3.0, 60.0)
    s = GuidelineSearch(corpus_dir, emb, cache)
    assert s._ensure_index() == "" and spy_http.clients == []                       # 캐시 로드는 Ollama 를 부르지 않는다
    r = s.search("커피", 1)
    assert spy_http.clients == [3.0] and r.method == "keyword" and "질의 임베딩 실패" in r.note   # 질의는 3초 상한으로 한 번만 시도하고 폴백


def test_document_embedding_uses_the_longer_build_timeout(corpus_dir, tmp_path, monkeypatch, spy_http):
    with_settings(monkeypatch, embed_timeout_s=3.0, embed_build_timeout_s=60.0)
    s = GuidelineSearch(corpus_dir, OllamaEmbedder(), tmp_path / "x" / "index.npz")
    with pytest.raises(EmbeddingError):
        s.build_index()
    assert spy_http.clients and set(spy_http.clients) == {60.0}


def test_slow_query_embedding_is_not_retried_for_a_while(corpus_dir, tmp_path):
    s = make_search(corpus_dir, tmp_path)
    s.build_index()
    s.embedder.fail = True
    s.search("커피", 1)
    calls = s.embedder.calls
    r = s.search("커피", 1)
    assert s.embedder.calls == calls and "최근 실패" in r.note                          # 15초 동안은 임베딩을 다시 부르지 않는다


def test_background_build_serves_keyword_first_then_vector(corpus_dir, tmp_path):
    s = GuidelineSearch(corpus_dir, FakeEmbedder(), tmp_path / "bg" / "index.npz", build_in_background=True)
    first = s.search("커피 카페인", 1)
    assert first.method == "keyword" and "백그라운드" in first.note                      # 요청 경로가 구축을 기다리지 않는다
    s._build_thread.join(timeout=10)
    assert s.search("커피 카페인", 1).method == "vector"


# ---------- 키워드 폴백의 결정론적 동점 처리 ----------
def test_keyword_ties_are_broken_by_doc_id_and_zero_scores_are_flagged(corpus_dir, tmp_path):
    s = make_search(corpus_dir, tmp_path, FakeEmbedder(fail=True))
    r1, r2 = s.search("ㅋㅋㅋ", 3), s.search("ㅋㅋㅋ", 3)                        # 질문과 겹치는 단서가 하나도 없다
    ids = [h.doc_id for h in r1.hits]
    assert ids == sorted(ids) == [h.doc_id for h in r2.hits] and all(h.score == 0 for h in r1.hits)
    assert "일치하는 키워드가 없어" in r1.note and "문서 id 순" in r1.note
    ok = s.search("커피 카페인", 1)
    assert ok.hits[0].doc_id == "caffeine" and "일치하는 키워드가 없어" not in ok.note


# ---------- 발췌는 문장 경계에서 자른다 ----------
from app.rag.chunking import chunk_corpus  # noqa: E402
from app.rag.search import EXCERPT_CHARS, trim_to_sentences  # noqa: E402

LONG = ("카페인은 심박수를 올릴 수 있습니다. 효과는 섭취 후 30분~1시간쯤 나타나요. 민감한 사람은 더 두드러집니다! 늦은 오후에는 피하세요? "
        "평균 72.4 bpm 처럼 소수점은 문장 끝이 아닙니다. 마지막 문장은 길게 이어져서 한도를 넘게 만듭니다. 그래서 여기서 잘려야 합니다.")


def test_short_text_is_untouched_without_ellipsis():
    assert trim_to_sentences("짧은 문장입니다. 두 번째입니다.") == "짧은 문장입니다. 두 번째입니다."
    assert trim_to_sentences("x" * EXCERPT_CHARS) == "x" * EXCERPT_CHARS


def test_long_text_is_cut_at_a_sentence_end_and_marked_with_ellipsis():
    out = trim_to_sentences(LONG, 120)
    assert out.endswith(" …") and out[:-2].rstrip().endswith(("다.", "요.", "!", "?")) and len(out) <= 120 + 2
    assert LONG.startswith(out[:-2].rstrip())                                                 # 원문의 앞부분 그대로(중간을 고치지 않음)
    assert "72.4" not in out or "72.4 bpm" in out                                             # 소수점 안에서 끊지 않는다


def test_decimal_points_and_newlines_are_handled():
    text = "첫 줄은 이렇게 끝납니다\n둘째 줄은 평균 72.4 bpm 이라고 말합니다. " + "셋째 문장입니다. " * 30
    out = trim_to_sentences(text, 80)
    assert "72.4 bpm" in out and out.endswith("…") and "\n" in out
    nl = trim_to_sentences("가" * 40 + "\n" + "나" * 40 + "\n" + "다" * 40, 90)
    assert nl == "가" * 40 + "\n" + "나" * 40 + " …"                                          # 줄바꿈도 경계


def test_single_overlong_sentence_is_cut_at_a_soft_boundary():
    out = trim_to_sentences("단어 " * 200, 60)
    assert out.endswith("…") and len(out) <= 61 and not out[:-1].endswith(" ") and "단어" in out
    assert trim_to_sentences("가" * 400, 50) == "가" * 50 + "…"                                  # 끊을 자리가 전혀 없으면 한도에서


def test_every_real_chunk_excerpt_ends_on_a_sentence_boundary_or_is_complete():
    chunks = chunk_corpus(load_corpus(GUIDELINES_DIR))
    trimmed = 0
    for c in chunks:
        out = trim_to_sentences(c.text)
        if out == c.text.strip():
            continue
        trimmed += 1
        body = out[:-2].rstrip() if out.endswith(" …") else out[:-1]
        assert out.endswith("…") and len(body) <= EXCERPT_CHARS and c.text.startswith(body), c.chunk_id
        if out.endswith(" …"):                                                                # 문장 경계에서 잘렸다면 직전이 문장 끝/줄 끝
            assert body.endswith((".", "!", "?", "。")) or c.text[len(body):].startswith("\n"), (c.chunk_id, body[-20:])
    assert trimmed > 0                                                                       # 실제 코퍼스에는 300자를 넘는 청크가 있다


def test_hit_carries_trimmed_text_and_full_text(corpus_dir, tmp_path):
    (corpus_dir / "long.md").write_text("---\nid: long\ntitle: 긴 문서\ntopics: [x]\nsource: 자체 작성 요약\n---\n## 긴 구간\n" + LONG * 2 + "\n", encoding="utf-8")
    s = make_search(corpus_dir, tmp_path, FakeEmbedder(fail=True))
    h = next(h for h in s.search("카페인 심박수", 5).hits if h.doc_id == "long")
    assert len(h.text) < len(h.full_text) and h.text.endswith("…") and h.to_dict()["full_text"] == h.full_text
    short = next(h for h in s.search("카페인 심박수", 5).hits if h.doc_id == "caffeine")
    assert short.text == short.full_text and not short.text.endswith("…")


# ---------- 문서 정보 섹션 / 출처 표기 ----------
def test_info_section_is_not_indexed_and_editing_it_does_not_stale_the_index(corpus_dir, tmp_path):
    """`## 문서 정보` 는 색인 제외라 출처·검증 문구를 고쳐도 청크 해시가 같아 캐시 인덱스가 그대로 유효하다 (재임베딩 불필요)."""
    s = make_search(corpus_dir, tmp_path)
    s.build_index()
    sleep = corpus_dir / "sleep.md"
    sleep.write_text(DOCS["sleep.md"].replace("source: 자체 작성 요약 (테스트)", "source: 자체 작성한 일반 상식 요약")
                     + "\n## 문서 정보\n- 의학적 조언이 아니며 임상 검토를 받지 않았습니다.\n- 수치는 대략적인 값이며 검증하지 않았습니다.\n", encoding="utf-8")
    again = make_search(corpus_dir, tmp_path)
    assert [c.content_hash for c in again.chunks] == [c.content_hash for c in s.chunks]            # 청크 해시 동일
    assert all("임상 검토" not in c.text and c.heading != "문서 정보" for c in again.chunks)
    assert VectorIndex.load_cached(again.chunks, again.cache_path, "fake") is not None             # 캐시 인덱스가 여전히 유효
    hit = next(h for h in again.search("잠잘 때 심박수가 낮아지나요", 3).hits if h.doc_id == "sleep")
    assert hit.source == "자체 작성한 일반 상식 요약"                                                # 새 출처 표기는 검색 결과로 전달된다


def test_real_corpus_every_doc_has_the_info_section_with_the_non_medical_line():
    for p in sorted(GUIDELINES_DIR.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        assert "\n## 문서 정보\n" in text, p.name
        info = text.split("\n## 문서 정보\n", 1)[1]
        assert "의학적 조언이 아니며 임상 검토를 받지 않았습니다" in info and ("자체 작성" in info or "자체 설명" in info), p.name
    for name, needle in (("exercise-hr-zones", "50~70%"), ("hr-recovery", "12회"), ("sleep-hr", "10~20%"), ("dehydration-heat-fever", "8~10회")):
        info = (GUIDELINES_DIR / f"{name}.md").read_text(encoding="utf-8").split("\n## 문서 정보\n", 1)[1]
        assert needle in info and "검증하지 않았습니다" in info, name                                   # 검증 못 한 수치는 해당 문서에 명시


# ---------- 부분 인덱스 (코퍼스 일부만 고쳐졌을 때) ----------
def _edit_sleep(corpus_dir):
    (corpus_dir / "sleep.md").write_text(DOCS["sleep.md"].replace("가장 낮습니다", "가장 낮게 나타납니다"), encoding="utf-8")


def test_load_cached_allow_partial_keeps_unchanged_chunks(corpus_dir, tmp_path):
    s = make_search(corpus_dir, tmp_path)
    s.build_index()
    _edit_sleep(corpus_dir)
    fresh = make_search(corpus_dir, tmp_path)
    assert VectorIndex.load_cached(fresh.chunks, fresh.cache_path, "fake") is None
    part = VectorIndex.load_cached(fresh.chunks, fresh.cache_path, "fake", allow_partial=True)
    assert part is not None and part.n_missing == 1 and len(part.chunks) == len(fresh.chunks) - 1


def test_stale_chunk_keeps_vector_search_with_note_when_embedder_is_down(corpus_dir, tmp_path):
    make_search(corpus_dir, tmp_path).build_index()
    _edit_sleep(corpus_dir)
    s = make_search(corpus_dir, tmp_path, FakeEmbedder(fail_first=1))
    r = s.search("커피와 카페인이 심박수를 올리나요", 2)
    assert r.method == "vector" and r.hits[0].doc_id == "caffeine" and "일부 낡아 1개" in r.note


def test_stale_chunk_is_reembedded_inline_when_embedder_is_up(corpus_dir, tmp_path):
    make_search(corpus_dir, tmp_path).build_index()
    _edit_sleep(corpus_dir)
    emb = FakeEmbedder()
    r = make_search(corpus_dir, tmp_path, emb).search("잠잘 때 심박수가 낮아지나요", 1)
    assert r.method == "vector" and r.note == "" and r.hits[0].doc_id == "sleep" and emb.texts == 2   # 바뀐 청크 1개 + 질의 1건


def test_stale_chunk_is_filled_in_background_after_serving_partial(corpus_dir, tmp_path):
    make_search(corpus_dir, tmp_path).build_index()
    _edit_sleep(corpus_dir)
    s = GuidelineSearch(corpus_dir, FakeEmbedder(), tmp_path / "cache" / "index.npz", build_in_background=True)
    first = s.search("커피 카페인", 1)
    assert first.method == "vector" and "일부 낡아" in first.note
    s._build_thread.join(timeout=10)
    assert s.search("커피 카페인", 1).note == ""

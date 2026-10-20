"""가이드라인 검색 진입점: 벡터 검색 + 키워드 폴백.

책임 경계:
    코퍼스 로드 -> 청킹 -> 인덱스 준비 -> 질의 검색을 한 클래스(`GuidelineSearch`)로 묶는다. LLM 호출·답변 생성은 하지 않는다
    (검색 결과를 쓰는 쪽은 에이전트이며, MCP 도구 `search_guidelines` 가 이 클래스를 노출한다).

실패 처리 (graceful degradation):
    - 임베딩이 비활성 설정(EMBED_ENABLED=false, LLM_ENABLED=false, LLM_BACKEND=fake)이면 Ollama 를 전혀 부르지 않고 곧바로 키워드 검색이다.
    - 저장된 인덱스(캐시)가 코퍼스와 맞으면 Ollama 없이 불러온다. 질의 임베딩만 Ollama 를 부르며 짧은 타임아웃(EMBED_TIMEOUT_S, 기본 3초) 안에
      끝나지 않으면 키워드로 대체하고 `method` 를 'keyword' 로 표시한다 (바쁜 Ollama 앞에서 서비스 응답이 밀리지 않게).
    - 인덱스가 없거나 코퍼스와 어긋나면 `build_in_background=True`(서비스 싱글턴)일 때 백그라운드 스레드가 구축하고 그동안은 키워드로 답한다
      (요청 경로가 구축을 기다리지 않음). 인덱스 준비 실패는 RETRY_AFTER_S, 질의 임베딩 실패는 QUERY_RETRY_AFTER_S 동안 다시 시도하지 않는다.
    키워드 검색은 기존 에이전트 폴백(`report.guideline_excerpt`)과 같은 아이디어(어절 + 앞 두 글자 단서)를 청크 단위로 적용한다.

협력 모듈: `corpus.py`, `chunking.py`, `embeddings.py`, `index.py`, `app.mcp_server.guidelines_server`(도구), `eval/rag_eval.py`(평가).
"""
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from app.config import settings
from app.rag.chunking import Chunk, chunk_corpus
from app.rag.corpus import GUIDELINES_DIR, load_corpus
from app.rag.embeddings import EmbeddingError, OllamaEmbedder
from app.rag.index import BuildStats, VectorIndex

log = logging.getLogger(__name__)
RETRY_AFTER_S = 60.0           # 인덱스 준비 실패 후 재시도까지
QUERY_RETRY_AFTER_S = 15.0     # 질의 임베딩 실패(시간 초과 등) 후 재시도까지: 바쁜 Ollama 앞에서 매 질의가 타임아웃을 기다리지 않게
_TOKEN = re.compile(r"[가-힣A-Za-z0-9]{2,}")


EXCERPT_CHARS = 300                 # 사용자에게 보이는 발췌 길이 상한 (청크는 최대 약 500자)
_SPLIT = re.compile(r"((?<=[.!?。])[ \t]+|\n+)")      # 문장 끝(. ! ? 。 뒤 공백) 또는 줄바꿈. 소수점 "72.4" 는 뒤에 공백이 없어 끊지 않는다
_SOFT = re.compile(r"[,;:、，] |\s")                 # 한 문장이 한도보다 길 때 마지막 수단으로 끊을 수 있는 자리


def trim_to_sentences(text: str, limit: int = EXCERPT_CHARS) -> str:
    """text 를 문장 경계(다./요./. ! ? 。 와 줄바꿈)에서 limit 자 이하로 줄인다. 잘렸을 때만 끝에 '…' 를 붙인다.

    Args:
        text: 원문 (청크 본문). limit: 최대 길이(자, '…' 제외 본문 기준).

    Returns:
        limit 이하면 원문 그대로. 넘으면 limit 안에 완전히 들어가는 앞쪽 문장들 + ' …'. 첫 문장 하나가 limit 보다 길면 그 문장을
        limit 안의 마지막 쉼표/공백에서 끊고 '…' 를 붙인다 (이 경우만 문장 중간에서 끊긴다).
    """
    text = text.strip()
    if len(text) <= limit:
        return text
    parts = _SPLIT.split(text)                                  # [문장, 구분자, 문장, 구분자, ...]
    out = ""
    for i in range(0, len(parts), 2):
        cand = out + parts[i]
        if len(cand) > limit:
            break
        out = cand + (parts[i + 1] if i + 1 < len(parts) else "")
    out = out.rstrip()
    if not out:                                                 # 첫 문장이 한도보다 길다
        cut = text[:limit]
        spots = [m.start() for m in _SOFT.finditer(cut)]
        out = cut[:spots[-1]].rstrip(",;:、， ") if spots else cut
        return out + "…"
    return out + " …"


@dataclass(frozen=True)
class Hit:
    """검색 결과 청크 하나. score 는 벡터 검색이면 코사인 유사도(-1~1), 키워드 검색이면 단서 일치 비율(0~1).

    Attributes:
        text: 사용자에게 보이는 발췌 - 문장 경계에서 EXCERPT_CHARS 이하로 줄인 것 (잘렸으면 끝에 '…').
        full_text: 청크 전문 (에이전트가 근거를 더 읽어야 할 때 쓴다). 화면에 그대로 내보이지 말 것.
    """
    doc_id: str
    title: str
    heading: str
    text: str
    score: float
    source: str
    full_text: str = ""

    def to_dict(self) -> dict:
        """JSON 직렬화용 dict (score 는 소수 4자리)."""
        return {"doc_id": self.doc_id, "title": self.title, "heading": self.heading, "text": self.text,
                "score": round(self.score, 4), "source": self.source, "full_text": self.full_text or self.text}


@dataclass(frozen=True)
class SearchResult:
    """검색 결과. method 는 'vector' | 'keyword', note 는 폴백 사유(없으면 빈 문자열)."""
    query: str
    method: str
    hits: list[Hit]
    note: str = ""


def keyword_scores(query: str, chunks: list[Chunk]) -> list[float]:
    """청크별 키워드 점수: 질의 어절과 그 앞 두 글자(조사·어미 변형 대응)가 청크에 들어 있는 비율.

    기존 `report.guideline_excerpt` 의 단서 방식을 따르되 제목/소제목까지 포함해 센다. 질의에서 뽑은 단서가 없으면 전부 0.
    """
    tokens = set(_TOKEN.findall(query))
    cues = tokens | {t[:2] for t in tokens if len(t) >= 3 and re.match(r"[가-힣]", t)}
    if not cues:
        return [0.0] * len(chunks)
    return [sum(cue in c.embed_text for cue in cues) / len(cues) for c in chunks]


class GuidelineSearch:
    """가이드라인 코퍼스에 대한 검색기.

    Attributes:
        chunks: 청킹된 코퍼스. cache_path: 임베딩 캐시 파일. embedder: 임베딩 생성기.
    """

    def __init__(self, directory: Path | None = None, embedder: OllamaEmbedder | None = None, cache_path: Path | None = None,
                 build_in_background: bool = False) -> None:
        """Args:
            directory: 생략=서비스 가이드라인 폴더. embedder: 생략=settings 기반. cache_path: 생략=data_dir/rag/index.npz.
            build_in_background: True 면 인덱스가 없을 때 첫 검색이 구축을 기다리지 않고 백그라운드 스레드가 만든다 (서비스용).
        """
        self.chunks = chunk_corpus(load_corpus(directory or GUIDELINES_DIR))
        self.embedder = embedder or OllamaEmbedder()
        self.cache_path = cache_path or settings.data_dir / "rag" / "index.npz"
        self._index: VectorIndex | None = None
        self._partial_missing = 0          # 부분 인덱스로 서비스 중일 때 아직 색인되지 않은 청크 수
        self._failed_at: float | None = None
        self._query_failed_at: float | None = None
        self._background = build_in_background
        self._build_thread: threading.Thread | None = None
        self.last_build: BuildStats | None = None

    def build_index(self) -> BuildStats:
        """인덱스를 (필요하면 임베딩을 만들어) 준비한다. 재실행하면 캐시로 즉시 끝난다.

        Raises:
            EmbeddingError: 새 임베딩이 필요한데 Ollama 를 쓸 수 없을 때.
        """
        self._index, self.last_build = VectorIndex.build(self.chunks, self.embedder, self.cache_path)
        self._partial_missing = 0
        self._failed_at = None
        return self.last_build

    def _ensure_index(self) -> str:
        """벡터 인덱스를 준비한다. 성공이면 '', 실패(최근 실패 후 대기 중 포함)이면 사유 문자열."""
        if self._index is not None:
            if self._partial_missing and self._background:          # 부분 인덱스로 답하는 동안 빠진 청크를 백그라운드에서 채운다
                self._start_background_build()
            return ""
        if self._failed_at is not None and time.monotonic() - self._failed_at < RETRY_AFTER_S:
            return "임베딩 인덱스를 준비하지 못해(최근 실패) 키워드 검색을 사용합니다"
        self._index = VectorIndex.load_cached(self.chunks, self.cache_path, self.embedder.model)      # Ollama 호출 없는 빠른 경로
        if self._index is not None:
            return ""
        partial = VectorIndex.load_cached(self.chunks, self.cache_path, self.embedder.model, allow_partial=True)
        if partial is not None:                                     # 일부 청크만 낡은 경우: 나머지는 벡터 검색을 계속 쓴다
            self._index, self._partial_missing = partial, partial.n_missing
            if self._background:
                self._start_background_build()
            else:
                try:
                    self.build_index()
                except EmbeddingError:                              # 임베딩을 못 만들면 부분 인덱스를 그대로 쓴다
                    self._failed_at = time.monotonic()
            return ""
        if self._background:
            self._start_background_build()
            return "임베딩 인덱스를 백그라운드에서 구축 중이라 키워드 검색을 사용합니다"
        try:
            self.build_index()
            return ""
        except EmbeddingError as e:
            self._failed_at = time.monotonic()
            return f"임베딩 사용 불가({e}) - 키워드 검색으로 대체했습니다"

    def _start_background_build(self) -> None:
        """백그라운드 구축 스레드를 (이미 돌고 있지 않고 최근 실패도 없으면) 시작한다."""
        if self._failed_at is not None and time.monotonic() - self._failed_at < RETRY_AFTER_S:
            return
        if self._build_thread is None or not self._build_thread.is_alive():
            self._build_thread = threading.Thread(target=self._build_quietly, name="rag-index-build", daemon=True)
            self._build_thread.start()

    def _build_quietly(self) -> None:
        """백그라운드 구축 스레드 본체: 실패는 예외로 새지 않고 실패 시각만 기록한다 (이후 RETRY_AFTER_S 동안 재시도 안 함)."""
        try:
            self.build_index()
        except EmbeddingError as e:
            self._failed_at = time.monotonic()
            log.warning("RAG 인덱스 백그라운드 구축 실패: %s", e)

    def search(self, query: str, k: int = 3) -> SearchResult:
        """질의와 가장 관련 있는 청크 k 개를 찾는다.

        Args:
            query: 자연어 질문 (한국어/영어 혼용 가능).
            k: 반환 개수 (1 이상).

        Returns:
            SearchResult. 벡터 검색이 되면 method='vector', Ollama 문제 등으로 안 되면 method='keyword' 와 note.
        """
        k = max(1, k)
        off = getattr(self.embedder, "disabled_reason", lambda: "")()
        if off:                                                    # 임베딩 비활성 설정: Ollama 를 부르지 않는다
            reason = f"임베딩 비활성({off}) - 키워드 검색을 사용합니다"
        elif self._query_failed_at is not None and time.monotonic() - self._query_failed_at < QUERY_RETRY_AFTER_S:
            reason = "질의 임베딩이 최근 실패해 키워드 검색을 사용합니다"
        else:
            reason = self._ensure_index()
        if not reason:
            try:
                qv = self.embedder.embed([query])[0]
                self._query_failed_at = None
                note = (f"색인이 일부 낡아 {self._partial_missing}개 문단은 아직 벡터 검색에 포함되지 않았습니다(재색인 필요)"
                        if self._partial_missing else "")
                return SearchResult(query, "vector", [self._hit(c, s) for c, s in self._index.top_k(qv, k)], note)
            except EmbeddingError as e:
                self._query_failed_at = time.monotonic()
                reason = f"질의 임베딩 실패({e}) - 키워드 검색으로 대체했습니다"
        scores = keyword_scores(query, self.chunks)
        # 동점은 (doc_id, chunk_id) 순으로 결정론적으로 정렬한다. 특히 점수가 전부 0 이면(질문과 겹치는 단서가 없음) 순서는 관련성이 아니라
        # 문서 id 순일 뿐이므로 note 로 밝힌다 - 호출 측은 score 0 인 결과를 근거로 쓰면 안 된다.
        order = sorted(range(len(self.chunks)), key=lambda i: (-scores[i], self.chunks[i].doc_id, self.chunks[i].chunk_id))[:k]
        if scores and max(scores) == 0:
            reason = (reason + " / " if reason else "") + "질문과 일치하는 키워드가 없어 결과가 관련 없을 수 있습니다 (동점은 문서 id 순)"
        return SearchResult(query, "keyword", [self._hit(self.chunks[i], scores[i]) for i in order], reason)

    @staticmethod
    def _hit(chunk: Chunk, score: float) -> Hit:
        """Chunk 를 Hit 으로 변환한다."""
        return Hit(chunk.doc_id, chunk.title, chunk.heading, trim_to_sentences(chunk.text), score, chunk.source, chunk.text)


_SINGLETON: GuidelineSearch | None = None


def get_search() -> GuidelineSearch:
    """프로세스 공용 검색기 (처음 호출 때 코퍼스를 읽고 청킹한다; 인덱스는 첫 검색 때 준비)."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = GuidelineSearch(build_in_background=True)     # 서비스: 요청 경로가 인덱스 구축을 기다리지 않는다
    return _SINGLETON


if __name__ == "__main__":
    import sys

    s = get_search()
    st = s.build_index()
    print(f"청크 {st.n_chunks}개 (새 임베딩 {st.n_embedded}, 캐시 {st.n_cached}), dim {st.dim}, {st.seconds}s, {st.cache_bytes:,} bytes")
    if len(sys.argv) > 1:
        r = s.search(" ".join(sys.argv[1:]))
        print(r.method, r.note)
        for h in r.hits:
            print(f"{h.score:.3f} [{h.doc_id}] {h.heading}: {h.text[:60]}")

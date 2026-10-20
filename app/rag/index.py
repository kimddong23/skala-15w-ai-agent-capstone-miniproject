"""영속 벡터 인덱스: 청크 임베딩 캐시 + 코사인 top-k.

책임 경계:
    (청크 내용 해시 -> 임베딩) 캐시를 .npz 한 파일로 관리하고, 그 위에서 코사인 유사도 top-k 를 계산한다.
    임베딩 생성은 `embeddings.OllamaEmbedder` 에 맡기고, 폴백 정책은 `search.py` 가 정한다.

설계 근거 (새 의존성 없이 numpy 만 사용):
    청크 수가 수십~수백 개 규모라 정규화된 행렬의 내적으로 충분하다(ANN 인덱스는 과잉). 캐시 키는 임베딩 입력의 sha256 이라
    문서가 바뀌면 바뀐 청크만 다시 임베딩하고, 재실행하면 Ollama 호출 없이 즉시 로드된다(멱등). 모델 이름이 바뀌면 캐시를 버린다.

데이터 형식 (.npz): hashes(str 배열), vectors(float32 N x dim), model(str).

협력 모듈: `chunking.Chunk`(content_hash), `embeddings.py`, `search.py`(소비자), `eval/rag_eval.py`(BuildStats 보고).
"""
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.rag.chunking import Chunk
from app.rag.embeddings import OllamaEmbedder


@dataclass(frozen=True)
class BuildStats:
    """인덱스 구축 결과 요약. seconds 는 임베딩 호출 포함 전체 소요, cache_bytes 는 저장된 .npz 크기."""
    n_chunks: int
    n_embedded: int
    n_cached: int
    seconds: float
    dim: int
    cache_bytes: int


def _load_cache(path: Path, model: str) -> dict[str, np.ndarray]:
    """캐시 파일을 {content_hash: 벡터} 로 읽는다. 없거나 깨졌거나 모델이 다르면 빈 dict."""
    try:
        with np.load(path, allow_pickle=False) as z:
            if str(z["model"]) != model:
                return {}
            return dict(zip(z["hashes"].tolist(), z["vectors"]))
    except (OSError, ValueError, KeyError):
        return {}


def _save_cache(path: Path, model: str, hashes: list[str], vectors: np.ndarray) -> None:
    """캐시를 임시 파일에 쓴 뒤 교체한다 (쓰는 도중 중단돼도 기존 캐시가 깨지지 않게)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, hashes=np.array(hashes), vectors=vectors, model=np.array(model))
    tmp.replace(path)


class VectorIndex:
    """청크와 정규화 임베딩 행렬을 들고 코사인 top-k 를 계산한다.

    Attributes:
        chunks: 인덱스 순서의 청크. matrix: (N, dim) float32 정규화 행렬. model: 임베딩 모델 이름.
    """

    def __init__(self, chunks: list[Chunk], matrix: np.ndarray, model: str, n_missing: int = 0) -> None:
        self.chunks, self.matrix, self.model, self.n_missing = chunks, matrix, model, n_missing

    @classmethod
    def load_cached(cls, chunks: list[Chunk], path: Path, model: str, allow_partial: bool = False) -> "VectorIndex | None":
        """캐시만으로 인덱스를 만든다 (Ollama 호출 없음).

        Args:
            allow_partial: True 면 일부 청크의 임베딩이 캐시에 없어도, 있는 청크만으로 *부분 인덱스*를 만든다(`n_missing` 에 빠진 수).
                코퍼스 문장을 조금 고친 뒤 재임베딩 전에도 나머지 청크는 벡터 검색에 쓰기 위함이다. 고친 청크는 재구축 전까지 벡터 검색에서 빠진다.

        Returns:
            모든 청크의 임베딩이 캐시에 있으면 VectorIndex. 하나라도 없으면 None (allow_partial=False) 이거나 있는 청크만의 부분 인덱스
            (allow_partial=True; 하나도 없으면 None).
        """
        cache = _load_cache(path, model)
        have = [c for c in chunks if c.content_hash in cache]
        if not have or (len(have) < len(chunks) and not allow_partial):
            return None
        return cls(have, np.stack([cache[c.content_hash] for c in have]).astype(np.float32), model, n_missing=len(chunks) - len(have))

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: OllamaEmbedder, path: Path) -> tuple["VectorIndex", BuildStats]:
        """캐시에 없는 청크만 임베딩해 인덱스를 만들고 캐시를 갱신한다 (재실행하면 임베딩 0건).

        Args:
            chunks: 인덱싱할 청크. embedder: 임베딩 생성기. path: 캐시 .npz 경로.

        Returns:
            (VectorIndex, BuildStats). 캐시에서 사라진 청크의 벡터는 저장 시 정리된다.

        Raises:
            app.rag.embeddings.EmbeddingError: 새 임베딩이 필요한데 생성에 실패했을 때.
        """
        t0 = time.perf_counter()
        cache = _load_cache(path, embedder.model)
        missing = [c for c in chunks if c.content_hash not in cache]
        if missing:
            embed_docs = getattr(embedder, "embed_documents", embedder.embed)     # 구축용(긴 타임아웃)이 있으면 그것을 쓴다
            new = embed_docs([c.embed_text for c in missing])
            cache.update({c.content_hash: v for c, v in zip(missing, new)})
        matrix = np.stack([cache[c.content_hash] for c in chunks]).astype(np.float32)
        if missing or not path.exists():
            _save_cache(path, embedder.model, [c.content_hash for c in chunks], matrix)
        stats = BuildStats(len(chunks), len(missing), len(chunks) - len(missing), round(time.perf_counter() - t0, 3),
                           int(matrix.shape[1]), path.stat().st_size)
        return cls(chunks, matrix, embedder.model), stats

    def top_k(self, query_vec: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        """질의 벡터(정규화됨)와 코사인 유사도가 높은 청크 k 개.

        Returns:
            [(Chunk, 점수)] 점수 내림차순 (점수는 -1~1, 보통 0.3~0.8). k 가 청크 수보다 크면 전부.
        """
        scores = self.matrix @ query_vec.astype(np.float32)
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]

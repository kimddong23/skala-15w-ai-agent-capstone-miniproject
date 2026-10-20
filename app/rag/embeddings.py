"""Ollama 임베딩 클라이언트.

책임 경계:
    텍스트 목록을 L2 정규화된 float32 행렬로 바꾼다. 캐시·인덱스·검색은 하지 않는다.

설계 근거 (로컬 LLM 과 같은 이유): 건강 관련 질의를 외부 임베딩 API 로 보내지 않는다. bge-m3 는 한국어 검색 품질이 좋고
이미 로컬에 설치되어 있다. 실패(서버 꺼짐, 모델 없음, 시간 초과)는 `EmbeddingError` 하나로 모아 호출자가 키워드 폴백을 결정하게 한다.

협력 모듈: `index.py`(인덱스 빌드), `search.py`(질의 임베딩, 실패 시 폴백), `app.config.settings`(URL/모델/타임아웃).
"""
import httpx
import numpy as np

from app.config import settings

BATCH = 16


class EmbeddingError(RuntimeError):
    """임베딩을 만들 수 없을 때 (연결 실패, 시간 초과, 모델 없음, 응답 형식 오류)."""


class OllamaEmbedder:
    """Ollama `/api/embed` 로 임베딩을 만드는 동기 클라이언트.

    질의용(`embed`, 짧은 타임아웃)과 인덱스 구축용(`embed_documents`, 긴 타임아웃)을 나눈다: 질의가 서비스 응답 시간에 들어가므로 바쁜
    Ollama 앞에서 오래 기다리지 않고 키워드로 내려가게 하되, 한 번만 하는 구축은 모델 로드 시간을 기다려 준다.

    Attributes:
        base_url: Ollama 서버 URL. model: 임베딩 모델 이름. timeout_s: 질의 임베딩 요청 상한(초). build_timeout_s: 문서 임베딩 상한(초).
    """

    def __init__(self, base_url: str | None = None, model: str | None = None, timeout_s: float | None = None,
                 build_timeout_s: float | None = None) -> None:
        """Args: 생략하면 settings 의 ollama_base_url / embed_model / embed_timeout_s / embed_build_timeout_s."""
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.embed_model
        self.timeout_s = timeout_s or settings.embed_timeout_s
        self.build_timeout_s = build_timeout_s or settings.embed_build_timeout_s

    @staticmethod
    def disabled_reason() -> str:
        """임베딩을 쓰면 안 되는 설정이면 사유, 쓸 수 있으면 빈 문자열.

        EMBED_ENABLED=false, LLM_ENABLED=false, LLM_BACKEND=fake 중 하나면 비활성이다 - LLM 없이 도는 환경(테스트·시연·도커 fake)에서
        바쁘거나 없는 Ollama 를 기다리다 응답이 늦어지는 일을 막는다. 설정은 호출 시점에 읽는다.
        """
        if not settings.embed_enabled:
            return "EMBED_ENABLED=false"
        if not settings.llm_enabled:
            return "LLM_ENABLED=false"
        if settings.llm_backend == "fake":
            return "LLM_BACKEND=fake"
        return ""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """인덱스 구축용 임베딩 (build_timeout_s). 반환·예외는 `embed` 와 같다."""
        return self.embed(texts, self.build_timeout_s)

    def embed(self, texts: list[str], timeout_s: float | None = None) -> np.ndarray:
        """texts 를 임베딩한다 (기본 상한은 질의용 timeout_s).

        Args:
            texts: 임베딩할 문자열 목록 (BATCH 개씩 나눠 호출).
            timeout_s: 요청 1회 상한(초). 생략하면 self.timeout_s.

        Returns:
            (len(texts), dim) float32, 각 행은 L2 정규화 (코사인 유사도 = 내적). 빈 입력이면 (0, 0).

        Raises:
            EmbeddingError: 임베딩 비활성 설정(HTTP 호출 없이 즉시), 서버에 연결할 수 없거나 시간 초과, HTTP 오류, 개수·형식이 맞지 않는 응답.
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        if reason := self.disabled_reason():
            raise EmbeddingError(f"임베딩 비활성({reason})")
        rows: list[list[float]] = []
        try:
            with httpx.Client(timeout=timeout_s or self.timeout_s) as client:
                for i in range(0, len(texts), BATCH):
                    batch = texts[i:i + BATCH]
                    r = client.post(f"{self.base_url}/api/embed", json={"model": self.model, "input": batch})
                    r.raise_for_status()
                    emb = r.json().get("embeddings")
                    if not emb or len(emb) != len(batch):
                        raise EmbeddingError(f"임베딩 응답 개수 불일치: 요청 {len(batch)}, 응답 {len(emb or [])}")
                    rows.extend(emb)
        except (httpx.HTTPError, ValueError, OSError) as e:
            raise EmbeddingError(f"{type(e).__name__}: {e}") from e
        mat = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.where(norms == 0, 1.0, norms)

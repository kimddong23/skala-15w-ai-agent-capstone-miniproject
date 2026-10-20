"""환경변수 기반 설정 (12-factor). 모든 모듈은 `from app.config import settings` 로 읽는다.

책임: 환경변수 → 타입이 있는 불변 설정 객체. 기본값은 로컬 개발(Ollama, 메모리 체크포인터)에 맞춘다.
주의: 값은 import 시점에 한 번 읽는다. 테스트는 `dataclasses.replace(settings, ...)` 로 만든 사본을 대상 모듈의 `settings` 에
monkeypatch 한다. (frozen 이라 직접 대입 불가 — 실행 중 설정 변경으로 인한 경합을 막기 위한 의도적 제약)
설계 근거: 모델 티어링·지연 설정, 로컬 Ollama, 체크포인터 선택.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env_file(path: Path) -> bool:
    """`.env` 가 있으면 환경변수로 불러온다 (이미 설정된 값은 덮어쓰지 않는다 — 셸 export 가 우선).

    예외: 과금 확인용 `OPENAI_CONFIRM` 은 .env 에서 읽지 않는다. 파일에 적어 두면 실행 스크립트의 y/N 확인을 건너뛰는 통로가 되므로,
    프로세스 환경(스크립트가 y/N 뒤에 export 하거나 `docker run -e`)에서 온 값만 인정한다.

    Args:
        path: .env 경로.

    Returns:
        파일을 읽었으면 True. python-dotenv 가 없거나 파일이 없으면 False (선택 기능이므로 조용히 건너뛴다).
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    from_process = os.environ.get("OPENAI_CONFIRM")
    loaded = path.is_file() and load_dotenv(path, override=False)
    if from_process is None:
        os.environ.pop("OPENAI_CONFIRM", None)   # 방금 .env 가 넣은 값이다
    return loaded


load_env_file(ROOT / ".env")  # Settings 의 기본값이 import 시점에 환경변수를 읽으므로 그 전에 불러온다


def _bool(name: str, default: bool) -> bool:
    """환경변수를 bool 로 해석한다 ("1", "true", "yes" 만 참, 대소문자 무시)."""
    return os.getenv(name, str(default)).lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class Settings:
    """애플리케이션 설정.

    Attributes:
        data_dir: 전처리 산출물(parquet, labels.json) 위치. env DATA_DIR.
        raw_dir: 원본 Kaggle CSV 위치. env RAW_DIR.
        ollama_base_url: Ollama 서버 URL. env OLLAMA_BASE_URL.
        fast_model: route/plan/analysis(기본) 용 소형 모델. env OLLAMA_FAST_MODEL (FAST_MODEL 은 별칭).
        quality_model: writing/simple_answer 용 모델. env OLLAMA_QUALITY_MODEL (QUALITY_MODEL 은 별칭).
        llm_backend: LLM 백엔드 "ollama"(기본)|"fake"(결정론, 테스트·시연)|"openai"(원격 API, 과금). env LLM_BACKEND.
        supervisor_version: "v2"(기본, 라우팅·게이트·안전·검증 포함)|"v1"(Before/After 비교용 선형 기준선, 실사용 금지). env SUPERVISOR_VERSION. v1 은 ALLOW_V1_BASELINE=yes 도 함께 있어야만 시작한다(graph.build_graph).
        openai_fast_model / openai_quality_model: openai 백엔드의 티어별 모델. env OPENAI_FAST_MODEL / OPENAI_QUALITY_MODEL.
        fault_inject_transient: True 면 프로세스의 *첫 도구 호출*이 일시 오류(TimeoutError)로 실패한다 — 재시도·오류 격리 시연과 retry 횟수 지표용. env FAULT_INJECT_TRANSIENT.
        hitl_enabled: True 면 에스컬레이션 시 "보호자 알림(시뮬레이션)" 발송 전에 사람 승인을 요청한다(LangGraph interrupt). 안내문 표시는 승인과 무관하다. UI 시연은 true, 테스트 기본은 false. env HITL_ENABLED.
        summarize_after_messages: 요약되지 않은 메시지가 이 개수를 넘으면 오래된 대화를 fast LLM 으로 요약해 상태에 보관한다(컨텍스트 길이 상한). env SUMMARIZE_AFTER_MESSAGES.
        keep_recent_messages: 요약 후에도 원문 그대로 프롬프트에 넣는 최근 메시지 수. env KEEP_RECENT_MESSAGES.
        llm_enabled: False 면 모든 LLM 호출을 건너뛰고 결정론 폴백으로 동작 (테스트·장애 시). env LLM_ENABLED.
        llm_timeout_s: LLM 호출 1회 상한(초). 구조화 호출은 별도로 더 짧게 제한된다 (llm.STRUCTURED_TIMEOUT_S).
        tool_timeout_s: MCP 도구 호출 1회 상한(초). 초과는 TimeoutError(= transient → 재시도 대상)로 처리한다. env TOOL_TIMEOUT_S.
        keep_alive: Ollama 모델 상주 시간. 유휴 후 첫 질의의 콜드 로드(수 초~10초+) 회피. env LLM_KEEP_ALIVE.
        analysis_tier: analysis 노드 모델 티어 "fast"|"quality". 기본 quality: fast 를 qwen3:0.6b 로 바꾼 측정(eval/results/fast_model_compare.md)에서 0.6b 의 해석이 사실 목록 복사에 그쳐(24/24) analysis 만 8b 로 올렸다. (qwen2.5:3b 일 때는 fast 가 -19s 로 유리했다). env ANALYSIS_TIER.
        merge_analysis: True 면 analysis LLM 호출을 생략하고 writing 이 해석을 겸한다. env MERGE_ANALYSIS.
        compact_facts: True 면 LLM 에 JSON 대신 압축 텍스트로 사실을 전달 (프롬프트 토큰 절감). env COMPACT_FACTS.
        num_predict_analysis: analysis 생성 토큰 상한 (0=무제한). env NUM_PREDICT_ANALYSIS.
        num_predict_writing: writing 생성 토큰 상한 (0=무제한). env NUM_PREDICT_WRITING.
        checkpointer: "memory"|"sqlite"|"postgres". env CHECKPOINTER.
        sqlite_path: sqlite 체크포인트 파일. env SQLITE_PATH.
        database_url: postgres 접속 문자열 (미검증 경로). env DATABASE_URL.
        max_replan_rounds: 데이터 부족 시 재계획 최대 횟수. env MAX_REPLAN_ROUNDS.
        max_validation_retries: 검증 실패 후 writing 재시도 최대 횟수. env MAX_VALIDATION_RETRIES.
        validator_judge: 규칙 통과 후 fast LLM judge 를 추가할지. 3b judge 가 정상 리포트를 오탐해 기본 off. env VALIDATOR_JUDGE.
        recursion_limit: LangGraph 재귀 상한. 설계된 두 루프 밖의 무한 루프 차단용 안전망. env GRAPH_RECURSION_LIMIT.
        external_dir: 외부 연구 데이터셋(Stanford 등) 폴더. 있으면 앱이 별도 데이터셋으로 읽고 없으면 무시한다 (커밋 금지 폴더). env EXTERNAL_DIR.
        embed_model: RAG 임베딩 모델(Ollama). env EMBED_MODEL.
        embed_timeout_s: 질의 임베딩 1회 상한(초, 기본 3). 초과·연결 실패는 키워드 검색 폴백을 일으킨다. env EMBED_TIMEOUT_S.
        embed_build_timeout_s: 인덱스 구축용 문서 임베딩 호출 1회 상한(초, 기본 60). env EMBED_BUILD_TIMEOUT_S.
        embed_enabled: False 면 Ollama 임베딩을 일절 호출하지 않고 키워드 검색만 쓴다 (LLM 이 꺼져 있어도 같은 효과). env EMBED_ENABLED.
        cors_origins: 허용 Origin 목록 (쉼표 구분). env CORS_ORIGINS.
    """

    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", ROOT / "data" / "processed")))
    raw_dir: Path = field(default_factory=lambda: Path(os.getenv("RAW_DIR", ROOT / "data" / "raw")))
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    fast_model: str = os.getenv("OLLAMA_FAST_MODEL") or os.getenv("FAST_MODEL", "qwen3:0.6b")
    quality_model: str = os.getenv("OLLAMA_QUALITY_MODEL") or os.getenv("QUALITY_MODEL", "qwen3:8b")
    llm_backend: str = os.getenv("LLM_BACKEND", "ollama")
    supervisor_version: str = os.getenv("SUPERVISOR_VERSION", "v2")
    openai_fast_model: str = os.getenv("OPENAI_FAST_MODEL", "gpt-4o-mini")
    openai_quality_model: str = os.getenv("OPENAI_QUALITY_MODEL", "gpt-4o")
    fault_inject_transient: bool = _bool("FAULT_INJECT_TRANSIENT", False)
    hitl_enabled: bool = _bool("HITL_ENABLED", False)
    summarize_after_messages: int = int(os.getenv("SUMMARIZE_AFTER_MESSAGES", "10"))
    keep_recent_messages: int = int(os.getenv("KEEP_RECENT_MESSAGES", "6"))
    llm_enabled: bool = _bool("LLM_ENABLED", True)
    llm_timeout_s: float = float(os.getenv("LLM_TIMEOUT_S", "120"))
    tool_timeout_s: float = float(os.getenv("TOOL_TIMEOUT_S", "20"))
    keep_alive: str = os.getenv("LLM_KEEP_ALIVE", "30m")
    analysis_tier: str = os.getenv("ANALYSIS_TIER", "quality")
    merge_analysis: bool = _bool("MERGE_ANALYSIS", False)
    compact_facts: bool = _bool("COMPACT_FACTS", True)
    num_predict_analysis: int = int(os.getenv("NUM_PREDICT_ANALYSIS", "220"))
    num_predict_writing: int = int(os.getenv("NUM_PREDICT_WRITING", "700"))
    checkpointer: str = os.getenv("CHECKPOINTER", "memory")
    sqlite_path: str = os.getenv("SQLITE_PATH", str(ROOT / "data" / "checkpoints.sqlite"))
    database_url: str = os.getenv("DATABASE_URL", "")
    max_replan_rounds: int = int(os.getenv("MAX_REPLAN_ROUNDS", "1"))
    max_validation_retries: int = int(os.getenv("MAX_VALIDATION_RETRIES", "1"))
    validator_judge: bool = _bool("VALIDATOR_JUDGE", False)
    recursion_limit: int = int(os.getenv("GRAPH_RECURSION_LIMIT", "40"))
    external_dir: Path = field(default_factory=lambda: Path(os.getenv("EXTERNAL_DIR", ROOT / "data" / "external")))
    embed_model: str = os.getenv("EMBED_MODEL", "bge-m3:latest")
    embed_timeout_s: float = float(os.getenv("EMBED_TIMEOUT_S", "3"))
    embed_build_timeout_s: float = float(os.getenv("EMBED_BUILD_TIMEOUT_S", "60"))
    embed_enabled: bool = _bool("EMBED_ENABLED", True)
    cors_origins: tuple = tuple(os.getenv("CORS_ORIGINS", "http://localhost:8600,http://127.0.0.1:8600,http://localhost:5173,http://127.0.0.1:5173").split(","))


settings = Settings()

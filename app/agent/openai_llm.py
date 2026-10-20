"""OpenAI 백엔드 (`LLM_BACKEND=openai`): 실제 원격 API 를 호출한다 — **호출마다 과금된다**.

책임: Ollama 백엔드와 같은 `LangChainLLM` 공통 구현 위에 OpenAI 채팅 모델(fast=gpt-4o-mini, quality=gpt-4o 기본)을 얹는다.
테스트·CI 에서는 절대 만들지 않는다 (`make_llm("openai")` 는 사용자가 명시적으로 고른 경우에만; 실행 스크립트 `sh/run_openai.sh` 는 y/N 확인을 받는다).
협력 모듈: `llm.py`(공통 구현·팩토리), `usage.py`(토큰·참고 비용), `config.py`(모델명).
사전 조건: `pip install langchain-openai` (requirements.txt 에 고정), 환경변수 `OPENAI_API_KEY`, 그리고 과금 확인용 `OPENAI_CONFIRM=yes`(없으면 시작 거부). 키는 코드·로그·트레이스에 남기지 않는다.
"""
import os

from app.agent.llm import LangChainLLM
from app.config import settings


class OpenAILLM(LangChainLLM):
    """OpenAI 채팅 모델 기반 구현 (fast/quality 두 모델)."""

    def __init__(self) -> None:
        """모델을 만든다. 키/패키지가 없으면 즉시 RuntimeError (나중에 호출 중간에 실패하지 않게)."""
        if os.getenv("OPENAI_CONFIRM") != "yes":  # 과금 방어선: 공유 키 사고를 막기 위해 명시적 확인 없이는 시작하지 않는다 (실행 스크립트가 y/N 뒤에 설정)
            raise RuntimeError("LLM_BACKEND=openai 는 과금되는 원격 호출입니다. 의도한 것이면 OPENAI_CONFIRM=yes 를 설정하세요 (sh/run_agent_openai.sh 는 y/N 확인 뒤 설정)")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("LLM_BACKEND=openai 는 OPENAI_API_KEY 환경변수가 필요합니다 (.env 에 넣고 키를 저장소에 커밋하지 마세요)")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise RuntimeError("langchain-openai 가 설치되어 있지 않습니다: pip install langchain-openai") from e
        common = dict(timeout=settings.llm_timeout_s, max_retries=0, stream_usage=True)  # 재시도는 앱의 with_retry 가 맡는다
        self.model_names = {"fast": settings.openai_fast_model, "quality": settings.openai_quality_model}
        self._models = {
            "fast": ChatOpenAI(model=settings.openai_fast_model, temperature=0, **common),
            "quality": ChatOpenAI(model=settings.openai_quality_model, temperature=0.2, **common),
        }

    def _extra(self, tier, max_tokens):
        """생성 토큰 상한은 OpenAI 의 max_tokens 로 전달한다."""
        return {"max_tokens": max_tokens} if max_tokens else {}

    async def health(self) -> dict:
        """네트워크 호출 없이 설정만 보고한다 (health 폴링이 과금·지연을 만들지 않도록)."""
        return {"enabled": True, "backend": "openai", "reachable": True, "models": sorted(set(self.model_names.values())), "missing": [],
                "note": "원격 API — 도달성은 프로브하지 않음"}

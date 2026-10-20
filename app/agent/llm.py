"""LLM 계층: 모델 티어링(fast/quality), 구조화 출력, 타임아웃, 실패 시 None 반환.

책임: Ollama 호출을 `LLM` 프로토콜 뒤로 숨기고, 어떤 실패(연결·타임아웃·파싱)도 예외 대신 None 으로 바꿔 호출 측이 결정론적
폴백으로 넘어가게 한다. 프롬프트 내용·폴백 선택은 `nodes.py` 의 몫이다.
협력 모듈: `nodes.py`(소비), `retry.py`(구조화 호출 재시도), `api/main.py`(health 조회).
설계 근거: 모델 티어링·지연, 로컬 Ollama, 실패 시 폴백.
테스트는 `LLM` 프로토콜을 만족하는 FakeLLM / NullLLM 을 주입하므로 Ollama 없이 돈다.
"""
import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Literal, Protocol, TypeVar

import httpx
from langchain_core.messages import AIMessageChunk
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from app.agent.retry import with_retry
from app.agent.usage import estimate_tokens, note_failure, record
from app.config import settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
Tier = Literal["fast", "quality"]
TokenCb = Callable[[str], Awaitable[None]]

# route/plan/judge 는 짧은 JSON 이라 전체 타임아웃(120s)을 줄 이유가 없다. Ollama 가 멈췄을 때 사용자를 오래 붙잡지 않는다.
STRUCTURED_TIMEOUT_S = 30
HEALTH_TTL_S = 5.0  # /health 폴링이 Ollama 를 두드리지 않도록 결과를 짧게 캐시
HEALTH_TIMEOUT_S = 1.5


class LLM(Protocol):
    """노드가 의존하는 LLM 인터페이스. 모든 호출은 실패 시 예외 대신 None 을 반환해야 한다."""

    enabled: bool

    async def structured(self, tier: Tier, name: str, schema: type[T], system: str, user: str) -> T | None:
        """pydantic 스키마로 구조화된 응답을 받는다. 실패하면 None."""
        ...

    async def text(
        self, tier: Tier, name: str, system: str, user: str, on_token: TokenCb | None = None, max_tokens: int | None = None
    ) -> str | None:
        """자유 텍스트를 생성한다. on_token 이 있으면 토큰 조각마다 호출한다. 실패하면 None."""
        ...


class NullLLM:
    """LLM_ENABLED=False 용 구현: 모든 호출이 None → 그래프 전체가 결정론적 폴백으로 동작한다 (테스트·장애 대응)."""

    enabled = False

    async def structured(self, tier, name, schema, system, user):
        """(NullLLM) 항상 None — 구조화 호출을 하지 않는다."""
        return None

    async def text(self, tier, name, system, user, on_token=None, max_tokens=None):
        """(NullLLM) 항상 None — 텍스트 생성을 하지 않는다."""
        return None

    async def health(self) -> dict:
        """(NullLLM) 비활성 상태를 그대로 보고한다."""
        return {"enabled": False, "reachable": False, "models": [], "missing": []}


class LangChainLLM:
    """LangChain 채팅 모델 기반 백엔드의 공통 구현 (Ollama·OpenAI 가 상속). fast/quality 두 모델을 보유한다.

    어떤 실패(연결·타임아웃·파싱)도 예외 대신 None 으로 바꾸고, 호출마다 사용량(토큰·지연)을 `usage.record` 로 남긴다.
    서브클래스는 `_models`(tier → 채팅 모델), `model_names`(tier → 이름), `_extra`(호출별 옵션), `health` 만 정의한다.
    """

    enabled = True
    _models: dict
    model_names: dict

    def _cfg(self, tier: Tier, name: str) -> dict:
        # LangSmith 트레이스에서 단계를 식별할 수 있도록 호출별 run_name/tags 를 단다.
        return {"run_name": name, "tags": [f"tier:{tier}", name]}

    def _extra(self, tier: Tier, max_tokens: int | None) -> dict:
        """호출별 생성 옵션 (백엔드마다 다르다). 기본은 없음."""
        return {}

    @staticmethod
    def _usage_of(msg) -> tuple[int, int] | None:
        """AIMessage(Chunk) 의 공급자 보고 사용량 (input, output) — 없으면 None."""
        um = getattr(msg, "usage_metadata", None)
        return (um.get("input_tokens", 0), um.get("output_tokens", 0)) if um else None

    async def structured(self, tier, name, schema, system, user):
        """구조화 응답. transient 오류는 1회 재시도하고, 그 외 모든 실패(파싱 포함)는 None.

        Args:
            tier: 사용할 모델 티어.
            name: 트레이스·사용량 기록용 단계 이름 (예: "route", "plan").
            schema: 응답 pydantic 스키마 (json_schema 형식 제약으로 전달).
            system: 시스템 프롬프트.
            user: 사용자 프롬프트.
        """
        runnable = self._models[tier].with_structured_output(schema, method="json_schema", include_raw=True).with_config(self._cfg(tier, name))
        started = time.perf_counter()

        async def call():
            """타임아웃을 건 구조화 호출 1회."""
            return await asyncio.wait_for(runnable.ainvoke([("system", system), ("user", user)]), min(settings.llm_timeout_s, STRUCTURED_TIMEOUT_S))

        try:
            out = await with_retry(call, attempts=2)
        except Exception as e:  # 연결·타임아웃 오류 모두 호출 측 폴백 대상이다. CancelledError 는 잡지 않는다.
            log.warning("LLM structured 실패(%s/%s): %s: %s", tier, name, type(e).__name__, e)
            note_failure("structured_error")
            return None
        u = self._usage_of(out.get("raw")) or (estimate_tokens(system, user), estimate_tokens(str(out.get("parsed"))))
        record(name, tier, self.model_names[tier], u[0], u[1], started, estimated=self._usage_of(out.get("raw")) is None)
        if out.get("parsing_error") or out.get("parsed") is None:
            log.warning("LLM structured 파싱 실패(%s/%s): %s", tier, name, out.get("parsing_error"))
            note_failure("structured_parse")
            return None
        return out["parsed"]

    async def text(self, tier, name, system, user, on_token=None, max_tokens=None):
        """텍스트를 스트리밍 생성한다.

        Args:
            tier: 사용할 모델 티어.
            name: 트레이스·사용량 기록용 단계 이름.
            system: 시스템 프롬프트.
            user: 사용자 프롬프트.
            on_token: 토큰 조각 콜백 (SSE token 이벤트로 이어진다).
            max_tokens: 생성 토큰 상한. None 이면 모델 기본.

        Returns:
            생성된 전체 텍스트. 빈 응답이나 실패(타임아웃·연결)는 None — 이미 일부 토큰을 내보냈더라도 호출 측이
            reset 토큰으로 교체한다.
        """
        model = self._models[tier].with_config(self._cfg(tier, name))
        extra = self._extra(tier, max_tokens)
        started = time.perf_counter()
        seen: list[tuple[int, int]] = []

        async def call() -> str:
            """토큰을 스트리밍하며 전체 텍스트를 모으는 호출 1회."""
            parts: list[str] = []
            async for chunk in model.astream([("system", system), ("user", user)], **extra):
                piece = chunk.content if isinstance(chunk, AIMessageChunk) else str(chunk.content)
                if u := self._usage_of(chunk):
                    seen.append(u)
                if piece:
                    parts.append(piece)
                    if on_token:
                        await on_token(piece)
            return "".join(parts)

        try:
            # 스트리밍이 시작된 뒤 재시도하면 토큰이 중복 방출되므로 재시도 없이 전체 타임아웃만 건다.
            out = (await asyncio.wait_for(call(), settings.llm_timeout_s)).strip()
        except Exception as e:
            log.warning("LLM text 실패(%s/%s): %s: %s", tier, name, type(e).__name__, e)
            note_failure("text_error")
            return None
        # 공급자가 청크마다 누적치를 주는지 마지막에 한 번만 주는지 달라서, 가장 큰 값(= 총량)을 쓴다.
        u = (max(i for i, _ in seen), max(o for _, o in seen)) if seen else (estimate_tokens(system, user), estimate_tokens(out))
        record(name, tier, self.model_names[tier], u[0], u[1], started, estimated=not seen)
        return out or None


class OllamaLLM(LangChainLLM):
    """로컬 Ollama 기반 구현. fast(소형, 라우팅·계획·해석)와 quality(대형, 리포트 작성) 두 모델을 보유한다."""

    def __init__(self) -> None:
        # quality(qwen3)는 reasoning=False 로 thinking 을 끈다: 지연·토큰이 크게 늘고 리포트 품질 이득이 없었다.
        # fast 는 라우팅·계획의 재현성을 위해 temperature 0.
        common = dict(
            base_url=settings.ollama_base_url, client_kwargs={"timeout": settings.llm_timeout_s}, keep_alive=settings.keep_alive or None
        )
        self._temp = {"fast": 0.0, "quality": 0.2}
        self._health: tuple[float, dict] | None = None
        self.model_names = {"fast": settings.fast_model, "quality": settings.quality_model}
        self._models = {
            # qwen3 계열(예: qwen3:0.6b)은 thinking 을 끈다. thinking 미지원 모델(qwen2.5)에 think 옵션을 보내면 오류이므로 qwen3 에만 지정한다.
            "fast": ChatOllama(model=settings.fast_model, temperature=0, **({"reasoning": False} if settings.fast_model.startswith("qwen3") else {}), **common),
            "quality": ChatOllama(model=settings.quality_model, temperature=0.2, reasoning=False, **common),
        }

    def _extra(self, tier, max_tokens):
        """options 를 넘기면 모델 기본 옵션이 통째로 대체되므로 temperature 를 함께 명시해야 설정이 유지된다."""
        return {"options": {"temperature": self._temp[tier], "num_predict": max_tokens}} if max_tokens else {}

    async def health(self) -> dict:
        """Ollama 도달성과 필수 모델(fast/quality) 설치 여부를 확인한다. 결과는 HEALTH_TTL_S 초 캐시.

        Returns:
            {"enabled": True, "backend": "ollama", "reachable": bool, "models": [설치돼 있는 필수 모델명], "missing": [없는 필수 모델명]}.
            네트워크 오류는 예외 없이 reachable=False 로 표현한다.
        """
        if self._health and time.monotonic() - self._health[0] < HEALTH_TTL_S:
            return self._health[1]
        need = {settings.fast_model, settings.quality_model}
        try:
            async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT_S) as c:
                r = await c.get(f"{settings.ollama_base_url}/api/tags")
                r.raise_for_status()
            installed = {m["name"] for m in r.json().get("models", [])}
            out = {"enabled": True, "backend": "ollama", "reachable": True, "models": sorted(need & installed), "missing": sorted(need - installed)}
        except (httpx.HTTPError, ValueError):
            out = {"enabled": True, "backend": "ollama", "reachable": False, "models": [], "missing": sorted(need)}
        self._health = (time.monotonic(), out)
        return out


class TieredLLM:
    """fast_llm / quality_llm 을 묶어 `LLM` 프로토콜로 보이게 하는 디스패처 (DI 번들 `AgentDeps` 의 `llm`).

    노드는 `llm.structured("fast", …)` 처럼 티어를 문자열로 고르므로, 티어에 맞는 쪽으로 위임한다.
    두 인자가 같은 객체여도 된다(테스트 더블·단일 백엔드).
    """

    def __init__(self, fast: LLM, quality: LLM) -> None:
        self.fast, self.quality = fast, quality

    @property
    def enabled(self) -> bool:
        """두 티어 모두 쓸 수 있을 때만 True."""
        return bool(self.fast.enabled and self.quality.enabled)

    def _pick(self, tier: Tier) -> LLM:
        return self.fast if tier == "fast" else self.quality

    async def structured(self, tier, name, schema, system, user):
        """티어에 맞는 백엔드로 위임."""
        return await self._pick(tier).structured(tier, name, schema, system, user)

    async def text(self, tier, name, system, user, on_token=None, max_tokens=None):
        """티어에 맞는 백엔드로 위임."""
        return await self._pick(tier).text(tier, name, system, user, on_token, max_tokens)

    async def health(self) -> dict:
        """두 백엔드 상태를 합친다 (같은 객체면 한 번만 조회). 백엔드가 health 를 지원하지 않으면 enabled 로만 보고한다."""
        parts = [self.fast] if self.fast is self.quality else [self.fast, self.quality]
        outs = [await p.health() if hasattr(p, "health") else {"enabled": bool(p.enabled), "reachable": bool(p.enabled), "models": [], "missing": []} for p in parts]
        return {
            "enabled": all(o["enabled"] for o in outs), "backend": outs[0].get("backend"), "reachable": all(o["reachable"] for o in outs),
            "models": sorted({m for o in outs for m in o["models"]}), "missing": sorted({m for o in outs for m in o["missing"]}),
        }


def make_llm(backend: str | None = None) -> LLM:
    """설정(`LLM_BACKEND`, `LLM_ENABLED`)에 따라 LLM 백엔드를 만든다.

    Args:
        backend: "fake" | "ollama" | "openai". None 이면 `settings.llm_backend`. `LLM_ENABLED=false` 면 항상 NullLLM(결정론 폴백).

    Returns:
        fast/quality 두 티어를 모두 처리하는 백엔드. openai 는 *실제 과금 호출*이므로 테스트에서 절대 만들지 않는다.

    Raises:
        ValueError: 알 수 없는 백엔드. RuntimeError: openai 선택 시 OPENAI_API_KEY 또는 패키지 누락.
    """
    if not settings.llm_enabled:
        return NullLLM()
    backend = (backend or settings.llm_backend).lower()
    if backend == "ollama":
        return OllamaLLM()
    if backend == "fake":
        from app.agent.fake_llm import FakeBackend

        return FakeBackend()
    if backend == "openai":
        from app.agent.openai_llm import OpenAILLM

        return OpenAILLM()
    raise ValueError(f"알 수 없는 LLM_BACKEND: {backend!r} (fake|ollama|openai)")


def default_llm() -> LLM:
    """설정에 따른 기본 LLM (`make_llm()` 의 별칭 — 기존 호출부 호환)."""
    return make_llm()

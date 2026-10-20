"""LLM 호출 사용량 계량: 요청 단위로 호출 수·토큰·지연·(원격 API일 때) 비용을 모은다.

책임: 백엔드(`llm.py`, `fake_llm.py`, `openai_llm.py`)가 호출 직후 `record()` 로 기록하고, API/평가 하네스가 `usage_scope()` 로 요청
범위를 열어 합계를 읽는다. 노드 코드는 사용량을 몰라도 된다 (ContextVar 로 암묵 전달 — 호출 수가 아니라 *요청*이 집계 단위이기 때문).
협력 모듈: `llm.py`·`fake_llm.py`·`openai_llm.py`(기록), `api/main.py`·`eval/compare_versions.py`(범위 열기·읽기), `nodes.py`(최종 상태에 요약 기록).
설계 근거: Before/After 비교(v1 vs v2)에서 "LLM 호출 수와 토큰이 얼마나 늘었는가"를 같은 잣대로 재기 위함이다.

ContextVar 의 값은 가변 객체(UsageMeter)라서, LangGraph 가 노드마다 복사하는 컨텍스트에서도 같은 미터를 공유한다.
`usage_scope()` 밖에서는 기록이 조용히 무시된다 (테스트·스크립트에서 계량이 필요 없을 때 비용 0).
"""
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Iterator

# 참고용 단가(USD / 100만 토큰, (입력, 출력)). 공급자 가격표는 바뀔 수 있으므로 *추정치*로만 쓰고 보고서에 "참고 단가"라고 밝힌다.
PRICES_USD_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """참고 단가로 비용을 추정한다. 단가표에 없는 모델(로컬 Ollama·fake)은 0."""
    p_in, p_out = PRICES_USD_PER_1M.get(model, (0.0, 0.0))
    return round((input_tokens * p_in + output_tokens * p_out) / 1_000_000, 6)


@dataclass
class CallRecord:
    """LLM 호출 1회의 기록.

    Attributes:
        name: 단계 이름 (route, plan, analysis, writing, simple_answer, validator_judge, v1_writing …).
        tier: "fast" | "quality".
        model: 모델 식별자.
        input_tokens / output_tokens: 토큰 수. estimated=True 면 공급자 값이 없어 글자 수/4 로 추정한 값이다.
        latency_s: 호출 시작~종료 시간(초). 스트리밍은 마지막 토큰까지.
        estimated: 토큰 수가 추정치인가.
    """

    name: str
    tier: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    estimated: bool = False


@dataclass
class UsageMeter:
    """한 요청 동안의 호출 기록 모음."""

    calls: list[CallRecord] = field(default_factory=list)
    retries: int = 0  # 이 요청에서 일어난 transient 재시도 횟수(도구·LLM 합계)
    failures: dict[str, int] = field(default_factory=dict)  # LLM 호출 실패 종류별 횟수: structured_parse(JSON 스키마 파싱 실패)·structured_error·text_error

    def add(self, rec: CallRecord) -> None:
        """호출 기록을 추가한다."""
        self.calls.append(rec)

    def summary(self) -> dict:
        """합계 요약: 호출 수, 토큰, 티어별 분해, 추정 비용, 호출 목록(사람이 읽는 용도).

        Returns:
            {"llm_calls", "input_tokens", "output_tokens", "total_tokens", "cost_usd", "llm_seconds", "by_tier", "calls"}.
            cost_usd 는 원격 API 모델만 0 보다 크다 (참고 단가 기준 추정치).
        """
        by_tier: dict[str, dict] = {}
        for c in self.calls:
            t = by_tier.setdefault(c.tier, {"calls": 0, "tokens": 0})
            t["calls"] += 1
            t["tokens"] += c.input_tokens + c.output_tokens
        i = sum(c.input_tokens for c in self.calls)
        o = sum(c.output_tokens for c in self.calls)
        return {
            "retries": self.retries, "failures": dict(self.failures), "llm_calls": len(self.calls), "input_tokens": i, "output_tokens": o, "total_tokens": i + o,
            "cost_usd": round(sum(cost_usd(c.model, c.input_tokens, c.output_tokens) for c in self.calls), 6),
            "llm_seconds": round(sum(c.latency_s for c in self.calls), 2), "by_tier": by_tier,
            "calls": [asdict(c) for c in self.calls],
        }


_current: ContextVar[UsageMeter | None] = ContextVar("usage_meter", default=None)


@contextmanager
def usage_scope() -> Iterator[UsageMeter]:
    """요청 범위를 열어 그 안에서 일어난 모든 LLM 호출(하위 태스크 포함)을 하나의 미터에 모은다."""
    meter = UsageMeter()
    token = _current.set(meter)
    try:
        yield meter
    finally:
        _current.reset(token)


def current_summary() -> dict | None:
    """현재 범위의 요약 (범위 밖이면 None). 최종 노드가 상태·done 이벤트에 싣는 데 쓴다."""
    meter = _current.get()
    return meter.summary() if meter else None


def note_retry() -> None:
    """재시도 1회를 현재 범위에 기록한다 (`retry.with_retry` 가 호출; 범위 밖이면 무시)."""
    meter = _current.get()
    if meter is not None:
        meter.retries += 1


def note_failure(kind: str) -> None:
    """LLM 호출 실패 1건을 종류별로 센다 (범위 밖이면 무시). 소형 모델이 JSON 스키마를 못 지키는 빈도를 재는 데 쓴다."""
    meter = _current.get()
    if meter is not None:
        meter.failures[kind] = meter.failures.get(kind, 0) + 1


def record(name: str, tier: str, model: str, input_tokens: int, output_tokens: int, started: float, estimated: bool = False) -> None:
    """호출 1회를 현재 범위에 기록한다 (범위가 없으면 무시).

    Args:
        started: 호출 시작 시각(`time.perf_counter()`), 종료 시각은 이 함수가 읽는다.
    """
    meter = _current.get()
    if meter is not None:
        meter.add(CallRecord(name, tier, model, int(input_tokens), int(output_tokens), round(time.perf_counter() - started, 3), estimated))


def estimate_tokens(*texts: str) -> int:
    """글자 수 기반 토큰 추정 (한국어 혼합 텍스트에서 글자 수/2~4 사이라 보수적으로 /3). 공급자 사용량이 없을 때만 쓴다."""
    return sum(len(t) for t in texts) // 3 + 1

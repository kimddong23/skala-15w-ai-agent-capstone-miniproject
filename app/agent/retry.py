"""일시적(transient) 오류 재시도 유틸.

책임: 타임아웃·연결 계열 오류에만 exponential backoff + full jitter 로 재시도한다. 그 외 오류는 즉시 전파한다.
협력 모듈: `nodes.py`(MCP 도구 호출·리소스 로드), `llm.py`(구조화 출력 호출).
설계 근거 (4계층 오류 처리). 재시도 대상을 좁게 잡는 이유는 잘못된 인자·도구 오류가 재시도해도 같은 결과이기 때문이다.
"""
import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx

from app.agent.usage import note_retry

log = logging.getLogger(__name__)
T = TypeVar("T")

# 타임아웃·연결 계열만 일시 오류로 본다. ValueError/도구가 보고한 {"error"} 등은 재시도 대상이 아니다.
TRANSIENT: tuple[type[BaseException], ...] = (TimeoutError, ConnectionError, httpx.TransportError)


class FaultInjector:
    """옵트인 결함 주입: 켜져 있으면 *처음 한 번* `check()` 가 일시 오류(TimeoutError)를 던진다.

    재시도·오류 격리·retry 횟수 지표(Before/After)를 실제 흐름으로 보여 주기 위한 장치이다 (`FAULT_INJECT_TRANSIENT=1`). 기본은 꺼짐.
    """

    def __init__(self, enabled: bool) -> None:
        self.pending = enabled

    def check(self) -> None:
        """대기 중인 주입이 있으면 한 번만 TimeoutError 를 던진다."""
        if self.pending:
            self.pending = False
            raise TimeoutError("주입된 일시 오류 (FAULT_INJECT_TRANSIENT)")


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retry_on: tuple[type[BaseException], ...] = TRANSIENT,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_retry: Callable[[], None] | None = None,
) -> T:
    """`fn()` 을 최대 `attempts` 번 시도한다.

    n 번째 실패 뒤 대기 시간은 `uniform(0, min(max_delay, base_delay * 2**n))` 초이다 (full jitter: 동시 재시도 폭주 방지).

    Args:
        fn: 매 시도마다 새 코루틴을 만드는 인자 없는 callable (코루틴 객체 재사용 불가이므로 callable 을 받는다).
        attempts: 총 시도 횟수 (1 이면 재시도 없음).
        base_delay: 첫 재시도 대기 상한(초).
        max_delay: 대기 상한(초).
        retry_on: 재시도할 예외 타입 튜플.
        sleep: 대기 함수. 테스트가 시간 소모 없이 백오프를 검증하도록 주입 가능.
        on_retry: 재시도가 일어날 때마다 호출되는 콜백 (태스크별 재시도 횟수를 이벤트에 싣는 데 쓴다). 요청 단위 합계는 usage 미터에도 항상 기록된다.

    Returns:
        `fn()` 의 결과.

    Raises:
        retry_on 에 속하지 않는 예외는 즉시, 속하는 예외는 마지막 시도 실패 시 그대로 전파된다.
    """
    for n in range(attempts):
        try:
            return await fn()
        except retry_on as e:
            if n == attempts - 1:
                raise
            note_retry()
            if on_retry:
                on_retry()
            delay = random.uniform(0, min(max_delay, base_delay * 2**n))
            log.warning("transient 오류 %s — %.2fs 후 재시도 (%d/%d)", type(e).__name__, delay, n + 1, attempts - 1)
            await sleep(delay)
    raise AssertionError("unreachable")  # attempts < 1 로 호출한 경우의 방어

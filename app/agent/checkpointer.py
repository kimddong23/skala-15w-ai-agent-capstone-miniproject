"""Checkpointer Factory: thread_id 단위 대화·분석 상태의 영속화 백엔드 선택.

책임: 설정(CHECKPOINTER)에 따라 memory / sqlite / postgres 체크포인터를 비동기 컨텍스트로 제공한다.
협력 모듈: `api/main.py`(lifespan 에서 1회 생성), `graph.py`(compile 시 주입).
설계 근거 (Checkpointer Factory 와 thread 멀티턴 기억).
"""
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from app.config import settings


@asynccontextmanager
async def make_checkpointer(kind: str | None = None) -> AsyncIterator[BaseCheckpointSaver]:
    """체크포인터를 만들고 컨텍스트 종료 시 연결을 정리한다.

    Args:
        kind: "memory" | "sqlite" | "postgres". None 이면 `settings.checkpointer`.

    Yields:
        LangGraph `BaseCheckpointSaver` 구현체. sqlite 는 `settings.sqlite_path` 파일에 영속화되어 프로세스 재시작 후에도
        같은 thread_id 의 기억이 남는다.

    Raises:
        ValueError: 알 수 없는 kind.
        RuntimeError: postgres 선택 시 선택 의존성(`langgraph-checkpoint-postgres`)이 없을 때.
    """
    kind = kind or settings.checkpointer
    if kind == "memory":
        yield InMemorySaver()
    elif kind == "sqlite":
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        async with AsyncSqliteSaver.from_conn_string(settings.sqlite_path) as saver:
            yield saver
    elif kind == "postgres":
        # 미검증 경로: 이 개발 환경에 Postgres 가 없어 실행해 보지 못했다. 사용 전 통합 테스트가 필요하다.
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        except ImportError as e:
            raise RuntimeError("postgres 체크포인터는 `pip install langgraph-checkpoint-postgres psycopg[binary,pool]` 필요") from e
        async with AsyncPostgresSaver.from_conn_string(settings.database_url) as saver:
            await saver.setup()
            yield saver
    else:
        raise ValueError(f"알 수 없는 CHECKPOINTER: {kind!r} (memory|sqlite|postgres)")

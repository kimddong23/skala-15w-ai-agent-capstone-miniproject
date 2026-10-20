"""의존성 번들 `AgentDeps`: 그래프가 쓰는 모든 외부 의존성을 한곳에 명시한다 (의존성 주입).

책임: fast/quality LLM, 도구 모음, 체크포인터, 설정, 버전 토글을 하나의 값 객체로 묶어 `build_graph`·`create_app` 에 넘긴다.
노드는 전역/클로저에 암묵적으로 기대지 않고 이 번들만 읽으므로, 테스트는 가짜 번들로 갈아 끼우고 운영은 설정으로 백엔드를 고른다.
협력 모듈: `graph.py`(소비), `nodes.py`·`supervisor_v1.py`(노드), `llm.py`(백엔드 팩토리), `api/main.py`(lifespan 에서 조립).
설계 근거: lifespan 싱글턴·DI. 번들 필드는 fast_llm, quality_llm, toolbox(MCP), checkpointer, settings, version 이다.
"""
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver

from app.agent.llm import LLM, TieredLLM, make_llm
from app.agent.toolbox import ToolBox
from app.config import Settings, settings as default_settings

VERSIONS = ("v1", "v2")


@dataclass
class AgentDeps:
    """에이전트 그래프의 의존성 묶음.

    Attributes:
        fast_llm: 라우팅·계획·해석 같은 짧은 단계용 저비용 모델.
        quality_llm: 리포트 작성·후속 답변용 고품질 모델. fast 와 같은 객체여도 된다(백엔드가 티어 인자로 모델을 고른다).
        toolbox: 도구 모음 (운영은 MCP, 테스트는 fake).
        checkpointer: thread 단위 상태 저장소 (None 이면 멀티턴 기억 없음).
        settings: 이 번들이 쓰는 설정 스냅샷 (게이트·재시도 상한 등).
        version: "v2"(기본) 또는 "v1"(Before/After 비교용 선형 기준선).
        extras: 선택 확장 지점 (예: 향후 알림 승인 핸들러).
    """

    fast_llm: LLM
    quality_llm: LLM
    toolbox: ToolBox
    checkpointer: BaseCheckpointSaver | None = None
    settings: Settings = default_settings
    version: str = "v2"
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """버전 값을 검증한다 (오타로 조용히 v2 가 돌지 않게)."""
        if self.version not in VERSIONS:
            raise ValueError(f"알 수 없는 SUPERVISOR_VERSION: {self.version!r} ({'|'.join(VERSIONS)})")

    @property
    def llm(self) -> TieredLLM:
        """노드가 티어 문자열로 호출하는 통합 LLM (fast_llm/quality_llm 으로 위임)."""
        return TieredLLM(self.fast_llm, self.quality_llm)

    @classmethod
    def from_parts(
        cls, toolbox: ToolBox, checkpointer: BaseCheckpointSaver | None = None, llm: LLM | None = None, *, version: str | None = None
    ) -> "AgentDeps":
        """기존 `(tools, checkpointer, llm)` 호출 형태에서 번들을 만든다.

        Args:
            llm: 두 티어를 모두 처리하는 LLM 하나. None 이면 설정(`LLM_BACKEND`)에 따른 기본 백엔드.
            version: None 이면 `settings.supervisor_version`.
        """
        backend = llm if llm is not None else make_llm()
        return cls(backend, backend, toolbox, checkpointer, default_settings, version or default_settings.supervisor_version)

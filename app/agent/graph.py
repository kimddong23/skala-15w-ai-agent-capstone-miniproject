"""Supervisor 그래프 조립.

책임: 노드(`AgentNodes`)와 조건부 엣지를 아래 도식대로 연결해 컴파일한다. 비즈니스 로직은 두지 않는다.
협력 모듈: `nodes.py`(노드 구현), `toolbox.py`(주입되는 도구), `llm.py`(주입되는 LLM), `checkpointer.py`(영속화).
설계 근거: 11개 노드의 명시적 파이프라인, 재계획 루프, 검증 재시도 루프.

    START → route ─simple→ simple_answer → END
                 └analysis→ plan → gate → research ─(replan↺ | analysis | safety_gate)
                            → safety_gate → writing → validator ─fail&재시도<MAX→ writing
                                                                 └그 외→ finalize → END

루프는 두 개뿐이며 둘 다 코드로 상한이 있다 (replan: MAX_REPLAN_ROUNDS·30일 clamp, writing: MAX_VALIDATION_RETRIES).
호출 측은 추가로 `recursion_limit` 을 걸어 설계 밖의 무한 루프를 차단한다.
"""
import os
from dataclasses import replace

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.deps import AgentDeps
from app.agent.llm import LLM
from app.agent.nodes import AgentNodes
from app.agent.state import AgentState
from app.agent.toolbox import ToolBox  # noqa: F401  (타입 힌트)


def build_graph(
    tools: "ToolBox | AgentDeps", checkpointer: BaseCheckpointSaver | None = None, llm: LLM | None = None, *, version: str | None = None
) -> CompiledStateGraph:
    """에이전트 그래프를 컴파일한다. `SUPERVISOR_VERSION`(또는 `version`/`deps.version`)으로 v1(기준선)/v2(기본)를 고른다.

    Args:
        tools: `AgentDeps` 번들, 또는 도구 모음(`ToolBox`) — 후자는 `checkpointer`·`llm` 과 함께 번들로 자동 조립된다.
        checkpointer: thread 단위 상태 저장소 (ToolBox 로 호출할 때만 의미 있음).
        llm: 두 티어를 처리하는 LLM (ToolBox 로 호출할 때만). None 이면 `LLM_BACKEND` 설정.
        version: "v1" | "v2". None 이면 번들/설정 값.

    Returns:
        컴파일된 그래프. 입력은 `{"messages": [...], "user_id"?: str}`, config 에 `configurable.thread_id` 필요.
        v1 도 같은 상태·SSE 계약을 쓴다(UI 호환) — 단 라우팅·게이트·안전 게이트·검증·재계획이 없다.
    """
    deps = tools if isinstance(tools, AgentDeps) else AgentDeps.from_parts(tools, checkpointer, llm, version=version)
    if version and version != deps.version:
        deps = replace(deps, version=version)
    if deps.version == "v1":
        # v1 은 안전 게이트·검증이 없는 비교용 기준선이다: 실수로 서비스에 켜지지 않게 명시적 허용(ALLOW_V1_BASELINE=yes)을 요구한다.
        if os.getenv("ALLOW_V1_BASELINE") != "yes":
            raise RuntimeError("SUPERVISOR_VERSION=v1 은 Before/After 비교용 기준선이라 안전 게이트·검증이 없습니다. 비교 실험이 목적이면 "
                               "ALLOW_V1_BASELINE=yes 를 함께 설정하세요. (서비스는 v2 를 쓰세요)")
        from app.agent.supervisor_v1 import build_v1_graph

        return build_v1_graph(deps)
    n = AgentNodes(deps)
    g = StateGraph(AgentState)
    for name in ("route", "simple_answer", "plan", "gate", "research", "replan", "analysis", "safety_gate", "writing", "validator", "finalize", "notify_guardian"):
        g.add_node(name, getattr(n, name))

    g.add_edge(START, "route")
    g.add_conditional_edges("route", lambda s: s["route"], {"simple": "simple_answer", "analysis": "plan"})
    g.add_edge("simple_answer", END)
    g.add_edge("plan", "gate")
    g.add_edge("gate", "research")
    g.add_conditional_edges("research", n.replan_check, {"replan": "replan", "analysis": "analysis", "safety_gate": "safety_gate"})
    g.add_edge("replan", "research")
    g.add_edge("analysis", "safety_gate")
    g.add_edge("safety_gate", "writing")  # analysis 를 건너뛰어도 safety_gate 는 반드시 거친다
    g.add_edge("writing", "validator")
    g.add_conditional_edges("validator", n.validation_check, {"writing": "writing", "finalize": "finalize"})
    # HITL: 에스컬레이션이 있고 설정이 켜져 있으면 보호자 알림(시뮬레이션) 승인 단계를 거친다. 리포트·안내문은 이 단계 *이전*에 이미 확정되어 있다.
    g.add_conditional_edges("finalize", n.after_finalize, {"notify_guardian": "notify_guardian", "end": END})
    g.add_edge("notify_guardian", END)
    return g.compile(checkpointer=deps.checkpointer)

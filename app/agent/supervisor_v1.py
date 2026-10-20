"""v1 기준선(baseline): 라우팅·게이트·안전 게이트·검증·재계획이 없는 선형 파이프라인.

**실사용 금지.** Before/After 비교(`eval/compare_versions.py`)에서 "v2 의 구조적 장치가 무엇을 사 오는가"를 재기 위한 *대조군*이다.
`SUPERVISOR_VERSION=v1` 로 기동하면 이 구현이 실행되고, 환경변수 하나로 v2 로 되돌아간다.

v1 이 하는 일 (모든 질문 동일):
    route(항상 analysis) → research(요약+탐지, 그대로) → writing(quality 모델 1회, 원시 JSON 사실 전달) → END
v1 에 *없는* 것 (v2 와의 차이 = 측정 대상):
    - 라우팅: 인사·일반 지식 질문도 조사→작성으로 간다 (사용자 미지정이면 데이터 없이 LLM 이 답한다)
    - 이중 게이트: 기간 보정·필수 태스크 보강 없음 (질문 속 기간을 그대로 쓰고 없으면 7일)
    - 안전 게이트·고정 문구: 에스컬레이션 안내/면책/진단 거절은 LLM 이 *쓰면* 있고 안 쓰면 없다. `escalation` 플래그는 항상 False
    - 검증기·재시도·템플릿 대체: LLM 출력을 그대로 내보낸다 (수치 환각·진단 단정·안심 표현 검사 없음)
    - 재계획: 데이터가 부족해도 기간을 늘리지 않는다
    - 모델 티어링: 모든 LLM 호출을 quality 모델 하나로 한다 (fast 모델 미사용)
공정성: v1 도 v2 와 같은 LLM·도구·사실 형식(FACTS)과 합리적 프롬프트(수치 인용·응급 안내·면책 요청)를 받는다 — 허수아비가 아니다.
유지하는 것: 같은 상태·SSE 계약(route_decided, task_*, anomalies_found, token, done), 도구 호출의 transient 재시도·타임아웃, 체크포인터(멀티턴).
협력 모듈: `graph.py`(버전 토글), `deps.py`(번들), `nodes.py`(emit·토큰 방출기 재사용), `gates.py`(사용자 ID·기간 파싱만 사용).
"""
import asyncio
import logging

from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver  # noqa: F401  (타입 힌트용)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent import events as ev
from app.agent import gates
from app.agent import report as rpt
from app.agent.deps import AgentDeps
from app.agent.nodes import _history, _last_user_text, _Tokens, emit
from app.agent.retry import FaultInjector, with_retry
from app.agent.state import AgentState
from app.agent.toolbox import bounded
from app.agent.usage import current_summary

log = logging.getLogger(__name__)

# 공정성: v1 은 "허수아비"가 아니다. v2 와 같은 LLM·같은 도구·같은 사실 형식(압축 FACTS)을 쓰고, 합리적인 시스템 프롬프트(수치 인용·응급 안내·면책 요청)도
# 받는다. 다른 점은 오직 *코드로 강제하는 장치*(라우팅·게이트·안전 게이트·고정 문구·검증·재계획·티어링)가 없다는 것이다 — LLM 이 프롬프트를 지키는지에만 의존한다.
V1_SYSTEM = (
    "당신은 웨어러블 심박 건강 상담 도우미입니다. 아래 FACTS(도구 조회 결과)와 대화 기록을 보고 사용자의 질문에 한국어로 답하세요. "
    "FACTS 에 있는 수치만 인용하고, 질병을 단정 진단하지 말며, 위험 신호가 있으면 119/의료진 상담을 권고하고, 마지막에 의학적 진단이 아니라는 면책을 쓰세요. "
    "FACTS 가 비어 있으면 일반 지식으로 답하고, 점검 대상 사용자가 없으면 사용자 ID 를 물어보세요."
)


class V1Nodes:
    """v1 선형 파이프라인의 노드 (route → research → writing)."""

    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps
        self.tools, self.llm = deps.toolbox, deps.llm
        self._fault = FaultInjector(deps.settings.fault_inject_transient)  # v2 와 같은 조건으로 재시도 지표를 비교하기 위해 v1 에도 동일하게 적용

    async def route(self, state: AgentState) -> dict:
        """라우팅 없음: 항상 analysis. 사용자는 질문 속 ID 또는 요청 본문 값(없으면 None — 존재 여부 검증도 하지 않는다)."""
        uid = gates.extract_user_id(_last_user_text(state)) or state.get("user_id")
        reasoning = "v1: 라우팅 없음 — 모든 질문을 조사→작성 경로로 처리"
        emit(ev.RouteDecided(route="analysis", reasoning=reasoning))
        return {"route": "analysis", "route_reasoning": reasoning, "user_id": uid, "missing_user": False, "unknown_user": None}

    async def research(self, state: AgentState) -> dict:
        """요약 통계와 이상 탐지를 병렬로 조회한다. 기간은 질문 속 값 또는 7일(게이트·clamp·재계획 없음). 사용자가 없으면 조회하지 않는다."""
        uid = state.get("user_id")
        days = gates.extract_lookback_days(_last_user_text(state)) or 7.0
        if not uid:
            return {"summary": None, "detection": None, "anomalies": [], "research_errors": [], "lookback_days": days}

        async def run(tool: str, task_id: str):
            """도구 1회 호출 (실패는 이벤트만 남기고 None)."""
            fn = self.tools.summary if tool == "summary" else self.tools.detect
            emit(ev.TaskStarted(task_id=task_id, task_type="research"))
            retries = [0]

            def call():
                """재시도 대상 호출 1회 (주입된 결함이 있으면 먼저 던진다)."""
                self._fault.check()
                return bounded(fn(uid, days, None))

            try:
                res = await with_retry(call, on_retry=lambda: retries.__setitem__(0, retries[0] + 1))
                if isinstance(res, dict) and res.get("error"):
                    raise RuntimeError(res["error"])
                emit(ev.TaskCompleted(task_id=task_id, task_type="research", summary=f"{tool} 조회 완료", retries=retries[0]))
                return res
            except Exception as e:
                log.warning("v1 research 실패 %s: %s", task_id, e)
                emit(ev.TaskFailed(task_id=task_id, task_type="research", error="조회에 실패했습니다"))
                return None

        summary, detection = await asyncio.gather(run("summary", "v1_summary"), run("detect", "v1_detect"))
        anomalies = list((detection or {}).get("anomalies") or [])
        if detection is not None:
            emit(ev.AnomaliesFound(count=len(anomalies), counts_by_type=detection.get("counts") or {}, anomalies=anomalies[:50]))
        return {"summary": summary, "detection": detection, "anomalies": anomalies, "research_errors": [], "lookback_days": days}

    async def writing(self, state: AgentState) -> dict:
        """quality 모델 1회로 답한다. v2 와 같은 압축 FACTS 를 주고, 출력은 검증·고정 문구·템플릿 없이 그대로 쓴다."""
        tokens = _Tokens()
        emit(ev.TaskStarted(task_id="v1_writing", task_type="writing"))
        facts = rpt.facts_brief(state) if state.get("user_id") and (state.get("summary") or state.get("detection")) else "(조회된 데이터 없음)"  # v2 와 같은 사실 형식
        user = f"질문: {_last_user_text(state)}\n\nFACTS:\n{facts}\n\n대화:\n{_history(state)}"
        text = await self.llm.text("quality", "v1_writing", V1_SYSTEM, user, tokens)
        report = text or "응답을 생성하지 못했습니다."  # v1 에는 템플릿 폴백이 없다
        if text is None:
            await tokens.replay(report)
        emit(ev.TaskCompleted(task_id="v1_writing", task_type="writing", summary=f"리포트 {len(report)}자"))
        return {
            "report": report, "messages": [AIMessage(report)], "escalation": False, "escalation_message": "",
            "validation": {"passed": True, "issues": [], "attempt": 0, "degraded": False, "note": "v1: 검증 없음"},
            "usage": current_summary() or {},
        }


def build_v1_graph(deps: AgentDeps) -> CompiledStateGraph:
    """v1 선형 그래프를 컴파일한다: START → route → research → writing → END."""
    n = V1Nodes(deps)
    g = StateGraph(AgentState)
    for name in ("route", "research", "writing"):
        g.add_node(name, getattr(n, name))
    g.add_edge(START, "route")
    g.add_edge("route", "research")
    g.add_edge("research", "writing")
    g.add_edge("writing", END)
    return g.compile(checkpointer=deps.checkpointer)

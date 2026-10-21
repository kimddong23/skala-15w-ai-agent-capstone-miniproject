"""SSE 이벤트 스키마 — 백엔드가 스트림으로 내보내는 모든 이벤트의 단일 정의.

책임: 이벤트 모델(pydantic)과 SSE 프레임 직렬화. 이벤트를 *언제* 내보내는지는 `nodes.py` 가 정한다.
협력 모듈: `nodes.py`(emit), `api/main.py`(done/error 생성, 프레임 변환), 프런트엔드 `src/events.ts`(zod 스키마와 1:1 대응).
설계 근거: SSE, 프런트 zod 검증.

계약: 모든 data JSON 에 `type` 이 있고, SSE 프레임의 `event:` 이름은 항상 `data.type` 과 같다 (불일치 프레임은 프런트가 버린다).
이벤트 필드를 바꾸면 `frontend/src/events.ts` 와 API 계약 문서를 함께 수정해야 한다.
"""
import json
from typing import Any, Literal

from pydantic import BaseModel


class _Event(BaseModel):
    type: str


class TaskItem(BaseModel):
    """계획의 한 태스크. type 은 research(데이터 조회) | analysis(해석) | writing(리포트 작성)."""

    id: str
    type: Literal["research", "analysis", "writing"]
    description: str


class RouteDecided(_Event):
    """route 노드가 simple/analysis 를 결정했다. reasoning 에 LLM 근거 또는 코드 보정 사유가 들어간다."""

    type: Literal["route_decided"] = "route_decided"
    route: Literal["simple", "analysis"]
    reasoning: str


class PlanCreated(_Event):
    """plan 노드가 (게이트 적용 전) 계획을 만들었다."""

    type: Literal["plan_created"] = "plan_created"
    tasks: list[TaskItem]
    needs_analysis: bool
    lookback_days: float


class GateApplied(_Event):
    """이중 게이트 결과: 제거된 태스크, analysis 강제 여부, 변경 사유, 확정된 조회 기간."""

    type: Literal["gate_applied"] = "gate_applied"
    removed_tasks: list[str]
    forced_analysis: bool
    reason: str
    lookback_days: float | None = None  # 게이트가 확정한 실제 조회 기간(일). 프런트는 plan_created 보다 이 값을 우선해 표시한다


class TaskStarted(_Event):
    """태스크 시작 알림 (research 는 병렬이라 task_completed/failed 와 순서가 섞일 수 있다)."""
    type: Literal["task_started"] = "task_started"
    task_id: str
    task_type: str


class TaskCompleted(_Event):
    """태스크 성공. summary 는 한 줄 요약(집계 수치만 포함)."""

    type: Literal["task_completed"] = "task_completed"
    task_id: str
    task_type: str
    summary: str
    retries: int = 0  # 이 태스크가 성공하기까지 일어난 transient 재시도 횟수 (선택 필드)


class TaskFailed(_Event):
    """태스크 실패. 다른 태스크는 계속 진행되어 부분 결과가 나온다 (오류 격리)."""

    type: Literal["task_failed"] = "task_failed"
    task_id: str
    task_type: str
    error: str


class Replan(_Event):
    """데이터 부족으로 조회 기간을 늘려 research 를 다시 실행한다 (상한 있음)."""

    type: Literal["replan"] = "replan"
    round: int
    reason: str
    lookback_days: float


class AnomaliesFound(_Event):
    """탐지 결과. anomalies 는 심각도 순 상위 N건(집계 evidence 포함, 원시 시계열 없음)."""

    type: Literal["anomalies_found"] = "anomalies_found"
    count: int
    counts_by_type: dict[str, int]
    anomalies: list[dict[str, Any]]


class SafetyEscalation(_Event):
    """안전 게이트가 critical 이상을 감지해 고정 안내문을 강제했다 (LLM 판단과 무관)."""

    type: Literal["safety_escalation"] = "safety_escalation"
    reason: str
    message: str


class Token(_Event):
    """리포트 토큰 조각.

    reset=True 는 "지금까지 받은 토큰을 버리고 여기서부터 새로 시작"을 뜻한다. 검증 재시도, LLM 실패 후 템플릿 대체,
    코드블록 제거 등으로 이미 스트리밍한 초안을 교체해야 할 때 해당 시도의 첫 토큰에 붙는다.
    """

    type: Literal["token"] = "token"
    text: str
    reset: bool = False


class Validation(_Event):
    """validator 결과. attempt 는 1부터. passed=False 면 재시도 또는 템플릿 대체가 뒤따른다."""

    type: Literal["validation"] = "validation"
    passed: bool
    issues: list[str]
    attempt: int
    degraded: bool = False  # 재시도 상한을 넘겨 finalize 가 템플릿으로 대체한다 (선택 필드)


class ApprovalRequired(_Event):
    """에스컬레이션 후 "보호자 알림 발송"(시뮬레이션)에 대한 사람 승인 요청 (HITL). 이 이벤트 뒤에 `done` 이 오며, 승인/거절은 `POST /chat/approve` 로 보낸다.

    에스컬레이션 안내문은 승인과 무관하게 이미 리포트에 들어 있다. 승인하지 않아도 안내는 사라지지 않는다.
    """

    type: Literal["approval_required"] = "approval_required"
    thread_id: str
    reason: str
    message_preview: str
    approval_id: str  # POST /chat/approve 에 되돌려 보낸다 - 이 요청이 아닌 오래된 카드의 결정은 409 로 거절된다


class Done(_Event):
    """스트림 종료. report 는 최종(템플릿 대체 포함) 리포트로, 토큰 누적본보다 항상 우선한다. usage 는 선택 필드(없어도 호환)."""

    type: Literal["done"] = "done"
    thread_id: str
    report: str
    escalation: bool
    run_id: str | None = None  # 이 응답의 실행 ID — POST /feedback 에 되돌려 보내 트레이스에 피드백을 붙인다 (선택 필드)
    usage: dict[str, Any] | None = None  # 이번 요청의 LLM 사용량 요약(호출 수·토큰·지연·참고 비용). 계량 범위 밖이면 None


class ErrorEvent(_Event):
    """처리 실패. 내부 예외 내용은 노출하지 않고 일반 문구만 담는다."""

    type: Literal["error"] = "error"
    message: str


def format_sse(event: dict[str, Any] | BaseModel) -> str:
    """이벤트를 SSE 프레임 문자열로 직렬화한다.

    Args:
        event: 이벤트 모델 또는 `type` 키를 가진 dict.

    Returns:
        ``event: <type>\\ndata: <json>\\n\\n`` 형식 문자열 (한글은 이스케이프하지 않는다).
    """
    data = event.model_dump() if isinstance(event, BaseModel) else event
    return f"event: {data['type']}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"

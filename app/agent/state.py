"""에이전트 그래프 상태 스키마 (LangGraph `StateGraph` 의 state).

책임: 노드 간에 주고받는 키와 그 수명(thread 전체 vs 분석 1회)을 한곳에 정의한다. 로직은 두지 않는다.
협력 모듈: `nodes.py` (읽기/쓰기), `graph.py` (StateGraph 생성), `checkpointer.py` (thread 단위 영속화).
설계 근거: 노드 구성, thread 멀티턴 기억.

수명 규칙:
    - thread 전체에서 유지: messages, user_id, analyzed_user, analyzed_days, report, escalation(+message), validation
    - 분석 1회마다 `plan` 노드가 초기화: tasks 이하 research/analysis/validation 관련 키 전부
    - `analyzed_*` 와 직전 분석 결과(summary/detection/anomalies)는 simple 후속 질문이 재조회 없이 참조하므로 plan 에서만 덮어쓴다.
"""
from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """그래프 상태. 모든 키는 선택적(total=False)이며 노드가 필요한 키만 반환해 병합한다.

    Attributes:
        messages: 대화 기록 (add_messages 리듀서가 누적). 마지막 human 메시지가 현재 질문.
        user_id: 점검 대상 사용자 가명 ID (예: "U05"). 질문 속 ID > 요청 본문 > 이전 턴 값 순으로 결정.
        route: "simple" | "analysis" (route 노드가 결정).
        route_reasoning: 라우팅 근거 한 줄 (LLM 설명 또는 코드 보정 사유).
        missing_user: 분석 의도인데 사용자 미지정 → simple_answer 가 사용자 선택을 안내.
        unknown_user: 존재하지 않는 사용자 ID 를 지정한 경우 그 ID (simple_answer 가 가용 사용자 목록과 함께 안내).
        user_ids: 비교 요청이면 대상 사용자들(2~3명), 단일 요청이면 빈 리스트.
        users: 비교 요청의 사용자별 조회 결과 {uid: {"summary","detection","anomalies","errors"}}. 단일이면 None. 이 값이 있어도
            summary/detection 키에는 첫 사용자의 결과가 들어가고, anomalies 는 모든 사용자의 이상(각 항목에 user_id 추가)이다.
        excluded_users: 한 번에 비교 가능한 인원(3명) 초과로 제외된 ID.
        analyzed_user: 직전 분석의 대상 사용자. None 이면 thread 에 분석 이력이 없다는 뜻.
        analyzed_days: 직전 분석의 최종 조회 기간(일, replan 확대 반영). 기간 미지정 재분석이 이어받는다.
        tasks: 게이트를 통과한 실행 태스크 [{"id","type","description"}].
        needs_analysis: analysis 단계 실행 여부 (이중 게이트의 최종 결과).
        lookback_days: 현재 조회 기간(일, 1~30). replan 이 2배로 늘릴 수 있다.
        replan_round: 지금까지의 재계획 횟수 (상한 MAX_REPLAN_ROUNDS).
        summary: `get_vitals_summary` 결과 (실패 시 None).
        detection: `detect_anomalies` 결과 (실패 시 None).
        guidelines: 가이드라인 리소스 원문 {"ranges","red_flags","disclaimer"}.
        research_errors: 실패한 research 태스크의 오류 문자열 (부분 결과 허용).
        anomalies: detection["anomalies"] 의 사본 (안전 게이트와 리포트의 입력).
        analysis: analysis 노드의 해석 문장.
        escalation: 안전 게이트가 켠 에스컬레이션 여부 (LLM 판단과 무관).
        escalation_message: 고정 에스컬레이션 안내문.
        report: 사용자에게 보여줄 최종(또는 검증 대기 중인) 리포트 마크다운.
        validation: {"passed","issues","attempt","degraded"}. degraded=True 면 템플릿으로 대체된 리포트.
        validation_attempt: validator 실행 횟수.
        validation_issues: 직전 검증 지적사항 (writing 재시도 프롬프트에 전달).
        usage: 이번 요청의 LLM 사용량 요약 (호출 수·토큰·재시도·지연·참고 비용; `usage.UsageMeter.summary()`). 계량 범위 밖이면 비어 있다.
        approval: 보호자 알림(시뮬레이션) 사람 승인 결과 {"status","simulated","message_preview"}. HITL 을 거치지 않았으면 None.
        memory_summary: 오래된 대화의 요약(개인 정보·점검 대상·관심사). 프롬프트의 대화 기록 앞에 붙는다. 안전 정보(escalation·anomalies)는 이 요약에 의존하지 않는다.
        summarized_upto: messages 중 앞에서부터 이 개수까지가 memory_summary 에 반영되었다.
    """

    messages: Annotated[list, add_messages]
    user_id: str | None
    route: str
    route_reasoning: str
    missing_user: bool
    unknown_user: str | None
    user_ids: list[str]
    users: dict[str, dict[str, Any]] | None
    excluded_users: list[str]
    analyzed_user: str | None
    analyzed_days: float | None
    tasks: list[dict[str, Any]]
    needs_analysis: bool
    lookback_days: float
    replan_round: int
    summary: dict[str, Any] | None
    detection: dict[str, Any] | None
    guidelines: dict[str, str]
    research_errors: list[str]
    anomalies: list[dict[str, Any]]
    analysis: str
    escalation: bool
    escalation_message: str
    report: str
    validation: dict[str, Any]
    validation_attempt: int
    validation_issues: list[str]
    usage: dict[str, Any]
    approval: dict[str, Any] | None
    memory_summary: str
    summarized_upto: int

"""HTTP API (FastAPI): 프런트엔드가 쓰는 REST + SSE 엔드포인트.

책임: 요청 검증, 그래프 실행과 SSE 변환, 데이터 조회 엔드포인트(차트용). 에이전트 로직은 `app.agent` 에 있고 여기엔 두지 않는다.
수명: lifespan 에서 MCP 클라이언트·checkpointer·그래프를 1회 만들어 `app.state` 에 두고 요청 간 공유한다.
협력 모듈: `app.agent.graph`(실행), `app.agent.toolbox`(MCP 연결·ToolBox), `app.agent.events`(SSE 직렬화), `app.config`.
설계 근거: SSE, lifespan 싱글턴·DI, 조회 범위 제한.

보안 한계: 인증·인가가 없다. 로컬 데모 전제이므로 `uvicorn --host 127.0.0.1` 로만 띄우고 외부에 노출하지 않는다.
thread_id 는 클라이언트가 고르는 값이라 아는 사람은 누구나 그 대화를 이어갈 수 있다. 공개 배포하려면 인증, 사용자-스레드 결속,
서버 발급 thread_id 가 필요하다.

엔드포인트 계약은 API 계약 문서 참조. 프런트 개발 프록시가 `/api` 접두어를 떼므로 라우트는 루트에 둔다.
"""
import asyncio
import logging
import os
import subprocess
import time
import uuid
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.errors import GraphRecursionError
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.agent import events as ev
from app.agent.checkpointer import make_checkpointer
from dataclasses import replace

from app.agent.deps import AgentDeps
from app.agent.graph import build_graph
from app.agent.llm import LLM, default_llm
from app.agent.usage import usage_scope
from app.agent.toolbox import ToolBox, connect_mcp
from app.config import ROOT, settings

log = logging.getLogger("vital-sentinel.api")
_ID = r"^[A-Za-z0-9_-]{1,64}$"
MCP_PING_TIMEOUT_S = 2.0


def _git_commit() -> str:
    """현재 커밋 해시(짧게). 컨테이너처럼 git 이 없으면 환경변수 GIT_COMMIT, 그것도 없으면 "unknown"."""
    if env := os.getenv("GIT_COMMIT"):
        return env
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=3).stdout.strip()
        return out or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


GIT_COMMIT = _git_commit()


def _langsmith_client():
    """LangSmith 클라이언트 (API 키가 없으면 None → 피드백은 기록 없이 no-op)."""
    if not (os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")):
        return None
    from langsmith import Client

    return Client()


class FeedbackRequest(BaseModel):
    """응답에 대한 사용자 피드백 (엄지 위/아래).

    Attributes:
        run_id: 응답(`/chat`, `done`)이 돌려준 실행 ID.
        score: 1.0(좋음) ~ 0.0(나쁨). 엄지 위/아래는 1 / 0.
        comment: 선택 의견 (최대 500자). LangSmith 로 전송되므로 개인 정보를 쓰지 말라고 UI 가 안내해야 한다.
    """

    run_id: uuid.UUID
    score: float = Field(ge=0.0, le=1.0)
    comment: str | None = Field(default=None, max_length=500)


class ApproveRequest(BaseModel):
    """보호자 알림(시뮬레이션) 승인/거절 요청.

    Attributes:
        thread_id: 승인 대기 중인 대화 (`approval_required` 이벤트로 받은 값).
        approved: True 면 알림 발송(시뮬레이션), False 면 발송하지 않음. 어느 쪽이든 에스컬레이션 안내문은 이미 리포트에 있다.
        approval_id: `approval_required` 가 알려 준 승인 요청 ID (필수). 현재 대기 중인 요청의 ID 와 다르면 409 - 오래된 카드가 같은 대화의 새 요청을
            승인하는 사고를 막는다.
    """

    thread_id: str = Field(pattern=_ID)
    approved: bool
    approval_id: str = Field(min_length=1, max_length=100)


class ChatRequest(BaseModel):
    """채팅 요청 본문.

    Attributes:
        message: 사용자 발화 (1~2000자). 상한은 프롬프트 폭주·남용 방지.
        user_id: 점검 대상 가명 ID. 생략하면 발화 속 ID 또는 thread 의 이전 값을 쓴다.
        thread_id: 대화 식별자. 생략하면 새로 발급하며 응답/`done` 이벤트로 돌려준다. 같은 값이면 멀티턴 기억이 이어진다.
    """

    message: str = Field(min_length=1, max_length=2000)
    user_id: str | None = Field(default=None, pattern=_ID)
    thread_id: str | None = Field(default=None, pattern=_ID)


def create_app(
    toolbox: ToolBox | None = None, llm: LLM | None = None, checkpointer: str | None = None, *, deps: AgentDeps | None = None,
    version: str | None = None,
) -> FastAPI:
    """FastAPI 앱을 만든다.

    Args:
        toolbox: 주입하면 MCP subprocess 를 띄우지 않는다 (테스트용). None 이면 lifespan 에서 stdio MCP 서버를 기동한다.
        llm: LLM 구현. None 이면 설정에 따른 기본값.
        checkpointer: "memory"|"sqlite"|"postgres". None 이면 설정값.
        deps: 의존성 번들을 통째로 주입한다(주어지면 toolbox/llm/checkpointer 인자는 무시, 체크포인터는 번들의 것을 쓴다).
        version: "v1" | "v2". None 이면 번들/설정(`SUPERVISOR_VERSION`) 값.

    Returns:
        lifespan 이 연결된 FastAPI 앱. MCP 연결에 실패해도 앱은 뜨며 데이터 엔드포인트는 503 을 돌려준다 (health 로 확인 가능).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """앱 수명: MCP 연결(또는 주입된 도구)·체크포인터·그래프를 1회 만들어 app.state 에 둔다.

        app.state: llm, tools(None 이면 MCP 미연결), graph, locks(thread 별 직렬화 락), mcp_health(캐시된 핑 결과).
        종료 시 AsyncExitStack 이 체크포인터와 MCP subprocess 를 정리한다.
        """
        async with AsyncExitStack() as stack:
            toolbox_ = deps.toolbox if deps is not None else toolbox
            tools, mcp_ok = toolbox_, toolbox_ is not None
            if tools is None:
                try:
                    tools, mcp_ok = await connect_mcp(stack), True
                except Exception:
                    log.exception("MCP 서버 연결 실패 — 데이터 엔드포인트는 503 을 반환한다")
            saver = deps.checkpointer if deps is not None else await stack.enter_async_context(make_checkpointer(checkpointer))
            if deps is not None:
                bundle = replace(deps, version=version) if version else deps
            elif mcp_ok:
                bundle = AgentDeps.from_parts(tools, saver, llm, version=version)
            app.state.deps = bundle if mcp_ok else None
            app.state.llm = bundle.llm if mcp_ok else (llm or default_llm())
            app.state.tools = tools if mcp_ok else None
            app.state.graph = build_graph(bundle) if mcp_ok else None
            app.state.locks = {}
            app.state.checkpointer_kind = checkpointer or settings.checkpointer
            app.state.mcp_health = (0.0, False)
            yield

    app = FastAPI(title="Vital Sentinel — 심박 이상 징후 에이전트", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["*"], allow_headers=["*"],
    )

    def tools_of(request: Request) -> ToolBox:
        """요청이 사용할 ToolBox. MCP 미연결이면 503."""
        if request.app.state.tools is None:
            raise HTTPException(503, "MCP 서버에 연결되어 있지 않습니다")
        return request.app.state.tools

    def unwrap(result):
        """MCP 도구의 {"error": ...} 를 HTTP 오류로 변환한다: 알 수 없는 사용자는 404, 그 외 입력 검증 실패(기간·형식)는 422.

        구분은 서버가 알 수 없는 사용자에게 쓰는 문구("알 수 없는 user_id")에 기대는 느슨한 결합이다.
        """
        if isinstance(result, dict) and result.get("error"):
            raise HTTPException(404 if "알 수 없는 user_id" in result["error"] else 422, result["error"])
        return result

    def log_usage(thread_id: str, usage: dict) -> None:
        """요청마다 LLM 사용량을 한 줄로 남겨 비용(특히 원격 API)을 눈으로 볼 수 있게 한다."""
        log.info("usage thread=%s backend=%s llm_calls=%d tokens=%d(in %d/out %d) cost_usd=%.4f llm_s=%.1f", thread_id, backend_name(),
                 usage["llm_calls"], usage["total_tokens"], usage["input_tokens"], usage["output_tokens"], usage["cost_usd"], usage["llm_seconds"])

    @asynccontextmanager
    async def thread_lock(thread_id: str):
        """같은 thread_id 의 요청을 직렬화한다 (LangGraph 는 thread 를 잠그지 않아 동시 요청이 체크포인트를 섞는다).

        락 사전은 사용자가 없어지면 항목을 지워 무한히 커지지 않는다. 단일 프로세스 전제 (다중 워커면 외부 락이 필요).
        """
        entry = app.state.locks.setdefault(thread_id, [asyncio.Lock(), 0])
        entry[1] += 1
        try:
            async with entry[0]:
                yield
        finally:
            entry[1] -= 1
            if entry[1] == 0:
                app.state.locks.pop(thread_id, None)

    async def pending_approval(graph, cfg: dict) -> dict | None:
        """이 thread 가 사람 승인(interrupt)에서 멈춰 있으면 그 요청 내용 ``{reason, message_preview, ...}``, 아니면 None."""
        snap = await graph.aget_state(cfg)
        # approval_id: 이 승인 대기(체크포인트의 task)마다 다른 값 - 한 대화에서 여러 번 대기가 생겨도 오래된 카드와 새 요청을 구분한다.
        return next(({**t.interrupts[0].value, "approval_id": t.id} for t in snap.tasks if t.interrupts), None)

    async def deny_stale_approval(graph, cfg: dict) -> None:
        """승인 대기 중인 thread 에 새 메시지가 오면 기존 요청을 *거절*로 정리한다(무응답 = 안전한 쪽: 알림을 보내지 않음). 안내문은 영향받지 않는다."""
        if await pending_approval(graph, cfg):
            # 같은 run_id 로 두 번 실행하면 트레이스가 충돌하므로 거절 재개에는 run_id 를 빼고 돌린다
            await graph.ainvoke(Command(resume={"approved": False}), {k: v for k, v in cfg.items() if k != "run_id"})

    async def mcp_alive(request: Request) -> bool:
        """MCP 서버가 실제로 응답하는가: list_users 를 짧은 타임아웃으로 호출한다 (결과 5초 캐시)."""
        at, ok = request.app.state.mcp_health
        if time.monotonic() - at < 5.0:
            return ok
        try:
            await asyncio.wait_for(request.app.state.tools.list_users(), MCP_PING_TIMEOUT_S)
            ok = True
        except Exception:
            ok = False
        request.app.state.mcp_health = (time.monotonic(), ok)
        return ok

    async def datasets_of(request: Request) -> list[str]:
        """연결된 데이터셋 이름 목록 (list_users 의 dataset 필드, 없으면 fitbit). MCP 가 안 되면 빈 목록 - /version 은 장애 중에도 응답해야 한다."""
        if request.app.state.tools is None:
            return []
        try:
            us = await asyncio.wait_for(request.app.state.tools.list_users(), MCP_PING_TIMEOUT_S)
        except Exception:
            return []
        return sorted({u.get("dataset", "fitbit") for u in us})

    def graph_input(req: ChatRequest) -> tuple[dict, dict, str]:
        """요청을 (그래프 입력, 실행 config, thread_id) 로 변환한다."""
        thread_id = req.thread_id or str(uuid.uuid4())
        inp: dict = {"messages": [("user", req.message)]}
        if req.user_id:
            inp["user_id"] = req.user_id
        cfg = {  # LangSmith: LANGCHAIN_TRACING_V2 설정 시 자동 추적. 실행 이름/태그/메타데이터(가명 user_id)를 붙인다.
            "configurable": {"thread_id": thread_id},
            "recursion_limit": settings.recursion_limit,  # 무한 루프 가드
            "run_name": "vital-sentinel-chat",
            "tags": ["vital-sentinel"],
            "metadata": {"thread_id": thread_id, "user_id": req.user_id},
            "run_id": uuid.uuid4(),  # 루트 런 ID: 응답에 실어 보내면 사용자 피드백(POST /feedback)을 이 실행의 트레이스에 붙일 수 있다
        }
        return inp, cfg, thread_id

    @app.get("/health")
    async def health(request: Request):
        """서비스 상태. mcp: MCP 연결 여부, llm: 실제 사용 가능 여부, ollama: 도달성·모델 세부.

        llm=false 여도 서비스는 결정론 폴백으로 동작하므로 status 는 "ok" 이다 (가용성과 기능 수준을 분리해 표현).
        """
        llm = request.app.state.llm
        ollama = await llm.health() if hasattr(llm, "health") else {"enabled": bool(llm.enabled), "reachable": bool(llm.enabled), "models": [], "missing": []}
        # llm: 실제로 쓸 수 있는가 (활성화 + Ollama 도달 + 필요한 모델 설치). false 여도 그래프는 결정론 폴백으로 동작한다.
        deps_ = request.app.state.deps
        return {
            "status": "ok", "supervisor_version": deps_.version if deps_ else settings.supervisor_version, "llm_backend": backend_name(),
            "mcp": request.app.state.tools is not None and await mcp_alive(request),
            "llm": bool(ollama["enabled"] and ollama["reachable"] and not ollama["missing"]), "ollama": ollama,
        }

    def backend_name() -> str:
        """실제로 쓰는 LLM 백엔드 이름 (LLM_ENABLED=false 면 "none": 결정론 폴백만)."""
        return settings.llm_backend if settings.llm_enabled else "none"

    @app.get("/version")
    async def version_info(request: Request):
        """실행 구성 표시용: 상위 UI 헤더가 백엔드·모델을 보여줄 수 있게 한다 (비밀값 없음).

        Returns:
            {supervisor_version, llm_backend, fast_model, quality_model, checkpointer, git_commit, datasets}.
            모델 이름은 백엔드에 맞는 값(ollama→로컬 모델, openai→OpenAI 모델, fake/none→"-"). datasets: 지금 연결된 데이터셋 이름 목록
            (MCP 에서 알아낸다 - 연결 실패면 빈 목록; 외부 연구 데이터셋 폴더가 없으면 ["fitbit"] 만).
        """
        backend = backend_name()
        models = {"ollama": (settings.fast_model, settings.quality_model), "openai": (settings.openai_fast_model, settings.openai_quality_model)}.get(backend, ("-", "-"))
        deps_ = request.app.state.deps
        return {
            "supervisor_version": deps_.version if deps_ else settings.supervisor_version, "llm_backend": backend, "fast_model": models[0],
            "quality_model": models[1], "checkpointer": request.app.state.checkpointer_kind, "git_commit": GIT_COMMIT,
            "datasets": await datasets_of(request), "allow_v1_baseline": os.getenv("ALLOW_V1_BASELINE") == "yes",
        }

    @app.get("/users")
    async def users(request: Request, dataset: str | None = None):
        """사용자 가명 ID 목록 (MCP list_users). dataset=fitbit|stanford 로 거른다 - 생략하면 전체, 모르는 이름이면 빈 목록(MCP 계약)."""
        return await (tools_of(request).list_users(dataset) if dataset else tools_of(request).list_users())

    @app.get("/users/{user_id}/events")
    async def events(request: Request, user_id: str):
        """연구용 건강 사건 라벨 {user_id, dataset, events, note} (MCP get_health_events). FitBit 사용자는 빈 목록 200, 모르는 사용자는 404.

        UI 표시 전용이다: 에이전트 그래프·LLM 사실에는 넣지 않는다 (라벨은 진단이 아니고 질병 단정의 근거가 될 수 없다).
        """
        tools = tools_of(request)
        if tools.events is None:  # 사건 도구가 없는 서버: 사용자만 확인하고 빈 목록
            if not any(u["user_id"] == user_id for u in await tools.list_users()):
                raise HTTPException(404, f"알 수 없는 user_id: {user_id}")
            return {"user_id": user_id, "dataset": "fitbit", "events": [], "note": ""}
        return unwrap(await tools.events(user_id))

    @app.get("/users/{user_id}/series")
    async def series(request: Request, user_id: str, lookback_days: float = 1, end: str | None = None, resample: str | None = None):
        """차트용 다운샘플 심박 시계열 (MCP 가 5분 미만 집계를 거부하고 최대 기간을 제한한다 — 데이터 최소화)."""
        # 기간이 길수록 거칠게 집계해 차트 포인트 수를 억제 (명시하지 않은 경우)
        resample = resample or ("5min" if lookback_days <= 7 else "15min" if lookback_days <= 14 else "1h")
        return unwrap(await tools_of(request).series(user_id, lookback_days, end, resample))

    @app.get("/users/{user_id}/summary")
    async def summary(request: Request, user_id: str, lookback_days: float = 7, end: str | None = None):
        """기간 요약 통계와 일별 휴식기 심박 (MCP get_vitals_summary 결과 그대로)."""
        return unwrap(await tools_of(request).summary(user_id, lookback_days, end))

    @app.get("/users/{user_id}/anomalies")
    async def anomalies(request: Request, user_id: str, lookback_days: float = 7, end: str | None = None):
        """결정론적 탐지 결과 (LLM 비관여, MCP detect_anomalies 결과 그대로)."""
        return unwrap(await tools_of(request).detect(user_id, lookback_days, end))

    @app.post("/chat")
    async def chat(request: Request, req: ChatRequest):
        """비스트리밍 채팅. 그래프를 끝까지 실행한 뒤 {thread_id, route, report, anomalies, escalation, validation} 을 돌려준다."""
        tools_of(request)
        inp, cfg, thread_id = graph_input(req)
        try:
            async with thread_lock(thread_id):
                await deny_stale_approval(request.app.state.graph, cfg)
                with usage_scope() as meter:
                    st = await request.app.state.graph.ainvoke(inp, cfg)
                    usage = meter.summary()
                    log_usage(thread_id, usage)
                pending = await pending_approval(request.app.state.graph, cfg)
        except GraphRecursionError:
            raise HTTPException(500, "에이전트가 제한 횟수 안에 끝나지 않았습니다")
        out = {
            "thread_id": thread_id, "run_id": str(cfg["run_id"]), "route": st["route"], "report": st["report"], "anomalies": st.get("anomalies", []),
            "escalation": bool(st.get("escalation")), "validation": st["validation"], "usage": usage,
        }
        if pending:  # HITL: 보호자 알림 승인 대기 (POST /chat/approve 로 결정). 대기가 없으면 키 자체가 없다(기존 응답 형태 유지).
            out["approval_required"] = pending
        return out

    @app.post("/feedback")
    async def feedback(req: FeedbackRequest):
        """사용자 피드백을 LangSmith 에 `user_thumbs` 키로 남긴다 (키가 없거나 실패하면 기록하지 않고 recorded=false — 서비스에 영향 없음).

        Returns:
            {"recorded": bool, "backend": "langsmith"|"none"}.
        """
        client = _langsmith_client()
        if client is None:
            return {"recorded": False, "backend": "none"}
        try:
            await asyncio.to_thread(client.create_feedback, run_id=req.run_id, key="user_thumbs", score=req.score, comment=req.comment)
            return {"recorded": True, "backend": "langsmith"}
        except Exception:
            log.exception("피드백 기록 실패 run_id=%s", req.run_id)
            return {"recorded": False, "backend": "langsmith"}

    @app.post("/chat/approve")
    async def chat_approve(request: Request, req: ApproveRequest):
        """보호자 알림(시뮬레이션) 승인/거절로 멈춘 그래프를 재개한다 (`Command(resume=…)`).

        Returns:
            {thread_id, approved, notification("sent_simulated"|"not_sent"), report(결과 한 줄이 덧붙은 최종 리포트), approval}.
        Raises:
            HTTPException 409: 승인 대기 중인 요청이 없을 때(이미 처리됨·만료·잘못된 thread). 503: MCP 미연결.
        """
        tools_of(request)
        graph = request.app.state.graph
        cfg = {"configurable": {"thread_id": req.thread_id}, "recursion_limit": settings.recursion_limit}
        async with thread_lock(req.thread_id):
            pending = await pending_approval(graph, cfg)
            if not pending:
                raise HTTPException(409, "승인 대기 중인 요청이 없습니다")
            if req.approval_id != pending["approval_id"]:
                raise HTTPException(409, "승인 요청이 바뀌었습니다 - 최신 알림 카드에서 다시 결정하세요")
            st = await graph.ainvoke(Command(resume={"approved": req.approved}), cfg)
        return {"thread_id": req.thread_id, "approved": req.approved, "notification": "sent_simulated" if req.approved else "not_sent",
                "report": st["report"], "approval": st.get("approval")}

    @app.post("/chat/stream")
    async def chat_stream(request: Request, req: ChatRequest):
        """스트리밍 채팅 (text/event-stream). 노드가 방출한 계약 이벤트를 순서대로 전달하고 마지막에 done(또는 error)을 보낸다.

        클라이언트가 연결을 끊으면 진행 중이던 그래프 실행(도구·LLM 호출 포함)을 취소한다.
        """
        tools_of(request)
        graph = request.app.state.graph
        inp, cfg, thread_id = graph_input(req)

        async def pump(q: asyncio.Queue) -> None:
            """그래프를 소비해 SSE 프레임을 큐에 넣는다. 별도 태스크라서 클라이언트 이탈 시 cancel 로 즉시 중단할 수 있다."""
            stream = graph.astream(inp, cfg, stream_mode="custom")  # 노드가 get_stream_writer 로 방출한 계약 이벤트
            try:
                async with thread_lock(thread_id):
                    await deny_stale_approval(graph, cfg)
                    with usage_scope() as meter:
                        async for chunk in stream:
                            q.put_nowait(ev.format_sse(chunk))
                        st = (await graph.aget_state(cfg)).values
                        usage = meter.summary()
                        log_usage(thread_id, usage)
                    if pending := await pending_approval(graph, cfg):  # HITL: 승인 요청을 done 앞에 알린다(리포트·안내문은 이미 확정)
                        q.put_nowait(ev.format_sse(ev.ApprovalRequired(thread_id=thread_id, reason=pending["reason"], message_preview=pending["message_preview"], approval_id=pending["approval_id"])))
                q.put_nowait(ev.format_sse(ev.Done(thread_id=thread_id, report=st["report"], escalation=bool(st.get("escalation")), usage=usage, run_id=str(cfg["run_id"]))))
            except asyncio.CancelledError:
                log.info("클라이언트 이탈 → 그래프 실행 취소 thread=%s", thread_id)
                raise
            except GraphRecursionError:
                q.put_nowait(ev.format_sse(ev.ErrorEvent(message="에이전트가 제한 횟수 안에 끝나지 않았습니다")))
            except Exception:
                log.exception("chat/stream 실패 thread=%s", thread_id)
                q.put_nowait(ev.format_sse(ev.ErrorEvent(message="처리 중 오류가 발생했습니다")))
            finally:
                q.put_nowait(None)  # 생성기 종료 신호
                await stream.aclose()

        async def gen():
            # 주의: Starlette(ASGI spec 2.4+)의 StreamingResponse 는 이탈해도 생성기를 취소하지 않으므로(다음 write 실패 때만 감지)
            # 도구/LLM 호출을 기다리는 중에는 이탈을 못 본다 → 별도 watcher 가 폴링해 pump 를 취소한다.
            """SSE 프레임을 큐에서 꺼내 내보내는 생성기. 종료 시 watcher 와 그래프 태스크를 정리한다."""
            q: asyncio.Queue = asyncio.Queue()
            run = asyncio.create_task(pump(q))

            async def watch() -> None:
                """클라이언트 연결 종료를 0.5초 간격으로 감시해 끊기면 그래프 태스크를 취소한다."""
                while not await request.is_disconnected():
                    await asyncio.sleep(0.5)
                run.cancel()

            watcher = asyncio.create_task(watch())
            try:
                while (frame := await q.get()) is not None:
                    yield frame
            finally:
                watcher.cancel()
                run.cancel()

        return StreamingResponse(
            gen(), media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    return app


app = create_app()

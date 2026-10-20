"""그래프가 사용하는 도구 모음(ToolBox)과 다중 MCP 서버 연결.

책임: (1) 그래프가 의존하는 도구 인터페이스 정의, (2) 여러 MCP stdio 서버(vitals, guidelines)를 띄우고 **이름 탐색**으로
도구·리소스·프롬프트를 ToolBox 에 바인딩하는 어댑터. 어느 서버가 무엇을 제공하는지는 코드에 적어 두지 않고 연결 직후
`list_tools/list_resources/list_prompts` 로 알아낸다 (서버 구성이 바뀌어도 이름만 같으면 동작; 같은 이름이 둘 이상이면 먼저 연결된 서버가 우선).
그래프/노드는 MCP 를 직접 알지 못하고 ToolBox 의 async callable 만 호출한다 -> 테스트는 fake ToolBox 를 주입한다.

부분 성능 저하: 서버 하나가 기동하지 못해도 나머지로 계속한다. guidelines 가 없으면 `search=None`, 리소스 읽기 실패 -> 호출 측이
번들 요약/키워드 발췌로 폴백한다. vitals 가 없으면 데이터 도구 호출이 `ToolUnavailable` 로 실패해 research 가 task_failed 로 격리한다.
모두 실패해야만 `connect_mcp` 가 예외를 던진다. 연결 후 서버가 죽는 경우는 재연결하지 않는다(각 호출이 예외로 실패 -> 노드가 격리).

협력 모듈: `graph.py`/`nodes.py`(소비), `api/main.py`(lifespan 에서 connect_mcp), `mcp_server/vitals_server.py`·`guidelines_server.py`(서버 구현).
설계 근거: MCP subprocess+stdio, Tool/Resource/Prompt 분리, 부분 저하, 의존성 주입.
"""
import asyncio
import json
import logging
import os
import sys
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from langsmith import traceable

from app.config import ROOT, settings

log = logging.getLogger(__name__)


@dataclass
class ToolBox:
    """에이전트가 쓰는 도구 집합. 모든 callable 은 async 이며 잘못된 입력은 예외 대신 ``{"error": 사유}`` 를 반환한다(MCP 계약).

    Attributes:
        list_users: ([dataset]) → 사용자 목록 [{"user_id","dataset","has_sleep","start","end","days","coverage_pct"}]. dataset('fitbit'|'stanford')을 주면 그 데이터셋만.
        summary: (user_id, lookback_days, end) → 요약 통계. end=None 이면 그 사용자의 마지막 기록 시각 기준.
        detect: (user_id, lookback_days, end) → 결정론적 이상 탐지 결과.
        series: (user_id, lookback_days, end, resample) → 차트용 다운샘플 시계열 (그래프는 쓰지 않고 API 만 사용: 데이터 최소화).
        resource: 가이드라인 리소스 URI → 원문 텍스트.
        search: (질문, k) → ``{"method","note","hits":[{"doc_id","title","heading","text","score","source"}]}`` (MCP `search_guidelines`).
            선택 기능이다 — None 이거나 실패하면 호출 측이 키워드 발췌로 폴백한다.
        list_resources: 연결된 서버들이 제공하는 리소스 URI 목록 (이름 탐색 결과). 선택 - None 이면 호출 측이 알려진 URI 를 쓴다.
        prompt: (프롬프트 이름, 인자 dict) → 렌더링된 프롬프트 텍스트 (MCP Prompt). 선택 - 제공 서버가 없으면 None.
        servers: 서버 이름 → 기동 성공 여부 (health/로그용). 선택.
        events: (user_id) → ``{"user_id","dataset","events":[{"type","date","illness"}],"note"}`` 연구용 건강 사건 라벨 (MCP `get_health_events`).
            선택 - None 이면 호출 측이 빈 목록으로 본다. `connect_mcp` 는 서버에 도구가 없으면 ``{"events": []}`` 를 돌려주는 함수를 넣는다.
    """

    list_users: Callable[..., Awaitable[list[dict]]]      # (dataset=None)
    summary: Callable[[str, float, str | None], Awaitable[dict]]     # (user_id, lookback_days, end)
    detect: Callable[[str, float, str | None], Awaitable[dict]]      # (user_id, lookback_days, end)
    series: Callable[[str, float, str | None, str], Awaitable[dict]]  # (user_id, lookback_days, end, resample)
    resource: Callable[[str], Awaitable[str]]                         # guideline://... 원문
    search: Callable[[str, int], Awaitable[dict]] | None = None       # (query, k) → RAG 검색 결과. 서버가 지원하지 않으면 None → 키워드 경로
    list_resources: Callable[[], Awaitable[list[str]]] | None = None  # 탐색으로 알아낸 리소스 URI 목록
    prompt: Callable[[str, dict[str, str]], Awaitable[str]] | None = None  # MCP Prompt 렌더링
    servers: dict[str, bool] | None = None                            # 서버별 기동 상태
    events: Callable[[str], Awaitable[dict]] | None = None            # 연구용 건강 사건 라벨 (get_health_events)


async def bounded(coro: Awaitable[Any], timeout_s: float | None = None) -> Any:
    """MCP 호출에 상한을 건다. 응답 없는 subprocess 가 요청을 영구히 붙잡지 않게 하며, 초과는 TimeoutError(transient)이다."""
    return await asyncio.wait_for(coro, timeout_s if timeout_s is not None else settings.tool_timeout_s)


def parse_tool_result(raw: Any) -> Any:
    """MCP 어댑터 도구 결과를 파이썬 객체로 정규화한다.

    Args:
        raw: JSON 문자열 | content block 리스트(``[{"type":"text","text":...}]``) | 이미 파싱된 dict/list.

    Returns:
        파싱된 객체. fastmcp 가 list 반환을 ``{"result": [...]}`` 로 감싸므로 그 경우 안쪽 값을 돌려준다.

    Raises:
        json.JSONDecodeError: 문자열이 JSON 이 아닐 때.
    """
    if isinstance(raw, list) and raw and all(isinstance(b, dict) and "type" in b for b in raw):
        raw = "".join(b.get("text", "") for b in raw if b["type"] == "text")
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, dict) and set(raw) == {"result"}:
        raw = raw["result"]
    return raw


class ToolUnavailable(RuntimeError):
    """요청한 MCP 도구·리소스·프롬프트를 제공하는 연결된 서버가 없을 때 (서버 기동 실패 또는 이름이 없음)."""


def default_servers() -> dict[str, dict]:
    """운영 서버 구성: vitals(데이터 도구, 프롬프트)와 guidelines(가이드라인 리소스, 검색 도구). 이름은 로그용 키일 뿐 도구 탐색에 쓰이지 않는다."""
    env = {k: os.environ[k] for k in ("DATA_DIR", "RAW_DIR", "OLLAMA_BASE_URL", "EMBED_MODEL", "EMBED_TIMEOUT_S") if k in os.environ}
    base = {"transport": "stdio", "command": sys.executable, "cwd": str(ROOT), **({"env": env} if env else {})}
    return {
        "vitals": {**base, "args": ["-m", "app.mcp_server.vitals_server"]},
        "guidelines": {**base, "args": ["-m", "app.mcp_server.guidelines_server"]},
    }


def pick_guideline_uris(available: list[str], keys: list[str]) -> dict[str, str]:
    """탐색된 리소스 URI 중에서 키(ranges, red_flags, disclaimer ...)에 해당하는 것을 고른다.

    키의 `_` 를 `-` 로 바꾼 이름이 URI 끝 부분(`guideline://heart-rate-ranges` 의 `heart-rate-ranges`)의 접미와 맞는 URI 를 쓴다.
    노드가 URI 문자열을 상수로 들고 있지 않아도 되게 하기 위한 헬퍼이다.

    Args:
        available: `ToolBox.list_resources()` 결과. keys: 찾을 키 목록.

    Returns:
        {키: URI}. 맞는 URI 가 없는 키는 빠진다.
    """
    out = {}
    for k in keys:
        stem = k.replace("_", "-")
        match = next((u for u in available if u.rsplit("/", 1)[-1].endswith(stem)), None)
        if match:
            out[k] = match
    return out


@traceable(run_type="tool", name="mcp.read_resource")
async def _read_resource(session: Any, server: str, uri: str) -> str:
    """리소스를 읽는다 (LangSmith 트레이스에서 보이도록 span 으로 감쌈; server 는 트레이스 입력용)."""
    res = await bounded(session.read_resource(uri))
    return "\n".join(getattr(c, "text", "") for c in res.contents)


@traceable(run_type="tool", name="mcp.get_prompt")
async def _get_prompt(session: Any, server: str, name: str, arguments: dict[str, str]) -> str:
    """프롬프트를 렌더링한다 (span 으로 감쌈). 메시지 텍스트를 줄바꿈으로 이어 돌려준다."""
    res = await bounded(session.get_prompt(name, arguments))
    return "\n".join(getattr(m.content, "text", "") for m in res.messages)


DISCOVERY_TIMEOUT_S = 10.0      # 서버 하나의 탐색 호출(list_tools / list_resources / list_prompts) 1회 상한


async def connect_mcp(stack: AsyncExitStack, servers: dict[str, dict] | None = None,
                      discovery_timeout_s: float = DISCOVERY_TIMEOUT_S) -> ToolBox:
    """MCP 서버들을 stdio subprocess 로 기동하고 영속 세션 위에서 이름 탐색으로 ToolBox 를 만든다.

    `client.get_tools()` 는 호출마다 세션(=subprocess)을 새로 만들어 느리므로, 영속 세션 위에서 도구를 로드한다.
    서버별로 기동을 시도해 실패한 서버는 건너뛰고(경고 로그) 나머지로 ToolBox 를 만든다.

    Args:
        stack: 세션·subprocess 의 수명을 맡을 컨텍스트 스택. ``stack.aclose()`` 가 모두 정리한다.
        servers: {이름: MultiServerMCPClient stdio 설정}. 생략하면 `default_servers()` (테스트에서 고장 난 서버를 끼워 부분 저하를 확인한다).
        discovery_timeout_s: 서버별 탐색 호출 1회 상한(초). 응답하지 않거나 예외를 던지는 서버는 그 서버의 해당 목록만 건너뛰고 상태를 False 로 표시한다
            (다른 서버와 같은 서버의 다른 목록은 계속 쓴다). 서버 *기동* 단계에는 타임아웃을 걸지 않는다: anyio 컨텍스트는 다른 태스크에서 취소하면
            정리가 깨지므로(`wait_for` 가 새 태스크를 만든다) 기동 실패는 예외로만 다룬다.

    Returns:
        탐색 결과로 바인딩된 ToolBox. 제공 서버가 없는 도구를 호출하면 `ToolUnavailable`, search 는 None.

    Raises:
        RuntimeError: 서버가 하나도 기동하지 못했을 때 (호출 측이 처리: API 는 mcp=false 로 기동하고 데이터 엔드포인트는 503).
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools

    cfg = servers or default_servers()
    client = MultiServerMCPClient(cfg)
    status: dict[str, bool] = {}
    sessions: dict[str, Any] = {}
    for name in cfg:
        try:
            sessions[name] = await stack.enter_async_context(client.session(name))
            status[name] = True
        except Exception as e:      # 한 서버의 기동 실패가 나머지를 막지 않는다 (ExceptionGroup 포함)
            status[name] = False
            log.warning("MCP 서버 '%s' 기동 실패 - 이 서버의 도구 없이 계속합니다: %s", name, e)
    if not sessions:
        raise RuntimeError(f"MCP 서버를 하나도 기동하지 못했습니다: {list(cfg)}")

    tools: dict[str, tuple[str, Any]] = {}          # 도구 이름 -> (서버, LangChain 도구)
    resources: dict[str, str] = {}                  # URI -> 서버
    prompts: dict[str, str] = {}                    # 프롬프트 이름 -> 서버

    async def discover(server: str, what: str, coro: Awaitable[Any]) -> Any:
        """탐색 호출 하나를 타임아웃·예외로부터 격리한다. 실패하면 서버 상태를 False 로 하고 None 을 돌려준다 (다른 목록·서버는 계속)."""
        try:
            return await asyncio.wait_for(coro, discovery_timeout_s)
        except Exception as e:                      # TimeoutError, McpError, 프로토콜 오류 등 - 이 서버의 이 목록만 포기
            status[server] = False
            log.warning("MCP 서버 '%s' 의 %s 탐색 실패(%s) - 이 목록 없이 계속합니다: %s", server, what, type(e).__name__, e)
            return None

    for name, session in sessions.items():          # 연결 순서 = 같은 이름의 우선순위
        for tool in await discover(name, "도구", load_mcp_tools(session)) or []:
            tools.setdefault(tool.name, (name, tool))
        res = await discover(name, "리소스", session.list_resources())
        for r in res.resources if res else []:
            resources.setdefault(str(r.uri), name)
        pr = await discover(name, "프롬프트", session.list_prompts())
        for p in pr.prompts if pr else []:
            prompts.setdefault(p.name, name)
    log.info("MCP 탐색 결과 - 도구 %s, 리소스 %s, 프롬프트 %s",
             {n: s for n, (s, _) in tools.items()}, resources, prompts)

    def missing(kind: str, name: str) -> ToolUnavailable:
        """없는 이름에 대한 오류 (연결 상태를 포함해 원인을 바로 알 수 있게)."""
        return ToolUnavailable(f"{kind} '{name}' 을(를) 제공하는 MCP 서버가 없습니다 (서버 상태: {status})")

    async def call(name: str, **kwargs) -> Any:
        """이름으로 찾은 MCP 도구를 호출해 파싱된 결과를 돌려준다 (None 인자는 제외, 호출당 타임아웃 적용)."""
        if name not in tools:
            raise missing("도구", name)
        return parse_tool_result(await bounded(tools[name][1].ainvoke({k: v for k, v in kwargs.items() if v is not None})))

    async def resource(uri: str) -> str:
        """URI 를 제공하는 서버에서 리소스 원문을 읽는다 (타임아웃 적용, 트레이스 span)."""
        if uri not in resources:
            raise missing("리소스", uri)
        return await _read_resource(sessions[resources[uri]], resources[uri], uri)

    async def list_resources() -> list[str]:
        """탐색된 리소스 URI 목록 (정렬)."""
        return sorted(resources)

    async def prompt(name: str, arguments: dict[str, str]) -> str:
        """이름으로 찾은 MCP 프롬프트를 렌더링한다."""
        if name not in prompts:
            raise missing("프롬프트", name)
        return await _get_prompt(sessions[prompts[name]], prompts[name], name, arguments)

    async def events(user_id: str) -> dict:
        """`get_health_events` 를 이름으로 찾아 호출한다 (같은 타임아웃). 제공 서버가 없으면 빈 목록 모양을 돌려준다 - 라벨은 선택 기능이다."""
        if "get_health_events" not in tools:
            return {"user_id": user_id, "events": []}
        return await call("get_health_events", user_id=user_id)

    return ToolBox(
        list_users=lambda dataset=None: call("list_users", dataset=dataset),
        summary=lambda uid, days, end: call("get_vitals_summary", user_id=uid, lookback_days=days, end=end),
        detect=lambda uid, days, end: call("detect_anomalies", user_id=uid, lookback_days=days, end=end),
        series=lambda uid, days, end, rs: call("get_hr_series", user_id=uid, lookback_days=days, end=end, resample=rs),
        resource=resource,
        search=(lambda q, k: call("search_guidelines", query=q, k=k)) if "search_guidelines" in tools else None,
        list_resources=list_resources,
        prompt=prompt if prompts else None,
        servers=status,
        events=events,
    )

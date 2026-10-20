"""테스트 전용 고장 MCP 서버 (stdio): 모드에 따라 탐색 호출(list_*)이 멈추거나 예외를 던진다.

사용: `python tests/mcp_bad_server.py <hang_tools|raise_tools|raise_resources|raise_prompts|hang_resources>`.
도구 하나(`ping`)는 정상 동작하므로, 어떤 탐색이 실패해도 서버 기동 자체와 나머지 호출은 살아 있다. 협력: `tests/test_mcp_multiserver.py`.
"""
import sys

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

mode = sys.argv[1] if len(sys.argv) > 1 else "ok"
server = Server("bad")


async def maybe_fail(kind: str) -> None:
    """모드가 해당 목록을 겨냥하면 멈추거나(hang) 예외를 던진다(raise)."""
    if mode == f"hang_{kind}":
        await anyio.sleep(3600)
    if mode == f"raise_{kind}":
        raise RuntimeError(f"{kind} 탐색 고장")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """도구 목록 (고장 모드면 실패)."""
    await maybe_fail("tools")
    return [types.Tool(name="ping", description="항상 pong", inputSchema={"type": "object", "properties": {}})]


@server.list_resources()
async def list_resources() -> list[types.Resource]:
    """리소스 목록 (고장 모드면 실패)."""
    await maybe_fail("resources")
    return []


@server.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    """프롬프트 목록 (고장 모드면 실패)."""
    await maybe_fail("prompts")
    return []


async def main() -> None:
    """stdio 로 서버를 돌린다."""
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    anyio.run(main)

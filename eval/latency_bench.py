"""에이전트 지연 측정: 실제 MCP + 실제 Ollama 로 대표 질의 N건을 순차 실행해 구간별 시간·검증 통과 여부를 JSON 으로 출력.

설정 변형은 환경변수(ANALYSIS_TIER, MERGE_ANALYSIS, COMPACT_FACTS, NUM_PREDICT_*, LLM_KEEP_ALIVE)로 주고 변형마다 새 프로세스로 실행한다.
사용: VARIANT=v0 .venv/bin/python -m eval.latency_bench > eval/results/latency_v0.json
"""
import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack

from langgraph.checkpoint.memory import InMemorySaver

from app.agent.graph import build_graph
from app.agent.llm import default_llm
from app.agent.toolbox import connect_mcp

QUERIES = [
    "U05 최근 30일 심박 이상 점검해줘",
    "U03 최근 30일 이상 있었는지 점검해줘",
    "U12 지난 2주 심박 점검해줘",
    "U08 최근 한 달 심박이 평소와 다른지 확인해줘",
    "U06 최근 30일 이상 징후 분석해줘",
]


async def run_one(graph, msg: str, thread: str) -> dict:
    t0, marks, ntok = time.perf_counter(), {}, 0
    cfg = {"configurable": {"thread_id": thread}, "recursion_limit": 40}
    async for e in graph.astream({"messages": [("user", msg)]}, cfg, stream_mode="custom"):
        now = time.perf_counter() - t0
        t = e["type"]
        if t == "token":
            ntok += 1
            marks.setdefault("first_token", now)
        elif t == "task_completed" and e["task_type"] == "analysis":
            marks["analysis_done"] = now
        elif t == "task_completed" and e["task_type"] == "writing":
            marks["writing_done"] = now
        elif t in ("plan_created", "anomalies_found", "validation"):
            marks[t] = now
    st = (await graph.aget_state(cfg)).values
    return {
        "query": msg, "total_s": round(time.perf_counter() - t0, 1), "marks": {k: round(v, 1) for k, v in marks.items()},
        "attempts": st["validation"]["attempt"], "passed": st["validation"]["passed"], "degraded": st["validation"]["degraded"],
        "issues": st["validation"]["issues"], "report_chars": len(st["report"]), "n_anomalies": len(st.get("anomalies") or []),
    }


async def main() -> None:
    async with AsyncExitStack() as stack:
        graph = build_graph(await connect_mcp(stack), InMemorySaver(), default_llm())
        await run_one(graph, "심박수 정상 범위가 어떻게 돼?", "warmup")  # 모델 적재(콜드 로드) 제외
        out = [await run_one(graph, q, f"b{i}") for i, q in enumerate(QUERIES)]
    json.dump({"variant": os.getenv("VARIANT", "?"), "runs": out}, sys.stdout, ensure_ascii=False, indent=1)


asyncio.run(main())

"""컨텍스트 요약 효과 측정: 대화 길이(1/5/10/20턴)에 따라 *다음 질문의 프롬프트 대화 기록 부분*이 얼마나 커지는지 세 전략으로 비교한다.

전략: A 전체 기록(제한 없음, 가장 단순한 방식) / B 슬라이딩 윈도우(요약 도입 전 동작: 최근 6개 메시지, 메시지당 600자 절단) / C 요약 + 최근 6개(현재 동작).
지표: 대화 기록 부분의 토큰 추정치(글자 수/3 — 공급자 토크나이저 값이 아니라 *추정*), 1턴째에 말한 개인 정보(이름)가 프롬프트에 남는지, C 의 요약 비용(LLM 호출 수·토큰).
대화는 합성이다: 1턴째에 이름을 말하고 이후 질문 7종을 순환하며(이름은 다시 말하지 않는다) 도우미 답변은 실제 v2 리포트 길이(약 500자)의 고정 문장을 쓴다 → 절대값보다 *증가 추세*를 본다.
안전 정보(에스컬레이션·이상 목록)는 요약과 무관하게 상태 키에서 도출되므로 이 표에 포함하지 않는다(`tests/test_context_summary.py` 가 검증).

실행: `.venv/bin/python -m eval.context_summary_eval [--backend fake|ollama]`  → eval/results/context_summary.md
"""
import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).parent / "results" / "context_summary.md"
QUESTIONS = ["제 이름은 민수입니다", "서맥이 뭐예요?", "안정시 심박 정상 범위가 어떻게 돼?", "수면 중 심박은 왜 낮아져?", "카페인이 심박에 영향을 주나요?",
             "운동 중 심박이 높은 건 괜찮아?", "센서 결측은 왜 생겨요?", "언제 병원에 가야 해?"]
REPLY = ("일반적으로 성인의 안정시 심박수는 분당 60~100회이며, 잘 훈련된 사람은 더 낮을 수 있습니다. 수면 중에는 평소보다 낮아지는 것이 자연스럽고, 카페인이나 스트레스는 "
         "일시적으로 심박을 올릴 수 있습니다. 증상(흉통, 호흡곤란, 실신)이 함께 있으면 심박 수치와 무관하게 즉시 진료가 필요합니다. 기기 오차가 있으므로 한 번의 수치로 판단하지 말고 "
         "추세를 보세요. 걱정되면 의료진과 상담하세요. 센서 결측은 착용 상태나 피부 접촉 때문에 생길 수 있습니다.")


def conversation(turns: int) -> list:
    from langchain_core.messages import AIMessage, HumanMessage

    msgs = []
    for i in range(turns):
        msgs += [HumanMessage(QUESTIONS[0] if i == 0 else QUESTIONS[1 + (i - 1) % (len(QUESTIONS) - 1)]), AIMessage(REPLY)]
    return msgs


async def measure(n: int, backend: str) -> dict:
    from app.agent.deps import AgentDeps
    from app.agent.llm import make_llm
    from app.agent.nodes import AgentNodes, _history
    from app.agent.toolbox import ToolBox
    from app.agent.usage import estimate_tokens, usage_scope
    from app.config import settings

    msgs = conversation(n)
    full = "\n".join(f"{'사용자' if m.type == 'human' else '도우미'}: {m.content}" for m in msgs)
    window = _history({"messages": msgs}, 6)
    llm = make_llm(backend)
    noop = ToolBox(None, None, None, None, None)
    nodes = AgentNodes(AgentDeps(llm, llm, noop, None, settings, "v2"))
    state = {"messages": msgs}
    with usage_scope() as meter:
        state.update(await nodes._maybe_summarize(state))
    summarized = _history(state, 6)
    t = estimate_tokens
    u = meter.summary()
    return {"turns": n, "messages": len(msgs), "A_tokens": t(full), "B_tokens": t(window), "C_tokens": t(summarized),
            "A_keeps_name": "민수" in full, "B_keeps_name": "민수" in window, "C_keeps_name": "민수" in summarized,
            "C_summarized": bool(state.get("memory_summary")), "C_summary_chars": len(state.get("memory_summary", "")),
            "C_summary_cost_calls": u["llm_calls"], "C_summary_cost_tokens": u["total_tokens"]}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="fake")
    args = ap.parse_args()
    rows = [await measure(n, args.backend) for n in (1, 5, 10, 20)]
    ok = lambda b: "O" if b else "X"  # noqa: E731
    md = ["# 컨텍스트 요약 효과 (대화 기록 부분의 프롬프트 토큰)", "",
          f"- 측정: {datetime.now().isoformat(timespec='seconds')}, 요약기 백엔드 `{args.backend}`, 토큰은 글자 수/3 추정치(절대값보다 증가 추세를 볼 것)",
          "- 전략 — A: 전체 기록(제한 없음) / B: 슬라이딩 윈도우(요약 도입 전, 최근 6개·메시지당 600자) / C: 요약 + 최근 6개(현재)",
          "- 대화는 합성(질문 8종 순환, 답변 약 500자 고정)이며 1턴째에 이름을 말한다. 임계: 요약되지 않은 메시지 10개 초과 시 요약, 최근 6개 유지.", "",
          "| 턴 | 메시지 | A 전체 | B 윈도우 | C 요약+윈도우 | 이름 유지 A/B/C | C 요약 글자 | C 요약 비용(호출/토큰) |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['turns']} | {r['messages']} | {r['A_tokens']} | {r['B_tokens']} | {r['C_tokens']} | {ok(r['A_keeps_name'])}/{ok(r['B_keeps_name'])}/{ok(r['C_keeps_name'])} | "
                  f"{r['C_summary_chars']} | {r['C_summary_cost_calls']}/{r['C_summary_cost_tokens']} |")
    md += ["", "## 해석", "",
           "- A 는 턴 수에 비례해 커진다. B 는 상한이 있지만 오래된 개인 정보(이름)가 윈도우 밖으로 밀려나 사라진다. C 는 상한을 유지하면서 요약으로 그 정보를 남긴다.",
           "- 주의: 요약은 미요약 메시지가 임계(10개)를 넘을 때만 만들어지므로, 그 이전(예: 5턴=10개)에는 윈도우(6개) 밖으로 밀려난 정보가 사라질 수 있다 — 개인 정보 질문은 정규식이 *전체* 메시지를 보므로 영향이 없지만, LLM 프롬프트만 보면 그 구간에 공백이 있다(임계를 낮추면 해소되나 요약 호출이 늘어난다).",
           "- 비용: C 는 요약을 만들 때 fast LLM 호출이 추가된다(요약은 `summarize_after_messages` 를 넘는 시점에만, 증분 요약). 요약 실패·진단 표현 포함 시 규칙 요약으로 대체.",
           "- 한계: 합성 대화 + 토큰 추정치다. 실제 LLM 요약 품질(핵심 정보 보존율)은 `--backend ollama` 로 돌려 요약 문장을 사람이 확인해야 한다. 이름 같은 정규식 폴백이 있는 정보만 결정론적으로 보존된다.", ""]
    OUT.write_text("\n".join(md))
    print(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    asyncio.run(main())

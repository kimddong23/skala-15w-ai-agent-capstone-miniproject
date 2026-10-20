"""허용 목록 필터의 비용 측정(오프라인): 이상이 있는 리포트에서 LLM 본문이 사실 서술만 남을 때 글자가 얼마나 살아남는가.

입력: 저장된 Before/After 결과(`eval/results/before_after*.json`)의 v2 리포트 원문. LLM·MCP 를 부르지 않는다.
대상: route 가 analysis 이고 템플릿 대체(degraded)가 아니며 '이상 없음' 계열 문장이 없는 리포트 - 이상이 있는 리포트에서만 필터가 켜지기 때문이다.
        (저장된 결과에 이상 목록이 없어 문장으로 판별하는 근사다. 이상 없는 리포트가 섞이면 생존율이 높게 나온다.)
측정: 고정 문구(면책·안내문)를 뺀 본문에 `narration_only` 를 적용해 글자 생존율, 문장 수, 완전히 비어 템플릿이 필요해지는 리포트 수를 센다.
결과: eval/results/allowlist_cost.md.

실행:  .venv/bin/python -m eval.allowlist_cost
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = Path(__file__).parent / "results"
SOURCES = ("before_after.json", "before_after_fast3b_analysis8b.json")
MIN_PART = "이 서비스는 의료기기가 아니며"   # 서비스 면책 리소스 문단의 시작 - 코드가 붙인 고정 문구라 본문이 아니다


def sentences(text: str) -> list[str]:
    """제목 줄을 뺀 문장 목록 (줄바꿈과 문장부호 뒤 공백 기준)."""
    return [u for line in text.split("\n") if not line.lstrip().startswith("#") for u in re.split(r"(?<=[.!?。])\s+", line) if u.strip()]


def measure() -> list[dict]:
    """저장된 v2 리포트마다 허용 목록 필터 전후의 글자·문장 수를 잰다."""
    from app.agent.report import strip_fixed
    from app.agent.validator import is_fact_narration, narration_only

    rows = []
    for name in SOURCES:
        data = json.loads((RESULTS / name).read_text())
        for case_id, r in data["results"]["v2"].items():
            if r["route"] != "analysis" or r.get("degraded"):
                continue
            body = strip_fixed(r["report"]).split(MIN_PART)[0].strip()
            if "이상이 없" in body or "이상 없음" in body:
                continue
            out = narration_only(body)
            units = sentences(body)
            rows.append({
                "source": name.removesuffix(".json"), "case": case_id, "chars": len(body), "kept_chars": len(out), "survival": len(out) / max(1, len(body)),
                "sentences": len(units), "kept_sentences": sum(is_fact_narration(u) for u in units), "empty": not re.search(r"[가-힣]", out),
            })
    return rows


def render(rows: list[dict]) -> str:
    """측정 결과를 마크다운 표와 요약으로 만든다."""
    s = [r["survival"] for r in rows]
    mean = sum(s) / len(s)
    lines = [
        "# 허용 목록 필터의 비용 (오프라인 측정)", "",
        "이상이 있는 리포트에서는 LLM 본문이 *사실 서술*(시각·지속·수치·유형·맥락·데이터 한계)만 남고 조언·평가·안심·진료 언급은 코드가 쓰는 고정 문구로 대신한다. "
        "이 표는 그 대가로 LLM 본문 글자가 얼마나 살아남는지를 저장된 v2 리포트 원문에 `narration_only` 를 적용해 잰 것이다 (LLM 호출 없음).", "",
        f"- 대상: {len(rows)}건 (`{', '.join(SOURCES)}` 의 v2 analysis 리포트 중 템플릿 대체가 아니고 '이상 없음' 문장이 없는 것). 이상 목록이 저장돼 있지 않아 문장으로 판별한 근사다.",
        "- 주의: 실제 서비스는 닫힌 어휘 검사에 이번 분석의 FACTS 단어(동적 어휘)를 함께 쓰지만, 저장된 결과에는 FACTS 가 없어 이 측정은 고정 어휘만 쓴다 - 생존율의 *하한*에 가깝다.",
        f"- 글자 생존율: 평균 {mean:.2f}, 최소 {min(s):.2f}, 최대 {max(s):.2f}",
        f"- 사실 서술이 하나도 남지 않아 템플릿 본문이 필요해지는 리포트: {sum(r['empty'] for r in rows)}건 / {len(rows)}건",
        f"- 문장 생존율: {sum(r['kept_sentences'] for r in rows)} / {sum(r['sentences'] for r in rows)} "
        f"({sum(r['kept_sentences'] for r in rows) / max(1, sum(r['sentences'] for r in rows)):.2f})", "",
        "읽는 법: 사라지는 것은 주로 [해석]·[권고]의 조언·평가·원인 추정 문장이다(예: '~놓친 것으로 보여요', '일시적일 수 있으나 주의가 필요합니다'). "
        "안전을 얻는 대신 이상이 있는 리포트의 해석 깊이가 줄어든다는 트레이드오프이다.", "",
        "| 출처 | 케이스 | 본문 글자 | 남은 글자 | 생존율 | 문장 | 남은 문장 | 비어 있음 |", "|---|---|---|---|---|---|---|---|",
    ]
    lines += [f"| {r['source']} | {r['case']} | {r['chars']} | {r['kept_chars']} | {r['survival']:.2f} | {r['sentences']} | {r['kept_sentences']} | {'예' if r['empty'] else '아니오'} |"
              for r in rows]
    return "\n".join(lines) + "\n"


def main() -> None:
    """측정하고 eval/results/allowlist_cost.md 를 쓴다."""
    rows = measure()
    (RESULTS / "allowlist_cost.md").write_text(render(rows))
    s = [r["survival"] for r in rows]
    print(f"reports={len(rows)} mean={sum(s) / len(s):.2f} min={min(s):.2f} max={max(s):.2f} empty={sum(r['empty'] for r in rows)}")


if __name__ == "__main__":
    main()

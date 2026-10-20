"""eval/results/latency_v*.json 을 읽어 변형별 비교표(마크다운)를 출력."""
import glob
import json
import statistics as st

rows = []
for f in sorted(glob.glob("eval/results/latency_v*.json")):
    d = json.load(open(f))
    r = d["runs"]
    m = lambda k: round(st.mean(x["marks"].get(k, 0) for x in r), 1)  # noqa: E731
    rows.append((d["variant"], round(st.mean(x["total_s"] for x in r), 1), round(max(x["total_s"] for x in r), 1), m("analysis_done"), m("first_token"),
                 m("writing_done"), sum(x["passed"] and not x["degraded"] for x in r), sum(x["attempts"] == 1 and x["passed"] for x in r),
                 round(st.mean(x["report_chars"] for x in r)), len(r)))
print("| 변형 | 평균 총(s) | 최대(s) | 분석 완료(s) | 첫 토큰(s) | 작성 완료(s) | 검증 통과 | 1회 통과 | 평균 리포트 글자 |")
print("|---|---|---|---|---|---|---|---|---|")
for v, tot, mx, ana, ft, wd, ok, first, ch, n in rows:
    print(f"| {v} | {tot} | {mx} | {ana} | {ft} | {wd} | {ok}/{n} | {first}/{n} | {ch} |")

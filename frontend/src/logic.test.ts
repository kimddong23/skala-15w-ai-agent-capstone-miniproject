/**
 * 순수 로직 테스트 — 리포트 각주 매칭(오인용 방지), 시계열 유틸, 실행 트레이스(소요·상태·재계획·경로별 노드),
 * 서버 계약 보강(token.reset, validation.degraded, 고정 문구 분리, /summary 스키마).
 * 이벤트 도착 시각은 인위적으로 주입해 소요 시간 계산을 결정론적으로 검증한다.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { matchCitation, splitCitations } from "./cite";
import type { Anomaly, ChatEvent } from "./events";
import { bucketize, computeStats, lowerBound, numberInRange, percentile, resampleMs, smallestRange, toSeries } from "./series";
import { buildTrace } from "./trace";
import { applyEvent, latestApprovals, newTurn } from "./turn";
import { splitReport } from "./report";
import { describeLlm, formatUsage, formatVersion } from "./llm";
import { participantNo } from "./labels";
import { fmtDateTimeY, tsToMs } from "./time";
import { buildMarkers, eventLabel, leadsFor, shortEventLabel, warningText } from "./illness";
import { mockAnomalies, mockApprove, pendingMockApproval, mockEvents, mockSeries, mockUsers } from "./mock";
import { approveChat } from "./api";
import { parseEvent } from "./sse";
import { SummaryResponse } from "./events";

const A = (id: string, type: string, value: number, dur: number, start = "2016-05-10T07:30:00", severity: Anomaly["severity"] = "warning"): Anomaly => ({
  anomaly_id: id, type, start, end: start, duration_min: dur, severity, value, baseline: 62, context: "rest", evidence: "",
});

describe("각주 매칭", () => {
  const list = [A("a", "resting_tachycardia", 128, 25), A("b", "bradycardia", 38, 18), A("c", "personal_deviation", 98, 30)];
  it("유형 언급 + 수치 일치 시에만 인용한다", () => {
    expect(matchCitation("안정 시 빈맥: 평균 128 bpm이 25분간", 128, "bpm", list)?.anomaly_id).toBe("a");
    expect(matchCitation("안정 시 빈맥: 평균 128 bpm이 25분간", 25, "분", list)?.anomaly_id).toBe("a");
    expect(matchCitation("수면 중 서맥 38 bpm", 38, "bpm", list)?.anomaly_id).toBe("b");
  });
  it("유형 언급이 없거나 수치가 다르면 인용하지 않는다(오인용 방지)", () => {
    expect(matchCitation("평균 128 bpm이었습니다", 128, "bpm", list)).toBeNull();
    expect(matchCitation("안정 시 빈맥 130 bpm", 130, "bpm", list)).toBeNull();
    expect(matchCitation("기준선 64 bpm 대비", 64, "bpm", list)).toBeNull();
  });
  it("후보가 둘 이상이면 인용하지 않는다", () => {
    const dup = [A("x", "bradycardia", 38, 18), A("y", "bradycardia", 38, 40)];
    expect(matchCitation("서맥 38 bpm", 38, "bpm", dup)).toBeNull();
  });
  it("splitCitations: 수치 직후에 각주가 붙고 나머지 텍스트는 보존된다", () => {
    const text = "빈맥 128 bpm이 25분간 지속";
    const p = splitCitations(text, text, list);
    expect(p.map((x) => x.text).join("")).toBe(text);
    expect(p.filter((x) => "anomaly" in x)).toHaveLength(1); // 같은 줄·같은 이벤트는 한 번만
    expect(p[0].text).toBe("빈맥 128 bpm");
  });
  it("sensor_gap 은 bpm 인용 대상이 아니다", () => {
    expect(matchCitation("결측 0 bpm", 0, "bpm", [A("g", "sensor_gap", 0, 130)])).toBeNull();
    expect(matchCitation("센서 결측 130분", 130, "분", [A("g", "sensor_gap", 0, 130)])?.anomaly_id).toBe("g");
  });
});

describe("각주 매칭: 화면에 보이는 한글 종류·심각도 단서", () => {
  const list = [A("t", "resting_tachycardia", 128, 25, "2016-05-10T07:30:00", "critical"), A("p", "personal_deviation", 98, 30), A("g", "sensor_gap", 0, 130, "2016-05-12T04:30:00", "info")];
  it("정식 한글 종류 이름만 있어도(영문 코드·별칭 없이) 수치가 맞으면 인용한다", () => {
    expect(matchCitation("안정 시 빈맥(심각): 심박이 128 bpm으로 25분 동안 계속 높았어요", 128, "bpm", list)?.anomaly_id).toBe("t");
    expect(matchCitation("개인 기준선 이탈 — 98 bpm", 98, "bpm", list)?.anomaly_id).toBe("p");
  });
  it("예전 이름 \"휴식 중 빈맥\" 도 별칭으로 계속 인용된다", () => {
    expect(matchCitation("휴식 중 빈맥: 128 bpm", 128, "bpm", list)?.anomaly_id).toBe("t");
  });
  it("종류 단서가 없으면 심각도 표기로 후보가 하나일 때만 인용한다", () => {
    expect(matchCitation("심각: 128 bpm이 25분간 이어졌어요", 128, "bpm", list)?.anomaly_id).toBe("t");
    // 심각도 단서도 후보를 가르지 못하면(수치가 같은 둘 다 주의) 인용하지 않는다
    const twin = [A("a", "bradycardia", 40, 10), A("b", "personal_deviation", 40, 12)];
    expect(matchCitation("주의: 40 bpm", 40, "bpm", twin)).toBeNull();
    // 종류 단서가 있으면 심각도보다 우선한다
    expect(matchCitation("서맥(주의): 40 bpm", 40, "bpm", twin)?.anomaly_id).toBe("a");
  });
  it("종류 단서에서 둘 이상이 걸리면 더 약한 단서로 억지로 고르지 않는다", () => {
    const two = [A("a", "bradycardia", 40, 10, undefined, "critical"), A("b", "bradycardia", 40, 12, undefined, "warning")];
    expect(matchCitation("서맥 주의 40 bpm", 40, "bpm", two)).toBeNull();
  });
  it("'N시간 M분' 은 분으로 환산해 지속시간과 맞춘다", () => {
    const text = "센서 결측(참고): 심박이 2시간 10분 동안 기록되지 않았어요";
    const p = splitCitations(text, text, list);
    expect(p.map((x) => x.text).join("")).toBe(text);
    const cited = p.find((x) => "anomaly" in x) as { text: string; anomaly: Anomaly } | undefined;
    expect(cited?.anomaly.anomaly_id).toBe("g");
    expect(cited?.text.endsWith("2시간 10분")).toBe(true);
  });
});

describe("시계열 유틸", () => {
  const pts = Array.from({ length: 100 }, (_, i) => ({ ts: new Date(Date.UTC(2016, 4, 1, 0, i * 5)).toISOString().slice(0, 19), hr: i % 10 === 0 ? null : 60 + (i % 7), steps: i % 3 === 0 ? 0 : 5, asleep: i < 20 }));
  const s = toSeries(pts);
  it("lowerBound / percentile", () => {
    expect(lowerBound([1, 3, 5], 3)).toBe(1);
    expect(lowerBound([1, 3, 5], 6)).toBe(3);
    expect(percentile([1, 2, 3, 4, 5], 0.5)).toBe(3);
    expect(percentile([], 0.5)).toBeNull();
  });
  it("bucketize: 점이 충분히 적으면 원시 점 그대로, 많으면 nb 이하로 줄인다", () => {
    expect(bucketize(s, s.t[0], s.t[99], 200)).toHaveLength(100);
    const b = bucketize(s, s.t[0], s.t[99], 10);
    expect(b.length).toBeLessThanOrEqual(10);
    expect(b.reduce((n, x) => n + x.n, 0)).toBe(100);
    expect(b[0].min).toBeLessThanOrEqual(b[0].hr!);
  });
  it("bucketize: 간격이 허용치를 넘으면 버킷을 끊고 gapBefore 로 표시한다(서버가 빈 구간 점을 생략)", () => {
    const step = 5 * 60_000;
    const mk = (mins: number[]) => toSeries(mins.map((m) => ({ ts: new Date(Date.UTC(2016, 4, 1, 0, m)).toISOString().slice(0, 19), hr: 60, steps: 0, asleep: false })));
    const g = mk([0, 5, 10, 15, 200, 205, 210]); // 15 → 200 분: 3시간 넘는 공백
    const withGap = bucketize(g, g.t[0], g.t[6], 2, step * 1.5); // nb=2 라 원래는 4점씩 한 버킷
    const flags = withGap.map((b) => b.gapBefore);
    expect(flags.filter(Boolean)).toHaveLength(1);
    const after = withGap.find((b) => b.gapBefore)!;
    expect(after.tStart).toBe(g.t[4]); // 공백 뒤 첫 점에서 새 버킷
    expect(withGap[withGap.indexOf(after) - 1].tEnd).toBe(g.t[3]); // 공백 앞 버킷은 15분에서 끝남
    // 허용치를 주지 않으면(기본) 끊지 않는다
    expect(bucketize(g, g.t[0], g.t[6], 2).some((b) => b.gapBefore)).toBe(false);
  });
  it("resampleMs: 서버 문자열을 해석하고, 없으면 점 간격 중앙값으로 추정한다", () => {
    expect(resampleMs("5min", [])).toBe(300_000);
    expect(resampleMs("15min", [])).toBe(900_000);
    expect(resampleMs("1h", [])).toBe(3_600_000);
    expect(resampleMs(undefined, [0, 300_000, 600_000, 900_000, 20_000_000])).toBe(300_000); // 큰 공백이 섞여도 중앙값
    expect(resampleMs("???", [])).toBe(300_000);
  });
  it("computeStats: stepMs 가 있으면 커버리지는 기대 점 수 기준(빈 구간 생략 보정)", () => {
    const step = 5 * 60_000;
    const s = toSeries(Array.from({ length: 50 }, (_, i) => ({ ts: new Date(Date.UTC(2016, 4, 1, 0, i * 5)).toISOString().slice(0, 19), hr: 60, steps: 0, asleep: false })));
    const t0 = s.t[0];
    const t1 = t0 + 100 * step; // 기대 100점, 실제 50점
    expect(computeStats(s, [], t0, t1, step).coveragePct).toBeCloseTo(50, 0);
    expect(computeStats(s, [], t0, t1).coveragePct).toBe(100); // stepMs 없으면 존재하는 점 기준(부풀려짐)
  });
  it("computeStats: 커버리지와 심각도별 건수", () => {
    const st = computeStats(s, [A("a", "x", 1, 1, pts[50].ts, "critical"), A("b", "x", 1, 1, "2000-01-01T00:00:00")], s.t[0], s.t[99]);
    expect(st.coveragePct).toBe(90);
    expect(st.counts).toEqual({ critical: 1, warning: 0, info: 0 });
    expect(st.restingHr).not.toBeNull();
  });
  it("numberInRange: 기간 안에서 1부터 번호를 매기고, 기간이 바뀌면 다시 매긴다", () => {
    const DAY = 86_400_000;
    const end = Date.UTC(2016, 4, 12, 9, 30);
    const at = (daysBefore: number) => new Date(end - daysBefore * DAY).toISOString().slice(0, 19);
    const list = [A("old2", "x", 1, 1, at(20)), A("old1", "x", 1, 1, at(12)), A("new1", "x", 1, 1, at(3)), A("new2", "x", 1, 1, at(1))];
    const w7 = numberInRange(list, end, 7);
    expect(w7.rows.map((r) => r.anomaly_id)).toEqual(["new1", "new2"]);
    expect(w7.index.get("new1")).toBe(1); // 7일 뷰는 #1 부터(30일 로그의 #3 이 아니다)
    expect(w7.index.get("old1")).toBeUndefined();
    const w30 = numberInRange(list, end, 30);
    expect([...w30.index.entries()]).toEqual([["old2", 1], ["old1", 2], ["new1", 3], ["new2", 4]]);
    // 같은 시작 시각이면 ID 순
    expect(numberInRange([A("b", "x", 1, 1, at(1)), A("a", "x", 1, 1, at(1))], end, 7).rows.map((r) => r.anomaly_id)).toEqual(["a", "b"]);
  });
  it("smallestRange: 이상을 담는 가장 짧은 기간, 없으면 가장 긴 기간", () => {
    const DAY = 86_400_000;
    const end = 100 * DAY;
    expect(smallestRange(end - 0.5 * DAY, end, [1, 7, 30])).toBe(1);
    expect(smallestRange(end - 3 * DAY, end, [1, 7, 30])).toBe(7);
    expect(smallestRange(end - 20 * DAY, end, [1, 7, 30])).toBe(30);
    expect(smallestRange(end - 90 * DAY, end, [1, 7, 30])).toBe(30);
  });
  it("각주 번호: 기간 밖 이상은 전환될 기간의 번호를 쓴다", () => {
    const DAY = 86_400_000;
    const end = Date.UTC(2016, 4, 12);
    const at = (d: number) => new Date(end - d * DAY).toISOString().slice(0, 19);
    const list = [A("far", "x", 1, 1, at(15)), A("near", "x", 1, 1, at(2))];
    const n = Object.fromEntries([1, 7, 30].map((r) => [r, numberInRange(list, end, r)]));
    const labelFor = (id: string, cur: number) => {
      const a = list.find((x) => x.anomaly_id === id)!;
      const r = Date.parse(a.start + "Z") >= end - cur * DAY ? cur : smallestRange(Date.parse(a.start + "Z"), end, [1, 7, 30]);
      return n[r].index.get(id);
    };
    expect(labelFor("near", 7)).toBe(1); // 지금 뷰의 번호
    expect(labelFor("far", 7)).toBe(1); // 30일로 전환되면 far 가 #1, near 가 #2
    expect(n[30].index.get("near")).toBe(2);
  });

});

describe("실행 트레이스", () => {
  const run = (evs: [number, ChatEvent][]) => evs.reduce((t, [at, ev]) => applyEvent(t, ev, at), newTurn("1", "q"));
  const plan: ChatEvent = { type: "plan_created", needs_analysis: true, lookback_days: 7, tasks: [
    { id: "t1", type: "research", description: "요약" }, { id: "t2", type: "research", description: "탐지" },
    { id: "t3", type: "analysis", description: "해석" }, { id: "t4", type: "writing", description: "작성" } ] };

  it("노드별 소요 ms 와 상태를 이벤트 도착 시각에서 계산한다", () => {
    const t = run([
      [300, { type: "route_decided", route: "analysis", reasoning: "r" }],
      [900, plan],
      [1000, { type: "gate_applied", removed_tasks: [], forced_analysis: true, reason: "g" }],
      [1010, { type: "task_started", task_id: "t1", task_type: "research" }],
      [1010, { type: "task_started", task_id: "t2", task_type: "research" }],
      [1800, { type: "task_completed", task_id: "t1", task_type: "research", summary: "ok" }],
      [2500, { type: "task_completed", task_id: "t2", task_type: "research", summary: "ok" }],
    ]);
    t.status = "running";
    const s = Object.fromEntries(buildTrace(t, 2600).map((x) => [x.node, x]));
    expect(s.route).toMatchObject({ status: "done", ms: 300 });
    expect(s.plan.ms).toBe(600);
    expect(s.gate.lines.join()).toContain("유지");
    expect(s.research.ms).toBe(1500); // 1000 → 2500
    expect(s.research.subs.map((x) => x.ms)).toEqual([790, 1490]); // 병렬 서브작업 각자 측정
    expect(s.research.status).toBe("running"); // 아직 이 단계가 마지막 활동
    expect(s.analysis.status).toBe("pending");
  });

  it("task 일부 실패는 경고 톤, 재계획은 라운드로 구분, 에스컬레이션은 alert", () => {
    const t = run([
      [100, { type: "route_decided", route: "analysis", reasoning: "r" }],
      [200, plan],
      [300, { type: "task_started", task_id: "t1", task_type: "research" }],
      [500, { type: "task_failed", task_id: "t1", task_type: "research", error: "timeout" }],
      [600, { type: "replan", round: 1, reason: "cov", lookback_days: 14 }],
      [610, { type: "task_started", task_id: "t1", task_type: "research" }],
      [900, { type: "task_completed", task_id: "t1", task_type: "research", summary: "ok" }],
      [950, { type: "task_started", task_id: "t3", task_type: "analysis" }],
      [1100, { type: "task_completed", task_id: "t3", task_type: "analysis", summary: "ok" }],
      [1110, { type: "safety_escalation", reason: "critical 1건", message: "m" }],
      [1500, { type: "validation", passed: true, issues: [], attempt: 1 }],
      [1600, { type: "done", thread_id: "th", report: "r", escalation: true }],
    ]);
    const s = Object.fromEntries(buildTrace(t).map((x) => [x.node, x]));
    expect(s.research.tone).toBe("warn");
    expect(s.research.subs.map((x) => x.round)).toEqual([0, 1]);
    expect(s.replan_check.lines[0]).toContain("1회째 다시 조사");
    expect(s.safety_gate.status).toBe("alert");
    expect(s.finalize.status).toBe("done");
    expect(s.validator.status).toBe("done");
  });

  it("단순 경로는 4개 노드만, 게이트가 제거한 분석은 건너뜀", () => {
    const t = run([
      [100, { type: "route_decided", route: "simple", reasoning: "r" }],
      [900, { type: "validation", passed: true, issues: [], attempt: 1 }],
      [950, { type: "done", thread_id: "th", report: "r", escalation: false }],
    ]);
    expect(buildTrace(t).map((x) => x.node)).toEqual(["route", "writing", "validator", "finalize"]);
    const t2 = run([
      [100, { type: "route_decided", route: "analysis", reasoning: "r" }],
      [200, plan],
      [300, { type: "task_started", task_id: "t4", task_type: "writing" }],
      [900, { type: "validation", passed: true, issues: [], attempt: 1 }],
      [950, { type: "done", thread_id: "th", report: "r", escalation: false }],
    ]);
    expect(buildTrace(t2).find((x) => x.node === "analysis")!.status).toBe("skipped");
  });

  it("검증 실패 → 재작성 중에는 writing 이 진행, validator 는 진행 표시", () => {
    const t = run([
      [100, { type: "route_decided", route: "analysis", reasoning: "r" }],
      [200, { type: "token", text: "초안" }],
      [400, { type: "validation", passed: false, issues: ["면책 누락"], attempt: 1 }],
      [600, { type: "token", text: "재작성" }],
    ]);
    const s = Object.fromEntries(buildTrace(t, 700).map((x) => [x.node, x]));
    expect(s.writing.status).toBe("running");
    expect(s.validator.status).toBe("running");
    expect(s.validator.subs[0]).toMatchObject({ status: "failed", note: "면책 누락" });
  });
});

describe("서버 계약 보강", () => {
  it("anomaly.stats 는 선택 필드(임의 키/값)로 받아들인다", () => {
    const base = { anomaly_id: "a", type: "personal_deviation", start: "s", end: "e", duration_min: 1, severity: "warning", value: 1, baseline: 60, context: "rest", evidence: "평소보다 높음" };
    const ev = (extra: object) => parseEvent({ event: "anomalies_found", data: JSON.stringify({ type: "anomalies_found", count: 1, counts_by_type: {}, anomalies: [{ ...base, ...extra }] }) });
    expect(ev({})).toMatchObject({ anomalies: [{ anomaly_id: "a" }] });
    expect(ev({ stats: { z: 5.1, z_threshold: 4 } })).toMatchObject({ anomalies: [{ stats: { z: 5.1 } }] });
    expect(() => ev({ stats: "bad" })).toThrow();
  });
  it("token.reset=true 는 버퍼를 교체하고, reset 이 없으면 누적한다", () => {
    let t = newTurn("1", "q");
    t = applyEvent(t, { type: "token", text: "초안", reset: true }, 1);
    t = applyEvent(t, { type: "token", text: " 계속" }, 2);
    expect(t.report).toBe("초안 계속");
    t = applyEvent(t, { type: "token", text: "교체", reset: true }, 3);
    expect(t.report).toBe("교체");
  });
  it("zod: token.reset/validation.degraded 는 선택 필드", () => {
    expect(parseEvent({ event: "token", data: '{"type":"token","text":"a"}' })).toEqual({ type: "token", text: "a" });
    expect(parseEvent({ event: "token", data: '{"type":"token","text":"a","reset":true}' })).toMatchObject({ reset: true });
    expect(parseEvent({ event: "validation", data: '{"type":"validation","passed":false,"issues":[],"attempt":2,"degraded":true}' })).toMatchObject({ degraded: true });
    expect(() => parseEvent({ event: "token", data: '{"type":"token","text":"a","reset":"yes"}' })).toThrow();
  });
  it("트레이스의 조회 기간: gate_applied.lookback_days 가 있으면 그 값, 없으면 계획의 값", () => {
    const plan: ChatEvent = { type: "plan_created", needs_analysis: true, lookback_days: 7, tasks: [{ id: "t1", type: "research", description: "d" }] };
    const base = (gate: Partial<Extract<ChatEvent, { type: "gate_applied" }>>) => {
      let t = newTurn("1", "q");
      t = applyEvent(t, { type: "route_decided", route: "analysis", reasoning: "r" }, 1);
      t = applyEvent(t, plan, 2);
      return applyEvent(t, { type: "gate_applied", removed_tasks: [], forced_analysis: false, reason: "g", ...gate }, 3);
    };
    const line = (t: ReturnType<typeof base>) => buildTrace(t).find((s) => s.node === "plan")!.lines[0];
    expect(line(base({ lookback_days: 30 }))).toContain("최근 30일");
    expect(line(base({}))).toContain("최근 7일");
    expect(parseEvent({ event: "gate_applied", data: '{"type":"gate_applied","removed_tasks":[],"forced_analysis":false,"reason":"x","lookback_days":14}' })).toMatchObject({ lookback_days: 14 });
  });
  it("degraded 검증은 트레이스 서브스텝에 표시된다", () => {
    let t = newTurn("1", "q");
    t = applyEvent(t, { type: "route_decided", route: "analysis", reasoning: "r" }, 1);
    t = applyEvent(t, { type: "validation", passed: false, issues: ["x"], attempt: 2, degraded: true }, 5);
    expect(buildTrace(t).find((s) => s.node === "validator")!.subs[0].note).toContain("대체 문안");
    expect(t.validations[0].degraded).toBe(true);
  });
  it("splitReport: 맨 위 중요 안내와 맨 끝 면책을 본문에서 분리한다", () => {
    const r = splitReport("**중요 안내** 즉시 119에 연락하세요.\n\n## 점검\n\n- 빈맥 128 bpm\n\n본 서비스는 의료기기가 아니며 진단을 대신하지 않습니다.");
    expect(r.notice).toBe("즉시 119에 연락하세요.");
    expect(r.body).toBe("## 점검\n\n- 빈맥 128 bpm");
    expect(r.disclaimer).toContain("의료기기");
  });
  it("splitReport: 고정 문구가 없으면 그대로, 인용(>) 형태 면책도 인식", () => {
    expect(splitReport("## 제목\n\n본문")).toEqual({ notice: null, body: "## 제목\n\n본문", disclaimer: null });
    expect(splitReport("본문\n\n> 의료진과 상담하세요.").disclaimer).toBe("의료진과 상담하세요.");
    // 단락이 하나뿐이면 면책으로 떼지 않는다(본문 소실 방지)
    expect(splitReport("본 서비스는 의료기기가 아닙니다.").disclaimer).toBeNull();
  });
  it("/summary 응답 스키마: 널 허용 필드와 추가 필드를 받아들인다", () => {
    const ok = SummaryResponse.safeParse({ user_id: "U01", window: { start: "a", end: "b" }, summary: { coverage_pct: 90, hr_mean: 70, hr_p05: 52, hr_p95: 98, resting_hr: null, sleep_hr: 55, active_minutes: 100, n_minutes: 1000 }, daily_resting_hr: [{ date: "2016-05-01", resting_hr: 62, n_rest_minutes: 300 }] });
    expect(ok.success).toBe(true);
    expect(SummaryResponse.safeParse({ summary: { coverage_pct: "x" } }).success).toBe(false);
  });
});

describe("LLM 연결 상태 문구", () => {
  const h = (llm: boolean, ollama?: object) => ({ status: "ok", mcp: true, llm, ollama });
  it("연결되면 원인 없이 connected", () => {
    expect(describeLlm(h(true))).toEqual({ connected: true });
  });
  it("서버가 없으면(null) 표시하지 않는다", () => {
    expect(describeLlm(null)).toBeNull();
  });
  it("원인은 꺼짐 → 서버 미도달 → 모델 미설치 순으로 가장 앞선 것", () => {
    expect(describeLlm(h(false, { enabled: false, reachable: false }))?.reason).toContain("꺼져");
    expect(describeLlm(h(false, { enabled: true, reachable: false }))?.reason).toContain("연결되지");
    expect(describeLlm(h(false, { enabled: true, reachable: true, missing: ["bge-m3", "qwen3:8b"] }))?.reason).toContain("bge-m3, qwen3:8b");
    expect(describeLlm(h(false))).toEqual({ connected: false }); // 세부가 없어도 "미연결" 은 알린다
  });
});

describe("헤더 버전 문구", () => {
  const v = { supervisor_version: "v2", llm_backend: "ollama", fast_model: "qwen3:0.6b", quality_model: "qwen3:8b", checkpointer: "memory", git_commit: "ab12cd3" };
  it("기본 화면은 그래프·백엔드·모델만, 개발자 보기에서 저장소·커밋을 더한다", () => {
    expect(formatVersion(v)?.short).toBe("v2 · ollama · qwen3:0.6b / qwen3:8b");
    expect(formatVersion(v, true)?.short).toBe("v2 · ollama · qwen3:0.6b / qwen3:8b · memory · ab12cd3");
    expect(formatVersion(v)?.title).toContain("빠른 모델 qwen3:0.6b");
  });
  it("fake 백엔드는 모델을 생략하고, 값이 하나도 없거나 null 이면 표시하지 않는다", () => {
    expect(formatVersion({ ...v, llm_backend: "fake" })?.short).toBe("v2 · fake");
    expect(formatVersion(null)).toBeNull();
    expect(formatVersion({})).toBeNull();
    expect(formatVersion({ supervisor_version: "v1", fast_model: null })?.short).toBe("v1");
    // 서버가 값 없음을 "-"/"none" 으로 보내는 경우
    expect(formatVersion({ supervisor_version: "v2", llm_backend: "none", fast_model: "-", quality_model: "-" })?.short).toBe("v2 · LLM 없음");
  });
});

describe("기준 시각·가명 표기", () => {
  it("fmtDateTimeY: 연도를 포함한 고정폭 표기(UTC 필드 그대로)", () => {
    expect(fmtDateTimeY(tsToMs("2016-05-12T15:00:00"))).toBe("2016-05-12 15:00");
    expect(fmtDateTimeY(tsToMs("2016-01-02T03:04:00"))).toBe("2016-01-02 03:04");
  });
  it("participantNo: 가명 ID 에서 참가자 번호를 뽑는다", () => {
    expect(participantNo("U05")).toBe(5);
    expect(participantNo("U15")).toBe(15);
    expect(participantNo("U007")).toBe(7);
    expect(participantNo("abc")).toBeNull();
    expect(participantNo("U")).toBeNull();
  });
});

describe("실제 사건 라벨과 첫 경고", () => {
  const onsetDay = "2020-03-11";
  const markers = buildMarkers([
    { type: "recovery", date: "2020-03-19", illness: "covid" },
    { type: "symptom_onset", date: onsetDay, illness: "covid" },
    { type: "diagnosis", date: "2020-03-13", illness: "covid" },
  ]);
  it("eventLabel/buildMarkers: 종류·질병을 쉬운 말로, 시각 순으로, 하루의 정오에 둔다", () => {
    expect(eventLabel({ type: "symptom_onset", illness: "covid" })).toBe("증상 시작 · 코로나19");
    expect(eventLabel({ type: "recovery", illness: "other" })).toBe("회복 · 기타 질병");
    expect(markers.map((m) => m.type)).toEqual(["symptom_onset", "diagnosis", "recovery"]);
    expect(markers[0].t).toBe(tsToMs(`${onsetDay}T12:00:00`));
    expect(shortEventLabel(markers[0])).toBe("증상 시작 2020-03-11");
    expect(buildMarkers([{ type: "diagnosis", date: "날짜아님", illness: "covid" }])).toEqual([]);
  });
  const ko = (t: string) => ({ resting_tachycardia: "안정 시 빈맥" })[t as "resting_tachycardia"] ?? t;
  const W = { evaluable: true, hit: true, pre_days: 7, post_days: 3 };
  it("warningText: 서버 값을 경우별 쉬운 말로 — 전/당일/후, 경고 없음, 평가 불가, 필드 없음은 표시 생략", () => {
    expect(warningText({ ...W, lead_days: 3, first_alarm: "2020-03-08", types: ["resting_tachycardia"] }, ko)).toBe("증상 3일 전 첫 경고 (안정 시 빈맥, 2020-03-08)");
    expect(warningText({ ...W, lead_days: 0, first_alarm: "2020-03-11", types: [] }, ko)).toBe("증상 당일 첫 경고 (2020-03-11)");
    expect(warningText({ ...W, lead_days: -1, first_alarm: "2020-03-12", types: ["resting_tachycardia"] }, ko)).toBe("증상 1일 후 첫 경고 (안정 시 빈맥, 2020-03-12)");
    expect(warningText({ ...W, hit: false, lead_days: null, first_alarm: null }, ko)).toBe("증상 −7~+3일 안에는 경고가 없었습니다");
    expect(warningText({ evaluable: false, hit: false }, ko)).toBe("이 사건 주변은 평가할 수 없습니다");
    expect(warningText(undefined, ko)).toBeNull();
  });
  it("leadsFor: 서버가 계산한 경고 날짜(정오)만 쓴다 — 이상 목록·보는 구간과 무관", () => {
    const mk = buildMarkers([
      { type: "symptom_onset", date: onsetDay, illness: "covid", first_warning: { ...W, lead_days: 3, first_alarm: "2020-03-08", types: [] } },
      { type: "symptom_onset", date: "2020-05-01", illness: "covid", first_warning: { ...W, hit: false, lead_days: null, first_alarm: null } },
      { type: "diagnosis", date: "2020-03-13", illness: "covid" },
    ]);
    const leads = leadsFor(mk);
    expect(leads).toHaveLength(1);
    expect(leads[0]).toMatchObject({ text: "증상 3일 전 첫 경고", alarmMs: tsToMs("2020-03-08T12:00:00") });
  });
  it("목업 Stanford: 목록 분리·수면 여부·사건·사건 주변 구간의 첫 경고", () => {
    expect(mockUsers("fitbit").every((u) => u.dataset === "fitbit")).toBe(true);
    const st = mockUsers("stanford");
    expect(st.map((u) => u.user_id)).toEqual(["S001", "S002", "S003", "S004", "S005"]);
    expect(st.find((u) => u.user_id === "S002")?.has_sleep).toBe(false);
    expect(mockEvents("S003").events).toEqual([]);
    const ev = mockEvents("S001").events;
    expect(ev.map((e) => e.type)).toEqual(["symptom_onset", "diagnosis", "recovery"]);
    // S001 의 증상 시작은 기록 끝에서 40일 전 → 마지막 30일 밖이라 이상 목록에는 안 보여도 첫 경고는 서버 값이라 늘 같다
    const mk = buildMarkers(ev);
    expect(mockAnomalies("S001", 30).some((x) => x.type === "resting_tachycardia")).toBe(false);
    expect(leadsFor(mk).map((l) => l.text)).toEqual(["증상 3일 전 첫 경고"]);
    // S004: 증상 시작 앞에 경고를 심지 않았다 → 경고 없음 문구(차트 표시는 없음)
    const mk4 = buildMarkers(mockEvents("S004").events);
    expect(leadsFor(mk4)).toEqual([]);
    expect(warningText(mk4[0].warning, ko)).toBe("증상 −7~+3일 안에는 경고가 없었습니다");
    // 사용자 기록 기간 밖은 점이 없다
    expect(mockSeries("S001", 7, "2019-01-01T00:00:00")).toEqual([]);
  });
});

describe("서버 계약 보강 2: retrieval/prompt, 승인 요청, 사용량", () => {
  const P = (obj: object) => parseEvent({ event: (obj as { type: string }).type, data: JSON.stringify(obj) });
  it("task_* 이벤트가 retrieval/prompt 종류와 retries 를 받아들이고, 모르는 종류는 거부한다", () => {
    expect(P({ type: "task_started", task_id: "retrieval", task_type: "retrieval" })).toMatchObject({ task_type: "retrieval" });
    expect(P({ type: "task_completed", task_id: "mcp_prompt", task_type: "prompt", summary: "ok", retries: 2 })).toMatchObject({ retries: 2 });
    expect(() => P({ type: "task_started", task_id: "x", task_type: "mystery" })).toThrow();
  });
  it("approval_required 와 done.usage(선택) 를 검증한다", () => {
    expect(P({ type: "approval_required", thread_id: "t1", reason: "r", message_preview: "p", approval_id: "a1" })).toMatchObject({ approval_id: "a1" });
    expect(() => P({ type: "approval_required", thread_id: "t1", reason: "r", message_preview: "p" })).toThrow(); // approval_id 필수
    const done = P({ type: "done", thread_id: "t", report: "r", escalation: false, run_id: null, usage: { llm_calls: 3, total_tokens: 100, extra: "ok" } });
    expect(done).toMatchObject({ usage: { llm_calls: 3 } });
    expect(P({ type: "done", thread_id: "t", report: "r", escalation: false })).not.toHaveProperty("usage");
  });
  it("turn: approval_required 는 승인 대기를 만들고 done 은 usage 를 싣는다", () => {
    let t = newTurn("1", "q");
    t = applyEvent(t, { type: "approval_required", thread_id: "t1", reason: "r", message_preview: "미리보기", approval_id: "a1" }, 5);
    expect(t.approval).toMatchObject({ status: "pending", thread_id: "t1", approval_id: "a1", message_preview: "미리보기" });
    t = applyEvent(t, { type: "done", thread_id: "t1", report: "r", escalation: true, usage: { llm_calls: 2 } }, 9);
    expect(t.usage).toEqual({ llm_calls: 2 });
    expect(t.approval?.status).toBe("pending"); // 승인은 done 과 별개로 결정된다
  });
  it("trace: prompt 는 계획 단계, retrieval 은 작성 단계의 하위 작업(고정 라벨)이고 retries 가 붙는다", () => {
    let t = newTurn("1", "q");
    const ev = (e: ChatEvent, at: number) => (t = applyEvent(t, e, at));
    ev({ type: "route_decided", route: "analysis", reasoning: "r" }, 1);
    ev({ type: "task_started", task_id: "mcp_prompt", task_type: "prompt" }, 2);
    ev({ type: "task_completed", task_id: "mcp_prompt", task_type: "prompt", summary: "MCP Prompt 적용", retries: 1 }, 300);
    ev({ type: "plan_created", needs_analysis: true, lookback_days: 7, tasks: [{ id: "t1", type: "research", description: "d" }] }, 400);
    const plan = buildTrace(t).find((s) => s.node === "plan")!;
    expect(plan.subs[0]).toMatchObject({ label: "점검 절차 불러오기", kind: "prompt", status: "done", retries: 1 });
    // 계획 목록에는 섞이지 않는다
    expect(t.plan!.tasks.map((x) => x.id)).toEqual(["t1"]);
    let s2 = newTurn("2", "q");
    s2 = applyEvent(s2, { type: "route_decided", route: "simple", reasoning: "r" }, 1);
    s2 = applyEvent(s2, { type: "task_started", task_id: "retrieval", task_type: "retrieval" }, 2);
    s2 = applyEvent(s2, { type: "task_completed", task_id: "retrieval", task_type: "retrieval", summary: "근거 3건" }, 350);
    const writing = buildTrace(s2).find((x) => x.node === "writing")!;
    expect(writing.subs[0]).toMatchObject({ label: "가이드라인 검색", kind: "retrieval", status: "done" });
    expect(writing.subs[0].retries).toBeUndefined();
  });
  it("formatUsage: 값 있는 항목만, 비용은 참고 표시, 없으면 null", () => {
    expect(formatUsage({ llm_calls: 4, input_tokens: 5120, output_tokens: 980, retries: 1, cost_usd: 0 })).toBe("LLM 호출 4회, 토큰 입력 5,120 / 출력 980, 재시도 1회");
    expect(formatUsage({ llm_calls: 1, total_tokens: 10, cost_usd: 0.0123 })).toBe("LLM 호출 1회, 토큰 10, 비용 $0.0123 (참고)");
    expect(formatUsage({})).toBeNull();
    expect(formatUsage(undefined)).toBeNull();
  });
});

describe("보호자 알림 승인 API", () => {
  afterEach(() => vi.unstubAllGlobals());
  const stub = (status: number, body: unknown = {}) =>
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } })));
  it("200: 결과 한 줄이 덧붙은 최종 리포트를 돌려준다", async () => {
    stub(200, { thread_id: "t1", approved: true, notification: "sent_simulated", report: "본문\n\n보호자 알림: 전송", approval: { status: "approved" } });
    const r = await approveChat("t1", true, "a1");
    expect(r).toMatchObject({ ok: true, data: { notification: "sent_simulated" } });
  });
  it("409(이미 처리됨)는 오류가 아니라 conflict 로, 그 밖의 오류는 실패로 돌려준다", async () => {
    stub(409, { detail: "승인 대기 중인 요청이 없습니다" });
    expect(await approveChat("t1", true, "a1")).toEqual({ ok: false, conflict: true, status: 409 });
    stub(503);
    expect(await approveChat("t1", false, "a1")).toEqual({ ok: false, conflict: false, status: 503 });
  });
  it("요청 본문은 {thread_id, approved, approval_id} 이고 POST 로 보낸다", async () => {
    const f = vi.fn(async () => new Response(JSON.stringify({ thread_id: "t9", approved: false, notification: "not_sent", report: "r" }), { status: 200 }));
    vi.stubGlobal("fetch", f);
    await approveChat("t9", false, "a7");
    const [url, init] = f.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toMatch(/\/chat\/approve$/);
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({ thread_id: "t9", approved: false, approval_id: "a7" });
  });
  it("목업: 첫 결정은 성공, 같은 approval_id 의 두 번째는 409", async () => {
    const id = pendingMockApproval("mock-X");
    expect((await mockApprove("mock-X", true, id)).ok).toBe(true);
    expect(await mockApprove("mock-X", true, id)).toMatchObject({ ok: false, conflict: true });
  });
  it("오래된 카드는 새 요청을 승인할 수 없다: 같은 thread 의 최신 approval_id 만 유효", async () => {
    const turns = ["a1", "a2"].map((id, i) => applyEvent(newTurn(String(i), "q"), { type: "approval_required", thread_id: "t1", reason: "r", message_preview: "p", approval_id: id }, 1));
    const latest = latestApprovals(turns);
    expect(latest.get("t1")).toBe("a2");
    expect(turns.map((t) => latest.get(t.approval!.thread_id) !== t.approval!.approval_id)).toEqual([true, false]); // 첫 카드만 stale
    const old = pendingMockApproval("mock-Y");
    pendingMockApproval("mock-Y"); // 새 요청이 생김
    expect(await mockApprove("mock-Y", true, old)).toMatchObject({ ok: false, conflict: true });
  });
});

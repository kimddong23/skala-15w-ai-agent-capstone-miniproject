/**
 * SSE 전송 계층 테스트 — 프레임 파서(청크 경계·CRLF·주석), zod 검증(거부 사례), 이벤트 리듀서 기본 전이.
 * 네트워크는 쓰지 않는다. 근거: (Custom Transport + 런타임 검증).
 */
import { describe, expect, it } from "vitest";
import { createSseParser, parseEvent } from "./sse";
import { applyEvent, newTurn } from "./turn";

const frame = (type: string, data: object) => `event: ${type}\ndata: ${JSON.stringify({ type, ...data })}\n\n`;

describe("SSE 파서", () => {
  it("청크 경계가 프레임 중간(필드 한가운데)에서 잘려도 복원한다", () => {
    const wire = frame("token", { text: "안녕" }) + frame("done", { thread_id: "t1", report: "r", escalation: false });
    const p = createSseParser();
    const frames = [];
    for (let i = 0; i < wire.length; i += 7) frames.push(...p.push(wire.slice(i, i + 7)));
    expect(frames.map((f) => f.event)).toEqual(["token", "done"]);
    expect(parseEvent(frames[0])).toEqual({ type: "token", text: "안녕" });
  });

  it("한 청크에 여러 프레임, CRLF, 주석(keep-alive)을 처리한다", () => {
    const wire = ": ping\n\n" + frame("token", { text: "a" }).replace(/\n/g, "\r\n") + frame("token", { text: "b" });
    const frames = createSseParser().push(wire);
    expect(frames.map((f) => parseEvent(f))).toEqual([
      { type: "token", text: "a" },
      { type: "token", text: "b" },
    ]);
  });

  it("빈 줄 없이 끝난 마지막 프레임은 flush 로 회수한다", () => {
    const p = createSseParser();
    expect(p.push('event: error\ndata: {"type":"error","message":"x"}')).toEqual([]);
    expect(p.flush()).toHaveLength(1);
  });
});

describe("zod 검증", () => {
  const f = (event: string, data: string) => ({ event, data });
  it("알 수 없는 type 을 거부한다", () => {
    expect(() => parseEvent(f("nope", '{"type":"nope"}'))).toThrow();
  });
  it("필수 필드 누락/타입 오류를 거부한다", () => {
    expect(() => parseEvent(f("token", '{"type":"token"}'))).toThrow();
    expect(() => parseEvent(f("validation", '{"type":"validation","passed":"yes","issues":[],"attempt":1}'))).toThrow();
  });
  it("깨진 JSON 과 event/type 불일치를 거부한다", () => {
    expect(() => parseEvent(f("token", "{oops"))).toThrow();
    expect(() => parseEvent(f("done", '{"type":"token","text":"x"}'))).toThrow();
  });
  it("anomalies_found 의 severity enum 을 검증한다", () => {
    const a = { anomaly_id: "a", type: "bradycardia", start: "s", end: "e", duration_min: 1, severity: "bad", value: 1, baseline: null, context: "rest", evidence: "" };
    expect(() => parseEvent(f("anomalies_found", JSON.stringify({ type: "anomalies_found", count: 1, counts_by_type: {}, anomalies: [a] })))).toThrow();
  });
});

describe("turn 리듀서", () => {
  it("task 상태 전이와 검증 실패 후 재작성 시 본문을 교체한다", () => {
    let t = newTurn("1", "q");
    const ev = (e: Parameters<typeof applyEvent>[1]) => (t = applyEvent(t, e));
    ev({ type: "plan_created", needs_analysis: true, lookback_days: 7, tasks: [{ id: "t1", type: "research", description: "d" }] });
    ev({ type: "task_started", task_id: "t1", task_type: "research" });
    expect(t.plan!.tasks[0].status).toBe("running");
    ev({ type: "task_failed", task_id: "t1", task_type: "research", error: "boom" });
    expect(t.plan!.tasks[0]).toMatchObject({ status: "failed", note: "boom" });
    ev({ type: "token", text: "초안" });
    ev({ type: "validation", passed: false, issues: ["x"], attempt: 1 });
    ev({ type: "token", text: "재작성" });
    expect(t.report).toBe("재작성");
    ev({ type: "done", thread_id: "th", report: "최종", escalation: false });
    expect(t).toMatchObject({ status: "done", report: "최종", threadId: "th" });
  });
});

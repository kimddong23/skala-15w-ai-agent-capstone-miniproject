/**
 * 질문 1건(turn)의 상태 모델과, SSE 이벤트를 상태에 접어 넣는 순수 리듀서.
 *
 * 책임: `ChatEvent` 를 받아 화면이 필요로 하는 파생 상태(계획·게이트·이상·리포트·검증)로 변환하고,
 *       실행 트레이스(trace.ts)가 소요 시간을 계산할 수 있도록 이벤트 도착 시각을 기록한다.
 * 경계: React/네트워크를 모른다(테스트 가능한 순수 함수). 표시 로직은 Run.tsx, 시간 계산은 trace.ts 가 맡는다.
 * 협력: events.ts(입력 타입), trace.ts(log 소비), Run.tsx(렌더).
 */
import type { Anomaly, ChatEvent, PlanTask, Usage } from "./events";

/** 계획 작업의 진행 상태. `removed` 는 이중 게이트가 제거한 작업. */
export type TaskStatus = "pending" | "running" | "done" | "failed" | "removed";
export interface TaskState extends PlanTask {
  status: TaskStatus;
  /** 완료 시 요약, 실패 시 사유. */
  note?: string;
}

/**
 * 이벤트 도착 기록. `at` 은 질문 전송 후 경과 ms(클라이언트 측정).
 * `token` 은 양이 많아 기록하지 않고 `Turn.tokens` 에 요약한다.
 */
export interface LogEntry {
  at: number;
  ev: ChatEvent;
  /** `validation` 직전의 마지막 token 도착 시각. 검증 시도별 소요 계산에 쓴다. */
  prevToken?: number;
}

/** 질문 1건의 누적 상태. 불변 객체로 취급하고 `applyEvent` 로만 갱신한다. */
export interface Turn {
  id: string;
  question: string;
  status: "running" | "done" | "error" | "aborted";
  route?: { route: "simple" | "analysis"; reasoning: string };
  plan?: { tasks: TaskState[]; needs_analysis: boolean; lookback_days: number };
  gate?: { removed_tasks: string[]; forced_analysis: boolean; reason: string; lookback_days?: number };
  replans: { round: number; reason: string; lookback_days: number }[];
  anomalies?: { count: number; counts_by_type: Record<string, number>; anomalies: Anomaly[] };
  escalation?: { reason: string; message: string };
  /** 현재 리포트 본문(스트리밍 중에는 부분 텍스트). */
  report: string;
  /** 검증 실패 직후 true. 재작성의 첫 token 이 오면 본문을 비우고 새로 시작한다. */
  staleReport: boolean;
  validations: { passed: boolean; issues: string[]; attempt: number; degraded?: boolean }[];
  /**
   * 보호자 알림 승인 요청(HITL). 에스컬레이션 안내문은 승인과 무관하게 이미 리포트에 있다.
   * status: pending(결정 대기) → sending(전송 중) → approved|denied(결과) | conflict(이미 처리됨) | error(전송 실패, 다시 시도 가능).
   */
  approval?: { thread_id: string; approval_id: string; reason: string; message_preview: string; status: "pending" | "sending" | "approved" | "denied" | "conflict" | "error"; resultLine?: string };
  /** 요청 전체의 LLM 사용량(개발자 보기에서만 표시). */
  usage?: Usage;
  threadId?: string;
  error?: string;
  /** token 을 제외한 이벤트의 도착 기록(시간 순). */
  log: LogEntry[];
  /** 첫/마지막 token 도착 시각(ms). */
  tokens: { first?: number; last?: number };
  /** 마지막 이벤트까지의 경과 ms. */
  elapsedMs: number;
}

/**
 * 빈 turn 을 만든다.
 * @param id 고유 ID(렌더 key 와 DOM `data-run` 에 쓰임).
 * @param question 사용자가 보낸 질문 원문.
 */
export const newTurn = (id: string, question: string): Turn => ({
  id,
  question,
  status: "running",
  replans: [],
  report: "",
  staleReport: false,
  validations: [],
  log: [],
  tokens: {},
  elapsedMs: 0,
});

// replan 으로 계획에 없던 task 가 시작될 수 있어, 없으면 목록에 추가한다.
const setTask = (t: Turn, id: string, patch: Partial<TaskState>, fallback: PlanTask): Turn => {
  const tasks = t.plan?.tasks ?? [];
  const exists = tasks.some((x) => x.id === id);
  const next = exists
    ? tasks.map((x) => (x.id === id ? { ...x, ...patch } : x))
    : [...tasks, { ...fallback, status: "pending" as TaskStatus, ...patch }];
  return { ...t, plan: { needs_analysis: false, lookback_days: 0, ...t.plan, tasks: next } };
};

/**
 * 이벤트 1건을 turn 에 적용해 새 turn 을 반환한다(원본 불변).
 *
 * @param t 현재 상태.
 * @param ev 검증을 통과한 이벤트.
 * @param at 질문 전송 후 경과 ms. 테스트 등에서 시간을 쓰지 않으면 생략(0).
 * @remarks `token` 은 log 에 쌓지 않는다(수백 건이 되어 리렌더 비용만 키운다).
 */
export function applyEvent(t: Turn, ev: ChatEvent, at = 0): Turn {
  const base: Turn =
    ev.type === "token"
      ? { ...t, tokens: { first: t.tokens.first ?? at, last: at }, elapsedMs: at }
      : { ...t, log: [...t.log, { at, ev, prevToken: t.tokens.last }], elapsedMs: at };
  return reduce(base, ev);
}

function reduce(t: Turn, ev: ChatEvent): Turn {
  switch (ev.type) {
    case "route_decided":
      return { ...t, route: { route: ev.route, reasoning: ev.reasoning } };
    case "plan_created":
      return {
        ...t,
        plan: {
          needs_analysis: ev.needs_analysis,
          lookback_days: ev.lookback_days,
          tasks: ev.tasks.map((x) => ({ ...x, status: "pending" as TaskStatus })),
        },
      };
    case "gate_applied": {
      const tasks = t.plan?.tasks.map((x) => (ev.removed_tasks.includes(x.id) ? { ...x, status: "removed" as TaskStatus } : x));
      return {
        ...t,
        gate: { removed_tasks: ev.removed_tasks, forced_analysis: ev.forced_analysis, reason: ev.reason, lookback_days: ev.lookback_days },
        plan: t.plan && tasks ? { ...t.plan, tasks } : t.plan,
      };
    }
    // 가이드라인 검색·점검 절차 불러오기는 계획의 작업이 아니라 진행 알림이다 — 계획 목록에 섞지 않고 trace 가 log 로 그린다.
    case "task_started":
      if (ev.task_type === "retrieval" || ev.task_type === "prompt") return t;
      return setTask(t, ev.task_id, { status: "running" }, { id: ev.task_id, type: ev.task_type, description: ev.task_id });
    case "task_completed":
      if (ev.task_type === "retrieval" || ev.task_type === "prompt") return t;
      return setTask(t, ev.task_id, { status: "done", note: ev.summary }, { id: ev.task_id, type: ev.task_type, description: ev.task_id });
    case "task_failed":
      if (ev.task_type === "retrieval" || ev.task_type === "prompt") return t;
      return setTask(t, ev.task_id, { status: "failed", note: ev.error }, { id: ev.task_id, type: ev.task_type, description: ev.task_id });
    case "replan":
      return { ...t, replans: [...t.replans, { round: ev.round, reason: ev.reason, lookback_days: ev.lookback_days }] };
    case "anomalies_found":
      return { ...t, anomalies: { count: ev.count, counts_by_type: ev.counts_by_type, anomalies: ev.anomalies } };
    case "safety_escalation":
      return { ...t, escalation: { reason: ev.reason, message: ev.message } };
    case "token":
      // 서버의 reset 신호(재작성·폴백)와 검증 실패 후의 첫 토큰 모두 초안을 교체한다.
      return t.staleReport || ev.reset ? { ...t, report: ev.text, staleReport: false } : { ...t, report: t.report + ev.text };
    case "validation":
      return {
        ...t,
        validations: [...t.validations, { passed: ev.passed, issues: ev.issues, attempt: ev.attempt, degraded: ev.degraded }],
        staleReport: !ev.passed,
      };
    case "approval_required":
      return { ...t, approval: { thread_id: ev.thread_id, approval_id: ev.approval_id, reason: ev.reason, message_preview: ev.message_preview, status: "pending" } };
    case "done":
      return {
        ...t,
        usage: ev.usage ?? undefined,
        status: "done",
        staleReport: false,
        // done.report 는 서버가 확정한 최종본(폴백·고정 문구 반영)이라 누적 토큰보다 우선한다.
        report: ev.report || t.report,
        threadId: ev.thread_id,
        escalation: ev.escalation ? (t.escalation ?? { reason: "", message: "" }) : t.escalation,
      };
    case "error":
      return { ...t, status: "error", error: ev.message };
  }
}

/** thread 별로 가장 최근 승인 요청의 ID. 이보다 오래된 카드는 결정할 수 없다(서버도 409). */
export function latestApprovals(turns: Turn[]): Map<string, string> {
  const m = new Map<string, string>();
  for (const t of turns) if (t.approval) m.set(t.approval.thread_id, t.approval.approval_id);
  return m;
}

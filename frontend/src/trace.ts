/**
 * 실행 트레이스 도출 — 이벤트 도착 기록(`Turn.log`)에서 에이전트 그래프의 노드별 상태·소요 시간을 계산한다.
 *
 * 책임: SSE 이벤트 → "route → plan → gate → research → replan_check → analysis → safety_gate → writing →
 *       validator → finalize" 수직 트레이스(LangSmith 풍). 순수 함수이며 React 를 모른다.
 * 소요 시간의 정의: 단계의 "결과 이벤트" 도착 시각 − 직전 단계의 결과 도착 시각이다. 서버가 측정한 값이 아니라
 *   클라이언트 도착 기준이라 네트워크 지연이 섞인다. 서버 측 span 은 LangSmith에서 본다.
 * 협력: turn.ts(입력), Run.tsx(렌더). 노드 구성은 백엔드 그래프와 같은 이름을 쓴다.
 */
import type { ChatEvent, TaskKind } from "./events";
import type { Turn } from "./turn";

/** 단계/서브작업 상태. `alert` 는 안전 게이트의 에스컬레이션, `skipped` 는 게이트·경로 때문에 실행되지 않음. */
export type StepStatus = "pending" | "running" | "done" | "failed" | "skipped" | "alert";

/** 단계 안의 병렬 서브작업(조사 작업, 검증 시도 등). */
export interface SubStep {
  /** `task_id#round` 형태의 안정적 key. 재조회 라운드마다 같은 task 가 다시 실행될 수 있다. */
  key: string;
  label: string;
  status: StepStatus;
  /** 소요 ms. 실행 중이면 `now` 까지의 경과. */
  ms?: number;
  /** 완료 요약 / 실패 사유 / 검증 결과. */
  note?: string;
  /** 재계획 라운드(0 = 최초). */
  round: number;
  /** 하위 작업의 종류. retrieval/prompt 는 아이콘을 붙여 보인다. */
  kind?: TaskKind | "validation";
  /** 성공까지 일어난 일시 오류 재시도 횟수(0 이면 없음). */
  retries?: number;
}
/** 트레이스의 한 줄(그래프 노드 하나). */
export interface TraceStep {
  /** 그래프 노드 이름(백엔드와 동일, mono 표기). */
  node: string;
  /** 한글 설명. */
  label: string;
  status: StepStatus;
  /** 소요 ms. 미완료·측정 불가 단계는 undefined. */
  ms?: number;
  /** "warn": 일부 하위 작업이 실패했지만 부분 결과로 계속 진행. */
  tone?: "warn";
  /** 판단 근거 등 단계 아래에 인라인으로 보일 쉬운 말 설명. */
  lines: string[];
  /** 개발자 보기에서만 보이는 기술 상세(서버가 준 코드값·식별자 포함). */
  devLines: string[];
  subs: SubStep[];
}

const NODES = ["route", "plan", "gate", "research", "replan_check", "analysis", "safety_gate", "writing", "validator", "finalize"] as const;
const LABELS: Record<(typeof NODES)[number], string> = {
  route: "질문 분류",
  plan: "계획 수립",
  gate: "계획 점검",
  research: "데이터 조사",
  replan_check: "재조사 판단",
  analysis: "결과 해석",
  safety_gate: "안전 확인",
  writing: "리포트 작성",
  validator: "자동 검증",
  finalize: "마무리",
};

/**
 * 이벤트가 가리키는 그래프 단계 번호(NODES 인덱스). replan 은 research(3)로 되돌아간다는 뜻에서 3 이다.
 * error 는 단계를 진행시키지 않으므로 -1.
 */
function rank(ev: ChatEvent): number {
  switch (ev.type) {
    case "route_decided": return 0;
    case "plan_created": return 1;
    case "gate_applied": return 2;
    case "anomalies_found":
    case "replan": return 3;
    case "task_started":
    case "task_completed":
    case "task_failed":
      // prompt(점검 절차 불러오기)는 계획 노드 안에서, retrieval(가이드라인 검색)은 답변 작성 단계에서 일어난다.
      return ev.task_type === "prompt" ? 1 : ev.task_type === "research" ? 3 : ev.task_type === "analysis" ? 5 : 7;
    case "safety_escalation": return 6;
    case "approval_required": return 9; // 마무리 단계에서 사람 승인을 기다린다
    case "token": return 7;
    case "validation": return 8;
    case "done": return 9;
    case "error": return -1;
  }
}

// 값이 없는 항목을 무시하고 최댓값을 구한다(전부 없으면 undefined).
const maxAt = (xs: (number | undefined)[]) => {
  const v = xs.filter((x): x is number => x != null);
  return v.length ? Math.max(...v) : undefined;
};

/**
 * turn 의 이벤트 기록에서 트레이스 단계 목록을 만든다.
 *
 * @param t 대상 turn.
 * @param now 실행 중 단계의 경과를 계산할 현재 시각(질문 전송 후 ms). 기본은 마지막 이벤트 시각이라 정적 렌더에 쓸 수 있다.
 * @returns 분류 전이면 빈 배열. 단순 경로는 4단계(route/writing/validator/finalize)만, 분석 경로는 10단계 전부.
 */
export function buildTrace(t: Turn, now = t.elapsedMs): TraceStep[] {
  if (!t.route) return [];
  const L = t.log;
  const simple = t.route.route === "simple";
  const finished = t.status !== "running";
  // 현재 단계 = 가장 최근 활동. max 가 아니라 "마지막" 을 쓰는 이유: 검증 실패 후 재작성 중에는
  // validator(8) 에서 writing(7) 으로 되돌아오기 때문이다. token 은 log 에 없어 별도로 비교한다.
  const lastLog = L[L.length - 1];
  const cur = t.tokens.last != null && (!lastLog || t.tokens.last >= lastLog.at) ? 7 : lastLog ? rank(lastLog.ev) : -1;

  // 서브작업: 로그를 순서대로 훑어 (task_id, round) 단위로 시작/종료를 짝짓는다. 병렬 실행이라 도착 순서가
  // 섞이므로 task_id 로 짝을 찾고, 재조회로 같은 task 가 다시 시작돼도 라운드로 구분한다.
  const subs: Record<TaskKind, SubStep[]> = { research: [], analysis: [], writing: [], retrieval: [], prompt: [] };
  // 계획에 없는 작업(prompt/retrieval)은 설명을 고정 문구로 보인다.
  const FIXED_LABEL: Partial<Record<TaskKind, string>> = { retrieval: "가이드라인 검색", prompt: "점검 절차 불러오기" };
  const open = new Map<string, { sub: SubStep; start: number }>();
  let round = 0;
  const desc = (id: string) => t.plan?.tasks.find((x) => x.id === id)?.description ?? id;
  for (const { at, ev } of L) {
    if (ev.type === "replan") round = ev.round;
    if (ev.type === "task_started") {
      const sub: SubStep = { key: `${ev.task_id}#${round}`, label: FIXED_LABEL[ev.task_type] ?? desc(ev.task_id), status: "running", round, kind: ev.task_type };
      subs[ev.task_type].push(sub);
      open.set(`${ev.task_id}#${round}`, { sub, start: at });
    }
    if (ev.type === "task_completed" || ev.type === "task_failed") {
      const o = open.get(`${ev.task_id}#${round}`);
      if (o) {
        o.sub.status = ev.type === "task_completed" ? "done" : "failed";
        o.sub.ms = at - o.start;
        o.sub.note = ev.type === "task_completed" ? ev.summary : ev.error;
        if (ev.type === "task_completed" && ev.retries) o.sub.retries = ev.retries;
        open.delete(`${ev.task_id}#${round}`);
      }
    }
  }
  // 끝났는데 완료 이벤트가 없는 서브작업(중단·오류)은 실행 중으로 남기지 않는다.
  if (finished) for (const { sub } of open.values()) sub.status = "pending";
  else for (const { sub, start } of open.values()) sub.ms = Math.max(0, now - start);

  const endOf = (kind: "research" | "analysis" | "writing") =>
    maxAt(L.filter((e) => (e.ev.type === "task_completed" || e.ev.type === "task_failed") && e.ev.task_type === kind).map((e) => e.at));
  const esc = L.find((e) => e.ev.type === "safety_escalation");
  const vals = L.filter((e) => e.ev.type === "validation");
  const done = L.find((e) => e.ev.type === "done");
  const ends: Partial<Record<(typeof NODES)[number], number>> = {
    route: L.find((e) => e.ev.type === "route_decided")?.at,
    plan: L.find((e) => e.ev.type === "plan_created")?.at,
    gate: L.find((e) => e.ev.type === "gate_applied")?.at,
    research: endOf("research") ?? L.find((e) => e.ev.type === "anomalies_found")?.at,
    analysis: endOf("analysis"),
    safety_gate: esc?.at,
    writing: endOf("writing") ?? t.tokens.last,
    validator: vals.length ? vals[vals.length - 1].at : undefined,
    finalize: done?.at,
  };

  // 검증 시도별 소요: 그 시도가 검증한 초안의 마지막 토큰(없으면 직전 시도) 이후 경과.
  let prevAnchor: number | undefined;
  const validatorSubs: SubStep[] = vals.map((v, i) => {
    const ev = v.ev as Extract<ChatEvent, { type: "validation" }>;
    const anchor = v.prevToken ?? prevAnchor ?? ends.writing;
    prevAnchor = v.at;
    return {
      key: `v${i}`,
      label: `${ev.attempt}번째 검증`,
      status: ev.passed ? "done" : "failed",
      ms: anchor != null ? Math.max(0, v.at - anchor) : undefined,
      note: (ev.passed ? "통과" : ev.issues.join(", ")) + (ev.degraded ? ". 대체 문안을 사용했습니다." : ""),
      round: 0,
    };
  });
  const validatorMs = validatorSubs.reduce((s, x) => s + (x.ms ?? 0), 0);

  const names = simple ? (["route", "writing", "validator", "finalize"] as const) : NODES;
  let prevEnd = 0;
  return names.map((node): TraceStep => {
    const idx = NODES.indexOf(node);
    const end = ends[node];
    let status: StepStatus;
    if (idx < cur) status = "done";
    else if (idx === cur) status = !finished ? "running" : t.status === "error" ? "failed" : "done";
    else status = finished ? (t.status === "error" ? "pending" : "skipped") : "pending";
    // replan_check 는 코드 판정이라 자체 이벤트가 없다 — 재계획이 있었거나 분석 단계로 넘어갔으면 수행된 것으로 본다.
    if (node === "replan_check") status = t.replans.length || cur >= 5 ? "done" : finished && t.status === "done" ? "skipped" : "pending";
    if (node === "analysis" && !subs.analysis.length && cur > 5) status = "skipped";
    if (node === "validator" && vals.length) {
      const last = vals[vals.length - 1].ev as Extract<ChatEvent, { type: "validation" }>;
      // degraded 는 검증이 제 역할(불합격 판정)을 하고 기본 리포트로 넘긴 것이라 완료로 센다.
      status = last.passed || last.degraded ? "done" : finished ? "failed" : "running";
    }
    if (node === "safety_gate" && esc) status = "alert";

    const lines: string[] = [];
    const devLines: string[] = [];
    let ms: number | undefined;
    let tone: TraceStep["tone"];
    let stepSubs: SubStep[] = [];
    switch (node) {
      case "route":
        lines.push(`${t.route!.route === "analysis" ? "데이터 분석이 필요한 질문" : "바로 답할 수 있는 질문"}: ${t.route!.reasoning}`);
        break;
      case "plan":
        stepSubs = subs.prompt;
        // 실제 조회 기간은 게이트가 확정한다(이전 기간을 이어받는 후속 질문에서는 계획의 값과 다를 수 있다). 없으면 계획의 값.
        if (t.plan) lines.push(`할 일 ${t.plan.tasks.length}개, 최근 ${t.gate?.lookback_days ?? t.plan.lookback_days}일 확인`);
        if (t.plan) devLines.push(`needs_analysis=${t.plan.needs_analysis} (LLM 1차 판단)`);
        break;
      case "gate":
        if (t.gate) {
          if (t.gate.forced_analysis) lines.push("점검 질문이라 분석 단계를 빼지 않고 유지");
          if (t.gate.removed_tasks.length) lines.push(`필요 없는 작업을 뺌: ${t.gate.removed_tasks.join(", ")}`);
          if (!t.gate.forced_analysis && !t.gate.removed_tasks.length) lines.push("계획을 그대로 사용");
          lines.push(t.gate.reason);
        }
        break;
      case "research":
        stepSubs = subs.research;
        if (stepSubs.some((s) => s.status === "failed")) tone = "warn";
        if (stepSubs.length && stepSubs.every((s) => s.status === "failed")) status = "failed";
        if (t.anomalies) lines.push(`이상 ${t.anomalies.count}건 발견`);
        break;
      case "replan_check":
        if (t.replans.length) for (const r of t.replans) lines.push(`${r.round}회째 다시 조사: 기간을 ${r.lookback_days}일로 넓힘. ${r.reason}`);
        else if (status === "done") lines.push("데이터가 충분해 다시 조사하지 않음");
        break;
      case "analysis":
        stepSubs = subs.analysis;
        break;
      case "safety_gate":
        if (esc) {
          lines.push("심각한 이상이 있어 안전 안내문을 붙임");
          devLines.push(esc.ev.type === "safety_escalation" ? esc.ev.reason : "");
        } else if (status === "done") lines.push("심각한 이상 없음");
        break;
      case "writing":
        stepSubs = subs.retrieval;
        if (t.report) lines.push(`${t.report.length.toLocaleString("ko-KR")}자${status === "running" ? " 작성 중…" : ""}`);
        if (vals.some((v) => v.ev.type === "validation" && !v.ev.passed)) lines.push("검증에서 문제가 있어 다시 작성");
        break;
      case "validator":
        stepSubs = validatorSubs;
        if (vals.some((v) => v.ev.type === "validation" && v.ev.degraded)) lines.push("검증을 통과하지 못해 기본 리포트로 대체");
        break;
      case "finalize":
        if (t.status === "error") lines.push("실행 중 오류가 발생했습니다.");
        if (t.status === "error") devLines.push(t.error ?? "");
        else if (done) devLines.push(`thread_id=${(done.ev as Extract<ChatEvent, { type: "done" }>).thread_id}`);
        break;
    }

    // 소요 시간: 결과 이벤트가 있으면 (결과 시각 − 직전 단계 결과 시각), 실행 중이면 now 까지의 경과.
    if (end != null && node !== "safety_gate") {
      ms = Math.max(0, end - prevEnd);
      // 재작성 루프에서는 writing 구간에 앞선 실패 검증 시간이 끼어 있어 그만큼 뺀다.
      if (node === "writing") ms = Math.max(0, ms - (validatorMs - (validatorSubs.at(-1)?.ms ?? 0)));
      if (node === "validator") ms = validatorMs;
    } else if (node === "safety_gate" && esc) {
      ms = Math.max(0, esc.at - prevEnd);
    } else if (status === "running") {
      ms = Math.max(0, now - prevEnd);
    }
    if (end != null && node !== "replan_check") prevEnd = Math.max(prevEnd, node === "validator" ? (end ?? prevEnd) : end);
    if (node === "safety_gate" && esc) prevEnd = esc.at;

    return { node, label: node === "writing" && simple ? "답변 작성" : LABELS[node], status, ms, tone, lines: lines.filter(Boolean), devLines: devLines.filter(Boolean), subs: stepSubs };
  });
}

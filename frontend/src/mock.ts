/**
 * 목업 백엔드(`VITE_MOCK=1`) — 서버 없이 UI 를 개발·시연·E2E 하기 위한 합성 데이터와 기록된 이벤트 재생.
 *
 * 책임: REST 3종(사용자/시계열/이상/요약)과 `/chat/stream` 을 계약 스키마대로 흉내 낸다. 세 시나리오를 재생한다 —
 *   (1) 7일 점검: 재계획 + critical 이상 + 검증 재작성, (2) 수면 질문: 일부 작업 실패 후 부분 결과로 진행,
 *   (3) 일반 지식 질문: 단순 경로.
 * 경계: 운영 코드 경로에서는 api.ts/App.tsx 의 `MOCK` 분기로만 도달한다. 수치는 시연용 가상 값이며 실제 데이터가 아니다.
 * 결정성: 시계열은 사용자 ID 로 시드한 난수라 새로고침해도 같은 곡선이 나온다(스크린샷·E2E 재현성).
 */
import type { Anomaly, ApproveResponse, ChatEvent, EventsResponse, SeriesPoint, SummaryResponse, UserInfo } from "./events";
import { lowerBound, percentile, toSeries } from "./series";
import { toNaiveIso, tsToMs } from "./time";

// 중단 신호에 즉시 반응하는 sleep. 실제 스트림처럼 abort 시 AbortError 로 끝나야 중단 UX 를 검증할 수 있다.
const sleep = (ms: number, signal: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    if (signal.aborted) return reject(new DOMException("aborted", "AbortError"));
    const t = setTimeout(resolve, ms);
    signal.addEventListener("abort", () => (clearTimeout(t), reject(new DOMException("aborted", "AbortError"))), { once: true });
  });

// 서버가 리포트 끝에 코드로 삽입하는 면책 단락(report.ts 가 본문에서 분리한다).
const DISCLAIMER =
  "\n\n본 서비스는 의료기기가 아니며 진단을 대신하지 않습니다. 증상이 있거나 걱정되면 의료진과 상담하세요.";

// 서버 안전 게이트가 강제하는 고정 안내문의 모사.
const ESCALATION_MSG =
  "심각 수준의 이상이 감지되었습니다. 흉통·호흡곤란·실신 등 증상이 함께 있다면 즉시 119 또는 응급실에 연락하세요. 증상이 없더라도 가까운 시일 내 의료진 상담을 권합니다.";

const HOUR = 3_600_000;
const DAY = 24 * HOUR;

/** 목업 사용자: 계약의 `UserInfo` 에 실제 사건(증상 시작·확진·회복) 정의를 더한 것. */
interface MockUser extends UserInfo {
  /** 기록 끝에서 며칠 전에 일어난 사건인지(날짜는 목업이 계산한다). */
  illness?: { type: "symptom_onset" | "diagnosis" | "recovery"; daysBeforeEnd: number; illness: "covid" | "other" }[];
  /** 증상 시작 이 일수 전에 첫 경고(이상)를 심어 둔다. 없으면 사전 경고 없음. */
  earlyAlarmDays?: number;
}

const FITBIT_USERS: MockUser[] = [
  { user_id: "U01", dataset: "fitbit", has_sleep: true, start: "2016-03-12T00:00:00", end: "2016-05-12T09:30:00", days: 62, coverage_pct: 91.4 },
  { user_id: "U02", dataset: "fitbit", has_sleep: true, start: "2016-03-12T00:00:00", end: "2016-05-12T09:30:00", days: 62, coverage_pct: 84.0 },
  { user_id: "U03", dataset: "fitbit", has_sleep: true, start: "2016-03-29T00:00:00", end: "2016-05-12T09:30:00", days: 45, coverage_pct: 63.2 },
  { user_id: "U04", dataset: "fitbit", has_sleep: true, start: "2016-03-12T00:00:00", end: "2016-04-11T23:59:00", days: 31, coverage_pct: 96.8 },
  { user_id: "U05", dataset: "fitbit", has_sleep: true, start: "2016-04-12T00:00:00", end: "2016-05-12T09:30:00", days: 31, coverage_pct: 42.7 },
];

// Stanford COVID-19 웨어러블 연구 데이터 모사(날짜는 개인정보 보호를 위해 이동된 값). 사건 위치를 기록 끝에서 멀리 둬서
// "마지막 30일" 밖에 있게 하고, 사건 주변 구간으로 옮겨 보는 흐름을 시험할 수 있게 한다.
const STANFORD_USERS: MockUser[] = [
  { user_id: "S001", dataset: "stanford", has_sleep: true, start: "2020-01-10T00:00:00", end: "2020-04-20T00:00:00", days: 100, coverage_pct: 88.1, earlyAlarmDays: 3.3,
    illness: [{ type: "symptom_onset", daysBeforeEnd: 40, illness: "covid" }, { type: "diagnosis", daysBeforeEnd: 38, illness: "covid" }, { type: "recovery", daysBeforeEnd: 30, illness: "covid" }] },
  { user_id: "S002", dataset: "stanford", has_sleep: false, start: "2020-01-29T00:00:00", end: "2020-04-18T00:00:00", days: 80, coverage_pct: 74.5, earlyAlarmDays: 1.6,
    illness: [{ type: "symptom_onset", daysBeforeEnd: 12, illness: "other" }] },
  { user_id: "S003", dataset: "stanford", has_sleep: true, start: "2019-12-26T00:00:00", end: "2020-04-25T00:00:00", days: 120, coverage_pct: 93.0 },
  { user_id: "S004", dataset: "stanford", has_sleep: true, start: "2020-02-16T00:00:00", end: "2020-04-16T00:00:00", days: 60, coverage_pct: 81.2,
    illness: [{ type: "symptom_onset", daysBeforeEnd: 20, illness: "covid" }, { type: "diagnosis", daysBeforeEnd: 19, illness: "covid" }] },
  { user_id: "S005", dataset: "stanford", has_sleep: false, start: "2020-01-23T00:00:00", end: "2020-04-22T00:00:00", days: 90, coverage_pct: 69.4, earlyAlarmDays: 2.2,
    illness: [{ type: "symptom_onset", daysBeforeEnd: 6, illness: "covid" }, { type: "recovery", daysBeforeEnd: 1, illness: "covid" }] },
];

const ALL_USERS: MockUser[] = [...FITBIT_USERS, ...STANFORD_USERS];

/** 합성 사용자(가명 ID). 종료 시각이 서로 달라 "마지막 기록" 표시가 사용자별로 바뀐다. */
export const MOCK_USERS: UserInfo[] = ALL_USERS.map(({ illness: _i, earlyAlarmDays: _e, ...u }) => u);

/** 데이터셋별 사용자 목록(서버 `GET /users?dataset=` 모사). Stanford 는 로컬 전용이라 목업에서만 있는 것처럼 둔다. */
export const mockUsers = (dataset: "fitbit" | "stanford"): UserInfo[] => MOCK_USERS.filter((u) => u.dataset === dataset);

const userOf = (uid: string): MockUser => ALL_USERS.find((u) => u.user_id === uid) ?? ALL_USERS[0];

// FitBit 목업의 주입 이상 — 종료 시각 기준 "몇 시간 전부터 몇 분". 시계열 생성과 이상 목록이 어긋나지 않도록 한 곳에서 관리한다.
const BASE_EVENTS = [
  { type: "resting_tachycardia", sev: "critical", hoursBefore: 50, dur: 25, hr: 128, ctx: "rest", base: 62 },
  { type: "personal_deviation", sev: "warning", hoursBefore: 20, dur: 30, hr: 98, ctx: "rest", base: 64 },
  { type: "bradycardia", sev: "warning", hoursBefore: 8.5, dur: 18, hr: 38, ctx: "sleep", base: 56 },
  { type: "sensor_gap", sev: "info", hoursBefore: 5, dur: 130, hr: 0, ctx: "rest", base: 0 },
  { type: "resting_tachycardia", sev: "warning", hoursBefore: 240, dur: 15, hr: 112, ctx: "rest", base: 63 },
  { type: "sensor_gap", sev: "info", hoursBefore: 400, dur: 150, hr: 0, ctx: "rest", base: 0 },
  { type: "personal_deviation", sev: "warning", hoursBefore: 520, dur: 40, hr: 96, ctx: "rest", base: 63 },
] as const;

interface Injected {
  type: string;
  sev: "critical" | "warning" | "info";
  startMs: number;
  dur: number;
  hr: number;
  ctx: string;
  base: number;
}

// 사용자별 주입 이상(절대 시각). Stanford 는 증상 시작 며칠 전의 첫 경고와 시작 직후의 이탈을 심는다.
function injected(uid: string): Injected[] {
  const u = userOf(uid);
  const end = tsToMs(u.end);
  if (u.dataset !== "stanford") return BASE_EVENTS.map((e) => ({ ...e, startMs: end - e.hoursBefore * HOUR }));
  const out: Injected[] = [{ type: "sensor_gap", sev: "info", startMs: end - 3 * DAY, dur: 180, hr: 0, ctx: "rest", base: 0 }];
  const onset = u.illness?.find((e) => e.type === "symptom_onset");
  if (onset && u.earlyAlarmDays) {
    const t = end - onset.daysBeforeEnd * DAY;
    out.push(
      { type: "resting_tachycardia", sev: "warning", startMs: t - u.earlyAlarmDays * DAY, dur: 20, hr: 112, ctx: "rest", base: 60 },
      { type: "personal_deviation", sev: "warning", startMs: t + 0.5 * DAY, dur: 30, hr: 98, ctx: "rest", base: 60 },
    );
  }
  return out;
}

// 선형 합동 생성기(LCG). 암호학적 품질이 필요 없고, 외부 의존 없이 재현 가능한 난수만 필요하다.
const rand = (seed: number) => {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32);
};
const seedOf = (id: string) => [...id].reduce((a, c) => a * 31 + c.charCodeAt(0), 7);

/** 서버의 리샘플 규칙 모사: 7일 이하 5분, 14일 이하 15분, 그 이상 1시간. */
export function mockResample(days: number): string {
  return days <= 7 ? "5min" : days <= 14 ? "15min" : "1h";
}

/** `end`(ISO, 생략하면 사용자의 마지막 기록) 문자열을 epoch ms 로. */
const endMsOf = (uid: string, end?: string) => (end ? tsToMs(end) : tsToMs(userOf(uid).end));

/**
 * `end` 에서 거꾸로 `days` 일의 합성 시계열(수면·걸음·주입 이상 반영). 점 간격은 `mockResample` 을 따르고,
 * 사용자의 기록 기간(시작~마지막) 밖은 점이 없다. 센서 결측과 무작위 누락은 서버처럼 점을 보내지 않고 건너뛴다.
 */
export function mockSeries(uid: string, days: number, end?: string): SeriesPoint[] {
  const u = userOf(uid);
  const endMs = endMsOf(uid, end);
  const first = tsToMs(u.start);
  const last = tsToMs(u.end);
  const step = { "5min": 5, "15min": 15, "1h": 60 }[mockResample(days) as "5min" | "15min" | "1h"] * 60_000;
  const r = rand(seedOf(uid));
  const inj = injected(uid);
  const out: SeriesPoint[] = [];
  for (let t = endMs - days * DAY; t <= endMs; t += step) {
    if (t < first || t > last) continue;
    const d = new Date(t);
    const h = d.getUTCHours() + d.getUTCMinutes() / 60;
    const asleep = u.has_sleep !== false && (h >= 23.2 || h < 6.6);
    const walking = !(h >= 23.2 || h < 6.6) && r() < (h > 7 && h < 21 ? 0.28 : 0.05);
    const steps = walking ? Math.round(40 + r() * 420) : 0;
    let hr = h >= 23.2 || h < 6.6 ? 54 + 3 * Math.sin(h * 1.3) + (r() - 0.5) * 5 : 68 + 4 * Math.sin((h - 9) / 3.8) + (r() - 0.5) * 6;
    if (steps > 0) hr += 8 + steps / 28 + r() * 6;
    let val: number | null = Math.round(hr);
    let st = steps;
    for (const e of inj) {
      if (t >= e.startMs && t < e.startMs + e.dur * 60_000) ((val = e.type === "sensor_gap" ? null : Math.round(e.hr + (r() - 0.5) * 4)), (st = 0)); // 휴식 중 이상: 걸음 0
    }
    if (r() < 0.004) val = null;
    if (val === null) continue;
    out.push({ ts: toNaiveIso(t), hr: val, steps: st, asleep });
  }
  return out;
}

/** `end` 에서 거꾸로 `days` 일 안에 시작하는 주입 이상 목록(계약의 `Anomaly` 형태). ID 는 사용자 안에서 고정이다. */
export function mockAnomalies(uid: string, days: number, end?: string): Anomaly[] {
  const endMs = endMsOf(uid, end);
  return injected(uid)
    .map((e, i) => {
      const evidence: Record<string, string> = {
        resting_tachycardia: `휴식 중 평균 ${e.hr} bpm이 ${e.dur}분간 지속 (임계 100 bpm, 직전 10분 걸음 0)`,
        personal_deviation: `휴식 중 ${e.hr} bpm, 평소(${e.base} bpm)보다 크게 높은 상태가 ${e.dur}분 지속`,
        bradycardia: `수면 중 평균 ${e.hr} bpm이 ${e.dur}분간 지속 (수면 기준 35 bpm 이상 권장)`,
        sensor_gap: `기기 착용 중 심박 결측 ${e.dur}분 (2시간 이상)`,
      };
      // 서버 `stats` 모사: 판단에 쓴 통계 값(개발자 보기에서만 표시).
      const stats: Record<string, Record<string, number>> = {
        resting_tachycardia: { threshold_bpm: 100, min_duration_min: 10 },
        personal_deviation: { z: 5.1, z_threshold: 4, min_duration_min: 15 },
        bradycardia: { threshold_bpm: 35, min_duration_min: 10 },
        sensor_gap: { min_gap_min: 120 },
      };
      return {
        anomaly_id: `${uid}-A${i + 1}`,
        type: e.type,
        start: toNaiveIso(e.startMs),
        end: toNaiveIso(e.startMs + e.dur * 60_000),
        duration_min: e.dur,
        severity: e.sev,
        value: e.hr,
        baseline: e.base || null,
        context: e.ctx,
        evidence: evidence[e.type],
        stats: stats[e.type],
      } satisfies Anomaly;
    })
    .filter((a) => tsToMs(a.start) >= endMs - days * DAY && tsToMs(a.start) <= endMs);
}

/** 합성 시계열에서 계산한 구간 요약(서버 `/summary` 응답 형태). */
export function mockSummary(uid: string, days: number, end?: string): SummaryResponse {
  const endMs = endMsOf(uid, end);
  const s = toSeries(mockSeries(uid, days, end));
  const a = lowerBound(s.t, endMs - days * DAY);
  const all: number[] = [];
  const rest: number[] = [];
  const slp: number[] = [];
  const stepMin = { "5min": 5, "15min": 15, "1h": 60 }[mockResample(days) as "5min" | "15min" | "1h"];
  let active = 0;
  for (let i = a; i < s.t.length; i++) {
    const v = s.hr[i];
    if (s.steps[i] > 0) active += stepMin;
    if (v == null) continue;
    all.push(v);
    if (s.asleep[i]) slp.push(v);
    else if (s.steps[i] === 0) rest.push(v);
  }
  const so = (x: number[]) => [...x].sort((p, q) => p - q);
  // 서버는 빈 구간의 점을 생략하므로, 커버리지는 존재하는 점이 아니라 "기대 점 수" 기준으로 계산한다.
  const expected = (days * 1440) / stepMin;
  return {
    window: { start: toNaiveIso(endMs - days * DAY), end: toNaiveIso(endMs) },
    summary: {
      coverage_pct: Math.min(100, (all.length / expected) * 100),
      hr_mean: all.length ? all.reduce((x, y) => x + y, 0) / all.length : null,
      hr_p05: percentile(so(all), 0.05),
      hr_p95: percentile(so(all), 0.95),
      resting_hr: percentile(so(rest), 0.5),
      sleep_hr: percentile(so(slp), 0.5),
      active_minutes: active,
      n_minutes: all.length * stepMin,
    },
    daily_resting_hr: [],
  };
}

/** 실제 사건(증상 시작·확진·회복) 목록(서버 `GET /users/{id}/events` 모사). 사건이 없는 사용자는 빈 목록. */
export function mockEvents(uid: string): EventsResponse {
  const u = userOf(uid);
  const end = tsToMs(u.end);
  return {
    events: (u.illness ?? []).map((e) => ({
      type: e.type,
      date: toNaiveIso(end - e.daysBeforeEnd * DAY).slice(0, 10),
      illness: e.illness,
      // 서버가 미리 계산하는 첫 경고 모사: 경고를 심은 사용자는 정수 일 전, 아니면 "경고 없음".
      ...(e.type === "symptom_onset"
        ? {
            first_warning: u.earlyAlarmDays
              ? { evaluable: true, hit: true, first_alarm: toNaiveIso(end - (e.daysBeforeEnd + Math.floor(u.earlyAlarmDays)) * DAY).slice(0, 10), lead_days: Math.floor(u.earlyAlarmDays), types: ["resting_tachycardia"], pre_days: 7, post_days: 3 }
              : { evaluable: true, hit: false, first_alarm: null, lead_days: null, types: [], pre_days: 7, post_days: 3 },
          }
        : {}),
    })),
    note: u.illness?.length ? "연구용 라벨입니다. 진단이 아니며, 날짜는 개인정보 보호를 위해 이동되었습니다." : "",
  };
}

// 재생 단계: 이벤트 또는 대기(ms).
type Step = ChatEvent | number; // number = 대기(ms)

// 리포트를 `n`자 토큰으로 쪼갠다. 첫 토큰만 reset=true 로 내보내 서버의 "작성 시도 시작" 신호를 모사한다.
function chunk(text: string, n = 5): Step[] {
  const out: Step[] = [];
  for (let i = 0; i < text.length; i += n) out.push({ type: "token", text: text.slice(i, i + n), reset: i === 0 }, 22);
  return out;
}

function scenario(message: string, uid: string): Step[] {
  const simple = /범위|뭐야|무엇|란|정의|정상.*심박|알려/.test(message) && !/점검|이상|확인해|괜찮/.test(message);
  const sleepQ = /수면|밤|잠/.test(message);
  const threadDone = (report: string, esc: boolean): ChatEvent => {
    LAST_REPORT.set(`mock-${uid}`, report); // 승인 뒤 "결과 한 줄이 덧붙은 최종 리포트" 를 돌려주려고 마지막 리포트를 기억한다
    return { type: "done", thread_id: `mock-${uid}`, report, escalation: esc, usage: MOCK_USAGE };
  };

  if (simple) {
    const report =
      "## 안정시 심박 정상 범위\n\n- 성인의 안정시 심박은 보통 **분당 60~100회**입니다.\n- 규칙적으로 운동하는 사람은 40~60회까지 낮을 수 있습니다.\n- 수면 중에는 안정시보다 5~10회 정도 더 낮아지는 것이 흔합니다." +
      DISCLAIMER;
    return [
      { type: "route_decided", route: "simple", reasoning: "일반 지식 질문이라 데이터 조회 없이 가이드라인으로 답변 가능" },
      { type: "task_started", task_id: "retrieval", task_type: "retrieval" },
      350,
      { type: "task_completed", task_id: "retrieval", task_type: "retrieval", summary: "근거 3건 (embedding)" },
      400,
      ...chunk(report),
      { type: "validation", passed: true, issues: [], attempt: 1 },
      threadDone(report, false),
    ];
  }

  if (sleepQ) {
    const report =
      "## 어젯밤 수면 중 심박 점검\n\n측정 구간 최근 1일, 수면 심박 평균은 **54 bpm**이었습니다.\n\n### 확인된 점\n- 수면 중 약 18분간 평균 **38 bpm**이 관찰되었습니다 (수면 기준 35 bpm 이상 권장, 경계 수준).\n- 그 외 구간은 개인 기준선 범위 안이었습니다.\n\n### 권장\n- 같은 패턴이 반복되는지 며칠 더 지켜보세요.\n- 어지럼증이 동반되면 의료진과 상담하세요." +
      DISCLAIMER;
    return [
      { type: "route_decided", route: "analysis", reasoning: "수면 중 심박 점검 요청 — 데이터 조회와 분석이 필요" },
      500,
      { type: "plan_created", needs_analysis: true, lookback_days: 1, tasks: [
        { id: "t1", type: "research", description: "최근 1일 심박 요약 조회 (get_vitals_summary)" },
        { id: "t2", type: "research", description: "수면 구간 이상 탐지 (detect_anomalies)" },
        { id: "t3", type: "analysis", description: "탐지 결과를 수면 맥락에서 해석" },
        { id: "t4", type: "writing", description: "한국어 점검 리포트 작성" },
      ] },
      400,
      { type: "gate_applied", removed_tasks: [], forced_analysis: false, reason: "계획이 적절함 — 변경 없음" },
      { type: "task_started", task_id: "t1", task_type: "research" },
      { type: "task_started", task_id: "t2", task_type: "research" },
      700,
      { type: "task_failed", task_id: "t1", task_type: "research", error: "get_vitals_summary 타임아웃 (재시도 2회 소진) — 부분 결과로 계속" },
      400,
      { type: "task_completed", task_id: "t2", task_type: "research", summary: "이상 1건 탐지 (bradycardia 1)" },
      { type: "anomalies_found", count: 1, counts_by_type: { bradycardia: 1 }, anomalies: mockAnomalies(uid, 1).filter((a) => a.type === "bradycardia") },
      { type: "task_started", task_id: "t3", task_type: "analysis" },
      900,
      { type: "task_completed", task_id: "t3", task_type: "analysis", summary: "수면 후반부 일시적 저심박 1건, 위험 신호 없음" },
      { type: "task_started", task_id: "t4", task_type: "writing" },
      ...chunk(report),
      { type: "task_completed", task_id: "t4", task_type: "writing", summary: "리포트 작성 완료" },
      { type: "validation", passed: true, issues: [], attempt: 1 },
      threadDone(report, false),
    ];
  }

  const anomalies = mockAnomalies(uid, 7);
  const counts = anomalies.reduce<Record<string, number>>((m, a) => ((m[a.type] = (m[a.type] ?? 0) + 1), m), {});
  const bad = "## 최근 7일 심박 점검\n\n개인 기준선 대비 이상이 관찰되었습니다.";
  const report =
    `**중요 안내** ${ESCALATION_MSG}\n\n` +
    "## 최근 7일 심박 점검\n\n최근 7일간 안정시 심박은 평균 **63 bpm**이었고, 이상 **4건**이 탐지되었습니다.\n\n### 주요 이상\n- **안정 시 빈맥 (심각)**: 약 2일 전 휴식 중 평균 **128 bpm**이 25분간 지속되었습니다. 운동이 아닌 휴식 상태였습니다.\n- **개인 기준선 이탈 (주의)**: 휴식 중 98 bpm이 30분 지속 (기준선 64 bpm).\n- **수면 중 서맥 (주의)**: 평균 38 bpm이 18분.\n- **센서 결측 (참고)**: 착용 중 130분간 심박 미측정.\n\n### 권장\n1. 빈맥 시점의 상황(카페인, 스트레스, 증상)을 되짚어 보세요.\n2. 증상이 있거나 반복되면 의료진과 상담하세요." +
    DISCLAIMER;
  return [
    { type: "route_decided", route: "analysis", reasoning: "기간 내 이상 점검 요청 — 다단계 분석 필요" },
    { type: "task_started", task_id: "mcp_prompt", task_type: "prompt" },
    300,
    { type: "task_completed", task_id: "mcp_prompt", task_type: "prompt", summary: "MCP Prompt anomaly_review 적용 (412자)" },
    200,
    { type: "plan_created", needs_analysis: false, lookback_days: 7, tasks: [
      { id: "t1", type: "research", description: "최근 7일 심박 요약 조회" },
      { id: "t2", type: "research", description: "최근 7일 이상 탐지" },
      { id: "t3", type: "analysis", description: "탐지 결과 해석" },
      { id: "t4", type: "writing", description: "점검 리포트 작성" },
    ] },
    400,
    { type: "gate_applied", removed_tasks: [], forced_analysis: true, reason: "LLM이 분석 불필요로 판단했으나 질문이 '이상 점검' 의도라 분석을 강제 유지" },
    { type: "task_started", task_id: "t1", task_type: "research" },
    { type: "task_started", task_id: "t2", task_type: "research" },
    800,
    { type: "task_completed", task_id: "t1", task_type: "research", summary: "커버리지 41.2%, 휴식 분 38 — 데이터 부족" },
    { type: "task_completed", task_id: "t2", task_type: "research", summary: "이상 2건 탐지" },
    500,
    { type: "replan", round: 1, reason: "커버리지 41.2% < 50% — 조회 기간을 2배로 확장해 재조회", lookback_days: 14 },
    { type: "task_started", task_id: "t1", task_type: "research" },
    { type: "task_started", task_id: "t2", task_type: "research" },
    800,
    { type: "task_completed", task_id: "t1", task_type: "research", summary: "커버리지 78.5%, 휴식 분 1,204" },
    { type: "task_completed", task_id: "t2", task_type: "research", summary: `이상 ${anomalies.length}건 탐지`, retries: 1 },
    { type: "anomalies_found", count: anomalies.length, counts_by_type: counts, anomalies },
    { type: "task_started", task_id: "t3", task_type: "analysis" },
    900,
    { type: "task_completed", task_id: "t3", task_type: "analysis", summary: "빈맥 1건은 휴식 상태에서 발생 — 운동 오탐 아님" },
    { type: "safety_escalation", reason: "critical 이상 1건 (resting_tachycardia 128 bpm)", message: ESCALATION_MSG },
    { type: "task_started", task_id: "t4", task_type: "writing" },
    ...chunk(bad.slice(0, 20)),
    { type: "validation", passed: false, issues: ["escalation 안내문 누락", "면책 문구 누락"], attempt: 1 },
    500,
    ...chunk(report),
    { type: "task_completed", task_id: "t4", task_type: "writing", summary: "리포트 작성 완료 (재작성 1회)" },
    { type: "validation", passed: true, issues: [], attempt: 2 },
    // 에스컬레이션 뒤 보호자 알림 승인 요청(HITL). 안내문은 승인과 무관하게 이미 리포트에 들어 있다.
    { type: "approval_required", thread_id: `mock-${uid}`, approval_id: pendingMockApproval(`mock-${uid}`), reason: "심각한 이상이 있어 보호자에게 알릴지 확인이 필요합니다.", message_preview: `[Vital Sentinel] ${uid} 님의 심박에서 심각 수준의 이상(휴식 중 128 bpm, 25분)이 감지되었습니다. 상태를 확인해 주세요. (진단이 아닌 참고 알림)` },
    threadDone(report, true),
  ];
}

// 마지막 리포트(thread 별)와 이미 결정된 승인 요청. 서버의 "한 번 결정하면 409" 동작을 모사한다.
const LAST_REPORT = new Map<string, string>();
const DECIDED = new Set<string>(); // 결정된 approval_id
const LATEST = new Map<string, string>(); // thread 별 현재 대기 중인 approval_id
let APPROVAL_SEQ = 0;
export const pendingMockApproval = (thread: string) => {
  const id = `ap-${++APPROVAL_SEQ}`;
  LATEST.set(thread, id);
  return id;
};
const MOCK_USAGE = { llm_calls: 4, input_tokens: 5120, output_tokens: 980, total_tokens: 6100, retries: 1, cost_usd: 0 };

/** 보호자 알림 승인/거절(서버 `POST /chat/approve` 모사). 첫 결정은 결과 한 줄이 덧붙은 리포트를, 이미 결정됐거나 새 요청으로 바뀐 approval_id 는 409 를 돌려준다. */
export async function mockApprove(threadId: string, approved: boolean, approvalId: string): Promise<{ ok: true; data: ApproveResponse } | { ok: false; conflict: boolean; status: number }> {
  await new Promise((r) => setTimeout(r, 600));
  if (DECIDED.has(approvalId) || LATEST.get(threadId) !== approvalId) return { ok: false, conflict: true, status: 409 };
  DECIDED.add(approvalId);
  const line = approved ? "보호자 알림: 전송했습니다(시뮬레이션 — 실제로는 발송되지 않습니다)." : "보호자 알림: 보내지 않았습니다.";
  return { ok: true, data: { thread_id: threadId, approved, notification: approved ? "sent_simulated" : "not_sent", report: `${LAST_REPORT.get(threadId) ?? ""}\n\n${line}` } };
}

/**
 * 질문에 맞는 시나리오를 실제 스트림처럼 시간차를 두고 내보낸다.
 * @param signal abort 되면 AbortError 로 끝난다.
 */
export async function* mockStream(message: string, userId: string | null, signal: AbortSignal): AsyncGenerator<ChatEvent> {
  for (const s of scenario(message, userId ?? "U01")) {
    if (typeof s === "number") await sleep(s, signal);
    else {
      await sleep(40, signal);
      yield s;
    }
  }
}

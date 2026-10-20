/**
 * 백엔드 계약 스키마 (zod) — SSE 이벤트와 REST 응답의 단일 정의.
 *
 * 책임: 네트워크 경계에서 들어오는 JSON 을 런타임 검증하고 TypeScript 타입을 파생한다.
 * 경계: 전송(sse.ts)·상태 전이(turn.ts)·표시(컴포넌트)는 이 모듈의 타입만 의존한다.
 * 정합성: 백엔드 `app/agent/events.py` 와 필드명을 1:1 로 맞춘다.
 *   필드를 추가/변경할 때는 세 곳(CONTRACT, events.py, 이 파일)을 함께 고친다.
 * 설계: Custom Transport + zod 런타임 검증.
 */
import { z } from "zod";

/** 이상 심각도. critical 은 안전 게이트가 고정 안내문을 강제하는 수준이다. */
export const Severity = z.enum(["info", "warning", "critical"]);
export type Severity = z.infer<typeof Severity>;

/**
 * 탐지된 이상 1건. 시각(`start`, `end`)은 기기 로컬 시각의 naive ISO 문자열(타임존 없음)이며
 * 해석은 `time.ts` 의 `tsToMs` 로만 한다.
 */
export const Anomaly = z.object({
  anomaly_id: z.string(),
  /** resting_tachycardia | bradycardia | personal_deviation | elevated_resting_trend | sensor_gap */
  type: z.string(),
  start: z.string(),
  end: z.string(),
  /** 지속 시간(분). */
  duration_min: z.number(),
  severity: Severity,
  /** 대표값(bpm). sensor_gap 에서는 의미가 없다. */
  value: z.number(),
  /** 개인 기준선(bpm). 기준선을 산출하지 못한 경우 null/생략. */
  baseline: z.number().nullable().optional(),
  /** rest | sleep | active */
  context: z.string(),
  /** 사람이 읽는 한 줄 근거(쉬운 한국어). */
  evidence: z.string(),
  /** 판단에 쓴 통계 값(z 점수, 임계값 등). 키는 서버가 정하며 화면에서는 "개발자 보기"에서만 보인다. */
  stats: z.record(z.string(), z.unknown()).optional(),
});
export type Anomaly = z.infer<typeof Anomaly>;

/** LLM 사용량 요약(호출 수·토큰·재시도·참고 비용). 필드는 모두 선택이며 모르는 키는 그대로 둔다. */
export const Usage = z
  .object({
    llm_calls: z.number().optional(),
    input_tokens: z.number().optional(),
    output_tokens: z.number().optional(),
    total_tokens: z.number().optional(),
    retries: z.number().optional(),
    cost_usd: z.number().optional(),
    llm_seconds: z.number().optional(),
  })
  .loose();
export type Usage = z.infer<typeof Usage>;

/** 계획 단계가 만든 작업 단위. */
export const PlanTask = z.object({
  id: z.string(),
  type: z.enum(["research", "analysis", "writing"]),
  description: z.string(),
});
export type PlanTask = z.infer<typeof PlanTask>;

// 진행 이벤트의 작업 종류. 계획이 세운 3종(research/analysis/writing)에 더해, 서버가 기존 진행 이벤트를 재사용해 알리는
// retrieval(가이드라인 검색)·prompt(점검 절차 불러오기)가 온다. 모르는 값은 거부한다(스키마와 서버를 함께 고친다).
const taskType = z.enum(["research", "analysis", "writing", "retrieval", "prompt"]);
export type TaskKind = z.infer<typeof taskType>;

/**
 * `/chat/stream` 이 내보내는 이벤트의 판별 합집합(`type` 으로 판별).
 * 그래프 진행(route~done)과 리포트 토큰(`token`)을 한 채널로 받는다.
 */
export const ChatEvent = z.discriminatedUnion("type", [
  z.object({ type: z.literal("route_decided"), route: z.enum(["simple", "analysis"]), reasoning: z.string() }),
  z.object({
    type: z.literal("plan_created"),
    tasks: z.array(PlanTask),
    needs_analysis: z.boolean(),
    lookback_days: z.number(),
  }),
  z.object({
    type: z.literal("gate_applied"),
    removed_tasks: z.array(z.string()),
    forced_analysis: z.boolean(),
    reason: z.string(),
    /** 게이트가 확정한 실제 조회 기간(일). 후속 질문에서 이전 기간을 이어받으면 계획의 값과 다를 수 있다. 없으면 계획의 값을 쓴다. */
    lookback_days: z.number().optional(),
  }),
  z.object({ type: z.literal("task_started"), task_id: z.string(), task_type: taskType }),
  z.object({
    type: z.literal("task_completed"),
    task_id: z.string(),
    task_type: taskType,
    summary: z.string(),
    /** 이 작업이 성공하기까지 일어난 일시 오류 재시도 횟수(없으면 0). */
    retries: z.number().optional(),
  }),
  z.object({ type: z.literal("task_failed"), task_id: z.string(), task_type: taskType, error: z.string() }),
  z.object({ type: z.literal("replan"), round: z.number(), reason: z.string(), lookback_days: z.number() }),
  z.object({
    type: z.literal("anomalies_found"),
    count: z.number(),
    counts_by_type: z.record(z.string(), z.number()),
    anomalies: z.array(Anomaly),
  }),
  z.object({ type: z.literal("safety_escalation"), reason: z.string(), message: z.string() }),
  z.object({
    type: z.literal("token"),
    text: z.string(),
    /** true 이면 초안을 교체한다(작성 시도의 첫 토큰, 재작성, 템플릿 폴백). 생략은 false 로 본다. */
    reset: z.boolean().optional(),
  }),
  z.object({
    type: z.literal("validation"),
    passed: z.boolean(),
    issues: z.array(z.string()),
    attempt: z.number(),
    /** true 이면 검증 소진 후 템플릿 폴백이 사용되었다. */
    degraded: z.boolean().optional(),
  }),
  z.object({
    type: z.literal("approval_required"),
    thread_id: z.string(),
    reason: z.string(),
    /** 보호자에게 갈 알림 문구 미리보기. */
    message_preview: z.string(),
    /** 이 승인 요청의 ID. 승인/거절을 보낼 때 되돌려 보낸다(오래된 카드가 새 요청을 승인하지 못하게 한다). */
    approval_id: z.string(),
  }),
  z.object({
    type: z.literal("done"),
    thread_id: z.string(),
    report: z.string(),
    escalation: z.boolean(),
    run_id: z.string().nullable().optional(),
    /** 이번 요청의 LLM 사용량 요약. 계량 범위 밖이면 없음. 키는 서버가 정한다(개발자 보기에서만 보인다). */
    usage: Usage.nullable().optional(),
  }),
  z.object({ type: z.literal("error"), message: z.string() }),
]);
export type ChatEvent = z.infer<typeof ChatEvent>;
/** `POST /chat/approve` 응답: 보호자 알림 승인/거절 결과와, 결과 한 줄이 덧붙은 최종 리포트. */
export const ApproveResponse = z
  .object({
    thread_id: z.string(),
    approved: z.boolean(),
    notification: z.enum(["sent_simulated", "not_sent"]),
    report: z.string(),
  })
  .loose();
export type ApproveResponse = z.infer<typeof ApproveResponse>;

/** `ChatEvent` 중 특정 `type` 하나로 좁힌 타입. */
export type EventOf<T extends ChatEvent["type"]> = Extract<ChatEvent, { type: T }>;

/** `GET /users` 항목. 사용자 ID 는 가명(U01…)이다. */
export const UserInfo = z.object({
  user_id: z.string(),
  start: z.string(),
  end: z.string(),
  days: z.number(),
  /** 전체 기간 대비 심박 데이터 존재 비율(%). */
  coverage_pct: z.number(),
  /** 데이터셋: fitbit(개발용 공개 데이터) | stanford(외부 검증, 로컬 전용). 구버전 서버는 생략 → fitbit 로 본다. */
  dataset: z.enum(["fitbit", "stanford"]).optional(),
  /** 수면 기록이 있는 참가자인지. 생략하면 있다고 본다. */
  has_sleep: z.boolean().optional(),
});
export type UserInfo = z.infer<typeof UserInfo>;

/** 차트용 다운샘플 시계열의 한 점. `hr` 가 null 이면 해당 구간 결측. */
export const SeriesPoint = z.object({
  ts: z.string(),
  hr: z.number().nullable(),
  steps: z.number().nullable().optional(),
  asleep: z.boolean().nullable().optional(),
});
export type SeriesPoint = z.infer<typeof SeriesPoint>;

// 응답 최상위는 loose(): 서버가 부가 필드(user_id, window 등)를 추가해도 화면이 깨지지 않게 한다.
// resample: 서버가 실제로 적용한 리샘플 주기("5min", "15min", "1h"). 기록 없음(간격) 판정의 기준이다. 없으면 점 간격으로 추정한다.
export const SeriesResponse = z.object({ points: z.array(SeriesPoint), resample: z.string().optional() }).loose();
export const AnomaliesResponse = z.object({ anomalies: z.array(Anomaly) }).loose();

// 요약 지표는 데이터가 부족한 구간에서 null 이 될 수 있어 모두 nullable.
/**
 * `GET /users/{id}/events` — 연구자가 붙인 실제 사건 라벨(증상 시작·확진·회복). 진단이 아니며 날짜는 개인정보 보호를 위해
 * 이동된 값일 수 있다. `date` 는 하루 단위(YYYY-MM-DD)다.
 */
/**
 * 증상 시작 사건에 서버가 미리 계산해 붙이는 "첫 경고". 보는 구간과 무관한 값이다.
 * `lead_days` 는 경고가 증상보다 먼저면 양수, 같은 날이면 0, 증상 뒤면 음수. 계산 파일이 없으면 이 필드 자체가 없다.
 */
export const FirstWarning = z
  .object({
    evaluable: z.boolean(),
    hit: z.boolean(),
    first_alarm: z.string().nullable().optional(),
    lead_days: z.number().nullable().optional(),
    types: z.array(z.string()).optional(),
    pre_days: z.number().optional(),
    post_days: z.number().optional(),
  })
  .loose();
export type FirstWarning = z.infer<typeof FirstWarning>;

export const EventsResponse = z
  .object({
    events: z.array(
      z.object({
        type: z.enum(["symptom_onset", "diagnosis", "recovery"]),
        date: z.string(),
        illness: z.enum(["covid", "other"]),
        first_warning: FirstWarning.optional(),
      }),
    ),
    note: z.string().optional(),
  })
  .loose();
export type EventsResponse = z.infer<typeof EventsResponse>;

const num = z.number().nullable().optional();

/** `GET /users/{id}/summary` 응답. 상단 판독값과 개인 기준선 밴드의 출처. */
export const SummaryResponse = z
  .object({
    window: z.object({ start: z.string(), end: z.string() }).optional(),
    summary: z.object({
      coverage_pct: num,
      hr_mean: num,
      hr_p05: num,
      hr_p95: num,
      resting_hr: num,
      sleep_hr: num,
      active_minutes: num,
      n_minutes: num,
    }),
    daily_resting_hr: z.array(z.object({ date: z.string(), resting_hr: num, n_rest_minutes: num })).optional(),
  })
  .loose();
export type SummaryResponse = z.infer<typeof SummaryResponse>;

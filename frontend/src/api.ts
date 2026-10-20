/**
 * REST 클라이언트(차트·사용자·요약). 응답은 모두 zod 로 검증하고, `VITE_MOCK=1` 이면 mock.ts 의 합성 데이터를 돌려준다.
 * 채팅(SSE)은 sse.ts 의 Custom Transport 가 맡는다 — 이 모듈은 단발성 GET 만 다룬다.
 */
import { AnomaliesResponse, ApproveResponse, EventsResponse, SeriesResponse, SummaryResponse, UserInfo, type Anomaly, type SeriesPoint } from "./events";
import { mockAnomalies, mockApprove, mockEvents, mockResample, mockSeries, mockSummary, mockUsers } from "./mock";
import type { HealthInfo, VersionInfo } from "./llm";
import { API_BASE } from "./sse";
import { z } from "zod";

/** 목업 모드 여부(빌드 시점 환경변수 `VITE_MOCK=1`). */
export const MOCK = import.meta.env.VITE_MOCK === "1";
/**
 * 이상 목록을 한 번에 받는 조회 기간(일) — 서버 `lookback_days` 상한(30). 이상은 해상도와 무관해 30일치를 한 번 받아
 * 1/7/30일 전환을 클라이언트에서 처리한다. (시계열은 해상도가 기간마다 달라 기간별로 받는다.)
 */
export const LOAD_DAYS = 30;

// `end`(naive ISO) 쿼리. 값이 없으면 서버 기본(그 사용자의 마지막 기록)을 쓴다.
const endQuery = (end?: string) => (end ? `&end=${encodeURIComponent(end)}` : "");

// 비-2xx 는 본문을 파싱하지 않고 경로와 상태 코드만 담아 던진다(화면의 오류 상태에 그대로 노출).
async function get<T>(path: string, schema: z.ZodType<T>): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`HTTP ${res.status} (${path})`);
  return schema.parse(await res.json());
}

/** 사용자 목록(가명 ID, 데이터 기간, 커버리지). @throws HTTP 오류 또는 스키마 불일치. */
export const fetchUsers = async (dataset: "fitbit" | "stanford" = "fitbit"): Promise<UserInfo[]> => {
  if (MOCK) return mockUsers(dataset);
  const all = await get(`/users?dataset=${dataset}`, z.array(UserInfo));
  // 쿼리를 모르는 구버전 서버가 전체 목록을 돌려줘도 두 데이터셋이 섞이지 않게 거른다(dataset 필드가 없으면 fitbit).
  return all.filter((u) => (u.dataset ?? "fitbit") === dataset);
};

/**
 * 외부 검증용 Stanford 참가자 목록. 이 데이터는 로컬에만 있어서(내려받은 경우에만) 없거나 서버가 모르는 값이면 빈 목록을 돌려준다.
 * 그 경우 화면은 데이터셋 선택을 숨긴다.
 */
export const fetchStanfordUsers = async (): Promise<UserInfo[]> => {
  try {
    return await fetchUsers("stanford");
  } catch {
    return [];
  }
};

/**
 * 실제 사건 라벨(증상 시작·확진·회복). 사건이 없는 사용자·구버전 서버(엔드포인트 없음)·오류는 모두 "사건 없음" 으로 돌려
 * 사건 표시만 생략한다.
 */
export const fetchEvents = async (uid: string): Promise<EventsResponse> => {
  if (MOCK) return mockEvents(uid);
  try {
    return await get(`/users/${uid}/events`, EventsResponse);
  } catch {
    return { events: [] };
  }
};

/**
 * 차트용 다운샘플 시계열.
 * @param days 조회 기간(일). 서버가 기간에 맞는 해상도(≤7일 5분, ≤14일 15분, 그 이상 1시간)를 고르고 응답의 `resample` 로 알려준다.
 *   그래서 1일/7일/30일 뷰를 각각 따로 받는다(30일 한 번으로 모두 그리면 1일·7일 뷰도 1시간 해상도가 된다).
 */
export const fetchSeries = async (uid: string, days: number, end?: string): Promise<{ points: SeriesPoint[]; resample?: string }> =>
  MOCK
    ? { points: mockSeries(uid, days, end), resample: mockResample(days) }
    : (({ points, resample }) => ({ points, resample }))(await get(`/users/${uid}/series?lookback_days=${days}${endQuery(end)}`, SeriesResponse));

/** 구간 내 탐지된 이상 목록. @param days 조회 기간(일). 기본 `LOAD_DAYS`. @param end 구간의 끝(naive ISO). 생략하면 그 사용자의 마지막 기록. */
export const fetchAnomalies = async (uid: string, days = LOAD_DAYS, end?: string): Promise<Anomaly[]> =>
  MOCK ? mockAnomalies(uid, days, end) : (await get(`/users/${uid}/anomalies?lookback_days=${days}${endQuery(end)}`, AnomaliesResponse)).anomalies;

/** 백엔드 상태. 네트워크 오류는 예외 대신 null 로 돌려 헤더의 "오프라인" 표시에 쓴다. */
export const fetchHealth = async (): Promise<HealthInfo | null> => {
  // 목업: `?llm=off` 로 "LLM 미연결" 상태를 재현할 수 있다.
  if (MOCK) {
    const off = new URLSearchParams(location.search).get("llm") === "off";
    return { status: "mock", mcp: true, llm: !off, ollama: { enabled: true, reachable: !off, models: [], missing: [] } };
  }
  try {
    return await get(
      "/health",
      z.object({
        status: z.string(),
        mcp: z.boolean(),
        llm: z.boolean(),
        ollama: z.object({ enabled: z.boolean().optional(), reachable: z.boolean().optional(), models: z.array(z.string()).optional(), missing: z.array(z.string()).optional(), backend: z.string().nullable().optional() }).loose().optional(),
      }),
    );
  } catch {
    return null;
  }
};

/**
 * 백엔드 버전·모델 정보(`GET /version`). 엔드포인트가 없는 서버(구버전)나 오류는 null 로 돌려 헤더 표시만 생략한다.
 * 목업 모드는 가상 값을 돌려준다.
 */
export const fetchVersion = async (): Promise<VersionInfo | null> => {
  if (MOCK) return { supervisor_version: new URLSearchParams(location.search).get("v") === "1" ? "v1" : "v2", llm_backend: "mock", fast_model: "qwen3:0.6b", quality_model: "qwen3:8b", checkpointer: "memory", git_commit: "mock" };
  try {
    return await get(
      "/version",
      z.object({ supervisor_version: z.string().nullable().optional(), llm_backend: z.string().nullable().optional(), fast_model: z.string().nullable().optional(), quality_model: z.string().nullable().optional(), checkpointer: z.string().nullable().optional(), git_commit: z.string().nullable().optional() }).loose(),
    );
  } catch {
    return null;
  }
};

/** 보호자 알림 승인/거절 결과. 이미 처리된 요청(409)은 오류가 아니라 `conflict` 로 돌려준다. */
export type ApproveResult = { ok: true; data: ApproveResponse } | { ok: false; conflict: boolean; status: number };

/**
 * 보호자 알림(시뮬레이션) 승인/거절을 보낸다(`POST /chat/approve`, 카드가 가진 `approvalId` 를 항상 같이 보낸다). 응답의 `report` 가 결과 한 줄이 덧붙은 최종 리포트다.
 * @returns 성공이면 데이터, 409(대기 중인 요청 없음·새 요청으로 바뀜: 이미 처리됨·만료)는 `conflict`, 그 밖의 오류는 `status` 와 함께 실패로 돌려준다.
 *   네트워크 오류는 던진다.
 */
export async function approveChat(threadId: string, approved: boolean, approvalId: string): Promise<ApproveResult> {
  if (MOCK) return mockApprove(threadId, approved, approvalId);
  const res = await fetch(`${API_BASE}/chat/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ thread_id: threadId, approved, approval_id: approvalId }),
  });
  if (res.status === 409) return { ok: false, conflict: true, status: 409 };
  if (!res.ok) return { ok: false, conflict: false, status: res.status };
  return { ok: true, data: ApproveResponse.parse(await res.json()) };
}

/** 구간 요약(상단 판독값·개인 기준선 밴드의 출처). @param days 1·7·30 중 하나. @param end 구간의 끝(생략 시 마지막 기록). */
export const fetchSummary = async (uid: string, days: number, end?: string): Promise<SummaryResponse> =>
  MOCK ? mockSummary(uid, days, end) : get(`/users/${uid}/summary?lookback_days=${days}${endQuery(end)}`, SummaryResponse);

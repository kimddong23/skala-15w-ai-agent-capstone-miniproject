/**
 * LLM 연결 상태 해석 — 백엔드 `/health` 의 `llm`/`ollama` 필드를 화면 문구로 바꾼다.
 *
 * 책임: "LLM 이 안 붙어 있어도 서비스는 동작한다(기본 리포트 모드)"는 사실을 첫 화면에서 분명히 하기 위한 순수 함수.
 * 경계: 네트워크·React 를 모른다. 백엔드가 /health 의 `llm` 을 "활성화 + 도달 + 필요한 모델 설치"의 결과로 계산하므로
 *   여기서는 그 값을 신뢰하고, 사람이 읽을 원인만 `ollama` 세부에서 골라낸다.
 */

import type { Usage } from "./events";

/** `/health` 응답 중 이 모듈이 읽는 부분. */
export interface HealthInfo {
  status: string;
  mcp: boolean;
  /** 실제로 LLM 을 쓸 수 있는가. false 여도 결정론 폴백으로 동작한다. */
  llm: boolean;
  ollama?: { enabled?: boolean; reachable?: boolean; models?: string[]; missing?: string[]; backend?: string | null };
}

/** 화면에 보일 LLM 상태. */
export interface LlmState {
  connected: boolean;
  /** 연결되지 않은 이유(쉬운 말). 연결되었으면 없음. */
  reason?: string;
}

/**
 * @param h `/health` 응답. 없으면(서버 오프라인) null 을 돌려 LLM 상태를 표시하지 않는다.
 * @returns 연결 여부와, 끊긴 경우의 원인 한 줄. 원인은 꺼짐 → 서버 미도달 → 모델 미설치 순으로 가장 앞선 것을 고른다.
 */
export function describeLlm(h: HealthInfo | null): LlmState | null {
  if (!h) return null;
  if (h.llm) return { connected: true };
  const o = h.ollama;
  if (o?.enabled === false) return { connected: false, reason: "LLM 사용이 꺼져 있습니다." };
  if (o?.reachable === false) return { connected: false, reason: "Ollama 서버에 연결되지 않았습니다." };
  if (o?.missing?.length) return { connected: false, reason: `필요한 모델이 설치되어 있지 않습니다: ${o.missing.join(", ")}` };
  return { connected: false };
}

/** `GET /version` 응답. 어떤 필드든 서버 버전에 따라 없을 수 있다. */
export interface VersionInfo {
  supervisor_version?: string | null;
  llm_backend?: string | null;
  fast_model?: string | null;
  quality_model?: string | null;
  checkpointer?: string | null;
  git_commit?: string | null;
}

/**
 * 헤더에 보일 짧은 버전 문구. 어떤 그래프·백엔드·모델로 답하고 있는지를 한 줄로 알린다.
 *
 * @param dev 개발자 보기. 켜면 체크포인터와 git 커밋(내부 식별자)도 덧붙인다 — 기본 화면에는 두지 않는다.
 * @returns `short` 한 줄 문구, `title` 마우스를 올리면 보이는 라벨 붙은 전체 설명. 보여줄 값이 없으면 null.
 * @remarks fake 백엔드는 LLM 을 쓰지 않으므로 모델 이름을 생략한다.
 */
export function formatVersion(v: VersionInfo | null, dev = false): { short: string; title: string } | null {
  if (!v) return null;
  // 서버가 "값 없음"을 "-" 나 "none" 으로 보내는 경우를 걸러, 의미 없는 기호가 헤더에 나오지 않게 한다.
  const real = (x?: string | null) => (x && x !== "-" ? x : null);
  const noLlm = v.llm_backend === "fake" || v.llm_backend === "none";
  const backend = v.llm_backend === "none" ? "LLM 없음" : real(v.llm_backend);
  const fastM = real(v.fast_model);
  const qualM = real(v.quality_model);
  const fake = noLlm;
  const models = !fake && fastM && qualM ? `${fastM} / ${qualM}` : null;
  const parts = [real(v.supervisor_version), backend, models, dev ? real(v.checkpointer) : null, dev ? real(v.git_commit) : null].filter((x): x is string => !!x);
  if (!parts.length) return null;
  const labelled = [
    real(v.supervisor_version) && `그래프 ${v.supervisor_version}`,
    backend && `LLM ${backend}`,
    !fake && fastM && `빠른 모델 ${fastM}`,
    !fake && qualM && `작성 모델 ${qualM}`,
    real(v.checkpointer) && `대화 저장 ${v.checkpointer}`,
    real(v.git_commit) && `커밋 ${v.git_commit}`,
  ].filter(Boolean);
  return { short: parts.join(" · "), title: labelled.join(", ") };
}

/**
 * 요청 하나의 LLM 사용량을 한 줄로(개발자 보기 전용). 값이 없는 항목은 생략하고, 하나도 없으면 null.
 * 비용은 서버가 "참고 단가" 로 추정한 값이라 `(참고)` 를 붙인다. 로컬 모델은 0 이라 표시하지 않는다.
 */
export function formatUsage(u: Usage | null | undefined): string | null {
  if (!u) return null;
  const n = (x: number) => x.toLocaleString("ko-KR");
  const parts = [
    u.llm_calls != null ? `LLM 호출 ${n(u.llm_calls)}회` : null,
    u.input_tokens != null || u.output_tokens != null ? `토큰 입력 ${n(u.input_tokens ?? 0)} / 출력 ${n(u.output_tokens ?? 0)}` : u.total_tokens != null ? `토큰 ${n(u.total_tokens)}` : null,
    u.retries ? `재시도 ${n(u.retries)}회` : null,
    u.cost_usd ? `비용 $${u.cost_usd.toFixed(4)} (참고)` : null,
  ].filter((x): x is string => !!x);
  return parts.length ? parts.join(", ") : null;
}

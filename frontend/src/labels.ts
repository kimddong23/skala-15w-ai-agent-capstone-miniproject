/**
 * 화면에 노출하는 한글 라벨 사전. 백엔드 식별자(type, severity, context)를 사람이 읽는 말로 바꾼다.
 * 새 이상 유형이 생기면 `TYPE_LABEL` 과 `cite.ts` 의 별칭 표를 함께 갱신한다. 사전에 없는 값은 원문을 그대로 쓴다.
 */

/** 이상 유형 → 한글 이름. */
export const TYPE_LABEL: Record<string, string> = {
  // 휴식·수면 모두에서 탐지되는 유형이라 "휴식 중" 이 아니라 "안정 시"로 쓴다. 발생 맥락은 CTX_LABEL 이 따로 보여준다.
  resting_tachycardia: "안정 시 빈맥",
  bradycardia: "서맥",
  personal_deviation: "개인 기준선 이탈",
  elevated_resting_trend: "안정시 심박 상승 추세",
  sensor_gap: "센서 결측",
};
/** 심각도 → 한글. */
export const SEV_LABEL = { critical: "심각", warning: "주의", info: "참고" } as const;
/** 발생 맥락 → 한글. */
export const CTX_LABEL: Record<string, string> = { rest: "휴식", sleep: "수면", active: "활동" };
/** 계획 작업 유형 → 한글. */
export const TASK_LABEL = { research: "조사", analysis: "분석", writing: "작성" } as const;

/**
 * 가명 사용자 ID(U05, S012)에서 참가자 번호(5, 12)를 뽑는다. 원본 Fitbit 사용자 ID 대신 쓰는 가명이라, 화면에는 "참가자 5" 로 풀어 쓴다.
 * @returns 번호. `U`/`S` + 숫자 형태가 아니면 null(그 경우 ID 만 보인다).
 */
export const participantNo = (id: string): number | null => {
  const m = /^[US]0*(\d+)$/.exec(id);
  return m ? Number(m[1]) : null;
};

/** 이상 유형의 한글 이름. 미등록 유형은 원문을 반환한다. */
export const typeLabel = (t: string) => TYPE_LABEL[t] ?? t;

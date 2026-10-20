/**
 * 시각 유틸.
 *
 * 데이터의 시각은 기기 로컬 시각의 naive ISO 문자열(타임존 없음)이다. 사용자의 브라우저 타임존으로
 * 변환하면 "밤 2시의 심박"이 다른 시각으로 보이므로, 모든 파싱·표시를 UTC 필드로 통일해 값 그대로 보여준다.
 */

const HAS_ZONE = /[zZ]|[+-]\d\d:?\d\d$/;

/**
 * 타임존 표기가 없는 ISO 문자열을 UTC 로 간주해 epoch ms 로 바꾼다(명시적 타임존이 있으면 존중).
 * @param s `YYYY-MM-DDTHH:mm:ss` 형태의 문자열.
 */
export const tsToMs = (s: string): number => Date.parse(HAS_ZONE.test(s) ? s : s + "Z");

const p2 = (n: number) => String(n).padStart(2, "0");

/** `MM/DD HH:mm` (UTC 필드 기준). */
export const fmtDateTime = (ms: number): string => {
  const d = new Date(ms);
  return `${p2(d.getUTCMonth() + 1)}/${p2(d.getUTCDate())} ${p2(d.getUTCHours())}:${p2(d.getUTCMinutes())}`;
};

/**
 * `YYYY-MM-DD HH:mm` (UTC 필드 기준). 연도가 필요한 곳(기준 시각, 기간 머리글, 이상 시작 시각)에 쓴다.
 * 이 데이터는 2016년 공개 데이터라, 연도가 없으면 "최근 N일" 이 오늘 기준으로 오해된다.
 */
export const fmtDateTimeY = (ms: number): string => `${new Date(ms).getUTCFullYear()}-${fmtDateTime(ms).replace("/", "-")}`;

/** `YYYY.MM.DD` (UTC 필드 기준). */
export const fmtDate = (ms: number): string => {
  const d = new Date(ms);
  return `${d.getUTCFullYear()}.${p2(d.getUTCMonth() + 1)}.${p2(d.getUTCDate())}`;
};

/** `HH:mm` (UTC 필드 기준). */
export const fmtHM = (ms: number): string => {
  const d = new Date(ms);
  return `${p2(d.getUTCHours())}:${p2(d.getUTCMinutes())}`;
};

/** epoch ms → 타임존 없는 ISO(`YYYY-MM-DDTHH:mm:ss`). `tsToMs` 의 역변환. */
export const toNaiveIso = (ms: number): string => new Date(ms).toISOString().slice(0, 19);

/** 분 단위 지속 시간을 "25분", "2시간 10분" 으로. */
export const fmtDuration = (min: number): string =>
  min >= 60 ? `${Math.floor(min / 60)}시간${min % 60 ? ` ${min % 60}분` : ""}` : `${min}분`;

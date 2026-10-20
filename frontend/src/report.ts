/**
 * 리포트 본문에서 서버가 코드로 끼워 넣는 고정 문구를 분리한다.
 *
 * 서버는 LLM 서술과 무관하게 ① 에스컬레이션 시 맨 위 `**중요 안내** …` 단락, ② 맨 끝 면책 단락을 삽입한다
 * (안전 게이트, 검증기). 화면에서는 ①을 전폭 안전 띠로, ②를 작은 글씨로 구별해 보여주려고
 * 본문과 떼어 낸다. 구조가 달라져 분리에 실패하면 전체를 본문으로 두므로 내용이 사라지지는 않는다.
 */

/** 분리 결과. */
export interface ReportParts {
  /** 중요 안내 본문(굵은 라벨 제외). 없으면 null. */
  notice: string | null;
  /** 고정 문구를 뺀 본문. */
  body: string;
  /** 면책 문구(인용 표시 `>` 제거). 없으면 null. */
  disclaimer: string | null;
}

const NOTICE = /^\*\*중요 안내\*\*\s*/;
// 면책은 짧은 고정 문장이다. 길이 상한으로 본문 단락을 면책으로 오인하는 일을 줄인다.
const looksLikeDisclaimer = (p: string) => p.length < 500 && /의료기기|진단을 대신|진단.{0,6}아닙|의료진과 상담/.test(p);

/**
 * 리포트를 안내문·본문·면책으로 나눈다.
 * @param text 서버가 보낸 리포트(스트리밍 중의 부분 텍스트도 허용).
 * @remarks 단락이 하나뿐이면 면책으로 떼지 않는다 — 본문 전체가 사라지는 사고를 막는다.
 */
export function splitReport(text: string): ReportParts {
  let rest = text.trim();
  let notice: string | null = null;
  const first = rest.split(/\n\s*\n/)[0] ?? "";
  if (NOTICE.test(first)) {
    notice = first.replace(NOTICE, "").trim();
    rest = rest.slice(first.length).trim();
  }
  const paras = rest.split(/\n\s*\n/);
  let disclaimer: string | null = null;
  const last = paras[paras.length - 1] ?? "";
  // 헤딩·목록 항목은 면책이 아니다. 인용 표시(>)로 쓰인 면책은 허용한다.
  if (paras.length > 1 && !/^\s*(#|[-*]\s|\d+\.\s)/.test(last) && looksLikeDisclaimer(last)) {
    disclaimer = last.replace(/^>\s?/gm, "").trim();
    paras.pop();
  }
  return { notice, body: paras.join("\n\n").trim(), disclaimer };
}

/**
 * 리포트 수치 → 이벤트 각주 매칭.
 *
 * 책임: 리포트 문장 속 "128 bpm", "25분", "1시간 10분" 이 탐지된 이상의 값·지속시간과 일치할 때만 해당 이벤트를 가리킨다.
 * 원칙: 오인용(잘못된 근거 연결)은 인용 누락보다 해롭다. 그래서 보수적으로 매칭한다 —
 *   ① 수치(값 또는 지속시간)가 정확히 일치하는 후보를 모으고,
 *   ② 같은 줄이 그 이상을 가리키는 말(화면 이벤트 로그와 같은 한글 종류 이름 → 별칭 → 심각도 표기)을 담고 있어야 하며,
 *   ③ 단서를 단계적으로 적용해 후보가 정확히 하나로 좁혀질 때만 인용한다.
 * 단서의 순서: 종류 이름/별칭이 가장 강하고, 심각도 표기("심각"·"주의"·"참고")는 그보다 약해 종류 단서가 없을 때만 쓴다.
 *   내부 영문 type 코드는 리포트에 나오지 않으므로 쓰지 않는다(화면에 보이는 말로만 매칭).
 * 협력: Markdown.tsx 가 조각 단위로 호출해 위첨자 버튼을 그린다. 검증기가 "인용 숫자가 facts 에 존재"
 *   를 서버에서 보장하므로, 여기서는 연결만 책임진다.
 */
import type { Anomaly } from "./events";
import { SEV_LABEL, typeLabel } from "./labels";

// LLM 이 정식 한글 이름 대신 줄여 쓰는 표현. 유형을 구별하는 단어만 넣는다(일반 명사는 오인용을 부른다).
const ALIASES: Record<string, string[]> = {
  // "휴식 중 빈맥" 은 예전 정식 이름 — 이전 리포트도 계속 각주가 붙도록 별칭으로 둔다.
  resting_tachycardia: ["휴식 중 빈맥", "빈맥"],
  bradycardia: ["서맥"],
  personal_deviation: ["기준선", "이탈", "편차", "평소보다"],
  elevated_resting_trend: ["상승 추세", "추세"],
  sensor_gap: ["결측", "미측정", "공백", "기록 없음", "놓친"],
};

const mentionsType = (ctx: string, a: Anomaly) => [typeLabel(a.type), ...(ALIASES[a.type] ?? [])].some((w) => ctx.includes(w));
const mentionsSeverity = (ctx: string, a: Anomaly) => ctx.includes(SEV_LABEL[a.severity]);

/**
 * 수치 하나에 대응하는 이상을 찾는다.
 *
 * @param ctx 그 수치가 들어 있는 줄(문단/목록 항목) 전체. 종류·심각도 언급 여부 판단에 쓴다.
 * @param value 문장에서 읽은 수치.
 * @param unit `bpm` 이면 이상의 대표값(반올림 비교), `분` 이면 지속시간(정수 비교).
 * @param anomalies 에이전트가 이번 실행에서 보고한 이상 목록.
 * @returns 단서를 적용한 뒤 후보가 정확히 하나일 때만 그 이상, 아니면 null(미인용).
 */
export function matchCitation(ctx: string, value: number, unit: "bpm" | "분", anomalies: Anomaly[]): Anomaly | null {
  const numeric = anomalies.filter((a) =>
    // sensor_gap 의 value 는 의미가 없어 bpm 인용 대상에서 뺀다.
    unit === "bpm" ? a.type !== "sensor_gap" && Math.round(a.value) === Math.round(value) : Number.isInteger(value) && a.duration_min === value,
  );
  if (!numeric.length) return null;
  // 종류 단서(강) → 심각도 단서(약) 순으로, 후보가 하나로 좁혀지는 첫 단계에서 확정한다.
  for (const clue of [mentionsType, mentionsSeverity]) {
    const hits = numeric.filter((a) => clue(ctx, a));
    if (hits.length === 1) return hits[0];
    if (hits.length > 1) return null; // 같은 단서에서 둘 이상이면 모호 — 더 약한 단서로 억지로 고르지 않는다.
  }
  return null;
}

/** 텍스트 조각. `anomaly` 가 있으면 그 조각 바로 뒤에 각주를 붙인다. */
export type Piece = { text: string } | { text: string; anomaly: Anomaly };

/**
 * 한 줄의 텍스트를 각주 위치 기준으로 조각낸다. 조각을 이어 붙이면 원문과 같다.
 * "1시간 10분" 같은 시간 표기는 분(70)으로 환산해 지속시간과 비교한다.
 *
 * @param text 분할할 텍스트(굵게 등 인라인 단위일 수 있다).
 * @param ctx 같은 줄 전체(매칭 문맥).
 * @param anomalies 인용 후보.
 * @param seen 그 줄에서 이미 각주를 붙인 이벤트 ID. 같은 이벤트의 중복 각주를 막으려고 호출 간에 공유한다.
 */
export function splitCitations(text: string, ctx: string, anomalies: Anomaly[], seen: Set<string> = new Set()): Piece[] {
  if (!anomalies.length) return [{ text }];
  const out: Piece[] = [];
  // 앞이 숫자/소수점이면 "1128" 의 부분 문자열을 잡지 않도록 lookbehind 로 막는다.
  // 1) "N시간 [M분]"  2) "N bpm" / "N분"
  const re = /(?<![\d.])(?:(\d+)\s*시간(?:\s*(\d+)\s*분)?|(\d+(?:\.\d+)?)\s*(bpm|분))/g;
  let last = 0;
  for (let m = re.exec(text); m; m = re.exec(text)) {
    const a =
      m[1] != null
        ? matchCitation(ctx, Number(m[1]) * 60 + Number(m[2] ?? 0), "분", anomalies)
        : matchCitation(ctx, parseFloat(m[3]), m[4] as "bpm" | "분", anomalies);
    // 같은 줄에서 같은 이벤트는 처음 한 번만 각주를 붙인다(각주 도배 방지).
    if (!a || seen.has(a.anomaly_id)) continue;
    seen.add(a.anomaly_id);
    const end = m.index + m[0].length;
    out.push({ text: text.slice(last, end), anomaly: a });
    last = end;
  }
  if (last < text.length) out.push({ text: text.slice(last) });
  return out;
}

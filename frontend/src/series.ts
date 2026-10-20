/**
 * 시계열 가공(순수 함수) — 배열화, 구간 버킷팅(다운샘플), 판독값 통계, 이벤트 번호 부여.
 *
 * 책임: 서버가 준 점 목록을 차트와 상단 판독값이 쓰기 좋은 형태로 바꾼다. 서버 `/summary` 가 없거나 실패했을 때의
 *   대체 계산도 여기서 한다. 시각은 모두 epoch ms(`time.ts` 의 naive-UTC 규약)이다.
 * 협력: Timeline.tsx(bucketize), App.tsx(computeStats, restBaseline, numberAnomalies).
 */
import type { Anomaly, SeriesPoint } from "./events";
import { tsToMs } from "./time";

/** 열 지향(column-oriented) 시계열. 모든 배열은 같은 길이이고 `t` 는 오름차순이다. */
export interface Series {
  t: number[];
  hr: (number | null)[];
  steps: number[];
  asleep: boolean[];
}

/** REST 점 목록을 `Series` 로 변환한다. 누락된 steps 는 0, asleep 은 false 로 채운다. */
export const toSeries = (pts: SeriesPoint[]): Series => ({
  t: pts.map((p) => tsToMs(p.ts)),
  hr: pts.map((p) => p.hr),
  steps: pts.map((p) => p.steps ?? 0),
  asleep: pts.map((p) => !!p.asleep),
});

/**
 * 오름차순 배열에서 `x` 이상인 첫 인덱스(이분 탐색). 모두 작으면 `a.length`.
 * 30일(≈8,640점) 구간 슬라이스를 매 렌더 O(log n) 으로 자르려는 용도.
 */
export function lowerBound(a: number[], x: number): number {
  let lo = 0;
  let hi = a.length;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (a[m] < x) lo = m + 1;
    else hi = m;
  }
  return lo;
}

/**
 * 정렬된 배열의 선형보간 백분위수.
 * @param sorted 오름차순 정렬된 값.
 * @param p 0~1.
 * @returns 빈 배열이면 null.
 */
export function percentile(sorted: number[], p: number): number | null {
  if (!sorted.length) return null;
  const i = (sorted.length - 1) * p;
  const lo = Math.floor(i);
  const hi = Math.ceil(i);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
}

const sortedNums = (xs: number[]) => [...xs].sort((a, b) => a - b);

/** 다운샘플 버킷(원시 샘플 n개의 요약). */
export interface Bucket {
  /** 대표 시각(버킷 중앙 샘플). */
  t: number;
  /** 버킷의 첫/마지막 원시 샘플 시각. 기록 없음 구간의 경계를 그릴 때 쓴다. */
  tStart: number;
  tEnd: number;
  /** 바로 앞 샘플과의 간격이 허용치를 넘어 "기록 없음" 구간 뒤에서 시작하는 버킷이면 true. */
  gapBefore: boolean;
  /** 평균 심박. 유효 샘플이 없으면 null. */
  hr: number | null;
  min: number | null;
  max: number | null;
  /** 샘플당 평균 걸음수 — 합계가 아니라 평균이어야 확대/축소해도 막대 스케일이 일정하다. */
  steps: number;
  /** 수면 샘플이 과반이면 true. */
  asleep: boolean;
  /** 원시 샘플 수. 1 이면 버킷 = 원시 점(정확값). */
  n: number;
  /** 원시 인덱스 구간 [i0, i1). */
  i0: number;
  i1: number;
}

/**
 * `[t0, t1]` 구간을 최대 `nb` 개 버킷으로 요약한다.
 * @param nb 목표 버킷 수(보통 화면 폭 px / 2). 원시 점이 nb 이하이면 요약 없이 점 그대로.
 * @param gapMs 이보다 벌어진 인접 샘플 사이를 "기록 없음"으로 본다(보통 리샘플 주기의 1.5배). 서버는 빈 구간의 점을
 *   아예 보내지 않아서, 간격으로 판정하지 않으면 센서 결측이 직선으로 이어져 측정된 것처럼 보인다. 버킷은 이 틈을 넘지 않는다.
 * @returns 시간 순 버킷. 구간에 점이 없으면 빈 배열.
 */
export function bucketize(s: Series, t0: number, t1: number, nb: number, gapMs = Infinity): Bucket[] {
  const a = lowerBound(s.t, t0);
  const b = lowerBound(s.t, t1 + 1);
  const n = b - a;
  if (n <= 0) return [];
  const size = Math.max(1, Math.ceil(n / Math.max(1, nb)));
  const out: Bucket[] = [];
  for (let i = a; i < b; ) {
    let j = i + 1;
    while (j < b && j - i < size && s.t[j] - s.t[j - 1] <= gapMs) j++;
    let sum = 0;
    let cnt = 0;
    let mn = Infinity;
    let mx = -Infinity;
    let st = 0;
    let sl = 0;
    for (let k = i; k < j; k++) {
      const v = s.hr[k];
      if (v != null) ((sum += v), cnt++, (mn = Math.min(mn, v)), (mx = Math.max(mx, v)));
      st += s.steps[k];
      if (s.asleep[k]) sl++;
    }
    out.push({
      t: s.t[(i + j - 1) >> 1],
      tStart: s.t[i],
      tEnd: s.t[j - 1],
      gapBefore: i > a && s.t[i] - s.t[i - 1] > gapMs,
      hr: cnt ? sum / cnt : null,
      min: cnt ? mn : null,
      max: cnt ? mx : null,
      steps: st / (j - i),
      asleep: sl * 2 > j - i,
      n: j - i,
      i0: i,
      i1: j,
    });
    i = j;
  }
  return out;
}

/**
 * 서버의 리샘플 주기 문자열("5min", "15min", "1h", "1d")을 ms 로 바꾼다. 해석할 수 없거나 없으면 실제 점 간격의
 * 중앙값을 쓴다(결측으로 벌어진 간격이 섞여도 중앙값은 정상 주기를 가리킨다). 점이 둘 미만이면 5분.
 */
export function resampleMs(resample: string | undefined, t: number[]): number {
  const m = /^\s*(\d+(?:\.\d+)?)\s*(min|m|h|d)\s*$/i.exec(resample ?? "");
  if (m) return Number(m[1]) * { min: 60_000, m: 60_000, h: 3_600_000, d: 86_400_000 }[m[2].toLowerCase() as "min" | "m" | "h" | "d"];
  if (t.length < 2) return 5 * 60_000;
  const d = t.slice(1).map((x, i) => x - t[i]).sort((x, y) => x - y);
  return d[d.length >> 1] || 5 * 60_000;
}

/** 상단 판독값. 값이 없으면 null. */
export interface Stats {
  restingHr: number | null;
  p05: number | null;
  p95: number | null;
  coveragePct: number | null;
  lastTs: number | null;
  counts: { critical: number; warning: number; info: number };
}

/**
 * 구간 판독값을 시계열에서 직접 계산한다(서버 `/summary` 실패 시의 대체 경로).
 *
 * @param anomalies 건수 집계 대상. 시작 시각이 `[t0, t1]` 안인 것만 센다.
 * @param stepMs 리샘플 주기(ms). 주면 커버리지를 "기대 점 수(구간 길이 ÷ 주기)" 기준으로 계산한다 — 서버가 빈 구간의 점을
 *   생략하므로 존재하는 점만 세면 100% 에 가깝게 부풀려진다.
 * @remarks 안정시 심박은 "깨어 있고 걸음 0" 구간 중앙값의 근사다. 서버 `summarize()` 의 정의와 다를 수 있어
 *   화면에서 "(추정)" 으로 구분 표기한다.
 */
export function computeStats(s: Series, anomalies: Anomaly[], t0: number, t1: number, stepMs?: number): Stats {
  const a = lowerBound(s.t, t0);
  const b = lowerBound(s.t, t1 + 1);
  const all: number[] = [];
  const rest: number[] = [];
  let lastTs: number | null = null;
  for (let i = a; i < b; i++) {
    const v = s.hr[i];
    if (v == null) continue;
    all.push(v);
    lastTs = s.t[i];
    if (s.steps[i] === 0 && !s.asleep[i]) rest.push(v);
  }
  const sa = sortedNums(all);
  const counts = { critical: 0, warning: 0, info: 0 };
  for (const an of anomalies) {
    const st = tsToMs(an.start);
    if (st >= t0 && st <= t1) counts[an.severity]++;
  }
  return {
    restingHr: percentile(sortedNums(rest), 0.5),
    p05: percentile(sa, 0.05),
    p95: percentile(sa, 0.95),
    coveragePct: stepMs ? Math.min(100, (all.length / Math.max(1, (t1 - t0) / stepMs)) * 100) : b > a ? (all.length / (b - a)) * 100 : null,
    lastTs,
    counts,
  };
}

/**
 * 개인 기준선 밴드의 대체 계산: 걸음 0(휴식·수면) 구간 심박의 p05–p95.
 * @returns 유효 샘플이 없으면 null.
 */
export function restBaseline(s: Series): { lo: number; hi: number } | null {
  const v: number[] = [];
  for (let i = 0; i < s.t.length; i++) {
    const h = s.hr[i];
    if (h != null && s.steps[i] === 0) v.push(h);
  }
  const sv = sortedNums(v);
  const lo = percentile(sv, 0.05);
  const hi = percentile(sv, 0.95);
  return lo != null && hi != null ? { lo, hi } : null;
}

const DAY_MS = 86_400_000;

/**
 * 조회 기간 선택지 중 `startMs` 의 이상을 담는 가장 짧은 기간(일)을 고른다. 어느 것에도 안 들어가면 가장 긴 기간.
 * @param ranges 오름차순 기간 후보(일).
 */
export function smallestRange(startMs: number, end: number, ranges: readonly number[]): number {
  return ranges.find((r) => startMs >= end - r * DAY_MS) ?? ranges[ranges.length - 1];
}

/**
 * 조회 기간 안의 이상에 화면 번호(#1…#n)를 매긴다. 번호는 "지금 보는 기간" 안에서 1부터라서, 7일 뷰가 #4부터 시작해
 * 앞 번호가 빠진 것처럼 보이는 일이 없다. 차트 밴드·표·리포트 각주가 같은 번호를 쓰도록 단일 출처로 둔다.
 *
 * @param end 기간의 끝(epoch ms). 시작은 `end - days`.
 * @returns `rows` 기간 안의 이상(시작 시각 오름차순, 동률은 ID 순), `index` 이상 ID → 번호(1부터).
 */
export function numberInRange(list: Anomaly[], end: number, days: number): { rows: Anomaly[]; index: Map<string, number> } {
  const t0 = end - days * DAY_MS;
  const rows = list
    .filter((a) => {
      const s = tsToMs(a.start);
      return s >= t0 && s <= end;
    })
    .sort((x, y) => tsToMs(x.start) - tsToMs(y.start) || x.anomaly_id.localeCompare(y.anomaly_id));
  return { rows, index: new Map(rows.map((a, i) => [a.anomaly_id, i + 1])) };
}

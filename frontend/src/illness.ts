/**
 * 실제 사건(증상 시작·확진·회복) 표시와 "첫 경고" 문구 — 순수 함수.
 *
 * 책임: 연구 라벨(`EventsResponse`)을 차트 마커로 바꾸고, 서버가 미리 계산한 `first_warning` 을 쉬운 말로 옮긴다.
 *   첫 경고는 보는 구간에 따라 달라지면 안 되므로 화면에서 이상 목록으로 다시 계산하지 않는다.
 * 주의: 라벨은 연구용이며 진단이 아니다. "첫 경고"는 시스템이 이상을 처음 알린 시점일 뿐 질병을 예측했다는 뜻이 아니다.
 */
import type { EventsResponse, FirstWarning } from "./events";
import { tsToMs } from "./time";

type Ev = EventsResponse["events"][number];

const TYPE_KO = { symptom_onset: "증상 시작", diagnosis: "확진", recovery: "회복" } as const;
const ILLNESS_KO = { covid: "코로나19", other: "기타 질병" } as const;

/** 차트에 그릴 사건 마커. */
export interface Marker {
  /** 사건 시각(그날 정오, epoch ms — 하루 단위 라벨이라 칸 가운데에 둔다). */
  t: number;
  type: Ev["type"];
  illness: Ev["illness"];
  /** 화면 라벨. 예: "증상 시작 · 코로나19". */
  label: string;
  date: string;
  /** 증상 시작이면 서버가 계산한 첫 경고. 없으면 표시를 생략한다. */
  warning?: FirstWarning;
}

/** 사건 라벨 한 줄. 예: `증상 시작 · 코로나19`, `회복 · 기타 질병`. */
export const eventLabel = (e: Pick<Ev, "type" | "illness">): string => `${TYPE_KO[e.type]} · ${ILLNESS_KO[e.illness]}`;

/** 사건 목록을 시각 순 마커로. 해석할 수 없는 날짜는 건너뛴다. */
export function buildMarkers(events: Ev[]): Marker[] {
  return events
    .map((e) => ({ t: tsToMs(`${e.date.slice(0, 10)}T12:00:00`), type: e.type, illness: e.illness, label: eventLabel(e), date: e.date.slice(0, 10), warning: e.first_warning }))
    .filter((m) => Number.isFinite(m.t))
    .sort((a, b) => a.t - b.t);
}

/**
 * 첫 경고 한 줄. 서버 값이 없으면 null(아무것도 보이지 않는다).
 * @param typeLabel 이상 종류 코드를 한글로 바꾸는 함수
 */
export function warningText(w: FirstWarning | undefined, typeLabel: (t: string) => string): string | null {
  if (!w) return null;
  if (!w.evaluable) return "이 사건 주변은 평가할 수 없습니다";
  if (!w.hit || w.lead_days == null) return `증상 −${w.pre_days ?? 7}~+${w.post_days ?? 3}일 안에는 경고가 없었습니다`;
  const d = w.lead_days;
  const head = d > 0 ? `증상 ${d}일 전 첫 경고` : d === 0 ? "증상 당일 첫 경고" : `증상 ${-d}일 후 첫 경고`;
  const detail = [...(w.types?.length ? [w.types.map(typeLabel).join("·")] : []), ...(w.first_alarm ? [w.first_alarm] : [])].join(", ");
  return detail ? `${head} (${detail})` : head;
}

/** 차트에 그릴 첫 경고 표시: 경고 날짜(정오)와 증상 시작을 잇는다. */
export interface Lead {
  onset: Marker;
  /** 경고 날짜의 정오(epoch ms). 증상보다 늦으면 onset 보다 크다. */
  alarmMs: number;
  /** 차트 위 짧은 문구(괄호 설명 없음). */
  text: string;
}

/** 경고를 찾은 증상 시작만 차트용 표시로. */
export function leadsFor(markers: Marker[]): Lead[] {
  return markers.flatMap((onset) => {
    const w = onset.warning;
    if (onset.type !== "symptom_onset" || !w?.evaluable || !w.hit || !w.first_alarm) return [];
    const alarmMs = tsToMs(`${w.first_alarm.slice(0, 10)}T12:00:00`);
    const text = warningText(w, () => "")?.replace(/ \(.*\)$/, "");
    return Number.isFinite(alarmMs) && text ? [{ onset, alarmMs, text }] : [];
  });
}

/** 사건 이동 버튼의 짧은 이름. 예: `증상 시작 2020-03-11`. */
export const shortEventLabel = (m: Marker): string => `${TYPE_KO[m.type]} ${m.date}`;

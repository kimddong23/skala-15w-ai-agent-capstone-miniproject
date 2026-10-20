/**
 * 최상위 화면 — 상태 스트립(판독값) · 01 타임라인 · 02 이벤트 로그 · 03 에이전트 실행 기록을 조립한다.
 *
 * 책임(이 모듈이 소유하는 상태): 선택 사용자, 조회 기간·확대 구간, 선택된 이벤트, 사용자별 대화(turns)와 thread_id,
 *   진행 중 요청의 중단 핸들. 차트·표·인용이 같은 선택/확대 상태를 공유해야 하므로 한곳에서 들고 있다.
 * 비책임: 전송·파싱(sse.ts), 이벤트 → 상태 변환(turn.ts), 트레이스 계산(trace.ts), 그리기(Timeline/Run/EventTable).
 * 데이터 흐름: 사용자 선택 → (series·anomalies 30일 + summary 1/7/30일) 병렬 조회 → 1/7/30일 전환·확대는 클라이언트 처리.
 *   채팅은 `/chat/stream` SSE → `applyEvent` 로 turn 에 누적. thread_id 는 사용자별로 보존해 후속 질문을 잇는다.
* URL 동기화: 선택 사용자와 조회 기간을 `?user=U01&range=7` 로 반영한다(새로고침·링크 공유 시 같은 화면).
 * 개발자 보기: 노드 코드명·thread_id 같은 내부 식별자는 기본 화면에서 숨기고 토글로만 보여준다.
 * 키보드: Enter 전송 · Shift+Enter 줄바꿈 · Esc 중단/확대 해제 · "/" 입력창 포커스.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Activity } from "react-bootstrap-icons";
import { describeLlm, formatVersion, type HealthInfo, type VersionInfo } from "./llm";
import { approveChat, fetchAnomalies, fetchEvents, fetchHealth, fetchSeries, fetchStanfordUsers, fetchSummary, fetchUsers, fetchVersion, MOCK } from "./api";
import { EventTable } from "./EventTable";
import type { Anomaly, EventsResponse, SummaryResponse, UserInfo } from "./events";
import { buildMarkers, leadsFor, shortEventLabel, warningText } from "./illness";
import { typeLabel } from "./labels";
import { Markdown } from "./Markdown";
import { splitReport } from "./report";
import { Run } from "./Run";
import { computeStats, numberInRange, resampleMs, restBaseline, smallestRange, toSeries, type Series } from "./series";
import { Timeline } from "./Timeline";
import { UserSwitch } from "./UserSwitch";
import { fmtDateTimeY, toNaiveIso, tsToMs } from "./time";
import { mockStream } from "./mock";
import { streamChat } from "./sse";
import { applyEvent, latestApprovals, newTurn, type Turn } from "./turn";

/** 예시 질문. 순서가 `?ask=<번호>` 와 대응한다. */
const PROMPTS = ["최근 7일 심박 이상 있었는지 점검해줘", "어젯밤 수면 중 심박 괜찮았어?", "안정시 심박 정상 범위가 뭐야?"];
// localStorage 키. 사용자 ID → thread_id 맵과 테마 선택만 저장한다(대화 본문·건강 데이터는 저장하지 않는다).
const THREAD_KEY = "vs.threads";
const THEME_KEY = "vs.theme";
// 첫 방문 안내(기준 시각 설명)를 닫았는지. 닫으면 다시 보이지 않는다.
const NOTICE_KEY = "vs.notice.dismissed";
const NOTICE_KEY_STANFORD = "vs.notice.stanford.dismissed";
type Dataset = "fitbit" | "stanford";
const DAY = 86_400_000;
// 질문 최대 길이. 서버 한도(2000자)와 같다. 넘기면 서버가 422 로 거절하므로 입력 단계에서 막고 이유를 알려준다.
const MAX_QUESTION = 2000;
const COUNT_SHOWN_FROM = 1800;
// 조회 기간(일) 선택지. 30 은 서버 lookback_days 상한이자 한 번에 로드하는 범위(api.ts LOAD_DAYS)다.
const RANGES = [1, 7, 30] as const;
type Range = (typeof RANGES)[number];
type Theme = "auto" | "light" | "dark";

// 데모·문서 스크린샷·E2E 용 쿼리: `?theme=light|dark`, `?ask=<예시 번호 또는 문장>`(로드 직후 자동 전송).
const params = new URLSearchParams(typeof location === "undefined" ? "" : location.search);

const initialRange = (): Range => {
  const r = Number(params.get("range"));
  return (RANGES as readonly number[]).includes(r) ? (r as Range) : 7;
};

const EMPTY: Series = { t: [], hr: [], steps: [], asleep: [] };

// localStorage 접근 래퍼. 시크릿 모드·저장소 차단 환경에서는 접근 자체가 예외를 던지므로, 저장 실패를 기능 실패로 만들지 않는다.
const safe = {
  get: (k: string): string | null => {
    try {
      return localStorage.getItem(k);
    } catch {
      return null;
    }
  },
  set: (k: string, v: string) => {
    try {
      localStorage.setItem(k, v);
    } catch {
      // 저장은 편의 기능이라 실패해도 화면 동작에 영향이 없다.
    }
  },
};

// 저장된 thread 맵. 손상된 JSON 은 빈 맵으로 대체한다(새 대화로 시작).
const loadThreads = (): Record<string, string> => {
  try {
    return JSON.parse(safe.get(THREAD_KEY) ?? "{}");
  } catch {
    return {};
  }
};

/** 애플리케이션 루트 컴포넌트. */
export default function App() {
  // 데이터셋별 참가자 목록. 두 데이터셋은 한 목록에 섞지 않는다. stanford 는 로컬에 내려받은 경우에만 있다.
  const [usersByDs, setUsersByDs] = useState<Record<Dataset, UserInfo[]>>({ fitbit: [], stanford: [] });
  const [dataset, setDataset] = useState<Dataset>("fitbit");
  // 사건 주변 보기: 조회 구간의 끝(naive ISO). null 이면 그 참가자의 마지막 기록.
  const [anchor, setAnchor] = useState<string | null>(() => (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/.test(params.get("anchor") ?? "") ? params.get("anchor") : null));
  const [eventsInfo, setEventsInfo] = useState<EventsResponse | null>(null);
  const users = usersByDs[dataset];
  const [uid, setUid] = useState<string | null>(null);
  // 사용자가 데이터셋·참가자를 직접 고른 뒤에는 늦게 도착한 초기 로드 응답이 그 선택을 덮어쓰지 않게 한다.
  const picked = useRef(false);
  const [dev, setDev] = useState(false);
  const [noticeClosed, setNoticeClosed] = useState<Record<Dataset, boolean>>(() => ({ fitbit: safe.get(NOTICE_KEY) === "1", stanford: safe.get(NOTICE_KEY_STANFORD) === "1" }));
  // 새 대화로 지운 내용을 잠시 되돌릴 수 있게 보관한다(되돌릴 수 없는 삭제를 즉시 실행하지 않기 위해).
  const [undo, setUndo] = useState<{ uid: string; turns: Turn[]; thread?: string } | null>(null);
  const [range, setRange] = useState<Range>(initialRange);
  // 시계열은 해상도가 기간마다 달라(서버: ≤7일 5분, 14일 이하 15분, 그 이상 1시간) 사용자·기간별로 받아 캐시한다.
  const [seriesCache, setSeriesCache] = useState<Record<string, { series: Series; stepMs: number }>>({});
  const inflight = useRef(new Set<string>());
  const [seriesErr, setSeriesErr] = useState<string | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [anomsReady, setAnomsReady] = useState(false);
  const [anoms, setAnoms] = useState<Anomaly[]>([]);
  const [sums, setSums] = useState<Record<number, SummaryResponse["summary"] | null>>({});
  const [reloadKey, setReloadKey] = useState(0);
  const [zoom, setZoom] = useState<[number, number] | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [health, setHealth] = useState<"checking" | "ok" | "mock" | "down">("checking");
  const [healthInfo, setHealthInfo] = useState<HealthInfo | null>(null);
  const [versionInfo, setVersionInfo] = useState<VersionInfo | null>(null);
  const [theme, setTheme] = useState<Theme>(() => (params.get("theme") ?? safe.get(THEME_KEY) ?? "auto") as Theme);

  const [turns, setTurns] = useState<Record<string, Turn[]>>({});
  const [threads, setThreads] = useState<Record<string, string>>(loadThreads);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  // 실행 중 단계의 경과 시간 카운터를 갱신하기 위한 리렌더 트리거(값 자체는 쓰지 않는다).
  const [, setTick] = useState(0);
  const abortRef = useRef<AbortController | null>(null);
  // turn id → 질문 전송 시각(performance.now). 경과 시간을 상태에 넣으면 매 틱마다 turns 전체가 바뀌므로 ref 로 둔다.
  const startRef = useRef<Record<string, number>>({});
  const logRef = useRef<HTMLDivElement>(null);
  const innerRef = useRef<HTMLDivElement>(null);
  // 실행 기록 목록이 하단을 따라가는 중인지. 사용자가 위로 스크롤하면 꺼서 읽던 위치를 빼앗지 않는다.
  const stick = useRef(true);
  const composerRef = useRef<HTMLTextAreaElement>(null);

  // ---- 테마: "자동" 은 data-theme 를 지워 CSS 의 prefers-color-scheme 에 맡긴다.
  useEffect(() => {
    const el = document.documentElement;
    if (theme === "auto") el.removeAttribute("data-theme");
    else el.setAttribute("data-theme", theme);
    safe.set(THEME_KEY, theme);
    // 브라우저 UI 색(theme-color)을 실제 배경에 맞춘다(수동 테마 전환 포함). 스타일 적용 뒤의 값을 읽으려고 한 프레임 미룬다.
    requestAnimationFrame(() => document.querySelector('meta[name="theme-color"]')?.setAttribute("content", getComputedStyle(document.body).backgroundColor));
  }, [theme]);

  // ---- 데이터 ----------------------------------------------------------------------
  useEffect(() => {
    let alive = true; // StrictMode 이중 실행·재시도로 앞선 요청의 늦은 응답이 상태를 덮어쓰지 않게 한다
    fetchHealth().then((h) => {
      if (!alive) return;
      setHealth(!h ? "down" : MOCK ? "mock" : "ok");
      setHealthInfo(h);
    });
    fetchVersion().then((v) => alive && setVersionInfo(v));
    // FitBit 목록이 실패하면 오류 상태, Stanford 는 없어도 정상(빈 목록 → 데이터셋 선택을 숨김).
    Promise.all([fetchUsers("fitbit"), fetchStanfordUsers()])
      .then(([f, s]) => {
        if (!alive) return;
        setUsersByDs({ fitbit: f, stanford: s });
        if (picked.current) return;
        const ds: Dataset = params.get("dataset") === "stanford" && s.length ? "stanford" : "fitbit";
        setDataset(ds);
        const list = ds === "stanford" ? s : f;
        setUid((cur) => cur ?? list.find((x) => x.user_id === params.get("user"))?.user_id ?? list[0]?.user_id ?? null);
      })
      .catch((e) => alive && setLoadErr(String(e.message ?? e)));
    return () => {
      alive = false;
    };
  }, [reloadKey]);

  // 실제 사건 라벨(증상 시작·확진·회복). 사용자가 바뀌면 새로 받고, 없으면 사건 표시를 생략한다.
  useEffect(() => {
    if (!uid) return;
    let alive = true;
    setEventsInfo(null);
    fetchEvents(uid).then((e) => alive && setEventsInfo(e));
    return () => {
      alive = false;
    };
  }, [uid]);

  useEffect(() => {
    if (!uid) return;
    let alive = true;
    setAnomsReady(false);
    setAnoms([]);
    setLoadErr(null);
    setZoom(null);
    setSelectedId(null);
    // 서버 요약(/summary)은 판독값·기준선의 출처다. 요약 실패가 차트 로딩을 막으면 안 되므로 별도 체인으로 두고,
    // 실패하면 null 로 남겨 시계열에서 계산한 대체값을 쓴다. `alive` 는 사용자 전환 후 도착한 늦은 응답을 버리는 가드다.
    setSums({});
    Promise.all(RANGES.map((d) => fetchSummary(uid, d, anchor ?? undefined).then((r) => r.summary, () => null))).then((r) => alive && setSums(Object.fromEntries(RANGES.map((d, i) => [d, r[i]]))));
    fetchAnomalies(uid, undefined, anchor ?? undefined)
      .then((a) => alive && (setAnoms(a), setAnomsReady(true)))
      .catch((e) => alive && setLoadErr(String(e.message ?? e)));
    return () => {
      alive = false;
    };
  }, [uid, reloadKey, anchor]);

  // 시계열: 사용자·기간·구간 끝 키로 캐시. 늦게 도착한 응답은 키가 달라 다른 화면을 덮어쓰지 않는다. 진행 중 요청은 중복 호출하지 않는다.
  useEffect(() => {
    if (!uid) return;
    const key = `${uid}:${range}:${anchor ?? ""}`;
    if (seriesCache[key]) return setSeriesErr(null);
    if (inflight.current.has(key)) return;
    inflight.current.add(key);
    setSeriesErr(null);
    fetchSeries(uid, range, anchor ?? undefined)
      .then((r) => {
        const s = toSeries(r.points);
        setSeriesCache((c) => ({ ...c, [key]: { series: s, stepMs: resampleMs(r.resample, s.t) } }));
      })
      .catch((e) => setSeriesErr(String(e.message ?? e)))
      .finally(() => inflight.current.delete(key));
  }, [uid, range, anchor, reloadKey, seriesCache]);

  useEffect(() => safe.set(THREAD_KEY, JSON.stringify(threads)), [threads]);

  // 선택 상태를 URL 에 반영한다(히스토리는 쌓지 않는다).
  useEffect(() => {
    if (!uid) return;
    const q = new URLSearchParams(location.search);
    q.set("user", uid);
    q.set("range", String(range));
    if (dataset === "stanford") q.set("dataset", dataset);
    else q.delete("dataset");
    if (anchor) q.set("anchor", anchor);
    else q.delete("anchor");
    history.replaceState(null, "", `${location.pathname}?${q}`);
  }, [uid, range, dataset, anchor]);

  // 되돌리기 안내는 10초 뒤 사라진다.
  useEffect(() => {
    if (!undo) return;
    const id = setTimeout(() => setUndo(null), 10_000);
    return () => clearTimeout(id);
  }, [undo]);

  // ---- 파생 값 -----------------------------------------------------------------------
  const user = users.find((u) => u.user_id === uid);
  const curSeries = uid ? seriesCache[`${uid}:${range}:${anchor ?? ""}`] : undefined;
  const series = curSeries?.series ?? EMPTY;
  const stepMs = curSeries?.stepMs ?? 5 * 60_000;
  // 기간의 끝: 사건 주변 보기를 켰으면 그 시각, 아니면 사용자 목록의 마지막 기록 시각. (기간마다 받은 시계열의 마지막 점은
  // 리샘플 버킷 시작 시각이라 서로 다를 수 있어 시계열이 아니라 목록 값을 쓴다.)
  const end = anchor ? tsToMs(anchor) : user ? tsToMs(user.end) : 0;
  const markers = useMemo(() => buildMarkers(eventsInfo?.events ?? []), [eventsInfo]);
  // 첫 경고는 서버가 미리 계산해 사건에 붙여 준다(보는 구간과 무관). 화면에서 이상 목록으로 다시 계산하지 않는다.
  const leads = useMemo(() => leadsFor(markers), [markers]);
  const loadError = loadErr ?? seriesErr;
  const load: { s: "loading" | "ok" | "error"; msg?: string } = loadError ? { s: "error", msg: loadError } : !curSeries || !anomsReady ? { s: "loading" } : { s: "ok" };
  const rangeStart = end - range * DAY;
  const domain: [number, number] = zoom ?? [rangeStart, end];
  // 번호는 기간마다 1부터 새로 매긴다(7일 뷰가 #4 부터 시작해 앞 번호가 빠진 듯 보이지 않게). 각주가 가리키는 이상이
  // 현재 기간 밖이면 클릭 시 옮겨 갈 기간의 번호를 미리 보여주기 위해 세 기간 모두 계산해 둔다.
  const numbering = useMemo(() => Object.fromEntries(RANGES.map((r) => [r, numberInRange(anoms, end, r)])), [anoms, end]);
  const { rows, index } = numbering[range];
  const stats = useMemo(() => computeStats(series, anoms, rangeStart, end, stepMs), [series, anoms, rangeStart, end, stepMs]);
  const sm = sums[range] ?? null;
  const sm30 = sums[30] ?? null;
  const serverBaseline = sm30 && sm30.hr_p05 != null && sm30.hr_p95 != null ? { lo: sm30.hr_p05, hi: sm30.hr_p95 } : null;
  // 기준선은 객체 동일성이 아니라 값(lo/hi)이 바뀔 때만 재계산한다.
  const baseline = useMemo(() => serverBaseline ?? restBaseline(series), [serverBaseline?.lo, serverBaseline?.hi, series]);
  // 상단 판독값: 서버 집계를 우선하고, 없을 때만 시계열 추정값(화면에 "(추정)" 으로 구분 표기).
  const ro = {
    resting: sm?.resting_hr ?? stats.restingHr,
    p05: sm?.hr_p05 ?? stats.p05,
    p95: sm?.hr_p95 ?? stats.p95,
    cov: sm?.coverage_pct ?? stats.coveragePct,
  };
  const myTurns = (uid && turns[uid]) || [];
  const latestApproval = latestApprovals(myTurns);
  const lastDone = [...myTurns].reverse().find((t) => t.status === "done" && t.report);

  // ---- 데이터셋·참가자·사건 주변 보기 ---------------------------------------------------
  const pickUser = (id: string) => {
    picked.current = true;
    setUid(id);
    setAnchor(null); // 다른 참가자의 "사건 주변" 구간을 끌고 가지 않는다
  };
  const switchDataset = (ds: Dataset) => {
    const list = usersByDs[ds];
    if (ds === dataset || !list.length) return;
    picked.current = true;
    setDataset(ds);
    setUid(list[0].user_id);
    setAnchor(null);
    setZoom(null);
    setSelectedId(null);
  };
  // 사건 주변으로 이동: 사건 7일 뒤를 구간 끝으로 30일을 본다(사건 앞 23일 + 뒤 7일). 기록 끝을 넘지 않는다.
  // 연구 데이터는 몇 달이라, 라벨이 "마지막 30일" 밖에 있으면 이렇게 옮겨야 사건과 앞선 경고를 볼 수 있다.
  const jumpToEvent = (t: number) => {
    const last = user ? tsToMs(user.end) : t;
    setAnchor(toNaiveIso(Math.min(t + 7 * DAY, last)));
    setRange(30);
    setZoom(null);
    setSelectedId(null);
  };

  // ---- 이벤트 선택/확대 ------------------------------------------------------------------
  // 선택된 이벤트의 표 행을 보이는 위치로 스크롤한다(표는 자체 스크롤 컨테이너).
  const select = useCallback((id: string | null) => setSelectedId(id), []);
  useEffect(() => {
    if (selectedId) document.getElementById(`ev-${selectedId}`)?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [selectedId]);

  // 이벤트로 "이동": 필요하면 조회 기간을 넓히고, 이벤트 길이의 2배(최소 2시간) 여유를 두고 확대하며, 선택 상태로 만든다.
  // 표 행·번호 클릭과 리포트 각주 클릭이 모두 이 동작을 쓴다.
  const focusAnomaly = useCallback(
    (a: Anomaly) => {
      const s = tsToMs(a.start);
      const e = tsToMs(a.end);
      const need = smallestRange(s, end, RANGES);
      if (need > range) setRange(need as Range);
      const pad = Math.max((e - s) * 2, 2 * 3_600_000);
      setZoom([Math.max(end - RANGES[RANGES.length - 1] * DAY, s - pad), Math.min(end, e + pad)]);
      setSelectedId(a.anomaly_id);
    },
    [end, range],
  );

  // 에이전트가 보고한 이상 → 로그의 이벤트. 서버가 보고 시점마다 ID 를 다르게 부여할 가능성에 대비해 (유형, 시작 시각) 으로도 찾는다.
  const inLog = useCallback((a: Anomaly) => anoms.find((x) => x.anomaly_id === a.anomaly_id) ?? anoms.find((x) => x.type === a.type && x.start === a.start) ?? null, [anoms]);
  const cite = useMemo(
    () => ({
      anomalies: [] as Anomaly[],
      indexOf: (a: Anomaly) => {
        const m = inLog(a);
        if (!m) return null;
        // 현재 기간 안이면 지금 번호, 밖이면 클릭 후 전환될 기간의 번호.
        const r = tsToMs(m.start) >= end - range * DAY ? range : smallestRange(tsToMs(m.start), end, RANGES);
        return numbering[r].index.get(m.anomaly_id) ?? null;
      },
      isOutside: (a: Anomaly) => {
        const m = inLog(a);
        return !!m && tsToMs(m.start) < end - range * DAY;
      },
      onCite: (a: Anomaly) => {
        const m = inLog(a);
        if (m) {
          focusAnomaly(m);
          document.getElementById("sec-timeline")?.scrollIntoView({ block: "nearest", behavior: "smooth" });
        }
      },
    }),
    [inLog, numbering, range, end, focusAnomaly],
  );

  // ---- 채팅 ----------------------------------------------------------------------------
  // 스트리밍으로 내용이 자라는 동안 하단을 따라간다. 스크롤 이벤트가 아니라 ResizeObserver 를 쓰는 이유: 토큰/단계가
  // 늘어 높이가 변하는 시점은 스크롤 이벤트가 오지 않기 때문이다.
  useEffect(() => {
    const el = logRef.current;
    const inner = innerRef.current;
    if (!el || !inner) return;
    const go = () => stick.current && (el.scrollTop = el.scrollHeight);
    const ro = new ResizeObserver(go);
    ro.observe(inner);
    go();
    return () => ro.disconnect();
  }, [uid]);

  // 실행 중에만 120ms 간격으로 리렌더해 진행 단계의 경과 카운터를 살아 있게 한다(분석/작성이 20초 안팎 걸린다).
  useEffect(() => {
    if (!busy) return;
    const id = setInterval(() => setTick((n) => n + 1), 120);
    return () => clearInterval(id);
  }, [busy]);

  /**
   * 질문을 보내고 SSE 이벤트를 turn 에 누적한다. 한 번에 한 요청만 허용한다(진행 중이면 무시).
   * 결과는 질문 시점의 사용자(`owner`)의 대화에 귀속한다 — 스트리밍 중 사용자를 바꿔도 섞이지 않는다.
   */
  const send = useCallback(
    async (text: string) => {
      // 붙여넣기·프로그램 입력은 maxLength 를 우회할 수 있어 전송 직전에도 한도로 자른다.
      const q = text.trim().slice(0, MAX_QUESTION);
      if (!q || busy || !uid) return;
      const owner = uid;
      const id = crypto.randomUUID();
      const ac = new AbortController();
      const t0 = performance.now();
      startRef.current[id] = t0;
      abortRef.current = ac;
      stick.current = true;
      setInput("");
      setUndo(null);
      setBusy(true);
      const upd = (fn: (t: Turn) => Turn) => setTurns((s) => ({ ...s, [owner]: (s[owner] ?? []).map((t) => (t.id === id ? fn(t) : t)) }));
      setTurns((s) => ({ ...s, [owner]: [...(s[owner] ?? []), newTurn(id, q)] }));
      // 단일 열(모바일)에서는 에이전트 패널이 화면 아래에 있어, 새 실행 기록 위치로 스크롤해야 진행이 보인다.
      if (!params.has("ask") && matchMedia("(max-width: 1180px)").matches) setTimeout(() => document.querySelector(`[data-run="${id}"]`)?.scrollIntoView({ block: "start", behavior: "smooth" }), 60);

      const body = { message: q, user_id: owner, thread_id: threads[owner] ?? null };
      try {
        const stream = MOCK ? mockStream(q, owner, ac.signal) : streamChat(body, ac.signal);
        for await (const ev of stream) {
          const at = performance.now() - t0;
          upd((t) => applyEvent(t, ev, at));
          if (ev.type === "done") setThreads((th) => ({ ...th, [owner]: ev.thread_id }));
        }
        // done/error 없이 스트림이 닫힌 경우(서버 중단·프록시 타임아웃)를 "실행 중" 으로 방치하지 않는다.
        upd((t) => (t.status === "running" ? { ...t, status: "error", error: "완료 이벤트 없이 스트림이 종료되었습니다." } : t));
      } catch (e) {
        const at = performance.now() - t0;
        if (ac.signal.aborted) upd((t) => ({ ...t, status: "aborted", elapsedMs: at }));
        else upd((t) => ({ ...t, status: "error", elapsedMs: at, error: e instanceof Error ? e.message : String(e) }));
      } finally {
        setBusy(false);
        abortRef.current = null;
        // 끝나면 실행 기록의 머리(질문·안전 띠)가 보이도록 올린다(스트리밍 중에는 하단을 따라갔다). 80ms: 트레이스가
        // 접히며 높이가 확정된 뒤에 위치를 계산하기 위한 지연이다.
        setTimeout(() => {
          const box = logRef.current;
          const el = box?.querySelector<HTMLElement>(`[data-run="${id}"]`);
          if (box && el && params.has("ask")) {
            // 데모/캡처(?ask=)에서는 페이지를 움직이지 않는다.
          } else if (box && el && matchMedia("(max-width: 1180px)").matches) {
            el.scrollIntoView({ block: "start", behavior: "smooth" });
          } else if (box && el) {
            stick.current = false;
            box.scrollTop = el.getBoundingClientRect().top - box.getBoundingClientRect().top + box.scrollTop;
          }
        }, 80);
      }
    },
    [busy, uid, threads],
  );

  // `?ask=` 자동 전송은 한 번만. 첫 로드가 끝나면(load.s === "ok") 실행하고 null 로 비워 재실행을 막는다.
  const autoAsk = useRef(params.get("ask"));
  useEffect(() => {
    const a = autoAsk.current;
    if (a == null || load.s !== "ok") return;
    autoAsk.current = null;
    const q = PROMPTS[Number(a)] ?? a;
    void send(q);
  }, [load.s, send]);

  // 새 대화: 이 사용자의 thread_id 와 화면의 실행 기록을 비운다(서버 쪽 checkpoint 는 그대로 두고 새 thread 로 시작).
  // 보호자 알림 승인/거절: 결과로 받은 최종 리포트(결과 한 줄 포함)로 본문을 교체하고 카드를 결과 상태로 바꾼다.
  // 409(이미 처리됨·만료)는 오류가 아니라 "이미 처리된 요청" 안내로, 그 밖의 실패는 다시 누를 수 있게 둔다.
  const approve = useCallback(
    async (turnId: string, a: NonNullable<Turn["approval"]>, approved: boolean) => {
      if (!uid) return;
      const owner = uid;
      const patchTurn = (fn: (t: Turn) => Turn) => setTurns((s) => ({ ...s, [owner]: (s[owner] ?? []).map((t) => (t.id === turnId ? fn(t) : t)) }));
      patchTurn((t) => (t.approval ? { ...t, approval: { ...t.approval, status: "sending" } } : t));
      try {
        const r = await approveChat(a.thread_id, approved, a.approval_id);
        if (r.ok) {
          patchTurn((t) => (t.approval ? { ...t, report: r.data.report || t.report, approval: { ...t.approval, status: approved ? "approved" : "denied" } } : t));
        } else {
          patchTurn((t) => (t.approval ? { ...t, approval: { ...t.approval, status: r.conflict ? "conflict" : "error" } } : t));
        }
      } catch {
        patchTurn((t) => (t.approval ? { ...t, approval: { ...t.approval, status: "error" } } : t));
      }
    },
    [uid],
  );

  const newThread = () => {
    if (!uid || busy) return;
    setUndo({ uid, turns: turns[uid] ?? [], thread: threads[uid] });
    setThreads((t) => {
      const { [uid]: _removed, ...rest } = t;
      return rest;
    });
    setTurns((s) => ({ ...s, [uid]: [] }));
  };

  const undoClear = () => {
    if (!undo) return;
    setTurns((s) => ({ ...s, [undo.uid]: undo.turns }));
    if (undo.thread) setThreads((t) => ({ ...t, [undo.uid]: undo.thread! }));
    setUndo(null);
  };

  // ---- 전역 키보드. 입력 요소에 포커스가 있을 때는 "/" 를 문자로 입력하도록 가로채지 않는다.
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      const typing = tag === "TEXTAREA" || tag === "INPUT" || tag === "SELECT";
      if (e.key === "Escape") {
        if (abortRef.current) abortRef.current.abort();
        else setZoom(null);
      } else if (e.key === "/" && !typing && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        composerRef.current?.focus();
      }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, []);

  const loading = load.s === "loading";
  const llm = describeLlm(healthInfo);
  const ver = formatVersion(versionInfo, dev);
  const threadId = uid ? threads[uid] : undefined;
  const nextTheme: Record<Theme, Theme> = { auto: "light", light: "dark", dark: "auto" };
  const fmtN = (v: number | null, d = 0) => (v == null ? "—" : v.toFixed(d));

  return (
    <div className="app">
      <a className="skip" href="#main">본문으로 건너뛰기</a>
      {versionInfo?.supervisor_version === "v1" && (
        <div className="v1banner" role="alert" data-testid="v1-banner">
          기준선(v1) 모드 — 안전 장치 없음, 비교 실험 전용
        </div>
      )}
      <header className="strip">
        <div className="strip-top">
          <div className="brand">
            <Activity size={24} aria-hidden="true" />
            <h1 translate="no">VITAL SENTINEL</h1>
            <span>심박 이상 징후 감지 에이전트</span>
          </div>
          <UserSwitch users={users} value={uid} onChange={pickUser} />
          <div className="spacer" />
          <button className="textbtn" onClick={() => setTheme(nextTheme[theme])} aria-label={`화면 ${{ auto: "자동", light: "라이트", dark: "다크" }[theme]} — 눌러서 테마 전환`} data-testid="theme-toggle">
            화면 {{ auto: "자동", light: "라이트", dark: "다크" }[theme]}
          </button>
        </div>

        {usersByDs.stanford.length > 0 && (
          <div className="dsrow" role="group" aria-label="데이터셋">
            <span className="dslabel">데이터셋</span>
            <div className="seg">
              <button aria-pressed={dataset === "fitbit"} onClick={() => switchDataset("fitbit")} data-testid="dataset-fitbit">
                개발 데이터 · FitBit ({usersByDs.fitbit.length}명)
              </button>
              <button aria-pressed={dataset === "stanford"} onClick={() => switchDataset("stanford")} data-testid="dataset-stanford">
                외부 검증 · Stanford COVID-19 (로컬 전용, {usersByDs.stanford.length}명)
              </button>
            </div>
          </div>
        )}

        <div className="status">
          <span
            className="dataset-tag"
            title={
              dataset === "stanford"
                ? "Stanford 의 COVID-19 웨어러블 연구 데이터. 이 컴퓨터에만 있는 데이터이며, 개인정보 보호를 위해 날짜가 이동되어 있습니다."
                : "Furberg 등(2016)이 Zenodo 에 공개한 Fitbit 데이터(CC BY 4.0)를 Kaggle 'Fitbit Fitness Tracker Data'(arashnic/fitbit)로 내려받아 썼습니다. 2016년 3~5월에 측정한 공개 데이터입니다."
            }
            data-testid="dataset-tag"
            translate="no"
          >
            {dataset === "stanford" ? "Stanford COVID-19 연구 데이터 (로컬 전용)" : "2016 FitBit 공개 데이터 (CC BY 4.0)"}
          </span>
          {llm && (
            <div className={`conn llm ${llm.connected ? "ok" : "off"}`} data-testid="llm-status" data-state={llm.connected ? "on" : "off"}>
              <i />
              {llm.connected ? "LLM 연결됨" : "LLM 미연결 — 기본 리포트 모드"}
            </div>
          )}
          <div className={`conn ${health}`} title="백엔드 상태" data-testid="api-status" data-state={health}>
            <i />
            {{ checking: "서버 확인 중…", ok: "서버 연결됨", mock: "시연용 합성 데이터", down: "서버에 연결되지 않음" }[health]}
          </div>
          {ver && (
            <div className="ver mono" title={ver.title} data-testid="version-info" translate="no">
              {ver.short}
            </div>
          )}
        </div>

        <dl className={`readouts ${loading ? "is-loading" : ""}`} aria-label="판독값" aria-busy={loading}>
          <div className="ro" data-testid="readout-resting">
            <dt>안정시 심박 <small>최근 {range}일, 가만히 있을 때{sm?.resting_hr == null && " (추정)"}</small></dt>
            <dd><span className="big mono">{fmtN(ro.resting)}</span><span className="unit">bpm</span></dd>
          </div>
          <div className="ro" data-testid="readout-range">
            <dt>심박수 범위 <small>최근 {range}일, 대부분의 시간이 이 안</small></dt>
            <dd><span className="big mono">{fmtN(ro.p05)}<em>–</em>{fmtN(ro.p95)}</span><span className="unit">bpm</span></dd>
          </div>
          <div className="ro" data-testid="readout-coverage">
            <dt>기록된 시간 <small>최근 {range}일 중 심박이 있는 비율{sm?.coverage_pct == null && " (추정)"}</small></dt>
            <dd><span className="big mono">{fmtN(ro.cov, 1)}</span><span className="unit">%</span></dd>
          </div>
          <div className="ro" data-testid="readout-last">
            <dt>데이터 기준 시각 <small>{anchor ? "선택한 사건 주변 구간의 끝" : "이 참가자의 마지막 기록"}{dataset === "stanford" && " · 날짜는 이동된 값"}</small></dt>
            <dd><span className="big mono t">{end ? fmtDateTimeY(end) : "—"}</span></dd>
          </div>
          <div className="ro counts" data-testid="readout-counts">
            <dt>발견된 이상 <small>최근 {range}일</small></dt>
            <dd>
              {(["critical", "warning", "info"] as const).map((s) => (
                <span key={s} className={`cnt ${s} ${stats.counts[s] ? "" : "zero"}`} data-testid={`count-${s}`}>
                  <i />
                  <span className="big mono">{stats.counts[s]}</span>
                  <span className="unit">{{ critical: "심각", warning: "주의", info: "참고" }[s]}</span>
                </span>
              ))}
            </dd>
          </div>
        </dl>
      </header>

      {!noticeClosed[dataset] && (
        <div className="notice" role="note" data-testid="dataset-notice">
          <span>
            {dataset === "stanford"
              ? "연구 데이터라 날짜는 개인정보 보호를 위해 이동되어 있습니다. ‘최근 N일’은 각 참가자의 마지막 기록을 기준으로 계산합니다."
              : "2016년 공개 데이터라 ‘최근 N일’은 오늘이 아니라 각 참가자의 마지막 기록을 기준으로 계산합니다."}
          </span>
          <button
            className="textbtn"
            onClick={() => {
              setNoticeClosed((c) => ({ ...c, [dataset]: true }));
              safe.set(dataset === "stanford" ? NOTICE_KEY_STANFORD : NOTICE_KEY, "1");
            }}
            data-testid="notice-close"
          >
            닫기
          </button>
        </div>
      )}

      <main className="cols" id="main">
        <div className="col-main">
          <section className="panel" id="sec-timeline" aria-labelledby="h-tl">
            <div className="ph">
              <h2 id="h-tl">심박 추이</h2>
              <span className="ph-note mono">{fmtDateTimeY(domain[0])} ~ {fmtDateTimeY(domain[1])}</span>
              <div className="spacer" />
              {zoom && <button className="textbtn" onClick={() => setZoom(null)} data-testid="zoom-reset">확대 해제 <kbd>Esc</kbd></button>}
              <div className="seg" role="group" aria-label="조회 기간">
                {RANGES.map((r) => (
                  <button key={r} aria-pressed={range === r} onClick={() => (setRange(r), setZoom(null))} data-testid={`range-${r}`}>{r}일</button>
                ))}
              </div>
            </div>
            {load.s === "error" ? (
              <div className="state err" role="alert">
                <b>데이터를 불러오지 못했습니다.</b>
                <span>서버에 연결할 수 없습니다. 잠시 후 다시 시도해 주세요.</span>
                {dev && <span className="mono" translate="no">{load.msg}</span>}
                <button
                  className="textbtn"
                  onClick={() => {
                    setLoadErr(null);
                    setSeriesErr(null);
                    setSeriesCache({});
                    setReloadKey((k) => k + 1);
                  }}
                >
                  다시 시도
                </button>
              </div>
            ) : loading ? (
              <div className="skel-chart" aria-busy="true" aria-label="차트 불러오는 중…"><i /><i /><i /><i /></div>
            ) : series.t.length === 0 ? (
              <div className="state">이 사용자의 심박 데이터가 없습니다.</div>
            ) : (
              <Timeline
                series={series}
                anomalies={anoms}
                index={index}
                domain={domain}
                baseline={baseline}
                selectedId={selectedId}
                onSelect={select}
                onBrush={setZoom}
                onReset={() => setZoom(null)}
                stepMs={stepMs}
                markers={markers}
                leads={leads}
                label={`${uid} 심박 추이. ${fmtDateTimeY(domain[0])}부터 ${fmtDateTimeY(domain[1])}까지. 발견된 이상은 아래 목록 참고`}
              />
            )}
            <div className="legend">
              <span><i className="lg line" />심박수 (옅은 면은 구간 안 최저~최고)</span>
              <span><i className="lg gap" />기록 없음</span>
              <span><i className="lg base" />평소 범위{baseline ? ` ${Math.round(baseline.lo)}–${Math.round(baseline.hi)} bpm` : ""}<span className="muted">{serverBaseline ? " (최근 30일)" : " (움직이지 않을 때)"}</span></span>
              <span className="sevkey"><i className="lg crit" />심각<i className="lg warn" />주의<i className="lg info" />참고</span>
              <span className="hint">드래그로 확대, 더블클릭으로 해제, 색 띠를 누르면 이상 선택</span>
            </div>
            {markers.length > 0 && (
              <div className="events" data-testid="events-panel">
                <div className="events-row">
                  <span className="muted">실제 사건</span>
                  {markers.map((m) => (
                    <button key={`${m.type}${m.t}`} className="textbtn" onClick={() => jumpToEvent(m.t)} data-testid="event-jump" data-type={m.type}>
                      {shortEventLabel(m)}
                    </button>
                  ))}
                  {anchor && (
                    <button className="textbtn" onClick={() => setAnchor(null)} data-testid="anchor-reset">
                      마지막 기록으로
                    </button>
                  )}
                </div>
                {markers
                  .filter((m) => m.type === "symptom_onset" && warningText(m.warning, typeLabel))
                  .map((m) => (
                    <p key={m.t} className="lead-line" data-testid="lead-text">
                      <b>{m.label} {m.date}</b> — {warningText(m.warning, typeLabel)}
                    </p>
                  ))}
                <p className="muted">연구용 라벨이며 진단이 아닙니다.</p>
                <p className="muted">날짜는 개인정보 보호를 위해 이동되어 실제 날짜와 다릅니다.</p>
              </div>
            )}
          </section>

          <section className="panel" aria-labelledby="h-ev">
            <div className="ph">
              <h2 id="h-ev">발견된 이상</h2>
              <span className="ph-note">최근 {range}일, {rows.length}건</span>
            </div>
            {loading ? (
              <div className="skel" aria-busy="true"><i /><i /><i /></div>
            ) : rows.length ? (
              <EventTable rows={rows} index={index} selectedId={selectedId} onFocus={focusAnomaly} dev={dev} />
            ) : (
              <div className="state">이 기간에 탐지된 이상이 없습니다.</div>
            )}
          </section>
        </div>

        <aside className="panel agent" aria-labelledby="h-ag">
          <div className="ph">
            <h2 id="h-ag">에이전트 실행 기록</h2>
            {dev && threadId && <span className="ph-note mono" translate="no">thread_id={threadId}</span>}
            <div className="spacer" />
            <button className="textbtn" aria-pressed={dev} onClick={() => setDev((d) => !d)} data-testid="dev-toggle">개발자 보기</button>
            <button className="textbtn" onClick={() => window.print()} disabled={!lastDone} data-testid="print-report">리포트 인쇄</button>
            <button className="textbtn" onClick={newThread} disabled={busy || !myTurns.length} data-testid="new-thread">새 대화</button>
          </div>

          {llm && !llm.connected && (
            <div className="llmbar" role="status" data-testid="llm-notice">
              <b>LLM 미연결 — 기본 리포트 모드</b>
              <span>이상 탐지와 안전 안내는 그대로 동작하고, 리포트는 규칙으로 만든 기본 문장으로 나옵니다.</span>
              {llm.reason && <span className="muted">원인: {llm.reason}</span>}
            </div>
          )}

          {/* tabIndex: 스크롤 영역에 키보드로 접근해 위/아래 키로 읽을 수 있게 한다. */}
          <div
            className="runs"
            role="region"
            aria-label="실행 기록 목록"
            tabIndex={0}
            ref={logRef}
            onScroll={(e) => {
              const el = e.currentTarget;
              stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
            }}
          >
            <div ref={innerRef}>
            {!myTurns.length && (
              <div className="empty">
                <p>질문을 보내면 AI가 데이터를 조사하는 과정이 여기에 쌓입니다.</p>
                <p className="muted">단계마다 걸린 시간과 판단 이유를 볼 수 있습니다.</p>
              </div>
            )}
            {myTurns.map((t, i) => (
              <Run key={t.id} turn={t} onApprove={(a, ok) => approve(t.id, a, ok)} approvalStale={!!t.approval && latestApproval.get(t.approval.thread_id) !== t.approval.approval_id} n={i + 1} anchor={end ? fmtDateTimeY(end) : undefined} dev={dev} open={i === myTurns.length - 1} now={Math.max(0, performance.now() - (startRef.current[t.id] ?? 0))} cite={cite} />
            ))}
            </div>
          </div>
          {undo && undo.uid === uid && (
            <div className="undo" role="status" data-testid="undo">
              <span>대화를 지웠습니다.</span>
              <button className="textbtn" onClick={undoClear}>되돌리기</button>
            </div>
          )}

          <div className={`composer ${myTurns.length ? "has-runs" : ""}`}>
            <div className="prompts" role="group" aria-label="예시 질문">
              <span className="muted">예시</span>
              {PROMPTS.map((p) => (
                <button key={p} className="link" disabled={busy || !uid} onClick={() => send(p)} data-testid="example-prompt">{p}</button>
              ))}
            </div>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                send(input);
              }}
            >
              <textarea
                ref={composerRef}
                rows={2}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                    e.preventDefault();
                    send(input);
                  }
                }}
                placeholder={uid ? `${uid}의 심박에 대해 질문하세요…  ( / 키로 이동 )` : "대상을 먼저 선택하세요"}
                disabled={!uid}
                aria-label="질문 입력"
                name="question"
                maxLength={MAX_QUESTION}
                autoComplete="off"
                data-testid="composer"
              />
              {busy ? (
                <button type="button" className="act stop" onClick={() => abortRef.current?.abort()} aria-label="중단" data-testid="stop">중단 <kbd aria-hidden="true">Esc</kbd></button>
              ) : (
                <button type="submit" className="act" disabled={!input.trim() || !uid} aria-label="전송" data-testid="send">전송</button>
              )}
            </form>
            {input.length >= COUNT_SHOWN_FROM && (
              <div className={`count mono ${input.length >= MAX_QUESTION ? "full" : ""}`} role="status" data-testid="char-count">
                {input.length.toLocaleString("ko-KR")} / {MAX_QUESTION.toLocaleString("ko-KR")}자
                {input.length >= MAX_QUESTION && <span className="count-msg"> 최대 {MAX_QUESTION.toLocaleString("ko-KR")}자까지 입력할 수 있습니다. 질문을 줄여 주세요.</span>}
              </div>
            )}
            <div className="keys muted">Enter로 전송, Shift+Enter로 줄바꿈</div>
          </div>
        </aside>
      </main>

      <footer className="disclaimer">
        <p>본 서비스는 <b>의료기기가 아니며</b> 진단이나 치료를 대신하지 않습니다.</p>
        <p>웨어러블 심박 데이터에서 통계적으로 벗어난 구간만 알려 드립니다.</p>
        <p>증상이 있거나 걱정되면 의료진과 상담하고, 응급 상황에서는 바로 119에 연락하세요.</p>
      </footer>

      {/* 인쇄 전용 문서. 화면에서는 숨기고(@media print 에서만 표시), 표는 읽기 전용이라 DOM id/testid 중복을 만들지 않는다. */}
      <div className="print-only">
        <p className="print-title">심박 이상 징후 점검 리포트</p>
        <p className="mono">대상 {uid}, 데이터 기간 {fmtDateTimeY(domain[0])} ~ {fmtDateTimeY(domain[1])}</p>
        {lastDone && (
          <>
            <h2>질문</h2>
            <p>{lastDone.question}</p>
            {lastDone.escalation && (
              <div className="print-safety"><b>안전 안내</b> {lastDone.escalation.message || splitReport(lastDone.report).notice}</div>
            )}
            <h2>리포트</h2>
            <Markdown text={splitReport(lastDone.report).body} />
          </>
        )}
        <h2>발견된 이상 (최근 {range}일)</h2>
        <EventTable rows={rows} index={index} selectedId={null} />
        <p className="print-disc">본 문서는 의료기기가 생성한 진단이 아니며, 웨어러블 심박 데이터의 통계적 이상 징후를 참고용으로 정리한 것입니다. 증상이 있거나 우려되면 의료진과 상담하세요.</p>
      </div>
    </div>
  );
}

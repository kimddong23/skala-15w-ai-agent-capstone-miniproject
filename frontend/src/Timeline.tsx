/**
 * 타임라인 차트 — 직접 만든 SVG. 개인 기준선 밴드, 심박선(+구간 최소–최대), 수면/걸음 트랙, 심각도 밴드,
 * 크로스헤어·툴팁, 드래그 확대를 한 컴포넌트로 그린다.
 *
 * 책임: 주어진 시계열·이상·표시 구간(domain)을 그리고, 사용자 조작(밴드 선택, 확대)을 콜백으로 올린다.
 *   표시 구간·선택 상태는 부모가 소유한다(표·인용과 동기화해야 하므로 제어 컴포넌트).
 * 차트 라이브러리를 쓰지 않은 이유: 이 화면은 밴드·트랙·브러시를 한 좌표계에 겹쳐야 하는데, 범용 차트 라이브러리로는
 *   커스터마이징 비용이 직접 그리는 것보다 크고(번들 크기·테마 토큰 연동 포함) 의존성만 늘어난다.
 * 기록 없음: 서버는 빈 구간의 점을 보내지 않으므로, 인접 점 간격이 리샘플 주기의 1.5배를 넘는 곳을 빗금 영역으로 그리고 선을 끊는다.
 * 성능: 원시 점이 아니라 화면 폭에 맞춘 버킷(series.ts `bucketize`)을 그린다 — 30일(≈8.6k점)도 픽셀당 1~2개 요소.
 * 접근성: SVG 는 `role="img"` 요약만 제공하고, 같은 정보는 이벤트 로그 표(키보드·스크린 리더 가능)가 담는다.
 * E2E: `data-testid` = timeline(SVG) / timeline-hit(조작 영역).
 */
import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { Anomaly } from "./events";
import { typeLabel, SEV_LABEL } from "./labels";
import { bucketize, type Series } from "./series";
import type { Lead, Marker } from "./illness";
import { fmtDateTime, fmtDateTimeY, fmtHM, tsToMs } from "./time";

// 레이아웃 상수(px). 좌측 여백은 y축 눈금·트랙 라벨 폭이다.
const PAD = { l: 38, r: 10, t: 10 };
// 플롯 높이: 노트북(낮은 창)에서는 차트와 이벤트 표 머리가 한 화면에 들어오도록 줄인다.
const H_PLOT_TALL = 250;
const H_PLOT_SHORT = 190;
const SHORT_VIEWPORT = 860;
const H_SLEEP = 10;
const H_STEPS = 38;
const GAP = 6;
const H_AXIS = 20;

// 심각도 → 색 토큰. 색은 심각도 의미에만 쓴다.
const sevVar = { critical: "var(--crit)", warning: "var(--warn)", info: "var(--info)" } as const;
const HOUR = 3_600_000;
// x축 눈금 후보 간격(작은 것부터). 화면 폭에 눈금 라벨이 겹치지 않는 가장 촘촘한 간격을 고른다.
const TICKS = [HOUR, 3 * HOUR, 6 * HOUR, 12 * HOUR, 24 * HOUR, 48 * HOUR, 120 * HOUR, 168 * HOUR];

// 창 높이가 낮은지 추적한다(리사이즈 시 갱신).
function useShortViewport() {
  const [short, setShort] = useState(() => typeof innerHeight === "number" && innerHeight < SHORT_VIEWPORT);
  useEffect(() => {
    const h = () => setShort(innerHeight < SHORT_VIEWPORT);
    addEventListener("resize", h);
    return () => removeEventListener("resize", h);
  }, []);
  return short;
}

// 컨테이너 폭을 추적한다. 300px 하한: 극단적으로 좁은 레이아웃 전환 중에도 음수/0 폭 계산이 나오지 않게 한다.
function useWidth() {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(720);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(300, Math.floor(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

/** `Timeline` 의 입력. */
interface Props {
  /** 전체 로드 구간의 시계열(표시 구간은 `domain` 으로 자른다). */
  series: Series;
  /** 이상 목록(표시 구간 밖의 것은 그리지 않는다). */
  anomalies: Anomaly[];
  /** 이상 ID → 화면 번호(#). 밴드 위에 번호를 표기한다. */
  index: Map<string, number>;
  /** 표시 구간 `[시작, 끝]` (epoch ms). 확대 시 부모가 좁힌다. */
  domain: [number, number];
  /** 개인 기준선 `[lo, hi]` bpm. 없으면 밴드를 그리지 않는다. */
  baseline: { lo: number; hi: number } | null;
  /** 강조할 이상 ID. */
  selectedId: string | null;
  /** 밴드를 클릭하면 그 ID, 빈 곳을 클릭하면 null. */
  onSelect: (id: string | null) => void;
  /** 드래그로 구간을 고르면 그 구간(최소 20분)을 올린다. */
  onBrush: (d: [number, number]) => void;
  /** 더블클릭 시 확대 해제 요청. */
  onReset: () => void;
  /** 실제 사건(증상 시작·확진·회복) 마커. 표시 구간 밖의 것은 그리지 않는다. */
  markers?: Marker[];
  /** 증상 시작별 "첫 경고" — 경고 날짜와 증상 시작을 이어 붙이는 괄호와 문구. */
  leads?: Lead[];
  /** 서버 리샘플 주기(ms). 인접 점 간격이 이의 1.5배를 넘으면 "기록 없음"으로 보고 선을 끊는다. */
  stepMs: number;
  /** SVG 접근성 요약 문구. */
  label: string;
}

/** 타임라인 차트. */
export function Timeline({ series, anomalies, index, domain, baseline, selectedId, onSelect, onBrush, onReset, stepMs, label, markers = [], leads = [] }: Props) {
  const [wrap, W] = useWidth();
  const H_PLOT = useShortViewport() ? H_PLOT_SHORT : H_PLOT_TALL;
  const H_TOTAL = PAD.t + H_PLOT + GAP + H_SLEEP + 2 + H_STEPS + H_AXIS;
  const iw = W - PAD.l - PAD.r;
  const [t0, t1] = domain;
  const span = Math.max(1, t1 - t0);
  const [hover, setHover] = useState<{ x: number; bi: number; gap?: { t0: number; t1: number } } | null>(null);
  const hatchId = `hatch-${useId().replace(/[^a-zA-Z0-9]/g, "")}`;
  const [drag, setDrag] = useState<{ a: number; b: number } | null>(null);
  const dragRef = useRef<{ a: number; moved: boolean } | null>(null);

  const gapMs = stepMs * 1.5;
  const buckets = useMemo(() => bucketize(series, t0, t1, Math.max(40, Math.floor(iw / 2)), gapMs), [series, t0, t1, iw, gapMs]);

  // 기록 없음 구간: 인접 점 사이가 허용 간격을 넘는 곳과, 표시 구간의 앞뒤 가장자리에서 기록이 비는 곳.
  // 서버가 빈 구간의 점을 보내지 않으므로 이 판정이 없으면 센서 결측이 직선으로 이어져 측정된 것처럼 보인다.
  const gaps = useMemo(() => {
    const g: { t0: number; t1: number }[] = [];
    if (!buckets.length) return g;
    if (buckets[0].tStart - t0 > gapMs) g.push({ t0, t1: buckets[0].tStart });
    for (let i = 1; i < buckets.length; i++) if (buckets[i].gapBefore) g.push({ t0: buckets[i - 1].tEnd, t1: buckets[i].tStart });
    if (t1 - buckets[buckets.length - 1].tEnd > gapMs) g.push({ t0: buckets[buckets.length - 1].tEnd, t1 });
    return g;
  }, [buckets, t0, t1, gapMs]);

  const x = (t: number) => PAD.l + ((t - t0) / span) * iw;
  const tAt = (px: number) => t0 + ((px - PAD.l) / iw) * span;

  // y 범위: 데이터 범위에 정상 범위(55~105)를 항상 포함시켜, 평탄한 날도 기준선 대비 위치가 보이게 한다.
  let lo = Infinity;
  let hi = -Infinity;
  for (const b of buckets) {
    if (b.min != null) lo = Math.min(lo, b.min);
    if (b.max != null) hi = Math.max(hi, b.max);
  }
  if (!isFinite(lo)) (lo = 40), (hi = 120);
  lo = Math.max(20, Math.floor((Math.min(lo, 55) - 3) / 10) * 10);
  hi = Math.ceil((Math.max(hi, 105) + 3) / 10) * 10;
  const y = (v: number) => PAD.t + (1 - (v - lo) / (hi - lo)) * H_PLOT;
  const yGrid: number[] = [];
  for (let v = Math.ceil(lo / 20) * 20; v <= hi; v += 20) yGrid.push(v);

  // 심박선 + 범위 포락선. 결측(hr=null)에서 선을 끊는다 — 보간하면 "측정된 것처럼" 보여 센서 결측 이상을 가린다.
  const segs: typeof buckets[] = [];
  let cur: typeof buckets = [];
  for (const b of buckets) {
    if (b.gapBefore && cur.length) {
      segs.push(cur);
      cur = [];
    }
    if (b.hr == null) {
      if (cur.length) segs.push(cur);
      cur = [];
    } else cur.push(b);
  }
  if (cur.length) segs.push(cur);
  // 점 하나짜리 구간(양옆이 기록 없음)은 길이 0 선이라 안 보이므로, 아주 짧은 선분을 더해 둥근 끝으로 점처럼 보이게 한다.
  const line = segs
    .map((s) => s.map((b, i) => `${i ? "L" : "M"}${x(b.t).toFixed(1)} ${y(b.hr!).toFixed(1)}`).join("") + (s.length === 1 ? "h0.01" : ""))
    .join("");
  // 버킷이 원시 점 하나면 min=max 라 포락선이 의미 없다. 요약(n>1)된 구간에서만 그려 스파이크 손실을 보완한다.
  const showEnv = buckets.length > 0 && buckets[0].n > 1;
  const env = showEnv
    ? segs
        .filter((s) => s.length > 1)
        .map(
          (s) =>
            s.map((b, i) => `${i ? "L" : "M"}${x(b.t).toFixed(1)} ${y(b.max!).toFixed(1)}`).join("") +
            [...s].reverse().map((b) => `L${x(b.t).toFixed(1)} ${y(b.min!).toFixed(1)}`).join("") +
            "Z",
        )
        .join("")
    : "";

  // 수면 구간: 인접 버킷을 병합해 rect 수를 줄인다(1.5배 간격까지 이어 붙임 = 사이 결측 1개 허용).
  const sleepRuns: [number, number][] = [];
  for (const b of buckets) {
    if (!b.asleep) continue;
    const last = sleepRuns[sleepRuns.length - 1];
    if (last && b.t - last[1] <= (buckets[1] ? buckets[1].t - buckets[0].t : 1) * 1.5) last[1] = b.t;
    else sleepRuns.push([b.t, b.t]);
  }
  const yPlotBottom = PAD.t + H_PLOT;
  const ySleep = yPlotBottom + GAP;
  const ySteps = ySleep + H_SLEEP + 2;
  const stepMax = Math.max(1, ...buckets.map((b) => b.steps));
  const barW = Math.max(1, (iw / Math.max(1, buckets.length)) * 0.8);

  // x 눈금: 하루 미만 간격이면 자정에만 날짜를 함께 표기한다.
  const tickMs = TICKS.find((m) => span / m <= iw / 74) ?? TICKS[TICKS.length - 1];
  const ticks: number[] = [];
  for (let t = Math.ceil(t0 / tickMs) * tickMs; t <= t1; t += tickMs) ticks.push(t);
  const tickLabel = (t: number) => (tickMs < 24 * HOUR ? (t % (24 * HOUR) === 0 ? fmtDateTime(t).slice(0, 5) : fmtHM(t)) : fmtDateTime(t).slice(0, 5));

  const visible = anomalies.filter((a) => tsToMs(a.end) >= t0 && tsToMs(a.start) <= t1);
  // 밴드 판정에 ±5px 허용 오차를 둔다 — 5분짜리 이상은 7일 뷰에서 1px 도 안 돼 정확히 맞추기 어렵다.
  const hitAnomaly = (px: number) => {
    const tol = 5 / iw * span;
    const t = tAt(px);
    return visible.find((a) => t >= tsToMs(a.start) - tol && t <= tsToMs(a.end) + tol) ?? null;
  };

  // 포인터 x 를 SVG 로컬 좌표로 바꾸고 플롯 영역으로 제한한다(setPointerCapture 로 영역 밖에서도 이벤트가 온다).
  const localX = (e: React.PointerEvent) => {
    const r = (e.currentTarget as Element).closest("svg")!.getBoundingClientRect();
    return Math.min(Math.max(e.clientX - r.left, PAD.l), W - PAD.r);
  };
  const nearest = (px: number) => {
    const t = tAt(px);
    let bi = 0;
    let bd = Infinity;
    // 버킷 수가 화면 폭/2 이하(수백 개)라 선형 탐색으로 충분하다.
    for (let i = 0; i < buckets.length; i++) {
      const d = Math.abs(buckets[i].t - t);
      if (d < bd) ((bd = d), (bi = i));
    }
    return bi;
  };

  // 클릭과 드래그를 한 핸들러로 구분한다: 5px 미만 이동은 클릭(밴드 선택), 이상은 브러시(확대).
  const onDown = (e: React.PointerEvent<SVGRectElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    const px = localX(e);
    dragRef.current = { a: px, moved: false };
    setDrag({ a: px, b: px });
  };
  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const px = localX(e);
    if (dragRef.current) {
      if (Math.abs(px - dragRef.current.a) > 5) dragRef.current.moved = true;
      setDrag({ a: dragRef.current.a, b: px });
      setHover(null);
    } else if (buckets.length) {
      const t = tAt(px);
      setHover({ x: px, bi: nearest(px), gap: gaps.find((g) => t > g.t0 && t < g.t1) });
    }
  };
  const onUp = (e: React.PointerEvent<SVGRectElement>) => {
    const d = dragRef.current;
    dragRef.current = null;
    setDrag(null);
    if (!d) return;
    const px = localX(e);
    if (d.moved) {
      const [a, b] = [tAt(Math.min(d.a, px)), tAt(Math.max(d.a, px))];
      if (b - a >= 20 * 60_000) onBrush([Math.round(a), Math.round(b)]);
    } else {
      onSelect(hitAnomaly(px)?.anomaly_id ?? null);
    }
  };

  const hb = hover ? buckets[hover.bi] : null;
  const hoverAnom = hb ? visible.find((a) => hb.t >= tsToMs(a.start) && hb.t <= tsToMs(a.end)) : null;

  return (
    <div className="tl-wrap" ref={wrap}>
      <svg width={W} height={H_TOTAL} role="img" aria-label={label} className="tl" data-testid="timeline">
        {/* 개인 기준선 밴드 */}
        {baseline && (
          <g>
            <rect x={PAD.l} y={y(baseline.hi)} width={iw} height={Math.max(1, y(baseline.lo) - y(baseline.hi))} className="baseline" />
          </g>
        )}
        <defs>
          <pattern id={hatchId} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="6" className="hatch-line" />
          </pattern>
        </defs>
        {yGrid.map((v) => (
          <g key={v}>
            <line x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} className="gl" />
            <text x={PAD.l - 6} y={y(v) + 3.5} textAnchor="end" className="tick">{v}</text>
          </g>
        ))}
        {[60, 100].filter((v) => v > lo && v < hi).map((v) => (
          <line key={`r${v}`} x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} className="ref" />
        ))}
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={PAD.t} y2={ySteps + H_STEPS} className="gl v" />
            <text x={x(t)} y={H_TOTAL - 5} textAnchor="middle" className="tick">{tickLabel(t)}</text>
          </g>
        ))}

        {/* 기록 없음: 빗금 영역. 라벨은 폭이 충분할 때만(좁으면 호버 툴팁이 설명한다). */}
        {gaps.map((g) => {
          const gx = x(Math.max(g.t0, t0));
          const gw = Math.max(1, x(Math.min(g.t1, t1)) - gx);
          if (gw < 4) return null; // 선은 이미 끊겼다. 1~3px 틈은 빗금이 노이즈라 그리지 않는다.
          return (
            <g key={`gap${g.t0}`} className="gapzone" data-testid="gap-zone">
              <rect x={gx} y={PAD.t} width={gw} height={ySteps + H_STEPS - PAD.t} fill={`url(#${hatchId})`} />
              {gw >= 56 && <text x={gx + gw / 2} y={PAD.t + H_PLOT / 2} textAnchor="middle" className="gaplabel">기록 없음</text>}
            </g>
          );
        })}

        {/* 수면 음영(플롯 위) */}
        {sleepRuns.map(([a, b]) => (
          <rect key={`sp${a}`} x={x(a)} y={PAD.t} width={Math.max(2, x(b) - x(a))} height={H_PLOT} className="sleep-shade" />
        ))}

        {/* 심각도 밴드 */}
        {visible.map((a) => {
          const xs = x(Math.max(tsToMs(a.start), t0));
          const xe = x(Math.min(tsToMs(a.end), t1));
          const w = Math.max(3, xe - xs);
          const sel = a.anomaly_id === selectedId;
          return (
            <g key={a.anomaly_id} style={{ color: sevVar[a.severity] }}>
              <rect x={xs} y={PAD.t} width={w} height={H_PLOT} className={`band ${sel ? "sel" : ""}`} />
              <rect x={xs} y={PAD.t} width={w} height={3} fill="currentColor" />
              {sel && <rect x={xs} y={PAD.t} width={w} height={ySteps + H_STEPS - PAD.t} className="band-outline" />}
              {visible.length <= 14 && (
                <text x={xs + w / 2} y={PAD.t + 14} textAnchor="middle" className="bnum" style={{ fill: "currentColor" }}>{index.get(a.anomaly_id)}</text>
              )}
            </g>
          );
        })}

        {/* 실제 사건 마커: 세로선(종류별 선 모양)과 라벨. 색은 심각도 전용이라 쓰지 않는다.
            라벨은 세 줄로 번갈아 두고, 앞 라벨과 너무 가까우면(겹칠 만큼) 글자를 생략한다 — 선과 툴팁, 아래 "실제 사건" 버튼에는 그대로 남는다. */}
        {(() => {
          let lastX = -Infinity;
          let row = 0;
          return markers
            .filter((m) => m.t >= t0 && m.t <= t1)
            .sort((p, q) => p.t - q.t)
            .map((m) => {
              const mx = x(m.t);
              const right = mx > W - PAD.r - 130;
              const show = mx - lastX >= (right ? 230 : 120);
              if (show) lastX = mx;
              return (
                <g key={`${m.type}${m.t}`} className={`marker ${m.type}`} data-testid="event-marker" data-type={m.type}>
                  <line x1={mx} x2={mx} y1={PAD.t} y2={ySteps + H_STEPS} />
                  {show && <text x={right ? mx - 4 : mx + 4} y={PAD.t + 64 + (row++ % 3) * 14} textAnchor={right ? "end" : "start"}>{m.label}</text>}
                  <title>{`${m.label} (${m.date}) — 연구용 라벨, 진단 아님`}</title>
                </g>
              );
            });
        })()}
        {leads
          .filter((l) => Math.max(l.onset.t, l.alarmMs) >= t0 && Math.min(l.onset.t, l.alarmMs) <= t1)
          .map((l) => {
            const ax = x(Math.min(Math.max(t0, Math.min(l.alarmMs, l.onset.t)), t1));
            const ox = x(Math.min(Math.max(t0, Math.max(l.alarmMs, l.onset.t)), t1));
            const ly = PAD.t + 44;
            return (
              <g key={`lead${l.onset.t}`} className="lead" data-testid="lead-annotation">
                <path d={`M${ax} ${ly - 4}V${ly + 4}M${ax} ${ly}H${ox}M${ox} ${ly - 4}V${ly + 4}`} />
                <text x={(ax + ox) / 2} y={ly - 7} textAnchor="middle">{l.text}</text>
              </g>
            );
          })}

        {env && <path d={env} className="env" />}
        <path d={line} className="hr-line" />

        {/* 트랙: 수면 / 걸음 */}
        <rect x={PAD.l} y={ySleep} width={iw} height={H_SLEEP} className="lane" />
        {sleepRuns.map(([a, b]) => (
          <rect key={`s${a}`} x={x(a)} y={ySleep} width={Math.max(2, x(b) - x(a))} height={H_SLEEP} className="sleep" />
        ))}
        <rect x={PAD.l} y={ySteps} width={iw} height={H_STEPS} className="lane" />
        {buckets.map((b, i) =>
          b.steps > 0 ? (
            <rect key={i} x={x(b.t) - barW / 2} width={barW} y={ySteps + H_STEPS - Math.sqrt(b.steps / stepMax) * (H_STEPS - 2)} height={Math.sqrt(b.steps / stepMax) * (H_STEPS - 2)} className="steps" />
          ) : null,
        )}
        <text x={PAD.l - 6} y={ySleep + 8.5} textAnchor="end" className="tick">수면</text>
        <text x={PAD.l - 6} y={ySteps + H_STEPS - 3} textAnchor="end" className="tick">걸음</text>

        {drag && <rect x={Math.min(drag.a, drag.b)} y={PAD.t} width={Math.abs(drag.b - drag.a)} height={ySteps + H_STEPS - PAD.t} className="brush" />}
        {hover && hb && !drag && !hover.gap && (
          <g pointerEvents="none">
            <line x1={x(hb.t)} x2={x(hb.t)} y1={PAD.t} y2={ySteps + H_STEPS} className="cross" />
            {hb.hr != null && <circle cx={x(hb.t)} cy={y(hb.hr)} r={3.5} className="dot" />}
          </g>
        )}
        <rect
          x={PAD.l}
          y={PAD.t}
          width={iw}
          height={ySteps + H_STEPS - PAD.t}
          className="hit"
          data-testid="timeline-hit"
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerLeave={() => setHover(null)}
          onDoubleClick={onReset}
        />
      </svg>

      {hover && hover.gap && !drag && (
        <div className="tip" aria-hidden="true" style={{ left: Math.min(Math.max(hover.x, 96), W - 96) }}>
          <div className="tip-v mono">기록 없음</div>
          <div className="mono muted">{fmtDateTime(hover.gap.t0)} ~ {fmtDateTime(hover.gap.t1)}</div>
        </div>
      )}
      {hover && hb && !drag && !hover.gap && (
        <div className="tip" aria-hidden="true" style={{ left: Math.min(Math.max(hover.x, 96), W - 96) }}>
          <div className="mono tip-t">{fmtDateTimeY(hb.t)}{hb.n > 1 && <span className="muted"> ({hb.n}회 측정 평균)</span>}</div>
          <div className="tip-v mono">{hb.hr != null ? Math.round(hb.hr) : "—"}<small> bpm</small></div>
          {hb.n > 1 && hb.min != null && <div className="mono muted">최저 {Math.round(hb.min)}, 최고 {Math.round(hb.max!)}</div>}
          <div className="mono muted">걸음 {Math.round(hb.steps)}, {hb.asleep ? "수면 중" : "깨어 있음"}</div>
          {hoverAnom && (
            <div className="tip-a" style={{ color: sevVar[hoverAnom.severity] }}>
              #{index.get(hoverAnom.anomaly_id)} {typeLabel(hoverAnom.type)} · {SEV_LABEL[hoverAnom.severity]}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

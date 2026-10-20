/**
 * 이벤트 로그 표 — 탐지된 이상을 시간순으로 나열하고, 행 선택을 차트와 연동한다.
 *
 * 책임: 번호·시각·유형·지속·값 대비 기준선 Δ·심각도·근거의 표시. 선택/확대 같은 상태는 부모(App)가 소유한다.
 * 반응형: 좁은 폭에서는 CSS 가 표를 행 카드로 재배치한다(`.ev-inline` 근거가 유형 아래에 나온다).
 * 접근성: 행 전체 클릭은 편의일 뿐이고, 키보드/스크린 리더는 번호 버튼으로 같은 동작을 한다.
 * E2E: `data-testid="event-row"` + `data-anomaly-id`, 선택 행은 `aria-current="true"`.
 */
import type { Anomaly } from "./events";
import { CTX_LABEL, SEV_LABEL, typeLabel } from "./labels";
import { fmtDateTimeY, fmtDuration, tsToMs } from "./time";

interface Props {
  /** 표시할 이상(시작 시각 오름차순). */
  rows: Anomaly[];
  /** 이상 ID → 화면 번호(#). */
  index: Map<string, number>;
  /** 강조할 이상 ID. */
  selectedId: string | null;
  /** 행/번호 클릭 시 호출. 생략하면 읽기 전용(인쇄용)이 된다. */
  onFocus?: (a: Anomaly) => void;
  /** 개발자 보기: 서버가 준 `stats`(z 점수·임계값 등)를 근거 아래에 함께 보인다. */
  dev?: boolean;
}

// 서버 통계 값을 `키=값` 으로 한 줄에 나열한다(개발자 보기 전용, 내부 식별자라 번역 대상이 아니다).
function Stats({ s }: { s: Record<string, unknown> }) {
  return (
    <div className="dev-stats mono" translate="no" data-testid="event-stats">
      {Object.entries(s).map(([k, v]) => `${k}=${typeof v === "number" ? Math.round(v * 100) / 100 : String(v)}`).join(", ")}
    </div>
  );
}

/** 이벤트 로그 표. */
export function EventTable({ rows, index, selectedId, onFocus, dev = false }: Props) {
  return (
    <div className="tbl-wrap">
      <table className="evt">
        <thead>
          <tr>
            <th className="num">#</th>
            <th>시작 시각</th>
            <th>종류</th>
            <th className="r">지속 시간</th>
            <th className="r">측정값 (평소 대비)</th>
            <th>심각도</th>
            <th className="ev-col">근거</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((a) => {
            const d = a.baseline != null ? a.value - a.baseline : null;
            const sel = a.anomaly_id === selectedId;
            return (
              <tr
                key={a.anomaly_id}
                // 읽기 전용(인쇄용) 표는 id/testid 를 달지 않는다 — 화면 표와 중복되면 스크롤 이동·E2E 선택자가 모호해진다.
                id={onFocus ? `ev-${a.anomaly_id}` : undefined}
                className={sel ? "sel" : ""}
                aria-current={sel ? "true" : undefined}
                data-testid={onFocus ? "event-row" : undefined}
                data-anomaly-id={a.anomaly_id}
                data-severity={a.severity}
                onClick={() => onFocus?.(a)}
              >
                <td className="num mono">
                  {onFocus ? (
                    // 행 onClick 과 이중 호출되지 않도록 버블링을 막는다.
                    <button type="button" className="rowbtn" onClick={(e) => (e.stopPropagation(), onFocus(a))} aria-label={`이상 ${index.get(a.anomaly_id)}번 차트에서 보기`}>
                      {index.get(a.anomaly_id)}
                    </button>
                  ) : (
                    index.get(a.anomaly_id)
                  )}
                </td>
                <td className="mono nowrap">{fmtDateTimeY(tsToMs(a.start))}</td>
                <td>
                  {typeLabel(a.type)} <span className="muted ctx">{CTX_LABEL[a.context] ?? a.context}</span>
                  <div className="ev-inline muted">{a.evidence}</div>
                  {dev && a.stats && <Stats s={a.stats} />}
                </td>
                <td className="r mono nowrap">{fmtDuration(a.duration_min)}</td>
                <td className="r mono nowrap">
                  {a.type === "sensor_gap" ? "—" : <>{Math.round(a.value)}{d != null && <span className="muted"> ({d >= 0 ? "+" : "−"}{Math.abs(Math.round(d))})</span>}</>}
                </td>
                <td><span className={`sev ${a.severity}`}><i />{SEV_LABEL[a.severity]}</span></td>
                <td className="ev-col muted">{a.evidence}{dev && a.stats && <Stats s={a.stats} />}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

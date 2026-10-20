/**
 * 실행 기록(run) 카드 — 질문 1건에 대한 에이전트 실행을 "진행 과정 → 안전 안내 → 리포트 → 검증" 순서로 보여준다.
 *
 * 책임: `Turn` 상태를 표시로 옮기는 것. 단계별 상태·소요 계산은 trace.ts, 리포트 고정 문구 분리는 report.ts,
 *   각주 연결은 cite.ts/Markdown.tsx 에 위임하고, 여기서는 그 결과를 배치한다.
 * 경계: 상태를 소유하지 않는다(순수 표시). 인용 클릭 같은 부수 효과는 `cite` 콜백으로 App 에 올린다.
 * 접근성: 안전 띠와 오류는 `role="alert"`, 리포트는 `aria-live` 영역(작성 중에는 `aria-busy` 로 부분 낭독을 막는다).
 * 문구: 화면에는 쉬운 말만 쓰고, 노드 코드명·thread_id·서버 사유 같은 내부 식별자는 `dev`(개발자 보기)에서만 보인다
 *   (내부 식별자·전문 용어는 기본 화면에 두지 않는다).
 * 아이콘: Bootstrap Icons(react-bootstrap-icons)만, 16px 한 크기·한 굵기. 상태 표시(체크/엑스/경고/대시/원/반복)와 펼침 화살표 같은
 *   기능 아이콘에만 쓰고 장식에는 쓰지 않는다.
 * E2E: `data-testid` = run / trace-step(+data-node, data-status) / safety-band / report / validation-status.
 */
import { useState } from "react";
import { ArrowRepeat, CheckLg, ChevronRight, Circle, Dash, ExclamationTriangle, FileEarmarkText, Search, XLg } from "react-bootstrap-icons";
import { formatUsage } from "./llm";
import { Markdown, type CiteCtx } from "./Markdown";
import { splitReport } from "./report";
import { buildTrace, type StepStatus, type SubStep, type TraceStep } from "./trace";
import type { Turn } from "./turn";

/**
 * 소요 시간 표기. 1초 미만은 ms, 이상은 초.
 * @param live true 이면 소수 1자리 — 진행 중 카운터가 매 틱마다 자릿수가 바뀌어 흔들리는 것을 줄인다.
 */
export const fmtMs = (ms: number, live = false) => (ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(live ? 1 : 2)} s`);

const STATUS_TEXT: Record<StepStatus, string> = {
  pending: "대기",
  running: "진행 중",
  done: "완료",
  failed: "실패",
  skipped: "건너뜀",
  alert: "안내 추가",
};

/**
 * 상태 글리프(Bootstrap Icons 16px). 색에만 의존하지 않도록 상태마다 모양이 다르다
 * (대기=빈 원, 진행=회전 화살표, 완료=체크, 실패=X, 안내 추가=경고 삼각형, 건너뜀=줄).
 * 장식이므로 스크린 리더에서는 숨기고, 상태는 인접 텍스트(또는 `.sr` 보조 텍스트)가 전달한다.
 */
export function Glyph({ s }: { s: StepStatus }) {
  const common = { size: 16, "aria-hidden": true, className: `glyph ${s}` } as const;
  switch (s) {
    case "pending": return <Circle {...common} />;
    case "running": return <ArrowRepeat {...common} />;
    case "done": return <CheckLg {...common} />;
    case "failed": return <XLg {...common} />;
    case "alert": return <ExclamationTriangle {...common} />;
    case "skipped": return <Dash {...common} />;
  }
}

// 재조회(재계획) 라운드가 바뀌는 지점에 구분선을 넣어, 같은 작업이 다시 실행된 것을 한눈에 구별하게 한다.
function Subs({ subs }: { subs: SubStep[] }) {
  const rows: React.ReactNode[] = [];
  let round = -1;
  for (const s of subs) {
    if (s.round !== round && s.round > 0) rows.push(<li key={`r${s.round}`} className="round">재조회 라운드 {s.round}</li>);
    round = s.round;
    rows.push(
      <li key={s.key} className={`sub ${s.status}`} data-testid="trace-sub" data-status={s.status}>
        <Glyph s={s.status} />
        <span className="sr">{STATUS_TEXT[s.status]}: </span>
        {/* 가이드라인 검색·점검 절차 불러오기는 종류를 아이콘으로도 알려 준다(Bootstrap Icons 16px, 장식이라 읽지 않는다). */}
        {s.kind === "retrieval" && <Search size={16} aria-hidden="true" className="kind" />}
        {s.kind === "prompt" && <FileEarmarkText size={16} aria-hidden="true" className="kind" />}
        <span className="sub-label">{s.label}</span>
        {!!s.retries && <span className="retry" data-testid="retry-note">재시도 {s.retries}회</span>}
        {s.ms != null && <span className={`mono ms ${s.status === "running" ? "live" : ""}`}>{fmtMs(s.ms, s.status === "running")}</span>}
        {s.note && <span className="sub-note">{s.note}</span>}
      </li>,
    );
  }
  return <ul className="subs" aria-label="하위 작업">{rows}</ul>;
}

function Step({ s, dev }: { s: TraceStep; dev: boolean }) {
  return (
    <li className={`step ${s.status} ${s.tone ?? ""}`} data-testid="trace-step" data-node={s.node} data-status={s.status}>
      <Glyph s={s.status} />
      <div className="step-body">
        <div className="step-head">
          <span className="step-label">{s.label}</span>
          {dev && <code className="node" translate="no">{s.node}</code>}
          <span className="step-st">{STATUS_TEXT[s.status]}</span>
          <span className={`mono ms ${s.status === "running" ? "live" : ""}`}>{s.ms != null ? fmtMs(s.ms, s.status === "running") : ""}</span>
        </div>
        {s.lines.map((l, i) => <div key={i} className="step-line">{l}</div>)}
        {dev && s.devLines.map((l, i) => <div key={`d${i}`} className="step-line dev" translate="no">{l}</div>)}
        {s.subs.length > 0 && <Subs subs={s.subs} />}
      </div>
    </li>
  );
}

const RESULT_LINE: Record<string, string> = {
  approved: "보호자 알림을 보냈습니다(시뮬레이션 — 실제로는 발송되지 않습니다).",
  denied: "보호자 알림을 보내지 않았습니다.",
  conflict: "이 요청은 이미 처리됐거나 새 요청으로 바뀌었습니다.",
  error: "처리하지 못했습니다. 잠시 후 다시 눌러 주세요.",
};

/**
 * 보호자 알림 승인 카드(HITL). 심각한 이상이 있을 때 사람이 알림 발송을 결정한다.
 * 위 안전 안내(119·의료진 상담)는 승인 여부와 무관하게 항상 남는다는 것을 카드 안에서도 말해 준다.
 * 결정하면 버튼은 사라지고 결과 한 줄이 남는다. 전송 실패(error)만 다시 누를 수 있다.
 */
function ApprovalCard({ a, stale, onApprove }: { a: NonNullable<Turn["approval"]>; stale: boolean; onApprove: (a: NonNullable<Turn["approval"]>, approved: boolean) => void }) {
  // 같은 대화에서 더 새로운 승인 요청이 왔으면(stale) 이 카드는 결정할 수 없다. 결과가 이미 난 카드는 그대로 둔다.
  const open = a.status === "pending" || a.status === "error";
  const decidable = open && !stale;
  const status = open && stale ? "conflict" : a.status;
  return (
    <section className="approval" aria-label="보호자 알림 승인" data-testid="approval-card" data-status={status}>
      <h4>보호자 알림을 보낼까요?</h4>
      <p className="approval-why">{a.reason}</p>
      <blockquote className="approval-preview" data-testid="approval-preview">{a.message_preview}</blockquote>
      <p className="muted">승인하지 않아도 위의 안전 안내는 그대로 남습니다.</p>
      {decidable && (
        <div className="approval-actions">
          <button className="act" onClick={() => onApprove(a, true)} data-testid="approve">승인</button>
          <button className="textbtn" onClick={() => onApprove(a, false)} data-testid="deny">거절</button>
        </div>
      )}
      {a.status === "sending" && <p role="status" className="approval-result">처리 중…</p>}
      {RESULT_LINE[status] && (
        <p role="status" className={`approval-result ${status}`} data-testid="approval-result">
          {RESULT_LINE[status]}
        </p>
      )}
    </section>
  );
}

/** `Run` 의 입력. */
interface Props {
  /** 표시할 질문 상태. */
  turn: Turn;
  /** 화면 표시용 순번(1부터). */
  n: number;
  /** 이 질문이 쓴 데이터 기준 시각(그 참가자의 마지막 기록). "최근 N일" 이 이 시각부터 거꾸로 센 기간임을 보여준다. */
  anchor?: string;
  /** 보호자 알림 승인/거절을 보낸다(카드의 thread_id·approval_id 를 함께 넘긴다). 결과 반영(리포트 교체)은 호출한 쪽이 한다. */
  onApprove: (a: NonNullable<Turn["approval"]>, approved: boolean) => void;
  /** 같은 대화에 더 새로운 승인 요청이 있어 이 카드의 승인 요청이 낡았는지. */
  approvalStale?: boolean;
  /** 개발자 보기: 노드 코드명, thread_id, 서버 사유 같은 내부 식별자를 함께 보인다. */
  dev: boolean;
  /** 최신 실행 여부. 실행 중일 때만 트레이스를 자동으로 펼친다. */
  open: boolean;
  /** 질문 전송 후 현재 경과 ms. 실행 중 단계의 카운터 계산에 쓴다. */
  now: number;
  /** 리포트 각주 연결용 컨텍스트(`anomalies` 는 이 실행의 보고로 덮어쓴다). */
  cite: CiteCtx;
}

/** 실행 기록 카드 1개. */
export function Run({ turn: t, n, anchor, dev, onApprove, approvalStale = false, open, now, cite }: Props) {
  const running = t.status === "running";
  // 실행 중에는 펼쳐 보이고, 끝나면 접어 리포트를 앞세운다. 사용자가 직접 토글하면 그 선택을 유지한다.
  const [manual, setManual] = useState<boolean | null>(null);
  const traceOpen = manual ?? (running && open);
  const steps = buildTrace(t, now);
  const lastVal = t.validations[t.validations.length - 1];
  const statusText = { running: "실행 중", done: "완료", error: "오류", aborted: "중단됨" }[t.status];
  // 각주 후보는 "이 실행이 보고한" 이상으로 한정한다(다른 실행/기간의 이벤트로 오인용되지 않도록).
  const withCite: CiteCtx = { ...cite, anomalies: t.anomalies?.anomalies ?? [] };
  const parts = splitReport(t.report);
  // 에스컬레이션 문구는 전폭 띠 하나로만 보여준다(본문 첫 단락은 중복이라 report.ts 가 제거).
  // 이벤트 없이 본문에만 있는 경우에는 그 문구로 띠를 만든다.
  const escMsg = t.escalation ? t.escalation.message : parts.notice;

  return (
    <article className={`run ${t.status}`} aria-label={`실행 ${n}`} data-run={t.id} data-testid="run" data-status={t.status}>
      <header className="run-head">
        <span className="run-id">질문 {n}</span>
        <h3>{t.question}</h3>
        <span className={`run-st ${t.status}`}>{statusText}</span>
        <span className="mono ms">{fmtMs(running ? now : t.elapsedMs, running)}</span>
        {anchor && <span className="run-anchor mono" data-testid="run-anchor">기준 {anchor}</span>}
      </header>

      {(t.escalation || parts.notice) && (
        <div className="safety" role="alert" data-testid="safety-band">
          <div className="safety-label"><ExclamationTriangle size={20} aria-hidden="true" />안전 안내</div>
          <p>{escMsg || parts.notice || "심각한 이상이 발견되었습니다. 증상이 있다면 바로 병원이나 119에 연락하세요."}</p>
          {dev && t.escalation?.reason && <div className="safety-why mono" translate="no">{t.escalation.reason}</div>}
        </div>
      )}

      {t.approval && <ApprovalCard a={t.approval} stale={approvalStale} onApprove={onApprove} />}

      <details
        className="trace-box"
        open={traceOpen}
        // 프로그램이 open 을 바꿀 때도 toggle 이 발생한다. 현재 계산값과 다를 때만 "사용자 선택" 으로 기록한다.
        onToggle={(e) => {
          if (e.currentTarget.open !== traceOpen) setManual(e.currentTarget.open);
        }}
      >
        <summary>
          <ChevronRight size={16} aria-hidden="true" className="chev" />
          진행 과정{" "}
          <span className="muted">
            {steps.length
              ? `${steps.length}단계 · ${steps.filter((x) => x.status === "done" || x.status === "alert").length}개 완료${steps.some((x) => x.status === "skipped") ? ` · ${steps.filter((x) => x.status === "skipped").length}개 건너뜀` : ""}`
              : ""}
          </span>
        </summary>
        {steps.length ? (
          <ol className="trace" aria-label="진행 과정 단계">{steps.map((s) => <Step key={s.node} s={s} dev={dev} />)}</ol>
        ) : (
          <div className="skel" aria-busy="true"><i /><i /><i /></div>
        )}
      </details>

      {(t.report || running) && (
        <section className="doc" aria-label="리포트" aria-live="polite" aria-busy={running} data-testid="report">
          <div className="doc-label">리포트{running && <span className="muted"> · 작성 중</span>}</div>
          {t.report ? (
            <>
              <Markdown text={parts.body} cite={withCite} />
              {running && <span className="caret" aria-hidden />}
              {/* 면책은 화면 하단 고정 문구가 한 번 안내하므로 리포트 안에서는 반복하지 않는다. 인쇄본에는 별도로 들어간다. */}
            </>
          ) : (
            <div className="skel"><i /><i /><i /></div>
          )}
        </section>
      )}

      {lastVal && (
        <div className={`vline ${lastVal.passed ? "ok" : "bad"}`} role="status" data-testid="validation-status" data-passed={lastVal.passed} data-degraded={!!lastVal.degraded}>
          <b>{lastVal.passed ? "자동 검증 통과" : `자동 검증에서 문제 ${lastVal.issues.length}건`}</b>
          <span className="muted">{lastVal.attempt}번째 시도</span>
          {lastVal.degraded && <b className="degraded">대체 문안 사용</b>}
          {!lastVal.passed && <span>{lastVal.issues.join(", ")}</span>}
        </div>
      )}
      {dev && formatUsage(t.usage) && (
        <div className="usage mono" data-testid="usage-dev" translate="no">
          {formatUsage(t.usage)}
        </div>
      )}
      {t.status === "error" && (
        <div className="errline" role="alert">
          실행 중 오류가 발생했습니다. 잠시 후 다시 질문해 주세요.
          {dev && <span className="mono" translate="no"> {t.error}</span>}
        </div>
      )}
      {t.status === "aborted" && <div className="muted abort">중단했습니다.</div>}
    </article>
  );
}

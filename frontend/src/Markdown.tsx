/**
 * 리포트 렌더러 — 필요한 만큼만 지원하는 최소 마크다운과 이벤트 각주.
 *
 * 지원: 제목(#~###), 목록(-, *, 1.), 인용(>), **굵게**, `코드`, 문단. 그 외 문법은 문단 텍스트로 남는다.
 * 보안: 문자열을 React 노드로만 변환하고 HTML 을 해석하지 않는다(LLM 출력이 DOM 으로 주입되지 않도록).
 *   마크다운 라이브러리를 쓰지 않은 이유도 같다 — 리포트 문법이 고정돼 있고 의존성·공격 표면을 늘릴 이유가 없다.
 * 각주: `cite` 가 주어지면 탐지된 이상과 일치하는 수치 뒤에 위첨자 버튼을 붙인다(매칭 규칙은 cite.ts).
 */
import type { ReactNode } from "react";
import { splitCitations } from "./cite";
import type { Anomaly } from "./events";

/** 각주 연결에 필요한 컨텍스트. */
export interface CiteCtx {
  /** 인용 후보(이 실행이 보고한 이상). */
  anomalies: Anomaly[];
  /** 이상 → 이벤트 로그의 번호(#). 로그에 없으면 null 을 돌려주고, 그 경우 각주를 생략한다(가리킬 곳이 없으므로). */
  indexOf: (a: Anomaly) => number | null;
  /**
   * 가리키는 이상이 지금 조회 기간 밖이면 true. 이 경우 번호는 클릭 후 전환될 기간의 번호라 지금 표의 번호와
   * 겹칠 수 있어, 점선 테두리와 안내 문구로 구별한다.
   */
  isOutside: (a: Anomaly) => boolean;
  /** 각주 클릭 시 호출(차트·표에서 해당 이벤트로 이동). */
  onCite: (a: Anomaly) => void;
}

// 텍스트 조각 하나에 각주를 붙인다. `seen` 은 같은 줄 안에서 이미 각주가 붙은 이벤트(중복 방지).
function cited(text: string, ctx: string, key: string, seen: Set<string>, cite?: CiteCtx): ReactNode[] {
  if (!cite) return [text];
  return splitCitations(text, ctx, cite.anomalies, seen).flatMap((p, i): ReactNode[] => {
    if (!("anomaly" in p)) return [p.text];
    const n = cite.indexOf(p.anomaly);
    if (n == null) return [p.text];
    const far = cite.isOutside(p.anomaly);
    return [
      p.text,
      <sup key={`${key}c${i}`}>
        <button type="button" className={far ? "cite far" : "cite"} data-testid="citation" data-anomaly-id={p.anomaly.anomaly_id} onClick={() => cite.onCite(p.anomaly)} title={far ? `이상 ${n}번 (조회 기간을 넓혀 이동)` : `이상 ${n}번으로 이동`} aria-label={far ? `이상 ${n}번 근거 보기, 조회 기간이 바뀝니다` : `이상 ${n}번 근거 보기`}>
          {n}
        </button>
      </sup>,
    ];
  });
}

// 한 줄 안의 굵게/코드/일반 텍스트를 노드로 바꾼다. 각주 매칭 문맥(ctx)은 굵게 경계와 무관하게 "줄 전체"다.
function inline(s: string, key: string, cite?: CiteCtx): ReactNode[] {
  const seen = new Set<string>();
  return s.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((p, i): ReactNode => {
    const k = `${key}${i}`;
    if (p.startsWith("**") && p.endsWith("**") && p.length > 4) return <strong key={k}>{cited(p.slice(2, -2), s, k, seen, cite)}</strong>;
    if (p.startsWith("`") && p.endsWith("`") && p.length > 2) return <code key={k}>{p.slice(1, -1)}</code>;
    return <span key={k}>{cited(p, s, k, seen, cite)}</span>;
  });
}

/**
 * 마크다운 텍스트를 렌더한다.
 * @param text 렌더할 텍스트(스트리밍 중의 부분 텍스트 허용).
 * @param cite 생략하면 각주 없이 렌더(인쇄 등).
 */
export function Markdown({ text, cite }: { text: string; cite?: CiteCtx }) {
  const out: ReactNode[] = [];
  const lines = text.split("\n");
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const k = `b${i}`;
    let m: RegExpMatchArray | null;
    if (!line.trim()) {
      i++;
    } else if ((m = line.match(/^(#{1,3})\s+(.*)$/))) {
      const Tag = (`h${m[1].length + 2}`) as "h3" | "h4" | "h5";
      out.push(<Tag key={k}>{inline(m[2], k, cite)}</Tag>);
      i++;
    } else if (/^\s*([-*]|\d+\.)\s+/.test(line)) {
      const ordered = /^\s*\d+\./.test(line);
      const items: ReactNode[] = [];
      while (i < lines.length && /^\s*([-*]|\d+\.)\s+/.test(lines[i])) {
        items.push(<li key={`${k}-${i}`}>{inline(lines[i].replace(/^\s*([-*]|\d+\.)\s+/, ""), `${k}-${i}`, cite)}</li>);
        i++;
      }
      out.push(ordered ? <ol key={k}>{items}</ol> : <ul key={k}>{items}</ul>);
    } else if (line.startsWith(">")) {
      const buf: string[] = [];
      while (i < lines.length && lines[i].startsWith(">")) buf.push(lines[i++].replace(/^>\s?/, ""));
      // 인용문은 면책 등 고정 문구에 쓰이므로 각주 대상에서 제외한다.
      out.push(<blockquote key={k}>{inline(buf.join(" "), k)}</blockquote>);
    } else {
      const buf: string[] = [];
      while (i < lines.length && lines[i].trim() && !/^(#{1,3}\s|\s*([-*]|\d+\.)\s|>)/.test(lines[i])) buf.push(lines[i++]);
      out.push(<p key={k}>{inline(buf.join(" "), k, cite)}</p>);
    }
  }
  return <div className="md">{out}</div>;
}

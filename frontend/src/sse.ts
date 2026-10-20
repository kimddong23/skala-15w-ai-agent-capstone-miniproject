/**
 * Custom Transport — `POST /chat/stream` 의 SSE 를 직접 읽어 검증된 `ChatEvent` 스트림으로 바꾼다.
 *
 * 책임: ① 네트워크 청크 경계와 무관하게 SSE 프레임을 복원(`createSseParser`),
 *       ② 프레임을 zod 로 검증(`parseEvent`), ③ fetch 스트림을 비동기 이터레이터로 노출(`streamChat`).
 * 경계: 상태 전이는 turn.ts, 스키마는 events.ts 가 소유한다. 이 모듈은 UI/React 를 모른다.
 * 설계: SSE와 자체 포맷을 해석하는 Custom Transport. Vercel AI SDK `useChat` 은
 *   자체 스트림 포맷을 기대하므로 쓰지 않는다. EventSource 는 POST/헤더를 지원하지 않아 fetch 로 읽는다.
 */
import { ChatEvent } from "./events";

/** 파싱된 SSE 프레임(아직 검증 전). */
export interface SseFrame {
  /** `event:` 필드. 없으면 SSE 기본값 "message". */
  event: string;
  /** `data:` 라인을 개행으로 이은 원문. */
  data: string;
}

/**
 * 증분 SSE 파서를 만든다. 프레임은 빈 줄(`\n\n`, `\r\n\r\n`, `\r\r`)로 끝난다.
 *
 * @returns `push(chunk)` — 청크를 먹이고 완성된 프레임만 반환(미완성 꼬리는 내부 버퍼에 보관).
 *          `flush()` — 스트림 종료 시 마지막 빈 줄 없이 끝난 잔여 프레임을 회수.
 * @remarks `:` 로 시작하는 주석 줄(keep-alive)과 `data` 없는 프레임은 버린다.
 */
export function createSseParser() {
  let buf = "";
  return {
    push(chunk: string): SseFrame[] {
      buf += chunk;
      const frames: SseFrame[] = [];
      for (;;) {
        const m = /\r\n\r\n|\n\n|\r\r/.exec(buf);
        if (!m) break;
        const raw = buf.slice(0, m.index);
        buf = buf.slice(m.index + m[0].length);
        const f = parseFrame(raw);
        if (f) frames.push(f);
      }
      return frames;
    },
    flush(): SseFrame[] {
      const f = parseFrame(buf);
      buf = "";
      return f ? [f] : [];
    },
  };
}

function parseFrame(raw: string): SseFrame | null {
  let event = "message";
  const data: string[] = [];
  for (const line of raw.split(/\r\n|\n|\r/)) {
    if (!line || line.startsWith(":")) continue;
    const i = line.indexOf(":");
    const field = i < 0 ? line : line.slice(0, i);
    // SSE 규격: 값 앞의 공백 하나만 제거한다.
    const value = i < 0 ? "" : line.slice(i + 1).replace(/^ /, "");
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }
  return data.length ? { event, data: data.join("\n") } : null;
}

/**
 * 프레임을 검증된 `ChatEvent` 로 변환한다.
 *
 * @throws JSON 이 깨졌거나 스키마(`ChatEvent`)에 맞지 않을 때, 또는 `event:` 헤더와 `data.type` 이 다를 때.
 *   호출부(`streamChat`)는 이 예외를 잡아 해당 프레임만 건너뛴다.
 */
export function parseEvent(frame: SseFrame): ChatEvent {
  const ev = ChatEvent.parse(JSON.parse(frame.data));
  // 헤더/본문 불일치는 프록시·서버 버그의 신호라 조용히 받아들이지 않는다.
  if (frame.event !== "message" && frame.event !== ev.type) {
    throw new Error(`event 헤더(${frame.event})와 data.type(${ev.type}) 불일치`);
  }
  return ev;
}

/** `POST /chat/stream` 요청 본문. */
export interface StreamBody {
  message: string;
  user_id: string | null;
  /** 같은 thread 의 후속 질문이면 이전 `done.thread_id`. */
  thread_id: string | null;
}

/** API 기준 주소. 기본은 개발 서버 프록시(`/api` → :8100)이고 `VITE_API_BASE` 로 직접 지정할 수 있다. */
export const API_BASE: string = import.meta.env.VITE_API_BASE ?? "/api";

/**
 * 채팅 이벤트를 도착 순서대로 내보내는 비동기 제너레이터.
 *
 * @param body 요청 본문.
 * @param signal 중단 신호. abort 되면 fetch/reader 가 AbortError 로 끝난다(정상 중단 경로).
 * @throws HTTP 오류 또는 네트워크 오류. 형식이 깨진 개별 프레임은 던지지 않고 콘솔 경고 후 건너뛴다
 *   (한 프레임의 결함이 진행 중인 실행 전체를 죽이지 않게 하려는 선택).
 */
export async function* streamChat(body: StreamBody, signal: AbortSignal): AsyncGenerator<ChatEvent> {
  const res = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(`서버 응답 오류 (HTTP ${res.status})`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  const parser = createSseParser();
  const emit = function* (frames: SseFrame[]) {
    for (const f of frames) {
      try {
        yield parseEvent(f);
      } catch (e) {
        console.warn("[sse] 잘못된 이벤트 무시:", f, e);
      }
    }
  };
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      // stream: true — 멀티바이트(한글) 문자가 청크 경계에서 잘려도 깨지지 않게 한다.
      yield* emit(parser.push(decoder.decode(value, { stream: true })));
    }
    yield* emit(parser.push(decoder.decode()));
    yield* emit(parser.flush());
  } finally {
    // 소비자가 중간에 이터레이션을 끝내도 연결을 즉시 닫는다.
    reader.cancel().catch(() => {});
  }
}

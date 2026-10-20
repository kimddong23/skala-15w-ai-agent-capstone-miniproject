# Vital Sentinel — 프론트엔드

Vite + React + TypeScript. 계기판형 콘솔: 상태 스트립 · 타임라인 · 이벤트 로그 · 에이전트 실행 기록(run trace).

```bash
npm install
npm run dev          # http://localhost:8600  (백엔드 http://localhost:8100, /api 프록시)
npm run dev:mock     # 백엔드 없이 기록된 이벤트 재생(VITE_MOCK=1)
npm test             # SSE 파서 / zod / 리듀서 / 각주 매칭 / 시계열 / 트레이스 (vitest)
npm run build        # tsc --noEmit + vite build
```

- **포트 기본값**: 프론트 8600, 백엔드 8100. 바꾸려면:
  - 백엔드 주소만 바꿀 때 `VITE_PROXY_TARGET=http://localhost:8110 npm run dev` (dev·preview 프록시 대상).
  - `VITE_API_BASE=http://localhost:8100 npm run dev` 는 프록시 대신 백엔드를 직접 호출한다(CORS 는 백엔드가 8600·5173 을 허용, `CORS_ORIGINS` 로 변경).
  - 프론트 포트는 `npm run dev -- --port 8610`. 저장소 루트의 `sh/run_frontend.sh` 와 `docker compose` 는 `FRONTEND_PORT`/`BACKEND_PORT` 환경변수를 쓴다.
- 폰트(IBM Plex Sans KR / Plex Mono)는 `@fontsource`, 아이콘은 Bootstrap Icons(`react-bootstrap-icons`) 한 세트 — 오프라인 동작, CDN 없음.
- **데이터셋**: 기본은 FitBit(개발용 공개 데이터). Stanford COVID-19 연구 데이터(외부 검증, 로컬 전용)가 서버에 있으면 헤더 아래에 데이터셋 선택이 나타나고, 없으면 숨는다. 두 데이터셋의 참가자는 한 목록에 섞지 않는다.
  Stanford 참가자는 실제 사건 라벨(`GET /users/{id}/events`: 증상 시작·확진·회복)이 있으면 차트에 세로선으로 그리고, 증상 시작의 첫 경고를 서버가 미리 계산해 준 `first_warning` 으로 표시한다("증상 N일 전/당일/N일 후 첫 경고", "−7\~+3일 안에는 경고가 없었습니다", "평가할 수 없습니다", 필드가 없으면 생략). 보는 구간과 무관한 값이라 화면에서 이상 목록으로 다시 계산하지 않는다. 사건이 마지막 30일 밖이면 "실제 사건" 버튼으로 그 주변 30일 구간(`end` 파라미터)으로 옮겨 본다. 연구용 라벨이라 진단이 아니며 날짜는 개인정보 보호를 위해 이동된 값이다.
- 데모/스크린샷용 쿼리: `?theme=light|dark`, `?ask=0|1|2|<문장>` (로드 직후 예시 질문 자동 전송), `?dataset=stanford&user=S001&range=30&anchor=2020-03-18T00:00:00` (데이터셋·참가자·기간·구간 끝).
- 키보드: `Enter` 전송 · `Shift+Enter` 줄바꿈 · `Esc` 중단(실행 중) / 확대 해제 · `/` 입력창 포커스. 차트는 드래그 확대, 더블클릭 해제, 밴드 클릭 = 이벤트 선택.

## 구조
| 파일 | 역할 |
|---|---|
| `events.ts` · `sse.ts` | SSE 스키마(zod, CONTRACT 동일) · Custom Transport(POST → ReadableStream → 프레임 분리 → 검증, AbortController) |
| `turn.ts` | 이벤트 리듀서. 이벤트 도착 시각(`log`)을 함께 기록 |
| `trace.ts` | 도착 시각 → 노드별(route…finalize) 상태·소요 ms·서브작업·재계획 라운드 도출(순수 함수) |
| `cite.ts` | 리포트 수치("128 bpm", "25분") → 이벤트 각주. 유형 언급+수치 일치+후보 1개일 때만 인용 |
| `series.ts` | 버킷팅(다운샘플), 백분위, 판독값·기준선 계산 |
| `Timeline.tsx` · `EventTable.tsx` · `Run.tsx` · `Markdown.tsx` · `App.tsx` | UI |

## 백엔드 가정 (CONTRACT 범위 내)
- `/users/{id}/series?lookback_days=30` 과 `/users/{id}/anomalies?lookback_days=30` 을 사용자당 한 번 받아 1/7/30일 전환·확대는 클라이언트에서 처리한다.
- 상단 판독값은 시계열에서 계산한 값이다(안정시 심박 = 깨어 있고 걸음 0 구간의 중앙값, 커버리지 = 5분 구간 중 심박 존재 비율). 서버 `summarize()` 와 정의가 조금 다를 수 있다.
- 단계별 소요 시간은 서버 측정이 아니라 **SSE 이벤트의 클라이언트 도착 시각** 기준이다.
- 이벤트 번호(#)는 30일 로그에서 시작 시각 오름차순. 에이전트가 보고한 이상은 `anomaly_id`(없으면 유형+시작 시각)로 로그와 연결한다.

## E2E 선택자 (Playwright 등)
역할+이름 선택자를 우선하고, 반복 요소나 이름이 겹치는 요소에는 `data-testid` 를 쓴다.

| 대상 | 선택자 |
|---|---|
| 사용자 선택 | 버튼 `getByRole('combobox', { name: '대상 사용자 선택' })` (`user-select`) → 옵션 `getByRole('option')` (`user-option`, `data-user-id`). 네이티브 select 가 아니므로 `selectOption` 불가 |
| 조회 기간 | `getByRole('group', { name: '조회 기간' })` 안의 `1일/7일/30일` (`aria-pressed`) · `range-1/7/30` |
| 확대 해제 | `zoom-reset` (확대 중일 때만 존재) · 차트 조작 영역 `timeline-hit`, SVG `timeline` |
| 판독값 | `readout-resting/range/coverage/last/counts`, `count-critical/warning/info` |
| 이상 행 | `event-row` (+`data-anomaly-id`, `data-severity`, 선택 시 `aria-current="true"`) · 번호 버튼 `이상 N번 차트에서 보기` |
| 차트 기록 없음 | `gap-zone` (빗금 영역, 4px 이상일 때만 그려짐) · 범례 "기록 없음" |
| 입력 글자 수 | `char-count` (1,800자부터 표시, `role="status"`) · 입력창 `maxLength=2000` |
| 입력/전송/중단 | `getByRole('textbox', { name: '질문 입력' })` · `getByRole('button', { name: '전송' })` · `{ name: '중단' }` · `composer`/`send`/`stop` |
| 예시 질문 | `getByRole('group', { name: '예시 질문' })` 안의 버튼 · `example-prompt` |
| 실행 기록 | `run` (+`data-status`) · 진행 단계 `trace-step` (+`data-node`, `data-status`) · 서브작업 `trace-sub` |
| 안전 안내 | `safety-band` (`role="alert"`) |
| 리포트/각주 | `report` (`aria-live`) · `citation` (+`data-anomaly-id`, 이름 `이상 N번 근거 보기`; 가리키는 이상이 조회 기간 밖이면 `이상 N번 근거 보기, 조회 기간이 바뀝니다` + `class="cite far"`) |
| 검증 상태 | `validation-status` (`role="status"`, `data-passed`, `data-degraded`) |
| 데이터셋 | 그룹 `getByRole('group', { name: '데이터셋' })` 안의 버튼 · `dataset-fitbit` / `dataset-stanford` (`aria-pressed`). Stanford 가 없으면 그룹 자체가 없다 |
| 실제 사건 | 차트 마커 `event-marker`(`data-type`), 첫 경고 괄호 `lead-annotation`, 안내 영역 `events-panel`, 이동 버튼 `event-jump`, 복귀 `anchor-reset`, 문장 `lead-text` |
| 보호자 알림 승인 | 카드 `approval-card`(`data-status`: pending·sending·approved·denied·conflict·error) · 버튼 `approve`/`deny` · 미리보기 `approval-preview` · 결과 줄 `approval-result`. 승인·거절은 항상 `{thread_id, approved, approval_id}` 로 보낸다. 같은 대화에 더 새로운 승인 요청이 오면 이전 카드는 버튼이 사라지고 `data-status="conflict"`("이 요청은 이미 처리됐거나 새 요청으로 바뀌었습니다"). 승인 여부와 관계없이 `safety-band` 는 그대로 남는다 |
| 기준선 경고 | `v1-banner` (`role="alert"`, 서버 `/version.supervisor_version` 이 `v1` 일 때. 목업은 `?v=1`) |
| 개발자 보기 전용 | `usage-dev`(호출 수·토큰·비용), `retry-note`(서브작업 "재시도 N회") |
| 기타 | `api-status`(`data-state`), `theme-toggle`, `dev-toggle`(개발자 보기), `print-report`, `new-thread`, `undo`(되돌리기 줄) |

진행 과정은 `<details>` 안에 있고 실행이 끝나면 접힌다. `trace-step` 은 DOM 에 있지만 펼쳐야 보인다("진행 과정" 요약을 클릭).
목업 모드(`npm run dev:mock`)는 네트워크 없이 같은 DOM 을 만든다. `?ask=0` 으로 로드 직후 예시 질문 1번을 자동 전송(이때 페이지는 자동 스크롤하지 않는다), `?user=U03&range=30` 으로 사용자·기간을 지정할 수 있다.

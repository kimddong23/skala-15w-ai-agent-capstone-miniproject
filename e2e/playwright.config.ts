import { defineConfig, devices } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
/** 백엔드 LLM 모드. 기본 false(결정론). test:llm 스크립트가 true 로 켠다. */
const LLM = process.env.LLM_ENABLED === "true";
/**
 * 프론트엔드 서버. 기본 "preview": 시작 시점 소스를 e2e/.dist 로 빌드해 정적 서빙(개발 서버 HMR 이 소스 수정마다
 * 페이지를 리로드해 테스트가 흔들리는 것을 방지, frontend/dist 도 건드리지 않음). E2E_FRONTEND=dev 면 vite dev.
 */
const FRONTEND_DEV = process.env.E2E_FRONTEND === "dev";
const DIST = path.join(here, ".dist");
/** 프론트 포트(기본 8600). /api 프록시를 쓰므로 어느 포트여도 동작한다. 다른 개발 서버가 점유하면 E2E_PORT 로 피한다. */
const PORT = Number(process.env.E2E_PORT ?? 8600);
/** 백엔드 포트(기본 8100). E2E_BACKEND_PORT 로 변경, 프론트 프록시(VITE_PROXY_TARGET)도 같은 값을 따른다. */
const BACKEND = Number(process.env.E2E_BACKEND_PORT ?? 8100);
/** HITL(보호자 알림 승인) 전용 백엔드·프론트. 설정은 프로세스 시작 시 고정되므로 별도 서버가 필요하다. */
const BACKEND_H = Number(process.env.E2E_HITL_BACKEND_PORT ?? 8101);
const PORT_H = Number(process.env.E2E_HITL_PORT ?? 8601);

export default defineConfig({
  testDir: "./tests",
  outputDir: "./test-results",
  reporter: [["list"], ["html", { open: "never", outputFolder: "./playwright-report" }]],
  fullyParallel: false,
  workers: 1, // 백엔드 단일 인스턴스(MCP subprocess 1개)를 공유 → 직렬 실행으로 타이밍 의존 제거
  retries: 0,
  timeout: Number(process.env.E2E_TEST_TIMEOUT ?? 60_000),
  // 부하가 큰 기계(도커·다른 작업과 동시 실행)에서는 E2E_EXPECT_TIMEOUT=30000 으로 늘린다.
  expect: { timeout: Number(process.env.E2E_EXPECT_TIMEOUT ?? 10_000) },
  globalSetup: "./global-setup.ts",
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    ...devices["Desktop Chrome"],
    viewport: { width: 1440, height: 900 },
    locale: "ko-KR",
  },
  projects: [
    { name: "default", testIgnore: /(llm|hitl)\.spec\.ts/ },
    { name: "hitl", testMatch: /hitl\.spec\.ts/, use: { baseURL: `http://localhost:${PORT_H}` } },
    { name: "llm", testMatch: /llm\.spec\.ts/, timeout: 900_000, expect: { timeout: 600_000 } },
  ],
  // 이미 떠 있는 서버는 재사용한다(내가 띄운 것만 종료). 백엔드 /health 의 llm 플래그는 global-setup 에서 확인한다.
  webServer: [
    {
      command: `.venv/bin/python -m uvicorn app.api.main:app --port ${BACKEND} --log-level warning`,
      cwd: root,
      url: `http://127.0.0.1:${BACKEND}/health`,
      env: { LLM_ENABLED: LLM ? "true" : "false", CHECKPOINTER: "memory" },
      reuseExistingServer: true,
      timeout: 60_000,
    },
    {
      command: FRONTEND_DEV
        ? `npm run dev -- --port ${PORT} --strictPort`
        : `npx vite build --outDir "${DIST}" --emptyOutDir && npx vite preview --outDir "${DIST}" --port ${PORT} --strictPort`,
      cwd: path.join(root, "frontend"),
      env: { VITE_PROXY_TARGET: `http://localhost:${BACKEND}` },
      url: `http://localhost:${PORT}`,
      // 이미 떠 있는 서버(예: VITE_MOCK=1 개발 서버)를 재사용하면 결과가 오염되므로 기본은 점유 시 즉시 실패. E2E_REUSE=1 로 허용.
      reuseExistingServer: process.env.E2E_REUSE === "1",
      timeout: 60_000,
    },
    {
      command: `.venv/bin/python -m uvicorn app.api.main:app --port ${BACKEND_H} --log-level warning`,
      cwd: root,
      url: `http://127.0.0.1:${BACKEND_H}/health`,
      env: { LLM_ENABLED: "false", CHECKPOINTER: "memory", HITL_ENABLED: "true" },
      reuseExistingServer: false,
      timeout: 60_000,
    },
    {
      // 첫 프론트 서버가 빌드한 dist 를 같은 정적 서버로 한 번 더 서빙한다(프록시 대상만 다르다). 빌드가 끝날 때까지 기다린다.
      command: FRONTEND_DEV
        ? `npm run dev -- --port ${PORT_H} --strictPort`
        : `sh -c 'until [ -f "${DIST}/index.html" ]; do sleep 1; done; sleep 2; npx vite preview --outDir "${DIST}" --port ${PORT_H} --strictPort'`,
      cwd: path.join(root, "frontend"),
      env: { VITE_PROXY_TARGET: `http://localhost:${BACKEND_H}` },
      url: `http://localhost:${PORT_H}`,
      reuseExistingServer: false,
      timeout: 120_000,
    },
  ],
});

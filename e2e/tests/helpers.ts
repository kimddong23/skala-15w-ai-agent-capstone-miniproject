/** 선택자는 frontend/README.md 의 "E2E 선택자" 표(data-testid + 역할/이름)만 사용한다. */
import { expect, type Locator, type Page } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const SHOTS = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../screenshots");
export const Q30 = "최근 30일 심박 이상 점검해줘";
export const Q7 = "최근 7일 심박 이상 있었는지 점검해줘";
export const ESCALATION_SNIPPET = "즉시 119 또는 가까운 응급실";

/** 콘솔 error / 페이지 예외 수집. */
export function collectErrors(page: Page): string[] {
  const errs: string[] = [];
  page.on("console", (m) => m.type() === "error" && errs.push(`console: ${m.text()}`));
  page.on("pageerror", (e) => errs.push(`pageerror: ${e.message}`));
  return errs;
}

export const tid = (p: Page | Locator, id: string) => p.getByTestId(id);
export const picker = (p: Page) => p.getByTestId("user-select");
export const composer = (p: Page) => p.getByTestId("composer");
export const sendBtn = (p: Page) => p.getByTestId("send");
export const stopBtn = (p: Page) => p.getByTestId("stop");
export const runs = (p: Page) => p.getByTestId("run");
export const rangeBtn = (p: Page, d: 1 | 7 | 30) => p.getByTestId(`range-${d}`);
export const eventRows = (p: Page) => p.getByTestId("event-row");
export const readouts = (p: Page) => p.locator('[data-testid^="readout-"] .big');
export const selectedBand = (p: Page) => p.locator("svg[data-testid=timeline] rect.band.sel");

/** 첫 화면 로드 + 사용자 목록·차트 준비까지 대기. query 로 ?user=U05&range=30 같은 URL 상태를 줄 수 있다. */
export async function openApp(page: Page, query = "") {
  await page.goto(`/${query}`);
  await expect(tid(page, "api-status")).toHaveAttribute("data-state", "ok");
  await expect(picker(page)).toBeEnabled();
  await expect(page.getByLabel("판독값")).not.toHaveAttribute("aria-busy", "true");
  await expect(tid(page, "timeline")).toBeVisible();
}

export async function selectUser(page: Page, uid: string) {
  await picker(page).click();
  await page.locator(`[data-testid=user-option][data-user-id=${uid}]`).click();
  await expect(picker(page)).toContainText(uid);
  await expect(page.getByLabel("판독값")).not.toHaveAttribute("aria-busy", "true");
  await expect(tid(page, "timeline")).toBeVisible();
}

/** 질문을 보내고 가장 최근 실행이 끝날 때까지 기다린다. */
export async function ask(page: Page, text: string, timeout = 30_000): Promise<Locator> {
  const before = await runs(page).count();
  await composer(page).fill(text);
  await composer(page).press("Enter");
  const run = runs(page).nth(before);
  await expect(run).toBeVisible();
  await expect(run).not.toHaveAttribute("data-status", "running", { timeout });
  return run;
}

export const steps = (run: Locator) => run.getByTestId("trace-step");
export const openTrace = (run: Locator) => run.locator("details.trace-box > summary").click();

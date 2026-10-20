// HITL(보호자 알림 승인) 화면 E2E. HITL_ENABLED=true 백엔드(8101) + 프론트(8601)에서 실행한다 (playwright.config.ts 의 hitl 프로젝트).
import { expect, test } from "@playwright/test";
import { ask, collectErrors, composer, ESCALATION_SNIPPET, openApp, Q30, runs, tid } from "./helpers";

const CRIT = "?user=U05&range=30";
const card = (run: import("@playwright/test").Locator) => tid(run, "approval-card");

test("critical run shows a pending approval card with the preview and both buttons", async ({ page }) => {
  await openApp(page, CRIT);
  const run = await ask(page, Q30);
  await expect(card(run)).toHaveAttribute("data-status", "pending");
  await expect(tid(run, "approval-preview")).toContainText("시뮬레이션");
  await expect(tid(run, "approve")).toBeVisible();
  await expect(tid(run, "deny")).toBeVisible();
});

test("the safety band is shown together with the card and says approval is not required to keep it", async ({ page }) => {
  await openApp(page, CRIT);
  const run = await ask(page, Q30);
  await expect(tid(run, "safety-band")).toBeVisible();
  await expect(card(run)).toContainText("승인하지 않아도 위의 안전 안내는 그대로 남습니다");
});

test("approving marks the card approved, hides the buttons and appends the simulated line to the report", async ({ page }) => {
  await openApp(page, CRIT);
  const run = await ask(page, Q30);
  await tid(run, "approve").click();
  await expect(card(run)).toHaveAttribute("data-status", "approved");
  await expect(tid(run, "approval-result")).toContainText("시뮬레이션");
  await expect(tid(run, "approve")).toHaveCount(0);
  await expect(tid(run, "report")).toContainText("보호자 알림 발송됨(시뮬레이션)");
});

test("denying marks the card denied and the escalation notice stays", async ({ page }) => {
  await openApp(page, CRIT);
  const run = await ask(page, Q30);
  await tid(run, "deny").click();
  await expect(card(run)).toHaveAttribute("data-status", "denied");
  await expect(tid(run, "approval-result")).toContainText("보내지 않았습니다");
  await expect(tid(run, "safety-band")).toBeVisible();
  await expect(tid(run, "report")).toContainText("보호자 알림을 발송하지 않았습니다");
  await expect(run.getByText(ESCALATION_SNIPPET)).toHaveCount(1);
});

test("a non-critical run has no approval card", async ({ page }) => {
  await openApp(page, "?user=U07&range=7");
  const run = await ask(page, "최근 7일 심박 이상 있었는지 점검해줘");
  await expect(card(run)).toHaveCount(0);
});

test("the approval flow produces no console errors", async ({ page }) => {
  const errs = collectErrors(page);
  await openApp(page, CRIT);
  const run = await ask(page, Q30);
  await tid(run, "approve").click();
  await expect(card(run)).toHaveAttribute("data-status", "approved");
  expect(errs.filter((e) => !/\/events|404/.test(e))).toEqual([]);
});

test("sending a new message auto-denies: approving the old card afterwards shows 'already handled'", async ({ page }) => {
  await openApp(page, CRIT);
  const first = await ask(page, Q30);
  await ask(page, "그럼 운동할 때는 괜찮은 거야?");
  await tid(first, "approve").click();
  await expect(card(first)).toHaveAttribute("data-status", "conflict");
  await expect(tid(first, "approval-result")).toContainText("이미 처리됐거나 새 요청으로 바뀌었습니다");
});

test("a re-analysis creates a new pending card and approving it works", async ({ page }) => {
  await openApp(page, CRIT);
  await ask(page, Q30);
  const second = await ask(page, "다시 30일 점검해줘");
  await expect(card(second)).toHaveAttribute("data-status", "pending");
  await tid(second, "approve").click();
  await expect(card(second)).toHaveAttribute("data-status", "approved");
});

test("an older run's card is no longer actionable once a newer run exists in the thread", async ({ page }) => {
  await openApp(page, CRIT);
  const first = await ask(page, Q30);
  await ask(page, "다시 30일 점검해줘");
  await expect(tid(first, "approve")).toHaveCount(0);
});

test("a shorter re-analysis without critical results has no new card", async ({ page }) => {
  await openApp(page, CRIT);
  await ask(page, Q30);
  const second = await ask(page, "다시 7일 점검해줘");
  await expect(card(second)).toHaveCount(0);
});

test("a 500 from /chat/approve keeps the buttons so the user can retry", async ({ page }) => {
  await openApp(page, CRIT);
  const run = await ask(page, Q30);
  let fail = true;
  await page.route("**/api/chat/approve", (r) => (fail ? r.fulfill({ status: 500, body: "boom" }) : r.continue()));
  await tid(run, "approve").click();
  await expect(card(run)).toHaveAttribute("data-status", "error");
  await expect(tid(run, "approve")).toBeVisible();
  fail = false;
  await tid(run, "approve").click();
  await expect(card(run)).toHaveAttribute("data-status", "approved");
});

test("a network failure on approve also leaves the card retryable", async ({ page }) => {
  await openApp(page, CRIT);
  const run = await ask(page, Q30);
  let fail = true;
  await page.route("**/api/chat/approve", (r) => (fail ? r.abort() : r.continue()));
  await tid(run, "approve").click();
  await expect(card(run)).toHaveAttribute("data-status", "error");
  fail = false;
  await tid(run, "deny").click();
  await expect(card(run)).toHaveAttribute("data-status", "denied");
});

test("double-clicking approve sends one request and ends approved (no conflict flash)", async ({ page }) => {
  await openApp(page, CRIT);
  const run = await ask(page, Q30);
  let n = 0;
  page.on("request", (r) => r.url().endsWith("/api/chat/approve") && n++);
  await tid(run, "approve").dblclick();
  await expect(card(run)).toHaveAttribute("data-status", "approved");
  expect(n).toBe(1);
});

test("the approval card has an accessible name and named buttons", async ({ page }) => {
  await openApp(page, CRIT);
  const run = await ask(page, Q30);
  await expect(run.getByRole("region", { name: "보호자 알림 승인" })).toBeVisible();
  await expect(run.getByRole("button", { name: "승인" })).toBeVisible();
  await expect(run.getByRole("button", { name: "거절" })).toBeVisible();
});

test("the result line is announced politely (role=status)", async ({ page }) => {
  await openApp(page, CRIT);
  const run = await ask(page, Q30);
  await tid(run, "approve").click();
  await expect(tid(run, "approval-result")).toHaveAttribute("role", "status");
});

test("approval works on a mobile viewport without horizontal scroll", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openApp(page, CRIT);
  const run = await ask(page, Q30);
  await expect(card(run)).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(0);
  await tid(run, "deny").click();
  await expect(card(run)).toHaveAttribute("data-status", "denied");
});

test("print view includes the final report line after approval", async ({ page }) => {
  await openApp(page, CRIT);
  const run = await ask(page, Q30);
  await tid(run, "approve").click();
  await expect(card(run)).toHaveAttribute("data-status", "approved");
  await page.emulateMedia({ media: "print" });
  await expect(page.locator(".print-only")).toContainText("보호자 알림 발송됨(시뮬레이션)");
});

test("each user's pending approval is independent (switching users keeps the card)", async ({ page }) => {
  await openApp(page, CRIT);
  const run = await ask(page, Q30);
  await expect(card(run)).toHaveAttribute("data-status", "pending");
  await page.getByTestId("user-select").click();
  await page.locator("[data-testid=user-option][data-user-id=U07]").click();
  await expect(runs(page)).toHaveCount(0);
  await page.getByTestId("user-select").click();
  await page.locator("[data-testid=user-option][data-user-id=U05]").click();
  await expect(card(runs(page).first())).toHaveAttribute("data-status", "pending");
});

test("composer stays usable while an approval is pending", async ({ page }) => {
  await openApp(page, CRIT);
  await ask(page, Q30);
  await expect(composer(page)).toBeEnabled();
});

test("the approval card does not expose internal type codes in the normal view", async ({ page }) => {
  await openApp(page, CRIT);
  const run = await ask(page, Q30);
  await expect(card(run)).not.toContainText(/resting_tachycardia|bradycardia|personal_deviation/);
});

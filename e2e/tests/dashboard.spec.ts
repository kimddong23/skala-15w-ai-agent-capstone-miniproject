import { expect, test } from "@playwright/test";
import { collectErrors, eventRows, openApp, picker, rangeBtn, readouts, SHOTS, selectUser, selectedBand, tid } from "./helpers";

test("page loads with no console errors", async ({ page }) => {
  const errs = collectErrors(page);
  await openApp(page);
  await expect(page.getByRole("heading", { level: 1 })).toContainText("VITAL SENTINEL");
  await page.waitForTimeout(500);
  expect(errs).toEqual([]);
  await page.screenshot({ path: `${SHOTS}/01-initial.png`, fullPage: true });
});

test("health indicator shows the server connected", async ({ page }) => {
  await openApp(page);
  await expect(tid(page, "api-status")).toContainText("서버 연결됨");
});

test("user picker lists 15 users U01..U15", async ({ page }) => {
  await openApp(page);
  await picker(page).click();
  const ids = await page.getByTestId("user-option").evaluateAll((os) => os.map((o) => o.getAttribute("data-user-id")));
  expect(ids).toEqual(Array.from({ length: 15 }, (_, i) => `U${String(i + 1).padStart(2, "0")}`));
  await page.screenshot({ path: `${SHOTS}/01b-picker-open.png` });
});

test("user picker is operable by keyboard (arrows + Enter)", async ({ page }) => {
  await openApp(page);
  await picker(page).focus();
  await page.keyboard.press("ArrowDown"); // open
  await page.keyboard.press("ArrowDown");
  await page.keyboard.press("Enter");
  await expect(picker(page)).toContainText("U02");
});

test("Escape inside the open picker closes it without resetting chart zoom", async ({ page }) => {
  await openApp(page, "?user=U05&range=30");
  await eventRows(page).first().getByRole("button").click();
  await expect(tid(page, "zoom-reset")).toBeVisible();
  await picker(page).focus();
  await page.keyboard.press("ArrowDown");
  await page.keyboard.press("Escape");
  await expect(page.getByRole("listbox")).toHaveCount(0);
  await expect(tid(page, "zoom-reset")).toBeVisible();
});

test("selecting a user renders readouts, timeline and event log", async ({ page }) => {
  await openApp(page);
  await selectUser(page, "U05");
  expect((await readouts(page).allTextContents()).slice(0, 3).join("")).toMatch(/\d/);
  await expect(page.locator("svg[data-testid=timeline] path.hr-line")).toBeVisible();
  await rangeBtn(page, 30).click();
  await expect(eventRows(page).first()).toBeVisible(); // 기간별 조회 완료까지 대기
});

test("readouts change when the user changes", async ({ page }) => {
  await openApp(page);
  await selectUser(page, "U05");
  const a = (await readouts(page).allTextContents()).join("|");
  await selectUser(page, "U03");
  expect((await readouts(page).allTextContents()).join("|")).not.toEqual(a);
});

test("severity counts in the strip match the event rows for the 30d range", async ({ page }) => {
  await openApp(page, "?user=U05&range=30");
  const n = async (s: string) => Number(await tid(page, `count-${s}`).locator(".big").innerText());
  const total = (await n("critical")) + (await n("warning")) + (await n("info"));
  await expect(eventRows(page)).toHaveCount(total);
  expect(await n("critical")).toBeGreaterThanOrEqual(1);
});

test("range switch 1d/7d/30d toggles pressed state", async ({ page }) => {
  await openApp(page);
  for (const d of [1, 7, 30] as const) {
    await rangeBtn(page, d).click();
    await expect(rangeBtn(page, d)).toHaveAttribute("aria-pressed", "true");
    for (const o of ([1, 7, 30] as const).filter((x) => x !== d)) await expect(rangeBtn(page, o)).toHaveAttribute("aria-pressed", "false");
  }
});

test("range switch is reflected in the URL (?range=)", async ({ page }) => {
  await openApp(page);
  await rangeBtn(page, 30).click();
  await expect(page).toHaveURL(/range=30/);
});

test("URL state ?user=U05&range=30 is restored on load", async ({ page }) => {
  await openApp(page, "?user=U05&range=30");
  await expect(picker(page)).toContainText("U05");
  await expect(rangeBtn(page, 30)).toHaveAttribute("aria-pressed", "true");
});

test("invalid ?range= and ?user= fall back instead of breaking", async ({ page }) => {
  const errs = collectErrors(page);
  await openApp(page, "?user=ZZZ&range=999");
  await expect(rangeBtn(page, 7)).toHaveAttribute("aria-pressed", "true");
  expect(errs).toEqual([]);
});

test("30d window lists at least as many events as 7d", async ({ page }) => {
  await openApp(page, "?user=U05&range=7");
  await expect(page.getByLabel("판독값")).not.toHaveAttribute("aria-busy", "true");
  const n7 = await eventRows(page).count();
  await rangeBtn(page, 30).click();
  await expect(page.getByLabel("판독값")).not.toHaveAttribute("aria-busy", "true");
  await expect.poll(() => eventRows(page).count()).toBeGreaterThanOrEqual(n7);
});

test("clicking an event row highlights the matching chart band", async ({ page }) => {
  await openApp(page, "?user=U05&range=30");
  const row = eventRows(page).first();
  await row.getByRole("button").click();
  await expect(row).toHaveAttribute("aria-current", "true");
  await expect(selectedBand(page)).toHaveCount(1);
  await page.screenshot({ path: `${SHOTS}/02-event-selected.png` });
});

test("clicking the row body (not the number button) also selects it", async ({ page }) => {
  await openApp(page, "?user=U05&range=30");
  const row = eventRows(page).nth(1);
  await row.locator("td").nth(2).click();
  await expect(row).toHaveAttribute("aria-current", "true");
});

test("Escape clears chart zoom after selecting an event", async ({ page }) => {
  await openApp(page, "?user=U05&range=30");
  await eventRows(page).first().getByRole("button").click();
  await expect(tid(page, "zoom-reset")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(tid(page, "zoom-reset")).toHaveCount(0);
});

test("dragging on the chart zooms and the reset button appears", async ({ page }) => {
  await openApp(page, "?user=U05&range=30");
  const box = (await tid(page, "timeline-hit").boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.3, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.6, box.y + box.height / 2, { steps: 8 });
  await page.mouse.up();
  await expect(tid(page, "zoom-reset")).toBeVisible();
  await tid(page, "zoom-reset").click();
  await expect(tid(page, "zoom-reset")).toHaveCount(0);
});

test("switching range clears an active zoom", async ({ page }) => {
  await openApp(page, "?user=U05&range=30");
  await eventRows(page).first().getByRole("button").click();
  await expect(tid(page, "zoom-reset")).toBeVisible();
  await rangeBtn(page, 7).click();
  await expect(tid(page, "zoom-reset")).toHaveCount(0);
});

test("user with no detected events shows the empty message", async ({ page }) => {
  await openApp(page, "?user=U11&range=7");
  await expect(page.getByText("이 기간에 탐지된 이상이 없습니다.")).toBeVisible();
});

test("user with almost no data (U02) still renders without crashing", async ({ page }) => {
  const errs = collectErrors(page);
  await openApp(page, "?user=U02&range=30");
  await page.waitForTimeout(500);
  expect(errs).toEqual([]);
});

test("dark theme toggle cycles auto -> light -> dark and persists across reload", async ({ page }) => {
  await openApp(page);
  const btn = tid(page, "theme-toggle");
  await expect(btn).toContainText("자동");
  await btn.click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await btn.click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.screenshot({ path: `${SHOTS}/03-dark.png`, fullPage: true });
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expect(btn).toContainText("다크");
});

test("dark theme actually changes the page background", async ({ page }) => {
  await openApp(page);
  const bg = () => page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  await tid(page, "theme-toggle").click(); // light
  const light = await bg();
  await tid(page, "theme-toggle").click(); // dark
  expect(await bg()).not.toEqual(light);
});

test("mobile viewport 390px has no horizontal scroll", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openApp(page, "?user=U05&range=30");
  expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(0);
  await page.screenshot({ path: `${SHOTS}/04-mobile.png`, fullPage: true });
});

test("mobile viewport 390px has no horizontal scroll after a critical run", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openApp(page, "?user=U05&range=30");
  await page.getByTestId("composer").fill("최근 30일 심박 이상 점검해줘");
  await page.getByTestId("composer").press("Enter");
  await expect(page.getByTestId("run")).toHaveAttribute("data-status", "done", { timeout: 30_000 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(0);
  await page.screenshot({ path: `${SHOTS}/04b-mobile-run.png`, fullPage: true });
});

test("tablet viewport 820px has no horizontal scroll", async ({ page }) => {
  await page.setViewportSize({ width: 820, height: 1100 });
  await openApp(page, "?user=U05&range=30");
  expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(0);
});

test("disclaimer footer is always present", async ({ page }) => {
  await openApp(page);
  await expect(page.locator("footer.disclaimer")).toContainText("의료기기가 아니며");
});

test("skip link targets the main landmark", async ({ page }) => {
  await openApp(page);
  await expect(page.locator("a.skip")).toHaveAttribute("href", "#main");
  await expect(page.locator("main#main")).toHaveCount(1);
});

test("backend down shows offline state", async ({ page }) => {
  await page.route("**/api/**", (r) => r.abort());
  await page.goto("/");
  await expect(tid(page, "api-status")).toHaveAttribute("data-state", "down");
  await page.screenshot({ path: `${SHOTS}/05-backend-down.png`, fullPage: true });
});

test("backend down: user picker and composer are disabled", async ({ page }) => {
  await page.route("**/api/**", (r) => r.abort());
  await page.goto("/");
  await expect(picker(page)).toBeDisabled();
  await expect(tid(page, "composer")).toBeDisabled();
});

test("backend down: the user-facing error does not leak internal details", async ({ page }) => {
  await page.route("**/api/**", (r) => r.abort());
  await page.goto("/");
  await expect(page.getByRole("alert")).toBeVisible();
  await expect(page.getByRole("alert")).not.toContainText(/Failed to fetch|HTTP \d{3}|TypeError/);
});

test("series fetch failing shows the error state and retry recovers", async ({ page }) => {
  let fail = true;
  await page.route("**/api/users/*/series*", (r) => (fail ? r.fulfill({ status: 500, body: "boom" }) : r.continue()));
  await page.goto("/");
  await expect(page.getByRole("alert").filter({ hasText: "데이터를 불러오지 못했습니다" })).toBeVisible();
  fail = false;
  await page.getByRole("button", { name: "다시 시도" }).click();
  await expect(tid(page, "timeline")).toBeVisible();
});

test("slow series response shows a loading state then the chart", async ({ page }) => {
  await page.route("**/api/users/*/series*", async (r) => {
    await new Promise((res) => setTimeout(res, 1500));
    await r.continue();
  });
  await page.goto("/");
  await expect(page.getByLabel("차트 불러오는 중…")).toBeVisible();
  await expect(tid(page, "timeline")).toBeVisible();
});

test("sensor gaps break the heart-rate line into several segments (U01)", async ({ page }) => {
  await openApp(page, "?user=U01&range=7");
  const d = await page.locator("svg[data-testid=timeline] path.hr-line").getAttribute("d");
  expect((d!.match(/M/g) ?? []).length).toBeGreaterThan(3);
});

test("a fully covered user (U12) draws far fewer line breaks than U01", async ({ page }) => {
  await openApp(page, "?user=U12&range=7");
  const d12 = (await page.locator("svg[data-testid=timeline] path.hr-line").getAttribute("d"))!;
  await selectUser(page, "U01");
  const d01 = (await page.locator("svg[data-testid=timeline] path.hr-line").getAttribute("d"))!;
  expect((d12.match(/M/g) ?? []).length).toBeLessThan((d01.match(/M/g) ?? []).length);
});

test("the 1-day view fetches a fine-grained series for that range", async ({ page }) => {
  const urls: string[] = [];
  page.on("request", (r) => /\/api\/users\/U05\/series/.test(r.url()) && urls.push(r.url()));
  await openApp(page, "?user=U05&range=7");
  await rangeBtn(page, 1).click();
  await expect.poll(() => urls.length).toBeGreaterThanOrEqual(2);
  expect(urls.some((u) => /lookback_days=1(&|$)/.test(u))).toBe(true);
  const last = urls[urls.length - 1];
  // 1일 뷰는 5분 해상도(시간당 12점)여야 한다: 차트 점 수가 시간 단위(24)보다 훨씬 많다
  const res = await page.request.get(last);
  const body = await res.json();
  expect(body.points.length).toBeGreaterThan(100);
});

test("switching back to a visited range does not break the chart", async ({ page }) => {
  const errs = collectErrors(page);
  await openApp(page, "?user=U05&range=7");
  for (const d of [1, 30, 7, 1] as const) {
    await rangeBtn(page, d).click();
    await expect(page.locator("svg[data-testid=timeline] path.hr-line")).toBeVisible();
  }
  expect(errs).toEqual([]);
});

test("composer exposes maxlength 2000", async ({ page }) => {
  await openApp(page);
  await expect(page.getByTestId("composer")).toHaveAttribute("maxlength", "2000");
});

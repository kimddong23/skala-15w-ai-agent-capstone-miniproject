// 안전·정합성 회귀 중 화면에서 확인 가능한 것들: 조사가 붙은 사용자 ID, 중단된 재점검 뒤 에스컬레이션 유지, 사용자 혼입 방지 등.
import { expect, test } from "@playwright/test";
import { ask, composer, ESCALATION_SNIPPET, openApp, Q30, runs, stopBtn, tid, SHOTS } from "./helpers";

test("particle-suffixed user id: 'U03의 …' typed while U05 is selected analyzes U03", async ({ page }) => {
  await openApp(page, "?user=U05&range=30");
  const run = await ask(page, "U03의 최근 30일 점검해줘");
  await expect(tid(run, "report")).toContainText("(U03)");
  await expect(tid(run, "report")).not.toContainText("(U05)");
});

test("particle-suffixed user id: 'U03는 …' and 'u03를 …' variants analyze U03", async ({ page }) => {
  await openApp(page, "?user=U05&range=30");
  for (const q of ["U03는 괜찮은지 점검해줘", "u03를 점검해줘"]) {
    const run = await ask(page, q);
    await expect(tid(run, "report")).toContainText("(U03)");
  }
});

test("stopping a re-analysis and then asking a follow-up still shows the safety band", async ({ page }) => {
  await openApp(page, "?user=U05&range=30");
  const first = await ask(page, Q30);
  await expect(tid(first, "safety-band")).toBeVisible();

  let delay = true;
  await page.route("**/api/chat/stream", async (r) => {
    if (delay) await new Promise((res) => setTimeout(res, 4000));
    await r.continue().catch(() => {});
  });
  await composer(page).fill("다시 30일 점검해줘");
  await composer(page).press("Enter");
  await expect(stopBtn(page)).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(runs(page).nth(1)).toHaveAttribute("data-status", "aborted");
  delay = false;

  const follow = await ask(page, "그럼 괜찮은 거야?");
  await expect(follow).toHaveAttribute("data-status", "done");
  await expect(tid(follow, "safety-band")).toBeVisible();
  await expect(follow.getByText(ESCALATION_SNIPPET)).toHaveCount(1);
});

test("UI smoke: a normal run is not labelled all-clear (a real detect failure is covered in tests/test_e2e_safety.py)", async ({ page }) => {
  await openApp(page, "?user=U05&range=30");
  const run = await ask(page, "U05 최근 30일 이상 점검해줘");
  await expect(tid(run, "report")).not.toContainText("감지되지 않았습니다");
});

test("a follow-up after switching the typed user does not mix users", async ({ page }) => {
  await openApp(page, "?user=U01&range=7");
  await ask(page, "U02 이상 점검해줘");
  const follow = await ask(page, "가장 심했던 건 몇 시야?");
  await expect(follow).toHaveAttribute("data-status", "done");
  const txt = await tid(follow, "report").innerText();
  expect(txt).not.toMatch(/\(U01\).*\(U02\)|\(U02\).*\(U01\)/s);
});

const LIMIT_NOTICE = "이번 기간은 기록이 제한적이어서";

test("data-limit notice is shown for a low-coverage user (U02) exactly once", async ({ page }) => {
  await openApp(page, "?user=U02&range=7");
  const run = await ask(page, "최근 7일 심박 이상 있었는지 점검해줘");
  await expect(run.getByText(LIMIT_NOTICE)).toHaveCount(1);
  await page.screenshot({ path: `${SHOTS}/12-data-limit-notice.png`, fullPage: true });
});

test("data-limit notice is absent for a fully recorded user without gaps (U12, 7d)", async ({ page }) => {
  await openApp(page, "?user=U12&range=7");
  const run = await ask(page, "최근 7일 심박 이상 있었는지 점검해줘");
  await expect(run.getByText(LIMIT_NOTICE)).toHaveCount(0);
});

test("follow-up repeats the data-limit notice at most once", async ({ page }) => {
  await openApp(page, "?user=U02&range=7");
  await ask(page, "최근 7일 심박 이상 있었는지 점검해줘");
  const f = await ask(page, "그럼 운동할 때는 괜찮은 거야?");
  expect(await f.getByText(LIMIT_NOTICE).count()).toBeLessThanOrEqual(1);
});

test("context labels are Korean and the type label is '안정 시 빈맥'", async ({ page }) => {
  await openApp(page, "?user=U05&range=30");
  const run = await ask(page, Q30);
  const txt = await tid(run, "report").innerText();
  expect(txt).toContain("안정 시 빈맥");
  expect(txt).not.toMatch(/\b(rest|sleep|active)\b/);
});

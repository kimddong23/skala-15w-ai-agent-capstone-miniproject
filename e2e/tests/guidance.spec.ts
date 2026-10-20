// 코드가 쓰는 고정 안내(WARNING_GUIDANCE / NO_ANOMALY_GUIDANCE)가 화면에 정확히 한 번 나오는지, Stanford 실제 사건 화면.
import { expect, test } from "@playwright/test";
import { ask, collectErrors, openApp, picker, Q30, tid } from "./helpers";

const WG = "비슷한 패턴이 반복되거나 증상(가슴 두근거림·어지럼·숨참)이 있으면 의료진과 상담하세요.";
const NA = "데이터에서 이상이 보이지 않아도 가슴 통증·어지럼·숨참 같은 증상이 있으면 의료진과 상담하고, 증상이 심하면 119에 연락하세요.";
const ESC = "즉시 119 또는 가까운 응급실";
const count = async (run: import("@playwright/test").Locator, text: string) => (await run.innerText()).split(text).length - 1;

test.describe("fixed guidance texts on screen", () => {
  test("critical report: escalation band once, no warning/no-anomaly guidance, facts are readable", async ({ page }) => {
    await openApp(page, "?user=U05&range=30");
    const run = await ask(page, Q30);
    await expect(tid(run, "safety-band")).toHaveCount(1);
    expect(await count(run, ESC)).toBe(1);
    expect(await count(run, WG)).toBe(0);
    expect(await count(run, NA)).toBe(0);
    await expect(tid(run, "report")).toContainText("146 bpm");
  });

  test("warning report: WARNING_GUIDANCE once, no band", async ({ page }) => {
    await openApp(page, "?user=U05&range=7");
    const run = await ask(page, "U05 최근 7일 심박 이상 점검해줘");
    expect(await count(run, WG)).toBe(1);
    expect(await count(run, NA)).toBe(0);
    await expect(tid(run, "safety-band")).toHaveCount(0);
  });

  test("no-anomaly report: NO_ANOMALY_GUIDANCE once", async ({ page }) => {
    await openApp(page, "?user=U12&range=7");
    const run = await ask(page, "U12 최근 7일 심박 이상 점검해줘");
    expect(await count(run, NA)).toBe(1);
    expect(await count(run, WG)).toBe(0);
  });

  test("general guideline question: NO_ANOMALY_GUIDANCE once with the sources line", async ({ page }) => {
    await openApp(page);
    const run = await ask(page, "휴식기 심박 정상 범위가 어떻게 돼?");
    expect(await count(run, NA)).toBe(1);
    await expect(tid(run, "report")).toContainText("출처");
  });

  test("follow-up after a warning report repeats the guidance exactly once", async ({ page }) => {
    await openApp(page, "?user=U05&range=7");
    await ask(page, "U05 최근 7일 심박 이상 점검해줘");
    const f = await ask(page, "그럼 운동할 때는 괜찮은 거야?");
    expect(await count(f, WG)).toBe(1);
  });

  test("follow-up after a critical report keeps one escalation and adds nothing else", async ({ page }) => {
    await openApp(page, "?user=U05&range=30");
    await ask(page, Q30);
    const f = await ask(page, "그럼 운동할 때는 괜찮은 거야?");
    expect(await count(f, ESC)).toBe(1);
    expect(await count(f, WG)).toBe(0);
    expect(await count(f, NA)).toBe(0);
  });
});

test.describe("Stanford real events", () => {
  const S = "S004"; // COVID 라벨이 있는 참가자 (증상 시작 2023-12-06)

  const P = "S005"; // 서버가 계산한 첫 경고가 증상 5일 전(lead_days=5)인 참가자 (증상 시작 2023-10-13)

  async function openStanford(page: import("@playwright/test").Page, uid = S) {
    await openApp(page);
    const group = page.getByRole("group", { name: "데이터셋" });
    test.skip((await group.count()) === 0, "로컬 Stanford 데이터가 없는 환경");
    await tid(page, "dataset-stanford").click();
    await expect(tid(page, "dataset-stanford")).toHaveAttribute("aria-pressed", "true");
    await picker(page).click();
    await page.locator(`[data-testid=user-option][data-user-id=${uid}]`).click();
    await expect(picker(page)).toContainText(uid);
  }

  test("the real-event panel lists symptom onset, diagnosis and recovery with the research-label disclaimer", async ({ page }) => {
    await openStanford(page);
    await expect(tid(page, "events-panel")).toContainText("증상 시작");
    await expect(tid(page, "events-panel")).toContainText("연구용 라벨이며 진단이 아닙니다");
    await expect(tid(page, "events-panel")).toContainText("날짜는 개인정보 보호를 위해 이동");
  });

  test("the lead text says how many days before the symptom the first warning appeared", async ({ page }) => {
    await openStanford(page, P);
    await expect(tid(page, "lead-text").first()).toContainText("증상 5일 전 첫 경고");
  });

  test("a warning after the symptom day is worded as days after, not before", async ({ page }) => {
    await openStanford(page); // S004: 첫 경고가 증상 다음 날(lead_days=-1)
    await expect(tid(page, "lead-text").first()).toContainText("증상 1일 후 첫 경고");
    await expect(tid(page, "lead-text").first()).not.toContainText("일 전");
  });

  test("jumping to the event draws markers and the lead annotation; reset returns to the last record", async ({ page }) => {
    const errs = collectErrors(page);
    await openStanford(page);
    await tid(page, "event-jump").first().click();
    await expect(tid(page, "event-marker").first()).toBeVisible();
    expect(await tid(page, "event-marker").evaluateAll((e) => e.map((x) => x.getAttribute("data-type")))).toEqual(expect.arrayContaining(["symptom_onset", "diagnosis", "recovery"]));
    await expect(tid(page, "lead-annotation")).toHaveCount(1);
    await tid(page, "anchor-reset").click();
    await expect(tid(page, "anchor-reset")).toHaveCount(0);
    expect(errs).toEqual([]);
  });

  test("the first-warning day count is the same before and after jumping to the event ", async ({ page }) => {
    await openStanford(page, P); // lead_days>0 인 참가자여야 아래 정규식이 실제 값을 비교한다
    const before = (await tid(page, "lead-text").first().innerText()).match(/증상 (\d+)일 전/)?.[1];
    expect(before).toBe("5");
    await tid(page, "event-jump").first().click();
    await expect(tid(page, "lead-annotation")).toHaveCount(1);
    const after = (await tid(page, "lead-text").first().innerText()).match(/증상 (\d+)일 전/)?.[1];
    expect(after).toBe(before);
  });

  test("Stanford conversation: the request user id analyzes the S user and keeps the research disclaimer", async ({ page }) => {
    await openStanford(page);
    const run = await ask(page, "최근 30일 이상 점검해줘");
    await expect(tid(run, "report")).toContainText(`(${S})`);
  });

  test("switching back to FitBit restores the U user list and hides the event panel", async ({ page }) => {
    await openStanford(page);
    await tid(page, "dataset-fitbit").click();
    await expect(picker(page)).toContainText(/U\d{2}/);
    await expect(tid(page, "events-panel")).toHaveCount(0);
  });
});

test("dataset switch is not undone by a late initial load (reproduces only with E2E_FRONTEND=dev, StrictMode)", async ({ page }) => {
  await openApp(page);
  test.skip((await page.getByRole("group", { name: "데이터셋" }).count()) === 0, "로컬 Stanford 데이터가 없는 환경");
  await tid(page, "dataset-stanford").click();
  await page.waitForTimeout(3500);
  await expect(tid(page, "dataset-stanford")).toHaveAttribute("aria-pressed", "true");
});

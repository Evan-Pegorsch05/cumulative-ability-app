const { chromium } = require("playwright");

(async () => {
  const url = process.env.TARGET_URL;
  if (!url) throw new Error("TARGET_URL env var not set");

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  page.setDefaultNavigationTimeout(120000);

  const resp = await page.goto(url, { waitUntil: "domcontentloaded" });
  console.log("Visited:", url, "status:", resp?.status());

  const wakeBtn = page.getByRole("button", { name: /get this app back up/i });
  if (await wakeBtn.count()) {
    await wakeBtn.first().click();
    await page.waitForTimeout(15000);
    console.log("Clicked wake button");
  }

  await page.waitForTimeout(10000);
  await browser.close();
})();

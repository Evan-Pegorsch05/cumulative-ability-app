const { chromium } = require("playwright");

(async () => {
  const url = process.env.TARGET_URL;
  if (!url) throw new Error("TARGET_URL env var not set");

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  page.setDefaultNavigationTimeout(180000);

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  async function openAndMaybeWake(attempt) {
    console.log(`Attempt ${attempt}: goto ${url}`);
    await page.goto(url, { waitUntil: "domcontentloaded" });
    await sleep(8000);

    const wakeBtn = page.locator('button:has-text("Yes, get this app back up!")');

    if (await wakeBtn.count()) {
      console.log("Wake button found -> clicking");
      await wakeBtn.first().click({ force: true });
      await sleep(45000);
      await page.reload({ waitUntil: "domcontentloaded" });
      await sleep(10000);
    } else {
      console.log("No wake button found");
      await sleep(5000);
    }

    const stillAsleep = await page.locator('text=This app has gone to sleep').count();
    console.log("Still asleep screen?", stillAsleep ? "YES" : "NO");
    return stillAsleep === 0;
  }

  let ok = false;
  for (let i = 1; i <= 3; i++) {
    ok = await openAndMaybeWake(i);
    if (ok) break;
    console.log("Not awake yet, retrying...");
    await sleep(10000);
  }

  if (!ok) {
    throw new Error("App still shows sleep screen after retries");
  }

  await browser.close();
})();

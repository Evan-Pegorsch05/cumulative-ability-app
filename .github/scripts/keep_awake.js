const { chromium } = require("playwright");

(async () => {
  const url = process.env.TARGET_URL;
  if (!url) throw new Error("TARGET_URL env var not set");

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  page.setDefaultNavigationTimeout(180000);

  const wait = (ms) => new Promise((r) => setTimeout(r, ms));

  const visit = async () => {
    const resp = await page.goto(url, { waitUntil: "domcontentloaded" });
    console.log("Visited:", url, "status:", resp?.status());
  };

  await visit();

  // Give the page time to render the "sleep" UI if it's going to show up
  await wait(8000);

  // If the wake button exists, click it and wait for the app to come back
  const wakeBtn = page.getByRole("button", { name: /get this app back up/i });

  if (await wakeBtn.count()) {
    console.log("Wake button detected. Clicking...");
    await wakeBtn.first().click();

    // Wait longer for the container to spin up
    await wait(45000);

    // Reload once after wake to ensure app is actually running
    await visit();
    await wait(10000);
    console.log("Post-wake reload done.");
  } else {
    console.log("No wake button detected (app likely already awake).");
    await wait(5000);
  }

  await browser.close();
})();

// Capture every page at desktop and phone widths in light and dark themes.
//
//   node scripts/screenshots.mjs [baseUrl] [outDir] [filter]
//
// Expects a running dev server with the mock API (npm run dev:mock).
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { chromium } from "playwright";

const base = process.argv[2] ?? "http://localhost:5173";
const outDir = process.argv[3] ?? "screenshots";
const filter = process.argv[4];
// Optional "light:1440" style restriction, e.g. node scripts/screenshots.mjs url out run light:1440
const only = process.argv[5];
mkdirSync(outDir, { recursive: true });

const PAGES = [
  ["home", "/"],
  ["runs", "/runs?cols=suite:olmes:base,task:gsm8k:cot:olmo3"],
  ["run-overview", "/runs/{run}?baseline={base}"],
  ["run-tasks", "/runs/{run}?tab=tasks&baseline={base}"],
  ["run-instances", "/runs/{run}?tab=instances&task=gsm8k:cot:olmo3&baseline={base}"],
  ["run-inference", "/runs/{run}?tab=inference"],
  ["run-config", "/runs/{run}?tab=config&baseline={base}"],
  ["run-artifacts", "/runs/{run}?tab=artifacts"],
  ["task-drilldown", "/runs/{run}/tasks/gsm8k:cot:olmo3?baseline={base}"],
  ["compare-heatmap", "/compare?group=olmo3-7b-midtrain-ablations&scope=suite:olmes:base"],
  ["compare-delta", "/compare?group=olmo3-7b-midtrain-ablations&mode=delta&scope=all&baseline={base}"],
  ["compare-pairwise", "/compare?group=olmo3-7b-midtrain-ablations&view=pairwise&scope=suite:olmes:base"],
  ["compare-scatter", "/compare?group=olmo3-7b-midtrain-ablations&view=scatter&scope=all"],
  ["compare-profiles", "/compare?group=olmo3-7b-midtrain-ablations&view=profiles&scope=all"],
  ["compare-progression", "/compare?group=olmo3-7b-midtrain-ablations&view=progression&scope=all"],
  ["compare-disagree", "/compare?group=olmo3-7b-midtrain-ablations&view=disagree&scope=all"],
  ["models", "/models"],
  ["model", "/models/olmo3-7b-midtrain"],
  ["tasks", "/tasks"],
  ["task", "/tasks/gsm8k:cot:olmo3"],
  ["suite", "/suites/olmes:base"],
  ["groups", "/groups"],
  ["group", "/groups/olmo3-7b-midtrain-ablations"],
  ["views", "/views"],
];

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await ctx.newPage();
await page.goto(`${base}/runs?series=olmo3-7b-midtrain&sort=-step`);
// Find a complete 7B run and its previous checkpoint to use as the subject and baseline.
await page.waitForSelector('[role="row"] a[href^="/runs/"]', { timeout: 20000 });
const hrefs = await page.$$eval('[role="row"] a[href^="/runs/"]', (as) => as.map((a) => a.getAttribute("href")));
const ids = [...new Set(hrefs.map((h) => h.split("?")[0].split("/")[2]))];
const run = ids[2];
const baseRun = ids[3];
await ctx.close();

const shots = [];
for (const theme of ["light", "dark"]) {
  for (const width of [1440, 390]) {
    if (only && only !== `${theme}:${width}`) continue;
    const context = await browser.newContext({
      viewport: { width, height: width > 600 ? 900 : 844 },
      deviceScaleFactor: width > 600 ? 1 : 2,
      colorScheme: theme,
    });
    await context.addInitScript((t) => {
      localStorage.setItem("oe.theme", JSON.stringify(t));
      localStorage.setItem(
        "oe.recents",
        JSON.stringify([
          { type: "compare", key: "c1", label: "olmo3-7b midtrain ablations", href: "/compare?group=olmo3-7b-midtrain-ablations", at: Date.now() - 3600e3 },
          { type: "model", key: "olmo3-7b-midtrain", label: "olmo3-7b-midtrain", href: "/models/olmo3-7b-midtrain", at: Date.now() - 7200e3 },
        ]),
      );
    }, theme);
    const p = await context.newPage();
    for (const [name, path] of PAGES) {
      if (filter && !name.includes(filter)) continue;
      const url = base + path.replace("{run}", run).replace("{base}", `r:${baseRun}`);
      await p.goto(url);
      try {
        await p.waitForLoadState("networkidle", { timeout: 15000 });
      } catch {
        // keep going
      }
      await p.waitForTimeout(700);
      const file = join(outDir, `${name}-${width}-${theme}.png`);
      await p.screenshot({ path: file, fullPage: true });
      shots.push(file);
    }
    await context.close();
  }
}
await browser.close();
console.log(shots.join("\n"));

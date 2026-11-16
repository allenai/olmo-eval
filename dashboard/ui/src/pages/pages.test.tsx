/** Page render tests against the MSW mock API. */
import { QueryClientProvider } from "@tanstack/react-query";
import { createMemoryHistory, RouterProvider } from "@tanstack/react-router";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { runDetail, runSuites, runTaskResults } from "@/mocks/api";
import { server } from "@/mocks/node";
import { getWorld } from "@/mocks/fixtures";
import { createQueryClient } from "@/queryClient";
import { createAppRouter } from "@/router";

function renderAt(url: string) {
  const router = createAppRouter({ history: createMemoryHistory({ initialEntries: [url] }) });
  const qc = createQueryClient();
  render(
    <QueryClientProvider client={qc}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  );
  return router;
}

const world = getWorld();
const run = world.runs.find((r) => r.summary.status === "partial" && r.model.series === "olmo3-7b-midtrain")!;
const runId = run.summary.run_id;
const group = run.summary.experiment_group!;

describe("pages", () => {
  it("Home shows recent runs, groups and activity", async () => {
    renderAt("/");
    expect(await screen.findByText(/Your recent runs/i)).toBeInTheDocument();
    expect(await screen.findByText("Active groups")).toBeInTheDocument();
    await waitFor(() => expect(screen.getAllByText(group).length).toBeGreaterThan(0), { timeout: 5000 });
  });

  it("Runs list renders rows and score columns from the URL", async () => {
    renderAt("/runs?cols=suite:olmes:base&series=olmo3-7b-midtrain");
    expect(await screen.findByRole("heading", { name: "Runs" })).toBeInTheDocument();
    await waitFor(() => expect(screen.getAllByText("olmo3-7b-midtrain").length).toBeGreaterThan(3), { timeout: 5000 });
    expect(screen.getAllByText("olmes:base").length).toBeGreaterThan(0);
  });

  it("Run overview shows the suite tree and health", async () => {
    renderAt(`/runs/${runId}`);
    expect(await screen.findByText("Suites and tasks", {}, { timeout: 5000 })).toBeInTheDocument();
    expect(await screen.findByText("Health")).toBeInTheDocument();
    await waitFor(() => expect(screen.getAllByText("olmes:base").length).toBeGreaterThan(0), { timeout: 5000 });
  });

  it.each([
    ["interrupted", "This run was interrupted before recording task results."],
    ["running", "This run is still running. Task results appear as they finish."],
    ["failed", "This run recorded no task results."],
  ] as const)("Run overview of a %s run without results says so", async (status, message) => {
    const target = world.runs.find((r) => r.summary.status === "interrupted")!.summary.run_id;
    server.use(
      http.get(`/api/runs/${target}`, () => HttpResponse.json({ ...runDetail(target), status })),
      http.get(`/api/runs/${target}/task-results`, ({ request }) =>
        HttpResponse.json({ ...runTaskResults(target, new URL(request.url).searchParams), items: [] }),
      ),
      http.get(`/api/runs/${target}/suites`, ({ request }) =>
        HttpResponse.json({ ...runSuites(target, new URL(request.url).searchParams), items: [], unsuited_tasks: [] }),
      ),
    );
    renderAt(`/runs/${target}`);
    expect(await screen.findByText(message, {}, { timeout: 8000 })).toBeInTheDocument();
  });

  it("Run tasks and config tabs render", async () => {
    renderAt(`/runs/${runId}?tab=tasks`);
    await waitFor(() => expect(screen.getAllByText("gsm8k:cot:olmo3").length).toBeGreaterThan(0), { timeout: 5000 });
    expect(screen.getAllByText("gsm8k:cot:olmo3")[0].closest('[role="row"]')).toHaveStyle({ height: "30px" });
  });

  it("Compare heatmap renders the group's subjects", async () => {
    renderAt(`/compare?group=${group}&scope=suite:olmes:base`);
    expect(await screen.findByText("Heatmap", { selector: "h2" }, { timeout: 8000 })).toBeInTheDocument();
    await waitFor(() => expect(screen.getAllByText("mmlu_anatomy:mc").length).toBeGreaterThan(0), { timeout: 8000 });
  });

  it("Compare leaves out unknown subjects instead of failing", async () => {
    const other = world.runs.find((r) => r.summary.run_id !== runId && r.model.series === "olmo3-7b-midtrain")!.summary.run_id;
    renderAt(`/compare?subjects=r:${runId},r:${other},r:zzzzzzzzzzzz&scope=suite:arc:rc`);
    expect(await screen.findByText(/1 unknown subject ignored/, {}, { timeout: 8000 })).toBeInTheDocument();
    expect(await screen.findByText("Heatmap", { selector: "h2" }, { timeout: 8000 })).toBeInTheDocument();
    expect(screen.queryByText(/Something went wrong/)).not.toBeInTheDocument();
  });

  it("Compare pairwise renders win rates", async () => {
    renderAt(`/compare?group=${group}&scope=suite:arc:rc&view=pairwise`);
    await waitFor(() => expect(screen.getAllByText(/% wins/).length).toBeGreaterThan(2), { timeout: 10000 });
    expect(screen.getByText("mean win rate")).toBeInTheDocument();
  });

  it("Compare disagreements list instances as a keyboard listbox", async () => {
    renderAt(`/compare?group=${group}&scope=suite:arc:rc&view=disagree`);
    const list = await screen.findByRole("listbox", { name: "Disagreeing instances" }, { timeout: 10000 });
    await waitFor(() => expect(list.querySelectorAll('[role="option"]').length).toBeGreaterThan(0), { timeout: 10000 });
    expect(screen.getAllByRole("button", { name: /instances\. List them\./ }).length).toBeGreaterThan(0);
  });

  it("Groups, Models and Tasks indexes list their rows with totals", async () => {
    renderAt("/groups");
    await waitFor(() => expect(screen.getAllByText(group).length).toBeGreaterThan(0), { timeout: 5000 });
    cleanup();
    renderAt("/models");
    await waitFor(() => expect(screen.getAllByText("olmo3-7b-midtrain").length).toBeGreaterThan(0), { timeout: 5000 });
    cleanup();
    renderAt("/tasks");
    await waitFor(() => expect(screen.getAllByText("gsm8k:cot:olmo3").length).toBeGreaterThan(0), { timeout: 5000 });
  });

  it("Run overview reports instances that could not be scored", async () => {
    const failures = {
      instances_unscored: 3,
      instances_partially_scored: 2,
      outputs_unscored: 7,
      infrastructure: 5,
      by_type: { JudgeParseError: 5, KeyError: 2 },
      first_error: "JudgeParseError: no parseable verdict after 3 judge attempts",
    };
    const task = runTaskResults(runId, new URLSearchParams()).items.find((row) => !row.error)!;
    server.use(
      http.get(`/api/runs/${runId}/task-results`, ({ request }) => {
        const result = runTaskResults(runId, new URL(request.url).searchParams);
        return HttpResponse.json({
          ...result,
          items: result.items.map((row) => (row.task_result_id === task.task_result_id ? { ...row, scoring_failures: failures } : row)),
        });
      }),
    );
    renderAt(`/runs/${runId}`);
    const line = await screen.findByText(/instances could not be scored/, {}, { timeout: 8000 });
    expect(line.textContent).toContain("scored from part of their outputs");
    expect(await screen.findByText(/JudgeParseError ×5, KeyError ×2/)).toBeInTheDocument();
  });

  it.each(["tasks", "overview"])("%s keeps baseline generation diagnostics beside its score", async (tab) => {
    const baseline = world.runs.find((r) => r.summary.run_id !== runId && r.model.series === run.model.series)!;
    const counts = { generations: 123, cap_hit: 47, empty: 9 };
    server.use(
      http.get(`/api/runs/${runId}/task-results`, ({ request }) => {
        const result = runTaskResults(runId, new URL(request.url).searchParams);
        return HttpResponse.json({ ...result, items: result.items.map((row) => ({ ...row, baseline: row.baseline ? { ...row.baseline, generation_counts: counts } : null })) });
      }),
      http.get(`/api/runs/${runId}/suites`, ({ request }) => {
        const result = runSuites(runId, new URL(request.url).searchParams);
        return HttpResponse.json({ ...result, items: result.items.map((row) => ({ ...row, baseline: row.baseline ? { ...row.baseline, generation_counts: counts } : null })) });
      }),
    );
    const params = new URLSearchParams({ baseline: `r:${baseline.summary.run_id}` });
    const task = runTaskResults(runId, params).items.find((row) => row.baseline)!;
    params.set("tab", tab);
    if (tab === "tasks") params.set("tq", task.task_name);
    renderAt(`/runs/${runId}?${params}`);
    const diagnostics = await screen.findAllByText("cap 47/123 · empty 9", {}, { timeout: 8000 });
    expect(diagnostics.length).toBeGreaterThan(0);
    if (tab === "tasks") expect(diagnostics[0].closest('[role="row"]')).toHaveStyle({ height: "72px" });
  });

  it("Tasks distinguish instance truncation from output cap hits", async () => {
    const result = runTaskResults(runId, new URLSearchParams());
    const task = result.items[0];
    server.use(http.get(`/api/runs/${runId}/task-results`, () => HttpResponse.json({
      ...result, items: [{ ...task, truncation_rate: 0.2, generation_counts: { generations: 1000, cap_hit: 20 }, baseline: null }],
    })));
    renderAt(`/runs/${runId}?tab=tasks`);
    expect(await screen.findByText("cap 20/1000 · empty 0")).toBeInTheDocument();
    const header = screen.getByRole("columnheader", { name: /Trunc. inst./ });
    expect(header).toHaveAttribute("title", expect.stringContaining("Stored instances"));
    expect(screen.getByText("20%")).toBeInTheDocument();
  });

  it("Tasks keep Score and Base compact without generation diagnostics", async () => {
    const result = runTaskResults(runId, new URLSearchParams());
    server.use(http.get(`/api/runs/${runId}/task-results`, () => HttpResponse.json({
      ...result, items: result.items.map((row) => ({ ...row, generation_counts: null, baseline: row.baseline ? { ...row.baseline, generation_counts: null } : null })),
    })));
    renderAt(`/runs/${runId}?tab=tasks`);
    const score = await screen.findByRole("columnheader", { name: /^Score/ });
    const headers = Array.from(score.parentElement!.children);
    const widths = score.parentElement!.style.gridTemplateColumns.split(" ");
    expect(widths[headers.indexOf(score)]).toBe("104px");
    expect(widths[headers.indexOf(screen.getByRole("columnheader", { name: /^Base/ }))]).toBe("86px");
  });

  it("Unknown runs show a not-found state", async () => {
    renderAt("/runs/zzzzzzzzzzzz");
    expect(await screen.findByText(/not found/i, {}, { timeout: 5000 })).toBeInTheDocument();
  });
});

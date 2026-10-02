import type { LeaderboardResponse, LeaderboardRow, MetricMeta } from "@contract/api-types";
import { useParams } from "@tanstack/react-router";
import { scaleLinear } from "d3-scale";
import { useMemo } from "react";
import { useSuiteDetail, useSuiteLeaderboard, useTaskDetail, useTaskLeaderboard } from "@/api/hooks/catalog";
import { useDistributions, useResolveSubjects } from "@/api/hooks/compare";
import { useRunsFacets } from "@/api/hooks/runs";
import { CIWhisker, HistogramChart } from "@/charts/basic";
import { ChartPanel, ChartTooltip, Legend, TipRow, useSize, useTooltip } from "@/charts/core";
import { AppLink, useAppNavigate } from "@/components/AppLink";
import { MetricName, ScoreValue, SubjectLabel } from "@/components/cells";
import { type Column, DataTable } from "@/components/DataTable";
import { JsonTree } from "@/components/Json";
import { MetaItem, PageHeader } from "@/components/PageHeader";
import { Badge, Chip, ErrorPanel, KV, Panel, RelativeTime, Segmented, Select, Skeleton } from "@/components/primitives";
import { formatCount, formatScore, metricLabel, shortHash } from "@/lib/format";
import { catColor, extent, normalize, seqColor, seqTextColor } from "@/lib/scales";
import { modelLabel } from "@/lib/subjects";
import { useSubjectSlots } from "@/state/colors";
import { useBaseline, useSearchParams } from "@/state/nav";
import { addToTray, pushRecent, removeFromTray, trayStore } from "@/state/prefs";
import { useStore } from "@/state/store";
import { useEffect } from "react";
import cc from "@/charts/charts.module.css";

type LbSearch = { hash?: string; metric?: string; pm?: string; family?: string; group?: string; user?: string; sort?: string };

function histogramOf(values: number[], bins = 24) {
  if (!values.length) return { edges: [0, 1], counts: [0] };
  const lo = Math.min(...values);
  const hi = Math.max(...values) + 1e-9;
  const w = (hi - lo) / bins || 1;
  const counts = new Array(bins).fill(0);
  for (const v of values) counts[Math.min(bins - 1, Math.floor((v - lo) / w))] += 1;
  return { edges: Array.from({ length: bins + 1 }, (_, i) => lo + i * w), counts };
}

function ScoreOverTime({ rows, meta }: { rows: LeaderboardRow[]; meta: MetricMeta | null }) {
  const [ref, { width }] = useSize<HTMLDivElement>();
  const tip = useTooltip();
  const navigate = useAppNavigate();
  const fmt = meta?.display_format ?? "percent";
  const families = Object.entries(rows.reduce<Record<string, number>>((acc, r) => ((acc[r.model.family ?? "other"] = (acc[r.model.family ?? "other"] ?? 0) + 1), acc), {}))
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([f]) => f);
  const height = 200;
  const m = { top: 8, right: 12, bottom: 22, left: 40 };
  const innerW = Math.max(10, width - m.left - m.right);
  const innerH = height - m.top - m.bottom;
  const ts = rows.map((r) => new Date(r.date).getTime());
  const vals = rows.map((r) => r.score ?? 0);
  const x = scaleLinear().domain([Math.min(...ts), Math.max(...ts) + 1]).range([0, innerW]);
  const ext = extent(vals) ?? [0, 1];
  const y = scaleLinear().domain([ext[0] - (ext[1] - ext[0]) * 0.1, ext[1] + (ext[1] - ext[0]) * 0.1 || 1]).range([innerH, 0]);
  const colorOf = (r: LeaderboardRow) => {
    const i = families.indexOf(r.model.family ?? "other");
    return i >= 0 ? catColor(i) : "var(--cat-context)";
  };
  return (
    <div ref={ref}>
      <Legend items={[...families.map((f, i) => ({ label: f, color: catColor(i) })), { label: "other", color: "var(--cat-context)" }]} />
      {width > 0 && (
        <svg width={width} height={height} onMouseLeave={tip.hide} style={{ marginTop: 6 }}>
          <g transform={`translate(${m.left},${m.top})`}>
            {y.ticks(4).map((t) => (
              <g key={t}>
                <line x1={0} x2={innerW} y1={y(t)} y2={y(t)} stroke="var(--grid)" />
                <text x={-6} y={y(t)} dy="0.32em" textAnchor="end" className={cc.axisMono}>
                  {formatScore(t, fmt)}
                </text>
              </g>
            ))}
            {x.ticks(4).map((t) => (
              <text key={t} x={x(t)} y={innerH + 15} textAnchor="middle" className={cc.axisMono}>
                {new Date(t).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
              </text>
            ))}
            {rows.map((r) => (
              <circle
                key={r.task_result_id ?? r.run_id}
                cx={x(new Date(r.date).getTime())}
                cy={y(r.score ?? 0)}
                r={5}
                fill={colorOf(r)}
                stroke="var(--surface)"
                strokeWidth={2}
                style={{ cursor: "pointer" }}
                onClick={() => navigate(`/runs/${r.run_id}`)}
                onMouseMove={(e) =>
                  tip.show(
                    e,
                    <>
                      <div style={{ fontWeight: 800 }}>{modelLabel(r.model)}</div>
                      <TipRow label="score" value={formatScore(r.score, fmt)} />
                      <TipRow label="date" value={new Date(r.date).toLocaleDateString()} />
                      <TipRow label="family" value={r.model.family ?? "—"} />
                    </>,
                  )
                }
              />
            ))}
          </g>
        </svg>
      )}
      <ChartTooltip state={tip.state} />
    </div>
  );
}

function LeaderboardBody({ kind, name, lb, meta }: { kind: "task" | "suite"; name: string; lb: ReturnType<typeof useTaskLeaderboard>; meta: MetricMeta | null }) {
  const navigate = useAppNavigate();
  const [baseline] = useBaseline();
  const [search, setSearch] = useSearchParams<LbSearch>();
  const tray = useStore(trayStore);
  const slots = useSubjectSlots(tray.map((t) => t.key));
  const data: LeaderboardResponse | undefined = lb.data;
  const rows = useMemo(() => data?.items ?? [], [data]);
  const fmt = meta?.display_format ?? "percent";
  const domain = useMemo<[number, number]>(() => {
    const ext = extent(rows.flatMap((r) => [r.ci_low ?? r.score, r.ci_high ?? r.score]));
    if (!ext) return [0, 1];
    const pad = (ext[1] - ext[0]) * 0.05 || 0.01;
    return [ext[0] - pad, ext[1] + pad];
  }, [rows]);
  const children = data?.children ?? [];
  const childExt = children.map((ch) => extent(rows.map((r) => r.child_scores?.[`${ch.type}:${ch.name}`] ?? null)));
  const selected = new Set(rows.filter((r) => tray.some((t) => t.key === r.subject)).map((r) => r.subject));
  const columns: Column<LeaderboardRow>[] = [
    { id: "rank", header: "#", title: "Rank", width: 46, align: "right", pin: true, sortValue: (r) => r.rank, cell: (r) => <span className="mono muted">{r.rank}</span>, csv: (r) => r.rank },
    {
      id: "model",
      header: "Model",
      title: "Model",
      width: 250,
      pin: true,
      sortValue: (r) => r.model.series_label,
      cell: (r) => (
        <AppLink href={`/runs/${r.run_id}`} className="truncate" style={{ display: "inline-flex", minWidth: 0 }}>
          <SubjectLabel model={r.model} slot={slots[r.subject]} baseline={baseline === r.subject} />
        </AppLink>
      ),
      csv: (r) => r.model.name,
    },
    { id: "score", header: "Score", title: "Score", width: 104, align: "right", sortValue: (r) => r.score, cell: (r) => <ScoreValue score={r.score} stderr={r.stderr} format={fmt} strong={r.rank === 1} />, csv: (r) => r.score },
    {
      id: "ci",
      header: "95% CI",
      title: "Dot and whisker on a shared axis",
      width: 240,
      cell: (r) => (
        <span className="row" style={{ gap: 6 }}>
          <CIWhisker value={r.score} low={r.ci_low} high={r.ci_high} domain={domain} width={150} color={baseline === r.subject ? "var(--teal-line)" : "var(--fg-2)"} />
          {r.tied_with_leader && <Badge tone="outline">tied #1</Badge>}
        </span>
      ),
      csv: (r) => `${r.ci_low ?? ""}..${r.ci_high ?? ""}`,
    },
    ...children.map(
      (ch, j): Column<LeaderboardRow> => ({
        id: `child:${ch.name}`,
        header: <span style={{ textTransform: "none", letterSpacing: 0 }}>{ch.name}</span>,
        title: ch.name,
        width: 96,
        align: "center",
        sortValue: (r) => r.child_scores?.[`${ch.type}:${ch.name}`],
        cell: (r) => {
          const v = r.child_scores?.[`${ch.type}:${ch.name}`];
          if (v == null) return <span className="faint">—</span>;
          const t = normalize(v, childExt[j]);
          return (
            <span className="mono" style={{ background: seqColor(t), color: seqTextColor(t), padding: "2px 8px", borderRadius: 3, fontSize: 12, width: "100%", textAlign: "center" }}>
              {formatScore(v)}
            </span>
          );
        },
        csv: (r) => r.child_scores?.[`${ch.type}:${ch.name}`],
      }),
    ),
    { id: "n", header: "n", title: "Instances", width: 76, align: "right", sortValue: (r) => r.n, cell: (r) => <span className="mono muted">{formatCount(r.n)}</span>, csv: (r) => r.n },
    { id: "family", header: "Family", title: "Family", width: 90, sortValue: (r) => r.model.family, cell: (r) => <span className="muted">{r.model.family ?? "—"}</span>, csv: (r) => r.model.family },
    { id: "group", header: "Group", title: "Group", width: 190, sortValue: (r) => r.experiment_group, cell: (r) => <span className="truncate muted">{r.experiment_group ?? "—"}</span>, csv: (r) => r.experiment_group },
    { id: "hash", header: "Variant", title: "Variant", width: 84, defaultHidden: kind === "suite", cell: (r) => <span className="mono muted" style={{ fontSize: 11.5 }}>{shortHash(r.task_hash)}</span>, csv: (r) => r.task_hash },
    { id: "date", header: "Date", title: "Date", width: 90, sortValue: (r) => r.date, cell: (r) => <RelativeTime value={r.date} className="muted" />, csv: (r) => r.date },
  ];
  const baseRow = rows.find((r) => r.subject === baseline);
  const dist = useDistributions(kind === "task" ? { items: [{ task_name: name, task_hash: data?.task_hash ?? null, metric: data?.metric ?? "primary" }], per_model: "all" } : null);
  const distValues = kind === "task" ? (dist.data?.items[0]?.scores ?? []) : rows.map((r) => r.score).filter((v): v is number => v != null);
  return (
    <>
      <Panel title="Leaderboard" caption="Latest result per model by default. Ties: the row's 95% CI overlaps the leader's. Select rows to add them to the compare tray." pad="none" refetching={lb.isFetching && !lb.isLoading}>
        {lb.error ? (
          <ErrorPanel error={lb.error} onRetry={() => lb.refetch()} />
        ) : (
          <DataTable
            tableId={`leaderboard-${kind}`}
            columns={columns}
            rows={rows}
            getRowId={(r) => r.subject + r.run_id}
            loading={lb.isLoading}
            sort={search.sort}
            onSortChange={(v) => setSearch({ sort: v }, { replace: true })}
            clientSort
            selected={new Set(rows.filter((r) => selected.has(r.subject)).map((r) => r.subject + r.run_id))}
            onSelectionChange={(next) => {
              for (const r of rows) {
                const id = r.subject + r.run_id;
                if (next.has(id) && !selected.has(r.subject)) addToTray({ key: r.subject, label: modelLabel(r.model) });
                if (!next.has(id) && selected.has(r.subject)) removeFromTray(r.subject);
              }
            }}
            onRowOpen={(r) => navigate(`/runs/${r.run_id}${kind === "task" ? `?tab=tasks&task=${encodeURIComponent(name)}` : ""}`)}
            rowClassName={(r) => (r.subject === baseline ? "" : undefined)}
            total={data?.total}
            maxHeight={560}
            exportName={`${name}-leaderboard`}
          />
        )}
      </Panel>
      <div className="grid-12">
        <div className="span-6">
          <ChartPanel title="Score distribution across runs" caption={kind === "task" ? "Every finalized run of this variant." : "Latest result per model."}>
            {dist.isLoading && kind === "task" ? (
              <Skeleton height={170} />
            ) : (
              <HistogramChart
                series={[{ data: histogramOf(distValues), color: "var(--seq-3)", label: "runs", style: "fill" }]}
                markers={baseRow?.score != null ? [{ value: baseRow.score, color: "var(--teal-line)", label: "baseline" }] : []}
                formatX={(v) => formatScore(v, fmt)}
                xLabel={metricLabel(data?.metric === "primary" ? undefined : data?.metric) === "—" ? "score" : metricLabel(data?.metric)}
              />
            )}
          </ChartPanel>
        </div>
        <div className="span-6">
          <ChartPanel title="Score over time" caption="Each point is a model's result by run date, colored by the three most common families.">
            {lb.isLoading ? <Skeleton height={200} /> : <ScoreOverTime rows={rows} meta={meta} />}
          </ChartPanel>
        </div>
      </div>
    </>
  );
}

function Filters({
  search,
  setSearch,
  scopeFilter,
}: {
  search: LbSearch;
  setSearch: (p: Partial<LbSearch>, o?: { replace?: boolean }) => void;
  /** Runs filter for facet counts: the task or suite this leaderboard covers. */
  scopeFilter: { task?: string; suite?: string; suite_coverage?: string };
}) {
  const facets = useRunsFacets(scopeFilter);
  const values = (key: "family" | "group" | "user") =>
    (facets.data?.[key] ?? []).map((f) => ({ value: f.value, label: f.value, hint: String(f.count) }));
  return (
    <div className="row-wrap" style={{ gap: 8 }}>
      <Segmented
        value={search.pm === "all" ? "all" : "latest"}
        onChange={(v) => setSearch({ pm: v === "latest" ? undefined : v }, { replace: true })}
        label="Rows"
        options={[
          { value: "latest", label: "Latest per model" },
          { value: "all", label: "All runs" },
        ]}
      />
      <Select size="sm" label="Family" value={search.family ?? ""} onChange={(v) => setSearch({ family: v || undefined })} options={[{ value: "", label: "any" }, ...values("family")]} />
      <Select size="sm" label="Group" value={search.group ?? ""} onChange={(v) => setSearch({ group: v || undefined })} options={[{ value: "", label: "any" }, ...values("group")]} width={240} />
      <Select size="sm" label="User" value={search.user ?? ""} onChange={(v) => setSearch({ user: v || undefined })} options={[{ value: "", label: "any" }, ...values("user")]} />
    </div>
  );
}

export function TaskPage() {
  const { taskName } = useParams({ strict: false }) as { taskName: string };
  const name = decodeURIComponent(taskName);
  const [search, setSearch] = useSearchParams<LbSearch>();
  const detail = useTaskDetail(name);
  const variant = search.hash ?? detail.data?.variants[0]?.task_hash;
  const v = detail.data?.variants.find((x) => x.task_hash === variant);
  const metric = search.metric ?? "primary";
  const lb = useTaskLeaderboard({ task: name, hash: variant, metric, per_model: search.pm, family: search.family, group: search.group, user: search.user, limit: 500 });
  useEffect(() => pushRecent({ type: "task", key: name, label: name, href: `/tasks/${encodeURIComponent(name)}` }), [name]);
  if (detail.error) return <ErrorPanel error={detail.error} />;
  const meta = v ? v.metric_meta[metric === "primary" ? (v.primary_metric ?? "") : metric] ?? null : null;
  return (
    <div className="page">
      <PageHeader
        crumbs={[{ label: "Tasks", href: "/tasks" }]}
        title={name}
        meta={
          <>
            {detail.data?.suites.length ? (
              <MetaItem label="in suites">
                {detail.data.suites.map((s) => (
                  <AppLink key={s} href={`/suites/${encodeURIComponent(s)}`} className="link">
                    {s}
                  </AppLink>
                ))}
              </MetaItem>
            ) : null}
            {v && (
              <MetaItem label="primary">
                <MetricName metric={v.primary_metric} />
              </MetaItem>
            )}
            {v?.n_instances != null && <MetaItem>{formatCount(v.n_instances)} instances</MetaItem>}
            {v?.num_fewshot != null && <MetaItem>{v.num_fewshot}-shot</MetaItem>}
            {meta?.higher_is_better === false && <Badge tone="muted">lower is better</Badge>}
          </>
        }
      >
        {detail.data && (
          <div className="row-wrap" style={{ gap: 6 }}>
            <span className="t-overline">Variants</span>
            {detail.data.variants.map((x) => (
              <Chip
                key={x.task_hash}
                value={
                  <>
                    <span className="mono">{shortHash(x.task_hash)}</span> <span className="muted">({x.n_runs} runs{x.limit != null ? `, limit ${x.limit}` : ""})</span>
                  </>
                }
                active={x.task_hash === variant}
                onClick={() => setSearch({ hash: x.task_hash === detail.data!.variants[0].task_hash ? undefined : x.task_hash })}
              />
            ))}
            <span className="spacer" />
            <Select
              size="sm"
              label="Metric"
              value={metric}
              onChange={(m) => setSearch({ metric: m === "primary" ? undefined : m })}
              options={[{ value: "primary", label: `primary (${metricLabel(v?.primary_metric)})` }, ...Object.keys(v?.metric_meta ?? {}).map((k) => ({ value: k, label: metricLabel(k) }))]}
            />
          </div>
        )}
      </PageHeader>
      <Filters search={search} setSearch={setSearch} scopeFilter={{ task: name }} />
      <LeaderboardBody kind="task" name={name} lb={lb} meta={meta} />
      {v && (
        <div className="grid-12">
          <div className="span-5">
            <Panel title="About this variant">
              <KV
                items={[
                  ["Variant", <span className="mono">{v.task_hash}</span>],
                  ["Primary metric", <MetricName metric={v.primary_metric} />],
                  ["Metrics", <span className="mono" style={{ fontSize: 11.5 }}>{Object.keys(v.metric_meta).map(metricLabel).join(", ")}</span>],
                  ["Split", v.split],
                  ["Few-shot", v.num_fewshot],
                  ["Limit", v.limit ?? "none"],
                  ["Instances", formatCount(v.n_instances)],
                  ["Runs", formatCount(v.n_runs)],
                  ["First seen", <RelativeTime value={v.first_seen_at} />],
                  ["Last seen", <RelativeTime value={v.last_seen_at} />],
                ]}
              />
            </Panel>
          </div>
          <div className="span-7">
            <Panel title="Config">
              <JsonTree value={v.config} filename={`${name}-${v.task_hash}.json`} maxHeight={360} />
            </Panel>
          </div>
        </div>
      )}
    </div>
  );
}

export function SuitePage() {
  const { suiteName } = useParams({ strict: false }) as { suiteName: string };
  const name = decodeURIComponent(suiteName);
  const [search, setSearch] = useSearchParams<LbSearch>();
  const detail = useSuiteDetail(name);
  const lb = useSuiteLeaderboard({ suite: name, per_model: search.pm, family: search.family, group: search.group, user: search.user, limit: 500 });
  const [baseline] = useBaseline();
  const resolved = useResolveSubjects(baseline ? [baseline] : []);
  useEffect(() => pushRecent({ type: "suite", key: name, label: name, href: `/suites/${encodeURIComponent(name)}` }), [name]);
  if (detail.error) return <ErrorPanel error={detail.error} />;
  const d = detail.data;
  const leaves: string[] = [];
  const walk = (n: NonNullable<typeof d>["tree"]) => (n.type === "task" ? leaves.push(n.name) : n.children.forEach(walk));
  if (d) walk(d.tree);
  return (
    <div className="page">
      <PageHeader
        crumbs={[{ label: "Suites", href: "/tasks?tab=suites" }]}
        title={name}
        meta={
          d ? (
            <>
              <MetaItem label="aggregation">{d.aggregation.replace(/_/g, " ")}</MetaItem>
              <MetaItem>{d.tree.children.length} children</MetaItem>
              <MetaItem>{leaves.length} tasks</MetaItem>
              <MetaItem>{formatCount(d.n_runs)} runs</MetaItem>
              {d.definitions_seen > 1 && <Badge tone="warn">{d.definitions_seen} definitions seen</Badge>}
              {resolved.data?.items[0] && <MetaItem label="baseline">{resolved.data.items[0].label}</MetaItem>}
            </>
          ) : (
            <Skeleton width={300} height={14} />
          )
        }
      >
        {d?.description && <p className="muted" style={{ fontSize: 13 }}>{d.description}</p>}
        {d && (
          <div className="row-wrap" style={{ gap: 4 }}>
            {d.tree.children.map((ch) => (
              <AppLink key={ch.name} href={ch.type === "suite" ? `/suites/${encodeURIComponent(ch.name)}` : `/tasks/${encodeURIComponent(ch.name)}`}>
                <Badge tone={ch.type === "suite" ? "teal" : "muted"}>{ch.name}</Badge>
              </AppLink>
            ))}
          </div>
        )}
      </PageHeader>
      <Filters search={search} setSearch={setSearch} scopeFilter={{ suite: name, suite_coverage: "full" }} />
      <LeaderboardBody kind="suite" name={name} lb={lb} meta={lb.data?.meta ?? null} />
    </div>
  );
}

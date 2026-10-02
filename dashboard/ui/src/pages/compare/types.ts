import type { MatrixResponse, SubjectInfo, SubjectKey } from "@contract/api-types";
import type { UseQueryResult } from "@tanstack/react-query";

export type CompareSearch = {
  subjects?: string;
  group?: string;
  merge?: string;
  scope?: string;
  metric?: string;
  view?: string;
  mode?: string;
  alpha?: string;
  margin?: string;
  shared?: string;
  a?: string;
  b?: string;
  task?: string;
  cell?: string;
  filter?: string;
  rsort?: string;
  scale?: string;
  inst?: string;
  x?: string;
  size?: string;
  baseline?: string;
};

export const VIEWS = ["heatmap", "pairwise", "scatter", "profiles", "progression", "disagree"] as const;
export type CompareView = (typeof VIEWS)[number];

export interface CompareCtx {
  keys: SubjectKey[];
  subjects: SubjectInfo[];
  baseline?: SubjectKey;
  group?: string;
  scope: string;
  metric: string;
  alpha: number;
  shared: boolean;
  matrix: UseQueryResult<MatrixResponse>;
  slots: Record<string, number | null>;
  search: CompareSearch;
  setSearch: (patch: Partial<CompareSearch>, options?: { replace?: boolean }) => void;
  labelOf: (key: string) => string;
}

/** Short column label: series label plus compact step. */
export function shortLabel(info: SubjectInfo | undefined, fallback: string): string {
  if (!info) return fallback;
  const step = info.model.step;
  const base = info.model.series_label;
  if (step == null) return info.label.includes(" · ") ? info.label : base;
  const s = step >= 1000 && step % 1000 === 0 ? `${step / 1000}k` : step.toLocaleString("en-US");
  return `${base} @${s}`;
}

/**
 * Two-line column label for dense matrices: the step ("@11k") on top and the series below, so
 * checkpoints of one series stay distinguishable in narrow columns.
 */
export function columnLabel(ctx: Pick<CompareCtx, "subjects" | "labelOf">, key: string): { main: string; sub: string } {
  const info = ctx.subjects.find((x) => x.key === key);
  const full = ctx.labelOf(key);
  if (!info) return { main: full, sub: "" };
  if (info.model.step != null) return { main: full.replace(`${info.model.series_label} `, ""), sub: info.model.series_label };
  return { main: info.model.series_label, sub: info.kind === "model" ? `${info.run_ids.length} run${info.run_ids.length === 1 ? "" : "s"}` : "run" };
}

#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair~=6.2",
#     "numpy~=2.4",
#     "polars~=1.43",
#     "vl-convert-python~=1.9",
# ]
# ///
"""Turn the canonical science-eval results ledger into slide-ready charts.

Reads `data/results.csv` plus the hand-maintained metadata in `metadata.toml`,
prefers fixed dev-set results for agentic evals, and writes the analysis deck.

    uv run scripts/analysis/eval_summary.py

See plans/007_scilit_eval_summary.md for the data flow and the scoring rules.
"""

from __future__ import annotations

import argparse
import base64
import math
import tomllib
from dataclasses import dataclass
from html import escape
from pathlib import Path

import altair as alt
import numpy as np
import polars as pl

HERE = Path(__file__).parent

# Calm, cool neutrals and restrained accents keep the dense result slides
# legible while giving the deck a coherent presentation palette.
SURFACE = "#f8fafc"
INK = "#172033"
INK_SECONDARY = "#5f687a"
MUTED = "#8c95a5"
GRID = "#e5e9f0"
AXIS = "#bcc4d1"

PRIMARY = "#405bd8"
DEEMPHASIS = "#c9cfd9"

# Validated 4-slot categorical (adjacent pairlist, light mode). Aqua and yellow
# sit below 3:1 on the surface, so lines carrying them must be direct-labelled.
SERIES = [
    "#405bd8",
    "#df6b66",
    "#269587",
    "#835cc7",
    "#c1842f",
    "#c85b91",
    "#3185a6",
]
MODEL_COLORS = [*SERIES, "#697a27"]

# Reserved status palette — only the coverage grid uses it. The score bars use
# emphasis (one hue + gray) instead, so a green bar never reads as "good score"
# when it only means "valid row".
STATUS_COLOR = {
    "ok": "#288f71",
    "suspect": "#cf5353",
    "unsupported": "#778195",
    "not-run": GRID,
}
STATUS_ORDER = ["ok", "suspect", "unsupported", "not-run"]

# Provenance remains available in the ledger and tooltips, but is not a score
# validity state. Only `suspect` is withheld from numerical analysis.
RANKABLE_STATUSES = ("ok",)

TRUST_ORDER = ["usable", "suspect"]

# Secondary blue for the sentinel panel. Grey is spoken for — it means
# "suspect" on the score charts, and the two must not collide across slides.
SENTINEL = "#86a6df"

COLUMN_RENAMES = {
    "Model Name": "model",
    "Eval Name": "eval",
    "Metric": "metric",
    "Score": "score",
    "Beaker Run ID": "run_id",
    "Notes": "notes",
    "Valid for analysis": "valid_for_analysis",
    "replica": "replica",
}

DEV_TO_CANONICAL = {
    "ExpertQA-dev100": "ExpertQA",
    "LitSearch-open-dev50": "LitSearch-open",
    "SAGE-open-dev50": "SAGE-open",
    "SAGE-short-dev50": "SAGE-short",
    "DeepScholar-Bench-dev10": "DeepScholar-Bench",
}
CANONICAL_TO_DEV = {canonical: dev for dev, canonical in DEV_TO_CANONICAL.items()}
AGENTIC_EVALS = list(DEV_TO_CANONICAL.values())
BASE_EVALS = ["ARC", "MMLU-STEM", "MedMCQA", "MedQA", "SciQ"]
SENTINEL_EVALS = ["IFEval", "MMLU", "MATH-500"]
FRONTIER_EVALS = [
    "FS-Olympiad accuracy",
    "FS-Research success",
    "FS-Research rubric",
]
# Candidate predictors include the base-compatible suite plus direct and
# judged post-training proxies. Protocol differences remain visible in the
# source ledger rather than being treated as additional evaluations.
PROXY_EVALS = [*BASE_EVALS, "LitSearch-rerank", *FRONTIER_EVALS, *SENTINEL_EVALS]
# Operationally cheap relative to DeepScholar: no tools, no external judge,
# and already fast enough to run in full in this sweep.
CHEAP_EVALS = [*BASE_EVALS, "LitSearch-rerank", *SENTINEL_EVALS]
ALL_ANALYSIS_EVALS = [*AGENTIC_EVALS, *PROXY_EVALS]
COMPACT_EVAL_LABELS = {
    "ExpertQA": "ExpertQA",
    "LitSearch-open": "LitSearch-open",
    "SAGE-open": "SAGE-open",
    "SAGE-short": "SAGE-short",
    "DeepScholar-Bench": "DeepScholar",
}
FAMILY_ORDER = ["sci-lit", "base", "frontier-science", "sentinel"]
FAMILY_LABELS = {
    "sci-lit": "Sci-lit",
    "base": "Base",
    "frontier-science": "FrontierScience",
    "sentinel": "Sentinel",
}
FAMILY_COLOR = {
    "sci-lit": PRIMARY,
    "base": SERIES[3],
    "frontier-science": SERIES[2],
    "sentinel": SENTINEL,
}
FRONTIER_METRIC_EVALS = {
    ("FrontierScience-Olympiad", "accuracy:frontierscience_judge"): FRONTIER_EVALS[0],
    ("FrontierScience-Research", "success_rate:frontierscience_judge"): FRONTIER_EVALS[1],
    ("FrontierScience-Research", "rubric_score:frontierscience_judge"): FRONTIER_EVALS[2],
}
PRIMARY_METRICS = {
    "ExpertQA": "citation_recall:sqa_judge",
    "LitSearch-open": "found_rate:exact_match",
    "LitSearch-rerank": "recall@5:litsearch_rerank",
    "SAGE-open": "weighted_recall:weighted_recall",
    "SAGE-short": "exact_match:exact_match",
    "DeepScholar-Bench": "geomean_fixed:external",
    "ARC": "accuracy",
    "MMLU-STEM": "accuracy",
    "MedMCQA": "accuracy",
    "MedQA": "accuracy",
    "SciQ": "accuracy",
    "IFEval": "prompt_level_loose_acc:ifeval",
    "MMLU": "primary_score:average",
    "MATH-500": "accuracy:minerva_math_flex",
    "FS-Olympiad accuracy": "accuracy:frontierscience_judge",
    "FS-Research success": "success_rate:frontierscience_judge",
    "FS-Research rubric": "rubric_score:frontierscience_judge",
}
INVALID_NOTE_MARKERS = ("legacy stock tool template",)


def eval_family(eval_name: str) -> str:
    if eval_name in BASE_EVALS:
        return "base"
    if eval_name in FRONTIER_EVALS:
        return "frontier-science"
    if eval_name in SENTINEL_EVALS:
        return "sentinel"
    return "sci-lit"


@dataclass(frozen=True)
class Metadata:
    evals: pl.DataFrame
    models: pl.DataFrame
    flags: pl.DataFrame

    @property
    def eval_order(self) -> list[str]:
        return self.evals["eval"].to_list()

    @property
    def model_order(self) -> list[str]:
        return self.models.filter(pl.col("included"))["model"].to_list()

    def sci_lit_evals(self) -> list[str]:
        return self.evals.filter(pl.col("family") == "sci-lit")["eval"].to_list()


def load_metadata(path: Path) -> Metadata:
    """Read eval/model/flag metadata, preserving declaration order for evals."""
    raw = tomllib.loads(path.read_text())

    evals = pl.DataFrame(
        [
            {"eval": name, "eval_index": i, **body}
            for i, (name, body) in enumerate(raw["evals"].items())
        ]
    )
    models = pl.DataFrame(
        [
            {"model": name, "included": body.get("included", True), **body}
            for name, body in raw["models"].items()
        ]
    ).sort(["total_b", "model"])
    models = models.with_row_index("model_index")

    flag_rows = raw.get("flags", [])
    flags = (
        pl.DataFrame(flag_rows).rename(
            {"status": "flag_status", "reason": "flag_reason", "verified": "flag_verified"}
        )
        if flag_rows
        else pl.DataFrame(
            schema={
                "model": pl.String,
                "eval": pl.String,
                "flag_status": pl.String,
                "flag_reason": pl.String,
                "flag_verified": pl.Boolean,
            }
        )
    )
    return Metadata(evals=evals, models=models, flags=flags)


def _blank_to_null(column: str) -> pl.Expr:
    stripped = pl.col(column).str.strip_chars()
    return pl.when(stripped.str.len_chars() > 0).then(stripped).otherwise(None).alias(column)


def _attach_status(scores: pl.DataFrame, meta: Metadata) -> pl.DataFrame:
    joined = scores.join(meta.flags, on=["model", "eval"], how="left")
    status = (
        pl.when(pl.col("flag_status") == "unsupported")
        .then(pl.lit("unsupported"))
        .when(pl.col("score").is_not_null() & ~pl.col("valid_for_analysis"))
        .then(pl.lit("suspect"))
        .when(pl.col("score").is_not_null() & pl.col("flag_status").is_not_null())
        .then(pl.col("flag_status"))
        .when(pl.col("score").is_null())
        .then(pl.lit("not-run"))
        .otherwise(pl.lit("ok"))
        .alias("status")
    )
    return joined.with_columns(status).with_columns(
        trust=pl.when(pl.col("status") == "suspect")
        .then(pl.lit("suspect"))
        .otherwise(pl.lit("usable")),
        reason=pl.when(pl.col("score").is_not_null() & ~pl.col("valid_for_analysis"))
        .then(pl.col("notes").fill_null("Excluded by Valid for analysis=False"))
        .otherwise(pl.col("flag_reason").fill_null("")),
    )


def load_scores(csv_path: Path, meta: Metadata) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load primary metrics, preferring fixed dev scopes for agentic evals.

    The returned analysis frame has exactly one row per included model/eval.
    Only explicitly numbered replicas are included. Repeated validated runs of
    the same scope are averaged. The second frame retains both trusted full and
    dev aggregates for paired comparisons. The third retains one primary score
    per run at the analysis-selected scope for replica-expanded covariance.
    """
    analysis_eval = pl.col("eval").replace_strict(DEV_TO_CANONICAL, default=pl.col("eval"))
    for (source_eval, metric), display_eval in FRONTIER_METRIC_EVALS.items():
        analysis_eval = (
            pl.when((pl.col("eval") == source_eval) & (pl.col("metric") == metric))
            .then(pl.lit(display_eval))
            .otherwise(analysis_eval)
        )
    # Keep FrontierScience's subject-level metrics in the long-form ledger
    # without letting them appear as unknown evaluations. Only the exact
    # primary metrics survive ``metric_ok`` below; this mapping simply assigns
    # every auxiliary metric to its parent analysis evaluation first.
    analysis_eval = (
        pl.when(pl.col("eval") == "FrontierScience-Olympiad")
        .then(pl.lit(FRONTIER_EVALS[0]))
        .when(
            (pl.col("eval") == "FrontierScience-Research")
            & pl.col("metric").str.starts_with("success_rate")
        )
        .then(pl.lit(FRONTIER_EVALS[1]))
        .when(
            (pl.col("eval") == "FrontierScience-Research")
            & pl.col("metric").str.starts_with("rubric_score")
        )
        .then(pl.lit(FRONTIER_EVALS[2]))
        .otherwise(analysis_eval)
    )

    raw = (
        pl.read_csv(csv_path, infer_schema_length=0)
        .rename(COLUMN_RENAMES)
        .with_columns(
            _blank_to_null("model"),
            _blank_to_null("eval"),
            _blank_to_null("metric"),
            _blank_to_null("run_id"),
            _blank_to_null("notes"),
            _blank_to_null("valid_for_analysis"),
            _blank_to_null("replica"),
        )
        .with_columns(
            pl.col("score").str.strip_chars().cast(pl.Float64, strict=False),
            pl.col("replica").cast(pl.Int64, strict=False),
            valid_for_analysis=pl.col("valid_for_analysis")
            .fill_null("true")
            .str.to_lowercase()
            .replace_strict({"true": True, "false": False}),
        )
        .with_columns(
            source_eval=pl.col("eval"),
            scope=pl.when(pl.col("eval").is_in(list(DEV_TO_CANONICAL)))
            .then(pl.lit("dev"))
            .otherwise(pl.lit("full")),
            eval=analysis_eval,
        )
    )

    unknown_models = set(raw["model"].drop_nulls()) - set(meta.models["model"])
    unknown_evals = set(raw["eval"].drop_nulls()) - set(meta.eval_order)
    if unknown_models or unknown_evals:
        raise ValueError(
            f"CSV has entries missing from {HERE / 'metadata.toml'}: "
            f"models={sorted(unknown_models)} evals={sorted(unknown_evals)}"
        )

    expected_metric = pl.col("eval").replace_strict(PRIMARY_METRICS)
    metric_ok = (pl.col("metric") == expected_metric) | (
        (pl.col("scope") == "full") & pl.col("metric").is_null()
    )
    invalid_note = (
        pl.col("notes")
        .fill_null("")
        .str.to_lowercase()
        .str.contains("|".join(INVALID_NOTE_MARKERS))
    )
    included_models = meta.models.filter(pl.col("included"))["model"].to_list()
    candidates = raw.filter(
        pl.col("model").is_in(included_models)
        & pl.col("score").is_not_null()
        & pl.col("replica").is_not_null()
        & metric_ok
        & ~invalid_note
    )

    # Collapse any duplicate representations of a primary metric within a run
    # before averaging across runs. This makes the bar-chart aggregation an
    # explicit mean of replicas rather than a mean of raw ledger rows.
    run_scores = (
        candidates.group_by("model", "eval", "scope", "run_id")
        .agg(
            score=pl.col("score").mean(),
            replica=pl.col("replica").first(),
            metric=pl.col("metric").drop_nulls().first(),
            notes=pl.col("notes").drop_nulls().first(),
            valid_for_analysis=pl.col("valid_for_analysis").all(),
        )
        .sort(["model", "eval", "scope", "replica", "run_id"])
    )
    aggregates = (
        run_scores.group_by("model", "eval", "scope")
        .agg(
            score=pl.col("score").mean(),
            score_min=pl.col("score").min(),
            score_max=pl.col("score").max(),
            score_std=pl.col("score").std(),
            run_count=pl.col("run_id").drop_nulls().n_unique(),
            run_id=pl.col("run_id").drop_nulls().first(),
            metric=pl.col("metric").drop_nulls().first(),
            notes=pl.col("notes").drop_nulls().first(),
            valid_for_analysis=pl.col("valid_for_analysis").all(),
        )
        .with_columns(pl.col("score_std").fill_null(0.0))
    )

    scope_scores = _attach_status(
        aggregates.join(meta.evals, on="eval", how="left").join(
            meta.models, on="model", how="left"
        ),
        meta,
    )
    selected_scope = (pl.col("eval").is_in(AGENTIC_EVALS) & (pl.col("scope") == "dev")) | (
        ~pl.col("eval").is_in(AGENTIC_EVALS) & (pl.col("scope") == "full")
    )
    chosen = aggregates.filter(selected_scope)
    selected_runs = _attach_status(
        run_scores.filter(selected_scope)
        .join(meta.evals, on="eval", how="left")
        .join(meta.models, on="model", how="left"),
        meta,
    )
    grid = meta.models.filter(pl.col("included")).join(meta.evals, how="cross")
    result = _attach_status(grid.join(chosen, on=["model", "eval"], how="left"), meta).with_columns(
        scope=pl.when(pl.col("eval").is_in(AGENTIC_EVALS))
        .then(pl.lit("dev"))
        .otherwise(pl.lit("full")),
        run_count=pl.col("run_count").fill_null(0),
        score_std=pl.col("score_std").fill_null(0.0),
    )

    # Fixed-precision labels are built here rather than in the chart spec: a
    # suspect bar can round to ~0 and vanish, so the label is the only thing
    # that carries its state.
    scores_seq, statuses = result["score"].to_list(), result["status"].to_list()
    result = result.with_columns(
        cell_label=pl.Series(
            [
                f"{s:.3f}" if s is not None else ("N/A" if st == "unsupported" else "—")
                for s, st in zip(scores_seq, statuses, strict=True)
            ]
        ),
        value_label=pl.Series(
            [
                "" if s is None else (f"{s:.3f}  suspect" if st == "suspect" else f"{s:.3f}")
                for s, st in zip(scores_seq, statuses, strict=True)
            ]
        ),
        source_label=pl.when(pl.col("scope") == "dev")
        .then(pl.format("fixed dev mean (n={})", pl.col("run_count")))
        .otherwise(pl.lit("full/base")),
    )
    return result, scope_scores, selected_runs


def add_ranks(df: pl.DataFrame) -> pl.DataFrame:
    """Percentile rank within each eval, over rankable rows only.

    Rank rather than min-max: while runs are still landing, one new model
    setting a new extreme would retroactively move every other bar. A lone
    result in an eval gets 0.5, since a rank of one carries no information.

    Ranks over a handful of models are coarse by construction, so `n_in_eval`
    travels with them and is surfaced on the profile axis.
    """
    usable = df.filter(pl.col("status").is_in(RANKABLE_STATUSES))
    return (
        usable.with_columns(
            rank=pl.col("score").rank("average").over("eval"),
            n_in_eval=pl.len().over("eval"),
        )
        .with_columns(
            pct=pl.when(pl.col("n_in_eval") > 1)
            .then((pl.col("rank") - 1) / (pl.col("n_in_eval") - 1))
            .otherwise(pl.lit(0.5))
        )
        .with_columns(eval_label=pl.format("{} (n={})", pl.col("eval"), pl.col("n_in_eval")))
    )


def composite(ranked: pl.DataFrame, family: str) -> pl.DataFrame:
    return (
        ranked.filter(pl.col("family") == family)
        .group_by("model")
        .agg(
            mean_pct=pl.col("pct").mean(),
            n_evals=pl.len(),
            model_index=pl.col("model_index").first(),
        )
        .sort("mean_pct", descending=True)
    )


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Return average-tie percentile ranks on [0, 1]."""
    count = len(values)
    if count == 1:
        return np.asarray([0.5])
    order = np.argsort(values, kind="stable")
    ranks = np.empty(count, dtype=float)
    start = 0
    while start < count:
        end = start + 1
        while end < count and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks / (count - 1)


def bootstrap_rank_intervals(
    run_scores: pl.DataFrame,
    meta: Metadata,
    *,
    samples: int = 2_000,
    seed: int = 20_260_804,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Propagate replica variation through ranks and family composites.

    Each draw independently selects one valid run for every measured
    model/eval cell, then recomputes all within-eval percentile ranks. This is
    a deterministic Monte Carlo marginalization over observed replicas; it
    does not pretend that replica runs add new models.
    """
    usable = run_scores.filter(pl.col("status").is_in(RANKABLE_STATUSES))
    models = meta.model_order
    evals = meta.eval_order
    model_index = {model: index for index, model in enumerate(models)}
    eval_index = {eval_name: index for index, eval_name in enumerate(evals)}
    cell_scores = {
        (str(model), str(eval_name)): np.asarray(group["score"].to_list(), dtype=float)
        for (model, eval_name), group in usable.partition_by(
            ["model", "eval"], as_dict=True
        ).items()
    }
    rng = np.random.default_rng(seed)
    rank_draws = np.full((samples, len(models), len(evals)), np.nan, dtype=float)

    for draw in range(samples):
        for eval_name, eval_position in eval_index.items():
            measured_models: list[int] = []
            values: list[float] = []
            for model, model_position in model_index.items():
                runs = cell_scores.get((model, eval_name))
                if runs is None:
                    continue
                measured_models.append(model_position)
                values.append(float(runs[rng.integers(len(runs))]))
            if not values:
                continue
            ranks = _percentile_ranks(np.asarray(values, dtype=float))
            rank_draws[draw, measured_models, eval_position] = ranks

    rank_rows = []
    for (model, eval_name), _runs in cell_scores.items():
        distribution = rank_draws[:, model_index[model], eval_index[eval_name]]
        rank_rows.append(
            {
                "model": model,
                "eval": eval_name,
                "pct_low": float(np.quantile(distribution, 0.025)),
                "pct_median": float(np.median(distribution)),
                "pct_high": float(np.quantile(distribution, 0.975)),
                "bootstrap_samples": samples,
            }
        )

    family_rows = []
    for family in FAMILY_ORDER:
        family_evals = meta.evals.filter(pl.col("family") == family)["eval"].to_list()
        positions = [eval_index[eval_name] for eval_name in family_evals]
        for model, model_position in model_index.items():
            distributions = rank_draws[:, model_position, positions]
            measured = ~np.isnan(distributions)
            counts = measured.sum(axis=1)
            valid_draws = counts > 0
            if not valid_draws.any():
                continue
            totals = np.nansum(distributions, axis=1)
            composite_draws = totals[valid_draws] / counts[valid_draws]
            family_rows.append(
                {
                    "model": model,
                    "family": family,
                    "mean_pct_low": float(np.quantile(composite_draws, 0.025)),
                    "mean_pct_median": float(np.median(composite_draws)),
                    "mean_pct_high": float(np.quantile(composite_draws, 0.975)),
                    "bootstrap_samples": int(valid_draws.sum()),
                }
            )

    return pl.DataFrame(rank_rows), pl.DataFrame(family_rows)


def _register_theme() -> None:
    @alt.theme.register("scilit", enable=True)
    def _theme() -> alt.theme.ThemeConfig:
        return {
            "config": {
                "background": SURFACE,
                "font": 'system-ui, -apple-system, "Segoe UI", sans-serif',
                "view": {"stroke": None, "continuousWidth": 300},
                "title": {
                    "color": INK,
                    "fontSize": 17,
                    "fontWeight": 600,
                    "anchor": "start",
                    "subtitleColor": INK_SECONDARY,
                    "subtitleFontSize": 12,
                    "subtitlePadding": 8,
                },
                "axis": {
                    "domainColor": AXIS,
                    "tickColor": AXIS,
                    "gridColor": GRID,
                    "labelColor": INK_SECONDARY,
                    "titleColor": INK_SECONDARY,
                    "labelFontSize": 11,
                    "titleFontSize": 11,
                    "titleFontWeight": 500,
                    "labelPadding": 5,
                    "titlePadding": 10,
                },
                "legend": {
                    "labelColor": INK_SECONDARY,
                    "titleColor": INK_SECONDARY,
                    "labelFontSize": 11,
                    "titleFontSize": 11,
                    "symbolType": "square",
                },
                "header": {
                    "labelColor": INK,
                    "labelFontSize": 13,
                    "labelFontWeight": 600,
                    "titleColor": INK_SECONDARY,
                },
            }
        }


def chart_coverage(df: pl.DataFrame, meta: Metadata) -> alt.HConcatChart:
    model_order = meta.model_order
    panels = []
    for index, family in enumerate(FAMILY_ORDER):
        sub = df.filter(pl.col("family") == family).with_columns(
            replica_label=pl.when(pl.col("status") == "unsupported")
            .then(pl.lit("N/A"))
            .when(pl.col("status").is_in(RANKABLE_STATUSES))
            .then(pl.col("run_count").cast(pl.Int64).cast(pl.String))
            .otherwise(pl.lit("0"))
        )
        evals = [e for e in meta.eval_order if e in set(sub["eval"])]
        base = alt.Chart(sub).encode(
            x=alt.X(
                "eval:N",
                sort=evals,
                title=None,
                axis=alt.Axis(labelAngle=-35, orient="bottom", labelLimit=150, labelColor=INK),
            ),
            y=alt.Y(
                "model:N",
                sort=model_order,
                title=None,
                axis=alt.Axis(labelLimit=190, labelColor=INK) if index == 0 else None,
            ),
        )
        cells = base.mark_rect(stroke=SURFACE, strokeWidth=2, cornerRadius=3).encode(
            color=alt.Color(
                "status:N",
                scale=alt.Scale(domain=STATUS_ORDER, range=[STATUS_COLOR[s] for s in STATUS_ORDER]),
                legend=alt.Legend(title=None, orient="bottom", columns=5, direction="horizontal")
                if index == 0
                else None,
            ),
            tooltip=[
                "model:N",
                "eval:N",
                "score:Q",
                "score_min:Q",
                "score_max:Q",
                "source_label:N",
                "score_std:Q",
                alt.Tooltip("run_count:Q", title="valid replicas"),
                "status:N",
                "run_id:N",
                "reason:N",
            ],
        )
        # Status color never carries meaning alone: every cell states its value
        # or its state in text.
        labels = base.mark_text(fontSize=9, fontWeight=600).encode(
            text="replica_label:N",
            color=alt.condition(
                alt.datum.status == "not-run", alt.value(MUTED), alt.value("#ffffff")
            ),
        )
        panels.append(
            (cells + labels).properties(
                width=alt.Step(60),
                height=alt.Step(26),
                title=alt.Title(
                    FAMILY_LABELS[family],
                    anchor="middle",
                    color=FAMILY_COLOR[family],
                ),
            )
        )
    return alt.hconcat(*panels, spacing=26).properties(
        title=alt.Title(
            "Eval coverage",
            subtitle="Cell: valid replicas · N/A: unsupported · 0: not run or excluded",
        )
    )


def chart_scores(df: pl.DataFrame, meta: Metadata, family: str) -> alt.FacetChart:
    sub = df.filter((pl.col("family") == family) & pl.col("score").is_not_null())
    evals = [e for e in meta.eval_order if e in set(sub["eval"])]
    has_suspect = sub.filter(pl.col("trust") == "suspect").height > 0
    base = alt.Chart(sub).encode(
        y=alt.Y("model:N", sort=meta.model_order, title=None, axis=alt.Axis(labelLimit=175)),
    )
    bars = base.mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("score:Q", title=None),
        color=alt.Color(
            "trust:N",
            scale=alt.Scale(
                domain=TRUST_ORDER,
                range=[FAMILY_COLOR[family], DEEMPHASIS],
            ),
            legend=alt.Legend(title=None, orient="bottom") if has_suspect else None,
        ),
        tooltip=[
            "model:N",
            "eval:N",
            "score:Q",
            alt.Tooltip("score_min:Q", title="replica min", format=".4f"),
            alt.Tooltip("score_max:Q", title="replica max", format=".4f"),
            "source_label:N",
            "score_std:Q",
            "status:N",
            "run_id:N",
            "reason:N",
        ],
    )
    whisker_base = base.transform_filter(alt.datum.run_count > 1)
    whiskers = whisker_base.mark_rule(color=INK, strokeWidth=1.25).encode(
        x=alt.X("score_min:Q"),
        x2=alt.X2("score_max:Q"),
    )
    whisker_min = whisker_base.mark_tick(
        color=INK, orient="vertical", thickness=1.25, size=10
    ).encode(x=alt.X("score_min:Q"))
    whisker_max = whisker_base.mark_tick(
        color=INK, orient="vertical", thickness=1.25, size=10
    ).encode(x=alt.X("score_max:Q"))
    values = base.mark_text(align="left", dx=5, fontSize=9).encode(
        x=alt.X("score_max:Q"),
        text=alt.Text("value_label:N"),
        # A suspect score near zero draws no visible bar, so its label has to
        # do the work on its own.
        color=alt.condition(
            alt.datum.trust == "suspect", alt.value(MUTED), alt.value(INK_SECONDARY)
        ),
    )
    if family == "sci-lit":
        title = "Sci-lit scores"
        subtitle = "Replica mean · whiskers: min–max · agentic: fixed dev · rerank: full set"
        facet_width, row_step = 235, 23
    elif family == "frontier-science":
        title = "FrontierScience scores"
        subtitle = [
            "Replica mean · whiskers: min–max",
            "Research rubric: raw score · Research success: thresholded rubric",
        ]
        # Three equal, presentation-scale panels fill the slide instead of
        # leaving the FrontierScience charts clustered in its left half.
        facet_width, row_step = 315, 27
    elif family == "base":
        title = "Base eval scores"
        subtitle = "Base-compatible tasks · full-set replica mean · whiskers: min–max"
        facet_width, row_step = 235, 25
    else:
        title = "Sentinel scores"
        subtitle = "Full-set replica mean · whiskers: min–max"
        facet_width, row_step = 300, 27

    return (
        (bars + whiskers + whisker_min + whisker_max + values)
        .properties(width=facet_width, height=alt.Step(row_step))
        .facet(alt.Facet("eval:N", sort=evals, title=None), columns=3)
        .resolve_scale(x="independent")
        .properties(title=alt.Title(title, subtitle=subtitle))
    )


def chart_profile(ranked: pl.DataFrame, meta: Metadata, models: list[str]) -> alt.LayerChart:
    """Percentile-rank profile across agentic, FrontierScience and sentinel evals."""
    # Build an explicit model/eval grid before joining scores. Vega-Lite then
    # breaks a line at an unmeasured combination instead of connecting the two
    # neighboring observations and implying coverage that does not exist.
    eval_meta = (
        meta.evals.select("eval", "eval_index", "family")
        .join(ranked.select("eval", "eval_label").unique(subset=["eval"]), on="eval", how="left")
        .filter(pl.col("eval_label").is_not_null())
        .with_columns(family_label=pl.col("family").replace_strict(FAMILY_LABELS))
        .sort("eval_index")
    )
    sub = (
        pl.DataFrame({"model": models})
        .join(eval_meta, how="cross")
        .join(
            ranked.filter(pl.col("model").is_in(models)).select(
                "model", "eval", "pct", "pct_low", "pct_high"
            ),
            on=["model", "eval"],
            how="left",
        )
        .with_columns(
            last_observed=pl.when(pl.col("pct").is_not_null())
            .then(pl.col("eval_index"))
            .max()
            .over("model")
        )
        .with_columns(is_last=pl.col("eval_index") == pl.col("last_observed"))
    )
    # Carry the per-eval n onto the axis: ranks over these small model pools are
    # coarse, and the reader should see that directly.
    evals = eval_meta["eval_label"].to_list()
    plot_height = 330
    rail_size = 7
    # Points at percentile 0 extend visually beyond the plot boundary. Give
    # this rail a slightly larger gutter than the cell-based heatmaps so it
    # never competes with those markers.
    rail_gap = 6
    axis_padding = rail_size + rail_gap + 4
    color = alt.Color(
        "model:N",
        sort=models,
        scale=alt.Scale(domain=models, range=SERIES[: len(models)]),
        legend=None,
    )
    base = alt.Chart(sub).encode(
        x=alt.X(
            "eval_label:N",
            sort=evals,
            title=None,
            axis=alt.Axis(
                labelAngle=-30,
                labelLimit=150,
                labelPadding=axis_padding,
                domain=False,
                ticks=False,
            ),
        ),
        y=alt.Y("pct:Q", title="Percentile rank", scale=alt.Scale(domain=[0, 1])),
        color=color,
    )
    lines = base.mark_line(strokeWidth=2, invalid="break-paths-show-domains")
    points = base.mark_point(size=55, filled=True)
    category_rail = (
        alt.Chart(eval_meta)
        .mark_rect(clip=False)
        .encode(
            x=alt.X("eval_label:N", sort=evals, title=None),
            y=alt.value(plot_height + rail_gap),
            y2=alt.value(plot_height + rail_gap + rail_size),
            color=alt.Color(
                "family_label:N",
                scale=alt.Scale(
                    domain=[FAMILY_LABELS[name] for name in FAMILY_ORDER],
                    range=[FAMILY_COLOR[name] for name in FAMILY_ORDER],
                ),
                legend=alt.Legend(title=None, orient="top", direction="horizontal"),
            ),
        )
    )
    # Direct labels keep identity from resting on color alone.
    labels = (
        base.transform_filter(alt.datum.is_last)
        .mark_text(align="left", dx=8, fontSize=10, fontWeight=600)
        .encode(text="model:N")
    )
    return (
        (category_rail + lines + points + labels)
        .resolve_scale(color="independent")
        .properties(
            width=960,
            height=plot_height,
            title=alt.Title(
                "Cross-eval profile",
                subtitle="1 = best within eval · gaps: unmeasured",
            ),
        )
    )


def chart_summary(
    sci_lit: pl.DataFrame,
    base_scores: pl.DataFrame,
    frontier: pl.DataFrame,
    sentinel: pl.DataFrame,
    n_sci_lit: int,
    pool: tuple[int, int],
) -> alt.HConcatChart:
    order = sci_lit["model"].to_list()
    sci_lit = sci_lit.with_columns(label_x=pl.max_horizontal("mean_pct", "mean_pct_high"))
    main_base = alt.Chart(sci_lit).encode(
        y=alt.Y("model:N", sort=order, title=None, axis=alt.Axis(labelLimit=175, labelColor=INK))
    )
    main = main_base.mark_bar(cornerRadiusEnd=4, color=PRIMARY).encode(
        x=alt.X("mean_pct:Q", title="mean percentile rank", scale=alt.Scale(domain=[0, 1])),
        tooltip=[
            "model:N",
            "mean_pct:Q",
            alt.Tooltip("mean_pct_low:Q", title="bootstrap 2.5%", format=".3f"),
            alt.Tooltip("mean_pct_high:Q", title="bootstrap 97.5%", format=".3f"),
            "n_evals:Q",
        ],
    )
    main_interval = main_base.mark_rule(color=INK, strokeWidth=1.3).encode(
        x=alt.X("mean_pct_low:Q"), x2=alt.X2("mean_pct_high:Q")
    )
    main_low = main_base.mark_tick(color=INK, orient="vertical", thickness=1.3, size=10).encode(
        x=alt.X("mean_pct_low:Q")
    )
    main_high = main_base.mark_tick(color=INK, orient="vertical", thickness=1.3, size=10).encode(
        x=alt.X("mean_pct_high:Q")
    )
    # Coverage on every bar: a mean over 5 evals and a mean over 6 are not the
    # same quantity, and the chart should not pretend otherwise.
    coverage = main_base.mark_text(align="left", dx=4, fontSize=9, color=INK_SECONDARY).encode(
        x=alt.X("label_x:Q"),
        text=alt.Text("label:N"),
    )
    left = (main + main_interval + main_low + main_high + coverage).properties(
        width=350,
        height=alt.Step(29),
        title=alt.Title(
            "Sci-lit composite",
            subtitle=f"Mean percentile rank · {n_sci_lit} evals",
        ),
    )

    def companion_panel(
        data: pl.DataFrame, title: str, subtitle: str, color: str
    ) -> alt.LayerChart:
        data = data.with_columns(label_x=pl.max_horizontal("display_pct", "mean_pct_high"))
        companion_base = alt.Chart(data).encode(
            y=alt.Y("model:N", sort=order, title=None, axis=None)
        )
        bars = companion_base.mark_bar(cornerRadiusEnd=4, color=color).encode(
            x=alt.X("display_pct:Q", title="mean percentile rank", scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                "model:N",
                "mean_pct:Q",
                alt.Tooltip("mean_pct_low:Q", title="bootstrap 2.5%", format=".3f"),
                alt.Tooltip("mean_pct_high:Q", title="bootstrap 97.5%", format=".3f"),
                "n_evals:Q",
            ],
        )
        interval_data = companion_base.transform_filter(alt.datum.n_evals > 0)
        intervals = interval_data.mark_rule(color=INK, strokeWidth=1.3).encode(
            x=alt.X("mean_pct_low:Q"), x2=alt.X2("mean_pct_high:Q")
        )
        interval_low = interval_data.mark_tick(
            color=INK, orient="vertical", thickness=1.3, size=10
        ).encode(x=alt.X("mean_pct_low:Q"))
        interval_high = interval_data.mark_tick(
            color=INK, orient="vertical", thickness=1.3, size=10
        ).encode(x=alt.X("mean_pct_high:Q"))
        # Coverage labels make missing companion scores visible instead of
        # silently dropping a model from the panel.
        labels = companion_base.mark_text(align="left", dx=4, fontSize=9, color=INK).encode(
            x=alt.X("label_x:Q"),
            text=alt.Text("label:N"),
        )
        return (bars + intervals + interval_low + interval_high + labels).properties(
            width=245,
            height=alt.Step(29),
            title=alt.Title(title, subtitle=subtitle),
        )

    base_panel = companion_panel(
        base_scores,
        "Base",
        "Mean rank · 5 evals",
        SERIES[3],
    )
    frontier_panel = companion_panel(
        frontier,
        "FrontierScience",
        "Mean rank · 3 measures",
        SERIES[2],
    )
    sentinel_panel = companion_panel(
        sentinel,
        "Sentinels",
        "Comparison only",
        SENTINEL,
    )
    return alt.hconcat(left, base_panel, frontier_panel, sentinel_panel, spacing=18).properties(
        title=alt.Title(
            "Overall standing",
            subtitle=(
                f"Composite: Sci-lit only · ranks span {pool[0]}–{pool[1]} models/eval · "
                "whiskers: replica-bootstrap 95%"
            ),
        )
    )


def _spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and values[order[end]] == values[order[start]]:
                end += 1
            average = (start + end - 1) / 2
            for index in order[start:end]:
                out[index] = average
            start = end
        return out

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mean_rank = (n - 1) / 2
    num = sum((rx[i] - mean_rank) * (ry[i] - mean_rank) for i in range(n))
    den = math.sqrt(sum((v - mean_rank) ** 2 for v in rx) * sum((v - mean_rank) ** 2 for v in ry))
    return num / den if den else 0.0


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denominator = math.sqrt(
        sum(value**2 for value in centered_x) * sum(value**2 for value in centered_y)
    )
    if denominator == 0:
        return None
    return sum(x * y for x, y in zip(centered_x, centered_y, strict=True)) / denominator


def _sample_covariance(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / (len(xs) - 1)


def trusted_dev_full_pairs(scope_scores: pl.DataFrame) -> pl.DataFrame:
    dev = scope_scores.filter(
        pl.col("eval").is_in(AGENTIC_EVALS)
        & (pl.col("scope") == "dev")
        & pl.col("status").is_in(RANKABLE_STATUSES)
    ).select(
        "model",
        "eval",
        dev_score="score",
        dev_min="score_min",
        dev_max="score_max",
        dev_runs="run_count",
    )
    full = scope_scores.filter(
        pl.col("eval").is_in(AGENTIC_EVALS)
        & (pl.col("scope") == "full")
        & pl.col("status").is_in(RANKABLE_STATUSES)
    ).select(
        "model",
        "eval",
        full_score="score",
        full_min="score_min",
        full_max="score_max",
        full_runs="run_count",
    )
    pairs = dev.join(full, on=["model", "eval"], how="inner").drop_nulls(
        ["dev_score", "full_score"]
    )
    labels: dict[str, str] = {}
    for eval_name in AGENTIC_EVALS:
        sub = pairs.filter(pl.col("eval") == eval_name)
        if sub.is_empty():
            continue
        rho = _spearman(sub["full_score"].to_list(), sub["dev_score"].to_list())
        labels[eval_name] = f"{CANONICAL_TO_DEV[eval_name]} · n={sub.height} · rho={rho:+.2f}"
    return pairs.with_columns(
        facet_label=pl.col("eval").replace_strict(labels, default=pl.col("eval"))
    )


def chart_dev_vs_full(pairs: pl.DataFrame) -> alt.FacetChart | None:
    if pairs.is_empty():
        return None
    rows = [{**row, "kind": "point", "diag_order": None} for row in pairs.to_dicts()]
    for facet_label in pairs["facet_label"].unique(maintain_order=True):
        sub = pairs.filter(pl.col("facet_label") == facet_label)
        values = [*sub["full_score"].to_list(), *sub["dev_score"].to_list()]
        lo, hi = min(values), max(values)
        pad = (hi - lo) * 0.08 or 0.01
        rows.extend(
            [
                {
                    "facet_label": facet_label,
                    "full_score": lo - pad,
                    "dev_score": lo - pad,
                    "kind": "diagonal",
                    "diag_order": 0,
                },
                {
                    "facet_label": facet_label,
                    "full_score": hi + pad,
                    "dev_score": hi + pad,
                    "kind": "diagonal",
                    "diag_order": 1,
                },
            ]
        )
    combined = pl.DataFrame(rows, infer_schema_length=None)
    base = alt.Chart(combined)
    line = (
        base.transform_filter(alt.datum.kind == "diagonal")
        .mark_line(color=AXIS, strokeDash=[5, 4])
        .encode(
            x=alt.X("full_score:Q", title="Full-set score"),
            y=alt.Y("dev_score:Q", title="Fixed-dev score"),
            order="diag_order:Q",
        )
    )
    points = (
        base.transform_filter(alt.datum.kind == "point")
        .mark_point(filled=True, color=PRIMARY, size=85)
        .encode(
            x=alt.X("full_score:Q", title="Full-set score"),
            y=alt.Y("dev_score:Q", title="Fixed-dev score"),
            tooltip=[
                "model:N",
                "eval:N",
                "full_score:Q",
                "dev_score:Q",
                "full_runs:Q",
                "dev_runs:Q",
            ],
        )
    )
    point_base = base.transform_filter(alt.datum.kind == "point")
    horizontal_errors = point_base.mark_rule(color=INK_SECONDARY, opacity=0.65).encode(
        x=alt.X("full_min:Q"),
        x2=alt.X2("full_max:Q"),
        y=alt.Y("dev_score:Q"),
    )
    vertical_errors = point_base.mark_rule(color=INK_SECONDARY, opacity=0.65).encode(
        x=alt.X("full_score:Q"),
        y=alt.Y("dev_min:Q"),
        y2=alt.Y2("dev_max:Q"),
    )
    labels = (
        base.transform_filter(alt.datum.kind == "point")
        .mark_text(dx=6, align="left", fontSize=8, color=INK_SECONDARY)
        .encode(x="full_score:Q", y="dev_score:Q", text="model:N")
    )
    return (
        alt.layer(line, horizontal_errors, vertical_errors, points, labels)
        .properties(width=205, height=175)
        .facet(alt.Facet("facet_label:N", title=None), columns=3)
        .resolve_scale(x="independent", y="independent")
        .properties(
            title=alt.Title(
                "Fixed dev vs. full set",
                subtitle="Trusted pairs · whiskers: replica min–max · dashed line: dev = full",
            )
        )
    )


def _replica_pairs(run_scores: pl.DataFrame, x_eval: str, y_eval: str) -> pl.DataFrame:
    """Return every within-model Cartesian run pairing for two evals.

    The diagonal is paired by run rather than expanded against itself so it
    remains a conventional variance/correlation diagonal. Cross-eval cells use
    all i × j combinations, as a deterministic marginalization over replicas.
    """
    x_side = run_scores.filter(pl.col("eval") == x_eval).select(
        "model",
        x="score",
        x_run_id="run_id",
        x_replica="replica",
    )
    if x_eval == y_eval:
        return x_side.select(
            "model",
            "x",
            "x_run_id",
            "x_replica",
            y=pl.col("x"),
            y_run_id=pl.col("x_run_id"),
            y_replica=pl.col("x_replica"),
        )
    y_side = run_scores.filter(pl.col("eval") == y_eval).select(
        "model",
        y="score",
        y_run_id="run_id",
        y_replica="replica",
    )
    return x_side.join(y_side, on="model", how="inner").drop_nulls(["x", "y"])


def pairwise_replica_correlations(
    run_scores: pl.DataFrame, x_evals: list[str], y_evals: list[str]
) -> pl.DataFrame:
    """Compute cross-eval statistics over deterministic Cartesian run pairs.

    ``n_pairs`` is the number of expanded points, not an independent-sample
    count. ``n_models`` remains the relevant coverage diagnostic.
    """
    usable = run_scores.filter(pl.col("status").is_in(RANKABLE_STATUSES)).select(
        "model", "eval", "score", "run_id", "replica"
    )
    rows = []

    for x_eval in x_evals:
        for y_eval in y_evals:
            pairs = _replica_pairs(usable, x_eval, y_eval)
            xs = pairs["x"].to_list()
            ys = pairs["y"].to_list()
            n_models = pairs["model"].n_unique() if pairs.height else 0
            rho = _spearman(xs, ys) if pairs.height >= 3 else None
            pearson = _pearson(xs, ys)
            covariance = _sample_covariance(xs, ys)
            variant_rhos = []
            if pairs.height:
                for variant in pairs.partition_by(["x_replica", "y_replica"]):
                    if variant.height < 3 or variant["model"].n_unique() < 3:
                        continue
                    variant_rhos.append(_spearman(variant["x"].to_list(), variant["y"].to_list()))
            if not variant_rhos and rho is not None:
                variant_rhos = [rho]
            combinations = pairs.group_by("model").len()["len"].to_list() if pairs.height else []
            rows.append(
                {
                    "x_eval": x_eval,
                    "y_eval": y_eval,
                    "x_label": CANONICAL_TO_DEV.get(x_eval, x_eval),
                    "y_label": CANONICAL_TO_DEV.get(y_eval, y_eval),
                    "x_family_label": FAMILY_LABELS[eval_family(x_eval)],
                    "y_family_label": FAMILY_LABELS[eval_family(y_eval)],
                    "rho": rho,
                    "rho_variant_mean": (
                        sum(variant_rhos) / len(variant_rhos) if variant_rhos else None
                    ),
                    "rho_variant_median": (
                        float(np.median(variant_rhos)) if variant_rhos else None
                    ),
                    "rho_min": min(variant_rhos) if variant_rhos else None,
                    "rho_max": max(variant_rhos) if variant_rhos else None,
                    "abs_rho_min": min(map(abs, variant_rhos)) if variant_rhos else None,
                    "abs_rho_max": max(map(abs, variant_rhos)) if variant_rhos else None,
                    "rho_variant_count": len(variant_rhos),
                    "pearson_r": pearson,
                    "sample_covariance": covariance,
                    "n_models": n_models,
                    "n_pairs": pairs.height,
                    "min_pairs_per_model": min(combinations) if combinations else 0,
                    "max_pairs_per_model": max(combinations) if combinations else 0,
                    "strong": rho is not None and n_models >= 4 and abs(rho) >= 0.7,
                }
            )
    return pl.DataFrame(rows).with_columns(
        rho_label=pl.when(pl.col("rho").is_not_null())
        .then(pl.col("rho").round(2).cast(pl.String))
        .otherwise(pl.lit("—")),
        n_label=pl.format("m={} · p={}", pl.col("n_models"), pl.col("n_pairs")),
    )


def deepscholar_ifeval_pairs(run_scores: pl.DataFrame) -> pl.DataFrame:
    """Expand every valid IFEval × DeepScholar replica combination by model."""
    usable = run_scores.filter(pl.col("status").is_in(RANKABLE_STATUSES)).select(
        "model", "eval", "score", "run_id", "replica"
    )
    return (
        _replica_pairs(usable, "IFEval", "DeepScholar-Bench")
        .rename(
            {
                "x": "ifeval_score",
                "x_run_id": "ifeval_run_id",
                "x_replica": "ifeval_replica",
                "y": "deepscholar_score",
                "y_run_id": "deepscholar_run_id",
                "y_replica": "deepscholar_replica",
            }
        )
        .sort(["model", "ifeval_replica", "deepscholar_replica"])
    )


def bootstrap_deepscholar_regression(
    pairs: pl.DataFrame,
    *,
    samples: int = 2_000,
    seed: int = 20_260_803,
) -> tuple[pl.DataFrame, dict[str, float | int]]:
    """Fit DeepScholar ~ IFEval with a model-cluster bootstrap.

    All Cartesian replica pairs contribute to each fit. Bootstrap draws happen
    at the model level so repeated measurements of one model are not treated as
    independent models when estimating the regression uncertainty.
    """
    if pairs.is_empty() or pairs["model"].n_unique() < 2:
        return pl.DataFrame(), {}

    model_groups = {
        str(model): (
            np.asarray(group["ifeval_score"].to_list(), dtype=float),
            np.asarray(group["deepscholar_score"].to_list(), dtype=float),
        )
        for (model,), group in pairs.partition_by("model", as_dict=True).items()
    }
    models = sorted(model_groups)
    x_observed = np.asarray(pairs["ifeval_score"].to_list(), dtype=float)
    y_observed = np.asarray(pairs["deepscholar_score"].to_list(), dtype=float)
    x_grid = np.linspace(float(x_observed.min()), float(x_observed.max()), 160)
    rng = np.random.default_rng(seed)
    predictions: list[np.ndarray] = []
    slopes: list[float] = []

    for _ in range(samples):
        selected = rng.choice(models, size=len(models), replace=True)
        x_boot = np.concatenate([model_groups[str(model)][0] for model in selected])
        y_boot = np.concatenate([model_groups[str(model)][1] for model in selected])
        if np.unique(x_boot).size < 2:
            continue
        slope, intercept = np.polyfit(x_boot, y_boot, 1)
        slopes.append(float(slope))
        predictions.append(intercept + slope * x_grid)

    if not predictions:
        return pl.DataFrame(), {}

    prediction_array = np.stack(predictions)
    slope_array = np.asarray(slopes)
    band = pl.DataFrame(
        {
            "ifeval_score": x_grid,
            "fit": np.median(prediction_array, axis=0),
            "fit_low": np.quantile(prediction_array, 0.025, axis=0),
            "fit_high": np.quantile(prediction_array, 0.975, axis=0),
        }
    )
    xs = x_observed.tolist()
    ys = y_observed.tolist()
    pearson = _pearson(xs, ys)
    summary: dict[str, float | int] = {
        "n_models": len(models),
        "n_pairs": pairs.height,
        "bootstrap_samples": len(predictions),
        "pearson_r": pearson if pearson is not None else float("nan"),
        "spearman_rho": _spearman(xs, ys),
        "slope_median": float(np.median(slope_array)),
        "slope_low": float(np.quantile(slope_array, 0.025)),
        "slope_high": float(np.quantile(slope_array, 0.975)),
    }
    return band, summary


def chart_deepscholar_ifeval(
    pairs: pl.DataFrame,
    band: pl.DataFrame,
    regression: dict[str, float | int],
) -> alt.LayerChart:
    """Show per-model replica ranges and the model-cluster bootstrap regression."""
    models = pairs["model"].unique(maintain_order=True).to_list()
    model_colors = MODEL_COLORS[: len(models)]
    centers = (
        pairs.group_by("model", maintain_order=True)
        .agg(
            ifeval_median=pl.col("ifeval_score").median(),
            ifeval_min=pl.col("ifeval_score").min(),
            ifeval_max=pl.col("ifeval_score").max(),
            deepscholar_median=pl.col("deepscholar_score").median(),
            deepscholar_min=pl.col("deepscholar_score").min(),
            deepscholar_max=pl.col("deepscholar_score").max(),
            ifeval_replicas=pl.col("ifeval_run_id").n_unique(),
            deepscholar_replicas=pl.col("deepscholar_run_id").n_unique(),
        )
        .sort(pl.col("model").replace_strict({model: i for i, model in enumerate(models)}))
    )
    x_caps = pl.concat(
        [
            centers.select(
                "model",
                "deepscholar_median",
                pl.col("ifeval_min").alias("ifeval_extent"),
            ),
            centers.select(
                "model",
                "deepscholar_median",
                pl.col("ifeval_max").alias("ifeval_extent"),
            ),
        ]
    )
    y_caps = pl.concat(
        [
            centers.select(
                "model",
                "ifeval_median",
                pl.col("deepscholar_min").alias("deepscholar_extent"),
            ),
            centers.select(
                "model",
                "ifeval_median",
                pl.col("deepscholar_max").alias("deepscholar_extent"),
            ),
        ]
    )
    x_encoding = alt.X(
        "ifeval_score:Q",
        title="IFEval accuracy (full)",
        scale=alt.Scale(zero=False),
        axis=alt.Axis(format=".2f"),
    )
    band_chart = (
        alt.Chart(band)
        .mark_area(color=PRIMARY, opacity=0.14)
        .encode(
            x=x_encoding,
            y=alt.Y(
                "fit_low:Q",
                title="DeepScholar geomean (dev10)",
                scale=alt.Scale(zero=False),
                axis=alt.Axis(format=".3f"),
            ),
            y2="fit_high:Q",
        )
    )
    fit = alt.Chart(band).mark_line(color=PRIMARY, strokeWidth=2.6).encode(x=x_encoding, y="fit:Q")
    model_color = alt.Color(
        "model:N",
        scale=alt.Scale(domain=models, range=model_colors),
        legend=alt.Legend(title="Model", orient="right", symbolSize=125, symbolType="circle"),
    )
    x_ranges = (
        alt.Chart(centers)
        .mark_rule(strokeWidth=1.6)
        .encode(
            x=alt.X("ifeval_min:Q", scale=alt.Scale(zero=False)),
            x2="ifeval_max:Q",
            y=alt.Y("deepscholar_median:Q", scale=alt.Scale(zero=False)),
            color=model_color,
        )
    )
    y_ranges = (
        alt.Chart(centers)
        .mark_rule(strokeWidth=1.6)
        .encode(
            x=alt.X("ifeval_median:Q", scale=alt.Scale(zero=False)),
            y=alt.Y("deepscholar_min:Q", scale=alt.Scale(zero=False)),
            y2="deepscholar_max:Q",
            color=model_color,
        )
    )
    x_range_caps = (
        alt.Chart(x_caps)
        .mark_tick(orient="vertical", size=10, thickness=1.6)
        .encode(
            x=alt.X("ifeval_extent:Q", scale=alt.Scale(zero=False)),
            y=alt.Y("deepscholar_median:Q", scale=alt.Scale(zero=False)),
            color=model_color,
        )
    )
    y_range_caps = (
        alt.Chart(y_caps)
        .mark_tick(orient="horizontal", size=10, thickness=1.6)
        .encode(
            x=alt.X("ifeval_median:Q", scale=alt.Scale(zero=False)),
            y=alt.Y("deepscholar_extent:Q", scale=alt.Scale(zero=False)),
            color=model_color,
        )
    )
    points = (
        alt.Chart(centers)
        .mark_point(
            filled=True,
            shape="circle",
            opacity=0.9,
            size=145,
            stroke=INK,
            strokeWidth=0.65,
        )
        .encode(
            x=alt.X(
                "ifeval_median:Q",
                title="IFEval accuracy (full)",
                scale=alt.Scale(zero=False),
                axis=alt.Axis(format=".2f"),
            ),
            y=alt.Y(
                "deepscholar_median:Q",
                title="DeepScholar geomean (dev10)",
                scale=alt.Scale(zero=False),
                axis=alt.Axis(format=".3f"),
            ),
            color=model_color,
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("ifeval_median:Q", title="IFEval median", format=".4f"),
                alt.Tooltip("ifeval_min:Q", title="IFEval min", format=".4f"),
                alt.Tooltip("ifeval_max:Q", title="IFEval max", format=".4f"),
                alt.Tooltip("deepscholar_median:Q", title="DeepScholar median", format=".4f"),
                alt.Tooltip("deepscholar_min:Q", title="DeepScholar min", format=".4f"),
                alt.Tooltip("deepscholar_max:Q", title="DeepScholar max", format=".4f"),
                alt.Tooltip("ifeval_replicas:Q", title="IFEval replicas"),
                alt.Tooltip("deepscholar_replicas:Q", title="DeepScholar replicas"),
            ],
        )
    )
    subtitle = [
        "Points: replica median · whiskers: min–max · fit: all replica pairs, 95% band",
        (
            f"n={int(regression['n_models'])} models · Pearson r={regression['pearson_r']:+.2f} · "
            f"Spearman ρ={regression['spearman_rho']:+.2f} · slope "
            f"{regression['slope_median']:+.3f} "
            f"[{regression['slope_low']:+.3f}, {regression['slope_high']:+.3f}]"
        ),
    ]
    return (
        band_chart + fit + x_ranges + y_ranges + x_range_caps + y_range_caps + points
    ).properties(
        width=850,
        height=455,
        title=alt.Title("IFEval → DeepScholar-Bench", subtitle=subtitle),
    )


RIDGE_ALPHA_GRID = np.logspace(-2, 2, 9)
PROXY_REGRESSION_LABELS = {
    "ARC": "ARC",
    "MMLU-STEM": "MMLU-STEM",
    "MedMCQA": "MedMCQA",
    "MedQA": "MedQA",
    "SciQ": "SciQ",
    "LitSearch-rerank": "LitSearch-rerank",
    "FS-Olympiad accuracy": "FS-Olympiad",
    "FS-Research success": "FS-Research success",
    "FS-Research rubric": "FS-Research rubric",
    "IFEval": "IFEval",
    "MMLU": "MMLU",
    "MATH-500": "MATH-500",
}


def _ridge_fit(
    x: np.ndarray, y: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Fit ridge with an unpenalized intercept and standardized predictors."""
    x_mean = x.mean(axis=0)
    x_scale = x.std(axis=0)
    x_scale = np.where(x_scale < 1e-12, 1.0, x_scale)
    x_standard = (x - x_mean) / x_scale
    y_mean = float(y.mean())
    gram = x_standard.T @ x_standard + alpha * np.eye(x.shape[1])
    beta = np.linalg.solve(gram, x_standard.T @ (y - y_mean))
    return x_mean, x_scale, y_mean, beta


def _ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_mean, x_scale, y_mean, beta = _ridge_fit(x_train, y_train, alpha)
    prediction = y_mean + ((x_test - x_mean) / x_scale) @ beta
    return prediction, beta


def _select_ridge_alpha(x: np.ndarray, y: np.ndarray) -> float:
    """Select ridge strength by leave-one-model-out error."""
    errors: list[float] = []
    for alpha in RIDGE_ALPHA_GRID:
        predictions = np.empty(len(y), dtype=float)
        for held_out in range(len(y)):
            train = np.arange(len(y)) != held_out
            predicted, _ = _ridge_predict(
                x[train], y[train], x[held_out : held_out + 1], float(alpha)
            )
            predictions[held_out] = predicted[0]
        errors.append(float(np.mean((y - predictions) ** 2)))
    return float(RIDGE_ALPHA_GRID[int(np.argmin(errors))])


def bootstrap_deepscholar_proxy_regression(
    run_scores: pl.DataFrame,
    predictor_evals: list[str],
    *,
    samples: int = 1_000,
    seed: int = 20_260_804,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, float | int]]:
    """Predict DeepScholar from proxy evals with replica-bootstrap nested LOMO ridge.

    The model is evaluated out of sample: each model is held out, ridge strength
    is selected using only the other models' median scores, and every bootstrap
    draws one replica independently for each model/evaluation cell. This
    propagates run-to-run variation without pretending the replicas are new
    independent models.
    """
    evaluations = [*predictor_evals, "DeepScholar-Bench"]
    usable = run_scores.filter(
        pl.col("status").is_in(RANKABLE_STATUSES) & pl.col("eval").is_in(evaluations)
    ).select("model", "eval", "score")
    lookup: dict[tuple[str, str], np.ndarray] = {}
    for key, group in usable.partition_by("model", "eval", as_dict=True).items():
        model, eval_name = key
        lookup[(str(model), str(eval_name))] = np.asarray(group["score"], dtype=float)

    candidates = usable.filter(pl.col("eval") == "DeepScholar-Bench")["model"].unique(
        maintain_order=True
    )
    models = [
        str(model)
        for model in candidates
        if all((str(model), eval_name) in lookup for eval_name in evaluations)
    ]
    if len(models) < 3:
        return pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), {}

    median_x = np.asarray(
        [
            [np.median(lookup[(model, eval_name)]) for eval_name in predictor_evals]
            for model in models
        ],
        dtype=float,
    )
    median_y = np.asarray(
        [np.median(lookup[(model, "DeepScholar-Bench")]) for model in models], dtype=float
    )
    outer_alphas: list[float] = []
    for held_out in range(len(models)):
        train = np.arange(len(models)) != held_out
        outer_alphas.append(_select_ridge_alpha(median_x[train], median_y[train]))
    full_alpha = _select_ridge_alpha(median_x, median_y)

    rng = np.random.default_rng(seed)
    predictions = np.empty((samples, len(models)), dtype=float)
    coefficients = np.empty((samples, len(predictor_evals)), dtype=float)
    metric_rows: list[dict[str, float | int]] = []
    for sample in range(samples):
        x = np.asarray(
            [
                [rng.choice(lookup[(model, eval_name)]) for eval_name in predictor_evals]
                for model in models
            ],
            dtype=float,
        )
        y = np.asarray(
            [rng.choice(lookup[(model, "DeepScholar-Bench")]) for model in models],
            dtype=float,
        )
        for held_out, alpha in enumerate(outer_alphas):
            train = np.arange(len(models)) != held_out
            predicted, _ = _ridge_predict(x[train], y[train], x[held_out : held_out + 1], alpha)
            predictions[sample, held_out] = predicted[0]
        _, coefficients[sample] = _ridge_predict(x, y, x[:1], full_alpha)
        residual = y - predictions[sample]
        denominator = float(np.sum((y - y.mean()) ** 2))
        metric_rows.append(
            {
                "sample": sample + 1,
                "pearson_r": float(np.corrcoef(y, predictions[sample])[0, 1]),
                "spearman_rho": float(_spearman(y.tolist(), predictions[sample].tolist())),
                "mae": float(np.mean(np.abs(residual))),
                "r2": float(1 - np.sum(residual**2) / denominator),
            }
        )

    prediction_rows: list[dict[str, float | int | str]] = []
    for index, model in enumerate(models):
        actual = lookup[(model, "DeepScholar-Bench")]
        prediction_rows.append(
            {
                "model": model,
                "actual_median": float(np.median(actual)),
                "actual_min": float(actual.min()),
                "actual_max": float(actual.max()),
                "predicted_median": float(np.median(predictions[:, index])),
                "predicted_low": float(np.quantile(predictions[:, index], 0.025)),
                "predicted_high": float(np.quantile(predictions[:, index], 0.975)),
                "ridge_alpha": outer_alphas[index],
            }
        )
    coefficient_rows = [
        {
            "eval": eval_name,
            "eval_label": PROXY_REGRESSION_LABELS[eval_name],
            "coefficient_median": float(np.median(coefficients[:, index])),
            "coefficient_low": float(np.quantile(coefficients[:, index], 0.025)),
            "coefficient_high": float(np.quantile(coefficients[:, index], 0.975)),
            "ridge_alpha": full_alpha,
        }
        for index, eval_name in enumerate(predictor_evals)
    ]
    metrics = pl.DataFrame(metric_rows)
    summary: dict[str, float | int] = {
        "n_models": len(models),
        "n_predictors": len(predictor_evals),
        "bootstrap_samples": samples,
        "ridge_alpha_full": full_alpha,
        "ridge_alpha_outer_min": min(outer_alphas),
        "ridge_alpha_outer_max": max(outer_alphas),
    }
    for metric in ["pearson_r", "spearman_rho", "mae", "r2"]:
        values = np.asarray(metrics[metric], dtype=float)
        summary[f"{metric}_median"] = float(np.quantile(values, 0.5))
        summary[f"{metric}_low"] = float(np.quantile(values, 0.025))
        summary[f"{metric}_high"] = float(np.quantile(values, 0.975))
    return (
        pl.DataFrame(prediction_rows),
        pl.DataFrame(coefficient_rows),
        metrics,
        summary,
    )


def chart_deepscholar_proxy_regression(
    predictions: pl.DataFrame,
    coefficients: pl.DataFrame,
    summary: dict[str, float | int],
    predictor_evals: list[str],
    *,
    title: str,
    predictor_label: str,
) -> alt.HConcatChart:
    """Show held-out predictions and bootstrap coefficient stability."""
    models = predictions["model"].to_list()
    model_color = alt.Color(
        "model:N",
        scale=alt.Scale(domain=models, range=MODEL_COLORS[: len(models)]),
        legend=alt.Legend(
            title="Model", orient="bottom", direction="horizontal", columns=4, symbolType="circle"
        ),
    )
    bounds = [
        float(predictions["actual_min"].min()),
        float(predictions["predicted_low"].min()),
        float(predictions["actual_max"].max()),
        float(predictions["predicted_high"].max()),
    ]
    padding = max(bounds) - min(bounds)
    domain = [min(bounds) - 0.04 * padding, max(bounds) + 0.04 * padding]
    identity = pl.DataFrame({"actual": domain, "predicted": domain})
    x = alt.X(
        "actual_median:Q",
        title="Observed DeepScholar median",
        scale=alt.Scale(domain=domain, zero=False),
        axis=alt.Axis(format=".3f"),
    )
    y = alt.Y(
        "predicted_median:Q",
        title="Held-out predicted DeepScholar",
        scale=alt.Scale(domain=domain, zero=False),
        axis=alt.Axis(format=".3f"),
    )
    diagonal = (
        alt.Chart(identity)
        .mark_line(color=MUTED, strokeDash=[5, 4], strokeWidth=1.3)
        .encode(x="actual:Q", y="predicted:Q")
    )
    x_ranges = (
        alt.Chart(predictions)
        .mark_rule(strokeWidth=1.4)
        .encode(x="actual_min:Q", x2="actual_max:Q", y=y, color=model_color)
    )
    y_ranges = (
        alt.Chart(predictions)
        .mark_rule(strokeWidth=1.4)
        .encode(x=x, y="predicted_low:Q", y2="predicted_high:Q", color=model_color)
    )
    points = (
        alt.Chart(predictions)
        .mark_point(filled=True, shape="circle", size=125, stroke=INK, strokeWidth=0.6)
        .encode(
            x=x,
            y=y,
            color=model_color,
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("actual_median:Q", title="Observed median", format=".4f"),
                alt.Tooltip("actual_min:Q", title="Observed min", format=".4f"),
                alt.Tooltip("actual_max:Q", title="Observed max", format=".4f"),
                alt.Tooltip("predicted_median:Q", title="Predicted median", format=".4f"),
                alt.Tooltip("predicted_low:Q", title="Predicted 2.5%", format=".4f"),
                alt.Tooltip("predicted_high:Q", title="Predicted 97.5%", format=".4f"),
                alt.Tooltip("ridge_alpha:Q", title="Fold ridge alpha", format=".2g"),
            ],
        )
    )
    prediction_panel = (diagonal + x_ranges + y_ranges + points).properties(
        width=535,
        height=405,
        title=alt.Title(
            "Held-out predictions",
            subtitle="x: replica min–max · y: bootstrap 95%",
        ),
    )

    coefficient_order = [PROXY_REGRESSION_LABELS[name] for name in predictor_evals]
    zero = (
        alt.Chart(pl.DataFrame({"zero": [0]}))
        .mark_rule(color=MUTED, strokeDash=[4, 3], strokeWidth=1.2)
        .encode(x="zero:Q")
    )
    coefficient_ranges = (
        alt.Chart(coefficients)
        .mark_rule(color=SERIES[2], strokeWidth=2)
        .encode(
            x=alt.X("coefficient_low:Q", title="DeepScholar score per predictor SD"),
            x2="coefficient_high:Q",
            y=alt.Y("eval_label:N", sort=coefficient_order, title=None),
        )
    )
    coefficient_points = (
        alt.Chart(coefficients)
        .mark_point(filled=True, color=SERIES[2], size=95, stroke=INK, strokeWidth=0.5)
        .encode(
            x="coefficient_median:Q",
            y=alt.Y("eval_label:N", sort=coefficient_order, title=None),
            tooltip=[
                alt.Tooltip("eval_label:N", title="Predictor"),
                alt.Tooltip("coefficient_median:Q", title="Median", format="+.4f"),
                alt.Tooltip("coefficient_low:Q", title="2.5%", format="+.4f"),
                alt.Tooltip("coefficient_high:Q", title="97.5%", format="+.4f"),
                alt.Tooltip("ridge_alpha:Q", title="Full ridge alpha", format=".2g"),
            ],
        )
    )
    coefficient_panel = (zero + coefficient_ranges + coefficient_points).properties(
        width=315,
        height=405,
        title=alt.Title(
            "Ridge coefficients",
            subtitle="Standardized · bootstrap 95%",
        ),
    )
    subtitle = [
        (
            f"Nested leave-one-model-out ridge · n={int(summary['n_models'])} · "
            f"{int(summary['n_predictors'])} {predictor_label} · "
            f"{int(summary['bootstrap_samples']):,} bootstrap samples"
        ),
        (
            f"Held-out Pearson r={summary['pearson_r_median']:+.2f} "
            f"[{summary['pearson_r_low']:+.2f}, {summary['pearson_r_high']:+.2f}]; "
            f"Spearman ρ={summary['spearman_rho_median']:+.2f}; "
            f"MAE={summary['mae_median']:.3f}"
        ),
    ]
    return alt.hconcat(prediction_panel, coefficient_panel, spacing=42).properties(
        title=alt.Title(title, subtitle=subtitle)
    )


def _pca_from_correlation(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """PSD-project a correlation matrix and return its PCA decomposition."""
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 1.0)
    raw_eigenvalues, raw_eigenvectors = np.linalg.eigh(matrix)
    clipped = np.clip(raw_eigenvalues, 0.0, None)
    projected = raw_eigenvectors @ np.diag(clipped) @ raw_eigenvectors.T
    scale = np.sqrt(np.clip(np.diag(projected), 1e-12, None))
    projected = projected / np.outer(scale, scale)
    projected = (projected + projected.T) / 2
    np.fill_diagonal(projected, 1.0)

    eigenvalues, eigenvectors = np.linalg.eigh(projected)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    components = eigenvectors[:, order].T
    for pc in range(components.shape[0]):
        anchor = int(np.argmax(np.abs(components[pc])))
        if components[pc, anchor] < 0:
            components[pc] *= -1
    explained = eigenvalues / eigenvalues.sum()
    return raw_eigenvalues, projected, eigenvalues, components, explained, np.cumsum(explained)


def analyze_eval_pca(
    pairwise: pl.DataFrame, eval_order: list[str]
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """PCA of the pairwise-expanded rank-covariance structure.

    Pairwise-complete correlation matrices can be non-positive-semidefinite
    because different eval pairs have different model coverage and replica
    counts. Negative eigenvalues are clipped and the diagonal is renormalized
    before PCA; diagnostics make that correction explicit.
    """
    lookup = {(row["x_eval"], row["y_eval"]): row["rho"] for row in pairwise.iter_rows(named=True)}
    matrix = np.empty((len(eval_order), len(eval_order)), dtype=float)
    missing: list[tuple[str, str]] = []
    for i, x_eval in enumerate(eval_order):
        for j, y_eval in enumerate(eval_order):
            value = lookup.get((x_eval, y_eval))
            if value is None:
                missing.append((x_eval, y_eval))
                matrix[i, j] = np.nan
            else:
                matrix[i, j] = float(value)
    if missing:
        raise ValueError(f"PCA correlation matrix has missing cells: {missing}")

    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 1.0)
    raw_eigenvalues, projected, eigenvalues, components, explained, cumulative = (
        _pca_from_correlation(matrix)
    )
    retained = int(np.searchsorted(cumulative, 0.8) + 1)
    loading_correlations = components.T * np.sqrt(eigenvalues)

    variance = pl.DataFrame(
        {
            "pc": [f"PC{i + 1}" for i in range(len(eval_order))],
            "pc_index": list(range(1, len(eval_order) + 1)),
            "explained_variance": explained,
            "cumulative_variance": cumulative,
            "retained_80pct": [i < retained for i in range(len(eval_order))],
        }
    )
    loading_rows = []
    for eval_index, eval_name in enumerate(eval_order):
        for pc in range(len(eval_order)):
            loading_rows.append(
                {
                    "eval": eval_name,
                    "eval_label": CANONICAL_TO_DEV.get(eval_name, eval_name),
                    "family_label": FAMILY_LABELS[eval_family(eval_name)],
                    "pc": f"PC{pc + 1}",
                    "pc_index": pc + 1,
                    "component_loading": float(components[pc, eval_index]),
                    "loading_correlation": float(loading_correlations[eval_index, pc]),
                }
            )
    loadings = pl.DataFrame(loading_rows)

    redundancy_rows = []
    for i, x_eval in enumerate(eval_order):
        for j in range(i + 1, len(eval_order)):
            y_eval = eval_order[j]
            rho = matrix[i, j]
            source = pairwise.filter(
                (pl.col("x_eval") == x_eval) & (pl.col("y_eval") == y_eval)
            ).row(0, named=True)
            abs_rho = float(abs(rho))
            abs_rho_min = source["abs_rho_min"]
            abs_rho_max = source["abs_rho_max"]
            redundancy_rows.append(
                {
                    "x_eval": x_eval,
                    "y_eval": y_eval,
                    "x_family": eval_family(x_eval),
                    "y_family": eval_family(y_eval),
                    "pair_label": (
                        f"{COMPACT_EVAL_LABELS.get(x_eval, x_eval)} ↔ "
                        f"{COMPACT_EVAL_LABELS.get(y_eval, y_eval)}"
                    ),
                    "rho": float(rho),
                    "abs_rho": abs_rho,
                    "rho_min": source["rho_min"],
                    "rho_max": source["rho_max"],
                    "abs_rho_min": abs_rho_min,
                    "abs_rho_max": abs_rho_max,
                    "abs_rho_interval_min": (
                        min(abs_rho, abs_rho_min) if abs_rho_min is not None else abs_rho
                    ),
                    "abs_rho_interval_max": (
                        max(abs_rho, abs_rho_max) if abs_rho_max is not None else abs_rho
                    ),
                    "rho_variant_mean": source["rho_variant_mean"],
                    "rho_variant_median": source["rho_variant_median"],
                    "rho_variant_count": source["rho_variant_count"],
                    "n_models": source["n_models"],
                    "n_pairs": source["n_pairs"],
                }
            )
    redundancy = pl.DataFrame(redundancy_rows).sort("abs_rho", descending=True)
    diagnostics = pl.DataFrame(
        {
            "method": ["pairwise-expanded Spearman; PSD eigenvalue clipping + unit diagonal"],
            "eval_count": [len(eval_order)],
            "retained_components_80pct": [retained],
            "minimum_raw_eigenvalue": [float(raw_eigenvalues.min())],
            "negative_raw_eigenvalues": [int((raw_eigenvalues < -1e-10).sum())],
            "projection_frobenius_delta": [float(np.linalg.norm(projected - matrix))],
        }
    )
    return variance, loadings, redundancy, diagnostics


def bootstrap_pca_intervals(
    run_scores: pl.DataFrame,
    variance: pl.DataFrame,
    loadings: pl.DataFrame,
    eval_order: list[str],
    *,
    samples: int = 1_000,
    seed: int = 20_260_805,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Propagate replica selection through the PCA correlation structure."""
    usable = run_scores.filter(pl.col("status").is_in(RANKABLE_STATUSES))
    models = sorted(usable["model"].unique().to_list())
    model_index = {model: index for index, model in enumerate(models)}
    eval_index = {eval_name: index for index, eval_name in enumerate(eval_order)}
    cell_scores = {
        (str(model), str(eval_name)): np.asarray(group["score"].to_list(), dtype=float)
        for (model, eval_name), group in usable.partition_by(
            ["model", "eval"], as_dict=True
        ).items()
    }
    reference_components = np.empty((len(eval_order), len(eval_order)), dtype=float)
    for row in loadings.iter_rows(named=True):
        reference_components[int(row["pc_index"]) - 1, eval_index[row["eval"]]] = float(
            row["component_loading"]
        )

    rng = np.random.default_rng(seed)
    explained_draws: list[np.ndarray] = []
    cumulative_draws: list[np.ndarray] = []
    loading_draws: list[np.ndarray] = []
    for _ in range(samples):
        scores = np.full((len(models), len(eval_order)), np.nan, dtype=float)
        for (model, eval_name), runs in cell_scores.items():
            if eval_name not in eval_index:
                continue
            scores[model_index[model], eval_index[eval_name]] = runs[rng.integers(len(runs))]

        correlation = np.eye(len(eval_order), dtype=float)
        valid = True
        for left in range(len(eval_order)):
            for right in range(left + 1, len(eval_order)):
                observed = ~np.isnan(scores[:, left]) & ~np.isnan(scores[:, right])
                if observed.sum() < 3:
                    valid = False
                    break
                rho = _spearman(scores[observed, left].tolist(), scores[observed, right].tolist())
                correlation[left, right] = rho
                correlation[right, left] = rho
            if not valid:
                break
        if not valid:
            continue

        _, _, eigenvalues, components, explained, cumulative = _pca_from_correlation(correlation)
        explained_draws.append(explained)
        cumulative_draws.append(cumulative)

        # Match bootstrap components to the reference by absolute vector
        # similarity, then orient their signs to the reference. This prevents
        # arbitrary sign flips and near-tied component swaps from inflating the
        # loading intervals.
        remaining = set(range(len(eval_order)))
        aligned_components = np.empty_like(components)
        aligned_eigenvalues = np.empty_like(eigenvalues)
        for reference_pc in range(len(eval_order)):
            candidate = max(
                remaining,
                key=lambda pc: abs(
                    float(np.dot(reference_components[reference_pc], components[pc]))
                ),
            )
            remaining.remove(candidate)
            sign = (
                1.0
                if np.dot(reference_components[reference_pc], components[candidate]) >= 0
                else -1.0
            )
            aligned_components[reference_pc] = components[candidate] * sign
            aligned_eigenvalues[reference_pc] = eigenvalues[candidate]
        loading_draws.append(aligned_components.T * np.sqrt(aligned_eigenvalues))

    if not explained_draws or not loading_draws:
        return variance, loadings

    explained_array = np.stack(explained_draws)
    cumulative_array = np.stack(cumulative_draws)
    variance_intervals = pl.DataFrame(
        {
            "pc_index": list(range(1, len(eval_order) + 1)),
            "explained_low": np.quantile(explained_array, 0.025, axis=0),
            "explained_high": np.quantile(explained_array, 0.975, axis=0),
            "cumulative_low": np.quantile(cumulative_array, 0.025, axis=0),
            "cumulative_high": np.quantile(cumulative_array, 0.975, axis=0),
            "bootstrap_samples": len(explained_draws),
        }
    )
    loading_array = np.stack(loading_draws)
    loading_rows = []
    for eval_name, eval_position in eval_index.items():
        for pc in range(len(eval_order)):
            distribution = loading_array[:, eval_position, pc]
            loading_rows.append(
                {
                    "eval": eval_name,
                    "pc_index": pc + 1,
                    "loading_low": float(np.quantile(distribution, 0.025)),
                    "loading_high": float(np.quantile(distribution, 0.975)),
                    "bootstrap_samples": len(loading_draws),
                }
            )
    return (
        variance.join(variance_intervals, on="pc_index", how="left"),
        loadings.join(pl.DataFrame(loading_rows), on=["eval", "pc_index"], how="left"),
    )


def chart_eval_pca(
    variance: pl.DataFrame,
    loadings: pl.DataFrame,
    redundancy: pl.DataFrame,
    diagnostics: pl.DataFrame,
    eval_order: list[str],
) -> alt.HConcatChart:
    retained = int(diagnostics["retained_components_80pct"][0])
    scree_data = variance.head(8)
    pc_order = scree_data["pc"].to_list()
    scree_base = alt.Chart(scree_data)
    scree = scree_base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("pc:N", sort=pc_order, title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y(
            "explained_variance:Q",
            title="share of rank variance",
            axis=alt.Axis(format=".0%"),
        ),
        color=alt.Color(
            "retained_80pct:N",
            scale=alt.Scale(domain=[True, False], range=[PRIMARY, DEEMPHASIS]),
            legend=None,
        ),
        tooltip=[
            "pc:N",
            alt.Tooltip("explained_variance:Q", format=".1%"),
            alt.Tooltip("cumulative_variance:Q", format=".1%"),
        ],
    )
    cumulative = scree_base.mark_line(point=True, color=SERIES[1]).encode(
        x=alt.X("pc:N", sort=pc_order),
        y=alt.Y("cumulative_variance:Q", axis=alt.Axis(format=".0%"), title=None),
    )
    scree_errors = scree_base.mark_rule(color=INK, strokeWidth=1.3).encode(
        x=alt.X("pc:N", sort=pc_order),
        y=alt.Y("explained_low:Q"),
        y2=alt.Y2("explained_high:Q"),
    )
    scree_error_low = scree_base.mark_tick(
        color=INK, orient="horizontal", thickness=1.3, size=9
    ).encode(x=alt.X("pc:N", sort=pc_order), y=alt.Y("explained_low:Q"))
    scree_error_high = scree_base.mark_tick(
        color=INK, orient="horizontal", thickness=1.3, size=9
    ).encode(x=alt.X("pc:N", sort=pc_order), y=alt.Y("explained_high:Q"))
    scree_panel = (
        scree + cumulative + scree_errors + scree_error_low + scree_error_high
    ).properties(
        width=260,
        height=330,
        title=alt.Title(
            "Variance by component",
            subtitle=f"{retained} PCs reach 80% · bootstrap 95%",
        ),
    )

    shown_pcs = pc_order[: min(3, len(pc_order))]
    loading_data = loadings.filter(pl.col("pc").is_in(shown_pcs))
    eval_labels = [CANONICAL_TO_DEV.get(name, name) for name in eval_order]
    loading_panels = []
    for pc_position, pc in enumerate(shown_pcs):
        pc_data = loading_data.filter(pl.col("pc") == pc)
        loading_base = alt.Chart(pc_data).encode(
            y=alt.Y(
                "eval_label:N",
                sort=eval_labels,
                title=None,
                axis=alt.Axis(labelLimit=145) if pc_position == 0 else None,
            )
        )
        zero = (
            alt.Chart(pl.DataFrame({"zero": [0.0]}))
            .mark_rule(color=AXIS, strokeDash=[3, 3])
            .encode(x=alt.X("zero:Q"))
        )
        loading_intervals = loading_base.mark_rule(color=INK, strokeWidth=1.15).encode(
            x=alt.X(
                "loading_low:Q",
                title="eval–PC corr.",
                scale=alt.Scale(domain=[-1, 1]),
                axis=alt.Axis(format=".1f", tickCount=3),
            ),
            x2=alt.X2("loading_high:Q"),
        )
        loading_points = loading_base.mark_point(
            filled=True, color=SERIES[pc_position], size=48
        ).encode(
            x=alt.X(
                "loading_correlation:Q",
                scale=alt.Scale(domain=[-1, 1]),
                axis=alt.Axis(format=".1f", tickCount=3),
            ),
            tooltip=[
                "eval_label:N",
                "pc:N",
                alt.Tooltip("loading_correlation:Q", format=".3f"),
                alt.Tooltip("loading_low:Q", title="bootstrap 2.5%", format=".3f"),
                alt.Tooltip("loading_high:Q", title="bootstrap 97.5%", format=".3f"),
            ],
        )
        loading_panels.append(
            (zero + loading_intervals + loading_points).properties(
                width=72,
                height=alt.Step(27),
                title=alt.Title(pc, anchor="middle", frame="group"),
            )
        )

    downstream_evals = [*AGENTIC_EVALS, "LitSearch-rerank", *FRONTIER_EVALS]
    base_sentinel_evals = [*BASE_EVALS, *SENTINEL_EVALS]
    x_is_downstream = pl.col("x_eval").is_in(downstream_evals)
    y_is_downstream = pl.col("y_eval").is_in(downstream_evals)
    x_is_base_sentinel = pl.col("x_eval").is_in(base_sentinel_evals)
    y_is_base_sentinel = pl.col("y_eval").is_in(base_sentinel_evals)
    redundancy_groups = [
        (
            "Downstream ↔ base/sentinel",
            (x_is_downstream & y_is_base_sentinel) | (x_is_base_sentinel & y_is_downstream),
        ),
        (
            "Base/sentinel ↔ base/sentinel",
            x_is_base_sentinel & y_is_base_sentinel,
        ),
        (
            "Downstream ↔ downstream",
            x_is_downstream & y_is_downstream,
        ),
    ]

    def redundancy_group_chart(
        title: str, predicate: pl.Expr, *, show_x_axis: bool
    ) -> alt.LayerChart:
        top_pairs = (
            redundancy.filter(predicate)
            .head(5)
            .with_columns(rho_label=pl.col("rho").round(2).cast(pl.String))
        )
        pair_order = top_pairs["pair_label"].to_list()
        pair_axis = alt.Y(
            "pair_label:N",
            sort=pair_order,
            title=None,
            bandPosition=0.5,
            axis=alt.Axis(labelLimit=245),
        )
        base = alt.Chart(top_pairs)
        bars = base.mark_bar(color=SERIES[4], cornerRadiusEnd=3, size=17).encode(
            x=alt.X(
                "abs_rho:Q",
                title="|Spearman ρ|" if show_x_axis else None,
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(tickCount=6) if show_x_axis else None,
            ),
            y=pair_axis,
            tooltip=[
                "pair_label:N",
                alt.Tooltip("rho:Q", format=".3f"),
                alt.Tooltip("rho_min:Q", title="pairing min ρ", format=".3f"),
                alt.Tooltip("rho_max:Q", title="pairing max ρ", format=".3f"),
                alt.Tooltip("rho_variant_count:Q", title="replica pairings"),
                "n_models:Q",
                "n_pairs:Q",
            ],
        )
        errors = base.mark_rule(color=INK, strokeWidth=1.2).encode(
            x=alt.X("abs_rho_interval_min:Q"),
            x2=alt.X2("abs_rho_interval_max:Q"),
            y=pair_axis,
        )
        error_min = base.mark_tick(color=INK, orient="vertical", thickness=1.2, size=9).encode(
            x=alt.X("abs_rho_interval_min:Q"), y=pair_axis
        )
        error_max = base.mark_tick(color=INK, orient="vertical", thickness=1.2, size=9).encode(
            x=alt.X("abs_rho_interval_max:Q"), y=pair_axis
        )
        labels = base.mark_text(align="left", dx=4, fontSize=8).encode(
            x=alt.X("abs_rho_interval_max:Q"), y=pair_axis, text="rho_label:N"
        )
        return (bars + errors + error_min + error_max + labels).properties(
            width=300,
            height=alt.Step(21),
            title=title,
        )

    redundancy_panels = [
        redundancy_group_chart(title, predicate, show_x_axis=index == 2)
        for index, (title, predicate) in enumerate(redundancy_groups)
    ]
    redundancy_panel = (
        alt.vconcat(*redundancy_panels, spacing=15)
        .resolve_scale(x="shared")
        .properties(
            title=alt.Title(
                "Most redundant pairs",
                subtitle="Top five per group · bar: pooled |ρ| · whisker: pooled + pairing range",
                offset=14,
            )
        )
    )

    return alt.hconcat(scree_panel, *loading_panels, redundancy_panel, spacing=16).properties(
        title=alt.Title(
            "Eval covariance",
            subtitle=[
                "Spearman rank correlation over all within-model replica pairs",
                "PCA: replica-bootstrap 95% · redundancy: pooled + pairing range",
                (
                    "Tentative interpretation: PC1 general capability · "
                    "PC2 instruction following ↔ open retrieval · "
                    "PC3 closed-form QA ↔ research synthesis"
                ),
            ],
        )
    )


def _correlation_heatmap(
    data: pl.DataFrame,
    x_order: list[str],
    y_order: list[str],
    title: str,
    width_step: int,
    height_step: int,
    show_legend: bool,
    show_family_legend: bool,
) -> alt.LayerChart:
    rail_size = 7
    rail_gap = 1
    axis_padding = rail_size + rail_gap + 4
    base = alt.Chart(data).encode(
        x=alt.X(
            "x_label:N",
            sort=x_order,
            title=None,
            axis=alt.Axis(
                labelAngle=-35,
                labelPadding=axis_padding,
                domain=False,
                ticks=False,
            ),
        ),
        y=alt.Y(
            "y_label:N",
            sort=y_order,
            title=None,
            axis=alt.Axis(labelPadding=axis_padding, domain=False, ticks=False),
        ),
    )
    cells = base.mark_rect(stroke=None).encode(
        color=alt.Color(
            "rho:Q",
            scale=alt.Scale(domain=[-1, 1], scheme="brownbluegreen", clamp=True),
            legend=alt.Legend(title="Spearman ρ") if show_legend else None,
        ),
        tooltip=[
            "x_label:N",
            "y_label:N",
            alt.Tooltip("rho:Q", title="Spearman ρ", format=".3f"),
            alt.Tooltip("rho_min:Q", title="pairing min ρ", format=".3f"),
            alt.Tooltip("rho_max:Q", title="pairing max ρ", format=".3f"),
            alt.Tooltip("rho_variant_count:Q", title="replica pairings"),
            alt.Tooltip("pearson_r:Q", title="Pearson r", format=".3f"),
            alt.Tooltip("sample_covariance:Q", title="sample covariance", format=".4f"),
            alt.Tooltip("n_models:Q", title="models"),
            alt.Tooltip("n_pairs:Q", title="replica pairs"),
            alt.Tooltip("min_pairs_per_model:Q", title="min pairs/model"),
            alt.Tooltip("max_pairs_per_model:Q", title="max pairs/model"),
        ],
    )
    labels = base.mark_text(fontSize=8).encode(
        text="rho_label:N",
        color=alt.condition("datum.rho < 0", alt.value("white"), alt.value(INK)),
    )
    family_scale = alt.Scale(
        domain=[FAMILY_LABELS[name] for name in FAMILY_ORDER],
        range=[FAMILY_COLOR[name] for name in FAMILY_ORDER],
    )
    plot_height = height_step * len(y_order)
    bottom_rail = (
        alt.Chart(data.unique(subset=["x_label"]))
        .mark_rect(clip=False)
        .encode(
            x=alt.X("x_label:N", sort=x_order, title=None),
            y=alt.value(plot_height + rail_gap),
            y2=alt.value(plot_height + rail_gap + rail_size),
            color=alt.Color(
                "x_family_label:N",
                scale=family_scale,
                legend=alt.Legend(title=None, orient="right", direction="vertical")
                if show_family_legend
                else None,
            ),
        )
    )
    left_rail = (
        alt.Chart(data.unique(subset=["y_label"]))
        .mark_rect(clip=False)
        .encode(
            x=alt.value(-rail_gap - rail_size),
            x2=alt.value(-rail_gap),
            y=alt.Y("y_label:N", sort=y_order, title=None),
            color=alt.Color("y_family_label:N", scale=family_scale, legend=None),
        )
    )
    return (
        (cells + bottom_rail + left_rail + labels)
        .resolve_scale(color="independent")
        .properties(
            width=alt.Step(width_step),
            height=alt.Step(height_step),
            title=title,
        )
    )


def chart_covariance(run_scores: pl.DataFrame) -> alt.HConcatChart:
    matrix_evals = ALL_ANALYSIS_EVALS
    matrix_labels = [CANONICAL_TO_DEV.get(name, name) for name in matrix_evals]
    matrix = pairwise_replica_correlations(run_scores, matrix_evals, matrix_evals)
    predictive = pairwise_replica_correlations(run_scores, AGENTIC_EVALS, PROXY_EVALS)
    left = _correlation_heatmap(
        matrix,
        matrix_labels,
        matrix_labels,
        "All evals",
        40,
        40,
        True,
        True,
    )
    right = _correlation_heatmap(
        predictive,
        [CANONICAL_TO_DEV[name] for name in AGENTIC_EVALS],
        PROXY_EVALS,
        "Proxy → agentic",
        76,
        56,
        False,
        False,
    )
    return alt.hconcat(left, right, spacing=36).properties(
        title=alt.Title(
            "Cross-eval rank correlation",
            subtitle="Spearman ρ · all within-model replica pairs",
        )
    )


def write_summary_md(
    df: pl.DataFrame, sci_lit: pl.DataFrame, excluded: list[str], path: Path
) -> None:
    counts = df.group_by("status").len().sort("len", descending=True).rows_by_key("status")
    lines = ["# Science eval sweep — generated summary", ""]

    total = df.height
    filled = df.filter(pl.col("score").is_not_null()).height
    lines += [f"{filled} of {total} cells have a score ({filled / total:.0%}).", ""]

    lines += ["## Row status", "", "| status | cells |", "|---|---|"]
    for status in STATUS_ORDER:
        lines.append(f"| {status} | {counts.get(status, [(0,)])[0][0]} |")

    lines += [
        "",
        "## Sci-lit standing",
        "",
        "| model | mean pct rank | replica-resampling 95% | evals |",
        "|---|---|---|---|",
    ]
    for row in sci_lit.iter_rows(named=True):
        interval = f"{row['mean_pct_low']:.3f}–{row['mean_pct_high']:.3f}"
        lines.append(f"| {row['model']} | {row['mean_pct']:.3f} | {interval} | {row['n_evals']} |")
    if excluded:
        lines += ["", f"Below the coverage threshold, not ranked: {', '.join(excluded)}."]

    suspect = df.filter(pl.col("status") == "suspect").sort(["model", "eval"])
    if suspect.height:
        lines += [
            "",
            "## Suspect scores",
            "",
            "| model | eval | score | run id | reason |",
            "|---|---|---|---|---|",
        ]
        for row in suspect.iter_rows(named=True):
            lines.append(
                f"| {row['model']} | {row['eval']} | {row['score']:.3f} | "
                f"{row['run_id'] or '—'} | {row['reason']} |"
            )

    path.write_text("\n".join(lines) + "\n")


def save(chart: alt.TopLevelMixin, out_dir: Path, name: str, formats: list[str]) -> None:
    for fmt in formats:
        target = out_dir / f"{name}.{fmt}"
        chart.save(target, scale_factor=2.0 if fmt == "png" else 1.0)
        print(f"  {target}")


SLIDE_TITLES = {
    "coverage": "Eval coverage",
    "scores_sci-lit": "Sci-lit scores",
    "scores_base": "Base eval scores",
    "scores_sentinel": "Sentinel scores",
    "scores_frontier-science": "FrontierScience scores",
    "profile": "Cross-eval profile",
    "dev_vs_full": "Fixed dev vs. full set",
    "covariance": "Cross-eval rank correlation",
    "deepscholar_ifeval": "IFEval and DeepScholar-Bench",
    "eval_pca": "Eval covariance",
    "summary": "Overall standing",
}


def _data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return ""
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def write_deck(
    out_dir: Path,
    generated: list[str],
    df: pl.DataFrame,
    sci_lit: pl.DataFrame,
    excluded: list[str],
) -> Path:
    """Fold the rendered charts into one self-contained HTML deck.

    Images are inlined as data URIs so the file opens over file:// with no
    network and no sibling assets to keep alongside it.
    """
    total, filled = df.height, df.filter(pl.col("score").is_not_null()).height
    counts = {s: df.filter(pl.col("status") == s).height for s in STATUS_ORDER}
    agentic = df.filter(pl.col("eval").is_in(AGENTIC_EVALS))
    supported_agentic = agentic.filter(pl.col("status") != "unsupported")
    scored_agentic = supported_agentic.filter(pl.col("score").is_not_null()).height

    tiles = "".join(
        f'<div class="tile"><div class="tile-v">{value}</div>'
        f'<div class="tile-k">{escape(key)}</div></div>'
        for key, value in [
            ("agentic dev", f"{scored_agentic}/{supported_agentic.height}"),
            ("scored cells", f"{filled}/{total}"),
            ("coverage", f"{filled / total:.0%}"),
            ("suspect", counts["suspect"]),
            ("unsupported", counts["unsupported"]),
            ("not run", counts["not-run"]),
        ]
    )

    slides = [
        f"""<section class="slide">
          <p class="eyebrow">Science and agentic benchmarks</p>
          <h1>Science eval sweep</h1>
          <div class="tiles">{tiles}</div>
        </section>"""
    ]

    predictor_taxonomy = _table(
        ["eval", "cheap", "base-compatible", "use"],
        [
            [
                "ARC, MMLU-STEM, MedMCQA, MedQA, SciQ",
                "Yes",
                "Yes",
                "Pre- and post-training science proxy",
            ],
            [
                "LitSearch-rerank",
                "Yes",
                "No — chat",
                "Post-training science proxy",
            ],
            [
                "IFEval",
                "Yes",
                "No — instruction following",
                "Post-training control",
            ],
            [
                "Full MMLU sentinel",
                "Yes",
                "MC/RC",
                "Broad knowledge retention",
            ],
            [
                "MATH-500",
                "Yes",
                "Completion/BPB",
                "Both; separate protocols",
            ],
            [
                "FS Olympiad",
                "No",
                "No — chat + judge",
                "Capability proxy",
            ],
            [
                "FS Research",
                "No",
                "No — chat + judge",
                "Downstream; shared responses",
            ],
        ],
    )
    slides.append(
        f"""<section class="slide">
          <h2>Cheap and base-compatible evals</h2>
          <p class="sub">Base-compatible: valid before instruction tuning ·
             Cheap: practical for frequent checks</p>
          {predictor_taxonomy}
        </section>"""
    )

    for name in generated:
        title = SLIDE_TITLES[name]
        png = out_dir / f"{name}.png"
        if not png.exists():
            continue
        slides.append(
            f"""<section class="slide">
              <div class="figure"><img src="{_data_uri(png)}" alt="{escape(title)}"></div>
            </section>"""
        )

    standing = _table(
        ["model", "mean rank", "95%", "evals"],
        [
            [
                r["model"],
                f"{r['mean_pct']:.3f}",
                f"{r['mean_pct_low']:.3f}–{r['mean_pct_high']:.3f}",
                r["n_evals"],
            ]
            for r in sci_lit.iter_rows(named=True)
        ],
    )
    suspect = _table(
        ["model", "eval", "score", "reason"],
        [
            [r["model"], r["eval"], f"{r['score']:.3f}", r["reason"]]
            for r in df.filter(pl.col("status") == "suspect")
            .sort(["model", "eval"])
            .iter_rows(named=True)
        ],
    )
    excluded_note = (
        f'<p class="sub">Not ranked below coverage threshold: {escape(", ".join(excluded))}.</p>'
        if excluded
        else ""
    )
    slides.append(
        f"""<section class="slide">
          <h2>Sci-lit standing</h2>{standing}{excluded_note}
          <h3>Excluded scores</h3>{suspect}
        </section>"""
    )

    document = _DECK_TEMPLATE.format(slides="\n".join(slides), count=len(slides))
    target = out_dir / "index.html"
    target.write_text(document)
    return target


_DECK_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Science eval sweep</title>
<style>
  :root {{
    --surface: #f8fafc; --card: #ffffff; --plane: #eef1f6; --ink: #172033;
    --ink2: #5f687a; --muted: #8c95a5; --rule: #e2e7ef; --accent: #405bd8;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--plane); color: var(--ink);
    font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  .deck {{ max-width: 1280px; margin: 0 auto; padding: 28px 24px 100px; }}
  .slide {{
    display: none; position: relative; overflow: hidden; background: var(--surface);
    border: 1px solid rgba(23, 32, 51, 0.07); border-radius: 20px;
    box-shadow: 0 22px 54px rgba(23, 32, 51, 0.10);
    padding: 38px 44px 34px; min-height: calc(100vh - 152px);
  }}
  .slide::before {{
    content: ""; position: absolute; inset: 0 0 auto; height: 5px;
    background: linear-gradient(90deg, var(--accent), #835cc7 54%, #269587);
  }}
  .slide.on {{ display: flex; flex-direction: column; }}
  h1 {{ font-size: clamp(36px, 4vw, 52px); line-height: 1.08; margin: 10px 0 12px;
        letter-spacing: -0.035em; }}
  h2 {{ font-size: 28px; line-height: 1.15; margin: 0 0 8px; letter-spacing: -0.025em; }}
  h3 {{ font-size: 13px; margin: 26px 0 8px; color: var(--ink2);
        text-transform: uppercase; letter-spacing: 0.05em; }}
  .eyebrow {{ color: var(--accent); font-size: 11px; font-weight: 700;
              letter-spacing: 0.11em; text-transform: uppercase; margin: 0 0 10px; }}
  .sub {{ color: var(--ink2); margin: 0 0 18px; max-width: 82ch; font-size: 15px; }}
  .figure {{ flex: 1; display: flex; align-items: center; justify-content: center;
             overflow: auto; padding: 8px 0 2px; }}
  .figure img {{ max-width: 100%; max-height: calc(100vh - 260px); width: auto; height: auto;
                 display: block; margin: auto; }}
  .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px; margin: 34px 0 24px; }}
  .tile {{ background: rgba(255, 255, 255, 0.72); border: 1px solid var(--rule);
           border-radius: 14px; padding: 17px 18px; }}
  .tile-v {{ color: var(--accent); font-size: 30px; font-weight: 650;
             letter-spacing: -0.035em; font-variant-numeric: tabular-nums; }}
  .tile-k {{ color: var(--ink2); font-size: 12px; margin-top: 3px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 4px 0 14px; font-size: 13px; }}
  th, td {{ text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--rule);
            vertical-align: top; }}
  th {{ color: var(--ink2); font-weight: 600; font-size: 11px;
        text-transform: uppercase; letter-spacing: 0.04em; }}
  td:nth-child(2), td:nth-child(3) {{ font-variant-numeric: tabular-nums; }}
  .bar {{
    position: fixed; left: 0; right: 0; bottom: 0; background: rgba(248, 250, 252, 0.94);
    backdrop-filter: blur(12px); border-top: 1px solid var(--rule); padding: 11px 20px;
    display: flex; gap: 12px; align-items: center; justify-content: center;
  }}
  button {{
    font: inherit; font-size: 13px; font-weight: 600; color: var(--ink); background: var(--card);
    border: 1px solid var(--rule); border-radius: 999px; padding: 7px 16px; cursor: pointer;
    box-shadow: 0 2px 8px rgba(23, 32, 51, 0.05);
  }}
  button:hover {{ border-color: var(--accent); color: var(--accent); }}
  .count {{ color: var(--ink2); font-size: 13px; font-variant-numeric: tabular-nums;
            min-width: 82px; text-align: center; }}
  @media print {{
    .bar {{ display: none; }}
    .slide {{ display: block; page-break-after: always; border: 0; min-height: 0; }}
  }}
  @media (max-width: 760px) {{
    .deck {{ padding: 12px 10px 84px; }}
    .slide {{ padding: 28px 22px; border-radius: 14px; min-height: calc(100vh - 108px); }}
    .figure {{ justify-content: flex-start; }}
    .figure img {{ max-width: none; max-height: none; }}
  }}
</style></head>
<body>
<div class="deck">{slides}</div>
<div class="bar">
  <button id="prev">&larr; Prev</button>
  <span class="count"><span id="now">1</span> / {count}</span>
  <button id="next">Next &rarr;</button>
</div>
<script>
  const slides = [...document.querySelectorAll('.slide')];
  const now = document.getElementById('now');
  let i = 0;
  function show(n) {{
    i = Math.max(0, Math.min(slides.length - 1, n));
    slides.forEach((s, k) => s.classList.toggle('on', k === i));
    now.textContent = i + 1;
    location.hash = i + 1;
    window.scrollTo(0, 0);
  }}
  document.getElementById('prev').onclick = () => show(i - 1);
  document.getElementById('next').onclick = () => show(i + 1);
  document.addEventListener('keydown', e => {{
    if (e.key === 'ArrowRight' || e.key === ' ') {{ e.preventDefault(); show(i + 1); }}
    if (e.key === 'ArrowLeft') {{ e.preventDefault(); show(i - 1); }}
  }});
  show(parseInt(location.hash.slice(1), 10) - 1 || 0);
</script>
</body></html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=HERE / "data" / "results.csv")
    parser.add_argument("--metadata", type=Path, default=HERE / "metadata.toml")
    parser.add_argument("--out", type=Path, default=HERE / "out")
    parser.add_argument(
        "--formats", nargs="+", default=["png", "html"], choices=["png", "html", "svg"]
    )
    parser.add_argument(
        "--min-coverage",
        type=int,
        default=5,
        help="sci-lit evals a model needs before it gets a composite (default: 5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    _register_theme()

    meta = load_metadata(args.metadata)
    df, scope_scores, run_scores = load_scores(args.csv, meta)
    rank_intervals, composite_intervals = bootstrap_rank_intervals(run_scores, meta)
    ranked = add_ranks(df).join(rank_intervals, on=["model", "eval"], how="left")

    n_sci_lit = len(meta.sci_lit_evals())
    n_base = meta.evals.filter(pl.col("family") == "base").height
    n_frontier = meta.evals.filter(pl.col("family") == "frontier-science").height
    n_sentinel = meta.evals.filter(pl.col("family") == "sentinel").height
    sci_lit_all = composite(ranked, "sci-lit").join(
        composite_intervals.filter(pl.col("family") == "sci-lit").drop("family"),
        on="model",
        how="left",
    )
    sci_lit = sci_lit_all.filter(pl.col("n_evals") >= args.min_coverage).with_columns(
        label=pl.format("n={}/{}", pl.col("n_evals"), pl.lit(n_sci_lit))
    )
    excluded = sci_lit_all.filter(pl.col("n_evals") < args.min_coverage)["model"].to_list()
    ranked_models = sci_lit["model"].to_list()
    model_grid = pl.DataFrame({"model": ranked_models})

    def companion_scores(family: str, n_evals: int) -> pl.DataFrame:
        return (
            model_grid.join(composite(ranked, family), on="model", how="left")
            .join(
                composite_intervals.filter(pl.col("family") == family).drop("family"),
                on="model",
                how="left",
            )
            .with_columns(
                n_evals=pl.col("n_evals").fill_null(0),
                display_pct=pl.col("mean_pct").fill_null(0.0),
            )
            .with_columns(label=pl.format("n={}/{}", pl.col("n_evals"), pl.lit(n_evals)))
        )

    base_scores = companion_scores("base", n_base)
    frontier = companion_scores("frontier-science", n_frontier)
    sentinel = companion_scores("sentinel", n_sentinel)

    print(f"writing to {args.out}")
    rank_intervals.write_csv(args.out / "rank_bootstrap_intervals.csv")
    composite_intervals.write_csv(args.out / "composite_bootstrap_intervals.csv")
    print(f"  {args.out / 'rank_bootstrap_intervals.csv'}")
    print(f"  {args.out / 'composite_bootstrap_intervals.csv'}")
    generated = [
        "coverage",
        "scores_sci-lit",
        "scores_base",
        "scores_frontier-science",
        "scores_sentinel",
    ]
    save(chart_coverage(df, meta), args.out, "coverage", args.formats)
    save(chart_scores(df, meta, "sci-lit"), args.out, "scores_sci-lit", args.formats)
    save(chart_scores(df, meta, "base"), args.out, "scores_base", args.formats)
    save(
        chart_scores(df, meta, "frontier-science"),
        args.out,
        "scores_frontier-science",
        args.formats,
    )
    save(chart_scores(df, meta, "sentinel"), args.out, "scores_sentinel", args.formats)
    dev_full_pairs = trusted_dev_full_pairs(scope_scores)
    dev_full_pairs.write_csv(args.out / "dev_full_pairs.csv")
    print(f"  {args.out / 'dev_full_pairs.csv'}")
    dev_full = chart_dev_vs_full(dev_full_pairs)
    if dev_full is not None:
        save(dev_full, args.out, "dev_vs_full", args.formats)
        generated.append("dev_vs_full")
    save(chart_covariance(run_scores), args.out, "covariance", args.formats)
    generated.append("covariance")
    deepscholar_pairs = deepscholar_ifeval_pairs(run_scores)
    deepscholar_pairs.write_csv(args.out / "deepscholar_ifeval_pairs.csv")
    print(f"  {args.out / 'deepscholar_ifeval_pairs.csv'}")
    regression_band, regression_summary = bootstrap_deepscholar_regression(deepscholar_pairs)
    if not regression_band.is_empty():
        regression_band.write_csv(args.out / "deepscholar_ifeval_bootstrap.csv")
        pl.DataFrame([regression_summary]).write_csv(args.out / "deepscholar_ifeval_regression.csv")
        print(f"  {args.out / 'deepscholar_ifeval_bootstrap.csv'}")
        print(f"  {args.out / 'deepscholar_ifeval_regression.csv'}")
        save(
            chart_deepscholar_ifeval(
                deepscholar_pairs,
                regression_band,
                regression_summary,
            ),
            args.out,
            "deepscholar_ifeval",
            args.formats,
        )
    regression_specs = [
        (
            "deepscholar_cheap_regression",
            CHEAP_EVALS,
            "Cheap evals → DeepScholar-Bench",
            "cheap direct evals",
        ),
        (
            "deepscholar_proxy_regression",
            PROXY_EVALS,
            "Proxy evals → DeepScholar-Bench",
            "available proxy evals",
        ),
    ]
    for output_name, predictor_evals, title, predictor_label in regression_specs:
        predictions, coefficients, bootstrap, regression_summary = (
            bootstrap_deepscholar_proxy_regression(run_scores, predictor_evals)
        )
        if predictions.is_empty():
            continue
        predictions.write_csv(args.out / f"{output_name}_predictions.csv")
        coefficients.write_csv(args.out / f"{output_name}_coefficients.csv")
        bootstrap.write_csv(args.out / f"{output_name}_bootstrap.csv")
        pl.DataFrame([regression_summary]).write_csv(args.out / f"{output_name}_summary.csv")
        for suffix in ["predictions", "coefficients", "bootstrap", "summary"]:
            print(f"  {args.out / f'{output_name}_{suffix}.csv'}")
        save(
            chart_deepscholar_proxy_regression(
                predictions,
                coefficients,
                regression_summary,
                predictor_evals,
                title=title,
                predictor_label=predictor_label,
            ),
            args.out,
            output_name,
            args.formats,
        )
    pairwise = pairwise_replica_correlations(run_scores, ALL_ANALYSIS_EVALS, ALL_ANALYSIS_EVALS)
    pairwise.write_csv(args.out / "pairwise_replica_correlations.csv")
    print(f"  {args.out / 'pairwise_replica_correlations.csv'}")
    proxy_agentic = pairwise_replica_correlations(run_scores, AGENTIC_EVALS, PROXY_EVALS)
    proxy_agentic.write_csv(args.out / "proxy_agentic_correlations.csv")
    print(f"  {args.out / 'proxy_agentic_correlations.csv'}")
    pca_variance, pca_loadings, redundancy, pca_diagnostics = analyze_eval_pca(
        pairwise, ALL_ANALYSIS_EVALS
    )
    pca_variance, pca_loadings = bootstrap_pca_intervals(
        run_scores,
        pca_variance,
        pca_loadings,
        ALL_ANALYSIS_EVALS,
    )
    pca_variance.write_csv(args.out / "eval_pca_variance.csv")
    pca_loadings.write_csv(args.out / "eval_pca_loadings.csv")
    redundancy.write_csv(args.out / "eval_redundancy.csv")
    pca_diagnostics.write_csv(args.out / "eval_pca_diagnostics.csv")
    for name in [
        "eval_pca_variance.csv",
        "eval_pca_loadings.csv",
        "eval_redundancy.csv",
        "eval_pca_diagnostics.csv",
    ]:
        print(f"  {args.out / name}")
    save(
        chart_eval_pca(
            pca_variance,
            pca_loadings,
            redundancy,
            pca_diagnostics,
            ALL_ANALYSIS_EVALS,
        ),
        args.out,
        "eval_pca",
        args.formats,
    )
    generated.append("eval_pca")
    if not regression_band.is_empty():
        generated.append("deepscholar_ifeval")
    if ranked_models:
        save(
            chart_profile(ranked, meta, ranked_models),
            args.out,
            "profile",
            args.formats,
        )
        pool_sizes = ranked.filter(pl.col("family") == "sci-lit")["n_in_eval"]
        save(
            chart_summary(
                sci_lit,
                base_scores,
                frontier,
                sentinel,
                n_sci_lit,
                (pool_sizes.min(), pool_sizes.max()),
            ),
            args.out,
            "summary",
            args.formats,
        )
        generated += ["profile", "summary"]
    write_summary_md(df, sci_lit, excluded, args.out / "summary.md")
    print(f"  {args.out / 'summary.md'}")

    if "png" in args.formats:
        print(f"  {write_deck(args.out, generated, df, sci_lit, excluded)}")
    else:
        print("  deck needs --formats png; skipped")


if __name__ == "__main__":
    main()

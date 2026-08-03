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

# Vega does not ship seaborn's `vlag` scheme, so use a sampled equivalent:
# cool blue for negative association, a neutral zero, and warm red positive.
VLAG_DOMAIN = [-1.0, -0.66, -0.33, 0.0, 0.33, 0.66, 1.0]
VLAG_RANGE = ["#315f9d", "#7893bc", "#b8c2d2", "#f3f2f1", "#d8b3b1", "#c27475", "#a9434d"]

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
SENTINEL_EVALS = ["IFEval", "MMLU", "MATH-500"]
FRONTIER_EVALS = [
    "FS-Olympiad accuracy",
    "FS-Research success",
    "FS-Research rubric",
]
BASE_EVALS = ["LitSearch-rerank", *FRONTIER_EVALS, *SENTINEL_EVALS]
ALL_ANALYSIS_EVALS = [*AGENTIC_EVALS, *BASE_EVALS]
COMPACT_EVAL_LABELS = {
    "ExpertQA": "ExpertQA",
    "LitSearch-open": "LitSearch-open",
    "SAGE-open": "SAGE-open",
    "SAGE-short": "SAGE-short",
    "DeepScholar-Bench": "DeepScholar",
}
FAMILY_ORDER = ["sci-lit", "frontier-science", "sentinel"]
FAMILY_LABELS = {
    "sci-lit": "Sci-lit",
    "frontier-science": "FrontierScience",
    "sentinel": "Sentinel",
}
FAMILY_COLOR = {
    "sci-lit": PRIMARY,
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
    "IFEval": "prompt_level_loose_acc:ifeval",
    "MMLU": "primary_score:average",
    "MATH-500": "accuracy:minerva_math_flex",
    "FS-Olympiad accuracy": "accuracy:frontierscience_judge",
    "FS-Research success": "success_rate:frontierscience_judge",
    "FS-Research rubric": "rubric_score:frontierscience_judge",
}
INVALID_NOTE_MARKERS = ("legacy stock tool template",)


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
        sub = df.filter(pl.col("family") == family)
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
                "status:N",
                "run_id:N",
                "reason:N",
            ],
        )
        # Status color never carries meaning alone: every cell states its value
        # or its state in text.
        labels = base.mark_text(fontSize=9, fontWeight=600).encode(
            text="cell_label:N",
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
            subtitle=[
                "Cell shows the score where one exists.",
                "N/A is an unsupported harness combination; — is genuinely not run.",
                "Suspect scores are excluded from numerical analysis.",
            ],
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
        title = "Agentic dev + base scores"
        subtitle = (
            "Bars are replica means; whiskers are replica min–max. Agentic panels use fixed-dev "
            "scores; rerank remains full/base."
        )
        facet_width, row_step = 235, 23
    elif family == "frontier-science":
        title = "FrontierScience scores"
        subtitle = (
            "Bars are replica means; whiskers are replica min–max. Olympiad accuracy uses 100 "
            "closed-form questions; Research success and rubric use the same 60 open-ended "
            "questions."
        )
        # Three equal, presentation-scale panels fill the slide instead of
        # leaving the FrontierScience charts clustered in its left half.
        facet_width, row_step = 315, 27
    else:
        title = "Sentinel scores"
        subtitle = "Bars are full-set replica means; whiskers are replica min–max."
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
            ranked.filter(pl.col("model").is_in(models)).select("model", "eval", "pct"),
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
        y=alt.Y("pct:Q", title="percentile rank within eval", scale=alt.Scale(domain=[0, 1])),
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
                subtitle=(
                    "Agentic dev, FrontierScience and sentinels; 1.0 is best-in-eval. "
                    "Gaps are unmeasured scores."
                ),
            ),
        )
    )


def chart_summary(
    sci_lit: pl.DataFrame,
    frontier: pl.DataFrame,
    sentinel: pl.DataFrame,
    n_sci_lit: int,
    pool: tuple[int, int],
) -> alt.HConcatChart:
    order = sci_lit["model"].to_list()
    main = (
        alt.Chart(sci_lit)
        .mark_bar(cornerRadiusEnd=4, color=PRIMARY)
        .encode(
            x=alt.X("mean_pct:Q", title="mean percentile rank", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y(
                "model:N", sort=order, title=None, axis=alt.Axis(labelLimit=175, labelColor=INK)
            ),
            tooltip=["model:N", "mean_pct:Q", "n_evals:Q"],
        )
    )
    # Coverage on every bar: a mean over 5 evals and a mean over 6 are not the
    # same quantity, and the chart should not pretend otherwise.
    coverage = (
        alt.Chart(sci_lit)
        .mark_text(align="left", dx=4, fontSize=9, color=INK_SECONDARY)
        .encode(
            x=alt.X("mean_pct:Q"),
            y=alt.Y("model:N", sort=order),
            text=alt.Text("label:N"),
        )
    )
    left = (main + coverage).properties(
        width=350,
        height=alt.Step(29),
        title=alt.Title(
            "Sci-lit composite",
            subtitle=f"mean percentile rank over {n_sci_lit} evals",
        ),
    )

    def companion_panel(
        data: pl.DataFrame, title: str, subtitle: str, color: str
    ) -> alt.LayerChart:
        bars = (
            alt.Chart(data)
            .mark_bar(cornerRadiusEnd=4, color=color)
            .encode(
                x=alt.X(
                    "display_pct:Q", title="mean percentile rank", scale=alt.Scale(domain=[0, 1])
                ),
                y=alt.Y("model:N", sort=order, title=None, axis=None),
                tooltip=["model:N", "mean_pct:Q", "n_evals:Q"],
            )
        )
        # Coverage labels make missing companion scores visible instead of
        # silently dropping a model from the panel.
        labels = (
            alt.Chart(data)
            .mark_text(align="left", dx=4, fontSize=9, color=INK)
            .encode(
                x=alt.X("display_pct:Q"),
                y=alt.Y("model:N", sort=order),
                text=alt.Text("label:N"),
            )
        )
        return (bars + labels).properties(
            width=245,
            height=alt.Step(29),
            title=alt.Title(title, subtitle=subtitle),
        )

    middle = companion_panel(
        frontier,
        "FrontierScience",
        "mean rank over three measures",
        SERIES[2],
    )
    right = companion_panel(
        sentinel,
        "Sentinels",
        "regression monitors, not the objective",
        SENTINEL,
    )
    return alt.hconcat(left, middle, right, spacing=24).properties(
        title=alt.Title(
            "Overall standing",
            subtitle=[
                "FrontierScience and sentinels are shown alongside, never folded into the "
                "sci-lit composite.",
                f"Ranks are taken over {pool[0]}-{pool[1]} models per eval, "
                "so small gaps between bars are not meaningful.",
            ],
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
        dev_runs="run_count",
    )
    full = scope_scores.filter(
        pl.col("eval").is_in(AGENTIC_EVALS)
        & (pl.col("scope") == "full")
        & pl.col("status").is_in(RANKABLE_STATUSES)
    ).select("model", "eval", full_score="score")
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
            x=alt.X("full_score:Q", title="trusted full-set score"),
            y=alt.Y("dev_score:Q", title="fixed-dev score"),
            order="diag_order:Q",
        )
    )
    points = (
        base.transform_filter(alt.datum.kind == "point")
        .mark_point(filled=True, color=PRIMARY, size=85)
        .encode(
            x=alt.X("full_score:Q", title="trusted full-set score"),
            y=alt.Y("dev_score:Q", title="fixed-dev score"),
            tooltip=["model:N", "eval:N", "full_score:Q", "dev_score:Q", "dev_runs:Q"],
        )
    )
    labels = (
        base.transform_filter(alt.datum.kind == "point")
        .mark_text(dx=6, align="left", fontSize=8, color=INK_SECONDARY)
        .encode(x="full_score:Q", y="dev_score:Q", text="model:N")
    )
    return (
        alt.layer(line, points, labels)
        .properties(width=205, height=175)
        .facet(alt.Facet("facet_label:N", title=None), columns=3)
        .resolve_scale(x="independent", y="independent")
        .properties(
            title=alt.Title(
                "Do fixed dev sets preserve full-set ordering?",
                subtitle=[
                    "Only model/eval pairs with trusted full and dev results are shown.",
                    "Dashed line is dev = full; rho is descriptive at these small n values.",
                ],
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

    def family(eval_name: str) -> str:
        if eval_name in FRONTIER_EVALS:
            return "frontier-science"
        if eval_name in SENTINEL_EVALS:
            return "sentinel"
        return "sci-lit"

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
                    "x_family_label": FAMILY_LABELS[family(x_eval)],
                    "y_family_label": FAMILY_LABELS[family(y_eval)],
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
    cumulative = np.cumsum(explained)
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
                    "family_label": FAMILY_LABELS[
                        "frontier-science"
                        if eval_name in FRONTIER_EVALS
                        else "sentinel"
                        if eval_name in SENTINEL_EVALS
                        else "sci-lit"
                    ],
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
            redundancy_rows.append(
                {
                    "x_eval": x_eval,
                    "y_eval": y_eval,
                    "pair_label": (
                        f"{COMPACT_EVAL_LABELS.get(x_eval, x_eval)} ↔ "
                        f"{COMPACT_EVAL_LABELS.get(y_eval, y_eval)}"
                    ),
                    "rho": float(rho),
                    "abs_rho": float(abs(rho)),
                    "rho_min": source["rho_min"],
                    "rho_max": source["rho_max"],
                    "abs_rho_min": source["abs_rho_min"],
                    "abs_rho_max": source["abs_rho_max"],
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


def chart_eval_pca(
    variance: pl.DataFrame,
    loadings: pl.DataFrame,
    redundancy: pl.DataFrame,
    diagnostics: pl.DataFrame,
    eval_order: list[str],
) -> alt.HConcatChart:
    retained = int(diagnostics["retained_components_80pct"][0])
    correction = float(diagnostics["projection_frobenius_delta"][0])
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
    scree_panel = (scree + cumulative).properties(
        width=260,
        height=330,
        title=alt.Title(
            "Variance by component", subtitle=f"first 8 shown · {retained} PCs reach 80%"
        ),
    )

    shown_pcs = pc_order[: min(4, len(pc_order))]
    loading_data = loadings.filter(pl.col("pc").is_in(shown_pcs)).with_columns(
        loading_label=pl.col("loading_correlation").round(2).cast(pl.String)
    )
    eval_labels = [CANONICAL_TO_DEV.get(name, name) for name in eval_order]
    loading_base = alt.Chart(loading_data).encode(
        x=alt.X("pc:N", sort=shown_pcs, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("eval_label:N", sort=eval_labels, title=None),
    )
    loading_cells = loading_base.mark_rect().encode(
        color=alt.Color(
            "loading_correlation:Q",
            title="eval–PC corr.",
            scale=alt.Scale(domain=VLAG_DOMAIN, range=VLAG_RANGE, clamp=True),
        ),
        tooltip=[
            "eval_label:N",
            "pc:N",
            alt.Tooltip("loading_correlation:Q", format=".3f"),
            alt.Tooltip("component_loading:Q", format=".3f"),
        ],
    )
    loading_text = loading_base.mark_text(fontSize=8).encode(
        text="loading_label:N",
        color=alt.condition(
            "abs(datum.loading_correlation) >= 0.58", alt.value("white"), alt.value(INK)
        ),
    )
    loading_panel = (loading_cells + loading_text).properties(
        width=alt.Step(55),
        height=alt.Step(27),
        title="Leading component loadings",
    )

    top_redundancy = redundancy.head(8).with_columns(
        rho_label=pl.col("rho").round(2).cast(pl.String),
    )
    pair_order = top_redundancy["pair_label"].to_list()
    redundancy_base = alt.Chart(top_redundancy).encode(
        x=alt.X("abs_rho:Q", title="|Spearman ρ|", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y(
            "pair_label:N",
            sort=pair_order,
            title=None,
            axis=alt.Axis(labelLimit=245),
        ),
    )
    redundancy_bars = redundancy_base.mark_bar(color=SERIES[4], cornerRadiusEnd=3).encode(
        tooltip=[
            "pair_label:N",
            alt.Tooltip("rho:Q", format=".3f"),
            alt.Tooltip("rho_min:Q", title="pairing min ρ", format=".3f"),
            alt.Tooltip("rho_max:Q", title="pairing max ρ", format=".3f"),
            alt.Tooltip("rho_variant_count:Q", title="replica pairings"),
            "n_models:Q",
            "n_pairs:Q",
        ]
    )
    redundancy_errors = redundancy_base.mark_rule(color=INK, strokeWidth=1.4).encode(
        x=alt.X("abs_rho_min:Q"),
        x2=alt.X2("abs_rho_max:Q"),
    )
    redundancy_error_min = redundancy_base.mark_tick(
        color=INK, orient="vertical", thickness=1.4, size=11
    ).encode(x=alt.X("abs_rho_min:Q"))
    redundancy_error_max = redundancy_base.mark_tick(
        color=INK, orient="vertical", thickness=1.4, size=11
    ).encode(x=alt.X("abs_rho_max:Q"))
    redundancy_labels = redundancy_base.mark_text(align="left", dx=5, fontSize=8).encode(
        x=alt.X("abs_rho_max:Q"), text="rho_label:N"
    )
    redundancy_panel = (
        redundancy_bars
        + redundancy_errors
        + redundancy_error_min
        + redundancy_error_max
        + redundancy_labels
    ).properties(
        width=300,
        height=alt.Step(34),
        title=alt.Title(
            "Most redundant eval pairs",
            subtitle="bar: pooled |ρ| · whisker: replica-pairing min–max",
        ),
    )

    return alt.hconcat(scree_panel, loading_panel, redundancy_panel, spacing=30).properties(
        title=alt.Title(
            "Evaluation covariance structure",
            subtitle=[
                "PCA uses the deterministic Cartesian replica-pair Spearman matrix.",
                "Score bars are replica means with min–max whiskers.",
                "Pairwise-complete matrix PSD-projected by eigenvalue clipping "
                f"(ΔF={correction:.3f}).",
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
            scale=alt.Scale(domain=VLAG_DOMAIN, range=VLAG_RANGE, clamp=True),
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
        color=alt.condition("abs(datum.rho) >= 0.58", alt.value("white"), alt.value(INK)),
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
    predictive = pairwise_replica_correlations(run_scores, AGENTIC_EVALS, BASE_EVALS)
    strongest = (
        predictive.filter(pl.col("strong"))
        .with_columns(abs_rho=pl.col("rho").abs())
        .sort("abs_rho", descending=True)
        .head(4)
    )
    strongest_label = "; ".join(
        f"{row['y_label']} → {row['x_label']} {row['rho']:+.2f}"
        for row in strongest.iter_rows(named=True)
    )
    left = _correlation_heatmap(
        matrix,
        matrix_labels,
        matrix_labels,
        "All-eval rank covariance",
        37,
        33,
        True,
        True,
    )
    right = _correlation_heatmap(
        predictive,
        [CANONICAL_TO_DEV[name] for name in AGENTIC_EVALS],
        BASE_EVALS,
        "Can base evals predict agentic scores?",
        76,
        56,
        False,
        False,
    )
    return alt.hconcat(left, right, spacing=36).properties(
        title=alt.Title(
            "Cross-eval rank covariance",
            subtitle=[
                "Every within-model run_i × run_j combination contributes one point.",
                "Cells show only rho; model and expanded-pair counts remain in tooltips.",
                "Expanded pairs are deterministic sensitivity points, not independent samples.",
                f"Strongest observed: {strongest_label}.",
            ],
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

    lines += ["", "## Sci-lit standing", "", "| model | mean pct rank | evals |", "|---|---|---|"]
    for row in sci_lit.iter_rows(named=True):
        lines.append(f"| {row['model']} | {row['mean_pct']:.3f} | {row['n_evals']} |")
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


SLIDE_NOTES = {
    "coverage": (
        "Where the sweep stands",
        "Agentic cells use fixed dev sets; FrontierScience, base and sentinel cells use full "
        "scores. Unsupported harness/model combinations remain blank by design.",
    ),
    "scores_sci-lit": (
        "Agentic dev scores",
        "Fixed-sample agentic scores are the primary comparison. Valid repeated runs are "
        "averaged, with min–max whiskers; LitSearch-rerank remains a full-set base eval.",
    ),
    "scores_sentinel": (
        "Sentinel scores",
        "Regression monitors for instruction-following, knowledge and reasoning. Read these for "
        "drift, not for standing; whiskers span the replica minimum and maximum.",
    ),
    "scores_frontier-science": (
        "FrontierScience",
        "Olympiad accuracy measures closed-form problem solving. Research success is the hard "
        "binary outcome; rubric score retains partial credit and is the more sensitive measure. "
        "Whiskers span replica minima and maxima where repeats exist.",
    ),
    "profile": (
        "Cross-eval profile",
        "Percentile rank across agentic dev, FrontierScience and sentinel evals. The per-eval "
        "n is on the axis; broken lines mark missing scores.",
    ),
    "dev_vs_full": (
        "Do the dev sets preserve full-set results?",
        "Trusted paired results only. The fixed subsets are useful for iteration, but deviations "
        "from the diagonal show why dev scores should not be mixed with full-benchmark history.",
    ),
    "covariance": (
        "Which base evals predict agentic performance?",
        "Rank-standardized covariance over every within-model Cartesian replica pairing. Pair "
        "counts are sensitivity points rather than independent observations; use model count to "
        "judge coverage.",
    ),
    "eval_pca": (
        "What structure do the evaluations share?",
        "PCA summarizes the pairwise-expanded rank-covariance matrix. Similar loadings and high "
        "absolute pair correlations flag possible redundancy; redundancy whiskers span the "
        "replica-pairing range. Signs of PCA components are arbitrary, and all relationships "
        "remain descriptive.",
    ),
    "summary": (
        "Overall standing",
        "FrontierScience and sentinels provide context beside the sci-lit composite; neither is "
        "folded into the primary standing.",
    ),
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
            ("supported agentic dev", f"{scored_agentic}/{supported_agentic.height}"),
            ("cells scored", f"{filled}/{total}"),
            ("coverage", f"{filled / total:.0%}"),
            ("suspect", counts["suspect"]),
            ("unsupported", counts["unsupported"]),
            ("not run", counts["not-run"]),
        ]
    )

    slides = [
        f"""<section class="slide">
          <p class="eyebrow">Model evaluation · science and agentic benchmarks</p>
          <h1>Science eval sweep</h1>
          <p class="sub">Complete first-pass coverage for every supported agentic dev profile.
             Unsupported tool-harness combinations remain intentionally blank.</p>
          <div class="tiles">{tiles}</div>
          <p class="foot">Generated by <code>scripts/analysis/eval_summary.py</code>.
             Re-run it after refreshing <code>data/results.csv</code>.</p>
        </section>"""
    ]

    for name in generated:
        title, note = SLIDE_NOTES[name]
        png = out_dir / f"{name}.png"
        if not png.exists():
            continue
        slides.append(
            f"""<section class="slide">
              <p class="eyebrow">Science eval sweep · results</p>
              <div class="figure"><img src="{_data_uri(png)}" alt="{escape(title)}"></div>
              <p class="reading-note"><span>Reading note</span>{escape(note)}</p>
            </section>"""
        )

    standing = _table(
        ["model", "mean pct rank", "evals"],
        [[r["model"], f"{r['mean_pct']:.3f}", r["n_evals"]] for r in sci_lit.iter_rows(named=True)],
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
        '<p class="sub">Below the coverage threshold, not ranked: '
        f"{escape(', '.join(excluded))}.</p>"
        if excluded
        else ""
    )
    slides.append(
        f"""<section class="slide">
          <h2>Detail</h2>
          <h3>Sci-lit standing</h3>{standing}{excluded_note}
          <h3>Suspect scores — excluded from ranking, all unverified</h3>{suspect}
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
  .foot {{ color: var(--muted); font-size: 13px; margin-top: auto; padding-top: 28px; }}
  code {{ font-size: 0.92em; background: var(--plane); padding: 1px 5px; border-radius: 4px; }}
  .figure {{ flex: 1; display: flex; align-items: center; justify-content: center;
             overflow: auto; padding: 8px 0 2px; }}
  .figure img {{ max-width: 100%; max-height: calc(100vh - 260px); width: auto; height: auto;
                 display: block; margin: auto; }}
  .reading-note {{ margin: 12px 0 0; padding: 12px 15px; color: var(--ink2);
                   background: rgba(255, 255, 255, 0.72); border: 1px solid var(--rule);
                   border-radius: 11px; font-size: 13px; }}
  .reading-note span {{ color: var(--ink); font-size: 10px; font-weight: 700;
                        letter-spacing: 0.08em; text-transform: uppercase; margin-right: 10px; }}
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
    ranked = add_ranks(df)

    n_sci_lit = len(meta.sci_lit_evals())
    n_frontier = meta.evals.filter(pl.col("family") == "frontier-science").height
    n_sentinel = meta.evals.filter(pl.col("family") == "sentinel").height
    sci_lit_all = composite(ranked, "sci-lit")
    sci_lit = sci_lit_all.filter(pl.col("n_evals") >= args.min_coverage).with_columns(
        label=pl.format("n={}/{}", pl.col("n_evals"), pl.lit(n_sci_lit))
    )
    excluded = sci_lit_all.filter(pl.col("n_evals") < args.min_coverage)["model"].to_list()
    ranked_models = sci_lit["model"].to_list()
    model_grid = pl.DataFrame({"model": ranked_models})

    def companion_scores(family: str, n_evals: int) -> pl.DataFrame:
        return (
            model_grid.join(composite(ranked, family), on="model", how="left")
            .with_columns(
                n_evals=pl.col("n_evals").fill_null(0),
                display_pct=pl.col("mean_pct").fill_null(0.0),
            )
            .with_columns(label=pl.format("n={}/{}", pl.col("n_evals"), pl.lit(n_evals)))
        )

    frontier = companion_scores("frontier-science", n_frontier)
    sentinel = companion_scores("sentinel", n_sentinel)

    print(f"writing to {args.out}")
    generated = [
        "coverage",
        "scores_sci-lit",
        "scores_frontier-science",
        "scores_sentinel",
    ]
    save(chart_coverage(df, meta), args.out, "coverage", args.formats)
    save(chart_scores(df, meta, "sci-lit"), args.out, "scores_sci-lit", args.formats)
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
    pairwise = pairwise_replica_correlations(run_scores, ALL_ANALYSIS_EVALS, ALL_ANALYSIS_EVALS)
    pairwise.write_csv(args.out / "pairwise_replica_correlations.csv")
    print(f"  {args.out / 'pairwise_replica_correlations.csv'}")
    base_agentic = pairwise_replica_correlations(run_scores, AGENTIC_EVALS, BASE_EVALS)
    base_agentic.write_csv(args.out / "base_agentic_correlations.csv")
    print(f"  {args.out / 'base_agentic_correlations.csv'}")
    pca_variance, pca_loadings, redundancy, pca_diagnostics = analyze_eval_pca(
        pairwise, ALL_ANALYSIS_EVALS
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

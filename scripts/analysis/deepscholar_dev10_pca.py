#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair~=6.2",
#     "numpy~=2.4",
#     "polars~=1.43",
#     "scikit-learn~=1.8",
#     "scipy~=1.17",
#     "vl-convert-python~=1.9",
# ]
# ///
"""PCA and redundancy analysis of DeepScholar-Bench dev10 sub-metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import altair as alt
import numpy as np
import polars as pl
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA

HERE = Path(__file__).parent
DEFAULT_INPUT = HERE / "data" / "results.csv"
DEFAULT_OUTPUT_DIR = HERE / "out"

METRIC_LABELS = {
    "cite_p_fixed": "Citation precision",
    "claim_coverage_fixed": "Claim coverage",
    "coverage_relevance_rate_fixed": "Coverage relevance",
    "document_importance_fixed": "Document importance",
    "nugget_coverage_fixed": "Nugget coverage",
    "organization_fixed": "Organization",
    "reference_coverage_fixed": "Reference coverage",
}

MODEL_LABELS = {
    "OLMo-3 7B Instruct": "OLMo Instruct",
    "Qwen3.5 9B Instruct": "Qwen 9B",
    "Gemma4 26B-A4B": "Gemma 26B",
    "GPT-OSS-20b": "GPT-OSS",
    "Qwen3.5-35B-A3B": "Qwen 35B",
    "Nemotron 3 Nano 30B-A3B": "Nemotron",
}

BLUE = "#2a78d6"
ORANGE = "#e56b2f"
GREEN = "#1b9e77"
INK = "#171717"
MUTED = "#66635e"
GRID = "#deddd7"


def load_matrix(path: Path) -> tuple[pl.DataFrame, list[str], pl.DataFrame]:
    results = (
        pl.read_csv(path, infer_schema_length=0)
        .with_columns(
            pl.col("Score").cast(pl.Float64, strict=False),
            pl.col("Metric").fill_null(""),
        )
        .with_columns(metric_base=pl.col("Metric").str.split(":").list.first())
    )

    constituents = results.filter(
        (pl.col("Eval Name") == "DeepScholar-Bench-dev10")
        & pl.col("metric_base").is_in(METRIC_LABELS)
        & pl.col("Score").is_not_null()
    ).select(
        pl.col("Model Name").alias("model"),
        pl.col("metric_base").alias("metric"),
        pl.col("Score").alias("score"),
    )
    duplicate_cells = constituents.group_by("model", "metric").len().filter(pl.col("len") != 1)
    if duplicate_cells.height:
        raise ValueError(f"Expected one score per model/metric: {duplicate_cells}")

    metrics = list(METRIC_LABELS)
    wide = constituents.pivot(index="model", on="metric", values="score").select("model", *metrics)
    incomplete = wide.filter(pl.any_horizontal(pl.col(metrics).is_null()))
    if incomplete.height:
        raise ValueError(f"Incomplete model rows: {incomplete['model'].to_list()}")

    aggregate = (
        results.filter(
            (pl.col("Eval Name") == "DeepScholar-Bench-dev10")
            & (pl.col("metric_base") == "geomean_fixed")
            & pl.col("Score").is_not_null()
        )
        .select(
            pl.col("Model Name").alias("model"),
            pl.col("Score").alias("geomean_fixed"),
        )
        .unique()
    )
    if aggregate.group_by("model").len().filter(pl.col("len") != 1).height:
        raise ValueError("Expected one geomean_fixed score per model")

    wide = wide.join(aggregate, on="model", how="inner", validate="1:1").sort("model")
    if wide.height < 3:
        raise ValueError(f"Need at least three complete models; found {wide.height}")
    return wide, metrics, constituents


def standardize_with_polars(wide: pl.DataFrame, metrics: list[str]) -> pl.DataFrame:
    normalized = wide.select(
        "model",
        *[
            ((pl.col(metric) - pl.col(metric).mean()) / pl.col(metric).std(ddof=0)).alias(metric)
            for metric in metrics
        ],
    )
    checks = normalized.select(
        *[pl.col(metric).mean().abs().alias(f"{metric}_mean") for metric in metrics],
        *[
            (pl.col(metric).std(ddof=0) - 1).abs().alias(f"{metric}_std_error")
            for metric in metrics
        ],
    )
    if max(checks.row(0)) > 1e-10:
        raise ValueError(f"Column standardization check failed: {checks}")
    return normalized


def orient_components(scores: np.ndarray, components: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Choose deterministic PCA signs: largest absolute feature loading is positive."""
    scores = scores.copy()
    components = components.copy()
    for pc in range(components.shape[0]):
        anchor = int(np.argmax(np.abs(components[pc])))
        if components[pc, anchor] < 0:
            components[pc] *= -1
            scores[:, pc] *= -1
    return scores, components


def analyze(wide: pl.DataFrame, normalized: pl.DataFrame, metrics: list[str]) -> dict[str, object]:
    x = np.asarray(normalized.select(metrics).rows(), dtype=float)
    n_components = min(wide.height - 1, len(metrics))
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(x)
    scores, components = orient_components(scores, pca.components_)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    retained = int(np.searchsorted(cumulative, 0.8) + 1)

    feature_corr = np.corrcoef(x, rowvar=False)
    loading_corr = np.empty((len(metrics), n_components))
    for feature in range(len(metrics)):
        for pc in range(n_components):
            loading_corr[feature, pc] = np.corrcoef(x[:, feature], scores[:, pc])[0, 1]

    geomean = np.asarray(wide["geomean_fixed"].to_list(), dtype=float)
    geomean_corr = np.asarray([np.corrcoef(x[:, i], geomean)[0, 1] for i in range(len(metrics))])
    raw = np.asarray(wide.select(metrics).rows(), dtype=float)
    raw_range = np.ptp(raw, axis=0)
    raw_std = np.std(raw, axis=0, ddof=0)

    redundancy = np.abs(feature_corr).copy()
    np.fill_diagonal(redundancy, -np.inf)
    closest_index = np.argmax(redundancy, axis=1)
    max_abs_corr = redundancy[np.arange(len(metrics)), closest_index]
    closest_corr = feature_corr[np.arange(len(metrics)), closest_index]
    communality = np.sum(loading_corr[:, :retained] ** 2, axis=1)
    retained_contribution = np.sum(
        (components[:retained, :].T ** 2) * explained[:retained], axis=1
    ) / np.sum(explained[:retained])

    variance = pl.DataFrame(
        {
            "pc": [f"PC{i + 1}" for i in range(n_components)],
            "explained_variance": explained,
            "cumulative_variance": cumulative,
        }
    )
    loadings = pl.DataFrame(
        {
            "metric": metrics,
            "metric_label": [METRIC_LABELS[m] for m in metrics],
            **{f"PC{i + 1}_loading": components[i, :] for i in range(n_components)},
            **{f"PC{i + 1}_correlation": loading_corr[:, i] for i in range(n_components)},
        }
    )
    diagnostics = pl.DataFrame(
        {
            "metric": metrics,
            "metric_label": [METRIC_LABELS[m] for m in metrics],
            "raw_range": raw_range,
            "raw_std": raw_std,
            "closest_metric": [METRIC_LABELS[metrics[i]] for i in closest_index],
            "closest_corr": closest_corr,
            "max_abs_corr": max_abs_corr,
            "corr_with_geomean": geomean_corr,
            "retained_communality": communality,
            "retained_contribution": retained_contribution,
        }
    ).sort("retained_contribution", descending=True)

    return {
        "x": x,
        "scores": scores,
        "components": components,
        "explained": explained,
        "cumulative": cumulative,
        "retained": retained,
        "feature_corr": feature_corr,
        "loading_corr": loading_corr,
        "variance": variance,
        "loadings": loadings,
        "diagnostics": diagnostics,
    }


def scree_chart(variance: pl.DataFrame, retained: int) -> alt.LayerChart:
    records = variance.to_dicts()
    bars = (
        alt.Chart(alt.Data(values=records))
        .mark_bar(color=BLUE, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("pc:N", sort=None, title=None),
            y=alt.Y("explained_variance:Q", title="Share of variance", axis=alt.Axis(format=".0%")),
            tooltip=["pc:N", alt.Tooltip("explained_variance:Q", format=".1%")],
        )
    )
    labels = (
        alt.Chart(alt.Data(values=records))
        .mark_text(dy=-8, fontSize=11)
        .encode(
            x=alt.X("pc:N", sort=None),
            y="explained_variance:Q",
            text=alt.Text("explained_variance:Q", format=".1%"),
        )
    )
    cumulative = (
        alt.Chart(alt.Data(values=records))
        .mark_line(color=ORANGE, point=True, strokeWidth=2)
        .encode(
            x=alt.X("pc:N", sort=None),
            y=alt.Y("cumulative_variance:Q", axis=alt.Axis(format=".0%")),
            tooltip=["pc:N", alt.Tooltip("cumulative_variance:Q", format=".1%")],
        )
    )
    threshold = (
        alt.Chart(alt.Data(values=[{"y": 0.8}]))
        .mark_rule(color=MUTED, strokeDash=[4, 4])
        .encode(y="y:Q")
    )
    return (bars + labels + cumulative + threshold).properties(
        width=390,
        height=270,
        title=alt.Title(
            "How many dimensions matter?",
            subtitle=f"Bars: per-PC variance · orange: cumulative · {retained} PCs reach 80%",
        ),
    )


def biplot_chart(
    wide: pl.DataFrame,
    metrics: list[str],
    scores: np.ndarray,
    loading_corr: np.ndarray,
    explained: np.ndarray,
) -> alt.LayerChart:
    scale = 0.83 / np.max(np.abs(scores[:, :2]))
    model_rows = []
    model_offsets = {
        "GPT-OSS-20b": (0.02, 0.04),
        "Nemotron 3 Nano 30B-A3B": (0.02, -0.04),
    }
    for index, model in enumerate(wide["model"].to_list()):
        dx, dy = model_offsets.get(model, (0.02, 0.02))
        model_rows.append(
            {
                "model": model,
                "label": MODEL_LABELS.get(model, model),
                "pc1": float(scores[index, 0] * scale),
                "pc2": float(scores[index, 1] * scale),
                "label_x": float(scores[index, 0] * scale + dx),
                "label_y": float(scores[index, 1] * scale + dy),
            }
        )
    metric_offsets = {
        "cite_p_fixed": (-0.03, 0.03),
        "claim_coverage_fixed": (-0.03, -0.07),
        "coverage_relevance_rate_fixed": (-0.07, 0.02),
        "reference_coverage_fixed": (0.00, 0.08),
        "nugget_coverage_fixed": (-0.04, -0.02),
        "document_importance_fixed": (-0.02, -0.02),
    }
    metric_rows = []
    for i, metric in enumerate(metrics):
        dx, dy = metric_offsets.get(metric, (0.0, 0.0))
        metric_rows.append(
            {
                "metric": METRIC_LABELS[metric],
                "x0": 0.0,
                "y0": 0.0,
                "pc1": float(loading_corr[i, 0]),
                "pc2": float(loading_corr[i, 1]),
                "label_x": float(loading_corr[i, 0] * 1.08 + dx),
                "label_y": float(loading_corr[i, 1] * 1.08 + dy),
            }
        )
    circle_rows = [
        {"angle": float(angle), "x": float(np.cos(angle)), "y": float(np.sin(angle))}
        for angle in np.linspace(0, 2 * np.pi, 121)
    ]
    domain = [-1.17, 1.17]
    circle = (
        alt.Chart(alt.Data(values=circle_rows))
        .mark_line(color="#b9b7af", strokeDash=[3, 3])
        .encode(
            x=alt.X("x:Q", title=f"PC1 ({explained[0]:.1%})", scale=alt.Scale(domain=domain)),
            y=alt.Y("y:Q", title=f"PC2 ({explained[1]:.1%})", scale=alt.Scale(domain=domain)),
            order="angle:Q",
        )
    )
    axes = (
        alt.Chart(
            alt.Data(
                values=[
                    {"x": -1.1, "x2": 1.1, "y": 0.0, "y2": 0.0},
                    {"x": 0.0, "x2": 0.0, "y": -1.1, "y2": 1.1},
                ]
            )
        )
        .mark_rule(color="#d2d0c8")
        .encode(x=alt.X("x:Q", axis=None), x2="x2:Q", y=alt.Y("y:Q", axis=None), y2="y2:Q")
    )
    arrows = (
        alt.Chart(alt.Data(values=metric_rows))
        .mark_rule(color=ORANGE, strokeWidth=2)
        .encode(
            x=alt.X("x0:Q", axis=None),
            y=alt.Y("y0:Q", axis=None),
            x2="pc1:Q",
            y2="pc2:Q",
            tooltip=[
                "metric:N",
                alt.Tooltip("pc1:Q", format=".3f"),
                alt.Tooltip("pc2:Q", format=".3f"),
            ],
        )
    )
    metric_points = (
        alt.Chart(alt.Data(values=metric_rows))
        .mark_point(color=ORANGE, filled=True, size=45)
        .encode(x=alt.X("pc1:Q", axis=None), y=alt.Y("pc2:Q", axis=None))
    )
    metric_labels = (
        alt.Chart(alt.Data(values=metric_rows))
        .mark_text(color="#a94417", fontSize=10, fontWeight=600)
        .encode(
            x=alt.X("label_x:Q", axis=None),
            y=alt.Y("label_y:Q", axis=None),
            text="metric:N",
        )
    )
    model_points = (
        alt.Chart(alt.Data(values=model_rows))
        .mark_circle(color=BLUE, size=85, opacity=0.9)
        .encode(x=alt.X("pc1:Q", axis=None), y=alt.Y("pc2:Q", axis=None), tooltip=["model:N"])
    )
    model_labels = (
        alt.Chart(alt.Data(values=model_rows))
        .mark_text(color=INK, align="left", fontSize=10)
        .encode(
            x=alt.X("label_x:Q", axis=None),
            y=alt.Y("label_y:Q", axis=None),
            text="label:N",
        )
    )
    return (
        circle + axes + arrows + metric_points + metric_labels + model_points + model_labels
    ).properties(
        width=480,
        height=430,
        title=alt.Title(
            "PC1–PC2 biplot",
            subtitle=[
                f"PC1 {explained[0]:.1%} · PC2 {explained[1]:.1%} · "
                "orange vectors are metric correlations",
                "Model positions are uniformly rescaled to share the correlation circle",
            ],
        ),
    )


def correlation_chart(feature_corr: np.ndarray, metrics: list[str]) -> alt.LayerChart:
    distance = np.clip(1 - np.abs(feature_corr), 0, 2)
    order_index = leaves_list(linkage(squareform(distance, checks=False), method="average"))
    order = [METRIC_LABELS[metrics[i]] for i in order_index]
    records = []
    for i, metric_y in enumerate(metrics):
        for j, metric_x in enumerate(metrics):
            records.append(
                {
                    "metric_x": METRIC_LABELS[metric_x],
                    "metric_y": METRIC_LABELS[metric_y],
                    "correlation": float(feature_corr[i, j]),
                }
            )
    base = alt.Chart(alt.Data(values=records)).encode(
        x=alt.X(
            "metric_x:N", sort=order, title=None, axis=alt.Axis(labelAngle=-38, labelLimit=130)
        ),
        y=alt.Y("metric_y:N", sort=order, title=None, axis=alt.Axis(labelLimit=135)),
    )
    cells = base.mark_rect(stroke="white", strokeWidth=1).encode(
        color=alt.Color(
            "correlation:Q",
            scale=alt.Scale(domain=[-1, 0, 1], range=["#b64b3c", "#f4f3ef", "#2a78d6"]),
            legend=alt.Legend(title="Pearson r", orient="bottom"),
        ),
        tooltip=["metric_x:N", "metric_y:N", alt.Tooltip("correlation:Q", format=".3f")],
    )
    labels = base.mark_text(fontSize=9).encode(
        text=alt.Text("correlation:Q", format=".2f"),
        color=alt.condition("abs(datum.correlation) > 0.55", alt.value("white"), alt.value(INK)),
    )
    return (cells + labels).properties(
        width=440,
        height=410,
        title=alt.Title(
            "Which metrics move together?",
            subtitle=(
                "Clustered by absolute Pearson correlation; n=6 models, so treat as descriptive"
            ),
        ),
    )


def attention_chart(diagnostics: pl.DataFrame, retained: int) -> alt.LayerChart:
    rows = diagnostics.to_dicts()
    label_offsets = {
        "Claim coverage": 0.012,
        "Citation precision": -0.012,
        "Coverage relevance": 0.009,
        "Reference coverage": -0.005,
    }
    for row in rows:
        row["label_y"] = row["raw_range"] + label_offsets.get(row["metric_label"], 0.0)
    median_range = float(diagnostics["raw_range"].median())
    base = alt.Chart(alt.Data(values=rows)).encode(
        x=alt.X(
            "max_abs_corr:Q",
            title="Redundancy: strongest |r| with another metric",
            scale=alt.Scale(domain=[0, 1.03]),
            axis=alt.Axis(format=".1f"),
        ),
        y=alt.Y("raw_range:Q", title="Observed model spread (max − min)"),
    )
    guides = alt.Chart(alt.Data(values=[{"x": 0.8, "y": median_range}])).mark_rule(
        color="#aaa79f", strokeDash=[4, 4]
    ).encode(x="x:Q") + alt.Chart(alt.Data(values=[{"x": 0.8, "y": median_range}])).mark_rule(
        color="#aaa79f", strokeDash=[4, 4]
    ).encode(y="y:Q")
    points = base.mark_circle(opacity=0.9, stroke="white", strokeWidth=1).encode(
        size=alt.Size(
            "retained_contribution:Q",
            title=f"Contribution to first {retained} PCs",
            scale=alt.Scale(range=[120, 900]),
        ),
        color=alt.Color(
            "corr_with_geomean:Q",
            title="r with geomean",
            scale=alt.Scale(domain=[-1, 0, 1], range=["#b64b3c", "#e8e7e1", BLUE]),
        ),
        tooltip=[
            "metric_label:N",
            alt.Tooltip("raw_range:Q", format=".3f"),
            alt.Tooltip("max_abs_corr:Q", title="Max |r|", format=".3f"),
            "closest_metric:N",
            alt.Tooltip("closest_corr:Q", format=".3f"),
            alt.Tooltip("corr_with_geomean:Q", format=".3f"),
            alt.Tooltip("retained_contribution:Q", format=".1%"),
            alt.Tooltip("retained_communality:Q", format=".1%"),
        ],
    )
    label_base = alt.Chart(alt.Data(values=rows)).encode(
        x=alt.X("max_abs_corr:Q", scale=alt.Scale(domain=[0, 1.03]), axis=None),
        y=alt.Y("label_y:Q", axis=None),
        text="metric_label:N",
    )
    labels_left = label_base.transform_filter("datum.max_abs_corr <= 0.85").mark_text(
        align="left", dx=8, fontSize=10, color=INK
    )
    labels_right = label_base.transform_filter("datum.max_abs_corr > 0.85").mark_text(
        align="right", dx=-8, fontSize=10, color=INK
    )
    return (guides + points + labels_left + labels_right).properties(
        width=470,
        height=330,
        title=alt.Title(
            "Attention map: separation versus redundancy",
            subtitle=[
                "Upper-left is attractive: separates models without duplicating another metric",
                f"Size: contribution to first {retained} PCs · "
                "color: correlation with derived geomean",
            ],
        ),
    )


def style(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    return (
        chart.configure(background="#fcfcfb")
        .configure_view(stroke=None)
        .configure_axis(
            gridColor=GRID,
            domainColor="#b5b3aa",
            labelColor=MUTED,
            titleColor=MUTED,
            labelFontSize=10,
            titleFontSize=11,
        )
        .configure_title(
            anchor="start",
            color=INK,
            fontSize=15,
            subtitleColor=MUTED,
            subtitleFontSize=11,
        )
    )


def save_outputs(
    output_dir: Path,
    variance: pl.DataFrame,
    loadings: pl.DataFrame,
    diagnostics: pl.DataFrame,
    charts: dict[str, alt.TopLevelMixin],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    variance.write_csv(output_dir / "deepscholar_dev10_pca_variance.csv")
    loadings.write_csv(output_dir / "deepscholar_dev10_pca_loadings.csv")
    diagnostics.write_csv(output_dir / "deepscholar_dev10_metric_diagnostics.csv")

    for name, chart in charts.items():
        styled = style(chart)
        styled.save(output_dir / f"deepscholar_dev10_{name}.html")
        styled.save(output_dir / f"deepscholar_dev10_{name}.png", scale_factor=2)

    dashboard = alt.vconcat(
        alt.hconcat(charts["scree"], charts["attention"], spacing=45),
        alt.hconcat(charts["biplot"], charts["correlations"], spacing=45),
        spacing=55,
        title=alt.Title(
            "DeepScholar-Bench dev10 sub-metric structure",
            subtitle="Six models · seven constituent metrics · columns standardized before PCA",
        ),
    )
    style(dashboard).save(output_dir / "deepscholar_dev10_pca_dashboard.html")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wide, metrics, _ = load_matrix(args.input)
    normalized = standardize_with_polars(wide, metrics)
    result = analyze(wide, normalized, metrics)

    variance = result["variance"]
    loadings = result["loadings"]
    diagnostics = result["diagnostics"]
    retained = result["retained"]
    assert isinstance(variance, pl.DataFrame)
    assert isinstance(loadings, pl.DataFrame)
    assert isinstance(diagnostics, pl.DataFrame)
    assert isinstance(retained, int)

    charts = {
        "scree": scree_chart(variance, retained),
        "biplot": biplot_chart(
            wide,
            metrics,
            result["scores"],
            result["loading_corr"],
            result["explained"],
        ),
        "correlations": correlation_chart(result["feature_corr"], metrics),
        "attention": attention_chart(diagnostics, retained),
    }
    save_outputs(args.output_dir, variance, loadings, diagnostics, charts)

    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=220):
        print("\nExplained variance")
        print(variance)
        print("\nMetric diagnostics")
        print(diagnostics)
        print("\nPC loadings and metric-PC correlations")
        print(loadings)
    print(f"\nRetained PCs for >=80% variance: {retained}")
    print(f"Wrote analysis to {args.output_dir}")


if __name__ == "__main__":
    main()

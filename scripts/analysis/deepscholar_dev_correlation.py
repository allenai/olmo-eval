#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair~=6.2",
#     "polars~=1.43",
#     "vl-convert-python~=1.9",
# ]
# ///
"""Compare matched DeepScholar-Bench full and dev10 geomean_fixed scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import altair as alt
import polars as pl

HERE = Path(__file__).parent
DEFAULT_INPUT = HERE / "data" / "results.csv"
DEFAULT_OUTPUT = HERE / "out" / "deepscholar_dev10_correlation.html"


def load_pairs(path: Path) -> pl.DataFrame:
    results = pl.read_csv(path, infer_schema_length=0).with_columns(
        pl.col("Score").cast(pl.Float64, strict=False),
        pl.col("Metric").fill_null(""),
    )
    metric_base = pl.col("Metric").str.split(":").list.first()

    full = (
        results.filter(
            (pl.col("Eval Name") == "DeepScholar-Bench")
            & pl.col("Score").is_not_null()
            & metric_base.is_in(["", "geomean_fixed"])
        )
        .select(
            pl.col("Model Name").alias("model"),
            pl.col("Score").alias("full_score"),
            pl.col("Beaker Run ID").alias("full_run_id"),
        )
        .unique()
    )
    dev10 = (
        results.filter(
            (pl.col("Eval Name") == "DeepScholar-Bench-dev10")
            & pl.col("Score").is_not_null()
            & (metric_base == "geomean_fixed")
        )
        .select(
            pl.col("Model Name").alias("model"),
            pl.col("Score").alias("dev10_score"),
            pl.col("Beaker Run ID").alias("dev10_run_id"),
        )
        .unique()
    )

    for scope, frame in (("full", full), ("dev10", dev10)):
        duplicates = frame.group_by("model").len().filter(pl.col("len") > 1)
        if duplicates.height:
            raise ValueError(f"Multiple distinct {scope} scores found: {duplicates}")

    return full.join(dev10, on="model", how="inner", validate="1:1").sort("full_score")


def correlations(pairs: pl.DataFrame) -> tuple[float, float]:
    if pairs.height < 2:
        raise ValueError(f"Need at least two matched pairs; found {pairs.height}")
    values = pairs.select(
        pl.corr("full_score", "dev10_score").alias("pearson"),
        pl.corr("full_score", "dev10_score", method="spearman").alias("spearman"),
    ).row(0)
    return float(values[0]), float(values[1])


def make_chart(pairs: pl.DataFrame, pearson: float, spearman: float) -> alt.LayerChart:
    short_names = {
        "OLMo-3 7B Instruct": "OLMo Instruct",
        "Qwen3.5 9B Instruct": "Qwen 9B",
        "Gemma4 26B-A4B": "Gemma 26B",
        "GPT-OSS-20b": "GPT-OSS",
        "Qwen3.5-35B-A3B": "Qwen 35B",
        "Nemotron 3 Nano 30B-A3B": "Nemotron",
    }
    label_offsets = {
        "GPT-OSS-20b": 0.004,
        "Nemotron 3 Nano 30B-A3B": -0.004,
        "Qwen3.5-35B-A3B": -0.003,
        "Gemma4 26B-A4B": 0.003,
    }
    records = pairs.to_dicts()
    for row in records:
        row["label"] = short_names.get(row["model"], row["model"])
        row["label_y"] = row["dev10_score"] + label_offsets.get(row["model"], 0.0)

    all_scores = pairs["full_score"].to_list() + pairs["dev10_score"].to_list()
    domain_min = max(0.0, min(all_scores) - 0.012)
    domain_max = max(all_scores) + 0.012
    shared_scale = alt.Scale(domain=[domain_min, domain_max], zero=False)

    base = alt.Chart(alt.Data(values=records))
    points = base.mark_circle(size=105, color="#2a78d6", opacity=0.9).encode(
        x=alt.X(
            "full_score:Q",
            title="Full benchmark geomean_fixed",
            scale=shared_scale,
            axis=alt.Axis(format=".3f"),
        ),
        y=alt.Y(
            "dev10_score:Q",
            title="Dev10 geomean_fixed",
            scale=shared_scale,
            axis=alt.Axis(format=".3f"),
        ),
        tooltip=[
            alt.Tooltip("model:N", title="Model"),
            alt.Tooltip("full_score:Q", title="Full", format=".4f"),
            alt.Tooltip("dev10_score:Q", title="Dev10", format=".4f"),
            alt.Tooltip("full_run_id:N", title="Full run"),
            alt.Tooltip("dev10_run_id:N", title="Dev10 run"),
        ],
    )
    labels = base.mark_text(align="left", dx=7, fontSize=11, color="#222222").encode(
        x=alt.X("full_score:Q", scale=shared_scale),
        y=alt.Y("label_y:Q", scale=shared_scale),
        text="label:N",
    )
    regression = (
        base.transform_regression("full_score", "dev10_score")
        .mark_line(color="#e56b2f", strokeWidth=2.5)
        .encode(
            x=alt.X("full_score:Q", scale=shared_scale),
            y=alt.Y("dev10_score:Q", scale=shared_scale),
        )
    )
    identity = (
        alt.Chart(
            alt.Data(
                values=[
                    {"full_score": domain_min, "dev10_score": domain_min},
                    {"full_score": domain_max, "dev10_score": domain_max},
                ]
            )
        )
        .mark_line(color="#898781", strokeDash=[5, 5], strokeWidth=1.5)
        .encode(
            x=alt.X("full_score:Q", scale=shared_scale),
            y=alt.Y("dev10_score:Q", scale=shared_scale),
        )
    )

    return (
        (identity + regression + points + labels)
        .properties(
            width=620,
            height=430,
            title=alt.Title(
                "DeepScholar-Bench: dev10 versus full",
                subtitle=[
                    f"Matched models n={pairs.height} · Pearson r={pearson:.3f} · "
                    f"Spearman ρ={spearman:.3f}",
                    "Orange: linear fit · dashed grey: identical dev/full score",
                ],
            ),
        )
        .configure_view(stroke=None)
        .configure_axis(
            gridColor="#e1e0d9",
            domainColor="#b5b3aa",
            labelColor="#52514e",
            titleColor="#52514e",
        )
        .configure_title(anchor="start", color="#0b0b0b", fontSize=16, subtitleFontSize=12)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = load_pairs(args.input)
    pearson, spearman = correlations(pairs)
    chart = make_chart(pairs, pearson, spearman)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    chart.save(args.output)
    png_path = args.output.with_suffix(".png")
    chart.save(png_path, scale_factor=2)

    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=140):
        print(pairs.select("model", "full_score", "dev10_score"))
    print(f"Pearson r:  {pearson:.6f}")
    print(f"Spearman ρ: {spearman:.6f}")
    print(f"Wrote {args.output}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()

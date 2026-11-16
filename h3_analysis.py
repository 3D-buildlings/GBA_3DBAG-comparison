#!/usr/bin/env python3
"""H3-based density analysis of the GBA vs 3DBAG comparison outputs.

Run after run_pipeline.py. It reads:
  <out>/GBA_vs_3DBAG.geojson
  <out>/GBA_vs_3DBAG_structural_groups.csv
and writes <out>/h3/*.csv, *.geojson, figures and metadata.

Density is defined by the number of 3DBAG reference buildings per occupied H3 cell.
Cells at or above the median of that distribution are labelled Dense; the rest Sparse.
Matching groups are assigned to the H3 cell of their reference centroid (for 1:1 groups)
or the centroid of the union of their reference buildings (for multi-feature groups).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio

try:
    import h3
except ImportError as exc:
    raise SystemExit("Install h3_analysis requirements: pip install h3 matplotlib") from exc


def summarize_groups(df: pd.DataFrame) -> dict:
    iou = pd.to_numeric(df["iou"], errors="coerce").dropna()
    h = pd.to_numeric(df["height_diff"], errors="coerce").dropna()
    return {
        "n_groups": int(len(df)),
        "n_groups_iou": int(len(iou)),
        "mean_iou": float(iou.mean()) if len(iou) else None,
        "median_iou": float(iou.median()) if len(iou) else None,
        "n_groups_height": int(len(h)),
        "mean_height_error_m": float(h.mean()) if len(h) else None,
        "median_height_error_m": float(h.median()) if len(h) else None,
        "mean_abs_height_error_m": float(h.abs().mean()) if len(h) else None,
        "rmse_height_m": float(np.sqrt((h * h).mean())) if len(h) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="H3 density analysis for GBA vs 3DBAG")
    parser.add_argument("--out", required=True, help="pipeline output directory")
    parser.add_argument("--resolution", type=int, default=8, help="H3 resolution (default 8)")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    h3out = out / "h3"
    h3out.mkdir(parents=True, exist_ok=True)
    main_path = out / "GBA_vs_3DBAG.geojson"
    struct_path = out / "GBA_vs_3DBAG_structural_groups.csv"
    if not main_path.exists() or not struct_path.exists():
        raise FileNotFoundError("run run_pipeline.py first; comparison outputs not found")

    print("reading comparison ...", flush=True)
    main = pyogrio.read_dataframe(
        main_path,
        columns=[
            "ref_id", "cmp_id_best", "match_category",
            "n_3dbag_in_group", "n_compared_in_group",
            "height_3dbag_m", "height_compared_m", "height_diff_m",
            "polygon_iou",
        ],
    )
    cent = main.geometry.centroid.to_crs(4326)
    main["h3"] = [
        h3.latlng_to_cell(lat, lng, args.resolution)
        for lat, lng in zip(cent.y, cent.x)
    ]
    counts = main["h3"].value_counts()
    threshold = float(counts.median())
    print(
        f"reference buildings={len(main)} occupied_cells={len(counts)} "
        f"median_ref_per_cell={threshold}",
        flush=True,
    )

    main["n_ref_cell"] = main["h3"].map(counts).astype(int)
    main["density"] = np.where(main["n_ref_cell"] >= threshold, "Dense", "Sparse")

    g11 = main[
        (main["n_3dbag_in_group"] == 1)
        & (main["n_compared_in_group"] == 1)
        & (main["match_category"] == "matched_1to1")
    ].copy()
    g11["group_id"] = "1to1:" + g11["ref_id"].astype(str)
    g11["iou"] = pd.to_numeric(g11["polygon_iou"], errors="coerce")
    g11["height_diff"] = pd.to_numeric(g11["height_diff_m"], errors="coerce")
    g11["height_abs"] = g11["height_diff"].abs()
    g11_points = gpd.GeoDataFrame(
        {
            "group_id": g11["group_id"],
            "ref_id": g11["ref_id"],
            "h3": g11["h3"],
            "density": g11["density"],
            "n_ref_cell": g11["n_ref_cell"],
            "iou": g11["iou"],
            "height_diff": g11["height_diff"],
            "height_abs": g11["height_abs"],
        },
        geometry=cent.loc[g11.index],
        crs=4326,
    )
    print("1:1 groups", len(g11_points), flush=True)

    struct = pd.read_csv(struct_path, encoding="utf-8-sig")
    geom_by_ref = dict(zip(main["ref_id"].astype(str), main.geometry))
    multi_rows = []
    for row in struct.itertuples(index=False):
        refs = [r for r in str(row.ref_ids).split(";") if r]
        geoms = [geom_by_ref[r] for r in refs if r in geom_by_ref]
        if not geoms:
            continue
        try:
            union = gpd.GeoSeries(geoms, crs=main.crs).union_all()
            p = union.centroid
        except Exception:
            p = geoms[0].centroid
        p = gpd.GeoSeries([p], crs=main.crs).to_crs(4326).iloc[0]
        cell = h3.latlng_to_cell(p.y, p.x, args.resolution)
        n_ref_cell = int(counts.get(cell, 0))
        density = "Dense" if n_ref_cell >= threshold else "Sparse"
        hdiff = pd.to_numeric(pd.Series([getattr(row, "height_diff_group_m", None)]), errors="coerce").iloc[0]
        iou = pd.to_numeric(pd.Series([getattr(row, "group_iou", None)]), errors="coerce").iloc[0]
        multi_rows.append(
            {
                "group_id": row.group_id,
                "ref_id": None,
                "h3": cell,
                "density": density,
                "n_ref_cell": n_ref_cell,
                "iou": iou,
                "height_diff": hdiff,
                "height_abs": abs(hdiff) if pd.notna(hdiff) else None,
                "geometry": p,
            }
        )
    multi = gpd.GeoDataFrame(multi_rows, geometry="geometry", crs=4326)
    print("multi groups", len(multi), flush=True)

    groups = pd.concat(
        [g11_points.drop(columns="geometry"), multi.drop(columns="geometry")],
        ignore_index=True,
    )
    groups["height_abs"] = groups["height_diff"].abs()
    print("total groups", len(groups), flush=True)

    rows = []
    for dens in ["Sparse", "Dense"]:
        s = summarize_groups(groups[groups["density"] == dens])
        s.update(
            {
                "density": dens,
                "h3_resolution": args.resolution,
                "density_threshold_ref_buildings": threshold,
            }
        )
        rows.append(s)
    summary = pd.DataFrame(rows)[
        [
            "density", "h3_resolution", "density_threshold_ref_buildings",
            "n_groups", "n_groups_iou", "mean_iou", "median_iou",
            "n_groups_height", "mean_height_error_m", "median_height_error_m",
            "mean_abs_height_error_m", "rmse_height_m",
        ]
    ]
    summary.to_csv(h3out / "h3_density_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False), flush=True)

    cell_rows = []
    for cell, sub in groups.groupby("h3"):
        iou = pd.to_numeric(sub["iou"], errors="coerce").dropna()
        h = pd.to_numeric(sub["height_diff"], errors="coerce").dropna()
        n_ref = int(counts.get(cell, 0))
        cell_rows.append(
            {
                "h3": cell,
                "n_ref_buildings": n_ref,
                "density": "Dense" if n_ref >= threshold else "Sparse",
                "n_groups": len(sub),
                "mean_iou": float(iou.mean()) if len(iou) else None,
                "median_iou": float(iou.median()) if len(iou) else None,
                "mean_height_error_m": float(h.mean()) if len(h) else None,
                "mean_abs_height_error_m": float(h.abs().mean()) if len(h) else None,
            }
        )
    pd.DataFrame(cell_rows).to_csv(h3out / "h3_cell_summary.csv", index=False, encoding="utf-8-sig")

    group_points = gpd.GeoDataFrame(
        pd.concat(
            [
                g11_points[["group_id", "h3", "density", "n_ref_cell", "iou", "height_diff", "height_abs", "geometry"]],
                multi[["group_id", "h3", "density", "n_ref_cell", "iou", "height_diff", "height_abs", "geometry"]],
            ],
            ignore_index=True,
        ),
        geometry="geometry",
        crs=4326,
    )
    group_points.to_file(h3out / "h3_group_metrics.geojson", driver="GeoJSON", engine="pyogrio", index=False)

    plt.rcParams.update({"figure.dpi": 140, "font.size": 10})
    order = ["Sparse", "Dense"]
    colors = {"Sparse": "#8ecae6", "Dense": "#ffb703"}

    fig, ax = plt.subplots(figsize=(6, 4))
    data = [pd.to_numeric(groups.loc[groups["density"] == d, "iou"], errors="coerce").dropna() for d in order]
    bp = ax.boxplot(data, positions=[1, 2], patch_artist=True, showfliers=False)
    for patch, d in zip(bp["boxes"], order):
        patch.set_facecolor(colors[d])
        patch.set_alpha(0.8)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(order)
    ax.set_ylabel("Polygon IoU")
    ax.set_title(f"Geometric agreement by H3 density (res {args.resolution})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(h3out / "figure_iou_by_density.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    data = [pd.to_numeric(groups.loc[groups["density"] == d, "height_diff"], errors="coerce").dropna() for d in order]
    bp = ax.boxplot(data, positions=[1, 2], patch_artist=True, showfliers=False)
    for patch, d in zip(bp["boxes"], order):
        patch.set_facecolor(colors[d])
        patch.set_alpha(0.8)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(order)
    ax.set_ylabel("Height difference (GBA - 3DBAG) [m]")
    ax.set_title(f"Vertical agreement by H3 density (res {args.resolution})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(h3out / "figure_height_error_by_density.png")
    plt.close(fig)

    meta = {
        "h3_resolution": args.resolution,
        "occupied_cells": int(len(counts)),
        "density_threshold_ref_buildings": threshold,
        "density_rule": "Dense if reference-building count per occupied H3 cell >= median; Sparse otherwise",
        "groups_total": int(len(groups)),
        "groups_1to1": int(len(g11_points)),
        "groups_multi": int(len(multi)),
        "outputs": {
            "summary": str(h3out / "h3_density_summary.csv"),
            "cells": str(h3out / "h3_cell_summary.csv"),
            "groups": str(h3out / "h3_group_metrics.geojson"),
            "iou_figure": str(h3out / "figure_iou_by_density.png"),
            "height_figure": str(h3out / "figure_height_error_by_density.png"),
        },
    }
    (h3out / "h3_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()

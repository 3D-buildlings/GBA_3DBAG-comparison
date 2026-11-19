"""make_figures_custom.py

Regenerate the GBA vs 3DBAG paper figures (figure 2..5 plus an optional
study-area map) from ANY dataset directory.

Your files are NOT modified. The script checks which columns the figures
need, finds them automatically when possible, and otherwise asks YOU which
of your columns holds each item. The chosen mapping is applied in memory
and saved to <figures>/column_map_used.json so you can reuse it later.

Quick start
-----------
1) Interactive (the script asks for every column it cannot find):
       python make_figures_custom.py --out "D:\\my_outputs"

2) Non-interactive with a mapping file:
       python make_figures_custom.py --out "D:\\my_outputs" --map columns.json

   Create a template for that JSON file:
       python make_figures_custom.py --init-map columns.json

3) Optional study-area map:
       python make_figures_custom.py --out "D:\\my_outputs" ^
           --study-bbox "4.728,52.278,5.079,52.431" --study-name "Amsterdam"

Expected inputs (override with --summary / --groups / --cells):
    <out>/comparison_summary.csv          -> figure2 overall comparison
    <out>/h3/h3_group_metrics.geojson     -> figure3, figure4 (CSV or GeoJSON)
    <out>/h3/h3_cell_summary.csv          -> figure5 (CSV; GeoJSON also works)

Outputs: <out>/figures/*.png and *.pdf, plus column_map_used.json.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from PIL import Image
from shapely.geometry import Point, Polygon, shape

try:
    import geopandas as gpd
except Exception:
    gpd = None

TILE_CACHE = Path(tempfile.gettempdir()) / "gba_3dbag_eox_tiles"
TILE_URL = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg"
HEADERS = {"User-Agent": "Mozilla/5.0"}

COLORS = {"reference": "#1565C0", "compared": "#E07B00", "unmatched": "#9E9E9E"}
DENSITY_COLORS = {"Sparse": "#8ECAE6", "Dense": "#FFB703"}
EXTRA_COLORS = ["#90BE6D", "#F4A261", "#BDE0FE", "#CDB4DB"]

ENCODINGS = ("utf-8-sig", "utf-8", "cp1256", "latin-1")

SECTION_FIELDS = {
    "summary": [
        "reference_count",
        "compared_count",
        "mean_group_iou",
        "mean_group_f1",
        "mean_group_precision",
        "mean_group_recall",
        "height_mae_m",
        "height_rmse_m",
        "height_mape_pct",
        "false_negative_count",
        "false_positive_count",
        "matched_reference_count",
        "matched_compared_count",
        "matched_group_count",
    ],
    "groups": ["density", "iou", "height_diff"],
    "cells": ["h3", "mean_iou", "mean_height_error_m", "mean_abs_height_error_m"],
}

SUMMARY_HARD = [
    "reference_count",
    "compared_count",
    "mean_group_iou",
    "mean_group_f1",
    "mean_group_precision",
    "mean_group_recall",
    "height_mae_m",
    "height_rmse_m",
    "height_mape_pct",
]

ALIASES = {
    "summary": {
        "reference_count": ["ref_count", "n_reference", "n_ref", "reference_buildings", "reference_total"],
        "compared_count": ["cmp_count", "n_compared", "n_cmp", "compared_buildings", "compared_total", "dataset_count"],
        "mean_group_iou": ["mean_iou", "iou_mean", "avg_iou"],
        "mean_group_f1": ["mean_f1", "f1_mean", "avg_f1"],
        "mean_group_precision": ["mean_precision", "precision_mean", "avg_precision"],
        "mean_group_recall": ["mean_recall", "recall_mean", "avg_recall"],
        "height_mae_m": ["mae_m", "height_mae", "mae"],
        "height_rmse_m": ["rmse_m", "height_rmse", "rmse"],
        "height_mape_pct": ["mape_pct", "height_mape", "mape"],
        "false_negative_count": ["fn_count", "false_negatives"],
        "false_positive_count": ["fp_count", "false_positives"],
        "matched_reference_count": ["matched_ref_count", "n_matched_ref"],
        "matched_compared_count": ["matched_cmp_count", "n_matched_cmp"],
        "matched_group_count": ["n_groups", "group_count", "matched_groups"],
    },
    "groups": {
        "density": ["density_class", "h3_density"],
        "iou": ["group_iou", "polygon_iou"],
        "height_diff": ["height_difference", "dh", "height_error", "signed_height_error"],
    },
    "cells": {
        "h3": ["h3_cell", "h3_id", "cell"],
        "mean_iou": ["iou_mean", "avg_iou", "mean_group_iou"],
        "mean_height_error_m": ["mean_height_diff", "mean_signed_height_error_m", "height_err_mean"],
        "mean_abs_height_error_m": ["mean_abs_height_error", "mean_abs_height_diff", "mae_m"],
    },
}

DESCRIPTIONS = {
    ("summary", "reference_count"): "number of reference buildings",
    ("summary", "compared_count"): "number of compared (your dataset) buildings",
    ("summary", "mean_group_iou"): "mean group-level IoU",
    ("summary", "mean_group_f1"): "mean group-level F1",
    ("summary", "mean_group_precision"): "mean group-level precision",
    ("summary", "mean_group_recall"): "mean group-level recall",
    ("summary", "height_mae_m"): "height MAE in metres",
    ("summary", "height_rmse_m"): "height RMSE in metres",
    ("summary", "height_mape_pct"): "height MAPE in percent",
    ("summary", "false_negative_count"): "reference buildings without a match",
    ("summary", "false_positive_count"): "compared polygons without a match",
    ("summary", "matched_reference_count"): "reference buildings with an accepted match",
    ("summary", "matched_compared_count"): "compared buildings with an accepted match",
    ("summary", "matched_group_count"): "number of matched overlap groups",
    ("groups", "density"): "density class (Dense / Sparse)",
    ("groups", "iou"): "group polygon IoU",
    ("groups", "height_diff"): "height difference compared - reference (m)",
    ("cells", "h3"): "H3 cell id (only needed when the file has no geometry)",
    ("cells", "mean_iou"): "mean group IoU per cell",
    ("cells", "mean_height_error_m"): "mean signed height error per cell (m)",
    ("cells", "mean_abs_height_error_m"): "mean absolute height error per cell (m)",
}

FIG = None
INTERACTIVE = False


def die(msg):
    raise SystemExit(f"[make_figures_custom] {msg}")


def save(fig, stem):
    png = FIG / f"{stem}.png"
    pdf = FIG / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {png.name} ({png.stat().st_size:,} bytes)")


def read_csv_any(path):
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return pd.read_csv(path)


def read_attrs(path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_csv_any(path)
    try:
        import pyogrio
        return pyogrio.read_dataframe(path, read_geometry=False)
    except Exception:
        pass
    if gpd is not None:
        try:
            return gpd.read_file(path)
        except Exception:
            pass
    data = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame([f.get("properties", {}) for f in data.get("features", [])])


def load_cells(path):
    if path.suffix.lower() == ".csv":
        return read_csv_any(path), None
    gdf = None
    if gpd is not None:
        try:
            gdf = gpd.read_file(path)
        except Exception:
            gdf = None
    if gdf is None or gdf.geometry.isna().all():
        return read_attrs(path), None
    if gdf.crs is not None and str(gdf.crs).upper() != "EPSG:4326":
        try:
            gdf = gdf.to_crs(4326)
        except Exception:
            pass
    geoms = gdf.geometry
    return pd.DataFrame(gdf.drop(columns=[gdf.geometry.name])), geoms


def ask_column(section, field, columns, interactive):
    if not interactive:
        return None
    desc = DESCRIPTIONS.get((section, field), "")
    print(f"\n[{section}] which column holds: {field}" + (f"  ({desc})" if desc else ""))
    for i, c in enumerate(columns, 1):
        print(f"  {i:>3}. {c}")
    try:
        ans = input("  column name or number (Enter = not available): ").strip()
    except EOFError:
        return None
    if not ans:
        return None
    if ans.isdigit():
        idx = int(ans)
        if 1 <= idx <= len(columns):
            return columns[idx - 1]
        print(f"  number out of range: {ans}")
        return None
    if ans in columns:
        return ans
    lower = {c.lower(): c for c in columns}
    if ans.lower() in lower:
        return lower[ans.lower()]
    print(f"  column not found: {ans}")
    return None


def resolve_columns(df, section, fields, mapping, interactive):
    cols = list(map(str, df.columns))
    by_lower = {c.lower().strip(): c for c in cols}
    section_map = mapping.get(section) or {}
    resolved = {}
    missing = []
    for field in fields:
        col = None
        mapped = section_map.get(field)
        if mapped:
            if mapped in cols:
                col = mapped
            elif str(mapped).lower() in by_lower:
                col = by_lower[str(mapped).lower()]
            else:
                die(f"[--map] column '{mapped}' for '{section}.{field}' not found. Available: {cols}")
        if col is None:
            for cand in [field] + ALIASES.get(section, {}).get(field, []):
                key = cand.lower().strip()
                if key in by_lower:
                    col = by_lower[key]
                    break
        if col is None:
            col = ask_column(section, field, cols, interactive)
        if col is None:
            missing.append(field)
        else:
            resolved[field] = col
    duplicates = [c for c in set(resolved.values()) if list(resolved.values()).count(c) > 1]
    if duplicates:
        die(f"[{section}] the same column is mapped to several fields: {duplicates}")
    return resolved, missing


def numeric_value(row, key, default=None):
    if key not in row.index:
        return default
    v = pd.to_numeric(pd.Series([row[key]]), errors="coerce").iloc[0]
    return default if pd.isna(v) else float(v)


def fig2(d, ref_name, cmp_name, source_name):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.2, 1.2, 1.0]})
    ax = axes[0]
    ref = int(d["reference_count"])
    cmp_ = int(d["compared_count"])
    mref = int(d["matched_reference_count"])
    mcmp = int(d["matched_compared_count"])
    fn = int(d["false_negative_count"])
    fp = int(d["false_positive_count"])
    ax.bar([0], [mref], color=COLORS["reference"], width=0.6, label="matched")
    ax.bar([0], [fn], bottom=[mref], color=COLORS["unmatched"], width=0.6, hatch="//", label="unmatched (FN)")
    ax.bar([1], [mcmp], color=COLORS["compared"], width=0.6)
    ax.bar([1], [fp], bottom=[mcmp], color=COLORS["unmatched"], width=0.6, hatch="//", label="unmatched (FP)")
    ax.text(0, ref, f"{ref:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(1, cmp_, f"{cmp_:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    delta = cmp_ - ref
    pct = d["building_count_error_pct"]
    ax.text(0.5, max(ref, cmp_) * 0.55, f"{delta:+,}\n({pct:+.2f}%)",
            ha="center", va="center", fontsize=9, color="#B00020",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#B00020", alpha=0.9))
    ax.set_xticks([0, 1], [ref_name, cmp_name])
    ax.set_ylabel("Building records")
    ax.set_ylim(0, max(ref, cmp_) * 1.18)
    ax.set_title("(a) Building counts", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
    ax.set_axisbelow(True)

    ax = axes[1]
    labels = ["IoU", "F1", "Precision", "Recall"]
    vals = [d["mean_group_iou"], d["mean_group_f1"], d["mean_group_precision"], d["mean_group_recall"]]
    ax.bar(range(4), vals, color=["#2E7D32", "#388E3C", "#43A047", "#66BB6A"], width=0.65)
    for x, v in zip(range(4), vals):
        ax.text(x, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(range(4), labels)
    ax.set_ylim(max(0.0, min(0.90, min(vals) - 0.03)), 1.005)
    groups = int(d["matched_group_count"])
    ax.set_title(f"(b) Group-level geometric metrics\n(n = {groups:,} groups)", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    ax = axes[2]
    vals = [d["height_mae_m"], d["height_rmse_m"]]
    ax.bar([0, 1], vals, color=["#6A1B9A", "#8E24AA"], width=0.6)
    for x, v in enumerate(vals):
        ax.text(x, v, f"{v:.4f} m", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks([0, 1], ["MAE", "RMSE"])
    ax.set_ylabel("Height error (m)")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title(f"(c) Height error\n(MAPE = {d['height_mape_pct']:.2f}%)", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    handles = [
        Patch(facecolor=COLORS["reference"], label=f"{ref_name} matched"),
        Patch(facecolor=COLORS["compared"], label=f"{cmp_name} matched"),
        Patch(facecolor=COLORS["unmatched"], hatch="//", label="unmatched (FN/FP)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{cmp_name} vs {ref_name}: overall comparison", fontsize=12.5, y=1.02)
    fig.text(0.99, -0.07, f"Source: {source_name}.", ha="right", va="top", fontsize=7, color="#555555")
    fig.tight_layout()
    save(fig, "figure2_overall_comparison")


def density_order(g):
    vals = [str(v) for v in pd.unique(g["density"].dropna())]
    known = [v for v in ("Sparse", "Dense") if v in vals]
    return known + [v for v in vals if v not in known]


def density_color(name, index):
    if name in DENSITY_COLORS:
        return DENSITY_COLORS[name]
    return EXTRA_COLORS[index % len(EXTRA_COLORS)]


def fig3(g):
    order = density_order(g)
    if not order:
        die("figure 3 skipped: density column has no values")
    data = [g.loc[g["density"] == d, "iou"].dropna() for d in order]
    all_vals = pd.concat(data) if data else pd.Series(dtype=float)
    floor = float(np.nanpercentile(all_vals, 1)) if len(all_vals) else 0.9
    lo = math.floor(max(0.0, min(0.94, floor - 0.01)) * 100) / 100
    colors = [density_color(d, i) for i, d in enumerate(order)]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bp = ax.boxplot(data, positions=list(range(1, len(order) + 1)), widths=0.5, patch_artist=True,
                    showfliers=False, showmeans=True,
                    medianprops=dict(color="black", linewidth=1.2),
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=5),
                    whiskerprops=dict(color="#444444"), capprops=dict(color="#444444"))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
    ax.set_xticks(list(range(1, len(order) + 1)))
    ax.set_xticklabels([f"{d}\n(n = {len(x):,})" for d, x in zip(order, data)])
    ax.set_ylim(lo, 1.0)
    ax.set_ylabel("Group-level polygon IoU")
    ax.set_title("Geometric agreement by H3 density", fontsize=10.5)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for x, s in zip(range(1, len(order) + 1), data):
        ax.text(x, lo + (1.0 - lo) * 0.05, f"mean {s.mean():.4f}\nmedian {s.median():.4f}",
                ha="center", va="bottom", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", alpha=0.9))
    fig.tight_layout()
    save(fig, "figure3_iou_by_h3_density")


def fig4(g):
    order = density_order(g)
    if not order:
        die("figure 4 skipped: density column has no values")
    data = [g.loc[g["density"] == d, "height_diff"].dropna() for d in order]
    all_vals = pd.concat(data) if data else pd.Series(dtype=float)
    floor = float(np.nanpercentile(np.abs(all_vals), 98)) if len(all_vals) else 0.0
    lim = max(10.0, floor * 1.1)
    colors = [density_color(d, i) for i, d in enumerate(order)]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bp = ax.boxplot(data, positions=list(range(1, len(order) + 1)), widths=0.5, patch_artist=True,
                    showfliers=False, showmeans=True,
                    medianprops=dict(color="black", linewidth=1.2),
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=5),
                    whiskerprops=dict(color="#444444"), capprops=dict(color="#444444"))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(list(range(1, len(order) + 1)))
    ax.set_xticklabels([f"{d}\n(n = {len(x):,})" for d, x in zip(order, data)])
    ax.set_ylim(-lim, lim)
    ax.set_ylabel("Height difference, compared - reference (m)")
    ax.set_title("Vertical agreement by H3 density", fontsize=10.5)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.text(0.01, 0.99, "Negative values: compared dataset lower than reference.\nDiamond = mean; line = median.",
            transform=ax.transAxes, fontsize=7, color="#555555", va="top")
    for x, s in zip(range(1, len(order) + 1), data):
        ax.text(x, -lim * 0.94, f"mean {s.mean():.3f}\nmedian {s.median():.3f}",
                ha="center", va="bottom", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", alpha=0.9))
    fig.tight_layout()
    save(fig, "figure4_height_error_by_h3_density")


def deg2num(lat, lon, z):
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def num2deg(x, y, z):
    n = 2.0 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lat, lon


def pick_zoom(extent):
    span = max(extent[2] - extent[0], extent[3] - extent[1])
    for max_span, z in ((0.08, 13), (0.18, 12), (0.4, 11), (0.9, 10), (2.0, 9), (4.5, 8), (9.0, 7)):
        if span < max_span:
            return z
    return 6


def tile_span(bbox, z):
    lon0, lat0, lon1, lat1 = bbox
    x0, y0 = deg2num(lat1, lon0, z)
    x1, y1 = deg2num(lat0, lon1, z)
    x0, x1 = int(math.floor(x0)), int(math.floor(x1))
    y0, y1 = int(math.floor(y0)), int(math.floor(y1))
    return x0, y0, x1, y1


def fetch_tile(z, x, y):
    fp = TILE_CACHE / f"{z}_{x}_{y}.jpg"
    if fp.exists() and fp.stat().st_size > 0:
        return Image.open(fp).convert("RGB")
    for attempt in range(4):
        try:
            req = urllib.request.Request(TILE_URL.format(z=z, x=x, y=y), headers=HEADERS)
            with urllib.request.urlopen(req, timeout=45) as r:
                data = r.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
            img.save(fp, quality=92)
            return img
        except Exception:
            time.sleep(1.2 * (attempt + 1))
    return Image.new("RGB", (256, 256), (180, 180, 180))


def mosaic(bbox, z):
    while z > 3:
        x0, y0, x1, y1 = tile_span(bbox, z)
        if (x1 - x0 + 1) * (y1 - y0 + 1) <= 220:
            break
        z -= 1
    canvas = Image.new("RGB", ((x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256))
    for ix in range(x0, x1 + 1):
        for iy in range(y0, y1 + 1):
            canvas.paste(fetch_tile(z, ix, iy), ((ix - x0) * 256, (iy - y0) * 256))
    top_lat, left_lon = num2deg(x0, y0, z)
    bottom_lat, right_lon = num2deg(x1 + 1, y1 + 1, z)
    return np.asarray(canvas), (left_lon, right_lon, bottom_lat, top_lat)


def find_country(bbox):
    url = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
           "master/geojson/ne_110m_admin_0_countries.geojson")
    fp = TILE_CACHE / "countries.geojson"
    try:
        if not fp.exists():
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as r:
                fp.write_bytes(r.read())
        data = json.loads(fp.read_text(encoding="utf-8"))
        pt = Point((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        for feature in data.get("features", []):
            try:
                geom = shape(feature["geometry"])
            except Exception:
                continue
            if geom.contains(pt):
                return geom
    except Exception:
        return None
    return None


def plot_polygon(ax, geom, **kwargs):
    if geom is None:
        return
    geoms = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    for g in geoms:
        x, y = g.exterior.xy
        ax.plot(x, y, **kwargs)


def draw_study_box(ax, bbox, name):
    dx = bbox[2] - bbox[0]
    dy = bbox[3] - bbox[1]
    ax.add_patch(Rectangle((bbox[0], bbox[1]), dx, dy, fill=False,
                           edgecolor="#D32F2F", linewidth=2.0, zorder=5))
    ax.text(bbox[0] + dx * 0.02, bbox[3] - dy * 0.04, name,
            fontsize=8, color="white", fontweight="bold", ha="left", va="top", zorder=6,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#D32F2F", edgecolor="none", alpha=0.92))


def fig_study_area(bbox, name):
    dx = bbox[2] - bbox[0]
    dy = bbox[3] - bbox[1]
    extent = (bbox[0] - dx * 0.45 - 0.02, bbox[1] - dy * 0.45 - 0.02,
              bbox[2] + dx * 0.45 + 0.02, bbox[3] + dy * 0.45 + 0.02)
    z = pick_zoom(extent)
    img, im_ext = mosaic(extent, z)
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    ax.imshow(img, extent=im_ext, origin="upper", zorder=1)
    plot_polygon(ax, find_country(bbox), color="white", linewidth=1.1, zorder=3)
    draw_study_box(ax, bbox, name)
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(f"Study area: {name}", fontsize=11)
    fig.tight_layout()
    save(fig, "study_area_map")


def finite_values(series):
    s = pd.to_numeric(series, errors="coerce")
    arr = s.to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def robust_iou_range(series):
    vals = finite_values(series)
    if vals.size == 0:
        return 0.85, 1.0
    lo = float(np.nanpercentile(vals, 1))
    lo = math.floor(max(0.0, min(0.94, lo - 0.01)) * 100) / 100
    return lo, 1.0


def robust_symmetric_limit(series, floor):
    vals = finite_values(series)
    if vals.size == 0:
        return floor
    return max(floor, float(np.nanpercentile(np.abs(vals), 98)) * 1.1)


def robust_upper_limit(series, floor):
    vals = finite_values(series)
    if vals.size == 0:
        return floor
    return max(floor, float(np.nanpercentile(vals, 98)) * 1.1)


def build_cells_gdf(df, resolved, geoms):
    if gpd is None:
        die("geopandas is required for figure 5 (pip install geopandas shapely)")
    renames = {src: dst for dst, src in resolved.items()}
    data = df.rename(columns=renames)
    for col in ("mean_iou", "mean_height_error_m", "mean_abs_height_error_m"):
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    if geoms is not None:
        return gpd.GeoDataFrame(data, geometry=list(geoms), crs=4326)
    try:
        import h3
    except ImportError:
        die("the 'h3' package is required when the cells file has no geometry (pip install h3)")
    rows = []
    for _, r in data.iterrows():
        cell = str(r["h3"])
        try:
            boundary = h3.cell_to_boundary(cell)
        except AttributeError:
            boundary = h3.h3_to_geo_boundary(cell, geo_json=False)
        coords = [(lng, lat) for lat, lng in boundary]
        if len(coords) >= 3:
            rows.append({**r.to_dict(), "geometry": Polygon(coords)})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)


def fig5(gdf, study_bbox=None, study_name="Study area"):
    if gdf is None or len(gdf) == 0:
        die("figure 5 skipped: no H3 cells could be built from the cells file")
    minx, miny, maxx, maxy = gdf.total_bounds
    dx = max(maxx - minx, 1e-3)
    dy = max(maxy - miny, 1e-3)
    extent = (minx - max(dx * 0.15, 0.01), miny - max(dy * 0.15, 0.01),
              maxx + max(dx * 0.15, 0.01), maxy + max(dy * 0.15, 0.01))
    img, im_ext = mosaic(extent, pick_zoom(extent))
    iou_lo, iou_hi = robust_iou_range(gdf["mean_iou"])
    sig = robust_symmetric_limit(gdf["mean_height_error_m"], 5.0)
    abs_max = robust_upper_limit(gdf["mean_abs_height_error_m"], 6.0)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), gridspec_kw={"wspace": 0.12})
    specs = [
        ("mean_iou", "Mean group IoU", "viridis", iou_lo, iou_hi, "min"),
        ("mean_height_error_m", "Mean signed height error (m)", "RdBu_r", -sig, sig, "both"),
        ("mean_abs_height_error_m", "Mean absolute height error (m)", "magma", 0.0, abs_max, "max"),
    ]
    titles = ["(a) Geometric agreement", "(b) Signed vertical bias", "(c) Absolute vertical error"]
    for ax, (col, label, cmap_name, vmin, vmax, extend), title in zip(axes, specs, titles):
        ax.imshow(img, extent=im_ext, origin="upper", zorder=1)
        cmap = plt.get_cmap(cmap_name).copy()
        if extend in ("min", "both"):
            cmap.set_under("#BDBDBD")
        if extend in ("max", "both"):
            cmap.set_over("#F7F7F7")
        g = gdf.dropna(subset=[col])
        if len(g) > 0:
            g.plot(column=col, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax,
                   edgecolor="white", linewidth=0.12, alpha=0.82, zorder=4)
        if study_bbox is not None:
            draw_study_box(ax, study_bbox, study_name)
        ax.set_xlim(extent[0], extent[2])
        ax.set_ylim(extent[1], extent[3])
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"{title}\n{label}", fontsize=9.5, pad=4)
        sm = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.045, pad=0.02,
                          extend=extend, shrink=0.85)
        cb.ax.tick_params(labelsize=6.5)
        cb.set_label(label, fontsize=7)
    n_cells = int(gdf.dropna(subset=["mean_iou", "mean_height_error_m", "mean_abs_height_error_m"]).shape[0])
    fig.suptitle("Spatial distribution of H3-based performance", fontsize=12, y=0.99)
    fig.text(0.99, -0.02,
             f"H3 cells containing matched groups (n = {n_cells:,}).\n"
             "Signed error = compared - reference; negative values indicate lower compared heights.\n"
             "Imagery: Sentinel-2 cloudless 2020 (EOX/Copernicus).",
             ha="right", va="bottom", fontsize=6.3, color="#444444")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "figure5_h3_spatial_performance")


def load_mapping(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        die(f"could not read --map file '{path}': {exc}")
    if not isinstance(data, dict):
        die("--map file must contain a JSON object")
    return data


def write_template(path):
    template = {section: {field: None for field in fields} for section, fields in SECTION_FIELDS.items()}
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(template, indent=2), encoding="utf-8")
    print(f"template written: {out.resolve()}")


def parse_bbox(text):
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        die("--study-bbox must be min_lon,min_lat,max_lon,max_lat")
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        die("--study-bbox contains non-numeric values")
    if not (vals[0] < vals[2] and vals[1] < vals[3]):
        die("--study-bbox must satisfy min_lon < max_lon and min_lat < max_lat")
    return tuple(vals)


def main():
    parser = argparse.ArgumentParser(description="Build the paper figures from any dataset with any column names")
    parser.add_argument("--out", required=True, help="dataset/pipeline output directory")
    parser.add_argument("--figures", default=None, help="figures output directory (default: <out>/figures)")
    parser.add_argument("--summary", default=None, help="summary CSV/GeoJSON (default: <out>/comparison_summary.csv)")
    parser.add_argument("--groups", default=None, help="matched-group file (default: <out>/h3/h3_group_metrics.geojson)")
    parser.add_argument("--cells", default=None, help="H3 cell file (default: <out>/h3/h3_cell_summary.csv)")
    parser.add_argument("--map", default=None, help="JSON file with your column names per field")
    parser.add_argument("--init-map", default=None, help="write a JSON template of all required fields and exit")
    parser.add_argument("--no-prompt", action="store_true", help="never ask interactively; fail if a column is missing")
    parser.add_argument("--skip", default="", help="comma list: fig2,fig3,fig4,fig5,study")
    parser.add_argument("--reference-name", default="3DBAG", help="label for the reference dataset")
    parser.add_argument("--compared-name", default="GBA", help="label for the compared dataset")
    parser.add_argument("--study-bbox", default=None, help="min_lon,min_lat,max_lon,max_lat for the study-area map")
    parser.add_argument("--study-name", default="Study area", help="label for the study area")
    args = parser.parse_args()

    global FIG, INTERACTIVE
    INTERACTIVE = bool(not args.no_prompt and sys.stdin is not None and sys.stdin.isatty())
    TILE_CACHE.mkdir(parents=True, exist_ok=True)

    if args.init_map:
        write_template(args.init_map)
        return

    base = Path(args.out).resolve()
    if not base.exists():
        die(f"--out directory not found: {base}")
    FIG = Path(args.figures).resolve() if args.figures else base / "figures"
    FIG.mkdir(parents=True, exist_ok=True)
    mapping = load_mapping(args.map) if args.map else {}
    skip = {s.strip().lower() for s in args.skip.split(",") if s.strip()}
    used_map = {}

    summary_path = Path(args.summary).resolve() if args.summary else base / "comparison_summary.csv"
    if "fig2" in skip:
        print("figure 2 skipped by --skip")
    elif not summary_path.exists():
        die(f"summary file not found: {summary_path} (use --summary or --skip fig2)")
    else:
        df = read_csv_any(summary_path)
        if df.empty:
            die(f"summary file is empty: {summary_path}")
        resolved, missing = resolve_columns(df, "summary", SECTION_FIELDS["summary"], mapping, INTERACTIVE)
        hard_missing = [f for f in SUMMARY_HARD if f in missing]
        if hard_missing:
            die("could not resolve summary columns: " + ", ".join(hard_missing) +
                "\nUse --map or run interactively (see --init-map).")
        used_map["summary"] = dict(resolved)
        renames = {src: dst for dst, src in resolved.items()}
        row = df.iloc[0].rename(index=renames)
        ref = numeric_value(row, "reference_count")
        cmp_ = numeric_value(row, "compared_count")
        if ref is None or cmp_ is None:
            die("reference_count / compared_count values are missing or not numeric")
        fn = numeric_value(row, "false_negative_count", 0.0)
        fp = numeric_value(row, "false_positive_count", 0.0)
        mref = numeric_value(row, "matched_reference_count", ref - fn)
        mcmp = numeric_value(row, "matched_compared_count", cmp_ - fp)
        gcount = numeric_value(row, "matched_group_count", mref)
        for field in missing:
            print(f"warning: summary column '{field}' not found; using a derived/default value")
        d = {
            "reference_count": ref,
            "compared_count": cmp_,
            "matched_reference_count": mref,
            "matched_compared_count": mcmp,
            "false_negative_count": fn,
            "false_positive_count": fp,
            "matched_group_count": gcount,
            "building_count_error_pct": ((cmp_ - ref) / ref * 100.0) if ref else float("nan"),
        }
        for field in SUMMARY_HARD:
            if field in ("reference_count", "compared_count"):
                continue
            val = numeric_value(row, field)
            if val is None:
                die(f"summary value '{field}' is missing or not numeric")
            d[field] = val
        fig2(d, args.reference_name, args.compared_name, summary_path.name)

    groups_path = Path(args.groups).resolve() if args.groups else base / "h3" / "h3_group_metrics.geojson"
    need_groups = [fig for fig in ("fig3", "fig4") if fig not in skip]
    if not need_groups:
        if "fig3" in skip and "fig4" in skip:
            print("figures 3 and 4 skipped by --skip")
    elif not groups_path.exists():
        die(f"groups file not found: {groups_path} (use --groups or --skip fig3,fig4)")
    else:
        raw = read_attrs(groups_path)
        resolved, missing = resolve_columns(raw, "groups", SECTION_FIELDS["groups"], mapping, INTERACTIVE)
        if missing:
            die("could not resolve groups columns: " + ", ".join(missing) +
                "\nUse --map or run interactively (see --init-map).")
        used_map["groups"] = dict(resolved)
        renames = {src: dst for dst, src in resolved.items()}
        g = raw.rename(columns=renames)
        for col in ("iou", "height_diff"):
            g[col] = pd.to_numeric(g[col], errors="coerce")
        g["density"] = g["density"].astype(str).str.strip().str.title()
        if "fig3" not in skip:
            fig3(g)
        if "fig4" not in skip:
            fig4(g)

    cells_path = Path(args.cells).resolve() if args.cells else base / "h3" / "h3_cell_summary.csv"
    if "fig5" in skip:
        print("figure 5 skipped by --skip")
    elif not cells_path.exists():
        die(f"cells file not found: {cells_path} (use --cells or --skip fig5)")
    else:
        raw, geoms = load_cells(cells_path)
        fields = ["mean_iou", "mean_height_error_m", "mean_abs_height_error_m"]
        if geoms is None:
            fields = fields + ["h3"]
        resolved, missing = resolve_columns(raw, "cells", fields, mapping, INTERACTIVE)
        if missing:
            die("could not resolve cells columns: " + ", ".join(missing) +
                "\nUse --map or run interactively (see --init-map).")
        used_map["cells"] = dict(resolved)
        gdf = build_cells_gdf(raw, resolved, geoms)
        study_bbox = parse_bbox(args.study_bbox) if args.study_bbox else None
        fig5(gdf, study_bbox, args.study_name)
        if study_bbox is not None and "study" not in skip:
            fig_study_area(study_bbox, args.study_name)

    if used_map:
        out_map = FIG / "column_map_used.json"
        out_map.write_text(json.dumps(used_map, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"column mapping saved: {out_map}")


if __name__ == "__main__":
    main()

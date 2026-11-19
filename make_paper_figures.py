from __future__ import annotations

import argparse
import io
import json
import math
import tempfile
import time
import urllib.request
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
import pyogrio
from PIL import Image
from shapely.geometry import Polygon
from shapely.ops import unary_union

import h3

BASE = None
FIG = None
META = {}
TILE_CACHE = Path(tempfile.gettempdir()) / "gba_3dbag_eox_tiles"
TILE_CACHE.mkdir(parents=True, exist_ok=True)
TILE_URL = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg"
HEADERS = {"User-Agent": "Mozilla/5.0"}

AMSTERDAM_BBOX = (4.728, 52.278, 5.079, 52.431)
UTRECHT_BBOX = (5.0, 52.03, 5.18, 52.14)
AMSTERDAM = (4.9041, 52.3676)
UTRECHT = (5.1214, 52.0907)
COLORS = {"3DBAG": "#1565C0", "GBA": "#E07B00", "unmatched": "#9E9E9E"}


def save(fig, stem):
    png = FIG / f"{stem}.png"
    pdf = FIG / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(stem, png.stat().st_size)


def finite_values(series):
    s = pd.to_numeric(series, errors="coerce")
    arr = s.to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def robust_iou_range(series):
    vals = finite_values(series)
    if vals.size == 0:
        return 0.85, 1.0
    lo = math.floor(max(0.0, min(0.94, float(np.nanpercentile(vals, 1)) - 0.01)) * 100) / 100
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


def resolution_suffix():
    res = META.get("h3_resolution")
    return f" (resolution {res})" if res is not None else ""


def fig2():
    s = pd.read_csv(BASE / "comparison_summary.csv", encoding="utf-8-sig").iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.2, 1.2, 1.0]})
    ax = axes[0]
    ref = int(s["reference_count"])
    cmp = int(s["compared_count"])
    mref = int(s["matched_reference_count"])
    mcmp = int(s["matched_compared_count"])
    fn = int(s["false_negative_count"])
    fp = int(s["false_positive_count"])
    ax.bar([0], [mref], color=COLORS["3DBAG"], width=0.6, label="matched")
    ax.bar([0], [fn], bottom=[mref], color=COLORS["unmatched"], width=0.6, hatch="//", label="unmatched (FN)")
    ax.bar([1], [mcmp], color=COLORS["GBA"], width=0.6)
    ax.bar([1], [fp], bottom=[mcmp], color=COLORS["unmatched"], width=0.6, hatch="//", label="unmatched (FP)")
    ax.text(0, ref, f"{ref:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(1, cmp, f"{cmp:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(0.5, max(ref, cmp) * 0.55, f"+{cmp - ref:,}\n({s['building_count_error_pct']:.2f}%)",
            ha="center", va="center", fontsize=9, color="#B00020",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#B00020", alpha=0.9))
    ax.set_xticks([0, 1], ["3DBAG", "GBA"])
    ax.set_ylabel("Building records")
    ax.set_ylim(0, cmp * 1.18)
    ax.set_title("(a) Building counts", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
    ax.set_axisbelow(True)

    ax = axes[1]
    labels = ["IoU", "F1", "Precision", "Recall"]
    vals = [s["mean_group_iou"], s["mean_group_f1"], s["mean_group_precision"], s["mean_group_recall"]]
    bars = ax.bar(range(4), vals, color=["#2E7D32", "#388E3C", "#43A047", "#66BB6A"], width=0.65)
    for x, v in zip(range(4), vals):
        ax.text(x, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(range(4), labels)
    ax.set_ylim(max(0.90, min(vals) - 0.03), 1.005)
    ax.set_title(f"(b) Group-level geometric metrics\n(n = {int(s['matched_group_count']):,} groups)", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    ax = axes[2]
    vals = [s["height_mae_m"], s["height_rmse_m"]]
    ax.bar([0, 1], vals, color=["#6A1B9A", "#8E24AA"], width=0.6)
    for x, v in enumerate(vals):
        ax.text(x, v, f"{v:.4f} m", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks([0, 1], ["MAE", "RMSE"])
    ax.set_ylabel("Height error (m)")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title(f"(c) Height error\n(MAPE = {s['height_mape_pct']:.2f}%)", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    handles = [
        Patch(facecolor=COLORS["3DBAG"], label="3DBAG matched"),
        Patch(facecolor=COLORS["GBA"], label="GBA matched"),
        Patch(facecolor=COLORS["unmatched"], hatch="//", label="unmatched (FN/FP)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("GBA vs 3DBAG: overall comparison", fontsize=12.5, y=1.02)
    fig.text(0.99, -0.07, "Source: comparison_summary.csv (latest pipeline output).",
             ha="right", va="top", fontsize=7, color="#555555")
    fig.tight_layout()
    save(fig, "figure2_overall_comparison")


def load_groups():
    g = pyogrio.read_dataframe(
        BASE / "h3" / "h3_group_metrics.geojson",
        columns=["density", "iou", "height_diff"],
        read_geometry=False,
    )
    g["iou"] = pd.to_numeric(g["iou"], errors="coerce")
    g["height_diff"] = pd.to_numeric(g["height_diff"], errors="coerce")
    g["height_abs"] = g["height_diff"].abs()
    return g


def fig3(g):
    order = ["Sparse", "Dense"]
    colors = {"Sparse": "#8ECAE6", "Dense": "#FFB703"}
    data = [g.loc[g["density"] == d, "iou"].dropna() for d in order]
    lo = robust_iou_range(pd.concat(data) if data else pd.Series(dtype=float))[0]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bp = ax.boxplot(data, positions=[1, 2], widths=0.5, patch_artist=True,
                    showfliers=False, showmeans=True,
                    medianprops=dict(color="black", linewidth=1.2),
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=5),
                    whiskerprops=dict(color="#444444"), capprops=dict(color="#444444"))
    for patch, d in zip(bp["boxes"], order):
        patch.set_facecolor(colors[d]); patch.set_alpha(0.9)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"{d}\n(n = {len(x):,})" for d, x in zip(order, data)])
    ax.set_ylim(lo, 1.0)
    ax.set_ylabel("Group-level polygon IoU")
    ax.set_title("Geometric agreement by H3 density" + resolution_suffix(), fontsize=10.5)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for x, s in zip([1, 2], data):
        ax.text(x, lo + (1.0 - lo) * 0.05, f"mean {s.mean():.4f}\nmedian {s.median():.4f}",
                ha="center", va="bottom", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", alpha=0.9))
    fig.tight_layout()
    save(fig, "figure3_iou_by_h3_density")


def fig4(g):
    order = ["Sparse", "Dense"]
    colors = {"Sparse": "#8ECAE6", "Dense": "#FFB703"}
    data = [g.loc[g["density"] == d, "height_diff"].dropna() for d in order]
    lim = robust_symmetric_limit(pd.concat(data) if data else pd.Series(dtype=float), 10.0)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bp = ax.boxplot(data, positions=[1, 2], widths=0.5, patch_artist=True,
                    showfliers=False, showmeans=True,
                    medianprops=dict(color="black", linewidth=1.2),
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=5),
                    whiskerprops=dict(color="#444444"), capprops=dict(color="#444444"))
    for patch, d in zip(bp["boxes"], order):
        patch.set_facecolor(colors[d]); patch.set_alpha(0.9)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"{d}\n(n = {len(x):,})" for d, x in zip(order, data)])
    ax.set_ylim(-lim, lim)
    ax.set_ylabel("Height difference, GBA - 3DBAG (m)")
    ax.set_title("Vertical agreement by H3 density" + resolution_suffix(), fontsize=10.5)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.text(0.01, 0.99, "Negative values: GBA lower than 3DBAG.\nDiamond = mean; line = median.",
            transform=ax.transAxes, fontsize=7, color="#555555", va="top")
    for x, s in zip([1, 2], data):
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
    lon0, lat0, lon1, lat1 = bbox
    x0, y0 = deg2num(lat1, lon0, z)
    x1, y1 = deg2num(lat0, lon1, z)
    x0, x1 = int(math.floor(x0)), int(math.floor(x1))
    y0, y1 = int(math.floor(y0)), int(math.floor(y1))
    canvas = Image.new("RGB", ((x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256))
    for ix in range(x0, x1 + 1):
        for iy in range(y0, y1 + 1):
            canvas.paste(fetch_tile(z, ix, iy), ((ix - x0) * 256, (iy - y0) * 256))
    top_lat, left_lon = num2deg(x0, y0, z)
    bottom_lat, right_lon = num2deg(x1 + 1, y1 + 1, z)
    return np.asarray(canvas), (left_lon, right_lon, bottom_lat, top_lat)


def cell_polygons(cells):
    rows = []
    for cell, row in cells.iterrows():
        boundary = h3.cell_to_boundary(cell)
        coords = [(lng, lat) for lat, lng in boundary]
        if len(coords) >= 3:
            rows.append({**row.to_dict(), "geometry": Polygon(coords)})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)


def add_context(ax):
    ax.add_patch(Rectangle((AMSTERDAM_BBOX[0], AMSTERDAM_BBOX[1]),
                           AMSTERDAM_BBOX[2] - AMSTERDAM_BBOX[0],
                           AMSTERDAM_BBOX[3] - AMSTERDAM_BBOX[1],
                           fill=False, edgecolor="#FFEB3B", linewidth=1.3, zorder=7))
    ax.add_patch(Rectangle((UTRECHT_BBOX[0], UTRECHT_BBOX[1]),
                           UTRECHT_BBOX[2] - UTRECHT_BBOX[0],
                           UTRECHT_BBOX[3] - UTRECHT_BBOX[1],
                           fill=False, edgecolor="#FFEB3B", linewidth=1.3, zorder=7))
    ax.plot(*AMSTERDAM, marker="o", markersize=4, color="white", markeredgecolor="black", zorder=8)
    ax.plot(*UTRECHT, marker="o", markersize=4, color="white", markeredgecolor="black", zorder=8)
    ax.text(AMSTERDAM[0] + 0.012, AMSTERDAM[1] + 0.012, "Amsterdam", color="white", fontsize=7.5,
            fontweight="bold", zorder=9,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.55, edgecolor="none"))
    ax.text(UTRECHT[0] + 0.012, UTRECHT[1] + 0.012, "Utrecht", color="white", fontsize=7.5,
            fontweight="bold", zorder=9,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.55, edgecolor="none"))


def fig5():
    cells = pd.read_csv(BASE / "h3" / "h3_cell_summary.csv", encoding="utf-8-sig")
    cells = cells.set_index("h3")
    gdf = cell_polygons(cells)
    minx, miny, maxx, maxy = gdf.total_bounds
    dx = max(maxx - minx, 1e-3)
    dy = max(maxy - miny, 1e-3)
    extent = (minx - max(dx * 0.15, 0.01), miny - max(dy * 0.15, 0.01),
              maxx + max(dx * 0.15, 0.01), maxy + max(dy * 0.15, 0.01))
    img, im_ext = mosaic(extent, 10)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), gridspec_kw={"wspace": 0.12})
    iou_lo, iou_hi = robust_iou_range(gdf["mean_iou"])
    sig = robust_symmetric_limit(gdf["mean_height_error_m"], 5.0)
    abs_max = robust_upper_limit(gdf["mean_abs_height_error_m"], 6.0)
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
        g.plot(column=col, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax,
               edgecolor="white", linewidth=0.12, alpha=0.82, zorder=4)
        add_context(ax)
        ax.set_xlim(extent[0], extent[2]); ax.set_ylim(extent[1], extent[3])
        ax.set_aspect("equal"); ax.set_axis_off()
        ax.set_title(f"{title}\n{label}", fontsize=9.5, pad=4)
        sm = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.045, pad=0.02,
                          extend=extend, shrink=0.85)
        cb.ax.tick_params(labelsize=6.5)
        cb.set_label(label, fontsize=7)
    n_cells = int(len(cells))
    note = f"H3 cells containing matched groups (n = {n_cells:,})"
    if META.get("cells_without_groups") is not None:
        note += f"; {int(META['cells_without_groups']):,} occupied cells without groups omitted"
    note += "."
    fig.suptitle("Spatial distribution of H3-based performance" + resolution_suffix(), fontsize=12, y=0.99)
    fig.text(0.99, -0.02, note + "\n"
             "Signed error = GBA - 3DBAG; negative values indicate lower GBA heights.\n"
             "Imagery: Sentinel-2 cloudless 2020 (EOX/Copernicus). Yellow outlines: study bboxes.",
             ha="right", va="bottom", fontsize=6.3, color="#444444")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "figure5_h3_spatial_performance")


def main():
    parser = argparse.ArgumentParser(description="Create the paper figures from pipeline outputs")
    parser.add_argument("--out", required=True, help="pipeline output directory")
    args = parser.parse_args()
    global BASE, FIG, META
    BASE = Path(args.out).resolve()
    FIG = BASE / "figures"
    FIG.mkdir(parents=True, exist_ok=True)
    meta_path = BASE / "h3" / "h3_metadata.json"
    if meta_path.exists():
        try:
            META = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        except Exception:
            META = {}
    TILE_CACHE.mkdir(parents=True, exist_ok=True)
    fig2()
    groups = load_groups()
    fig3(groups)
    fig4(groups)
    fig5()


if __name__ == "__main__":
    main()

"""study_area_map.py

Build the study-area figure (context map + one inset per study area).

Running without --area reproduces the paper figure for the Netherlands
(Amsterdam + Utrecht). For any other region, pass one or more --area values:

    python study_area_map.py --out /path/to/output ^
        --area "Delft:4.34,51.98,4.41,52.03"

    python study_area_map.py --out /path/to/output ^
        --area "Area A:4.34,51.98,4.41,52.03" ^
        --area "Area B:4.44,52.13,4.53,52.19" ^
        --title "My study areas"

Outputs <out>/figures/study_areas_netherlands.png/pdf for the default
areas, or <out>/figures/study_area_map.png/pdf otherwise (override the
file stem with --stem).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import tempfile
import time
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
from PIL import Image
from shapely.geometry import Point, shape

TILE_CACHE = Path(tempfile.gettempdir()) / "gba_3dbag_eox_tiles"
TILE_URL = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg"
HEADERS = {"User-Agent": "Mozilla/5.0"}

DEFAULT_AREAS = [
    ("Amsterdam", (4.728, 52.278, 5.079, 52.431)),
    ("Utrecht", (5.0, 52.03, 5.18, 52.14)),
]
DEFAULT_CONTEXT = (3.0, 50.6, 7.4, 53.7)
PALETTE = ["#D32F2F", "#1565C0", "#2E7D32", "#6A1B9A", "#E07B00", "#00838F"]


def die(msg):
    raise SystemExit(f"[study_area_map] {msg}")


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
    url = TILE_URL.format(z=z, x=x, y=y)
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
            img.save(fp, quality=92)
            return img
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return Image.new("RGB", (256, 256), (200, 200, 200))


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


def find_country(lon, lat):
    url = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
           "master/geojson/ne_110m_admin_0_countries.geojson")
    fp = TILE_CACHE / "countries.geojson"
    try:
        if not fp.exists():
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as r:
                fp.write_bytes(r.read())
        data = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return None
    pt = Point(lon, lat)
    for feature in data.get("features", []):
        try:
            geom = shape(feature["geometry"])
        except Exception:
            continue
        if geom.contains(pt):
            return geom
    return None


def plot_polygon(ax, geom, **kwargs):
    if geom is None:
        return
    geoms = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    for g in geoms:
        x, y = g.exterior.xy
        ax.plot(x, y, **kwargs)


def add_bbox(ax, bbox, color, label, xytext, ha, va):
    lon0, lat0, lon1, lat1 = bbox
    ax.add_patch(Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0,
                           fill=False, edgecolor=color, linewidth=2.0, zorder=6))
    ax.annotate(
        label,
        xy=((lon0 + lon1) / 2, (lat0 + lat1) / 2),
        xytext=xytext,
        textcoords="offset points",
        fontsize=8.5,
        color="white",
        fontweight="bold",
        ha=ha,
        va=va,
        zorder=8,
        bbox=dict(boxstyle="round,pad=0.22", facecolor=color, edgecolor="none", alpha=0.9),
    )


def add_inset_box(ax, bbox, color, label):
    lon0, lat0, lon1, lat1 = bbox
    ax.add_patch(Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0,
                           fill=False, edgecolor=color, linewidth=2.0, zorder=6))
    ax.text(
        lon0 + (lon1 - lon0) * 0.02, lat1 - (lat1 - lat0) * 0.03, label,
        fontsize=8.0, color="white", fontweight="bold", ha="left", va="top", zorder=9,
        bbox=dict(boxstyle="round,pad=0.22", facecolor=color, edgecolor="none", alpha=0.9),
    )


def add_scalebar(ax, bbox, length_km):
    lon0, lat0, lon1, lat1 = bbox
    lat_c = (lat0 + lat1) / 2
    km_per_deg_lon = 111.32 * math.cos(math.radians(lat_c))
    d_lon = length_km / km_per_deg_lon
    x = lon1 - d_lon - (lon1 - lon0) * 0.04
    y = lat0 + (lat1 - lat0) * 0.06
    ax.plot([x, x + d_lon], [y, y], color="white", linewidth=3, solid_capstyle="butt", zorder=9)
    ax.plot([x, x + d_lon], [y, y], color="black", linewidth=1.2, solid_capstyle="butt", zorder=10)
    ax.text(x + d_lon / 2, y + (lat1 - lat0) * 0.012, f"{length_km:g} km",
            color="white", fontsize=7.5, ha="center", va="bottom", zorder=10)


def style_ax(ax):
    ax.set_axis_off()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("white")
        spine.set_linewidth(1.5)


def parse_bbox(text, what="bbox"):
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        die(f"{what} must be min_lon,min_lat,max_lon,max_lat (got: {text})")
    try:
        vals = tuple(float(p) for p in parts)
    except ValueError:
        die(f"{what} contains non-numeric values (got: {text})")
    if not (vals[0] < vals[2] and vals[1] < vals[3]):
        die(f"{what} must satisfy min_lon < max_lon and min_lat < max_lat (got: {text})")
    return vals


def parse_areas(items):
    areas = []
    for item in items:
        if ":" not in item:
            die(f"--area must be NAME:min_lon,min_lat,max_lon,max_lat (got: {item})")
        name, coords = item.rsplit(":", 1)
        name = name.strip()
        if not name:
            die(f"--area name is empty (got: {item})")
        areas.append((name, parse_bbox(coords, f"--area '{name}'")))
    return areas


def auto_context(areas):
    minx = min(b[0] for _, b in areas)
    miny = min(b[1] for _, b in areas)
    maxx = max(b[2] for _, b in areas)
    maxy = max(b[3] for _, b in areas)
    padx = max((maxx - minx) * 0.5, 0.02)
    pady = max((maxy - miny) * 0.5, 0.02)
    return (minx - padx, miny - pady, maxx + padx, maxy + pady)


def inset_extent(bbox, pad_ratio=0.4):
    padx = max((bbox[2] - bbox[0]) * pad_ratio, 0.01)
    pady = max((bbox[3] - bbox[1]) * pad_ratio, 0.01)
    return (bbox[0] - padx, bbox[1] - pady, bbox[2] + padx, bbox[3] + pady)


def nice_km(km):
    for value in (1, 2, 5, 10, 20, 25, 50, 100, 200, 500):
        if km <= value:
            return value
    return 1000


def bbox_width_km(bbox):
    lat_c = (bbox[1] + bbox[3]) / 2
    return (bbox[2] - bbox[0]) * 111.32 * math.cos(math.radians(lat_c))


def main():
    parser = argparse.ArgumentParser(description="Study-area figure with context map and one inset per area")
    parser.add_argument("--out", required=True, help="pipeline output directory")
    parser.add_argument("--figures", default=None, help="figures output directory (default: <out>/figures)")
    parser.add_argument("--area", action="append", default=None, metavar="NAME:min_lon,min_lat,max_lon,max_lat",
                        help="study area (repeatable); omit for Amsterdam + Utrecht")
    parser.add_argument("--context-bbox", default=None, help="context map extent (default: auto around the areas)")
    parser.add_argument("--context-zoom", type=int, default=None, help="tile zoom of the context map")
    parser.add_argument("--inset-zoom", type=int, default=None, help="tile zoom of the insets")
    parser.add_argument("--scalebar-km", type=float, default=None, help="scale-bar length in km (default: auto)")
    parser.add_argument("--title", default=None, help="figure title")
    parser.add_argument("--stem", default=None, help="output file stem (default: study_areas_netherlands or study_area_map)")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    fig_dir = Path(args.figures).resolve() if args.figures else out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    TILE_CACHE.mkdir(parents=True, exist_ok=True)

    using_defaults = not args.area
    areas = parse_areas(args.area) if args.area else list(DEFAULT_AREAS)
    context = parse_bbox(args.context_bbox, "--context-bbox") if args.context_bbox else (
        DEFAULT_CONTEXT if using_defaults else auto_context(areas))
    context_zoom = args.context_zoom or (8 if using_defaults else pick_zoom(context))
    fig_title = args.title or ("Study areas used in the GBA vs 3DBAG comparison" if using_defaults else "Study areas")
    stem = args.stem or ("study_areas_netherlands" if using_defaults else "study_area_map")
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(areas))]

    country = find_country((context[0] + context[2]) / 2, (context[1] + context[3]) / 2)

    fig = plt.figure(figsize=(13.5, 9.2), facecolor="white")
    n = len(areas)
    gs = fig.add_gridspec(n, 2, width_ratios=[1.55, 1], wspace=0.04, hspace=0.08,
                          left=0.02, right=0.98, top=0.91, bottom=0.03)

    ax = fig.add_subplot(gs[:, 0])
    img, ext = mosaic(context, context_zoom)
    ax.imshow(img, extent=ext, origin="upper", zorder=1)
    plot_polygon(ax, country, color="white", linewidth=1.2, zorder=4)
    for i, ((name, bbox), color) in enumerate(zip(areas, colors)):
        if i == 0:
            add_bbox(ax, bbox, color, f"{name} study area", (-14, -20), "right", "top")
        else:
            add_bbox(ax, bbox, color, f"{name} study area", (12, 14), "left", "bottom")
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        ax.plot(cx, cy, marker="o", markersize=6, color="white", markeredgecolor="black", zorder=7)
        ax.text(cx + 0.045, cy + 0.015, name, color="white", fontsize=9,
                fontweight="bold", ha="left", va="bottom", zorder=8,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="black", alpha=0.55, edgecolor="none"))
    ax.set_xlim(context[0], context[2])
    ax.set_ylim(context[1], context[3])
    ax.set_aspect("equal")
    style_ax(ax)
    ax.text(0.99, 0.01, "Sentinel-2 cloudless (2020) by EOX IT Services GmbH; contains modified Copernicus Sentinel data",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="white",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.5, edgecolor="none"), zorder=11)
    ax.legend(handles=[Patch(facecolor="none", edgecolor=color, label=f"{name} study area")
                       for (name, _), color in zip(areas, colors)],
              loc="lower left", fontsize=7.5, framealpha=0.85)

    for i, ((name, bbox), color) in enumerate(zip(areas, colors)):
        ax_i = fig.add_subplot(gs[i, 1])
        extent = inset_extent(bbox)
        zoom = args.inset_zoom or (12 if using_defaults else pick_zoom(extent))
        img, ext = mosaic(extent, zoom)
        ax_i.imshow(img, extent=ext, origin="upper", zorder=1)
        add_inset_box(ax_i, bbox, color, f"{name} study area")
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        ax_i.plot(cx, cy, marker="o", markersize=5, color="white", markeredgecolor="black", zorder=8)
        ax_i.text(cx, cy - (bbox[3] - bbox[1]) * 0.08, name, color="white", fontsize=7.5,
                  ha="center", va="top", zorder=9,
                  bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6, edgecolor="none"))
        ax_i.set_xlim(extent[0], extent[2])
        ax_i.set_ylim(extent[1], extent[3])
        ax_i.set_aspect("equal")
        style_ax(ax_i)
        ax_i.set_title(f"{chr(ord('A') + i)}. {name} study area", fontsize=9.5, pad=4)
        scalebar_km = args.scalebar_km or (5 if using_defaults else nice_km(bbox_width_km(bbox) * 0.25))
        add_scalebar(ax_i, bbox, scalebar_km)

    fig.suptitle(fig_title, fontsize=13, y=0.965)
    fig.text(0.99, 0.005, "Bounding boxes correspond to the study extents used in the pipeline.",
             ha="right", va="bottom", fontsize=7, color="#444444")

    png = fig_dir / f"{stem}.png"
    pdf = fig_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {png.name} ({png.stat().st_size:,} bytes)")
    print(f"saved {pdf.name} ({pdf.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

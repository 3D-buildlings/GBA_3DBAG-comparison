from __future__ import annotations

import io
import json
import math
import time
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
from PIL import Image
from shapely.geometry import shape

OUT = Path(r"D:\GBA_3DBAG_outputs\figures")
OUT.mkdir(parents=True, exist_ok=True)
TILE_CACHE = Path(r"C:\Users\hamid\AppData\Local\Temp\opencode\eox_tiles")
TILE_CACHE.mkdir(parents=True, exist_ok=True)

TILE_URL = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg"
HEADERS = {"User-Agent": "Mozilla/5.0"}

NETHERLANDS = (3.0, 50.6, 7.4, 53.7)
AMSTERDAM_BBOX = (4.728, 52.278, 5.079, 52.431)
UTRECHT_BBOX = (5.0, 52.03, 5.18, 52.14)
AMSTERDAM = (4.9041, 52.3676)
UTRECHT = (5.1214, 52.0907)


def deg2num(lat, lon, z):
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
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
    lon0, lat0, lon1, lat1 = bbox
    x0, y0 = deg2num(lat1, lon0, z)
    x1, y1 = deg2num(lat0, lon1, z)
    x0, x1 = int(math.floor(x0)), int(math.floor(x1))
    y0, y1 = int(math.floor(y0)), int(math.floor(y1))
    width = (x1 - x0 + 1) * 256
    height = (y1 - y0 + 1) * 256
    canvas = Image.new("RGB", (width, height))
    for ix in range(x0, x1 + 1):
        for iy in range(y0, y1 + 1):
            canvas.paste(fetch_tile(z, ix, iy), ((ix - x0) * 256, (iy - y0) * 256))
    top_lat, left_lon = num2deg(x0, y0, z)
    bottom_lat, right_lon = num2deg(x1 + 1, y1 + 1, z)
    return np.asarray(canvas), (left_lon, right_lon, bottom_lat, top_lat)


def download_country():
    url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
    fp = TILE_CACHE / "countries.geojson"
    if not fp.exists():
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=60) as r:
            fp.write_bytes(r.read())
    data = json.loads(fp.read_text(encoding="utf-8"))
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        name = props.get("ADMIN") or props.get("NAME") or ""
        if str(name).lower() in {"netherlands", "the netherlands"}:
            return shape(feature["geometry"])
    return None


def plot_polygon(ax, geom, **kwargs):
    if geom is None:
        return
    geoms = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    for g in geoms:
        x, y = g.exterior.xy
        ax.plot(x, y, **kwargs)


def add_bbox(ax, bbox, color, label, xytext=None, ha="right", va="bottom", label_xy=None):
    lon0, lat0, lon1, lat1 = bbox
    ax.add_patch(Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0,
                           fill=False, edgecolor=color, linewidth=2.0, zorder=6))
    if label_xy is not None:
        ax.text(
            label_xy[0], label_xy[1], label,
            fontsize=8.0, color="white", fontweight="bold",
            ha="left", va="top", zorder=9,
            bbox=dict(boxstyle="round,pad=0.22", facecolor=color, edgecolor="none", alpha=0.9),
        )
        return
    if xytext is None:
        xytext = (-8, 16 if label.startswith("Amsterdam") else -22)
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


def add_scalebar(ax, bbox, length_km=5):
    lon0, lat0, lon1, lat1 = bbox
    lat_c = (lat0 + lat1) / 2
    km_per_deg_lon = 111.32 * math.cos(math.radians(lat_c))
    d_lon = length_km / km_per_deg_lon
    x = lon1 - d_lon - (lon1 - lon0) * 0.04
    y = lat0 + (lat1 - lat0) * 0.06
    ax.plot([x, x + d_lon], [y, y], color="white", linewidth=3, solid_capstyle="butt", zorder=9)
    ax.plot([x, x + d_lon], [y, y], color="black", linewidth=1.2, solid_capstyle="butt", zorder=10)
    ax.text(x + d_lon / 2, y + (lat1 - lat0) * 0.012, f"{length_km} km",
            color="white", fontsize=7.5, ha="center", va="bottom", zorder=10)


def style_ax(ax):
    ax.set_axis_off()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("white")
        spine.set_linewidth(1.5)


country = download_country()

fig = plt.figure(figsize=(13.5, 9.2), facecolor="white")
gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], wspace=0.04, hspace=0.08,
                      left=0.02, right=0.98, top=0.91, bottom=0.03)

# Netherlands overview
ax = fig.add_subplot(gs[:, 0])
img, ext = mosaic(NETHERLANDS, 8)
ax.imshow(img, extent=ext, origin="upper", zorder=1)
plot_polygon(ax, country, color="white", linewidth=1.2, zorder=4)
add_bbox(ax, AMSTERDAM_BBOX, "#D32F2F", "Amsterdam study area", xytext=(-14, -20), ha="right", va="top")
add_bbox(ax, UTRECHT_BBOX, "#1565C0", "Utrecht study area", xytext=(12, 14), ha="left", va="bottom")
ax.plot(*AMSTERDAM, marker="o", markersize=6, color="white", markeredgecolor="black", zorder=7)
ax.plot(*UTRECHT, marker="o", markersize=6, color="white", markeredgecolor="black", zorder=7)
ax.text(AMSTERDAM[0] + 0.045, AMSTERDAM[1] + 0.015, "Amsterdam", color="white", fontsize=9,
        fontweight="bold", ha="left", va="bottom", zorder=8,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="black", alpha=0.55, edgecolor="none"))
ax.text(UTRECHT[0] + 0.045, UTRECHT[1] - 0.02, "Utrecht", color="white", fontsize=9,
        fontweight="bold", ha="left", va="top", zorder=8,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="black", alpha=0.55, edgecolor="none"))
ax.set_xlim(NETHERLANDS[0], NETHERLANDS[2])
ax.set_ylim(NETHERLANDS[1], NETHERLANDS[3])
ax.set_aspect("equal")
style_ax(ax)
ax.text(0.99, 0.01, "Sentinel-2 cloudless (2020) by EOX IT Services GmbH; contains modified Copernicus Sentinel data",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="white",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.5, edgecolor="none"), zorder=11)
ax.legend(handles=[Patch(facecolor="none", edgecolor="#D32F2F", label="Amsterdam study area"),
                   Patch(facecolor="none", edgecolor="#1565C0", label="Utrecht study area")],
          loc="lower left", fontsize=7.5, framealpha=0.85)

# Amsterdam inset
ax1 = fig.add_subplot(gs[0, 1])
a_bbox = (AMSTERDAM_BBOX[0] - 0.08, AMSTERDAM_BBOX[1] - 0.07,
          AMSTERDAM_BBOX[2] + 0.08, AMSTERDAM_BBOX[3] + 0.07)
img1, ext1 = mosaic(a_bbox, 12)
ax1.imshow(img1, extent=ext1, origin="upper", zorder=1)
add_bbox(ax1, AMSTERDAM_BBOX, "#D32F2F", "Amsterdam study area", label_xy=(AMSTERDAM_BBOX[0] + 0.006, AMSTERDAM_BBOX[3] - 0.004))
ax1.plot(*AMSTERDAM, marker="o", markersize=5, color="white", markeredgecolor="black", zorder=8)
ax1.text(AMSTERDAM[0], AMSTERDAM[1] - 0.012, "Amsterdam", color="white", fontsize=7.5,
         ha="center", va="top", zorder=9,
         bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6, edgecolor="none"))
ax1.set_xlim(a_bbox[0], a_bbox[2]); ax1.set_ylim(a_bbox[1], a_bbox[3])
ax1.set_aspect("equal"); style_ax(ax1)
ax1.set_title("A. Amsterdam study area", fontsize=9.5, pad=4)
add_scalebar(ax1, AMSTERDAM_BBOX, 5)

# Utrecht inset
ax2 = fig.add_subplot(gs[1, 1])
u_bbox = (UTRECHT_BBOX[0] - 0.06, UTRECHT_BBOX[1] - 0.05,
          UTRECHT_BBOX[2] + 0.06, UTRECHT_BBOX[3] + 0.05)
img2, ext2 = mosaic(u_bbox, 12)
ax2.imshow(img2, extent=ext2, origin="upper", zorder=1)
add_bbox(ax2, UTRECHT_BBOX, "#1565C0", "Utrecht study area", label_xy=(UTRECHT_BBOX[0] + 0.004, UTRECHT_BBOX[3] - 0.003))
ax2.plot(*UTRECHT, marker="o", markersize=5, color="white", markeredgecolor="black", zorder=8)
ax2.text(UTRECHT[0], UTRECHT[1] - 0.009, "Utrecht", color="white", fontsize=7.5,
         ha="center", va="top", zorder=9,
         bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6, edgecolor="none"))
ax2.set_xlim(u_bbox[0], u_bbox[2]); ax2.set_ylim(u_bbox[1], u_bbox[3])
ax2.set_aspect("equal"); style_ax(ax2)
ax2.set_title("B. Utrecht study area", fontsize=9.5, pad=4)
add_scalebar(ax2, UTRECHT_BBOX, 5)

fig.suptitle("Study areas used in the GBA vs 3DBAG comparison", fontsize=13, y=0.965)
fig.text(0.99, 0.005, "Bounding boxes correspond to the study extents used in the pipeline.",
         ha="right", va="bottom", fontsize=7, color="#444444")

png = OUT / "study_areas_netherlands.png"
pdf = OUT / "study_areas_netherlands.pdf"
fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(pdf, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(png)
print(pdf)
print("size", png.stat().st_size, pdf.stat().st_size)


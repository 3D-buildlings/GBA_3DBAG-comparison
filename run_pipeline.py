#!/usr/bin/env python3
"""Reproduce the GBA vs 3DBAG (Amsterdam + Utrecht) deliverables.

The pipeline:
1. downloads the pinned 3DBAG LoD1.2 CityJSON tiles,
2. extracts per-building 3DBAG site height (z_roof - z_base) and b3_opp_grond,
3. fetches the GBA LoD1 footprints and heights (or reads a local GBA file),
4. builds the GBA / 3DBAG clean and raw GeoJSON deliverables,
5. runs the GBA vs 3DBAG comparison and writes all summary files and zips.

All outputs are written under --out (for Colab, point this to Google Drive).
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import math
import os
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from pyproj import Transformer
from shapely import make_valid
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import pipeline_lib as p  # noqa: E402

CITIES = {
    "amsterdam": (4.728, 52.278, 5.079, 52.431),
    "utrecht": (5.0, 52.03, 5.18, 52.14),
}
CITY_LABEL = {"amsterdam": "Amsterdam", "utrecht": "Utrecht"}
TDBAG_VERSION = "v20250903"
TDBAG_RELEASE = "v2025.09.03"
TILE_INDEX_URL = f"https://data.3dbag.nl/{TDBAG_VERSION}/tile_index.fgb"
MIN_AREA_M2 = 1.0

REF_DEF = "3DBAG LoD1.2 site height (z_roof - z_base)"
CMP_DEF = "GBA own height (height attribute)"
NOTES = (
    "GBA vs 3DBAG only. 3DBAG reference uses the LoD1.2 site heights "
    "(z_roof - z_base) and b3_opp_grond areas; GBA uses its own height attribute. "
    "Main comparison GeoJSONs use 3DBAG geometry; false positives are separate files."
)


def log(msg: str = "") -> None:
    print(msg, flush=True)


def round_half_up(value: float, ndigits: int) -> float:
    factor = 10 ** ndigits
    return math.floor(value * factor + 0.5) / factor


def _r(value, ndigits: int, half: bool = False):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    number = float(value)
    return round_half_up(number, ndigits) if half else round(number, ndigits)


def city_boxes_rd() -> dict[str, Polygon]:
    tr = Transformer.from_crs(p.WGS84, p.WORK_CRS, always_xy=True)
    out = {}
    for name, bb in CITIES.items():
        x0, y0 = tr.transform(bb[0], bb[1])
        x1, y1 = tr.transform(bb[2], bb[3])
        out[name] = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    return out


def _surface_polygon(surface, xy):
    if not surface or not surface[0]:
        return None
    shell = [xy[i] for i in surface[0] if 0 <= i < len(xy)]
    if len(shell) < 3:
        return None
    holes = []
    for ring in surface[1:]:
        pts = [xy[i] for i in ring if 0 <= i < len(xy)]
        if len(pts) >= 3:
            holes.append(pts)
    try:
        poly = Polygon(shell, holes)
    except Exception:
        return None
    if not poly.is_valid:
        poly = make_valid(poly)
    if poly.is_empty or poly.area <= 0:
        return None
    return poly


def _footprint(obj, xy):
    geoms = obj.get("geometry") or []
    lod0 = [g for g in geoms if str(g.get("lod")) == "0"]
    candidates = lod0 if lod0 else geoms
    for geom in candidates:
        type_ = geom.get("type")
        polys = []
        if type_ == "MultiSurface":
            surfaces = geom.get("boundaries") or []
        elif type_ == "Solid":
            surfaces = [s for shell in (geom.get("boundaries") or []) for s in shell]
        else:
            continue
        for surface in surfaces:
            poly = _surface_polygon(surface, xy)
            if poly is not None:
                polys.append(poly)
        if not polys:
            continue
        try:
            merged = unary_union([make_valid(x) for x in polys])
        except Exception:
            merged = unary_union(polys)
        if isinstance(merged, (Polygon, MultiPolygon)) and not merged.is_empty:
            return merged
        if isinstance(merged, GeometryCollection):
            parts = [g for g in merged.geoms if isinstance(g, (Polygon, MultiPolygon))]
            if parts:
                merged = unary_union(parts)
                if isinstance(merged, (Polygon, MultiPolygon)) and not merged.is_empty:
                    return merged
    return None


def _collect_indices(target: set[int], value) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        target.add(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_indices(target, item)


def _lod12_z(obj, objects, vertices, transform):
    sx, sy, sz = transform["scale"]
    tx, ty, tz = transform["translate"]
    nodes = [obj]
    seen = set()
    stack = list(obj.get("children") or [])
    while stack:
        key = stack.pop()
        if key in seen:
            continue
        seen.add(key)
        child = objects.get(key)
        if not child:
            continue
        nodes.append(child)
        stack.extend(child.get("children") or [])
    idx = set()
    for node in nodes:
        for geom in node.get("geometry") or []:
            if str(geom.get("lod")) == "1.2":
                _collect_indices(idx, geom.get("boundaries") or [])
    if not idx:
        return None, None
    zs = [tz + sz * vertices[i][2] for i in idx if 0 <= i < len(vertices)]
    if not zs:
        return None, None
    return min(zs), max(zs)


def _city_for(geom, city_boxes):
    centroid = geom.centroid
    for name, b in city_boxes.items():
        if b.contains(centroid):
            return name
    best = None
    best_area = 0.0
    for name, b in city_boxes.items():
        if geom.intersects(b):
            try:
                area = float(geom.intersection(b).area)
            except Exception:
                area = 0.0
            if area > best_area:
                best = name
                best_area = area
    return best


def select_tiles(index_path: Path) -> pd.DataFrame:
    idx = gpd.read_file(index_path)
    city_boxes = city_boxes_rd()
    parts = []
    for b in city_boxes.values():
        parts.append(idx[idx.geometry.intersects(b)])
    tiles = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True), geometry="geometry", crs=idx.crs
    ).drop_duplicates("tile_id")
    log(f"3DBAG tiles selected: {len(tiles)}")
    return tiles[["tile_id", "cj_download"]].reset_index(drop=True)


def download(url: str, dest: Path, retries: int = 4) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, dest)
            return True
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return False


def download_tiles(tiles: pd.DataFrame, work: Path, threads: int = 8) -> list[Path]:
    tile_dir = work / "3dbag_tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for row in tiles.itertuples(index=False):
        dest = tile_dir / Path(row.cj_download).name
        jobs.append((row.cj_download, dest))
    done = 0
    failed = []
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = {pool.submit(download, url, dest): dest for url, dest in jobs}
        for fut in as_completed(futures):
            dest = futures[fut]
            ok = fut.result()
            done += 1
            if not ok:
                failed.append(str(dest))
            if done % 50 == 0:
                log(f"  downloaded {done}/{len(jobs)} tiles")
    if failed:
        raise RuntimeError(f"failed to download {len(failed)} 3DBAG tiles")
    log(f"3DBAG tiles downloaded: {len(jobs)}")
    return [dest for _, dest in jobs]


def parse_3dbag(tile_files: list[Path], city_boxes) -> gpd.GeoDataFrame:
    rows = []
    for n, path in enumerate(tile_files, 1):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        transform = data.get("transform") or {}
        try:
            sx, sy, sz = transform["scale"]
            tx, ty, tz = transform["translate"]
        except Exception:
            continue
        objects = data.get("CityObjects") or {}
        vertices = data.get("vertices") or []
        xy = [(tx + sx * v[0], ty + sy * v[1]) for v in vertices]
        for cid, obj in objects.items():
            if obj.get("type") != "Building":
                continue
            geom = _footprint(obj, xy)
            if geom is None:
                continue
            if not any(b.intersects(geom) for b in city_boxes.values()):
                continue
            city = _city_for(geom, city_boxes)
            if city is None:
                continue
            attrs = obj.get("attributes") or {}
            zmin, zmax = _lod12_z(obj, objects, vertices, transform)
            area = float(geom.area)
            if area < MIN_AREA_M2:
                continue
            rows.append(
                {
                    "id": cid,
                    "city": CITY_LABEL[city],
                    "zmin": zmin,
                    "zmax": zmax,
                    "h50": attrs.get("b3_h_dak_50p"),
                    "ground": attrs.get("b3_h_maaiveld"),
                    "opp_grond": attrs.get("b3_opp_grond"),
                    "area": area,
                    "geometry": geom,
                }
            )
        if n % 25 == 0 or n == len(tile_files):
            log(f"  parsed {n}/{len(tile_files)} tiles, buildings={len(rows)}")
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=p.WORK_CRS)
    gdf = gdf.drop_duplicates("id").reset_index(drop=True)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].reset_index(drop=True)
    log(f"3DBAG buildings in study cities: {len(gdf)}")
    return gdf


def build_3dbag_layers(gdf: gpd.GeoDataFrame, out_raw: Path):
    g = gdf.copy()
    site_h = g["zmax"] - g["zmin"]
    orig_h = g["h50"] - g["ground"]
    matched = site_h.notna() & g["opp_grond"].notna()
    no_lod12 = site_h.isna()
    site_match = np.where(matched, "matched", np.where(no_lod12, "no_lod12", "no_site_area"))

    height_m = np.where(matched, site_h, orig_h)
    height_nap = np.where(matched, g["zmax"], g["h50"])
    area_m2 = np.where(matched, g["opp_grond"], g["area"])
    height_m = [_r(v, 1 if m else 3, bool(m)) for v, m in zip(height_m, matched)]
    height_nap = [_r(v, 3) for v in height_nap]
    area_m2 = [_r(v, 2 if m else 3) for v, m in zip(area_m2, matched)]
    clean = gpd.GeoDataFrame(
        {
            "id": g["id"],
            "height_m": height_m,
            "height_nap_m": height_nap,
            "area_m2": area_m2,
            "city": g["city"],
            "orig_height_m": [_r(v, 3) for v in orig_h],
            "orig_height_nap_m": [_r(v, 3) for v in g["h50"]],
            "orig_area_m2": [_r(v, 3) for v in g["area"]],
            "site_ground_nap_m": [_r(v, 3) for v in g["zmin"]],
            "site_3dbag_id": np.where(matched, g["id"], None),
            "site_match": site_match,
        },
        geometry=g.geometry,
        crs=p.WORK_CRS,
    )
    clean_path = out_raw / "3DBAG_Amsterdam_Utrecht_lod12_clean.geojson"
    pyogrio.write_dataframe(
        clean.to_crs(p.WGS84), clean_path, driver="GeoJSON",
        layer_options={"COORDINATE_PRECISION": 6},
    )
    log(f"wrote {clean_path.name}: {len(clean)} features")

    ref = gpd.GeoDataFrame(
        {
            "ref_id": g["id"].astype(str),
            "ref_bag_id": g["id"].astype(str),
            "ref_source": "3DBAG",
            "height_3dbag_m": clean["height_m"],
            "height_3dbag_50p_m": clean["height_m"],
            "height_3dbag_max_m": clean["height_m"],
            "height_3dbag_definition": REF_DEF,
            "height_3dbag_max_definition": REF_DEF,
            "h_roof50_3dbag_m": clean["height_nap_m"],
            "h_roofmax_3dbag_m": clean["height_nap_m"],
            "h_ground_3dbag_m": clean["site_ground_nap_m"],
            "h_nok_3dbag_m": None,
            "ref_area_m2": clean["orig_area_m2"],
            "city": clean["city"],
        },
        geometry=clean.geometry,
        crs=p.WORK_CRS,
    )
    site_map = {
        str(row.id): {
            "height": row.height_m,
            "height_nap": row.height_nap_m,
            "area": row.area_m2,
            "ground": row.site_ground_nap_m,
        }
        for row in clean.itertuples(index=False)
        if row.site_match == "matched"
    }
    return clean, ref, site_map


def _tile_bboxes(bbox, step):
    xmin, ymin, xmax, ymax = bbox
    nx = max(1, int(math.ceil((xmax - xmin) / step)))
    ny = max(1, int(math.ceil((ymax - ymin) / step)))
    out = []
    for ix in range(nx):
        for iy in range(ny):
            x0 = xmin + ix * step
            y0 = ymin + iy * step
            out.append((x0, y0, min(x0 + step, xmax), min(y0 + step, ymax)))
    return out


def fetch_gba_tiled(work: Path, step: float) -> gpd.GeoDataFrame:
    p.MAX_GBA_FEATURES = 3_000_000
    tile_dir = work / "gba_tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for city, bbox in CITIES.items():
        tiles = _tile_bboxes(bbox, step)
        log(f"GBA {city}: {len(tiles)} tiles (step={step} deg)")
        for i, tile in enumerate(tiles):
            part = tile_dir / f"{city}_{i:04d}.gpkg"
            if part.exists() and part.stat().st_size > 0:
                g = pyogrio.read_dataframe(part)
            else:
                g = None
                for attempt in range(3):
                    try:
                        g = p.fetch_gba(tile)
                        break
                    except RuntimeError as exc:
                        if "no usable building footprints" in str(exc):
                            g = None
                            break
                        time.sleep(3 * (attempt + 1))
                if g is not None and len(g):
                    pyogrio.write_dataframe(g, part, driver="GPKG", layer="buildings")
                else:
                    log(f"  [WARN] GBA tile {city} #{i} returned no usable buildings")
            if g is not None and len(g):
                g["city"] = CITY_LABEL[city]
                frames.append(g)
            if (i + 1) % 10 == 0 or i + 1 == len(tiles):
                log(f"  {city}: {i + 1}/{len(tiles)} tiles, rows={sum(len(x) for x in frames)}")
    if not frames:
        raise RuntimeError("GBA returned no buildings")
    gba = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=p.WORK_CRS)
    gba = gba.drop_duplicates("cmp_id").reset_index(drop=True)
    gba = gba[gba.geometry.notna() & ~gba.geometry.is_empty].reset_index(drop=True)
    gba = gba[gba.geometry.area >= MIN_AREA_M2].reset_index(drop=True)
    log(f"GBA buildings: {len(gba)}")
    return gba


def normalize_gba(gdf: gpd.GeoDataFrame, city_boxes) -> gpd.GeoDataFrame:
    g = gdf.to_crs(p.WORK_CRS).copy()
    id_col = next((c for c in ("cmp_id", "id", "fid") if c in g.columns), None)
    if id_col is None:
        g["cmp_id"] = [f"GBA_{i}" for i in range(len(g))]
        id_col = "cmp_id"
    height_col = None
    for c in ("orig_height_m", "height_compared_m", "height_m", "height"):
        if c in g.columns and g[c].notna().any():
            height_col = c
            break
    if height_col is None:
        g["height_compared_m"] = None
        height_col = "height_compared_m"
    if "city" not in g.columns:
        g["city"] = [CITY_LABEL.get(_city_for(geom, city_boxes), None) for geom in g.geometry]
    else:
        fallback = [CITY_LABEL.get(_city_for(geom, city_boxes), None) for geom in g.geometry]
        g["city"] = g["city"].where(g["city"].notna(), fallback)
    return gpd.GeoDataFrame(
        {
            "cmp_id": g[id_col].astype(str),
            "cmp_source_id": g[id_col].astype(str),
            "cmp_source": "GBA",
            "height_compared_m": pd.to_numeric(g[height_col], errors="coerce"),
            "height_compared_50p_m": pd.to_numeric(g[height_col], errors="coerce"),
            "height_compared_max_m": pd.to_numeric(g[height_col], errors="coerce"),
            "height_compared_method": CMP_DEF,
            "height_compared_definition": CMP_DEF,
            "h_roof50_cmp_m": None,
            "h_ground_cmp_m": None,
            "h_roof70_cmp_m": None,
            "h_roofmax_cmp_m": None,
            "year_built_cmp": None,
            "status_cmp": None,
            "n_parts_cmp": 1,
            "gba_source": None,
            "gba_var": None,
            "cmp_area_m2": g.geometry.area.astype(float),
            "city": g["city"],
        },
        geometry=g.geometry,
        crs=p.WORK_CRS,
    )


def assign_cmp_to_ref(groups: pd.DataFrame, ref: gpd.GeoDataFrame, cmp: gpd.GeoDataFrame) -> dict[str, str]:
    ref_geom = {str(row.ref_id): row.geometry for row in ref.itertuples(index=False)}
    cmp_geom = {str(row.cmp_id): row.geometry for row in cmp.itertuples(index=False)}
    assigned = {}
    for row in groups.itertuples(index=False):
        refs = [r for r in str(row.ref_ids).split(";") if r]
        cmps = [c for c in str(row.cmp_ids).split(";") if c]
        if not refs or not cmps:
            continue
        if len(refs) == 1:
            for c in cmps:
                assigned[c] = refs[0]
            continue
        for c in cmps:
            cg = cmp_geom.get(c)
            best = refs[0]
            best_area = -1.0
            if cg is not None:
                for r in refs:
                    rg = ref_geom.get(r)
                    if rg is None:
                        continue
                    try:
                        area = float(cg.intersection(rg).area)
                    except Exception:
                        area = 0.0
                    if area > best_area:
                        best_area = area
                        best = r
            assigned[c] = best
    return assigned


def build_gba_outputs(
    cmp: gpd.GeoDataFrame,
    site_map: dict,
    assigned: dict[str, str],
    out_raw: Path,
) -> None:
    clean_rows = []
    raw_rows = []
    for row in cmp.itertuples(index=False):
        cid = str(row.cmp_id)
        own_h = None if pd.isna(row.height_compared_m) else float(row.height_compared_m)
        own_area = float(row.cmp_area_m2)
        ref_id = assigned.get(cid)
        site = site_map.get(ref_id) if ref_id else None
        matched = site is not None and site.get("height") is not None and site.get("area") is not None
        no_lod12 = ref_id is not None and not matched
        city = getattr(row, "city", None)
        if matched:
            clean_rows.append(
                {
                    "id": cid,
                    "height_m": round_half_up(float(site["height"]), 1),
                    "height_nap_m": None if site.get("height_nap") is None else round(float(site["height_nap"]), 3),
                    "area_m2": round(float(site["area"]), 2),
                    "city": city,
                    "orig_height_m": None if own_h is None else round(own_h, 3),
                    "orig_height_nap_m": None,
                    "orig_area_m2": round(own_area, 3),
                    "site_ground_nap_m": None if site.get("ground") is None else round(float(site["ground"]), 3),
                    "site_3dbag_id": ref_id,
                    "site_match": "matched",
                }
            )
            raw_rows.append(
                {
                    "id": cid,
                    "height_m": None if own_h is None else round(own_h, 3),
                    "height_nap_m": None,
                    "area_m2": round(own_area, 3),
                    "city": city,
                    "site_height_m": round_half_up(float(site["height"]), 1),
                    "site_area_m2": round(float(site["area"]), 2),
                    "site_height_nap_m": None if site.get("height_nap") is None else round(float(site["height_nap"]), 3),
                    "site_ground_nap_m": None if site.get("ground") is None else round(float(site["ground"]), 3),
                    "site_3dbag_id": ref_id,
                    "site_match": "matched",
                }
            )
        else:
            match = "no_lod12" if no_lod12 else "no_3dbag_match"
            clean_rows.append(
                {
                    "id": cid,
                    "height_m": None if own_h is None else round(own_h, 3),
                    "height_nap_m": None,
                    "area_m2": round(own_area, 3),
                    "city": city,
                    "orig_height_m": None,
                    "orig_height_nap_m": None,
                    "orig_area_m2": None,
                    "site_ground_nap_m": None,
                    "site_3dbag_id": None,
                    "site_match": match,
                }
            )
            raw_rows.append(
                {
                    "id": cid,
                    "height_m": None if own_h is None else round(own_h, 3),
                    "height_nap_m": None,
                    "area_m2": round(own_area, 3),
                    "city": city,
                    "site_height_m": None,
                    "site_area_m2": None,
                    "site_height_nap_m": None,
                    "site_ground_nap_m": None,
                    "site_3dbag_id": None,
                    "site_match": match,
                }
            )
    clean = gpd.GeoDataFrame(clean_rows, geometry=cmp.geometry, crs=p.WORK_CRS)
    raw = gpd.GeoDataFrame(raw_rows, geometry=cmp.geometry, crs=p.WORK_CRS)
    clean["site_3dbag_id"] = clean["site_3dbag_id"].astype(object).where(clean["site_3dbag_id"].notna(), None)
    raw["site_3dbag_id"] = raw["site_3dbag_id"].astype(object).where(raw["site_3dbag_id"].notna(), None)
    clean_path = out_raw / "GBA_Amsterdam_Utrecht_lod12_clean.geojson"
    raw_path = out_raw / "GBA_Amsterdam_Utrecht_raw.geojson"
    pyogrio.write_dataframe(clean.to_crs(p.WGS84), clean_path, driver="GeoJSON", layer_options={"COORDINATE_PRECISION": 6})
    pyogrio.write_dataframe(raw.to_crs(p.WGS84), raw_path, driver="GeoJSON", layer_options={"COORDINATE_PRECISION": 6})
    log(f"wrote {clean_path.name}: {len(clean)} features")
    log(f"wrote {raw_path.name}: {len(raw)} features")


def main() -> None:
    parser = argparse.ArgumentParser(description="GBA vs 3DBAG pipeline")
    parser.add_argument("--out", required=True, help="output directory (use Google Drive in Colab)")
    parser.add_argument("--work", default=None, help="working/cache directory (default: <out>/work)")
    parser.add_argument("--gba-file", default=None, help="optional local GBA footprints with height attribute")
    parser.add_argument("--gba-tile-step", type=float, default=0.03, help="GBA fetch tile size in degrees")
    parser.add_argument("--threads", type=int, default=8, help="3DBAG tile download threads")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    work = Path(args.work).resolve() if args.work else out / "work"
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GBA_3DBAG_PROJECT_DIR", str(work))

    city_boxes = city_boxes_rd()

    index_path = work / "tile_index.fgb"
    if not index_path.exists():
        log("downloading 3DBAG tile index ...")
        if not download(TILE_INDEX_URL, index_path):
            raise RuntimeError("could not download the 3DBAG tile index")
    tiles = select_tiles(index_path)
    tile_files = download_tiles(tiles, work, args.threads)

    buildings = parse_3dbag(tile_files, city_boxes)
    clean_ref, ref, site_map = build_3dbag_layers(buildings, raw_dir)
    del buildings, clean_ref
    gc.collect()

    if args.gba_file:
        log(f"reading GBA from {args.gba_file}")
        cmp = normalize_gba(gpd.read_file(args.gba_file), city_boxes)
    else:
        cmp = fetch_gba_tiled(work, args.gba_tile_step)
        cmp = gpd.GeoDataFrame(cmp, geometry="geometry", crs=p.WORK_CRS)

    log("running GBA vs 3DBAG matching ...")
    p.SLIM_OUTPUT = True
    res = p.compare_datasets(ref, cmp, "GBA_vs_3DBAG", height_mode="50p")
    counts = res["counts"]
    groups = res["groups"]
    assigned = assign_cmp_to_ref(groups, ref, cmp)
    for key in ("pairs", "reference", "compared"):
        res.pop(key, None)
    gc.collect()

    out.mkdir(parents=True, exist_ok=True)
    p.write_geojson(res["main"], out / "GBA_vs_3DBAG.geojson")
    p.write_geojson(res["false_positives"], out / "GBA_vs_3DBAG_false_positives.geojson")
    if groups is not None and not groups.empty:
        structural = groups[
            (pd.to_numeric(groups["n_3dbag_in_group"], errors="coerce") > 1)
            | (pd.to_numeric(groups["n_compared_in_group"], errors="coerce") > 1)
        ].copy()
    else:
        structural = groups
    structural.to_csv(out / "GBA_vs_3DBAG_structural_groups.csv", index=False, encoding="utf-8-sig")
    res.pop("main", None)
    res.pop("false_positives", None)
    res.pop("groups", None)
    del groups, structural, ref
    gc.collect()

    union_bbox = (
        min(v[0] for v in CITIES.values()), min(v[1] for v in CITIES.values()),
        max(v[2] for v in CITIES.values()), max(v[3] for v in CITIES.values()),
    )
    area_km2 = sum(p.bbox_area_km2(v) for v in CITIES.values())
    wide, long = p.build_summary_tables([{"comparison": "GBA_vs_3DBAG", "counts": counts}], union_bbox, area_km2)
    wide["height_3dbag_definition"] = REF_DEF
    wide["height_compared_definition"] = CMP_DEF
    wide["height_mode_used"] = "site_lod12_reference"
    wide["notes"] = NOTES
    summary_path = out / "comparison_summary.csv"
    breakdown_path = out / "comparison_category_breakdown.csv"
    wide.to_csv(summary_path, index=False, encoding="utf-8-sig")
    long.to_csv(breakdown_path, index=False, encoding="utf-8-sig")

    meta = {
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "tdbag_version": TDBAG_RELEASE,
        "cities": {k: list(v) for k, v in CITIES.items()},
        "reference_height_definition": REF_DEF,
        "compared_height_definition": CMP_DEF,
        "results": {"GBA_vs_3DBAG": counts},
        "outputs": {
            "GBA_vs_3DBAG": str(out / "GBA_vs_3DBAG.geojson"),
            "GBA_vs_3DBAG_false_positives": str(out / "GBA_vs_3DBAG_false_positives.geojson"),
            "GBA_vs_3DBAG_structural_groups_csv": str(out / "GBA_vs_3DBAG_structural_groups.csv"),
            "comparison_summary_csv": str(summary_path),
            "comparison_category_breakdown_csv": str(breakdown_path),
        },
        "notes": NOTES,
    }
    meta_path = out / "run_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    build_gba_outputs(cmp, site_map, assigned, raw_dir)

    with zipfile.ZipFile(out / "comparison_outputs.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in (
            "GBA_vs_3DBAG.geojson",
            "GBA_vs_3DBAG_false_positives.geojson",
            "GBA_vs_3DBAG_structural_groups.csv",
            "comparison_summary.csv",
            "comparison_category_breakdown.csv",
            "run_metadata.json",
        ):
            zf.write(out / name, arcname=name)
    with zipfile.ZipFile(raw_dir / "lod12_clean.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in ("3DBAG_Amsterdam_Utrecht_lod12_clean.geojson", "GBA_Amsterdam_Utrecht_lod12_clean.geojson"):
            zf.write(raw_dir / name, arcname=name)

    log(f"matched reference: {counts['matched_reference_count']}")
    log(f"false negatives: {counts['false_negative_count']}, false positives: {counts['false_positive_count']}")
    log(f"height MAE: {counts['height_mae_m']}, RMSE: {counts['height_rmse_m']}, MAPE: {counts['height_mape_pct']}")
    log(f"outputs written to {out}")


if __name__ == "__main__":
    main()





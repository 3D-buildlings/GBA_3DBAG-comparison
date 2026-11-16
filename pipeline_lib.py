from __future__ import annotations

import json
import math
import os
import shutil
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import requests
from pyproj import Transformer
from shapely import STRtree, force_2d, make_valid
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box, shape
from shapely.ops import unary_union

WGS84 = "EPSG:4326"
WORK_CRS = "EPSG:28992"
TDBAG_CRS = "EPSG:7415"

NETHERLANDS_BBOX = (3.36, 50.75, 7.23, 53.55)
BBOX_PRESETS = {
    "delft_demo": (4.355, 52.000, 4.362, 52.005),
    "utrecht_demo": (5.115, 52.085, 5.125, 52.095),
    "netherlands": NETHERLANDS_BBOX,
}

BBOX_NAME = "delft_demo"
STUDY_BBOX = BBOX_PRESETS[BBOX_NAME]
MAX_STUDY_AREA_KM2 = 25.0
ALLOW_LARGE_STUDY_AREA = False

MIN_BUILDING_AREA_M2 = 1.0
MIN_INTERSECTION_AREA_M2 = 0.5
PAIR_IOU_MIN = 0.10
CONTAINMENT_OVERLAP_MIN = 0.50
CENTROID_MAX_DIST_M = 10.0
CENTROID_MIN_IOU = 0.05
IOU_FN_THRESHOLD = 0.10

TDBAG_CELL_SIZE_M = 250.0
TDBAG_MIN_CELL_SIZE_M = 25.0
TDBAG_MAX_REQUESTS = 500
MAX_GBA_FEATURES = 50000

PROJECT_DIR = Path(os.environ.get("GBA_3DBAG_PROJECT_DIR", Path.cwd() / "gba_3dbag_work"))
CACHE_DIR = PROJECT_DIR / "cache"
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}

CATEGORY_META = {
    "matched_1to1": {
        "label": "1:1 matched",
        "color": "#2E7D32",
        "description": "One 3DBAG building corresponds to one compared polygon.",
    },
    "many_to_one_3dbag_to_compared": {
        "label": "Many 3DBAG -> one compared (merged)",
        "color": "#1565C0",
        "description": "Several 3DBAG buildings are represented by one merged compared-dataset polygon.",
    },
    "one_to_many_3dbag_to_compared": {
        "label": "One 3DBAG -> many compared (split)",
        "color": "#6A1B9A",
        "description": "One 3DBAG building is split across several compared-dataset polygons.",
    },
    "many_to_many": {
        "label": "Many-to-many",
        "color": "#00838F",
        "description": "Several 3DBAG buildings and several compared polygons form one overlap group.",
    },
    "false_negative_missed_3dbag": {
        "label": "False negative: 3DBAG building missed",
        "color": "#FF9800",
        "description": "3DBAG building with no significant overlap with any compared polygon; segmentation metrics are 0 and height metrics are null.",
    },
    "false_positive_extra_detection": {
        "label": "False positive: extra compared building",
        "color": "#D32F2F",
        "description": "Compared-dataset polygon with no accepted 3DBAG match; shown in the companion false-positive GeoJSON.",
    },
}

# By default only these 17 properties are written to the GeoJSON deliverables.
# Set SLIM_OUTPUT = False if you also need the full diagnostic property set.
SLIM_OUTPUT = True
SLIM_PROPERTIES = [
    "ref_id",
    "cmp_id_best",
    "match_category",
    "n_3dbag_in_group",
    "n_compared_in_group",
    "height_3dbag_m",
    "height_compared_m",
    "height_diff_m",
    "polygon_iou",
    "f1_score",
    "precision",
    "recall",
]

# Attribute floats are rounded to 3 decimal places. Coordinates must keep
# enough precision for WGS84 footprints: 6 decimals is about 0.1 m.
SLIM_ATTRIBUTE_DECIMALS = 3
SLIM_COORDINATE_DECIMALS = 6
SLIM_METRIC_DECIMALS = 6
SLIM_METRIC_FIELDS = {"polygon_iou", "f1_score", "precision", "recall"}


def log(msg=""):
    print(msg, flush=True)


def bbox_area_km2(bbox):
    minx, miny, maxx, maxy = bbox
    lat = math.radians((miny + maxy) / 2.0)
    width = (maxx - minx) * 111.320 * math.cos(lat)
    height = (maxy - miny) * 110.574
    return abs(width * height)


def check_study_area():
    area = bbox_area_km2(STUDY_BBOX)
    log(f"Study bbox ({BBOX_NAME}): {STUDY_BBOX}")
    log(f"Approx. bbox area: {area:.3f} km2")
    if area > MAX_STUDY_AREA_KM2 and not ALLOW_LARGE_STUDY_AREA:
        raise ValueError(
            f"The study bbox is about {area:.1f} km2, above the configured limit "
            f"of {MAX_STUDY_AREA_KM2} km2. Reduce the bbox or set "
            "ALLOW_LARGE_STUDY_AREA = True after reading the scaling notes."
        )
    return area


def clean_polygonal(geom):
    if geom is None or geom.is_empty:
        return None
    try:
        geom = force_2d(geom)
        geom = make_valid(geom)
    except Exception:
        try:
            geom = geom.buffer(0)
        except Exception:
            return None
    polys = []
    if isinstance(geom, Polygon):
        polys = [geom]
    elif isinstance(geom, MultiPolygon):
        polys = list(geom.geoms)
    elif isinstance(geom, GeometryCollection):
        for part in geom.geoms:
            if isinstance(part, Polygon):
                polys.append(part)
            elif isinstance(part, MultiPolygon):
                polys.extend(list(part.geoms))
    polys = [p for p in polys if not p.is_empty and p.area > 0]
    if not polys:
        return None
    try:
        return unary_union(polys)
    except Exception:
        return None


def to_float(value):
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def normalize_id(value):
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text or None


def pick_series(df, names, default=None):
    for name in names:
        if name in df.columns:
            return df[name]
    if default is None:
        return pd.Series([None] * len(df), index=df.index)
    return pd.Series([default] * len(df), index=df.index)


def numeric_series(df, names):
    s = pick_series(df, names)
    return pd.to_numeric(s, errors="coerce")


def make_http_session():
    s = requests.Session()
    s.headers.update(HTTP_HEADERS)
    return s


def download_file(session, url, dest, timeout=300, min_bytes=1):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        log(f"  cached: {dest.name} ({dest.stat().st_size:,} bytes)")
        return dest
    tmp = dest.with_name(dest.name + ".part")
    log(f"  downloading: {url}")
    with session.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = 0
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                fh.write(chunk)
                total += len(chunk)
    if total < min_bytes:
        tmp.unlink(missing_ok=True)
        raise IOError(f"Downloaded file is too small ({total} bytes): {url}")
    os.replace(tmp, dest)
    log(f"  saved: {dest.name} ({total:,} bytes)")
    return dest


print("Imports and configuration loaded.")

TDBAG_API = "https://api.3dbag.nl/collections/pand/items"


def _tdbag_surface_polygon(surface, xy):
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
    return poly if poly.is_valid and poly.area > 0 else None


def extract_3dbag_footprint(building, xy):
    """Return a 2D footprint from a 3DBAG CityJSON Building object."""
    geoms = building.get("geometry", []) or []
    if not geoms:
        return None

    lod0 = [g for g in geoms if str(g.get("lod")) == "0"]
    candidates = lod0 if lod0 else sorted(geoms, key=lambda g: float(str(g.get("lod", "9")) or 9))

    for geom in candidates:
        gtype = geom.get("type")
        polys = []
        if gtype == "MultiSurface":
            for surface in geom.get("boundaries", []):
                p = _tdbag_surface_polygon(surface, xy)
                if p is not None:
                    polys.append(p)
        elif gtype == "Solid":
            for shell in geom.get("boundaries", []):
                for surface in shell:
                    p = _tdbag_surface_polygon(surface, xy)
                    if p is not None:
                        polys.append(p)
        if polys:
            try:
                return clean_polygonal(unary_union([make_valid(p) for p in polys]))
            except Exception:
                return None
    return None


def _tdbag_request(session, cell_bounds):
    x0, y0, x1, y1 = cell_bounds
    params = {
        "bbox": f"{x0:.3f},{y0:.3f},{x1:.3f},{y1:.3f}",
        "limit": 1000,
    }
    last_exc = None
    for attempt in range(4):
        try:
            r = session.get(TDBAG_API, params=params, timeout=180)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"3DBAG request failed for cell {cell_bounds}: {last_exc}")


def _parse_tdbag_page(data, store):
    features = data.get("features", []) or []
    if not features:
        return 0
    transform = data.get("metadata", {}).get("transform")
    if not transform:
        raise RuntimeError("3DBAG response has features but no metadata.transform.")
    sx, sy, _sz = transform["scale"]
    tx, ty, _tz = transform["translate"]

    added = 0
    for feature in features:
        ref_id = feature.get("id")
        if not ref_id or ref_id in store:
            continue
        vertices = feature.get("vertices", []) or []
        xy = [(tx + sx * v[0], ty + sy * v[1]) for v in vertices]
        cityobjects = feature.get("CityObjects", {}) or {}
        building = cityobjects.get(ref_id)
        if building is None or building.get("type") != "Building":
            building = next(
                (obj for obj in cityobjects.values() if obj.get("type") == "Building"),
                None,
            )
        if building is None:
            continue
        geom = extract_3dbag_footprint(building, xy)
        if geom is None:
            continue
        attrs = building.get("attributes", {}) or {}
        roof50 = to_float(attrs.get("b3_h_dak_50p"))
        roofmax = to_float(attrs.get("b3_h_dak_max"))
        ground = to_float(attrs.get("b3_h_maaiveld"))
        height = None
        if roof50 is not None and ground is not None and roof50 > ground:
            height = roof50 - ground
        height_max = None
        if roofmax is not None and ground is not None and roofmax > ground:
            height_max = roofmax - ground
        store[ref_id] = {
            "ref_id": ref_id,
            "ref_bag_id": normalize_id(attrs.get("identificatie")) or ref_id,
            "ref_source": "3DBAG",
            "height_3dbag_m": height,
            "height_3dbag_method": "b3_h_dak_50p - b3_h_maaiveld",
            "height_3dbag_50p_m": height,
            "height_3dbag_max_m": height_max,
            "height_3dbag_definition": "b3_h_dak_50p - b3_h_maaiveld",
            "height_3dbag_max_definition": "b3_h_dak_max - b3_h_maaiveld",
            "h_roof50_3dbag_m": roof50,
            "h_roofmax_3dbag_m": roofmax,
            "h_ground_3dbag_m": ground,
            "h_nok_3dbag_m": to_float(attrs.get("b3_h_nok")),
            "bouwjaar_3dbag": attrs.get("oorspronkelijkbouwjaar"),
            "status_3dbag": attrs.get("status"),
            "b3_bouwlagen": attrs.get("b3_bouwlagen"),
            "b3_dak_type": attrs.get("b3_dak_type"),
            "b3_kwaliteitsindicator": attrs.get("b3_kwaliteitsindicator"),
            "geometry": geom,
        }
        added += 1
    return added


def fetch_3dbag(bbox_wgs84, cell_size=TDBAG_CELL_SIZE_M):
    """Fetch 3DBAG footprints for a WGS84 bbox via cell-wise API requests.

    The public OGC API caps responses and its CityObject-based offset pagination
    can repeat features. We therefore subdivide the study extent into small
    cells, accept only responses that are complete, and de-duplicate by BAG id.
    """
    transformer = Transformer.from_crs(WGS84, TDBAG_CRS, always_xy=True)
    xmin, ymin = transformer.transform(bbox_wgs84[0], bbox_wgs84[1])
    xmax, ymax = transformer.transform(bbox_wgs84[2], bbox_wgs84[3])
    xmin, xmax = sorted([xmin, xmax])
    ymin, ymax = sorted([ymin, ymax])

    def make_cells(bounds, size):
        x0, y0, x1, y1 = bounds
        nx = max(1, int(math.ceil((x1 - x0) / size)))
        ny = max(1, int(math.ceil((y1 - y0) / size)))
        cells = []
        for ix in range(nx):
            for iy in range(ny):
                cx0 = x0 + ix * size
                cy0 = y0 + iy * size
                cells.append((cx0, cy0, min(cx0 + size, x1), min(cy0 + size, y1)))
        return cells

    queue = make_cells((xmin, ymin, xmax, ymax), cell_size)
    session = make_http_session()
    session.headers["Accept"] = "application/city+json, application/json"
    store = {}
    request_count = 0
    subdivided = 0
    cityobject_matched_total = 0

    log(f"Fetching 3DBAG in {len(queue)} initial cell(s) of {cell_size:.0f} m ...")
    while queue:
        bounds = queue.pop(0)
        request_count += 1
        if request_count > TDBAG_MAX_REQUESTS:
            raise RuntimeError(
                f"3DBAG request guard exceeded ({TDBAG_MAX_REQUESTS}). "
                "Reduce the study bbox or increase TDBAG_MAX_REQUESTS deliberately."
            )
        data = _tdbag_request(session, bounds)
        matched = int(data.get("numberMatched") or 0)
        returned = int(data.get("numberReturned") or 0)
        features = data.get("features", []) or []
        cityobject_matched_total += matched
        added = _parse_tdbag_page(data, store)

        if matched > 0 and returned < matched:
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            if max(width, height) > TDBAG_MIN_CELL_SIZE_M:
                cx = (bounds[0] + bounds[2]) / 2.0
                cy = (bounds[1] + bounds[3]) / 2.0
                queue.extend(
                    [
                        (bounds[0], bounds[1], cx, cy),
                        (cx, bounds[1], bounds[2], cy),
                        (bounds[0], cy, cx, bounds[3]),
                        (cx, cy, bounds[2], bounds[3]),
                    ]
                )
                subdivided += 1
            else:
                raise RuntimeError(
                    "A 3DBAG cell is still capped at the minimum cell size. "
                    f"Cell={bounds}, matched={matched}, returned={returned}."
                )
        elif request_count % 10 == 0:
            log(
                f"  3DBAG requests={request_count} cells_in_queue={len(queue)} "
                f"unique_buildings={len(store)}"
            )
        time.sleep(0.05)

    log(
        f"3DBAG requests: {request_count}; subdivided cells: {subdivided}; "
        f"unique buildings: {len(store)}; API matched CityObjects: {cityobject_matched_total}"
    )

    if not store:
        raise RuntimeError("No 3DBAG buildings were returned for the study bbox.")

    gdf = gpd.GeoDataFrame(list(store.values()), geometry="geometry", crs=TDBAG_CRS)
    gdf = gdf.to_crs(WORK_CRS)
    gdf["geometry"] = gdf.geometry.apply(clean_polygonal)
    gdf = gdf[gdf.geometry.notna()].copy()
    study_box_work = box(
        *Transformer.from_crs(WGS84, WORK_CRS, always_xy=True).transform_bounds(
            bbox_wgs84[0], bbox_wgs84[1], bbox_wgs84[2], bbox_wgs84[3]
        )
    )
    gdf = gdf[gdf.geometry.intersects(study_box_work)].copy()
    gdf["ref_area_m2"] = gdf.geometry.area
    gdf = gdf[gdf["ref_area_m2"] >= MIN_BUILDING_AREA_M2].copy()
    gdf = gdf.drop_duplicates("ref_id").reset_index(drop=True)
    invalid_h = int(gdf["height_3dbag_m"].isna().sum())
    log(
        f"3DBAG reference: {len(gdf)} buildings in study bbox; "
        f"{invalid_h} without a valid above-ground height."
    )
    return gdf

GBA_BASE = "https://tubvsig-so2sat-vm1.srv.mwn.de"
GBA_WFS = GBA_BASE + "/geoserver/ows"


def _new_gba_session():
    session = make_http_session()
    session.headers.update(
        {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": GBA_BASE + "/",
            "Origin": GBA_BASE,
        }
    )
    r = session.get(GBA_BASE + "/viewer-session", timeout=60)
    r.raise_for_status()
    token = r.json().get("viewerSession")
    if not token:
        raise RuntimeError("GBA viewer session did not return a viewerSession token.")
    return session, token


def fetch_gba(bbox_wgs84):
    """Fetch GBA LoD1 footprints and heights from the official viewer WFS."""
    session, token = _new_gba_session()
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": "global3D:lod1_global",
        "outputFormat": "application/json",
        "srsName": "EPSG:4326",
        "bbox": f"{bbox_wgs84[0]},{bbox_wgs84[1]},{bbox_wgs84[2]},{bbox_wgs84[3]},EPSG:4326",
        "viewerSession": token,
    }

    log("Fetching GBA LoD1 via WFS ...")
    r = session.get(GBA_WFS, params=params, timeout=300)
    if r.status_code == 403:
        log("  GBA session rejected (403); requesting a fresh session and retrying once.")
        session, token = _new_gba_session()
        params["viewerSession"] = token
        r = session.get(GBA_WFS, params=params, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"GBA WFS HTTP {r.status_code}: {r.text[:500]}")
    try:
        data = r.json()
    except Exception as exc:
        raise RuntimeError(f"GBA WFS did not return JSON: {r.text[:500]}") from exc

    features = data.get("features", []) or []
    if len(features) > MAX_GBA_FEATURES:
        raise RuntimeError(
            f"GBA returned {len(features)} features, above MAX_GBA_FEATURES={MAX_GBA_FEATURES}. "
            "Reduce the study bbox."
        )
    rows = []
    for idx, feature in enumerate(features):
        geom_raw = feature.get("geometry")
        if not geom_raw:
            continue
        try:
            geom = shape(geom_raw)
        except Exception:
            continue
        geom = clean_polygonal(geom)
        if geom is None:
            continue
        props = feature.get("properties", {}) or {}
        ogc_fid = props.get("ogc_fid", feature.get("id", idx))
        height = to_float(props.get("height"))
        rows.append(
            {
                "cmp_id": f"GBA_{normalize_id(ogc_fid) or idx}",
                "cmp_source_id": normalize_id(props.get("id")),
                "cmp_source": "GBA",
                "height_compared_m": height,
                "height_compared_method": "GBA LoD1 height attribute",
                "height_compared_50p_m": None,
                "height_compared_max_m": height,
                "height_compared_definition": "GBA height attribute (maximum within footprint)",
                "h_roof50_cmp_m": None,
                "h_ground_cmp_m": None,
                "h_roof70_cmp_m": None,
                "h_roofmax_cmp_m": None,
                "year_built_cmp": None,
                "status_cmp": props.get("source"),
                "gba_source": props.get("source"),
                "gba_var": to_float(props.get("var")),
                "n_parts_cmp": 1,
                "geometry": geom,
            }
        )

    if not rows:
        raise RuntimeError("GBA returned no usable building footprints for the study bbox.")
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=WGS84)
    study_box = box(*bbox_wgs84)
    gdf = gdf[gdf.geometry.intersects(study_box)].copy()
    gdf = gdf.to_crs(WORK_CRS)
    gdf["cmp_area_m2"] = gdf.geometry.area
    gdf = gdf[gdf["cmp_area_m2"] >= MIN_BUILDING_AREA_M2].copy()
    gdf = gdf.drop_duplicates("cmp_id").reset_index(drop=True)
    valid_h = int(gdf["height_compared_m"].notna().sum())
    log(f"GBA compared dataset: {len(gdf)} polygons; {valid_h} with a height value.")
    return gdf

def _wavg(values, weights):
    values = pd.Series(pd.to_numeric(values, errors="coerce")).reset_index(drop=True)
    weights = pd.Series(pd.to_numeric(weights, errors="coerce")).reset_index(drop=True)
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return None
    return float((values[mask] * weights[mask]).sum() / weights[mask].sum())


def _pair_f1(precision, recall):
    if precision is None or recall is None:
        return None
    return 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def _pair_metrics(pair):
    if pair is None:
        return {
            "pair_iou_best": None,
            "pair_precision_best": None,
            "pair_recall_best": None,
            "pair_f1_best": None,
        }
    return {
        "pair_iou_best": pair["iou"],
        "pair_precision_best": pair["precision"],
        "pair_recall_best": pair["recall"],
        "pair_f1_best": pair["f1"],
    }


def _height_metrics(h_ref, h_cmp):
    out = {
        "height_diff_m": None,
        "height_abs_error_m": None,
        "height_sq_error_m2": None,
        "height_mae_m": None,
        "height_rmse_m": None,
        "height_diff_pct": None,
        "height_abs_diff_pct": None,
        "height_metrics_valid": False,
    }
    if h_ref is None or h_cmp is None:
        return out
    if not (math.isfinite(h_ref) and math.isfinite(h_cmp)) or h_ref <= 0:
        return out
    diff = h_cmp - h_ref
    abs_err = abs(diff)
    out.update(
        {
            "height_diff_m": diff,
            "height_abs_error_m": abs_err,
            "height_sq_error_m2": diff * diff,
            "height_mae_m": abs_err,
            "height_rmse_m": abs_err,
            "height_diff_pct": 100.0 * diff / h_ref,
            "height_abs_diff_pct": 100.0 * abs_err / h_ref,
            "height_metrics_valid": True,
        }
    )
    return out


def _category_for_counts(n_ref, n_cmp):
    if n_ref == 1 and n_cmp == 1:
        return "matched_1to1"
    if n_ref > 1 and n_cmp == 1:
        return "many_to_one_3dbag_to_compared"
    if n_ref == 1 and n_cmp > 1:
        return "one_to_many_3dbag_to_compared"
    return "many_to_many"


def _ref_base(ref, idx, comparison):
    row = ref.iloc[idx]
    return {
        "comparison": comparison,
        "ref_index": int(idx),
        "ref_id": row.get("ref_id"),
        "ref_bag_id": row.get("ref_bag_id"),
        "ref_source": "3DBAG",
        "ref_area_m2": to_float(row.get("ref_area_m2")),
        "height_3dbag_m": to_float(row.get("height_3dbag_m")),
        "height_3dbag_method": row.get("height_3dbag_method"),
        "height_3dbag_50p_m": to_float(row.get("height_3dbag_50p_m")),
        "height_3dbag_max_m": to_float(row.get("height_3dbag_max_m")),
        "height_3dbag_definition": row.get("height_3dbag_definition"),
        "height_3dbag_max_definition": row.get("height_3dbag_max_definition"),
        "h_roof50_3dbag_m": to_float(row.get("h_roof50_3dbag_m")),
        "h_roofmax_3dbag_m": to_float(row.get("h_roofmax_3dbag_m")),
        "h_ground_3dbag_m": to_float(row.get("h_ground_3dbag_m")),
        "h_nok_3dbag_m": to_float(row.get("h_nok_3dbag_m")),
        "bouwjaar_3dbag": row.get("bouwjaar_3dbag"),
        "status_3dbag": row.get("status_3dbag"),
        "b3_bouwlagen": row.get("b3_bouwlagen"),
        "b3_dak_type": row.get("b3_dak_type"),
        "b3_kwaliteitsindicator": row.get("b3_kwaliteitsindicator"),
        "output_geometry_source": "3DBAG",
        "is_matched": False,
        "is_false_negative": False,
        "is_false_positive": False,
        "low_quality_match": False,
    }


def _cmp_base(cmp, idx, comparison):
    row = cmp.iloc[idx]
    return {
        "comparison": comparison,
        "cmp_index": int(idx),
        "cmp_id": row.get("cmp_id"),
        "cmp_source": row.get("cmp_source"),
        "cmp_source_id": row.get("cmp_source_id"),
        "cmp_area_m2": to_float(row.get("cmp_area_m2")),
        "height_compared_m": to_float(row.get("height_compared_m")),
        "height_compared_method": row.get("height_compared_method"),
        "height_compared_50p_m": to_float(row.get("height_compared_50p_m")),
        "height_compared_max_m": to_float(row.get("height_compared_max_m")),
        "height_compared_definition": row.get("height_compared_definition"),
        "h_roof50_cmp_m": to_float(row.get("h_roof50_cmp_m")),
        "h_ground_cmp_m": to_float(row.get("h_ground_cmp_m")),
        "h_roof70_cmp_m": to_float(row.get("h_roof70_cmp_m")),
        "h_roofmax_cmp_m": to_float(row.get("h_roofmax_cmp_m")),
        "year_built_cmp": row.get("year_built_cmp"),
        "status_cmp": row.get("status_cmp"),
        "pc_source_cmp": row.get("pc_source_cmp"),
        "pc_year_cmp": row.get("pc_year_cmp"),
        "reconstruction_success_cmp": row.get("reconstruction_success_cmp"),
            "n_parts_cmp": row.get("n_parts_cmp"),
            "gba_source": row.get("gba_source"),
        "gba_var": to_float(row.get("gba_var")),
    }


def _fn_record(ref, idx, comparison, best_pair):
    rec = _ref_base(ref, idx, comparison)
    rec.update(_pair_metrics(best_pair))
    rec.update(
        {
            "group_id": None,
            "n_3dbag_in_group": 1,
            "n_compared_in_group": 0,
            "cmp_ids": "",
            "cmp_id_best": best_pair["cmp_id"] if best_pair else None,
            "match_category": "false_negative_missed_3dbag",
            "match_category_label": CATEGORY_META["false_negative_missed_3dbag"]["label"],
            "qgis_color": CATEGORY_META["false_negative_missed_3dbag"]["color"],
            "cmp_metadata_scope": "none",
            "segmentation_scope": "none",
            "polygon_iou": 0.0,
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "group_iou": 0.0,
            "group_f1": 0.0,
            "group_precision": 0.0,
            "group_recall": 0.0,
            "match_method": "none",
            "match_quality": "none",
            "centroid_distance_m": best_pair["centroid_distance"] if best_pair else None,
            "centroid_overlap": bool(best_pair["centroid_overlap"]) if best_pair else False,
            "centroid_auxiliary_used": False,
            "height_compared_m": None,
            "height_compared_group_m": None,
            "height_3dbag_group_m": None,
            "cmp_height_polygon_count": 0,
            "is_matched": False,
            "is_false_negative": True,
            "is_false_positive": False,
            "low_quality_match": False,
        }
    )
    rec.update(_height_metrics(None, None))
    return rec


def _fp_record(cmp, idx, comparison, best_pair):
    rec = _cmp_base(cmp, idx, comparison)
    rec.update(_pair_metrics(best_pair))
    rec.update(
        {
            "group_id": None,
            "n_3dbag_in_group": 0,
            "n_compared_in_group": 1,
            "cmp_ids": rec.get("cmp_id"),
            "cmp_id_best": rec.get("cmp_id"),
            "ref_id": None,
            "ref_bag_id": None,
            "ref_source": "3DBAG",
            "ref_id_best": best_pair["ref_id"] if best_pair else None,
            "max_iou_with_any_3dbag": best_pair["iou"] if best_pair else 0.0,
            "match_category": "false_positive_extra_detection",
            "match_category_label": CATEGORY_META["false_positive_extra_detection"]["label"],
            "qgis_color": CATEGORY_META["false_positive_extra_detection"]["color"],
            "cmp_metadata_scope": "self",
            "output_geometry_source": "compared dataset (false-positive companion)",
            "segmentation_scope": "none",
            "polygon_iou": None,
            "f1_score": None,
            "precision": None,
            "recall": None,
            "group_iou": None,
            "group_f1": None,
            "group_precision": None,
            "group_recall": None,
            "match_method": "none",
            "match_quality": "none",
            "centroid_distance_m": best_pair["centroid_distance"] if best_pair else None,
            "centroid_overlap": bool(best_pair["centroid_overlap"]) if best_pair else False,
            "centroid_auxiliary_used": False,
            "height_3dbag_m": None,
            "height_3dbag_group_m": None,
            "height_compared_group_m": None,
            "cmp_height_polygon_count": 1,
            "is_matched": False,
            "is_false_negative": False,
            "is_false_positive": True,
            "low_quality_match": False,
        }
    )
    rec.update(_height_metrics(None, None))
    return rec


def compare_datasets(ref, cmp, comparison_name, height_mode="50p"):
    """Match compared polygons to 3DBAG reference buildings and compute metrics."""
    if height_mode not in ("50p", "max"):
        raise ValueError("height_mode must be '50p' or 'max'.")
    ref = ref.reset_index(drop=True).copy()
    cmp = cmp.reset_index(drop=True).copy()
    if height_mode == "max":
        ref["height_3dbag_m"] = ref["height_3dbag_max_m"]
        ref["height_3dbag_method"] = ref["height_3dbag_max_definition"]
        ref["height_3dbag_definition"] = ref["height_3dbag_max_definition"]
        cmp["height_compared_m"] = cmp["height_compared_max_m"]
        if "height_compared_definition" in cmp.columns:
            cmp["height_compared_method"] = cmp["height_compared_definition"]
    ref = ref[ref.geometry.notna() & ~ref.geometry.is_empty].copy()
    cmp = cmp[cmp.geometry.notna() & ~cmp.geometry.is_empty].copy()
    ref = ref[ref.geometry.area >= MIN_BUILDING_AREA_M2].reset_index(drop=True)
    cmp = cmp[cmp.geometry.area >= MIN_BUILDING_AREA_M2].reset_index(drop=True)
    if ref.empty or cmp.empty:
        raise RuntimeError(f"{comparison_name}: empty reference or compared dataset.")

    log(f"Building overlap graph for {comparison_name}: {len(ref)} reference x {len(cmp)} compared ...")
    tree = STRtree(ref.geometry.values)
    ref_centroids = ref.geometry.centroid.values
    cmp_centroids = cmp.geometry.centroid.values
    pairs = []
    for j, cgeom in enumerate(cmp.geometry.values):
        for i in tree.query(cgeom, predicate="intersects"):
            rgeom = ref.geometry.values[i]
            try:
                inter = rgeom.intersection(cgeom)
                ia = float(inter.area)
            except Exception:
                continue
            if not math.isfinite(ia) or ia < MIN_INTERSECTION_AREA_M2:
                continue
            ra = float(rgeom.area)
            ca = float(cgeom.area)
            if ra <= 0 or ca <= 0:
                continue
            union_area = ra + ca - ia
            iou = ia / union_area if union_area > 0 else 0.0
            overlap_ref = ia / ra
            overlap_cmp = ia / ca
            centroid_distance = float(ref_centroids[i].distance(cmp_centroids[j]))
            ref_centroid_in_cmp = bool(cgeom.covers(ref_centroids[i]))
            cmp_centroid_in_ref = bool(rgeom.covers(cmp_centroids[j]))
            centroid_overlap = ref_centroid_in_cmp or cmp_centroid_in_ref
            centroid_aux = (
                centroid_distance <= CENTROID_MAX_DIST_M
                and centroid_overlap
                and iou >= CENTROID_MIN_IOU
            )
            keep = (
                iou >= PAIR_IOU_MIN
                or max(overlap_ref, overlap_cmp) >= CONTAINMENT_OVERLAP_MIN
                or centroid_aux
            )
            if not keep:
                continue
            precision = ia / ca
            recall = ia / ra
            pairs.append(
                {
                    "ref_idx": int(i),
                    "cmp_idx": int(j),
                    "inter_area": ia,
                    "ref_area": ra,
                    "cmp_area": ca,
                    "iou": iou,
                    "precision": precision,
                    "recall": recall,
                    "f1": _pair_f1(precision, recall),
                    "overlap_ref": overlap_ref,
                    "overlap_cmp": overlap_cmp,
                    "overlap_max": max(overlap_ref, overlap_cmp),
                    "centroid_distance": centroid_distance,
                    "centroid_overlap": centroid_overlap,
                    "centroid_aux": centroid_aux,
                    "ref_id": ref.iloc[i].get("ref_id"),
                    "cmp_id": cmp.iloc[j].get("cmp_id"),
                }
            )
    log(f"  candidate overlap pairs: {len(pairs)}")
    if not pairs:
        log(
            f"WARNING: {comparison_name}: no overlap pairs passed the matching thresholds. "
            "All reference buildings will be exported as false negatives and all compared "
            "polygons as false positives."
        )

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for p in pairs:
        union(("r", p["ref_idx"]), ("c", p["cmp_idx"]))

    comps = defaultdict(lambda: {"refs": set(), "cmps": set(), "pairs": []})
    for p in pairs:
        root = find(("r", p["ref_idx"]))
        comp = comps[root]
        comp["refs"].add(p["ref_idx"])
        comp["cmps"].add(p["cmp_idx"])
        comp["pairs"].append(p)

    pairs_by_ref = defaultdict(list)
    pairs_by_cmp = defaultdict(list)
    for p in pairs:
        pairs_by_ref[p["ref_idx"]].append(p)
        pairs_by_cmp[p["cmp_idx"]].append(p)

    matched_ref = set()
    matched_cmp = set()
    matched_records = []
    group_rows = []
    group_counter = 0
    low_quality_groups = 0

    for comp in comps.values():
        refs = sorted(comp["refs"])
        cmps = sorted(comp["cmps"])
        group_pairs = comp["pairs"]
        if not refs or not cmps:
            continue
        u_ref = unary_union([ref.geometry.values[i] for i in refs])
        u_cmp = unary_union([cmp.geometry.values[j] for j in cmps])
        try:
            inter = u_ref.intersection(u_cmp)
            inter_area = float(inter.area)
        except Exception:
            inter_area = 0.0
        union_area = float(u_ref.area) + float(u_cmp.area) - inter_area
        group_iou = inter_area / union_area if union_area > 0 else 0.0
        group_precision = inter_area / float(u_cmp.area) if u_cmp.area > 0 else 0.0
        group_recall = inter_area / float(u_ref.area) if u_ref.area > 0 else 0.0
        group_f1 = _pair_f1(group_precision, group_recall)
        best_pair = max(group_pairs, key=lambda p: (p["iou"], -p["centroid_distance"]))
        max_containment = max(p["overlap_max"] for p in group_pairs)
        any_centroid_aux = any(p["centroid_aux"] for p in group_pairs)
        if best_pair["iou"] >= PAIR_IOU_MIN:
            match_method = "max_iou"
        elif max_containment >= CONTAINMENT_OVERLAP_MIN:
            match_method = "overlap_containment"
        elif any_centroid_aux:
            match_method = "centroid_auxiliary"
        else:
            match_method = "component"
        low_quality = group_iou < IOU_FN_THRESHOLD and max_containment < CONTAINMENT_OVERLAP_MIN
        if low_quality:
            low_quality_groups += 1

        group_counter += 1
        group_id = f"{comparison_name}_G{group_counter:05d}"
        category = _category_for_counts(len(refs), len(cmps))
        meta = CATEGORY_META[category]

        ref_h = pd.to_numeric(ref.loc[refs, "height_3dbag_m"], errors="coerce")
        ref_w = pd.to_numeric(ref.loc[refs, "ref_area_m2"], errors="coerce")
        h_ref_group = _wavg(ref_h, ref_w)
        cmp_h = pd.to_numeric(cmp.loc[cmps, "height_compared_m"], errors="coerce")
        cmp_w = []
        for j in cmps:
            try:
                w = float(cmp.geometry.values[j].intersection(u_ref).area)
            except Exception:
                w = 0.0
            cmp_w.append(w if w > 0 else float(cmp.geometry.values[j].area))
        h_cmp_group = _wavg(cmp_h, pd.Series(cmp_w, index=cmps))
        if h_ref_group is not None and h_cmp_group is not None and h_ref_group > 0:
            group_diff = h_cmp_group - h_ref_group
            group_abs = abs(group_diff)
            group_sq = group_diff * group_diff
            group_pct = 100.0 * group_diff / h_ref_group
        else:
            group_diff = group_abs = group_sq = group_pct = None

        group_rows.append(
            {
                "comparison": comparison_name,
                "group_id": group_id,
                "match_category": category,
                "match_category_label": meta["label"],
                "qgis_color": meta["color"],
                "n_3dbag_in_group": len(refs),
                "n_compared_in_group": len(cmps),
                "ref_ids": ";".join(str(ref.iloc[i].get("ref_id")) for i in refs),
                "cmp_ids": ";".join(str(cmp.iloc[j].get("cmp_id")) for j in cmps),
                "polygon_iou": group_iou,
                "f1_score": group_f1,
                "precision": group_precision,
                "recall": group_recall,
                "group_iou": group_iou,
                "group_f1": group_f1,
                "group_precision": group_precision,
                "group_recall": group_recall,
                "match_method": match_method,
                "low_quality_match": bool(low_quality),
                "height_3dbag_group_m": h_ref_group,
                "height_compared_group_m": h_cmp_group,
                "height_diff_group_m": group_diff,
                "height_abs_error_group_m": group_abs,
                "height_sq_error_group_m2": group_sq,
                "height_diff_group_pct": group_pct,
            }
        )

        for i in refs:
            rgeom = ref.geometry.values[i]
            ref_pairs = pairs_by_ref[i]
            ref_best = max(ref_pairs, key=lambda p: (p["iou"], -p["centroid_distance"]))
            valid_cmp_parts = []
            valid_cmp_weights = []
            for j in cmps:
                h = to_float(cmp.iloc[j].get("height_compared_m"))
                if h is None or not math.isfinite(h) or h <= 0:
                    continue
                try:
                    w = float(rgeom.intersection(cmp.geometry.values[j]).area)
                except Exception:
                    w = 0.0
                if w <= 0:
                    continue
                valid_cmp_parts.append(h)
                valid_cmp_weights.append(w)
            h_cmp_assigned = None
            if valid_cmp_weights:
                h_cmp_assigned = float(
                    sum(h * w for h, w in zip(valid_cmp_parts, valid_cmp_weights))
                    / sum(valid_cmp_weights)
                )
            elif h_cmp_group is not None:
                h_cmp_assigned = h_cmp_group
            h_ref_own = to_float(ref.iloc[i].get("height_3dbag_m"))
            rec = _ref_base(ref, i, comparison_name)
            rec.update(_cmp_base(cmp, ref_best["cmp_idx"], comparison_name))
            rec.update(_pair_metrics(ref_best))
            rec.update(
                {
                    "cmp_metadata_scope": "best_pair",
                    "cmp_group_area_m2": float(
                        sum(cmp.geometry.values[j].area for j in cmps)
                    ),
                    "cmp_group_part_count": int(
                        sum(int(cmp.iloc[j].get("n_parts_cmp") or 1) for j in cmps)
                    ),
                    "group_id": group_id,
                    "n_3dbag_in_group": len(refs),
                    "n_compared_in_group": len(cmps),
                    "cmp_ids": ";".join(str(cmp.iloc[j].get("cmp_id")) for j in cmps),
                    "cmp_id_best": ref_best["cmp_id"],
                    "match_category": category,
                    "match_category_label": meta["label"],
                    "qgis_color": meta["color"],
                    "segmentation_scope": "pair" if len(refs) == 1 and len(cmps) == 1 else "group",
                    "polygon_iou": group_iou,
                    "f1_score": group_f1,
                    "precision": group_precision,
                    "recall": group_recall,
                    "group_iou": group_iou,
                    "group_f1": group_f1,
                    "group_precision": group_precision,
                    "group_recall": group_recall,
                    "match_method": match_method,
                    "match_quality": "low" if low_quality else "normal",
                    "centroid_distance_m": ref_best["centroid_distance"],
                    "centroid_overlap": bool(ref_best["centroid_overlap"]),
                    "centroid_auxiliary_used": bool(ref_best["centroid_aux"]),
                    "height_compared_m": h_cmp_assigned,
                    "height_compared_group_m": h_cmp_group,
                    "height_3dbag_group_m": h_ref_group,
                    "cmp_height_polygon_count": len(valid_cmp_parts),
                    "is_matched": True,
                    "is_false_negative": False,
                    "is_false_positive": False,
                    "low_quality_match": bool(low_quality),
                }
            )
            rec.update(_height_metrics(h_ref_own, h_cmp_assigned))
            matched_records.append(rec)

        matched_ref.update(refs)
        matched_cmp.update(cmps)

    fn_records = []
    for i in range(len(ref)):
        if i in matched_ref:
            continue
        ref_pairs = pairs_by_ref.get(i, [])
        best_pair = (
            max(ref_pairs, key=lambda p: (p["iou"], -p["centroid_distance"]))
            if ref_pairs
            else None
        )
        fn_records.append(_fn_record(ref, i, comparison_name, best_pair))

    fp_records = []
    for j in range(len(cmp)):
        if j in matched_cmp:
            continue
        cmp_pairs = pairs_by_cmp.get(j, [])
        best_pair = (
            max(cmp_pairs, key=lambda p: (p["iou"], -p["centroid_distance"]))
            if cmp_pairs
            else None
        )
        fp_records.append(_fp_record(cmp, j, comparison_name, best_pair))

    main_records = matched_records + fn_records
    main = gpd.GeoDataFrame(
        pd.DataFrame(main_records),
        geometry=[ref.geometry.values[int(r["ref_index"])] for r in main_records],
        crs=WORK_CRS,
    )
    main = main.sort_values("ref_id").reset_index(drop=True)
    if fp_records:
        fp = gpd.GeoDataFrame(
            pd.DataFrame(fp_records),
            geometry=[cmp.geometry.values[int(r["cmp_index"])] for r in fp_records],
            crs=WORK_CRS,
        )
        fp = fp.sort_values("cmp_id").reset_index(drop=True)
    else:
        fp = gpd.GeoDataFrame(
            {
                "cmp_id": pd.Series(dtype="object"),
                "match_category": pd.Series(dtype="object"),
                "qgis_color": pd.Series(dtype="object"),
                "is_false_positive": pd.Series(dtype="bool"),
            },
            geometry=[],
            crs=WORK_CRS,
        )
    groups = pd.DataFrame(group_rows)

    category_counts = groups["match_category"].value_counts().to_dict() if not groups.empty else {}
    matched_groups = len(groups)
    matched_ref_count = len(matched_ref)
    matched_cmp_count = len(matched_cmp)
    fn_count = len(fn_records)
    fp_count = len(fp_records)

    seg_stats = {}
    for col in ("group_iou", "group_f1", "group_precision", "group_recall"):
        if not groups.empty and col in groups.columns:
            seg_stats[f"mean_{col}"] = float(pd.to_numeric(groups[col], errors="coerce").mean())
        else:
            seg_stats[f"mean_{col}"] = None

    valid_groups = groups[
        groups["height_3dbag_group_m"].notna()
        & groups["height_compared_group_m"].notna()
        & (pd.to_numeric(groups["height_3dbag_group_m"], errors="coerce") > 0)
    ].copy() if not groups.empty else pd.DataFrame()
    if not valid_groups.empty:
        err = pd.to_numeric(valid_groups["height_compared_group_m"], errors="coerce") - pd.to_numeric(
            valid_groups["height_3dbag_group_m"], errors="coerce"
        )
        abs_err = err.abs()
        height_mae = float(abs_err.mean())
        height_rmse = float(math.sqrt(float((err * err).mean())))
        h_ref_valid = pd.to_numeric(valid_groups["height_3dbag_group_m"], errors="coerce")
        height_mape = float((abs_err / h_ref_valid * 100.0).mean())
        height_n = int(len(valid_groups))
    else:
        height_mae = height_rmse = height_mape = None
        height_n = 0

    counts = {
        "height_mode": height_mode,
        "no_overlap_pairs": int(len(pairs) == 0),
        "reference_count": int(len(ref)),
        "compared_count": int(len(cmp)),
        "matched_reference_count": int(matched_ref_count),
        "matched_compared_count": int(matched_cmp_count),
        "matched_group_count": int(matched_groups),
        "false_negative_count": int(fn_count),
        "false_positive_count": int(fp_count),
        "building_count_error": int(len(cmp) - len(ref)),
        "building_count_error_pct": (
            100.0 * (len(cmp) - len(ref)) / len(ref) if len(ref) > 0 else None
        ),
        "groups_1to1": int(category_counts.get("matched_1to1", 0)),
        "groups_many_to_one": int(category_counts.get("many_to_one_3dbag_to_compared", 0)),
        "groups_one_to_many": int(category_counts.get("one_to_many_3dbag_to_compared", 0)),
        "groups_many_to_many": int(category_counts.get("many_to_many", 0)),
        "ref_features_many_to_one": int(
            groups.loc[groups["match_category"] == "many_to_one_3dbag_to_compared", "n_3dbag_in_group"].sum()
        ) if not groups.empty else 0,
        "cmp_features_many_to_one": int(
            groups.loc[groups["match_category"] == "many_to_one_3dbag_to_compared", "n_compared_in_group"].sum()
        ) if not groups.empty else 0,
        "ref_features_one_to_many": int(
            groups.loc[groups["match_category"] == "one_to_many_3dbag_to_compared", "n_3dbag_in_group"].sum()
        ) if not groups.empty else 0,
        "cmp_features_one_to_many": int(
            groups.loc[groups["match_category"] == "one_to_many_3dbag_to_compared", "n_compared_in_group"].sum()
        ) if not groups.empty else 0,
        "ref_features_many_to_many": int(
            groups.loc[groups["match_category"] == "many_to_many", "n_3dbag_in_group"].sum()
        ) if not groups.empty else 0,
        "cmp_features_many_to_many": int(
            groups.loc[groups["match_category"] == "many_to_many", "n_compared_in_group"].sum()
        ) if not groups.empty else 0,
        "low_quality_group_count": int(low_quality_groups),
        "height_group_count": int(height_n),
        "height_mae_m": height_mae,
        "height_rmse_m": height_rmse,
        "height_mape_pct": height_mape,
    }
    counts.update(seg_stats)
    log(
        f"{comparison_name}: matched {matched_ref_count}/{len(ref)} 3DBAG buildings, "
        f"FN={fn_count}, FP={fp_count}, groups={matched_groups} "
        f"(1:1={counts['groups_1to1']}, N:1={counts['groups_many_to_one']}, "
        f"1:N={counts['groups_one_to_many']}, N:N={counts['groups_many_to_many']})."
    )
    return {
        "comparison": comparison_name,
        "reference": ref,
        "compared": cmp,
        "main": main,
        "false_positives": fp,
        "groups": groups,
        "counts": counts,
        "pairs": pairs,
    }

REQUIRED_MAIN_PROPERTIES = [
    "comparison",
    "ref_id",
    "ref_source",
    "height_3dbag_m",
    "height_compared_m",
    "polygon_iou",
    "f1_score",
    "precision",
    "recall",
    "height_mae_m",
    "height_rmse_m",
    "height_diff_pct",
    "height_abs_diff_pct",
    "match_category",
    "match_category_label",
    "qgis_color",
    "is_matched",
    "is_false_negative",
    "is_false_positive",
    "group_id",
    "n_3dbag_in_group",
    "n_compared_in_group",
]


def _json_scalar(value):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not math.isfinite(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def write_geojson(gdf, path):
    out = gdf.copy()
    out = out.drop(columns=[c for c in ("ref_index", "cmp_index") if c in out.columns])
    if SLIM_OUTPUT:
        missing = [c for c in SLIM_PROPERTIES if c not in out.columns]
        if missing:
            raise ValueError(f"Cannot write slim GeoJSON; missing properties: {missing}")
        out = out[SLIM_PROPERTIES + ["geometry"]]
    for col in out.columns:
        if col == "geometry":
            continue
        if pd.api.types.is_numeric_dtype(out[col]) and not pd.api.types.is_bool_dtype(out[col]):
            decimals = SLIM_METRIC_DECIMALS if col in SLIM_METRIC_FIELDS else SLIM_ATTRIBUTE_DECIMALS
            out[col] = out[col].round(decimals)
        out[col] = out[col].apply(_json_scalar)
    out = out.to_crs(WGS84)
    out.to_file(
        path,
        driver="GeoJSON",
        engine="pyogrio",
        index=False,
        layer_options={"COORDINATE_PRECISION": SLIM_COORDINATE_DECIMALS},
    )
    log(f"Wrote {path.name}: {len(out)} features, {Path(path).stat().st_size:,} bytes")


def _summary_note(comparison):
    return (
        "GBA provides one LoD1 height per footprint, documented as derived from the maximum "
        "height within the footprint. GBA source imagery is older than the 3DBAG data; "
        "treat height metrics as inter-dataset agreement, not absolute accuracy."
    )


def build_summary_tables(results, study_bbox, study_area_km2):
    wide_rows = []
    long_rows = []
    for res in results:
        counts = res["counts"]
        comp = res["comparison"]
        meta_fn = CATEGORY_META["false_negative_missed_3dbag"]
        meta_fp = CATEGORY_META["false_positive_extra_detection"]
        height_def_3dbag = (
            "b3_h_dak_max - b3_h_maaiveld"
            if counts.get("height_mode") == "max"
            else "b3_h_dak_50p - b3_h_maaiveld"
        )
        height_def_cmp = "GBA height attribute (maximum within footprint)"
        wide_rows.append(
            {
                "comparison": comp,
                "reference_dataset": "3DBAG",
                "compared_dataset": "GBA",
                "study_bbox_wgs84": ",".join(f"{v:.6f}" for v in study_bbox),
                "study_area_km2": round(study_area_km2, 4),
                "reference_count": counts["reference_count"],
                "compared_count": counts["compared_count"],
                "building_count_error": counts["building_count_error"],
                "building_count_error_pct": counts["building_count_error_pct"],
                "matched_reference_count": counts["matched_reference_count"],
                "matched_compared_count": counts["matched_compared_count"],
                "matched_group_count": counts["matched_group_count"],
                "groups_1to1": counts["groups_1to1"],
                "groups_many_to_one": counts["groups_many_to_one"],
                "groups_one_to_many": counts["groups_one_to_many"],
                "groups_many_to_many": counts["groups_many_to_many"],
                "reference_features_many_to_one": counts["ref_features_many_to_one"],
                "compared_features_many_to_one": counts["cmp_features_many_to_one"],
                "reference_features_one_to_many": counts["ref_features_one_to_many"],
                "compared_features_one_to_many": counts["cmp_features_one_to_many"],
                "reference_features_many_to_many": counts["ref_features_many_to_many"],
                "compared_features_many_to_many": counts["cmp_features_many_to_many"],
                "false_negative_count": counts["false_negative_count"],
                "false_negative_description": meta_fn["description"],
                "false_negative_qgis_color": meta_fn["color"],
                "false_positive_count": counts["false_positive_count"],
                "false_positive_description": meta_fp["description"],
                "false_positive_qgis_color": meta_fp["color"],
                "mean_group_iou": counts.get("mean_group_iou"),
                "mean_group_f1": counts.get("mean_group_f1"),
                "mean_group_precision": counts.get("mean_group_precision"),
                "mean_group_recall": counts.get("mean_group_recall"),
                "height_group_count": counts["height_group_count"],
                "height_mae_m": counts["height_mae_m"],
                "height_rmse_m": counts["height_rmse_m"],
                "height_mape_pct": counts["height_mape_pct"],
                "height_3dbag_definition": height_def_3dbag,
                "height_compared_definition": height_def_cmp,
                "height_mode_used": counts.get("height_mode", "50p"),
                "category_legend_json": json.dumps(
                    {k: v for k, v in CATEGORY_META.items()}, ensure_ascii=False
                ),
                "notes": _summary_note(comp),
            }
        )

        def add_long(record_type, category, label, description, count, count_unit, ref_feat, cmp_feat, color, **extra):
            row = {
                "comparison": comp,
                "record_type": record_type,
                "category": category,
                "label": label,
                "description": description,
                "count": count,
                "count_unit": count_unit,
                "reference_features": ref_feat,
                "compared_features": cmp_feat,
                "qgis_color": color,
            }
            row.update(extra)
            long_rows.append(row)

        add_long(
            "building_count",
            "overall",
            "Overall building-count summary",
            f"{comp}: compared dataset has {counts['compared_count']} buildings vs {counts['reference_count']} 3DBAG reference buildings; error {counts['building_count_error']:+d} ({counts['building_count_error_pct']:+.2f}%).",
            counts["building_count_error"],
            "building_count_error",
            counts["matched_reference_count"],
            counts["matched_compared_count"],
            "",
            mean_iou=counts.get("mean_group_iou"),
            mean_f1=counts.get("mean_group_f1"),
            mean_precision=counts.get("mean_group_precision"),
            mean_recall=counts.get("mean_group_recall"),
            height_n_groups=counts["height_group_count"],
            height_mae_m=counts["height_mae_m"],
            height_rmse_m=counts["height_rmse_m"],
            height_mape_pct=counts["height_mape_pct"],
        )
        for category, count_key, ref_key, cmp_key in [
            ("matched_1to1", "groups_1to1", None, None),
            ("many_to_one_3dbag_to_compared", "groups_many_to_one", "ref_features_many_to_one", "cmp_features_many_to_one"),
            ("one_to_many_3dbag_to_compared", "groups_one_to_many", "ref_features_one_to_many", "cmp_features_one_to_many"),
            ("many_to_many", "groups_many_to_many", "ref_features_many_to_many", "cmp_features_many_to_many"),
        ]:
            meta = CATEGORY_META[category]
            add_long(
                "structural_discrepancy",
                category,
                meta["label"],
                meta["description"],
                counts[count_key],
                "overlap_groups",
                counts[ref_key] if ref_key else counts[count_key],
                counts[cmp_key] if cmp_key else counts[count_key],
                meta["color"],
            )
        add_long(
            "false_negative",
            "false_negative_missed_3dbag",
            meta_fn["label"],
            meta_fn["description"],
            counts["false_negative_count"],
            "3DBAG_buildings",
            counts["false_negative_count"],
            0,
            meta_fn["color"],
        )
        add_long(
            "false_positive",
            "false_positive_extra_detection",
            meta_fp["label"],
            meta_fp["description"],
            counts["false_positive_count"],
            "compared_buildings",
            0,
            counts["false_positive_count"],
            meta_fp["color"],
        )
        add_long(
            "segmentation_metrics",
            "mean_over_matched_groups",
            "Mean segmentation agreement",
            "Mean group-level IoU, F1, precision and recall over all accepted overlap groups.",
            counts["matched_group_count"],
            "matched_groups",
            counts["matched_reference_count"],
            counts["matched_compared_count"],
            "",
            mean_iou=counts.get("mean_group_iou"),
            mean_f1=counts.get("mean_group_f1"),
            mean_precision=counts.get("mean_group_precision"),
            mean_recall=counts.get("mean_group_recall"),
        )
        add_long(
            "height_metrics",
            "group_level_height_agreement",
            "Group-level height agreement",
            f"Height MAE/RMSE/MAPE over {counts['height_group_count']} matched groups with valid heights.",
            counts["height_group_count"],
            "matched_groups_with_heights",
            counts["matched_reference_count"],
            counts["matched_compared_count"],
            "",
            height_n_groups=counts["height_group_count"],
            height_mae_m=counts["height_mae_m"],
            height_rmse_m=counts["height_rmse_m"],
            height_mape_pct=counts["height_mape_pct"],
        )

    wide = pd.DataFrame(wide_rows)
    long = pd.DataFrame(long_rows)
    return wide, long


def validate_results(results, output_paths):
    log("Validating outputs ...")
    for res in results:
        comp = res["comparison"]
        ref = res["reference"]
        main = res["main"]
        fp = res["false_positives"]
        counts = res["counts"]

        assert len(main) == len(ref), f"{comp}: main rows must equal reference buildings"
        assert main["ref_id"].is_unique, f"{comp}: duplicate reference ids"
        assert main["ref_source"].eq("3DBAG").all(), f"{comp}: main must use 3DBAG source"
        assert main["is_false_positive"].eq(False).all(), f"{comp}: main contains FP flags"
        assert set(main.geometry.geom_type) <= {"Polygon", "MultiPolygon"}
        assert set(fp.geometry.geom_type) <= {"Polygon", "MultiPolygon"} if len(fp) else True
        assert counts["matched_reference_count"] + counts["false_negative_count"] == counts["reference_count"]
        assert counts["matched_compared_count"] + counts["false_positive_count"] == counts["compared_count"]
        assert len(fp) == counts["false_positive_count"]
        assert main["is_false_negative"].sum() == counts["false_negative_count"]
        assert (main["is_matched"] == (~main["is_false_negative"])).all()

        ref_lookup = ref.set_index("ref_id").geometry
        for ref_id, geom in zip(main["ref_id"], main.geometry):
            assert geom.equals(ref_lookup.loc[ref_id]), f"{comp}: geometry changed for {ref_id}"

        for col in REQUIRED_MAIN_PROPERTIES:
            assert col in main.columns, f"{comp}: missing property {col}"

        seg_cols = ["polygon_iou", "f1_score", "precision", "recall"]
        for col in seg_cols:
            vals = pd.to_numeric(main[col], errors="coerce").dropna()
            assert ((vals >= -1e-12) & (vals <= 1 + 1e-12)).all(), f"{comp}: {col} outside [0,1]"

        valid = main[main["height_metrics_valid"] == True].copy()
        if not valid.empty:
            calc = (valid["height_compared_m"] - valid["height_3dbag_m"]).abs()
            assert np.allclose(calc, valid["height_abs_error_m"], equal_nan=False, atol=1e-6)

        fn_colors = set(main.loc[main["is_false_negative"], "qgis_color"].dropna())
        assert fn_colors <= {CATEGORY_META["false_negative_missed_3dbag"]["color"]}
        if len(fp):
            assert fp["is_false_positive"].eq(True).all()
            assert set(fp["qgis_color"].dropna()) == {CATEGORY_META["false_positive_extra_detection"]["color"]}

        log(f"  {comp}: OK ({len(main)} reference features, {len(fp)} false positives)")

    for label, path in output_paths.items():
        if str(path).endswith(".geojson"):
            text = Path(path).read_text(encoding="utf-8")
            assert "NaN" not in text, f"{label}: NaN literal found in GeoJSON"
            data = json.loads(text)
            assert data.get("type") == "FeatureCollection"
            features = data.get("features", [])
            if SLIM_OUTPUT and features:
                actual = set(features[0].get("properties", {}).keys())
                assert actual == set(SLIM_PROPERTIES), (
                    f"{label}: slim GeoJSON property mismatch; "
                    f"extra={sorted(actual - set(SLIM_PROPERTIES))}, "
                    f"missing={sorted(set(SLIM_PROPERTIES) - actual)}"
                )
            log(f"  {label}: valid GeoJSON ({len(features)} features)")



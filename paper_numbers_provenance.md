# Provenance of the numbers in the manuscript

Reference run: `D:\GBA_3DBAG_outputs`
Run timestamp in `run_metadata.json` (`generated_utc`): `2026-11-16T08:51:34Z`
3DBAG release in `run_metadata.json`: `v2025.09.03`

All paths below are relative to the reference run directory. Paper values are rounded
display values; the exact values are in the listed fields.

## 1. Dataset counts and study area

| Paper location | Reported value | Evidence file | Field / basis |
|---|---:|---|---|
| Abstract, §3.1, §3.2, Table 1, §5.1, §8 | 473,690 | `comparison_summary.csv` | `reference_count` |
| Abstract, §3.2, Table 1, §5.1, §8 | 512,327 | `comparison_summary.csv` | `compared_count` |
| Abstract, §5.1, §8 | 8.16% | `comparison_summary.csv` | `building_count_error_pct` = 8.156600; also `building_count_error` = 38,637 |
| §3.1 | 553.51 km² | `comparison_summary.csv` | `study_area_km2` = 553.5074 |
| Table 1 / §5.1 | 473,690 | `raw/3DBAG_Amsterdam_Utrecht_lod12_clean.geojson` | feature count (one per 3DBAG building) |
| Table 1 / §5.1 | 512,327 | `raw/GBA_Amsterdam_Utrecht_lod12_clean.geojson` | feature count (one per GBA building) |
| §5.1 / §8 | 468,999 matched groups | `comparison_summary.csv` | `matched_group_count` |

## 2. Geometric and vertical metrics (Table 1, §5.1, §6.1, §6.3, §8)

| Paper value | Evidence file | Field |
|---|---:|---|
| Mean group IoU = 0.9724 | `comparison_summary.csv` | `mean_group_iou` = 0.972366573 |
| Mean group F1 = 0.9844 | `comparison_summary.csv` | `mean_group_f1` = 0.984359657 |
| Mean group precision = 0.9886 | `comparison_summary.csv` | `mean_group_precision` = 0.988569159 |
| Mean group recall = 0.9824 | `comparison_summary.csv` | `mean_group_recall` = 0.982380031 |
| Height MAE = 2.4663 m | `comparison_summary.csv` | `height_mae_m` = 2.466337525 |
| Height RMSE = 3.4057 m | `comparison_summary.csv` | `height_rmse_m` = 3.405719162 |
| MAPE = 43.15% (Figure 2) | `comparison_summary.csv` | `height_mape_pct` = 43.153482 |
| Height groups = 468,998 | `comparison_summary.csv` | `height_group_count` |

## 3. Structural matching (Table 2 and §5.2)

| Paper value | Evidence file | Field |
|---|---:|---|
| 466,184 one-to-one | `comparison_summary.csv` | `groups_1to1` |
| 744 many-to-one | `comparison_summary.csv` | `groups_many_to_one` |
| 1,505 one-to-many | `comparison_summary.csv` | `groups_one_to_many` |
| 566 many-to-many | `comparison_summary.csv` | `groups_many_to_many` |
| 1,850 false negatives | `comparison_summary.csv` | `false_negative_count` |
| 39,639 false positives | `comparison_summary.csv` | `false_positive_count` |
| 468,999 matched groups | `comparison_summary.csv` | `matched_group_count` |
| matched reference / compared | `comparison_summary.csv` | `matched_reference_count` = 471,840; `matched_compared_count` = 472,688 |

The same counts are also itemized in `comparison_category_breakdown.csv` (rows:
`structural_discrepancy`, `false_negative`, `false_positive`).

## 4. Matching thresholds (§4.1, §4.2)

These are code constants, not output fields:

| Paper value | Code location |
|---|---|
| minimum building area = 1.0 m² | `pipeline_lib.py` → `MIN_BUILDING_AREA_M2 = 1.0` |
| minimum intersection area = 0.5 m² | `pipeline_lib.py` → `MIN_INTERSECTION_AREA_M2 = 0.5` |
| pairwise IoU >= 0.10 | `pipeline_lib.py` → `PAIR_IOU_MIN = 0.10` |
| containment >= 0.50 | `pipeline_lib.py` → `CONTAINMENT_OVERLAP_MIN = 0.50` |
| centroid distance <= 10 m | `pipeline_lib.py` → `CENTROID_MAX_DIST_M = 10.0` |
| centroid-auxiliary IoU >= 0.05 | `pipeline_lib.py` → `CENTROID_MIN_IOU = 0.05` |
| low-quality group IoU < 0.10 and containment < 0.50 | `pipeline_lib.py` → `IOU_FN_THRESHOLD = 0.10` and `CONTAINMENT_OVERLAP_MIN` |
| EPSG:28992 (RD New) | `pipeline_lib.py` → `WORK_CRS = "EPSG:28992"` |

The logic itself is in `pipeline_lib.py` → `compare_datasets()` (candidate-pair
filtering and group construction).

## 5. Data versions and dates (§4.1)

| Paper value | Evidence |
|---|---|
| 3DBAG release `v2025.09.03` | `run_metadata.json` → `tdbag_version`; `run_pipeline.py` → `TDBAG_RELEASE` |
| GBA WFS retrieved 16 November 2026 | `run_metadata.json` → `generated_utc` = `2026-11-16T08:51:34Z` (run and access date) |
| 3DBAG site height definition | `run_metadata.json` → `reference_height_definition` |
| GBA own height definition | `run_metadata.json` → `compared_height_definition` |

The GBA service is live; `README.md` states that a fixed GBA input can be supplied with
`--gba-file` for exact reproduction.

## 6. H3 analysis (§4.5, §5.3, Table 3, §6.3, §6.4, §7, §8)

### 6.1 Cell and group counts

| Paper value | Evidence file | Field / basis |
|---|---:|---|---|
| H3 resolution = 8 | `h3/h3_metadata.json` | `h3_resolution` |
| 877 occupied H3 cells | `h3/h3_metadata.json` | `occupied_cells` |
| 874 cells containing matched groups | `h3/h3_metadata.json` | `cells_with_groups`; also number of rows in `h3/h3_cell_summary.csv` |
| 3 cells without matched groups | `h3/h3_metadata.json` | `cells_without_groups` (= 877 − 874) |
| 127 reference buildings/cell (median threshold) | `h3/h3_metadata.json`; `h3/h3_density_summary.csv` | `density_threshold_ref_buildings` |
| 434 Sparse cells | `h3/h3_metadata.json` | `cells_sparse` |
| 440 Dense cells | `h3/h3_metadata.json` | `cells_dense` |
| 15,670 Sparse groups | `h3/h3_metadata.json`; `h3/h3_density_summary.csv` | `groups_sparse` / `n_groups` |
| 453,329 Dense groups | `h3/h3_metadata.json`; `h3/h3_density_summary.csv` | `groups_dense` / `n_groups` |
| 468,999 total groups | `h3/h3_metadata.json`; `h3/h3_density_summary.csv` | `groups_total`; sum of `n_groups` |
| 3.3% Sparse share | `h3/h3_density_summary.csv` | `group_share_pct` = 3.341159 |
| 96.7% Dense share | `h3/h3_density_summary.csv` | `group_share_pct` = 96.658841 |

### 6.2 Group-level H3 metrics (Table 3)

All values are in `h3/h3_density_summary.csv`, one row per density class:

| Paper value | Field |
|---|---|
| Mean IoU Sparse 0.9619 / Dense 0.9727 | `mean_iou` |
| Median IoU Sparse 0.9948 / Dense 0.9912 | `median_iou` |
| Mean signed height error Sparse -1.589 / Dense -0.999 m | `mean_height_error_m` |
| Median signed height error Sparse -1.348 / Dense -0.813 m | `median_height_error_m` |
| Mean absolute height error Sparse 3.070 / Dense 2.445 m | `mean_abs_height_error_m` |
| RMSE Sparse 5.302 / Dense 3.321 m | `rmse_height_m` |
| Height-metric group count Sparse 15,670 / Dense 453,328 | `n_groups_height` |

Height metrics are computed over groups with a valid height. One Dense group has a
missing height, so the Dense height-metric count is 453,328 while the Dense matched-group
count is 453,329.

### 6.3 Group and cell-level data

- `h3/h3_group_metrics.geojson`: one point per matched group (468,999 features), with
  fields `group_id`, `h3`, `density`, `n_ref_cell`, `iou`, `height_diff`, `height_abs`.
- `h3/h3_cell_summary.csv`: 874 cells with matched groups, with `h3`, `n_ref_buildings`,
  `density`, `n_groups`, `mean_iou`, `median_iou`, `mean_height_error_m`,
  `mean_abs_height_error_m`.

## 7. Figures

| Figure | Source data | Script |
|---|---|---|
| Figure 2 | `comparison_summary.csv` | `make_paper_figures.py` → `fig2()` |
| Figure 3 | `h3/h3_group_metrics.geojson` | `make_paper_figures.py` → `fig3()` |
| Figure 4 | `h3/h3_group_metrics.geojson` | `make_paper_figures.py` → `fig4()` |
| Figure 5 | `h3/h3_cell_summary.csv` + Sentinel-2 cloudless 2020 imagery | `make_paper_figures.py` → `fig5()` |

Figure files: `figures/figure2_overall_comparison.*`, `figure3_iou_by_h3_density.*`,
`figure4_height_error_by_h3_density.*`, `figure5_h3_spatial_performance.*`
(PNG at 300 dpi and PDF).

## 8. Quick inspection commands

```bash
# comparison counts and metrics
python -c "import pandas as pd; print(pd.read_csv('comparison_summary.csv').T)"

# H3 summary and cell counts
python -c "import pandas as pd; print(pd.read_csv('h3/h3_density_summary.csv'))"
python -c "import json; print(json.load(open('h3/h3_metadata.json')))"
```

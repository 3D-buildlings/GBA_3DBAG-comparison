# GBA vs 3DBAG building-height pipeline (Amsterdam + Utrecht)

This repository contains an end-to-end, reproducible pipeline that builds per-building
height/area layers for **3DBAG** and the **Global Building Atlas (GBA)**, then compares them.

Repository outputs are generated from scratch and can be written directly to Google Drive
(e.g. from Colab).

## What it produces

For both study cities (Amsterdam and Utrecht):

- `raw/3DBAG_Amsterdam_Utrecht_lod12_clean.geojson`
  - one polygon per 3DBAG building
  - `height_m`: 3DBAG LoD1.2 site height = `z_roof - z_base` (roof is the flat top of the LoD1.2 model)
  - `area_m2`: `b3_opp_grond` (area shown by the 3DBAG viewer)
  - `height_nap_m`, `site_ground_nap_m`, `orig_*`, `city`
- `raw/GBA_Amsterdam_Utrecht_lod12_clean.geojson`
  - the GBA footprint with its matched 3DBAG site height/area in `height_m` / `area_m2`
  - the GBA own height/area in `orig_height_m` / `orig_area_m2`
- `raw/GBA_Amsterdam_Utrecht_raw.geojson`
  - the GBA footprint with the GBA own height/area as `height_m` / `area_m2`
  - the matched 3DBAG site values in `site_height_m`, `site_area_m2`, `site_3dbag_id`
- `GBA_vs_3DBAG.geojson` (one feature per 3DBAG footprint, incl. false negatives)
- `GBA_vs_3DBAG_false_positives.geojson` (GBA footprints with no accepted 3DBAG match)
- `GBA_vs_3DBAG_structural_groups.csv`
- `comparison_summary.csv`, `comparison_category_breakdown.csv`, `run_metadata.json`
- `comparison_outputs.zip`, `raw/lod12_clean.zip`
- `h3/h3_density_summary.csv`, `h3/h3_cell_summary.csv`,
  `h3/h3_group_metrics.geojson`, H3 figures and `h3/h3_metadata.json`

## Data sources

- 3DBAG: pinned release **v2025.09.03** (`https://data.3dbag.nl/v20250903/`), CityJSON tiles.
  The viewer and the comparison use the same LoD1.2 geometry.
- GBA: Global Building Atlas LoD1 footprints and height attribute, fetched from the
  official viewer WFS (`https://tubvsig-so2sat-vm1.srv.mwn.de`).

The GBA web service is a live source. If its data version changes, results will change.
For exact reproduction, pass a local GBA file with `--gba-file`.

PDOK is not used anywhere in this pipeline.

## Setup

```bash
python -m pip install -r requirements.txt
```

The pipeline was tested with `geopandas`, `pyogrio`, `shapely >= 2`, `pyproj`, `pandas`,
`numpy`, and `requests`.

## Run

```bash
python run_pipeline.py --out /path/to/output --work /path/to/work
```

Useful options:

| option | meaning |
|---|---|
| `--out` | output directory (use your Google Drive folder in Colab) |
| `--work` | cache/work directory (default: `<out>/work`) |
| `--gba-file` | optional local GBA footprint file (GPKG/GeoJSON) with `height_m`/`height` |
| `--gba-tile-step` | GBA WFS request tile size in degrees (default `0.03`) |
| `--threads` | parallel downloads for 3DBAG tiles |

### Google Colab

Open `GBA_3DBAG_Colab.ipynb` in Colab, set the repository path, mount Drive,
and run the final cell. Outputs are written to:

```
/content/drive/MyDrive/GBA_3DBAG_outputs
```

Typical runtime is about 40-90 minutes (3DBAG tile download + GBA WFS fetch + matching),
depending on network speed. Colab high-RAM is recommended for the matching step.

### H3 density analysis (after the pipeline)

```bash
python h3_analysis.py --out /path/to/output --resolution 8
```

This reads the comparison outputs and writes `h3/` tables, group points, figures,
and metadata. Density is the number of 3DBAG reference buildings per occupied H3 cell;
cells at or above the median are `Dense`, the rest `Sparse`. Match groups are assigned
to the H3 cell of the reference centroid (1:1) or the centroid of the union of their
reference buildings (multi-feature groups).

## Definitions

- **3DBAG site height**: for each building, all vertices of its LoD1.2 geometry are
  projected on the local up direction; the reported height is `max(z) - min(z)`.
  This is exactly the height the 3DBAG viewer shows when clicking the flat roof in the
  LoD1.2 view (`height.toFixed(1)`).
- **3DBAG area**: `b3_opp_grond` from the 3DBAG attributes.
- **GBA own height**: the `height` attribute from the GBA LoD1 footprint.
- **GBA clean file**: the GBA footprint with the matched 3DBAG site height/area for
  inspection; the GBA own values are kept in the `orig_*` fields.
- **Comparison**: 3DBAG reference = LoD1.2 site height; GBA compared = GBA own height.
  Height metrics are group-level (`MAE`, `RMSE`, `MAPE`) using the original matching
  algorithm of the study.
## Figures

python make_paper_figures.py --out /path/to/output
python study_area_map.py --out /path/to/output
python study_area_map.py --out /path/to/output --area "Name:min_lon,min_lat,max_lon,max_lat"

# any dataset with different column names (interactive or JSON mapping):
python make_figures_custom.py --out /path/to/output
python make_figures_custom.py --init-map columns.json
python make_figures_custom.py --out /path/to/output --map columns.json
## Method summary

For each 3DBAG building, all vertices of the LoD1.2 geometry are projected on the local
up direction; the site height is `max(z) - min(z)`, i.e. the flat-roof height shown by
the 3DBAG LoD1.2 viewer. The site area is `b3_opp_grond`. GBA provides one LoD1 footprint
and one height attribute per building. The two datasets are matched with the overlap-graph
algorithm in `pipeline_lib.py` (1:1, many-to-one, one-to-many, many-to-many;
false negatives and false positives are reported separately). Height agreement is measured
at group level with MAE, RMSE, and MAPE.

## Reference run

Command used for the reference output:

```bash
python run_pipeline.py --out D:\GBA_3DBAG_outputs --work D:\GBA_3DBAG_work
```

Data: 3DBAG `v2025.09.03` (pinned) + GBA WFS fetched 2026-11-16.

| item | value |
|---|---|
| 3DBAG buildings | 473,690 |
| GBA buildings | 512,327 |
| matched 3DBAG buildings | 471,840 |
| false negatives | 1,850 |
| false positives | 39,639 |
| matched overlap groups | 468,999 |
| height MAE | 2.4663 m |
| height RMSE | 3.4057 m |
| height MAPE | 43.15% |

Small variations in these counts are expected if the live GBA WFS changes. Use
`--gba-file` with a fixed GBA input for exact reproduction.

### Reference H3 results (Amsterdam + Utrecht, resolution 8)

- Occupied H3 cells: 877 (874 contain matched groups)
- Density threshold: median = 127 reference buildings per occupied cell
  (`Dense`: >= 127; `Sparse`: < 127)
- Groups: 468,999 (466,184 one-to-one + 2,815 multi-feature groups)

| Density | Groups | Mean IoU | Median IoU | Mean signed height error (m) | Median signed height error (m) | Mean absolute height error (m) | RMSE (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sparse | 15,670 | 0.9619 | 0.9948 | -1.589 | -1.348 | 3.070 | 5.302 |
| Dense | 453,329 | 0.9727 | 0.9912 | -0.999 | -0.813 | 2.445 | 3.321 |

The signed error is `GBA - 3DBAG`. Dense cells show a higher mean IoU and lower absolute
height error, while sparse cells have a higher median IoU but a heavier lower tail.

## Notes

- 3DBAG tile index and CityJSON files are pinned to `v2025.09.03` so the 3DBAG side is reproducible.
- Building matching uses spatial overlap groups (1:1, many-to-one, one-to-many, many-to-many).
- False negatives are 3DBAG buildings without an accepted GBA match; false positives are
  GBA footprints without an accepted 3DBAG match.
- The bundled `pipeline_lib.py` is the comparison engine used in the original study.
  Only its GBA and comparison functions are exercised by `run_pipeline.py`.

## Repository layout

```text
run_pipeline.py                  end-to-end pipeline
pipeline_lib.py                  matching and comparison engine
h3_analysis.py                   H3 resolution-8 density analysis
make_paper_figures.py            paper figures from pipeline outputs
GBA_3DBAG_Colab.ipynb            Colab runner
make_figures_custom.py           same figures from any dataset via column mapping
study_area_map.py                study-area map (default: Amsterdam + Utrecht)
requirements.txt
README.md
LICENSE
```

## Numbers and provenance

docs/paper_numbers_provenance.md maps every reported number in the manuscript to
the output file and field (or the code constant) that produced it.

## Code availability

https://github.com/farnia04/GBA_3DBAG-comparison

## Citation

If you use this code in a publication, please cite the repository above and the
3DBAG and GBA data sources listed under Attribution.

## Attribution

- 3DBAG by the 3D Geoinformation research group (TU Delft) and 3DGI, licensed under
  CC BY 4.0 (https://3dbag.nl).
- Global Building Atlas (GBA) data from its official web service; check the GBA terms of use.

## License

Code in this repository is released under the MIT License (see `LICENSE`).
Data licenses remain with the respective data providers.

# Prishtina Taxi Mobility

Preparing and visualizing GPS traces of 85 taxis in Prishtina, June 1 to August 31, 2023.
Project for the course *Përgatitja dhe vizualizimi i të dhënave* (FIEK, Master, 2026-27),
Prof. Asst. Dr. Mërgim H. Hoti.

Team: Nehar Jashari, Ora Osmani, Erëza Ismajli.

The project runs in three phases on the same code base: data preparation (Phase I),
outlier detection and exploration (Phase II), and visualization (Phase III).
Current phase: **II**. The code and outputs submitted for Phase I are at the tag `phase-1`.

## Data

Kadri Sylejmani, *Movement of taxi vehicles in Prishtina*, Kaggle, version 1 (last updated
2025-02-24), MIT licence.
https://www.kaggle.com/datasets/kadrisylejmani/movement-of-taxi-vehicles-in-prishtina

85 CSV files, one per taxi, named by vehicle id, 2,282,173,948 bytes and 28,590,588 data
rows. `8.csv` and `27.csv` are empty. Columns: `DeviceDateTime`, `Longitute` (sic),
`Latitude`, `Speed`, `Direction`, `Di1`, `Di2`, `Di3`.

The raw files are not in this repository. `reports/tables/raw_inventory.csv` pins them with
one SHA-256 per file, and `verify-raw` checks a download against it.

### External sources

**Kosovo boundaries.** Who's On First, *Kosovo: WOF Administrative Subdivisions and Human
Settlements*, Humanitarian Data Exchange, CC BY.
https://data.humdata.org/dataset/whosonfirst-data-admin-xkx
We use `whosonfirst-data-admin-xk-country-polygon.shp` for the Kosovo boundary and
`whosonfirst-data-admin-xk-localadmin-polygon.shp` for the municipalities. Downloaded on
2026-10-21 (352,936 bytes, SHA-256
`cab2553fb014e3d6de2742c3b20a05d65b306619bd5c855769e0d13e30530141`). The files are not in
this repository; get them with:

```bash
mkdir -p data/external/kosovo_admin
curl -L -o data/external/kosovo_admin/whosonfirst-data-admin-xk-latest.zip https://data.geocode.earth/wof/dist/shapefile/whosonfirst-data-admin-xk-latest.zip
unzip data/external/kosovo_admin/whosonfirst-data-admin-xk-latest.zip -d data/external/kosovo_admin
```

**Public holidays.** Ministry of Internal Affairs of the Republic of Kosovo, *Festat zyrtare
në Republikën e Kosovës për vitin 2023* (official holidays in 2023), issued under Law
No. 03/L-064 on Official Holidays.
https://mpb.rks-gov.net/Uploads/EditorFiles/TextHtml/79/Kalendari%20festave%20zyrtare%20_%202023.pdf
Law: https://gzk.rks-gov.net/ActDocumentDetail.aspx?ActID=2539
`data/external/calendar_2023.csv` (in this repository) has one row per day from June 1 to
September 1, 2023. The only official holiday in that period is Kurban Bajram, Wednesday
June 28, 2023.

## Setup

Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data

With a Kaggle API token: create one under *Settings > API* on kaggle.com and save it as
`~/.kaggle/kaggle.json` (or set `KAGGLE_USERNAME` and `KAGGLE_KEY`). Never put the token in
this repository.

```bash
kaggle datasets download kadrisylejmani/movement-of-taxi-vehicles-in-prishtina -p data/raw --unzip
```

Without a token, the public download link gives the same zip:

```bash
curl -L -o taxi.zip https://www.kaggle.com/api/v1/datasets/download/kadrisylejmani/movement-of-taxi-vehicles-in-prishtina
unzip taxi.zip -d data/raw && rm taxi.zip
```

Then check that you have exactly the bytes the pipeline was built on:

```bash
python -m taxi.pipeline verify-raw
```

## Run

```bash
python -m taxi.pipeline run --dev                 # every stage on the 3 dev vehicles (outputs in data/dev, reports/dev)
python -m taxi.pipeline run --stage detect --dev  # one stage
python -m taxi.pipeline check --dev               # row reconciliation, planned outputs, data dictionary
python -m taxi.pipeline run                       # all 83 vehicles; takes several minutes
python -m taxi.pipeline check
```

Stages, in order:

- Phase I, preparation: `verify-raw`, `ingest`, `profile-before`, `clean`, `profile-after`.
- Phase II, outliers: `detect` (rules O01-O04 flag impossible positions, speeds, and parked
  GPS drift; nothing is changed) and `screen` (removes or blanks what the validated rules
  found, logged in `reports/tables/cleaning_log.csv`).
- Phase I, preparation, now on the screened points: `aggregate`, `sample`, `features`,
  `trips`, `transform`, `reduce`.
- Phase II, exploration: `explore-points`, `explore-aggregates`, `explore-trips`.
- `export` writes the dataset package to `data/export/phase-2/`.

Self-checks of the shared functions: `python -m taxi.geo`, `python -m taxi.stats`,
`python -m taxi.outliers`.

Lint and format: `ruff check . && ruff format .`

### Reviewing the outlier rules

A rule may remove data only if most of what it flags are errors, so every rule is checked
by people first. `detect` draws a review sample
(`reports/tables/outlier_review_sample.csv`): flagged items and items just below each
rule's threshold, mixed so a reviewer cannot tell them apart. Each reviewer, once:

```python
from taxi import outliers
outliers.start_labels("<reviewer>")   # data/external/outlier_review_labels/<reviewer>.csv
outliers.render_review("<reviewer>")  # one plot per item in reports/dev/review_plots/<reviewer>/
```

Fill the `label` column with `error`, `real`, or `unsure` (and an optional `note`), judging
from the plot only. The plots show positions in metres relative to the item and times in
seconds relative to it, with no map, and stay on the reviewer's machine (`reports/dev/`
is not in git). The next `detect` run measures each rule's precision in
`reports/tables/outlier_validation.csv`.

## Layout

| Path | Content |
|---|---|
| `taxi/` | the pipeline: `config.py` (paths, thresholds, seed), `io.py`, `pipeline.py`, one module per stage |
| `notebooks/` | one narrative notebook per part of the work |
| `data/raw/` | the Kaggle CSVs, untouched (not in git) |
| `data/external/` | boundaries (not in git), `calendar_2023.csv`, and the reviewers' labels in `outlier_review_labels/` |
| `data/interim/`, `data/processed/`, `data/export/` | pipeline outputs (not in git) |
| `data/sample/points_sample.csv` | small stratified sample for reviewers |
| `reports/tables/`, `reports/figures/` | tables and figures used in the report |

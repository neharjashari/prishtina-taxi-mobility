# Prishtina Taxi Mobility

Preparing and visualizing GPS traces of 85 taxis in Prishtina, June 1 to August 31, 2023.
Project for the course *Përgatitja dhe vizualizimi i të dhënave* (FIEK, Master, 2026-27),
Prof. Asst. Dr. Mërgim H. Hoti.

Team: Nehar Jashari, Ora Osmani, Erëza Ismajli.

The project runs in three phases on the same code base: data preparation (Phase I),
outlier detection and exploration (Phase II), and visualization (Phase III).
Current phase: **I**.

## Data

Kadri Sylejmani, *Movement of taxi vehicles in Prishtina*, Kaggle, version 1 (last updated
2025-02-24), MIT licence.
https://www.kaggle.com/datasets/kadrisylejmani/movement-of-taxi-vehicles-in-prishtina

85 CSV files, one per taxi, named by vehicle id, 2,282,173,948 bytes and 28,590,588 data
rows. `8.csv` and `27.csv` are empty. Columns: `DeviceDateTime`, `Longitute` (sic),
`Latitude`, `Speed`, `Direction`, `Di1`, `Di2`, `Di3`.

The raw files are not in this repository. `reports/tables/raw_inventory.csv` pins them with
one SHA-256 per file, and `verify-raw` checks a download against it.

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
python -m taxi.pipeline run --stage ingest --dev  # one stage
python -m taxi.pipeline check --dev               # row reconciliation, planned outputs, data dictionary
python -m taxi.pipeline run                       # all 83 vehicles; takes long
python -m taxi.pipeline check
```

Stages, in order: `verify-raw`, `ingest`, `profile-before`, `clean`, `profile-after`,
`aggregate`, `sample`, `features`, `trips`, `transform`, `reduce`, `export`.

Lint and format: `ruff check . && ruff format .`

## Layout

| Path | Content |
|---|---|
| `taxi/` | the pipeline: `config.py` (paths, thresholds, seed), `io.py`, `pipeline.py`, one module per stage |
| `notebooks/` | one narrative notebook per part of the work |
| `data/raw/` | the Kaggle CSVs, untouched (not in git) |
| `data/interim/`, `data/processed/`, `data/export/` | pipeline outputs (not in git) |
| `data/sample/points_sample.csv` | small stratified sample for reviewers |
| `reports/tables/`, `reports/figures/` | tables and figures used in the report |
| `reports/phase-1/` | Phase I report and slides |

"""The extracted datasets for GClassroom: data/export/phase-1/ (data/dev/export/phase-1 in dev)."""

import hashlib
import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from taxi import PipelineError, config, io, pipeline

STAGE = "export"

CONTENTS = [
    (
        "points/",
        "processed/points",
        "final points: one Parquet file per vehicle, clean points with features, stop_id, and trip_id",
    ),
    ("trips.parquet", "processed/trips", "one row per trip"),
    ("trips.csv", "processed/trips", "the same trips as CSV"),
    (
        "vehicle_day.csv",
        "processed/vehicle_day",
        "one row per vehicle and local date with data",
    ),
    (
        "cell_hour.csv",
        "processed/cell_hour",
        "one row per 500 m grid cell, local date, and hour with at least one point",
    ),
    (
        "fleet_hour.csv",
        "processed/fleet_hour",
        "one row per local hour of the study window",
    ),
    (
        "model/trips_model.parquet",
        "processed/model/trips_model",
        "trip features, transformed, encoded, and z-scored",
    ),
    (
        "model/vehicle_day_model.parquet",
        "processed/model/vehicle_day_model",
        "vehicle-day features with trip statistics, transformed, z-scored, with the PCA scores",
    ),
    (
        "points_sample.csv",
        "sample/points_sample",
        "stratified sample of the clean points (vehicle x hour), under 5 MB; lat, lon, and cell_id withheld on night-time rows with the ignition off (S01)",
    ),
]

SOURCES = """\
- **Taxi GPS data.** Kadri Sylejmani, *Movement of taxi vehicles in Prishtina*, Kaggle,
  version 1 (last updated 2025-02-24), MIT licence.
  https://www.kaggle.com/datasets/kadrisylejmani/movement-of-taxi-vehicles-in-prishtina
- **Kosovo boundary and municipalities** (`in_kosovo`, `municipality`). Who's On First,
  *Kosovo: WOF Administrative Subdivisions and Human Settlements*, Humanitarian Data
  Exchange, CC BY. https://data.humdata.org/dataset/whosonfirst-data-admin-xkx
- **Public holidays** (`is_holiday`). Ministry of Internal Affairs of the Republic of Kosovo,
  *Festat zyrtare në Republikën e Kosovës për vitin 2023*.
  https://mpb.rks-gov.net/Uploads/EditorFiles/TextHtml/79/Kalendari%20festave%20zyrtare%20_%202023.pdf
"""


def _sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 24):
            sha.update(chunk)
    return sha.hexdigest()


def _columns_and_rows(path: Path) -> tuple[list[str], int]:
    if path.suffix == ".csv":
        columns = pipeline.file_columns(path)
        with path.open("rb") as fh:
            rows = (
                sum(chunk.count(b"\n") for chunk in iter(lambda: fh.read(1 << 24), b""))
                - 1
            )
        return columns, rows
    meta = pq.ParquetFile(path)
    return pipeline.file_columns(path), meta.metadata.num_rows


def _sources() -> dict[str, list[Path]]:
    """Source files of every package entry; stops when one is missing."""
    final = sorted(config.points_dir("final").glob("*.parquet"))
    src = {
        "points/": final,
        "trips.parquet": [io.table_path("trips")],
        "trips.csv": [io.table_path("trips")],
        "vehicle_day.csv": [io.table_path("vehicle_day")],
        "cell_hour.csv": [io.table_path("cell_hour")],
        "fleet_hour.csv": [io.table_path("fleet_hour")],
        "model/trips_model.parquet": [io.table_path("model/trips_model")],
        "model/vehicle_day_model.parquet": [io.table_path("model/vehicle_day_model")],
        "points_sample.csv": [config.sample_csv()],
    }
    missing = [
        str(p.relative_to(config.ROOT))
        for paths in src.values()
        for p in paths
        if not p.exists()
    ]
    if not final:
        missing.append(
            str(config.points_dir("final").relative_to(config.ROOT)) + "/*.parquet"
        )
    if missing:
        raise PipelineError(
            "Cannot build the export, inputs missing (run the earlier stages): "
            + ", ".join(sorted(set(missing)))
        )
    return src


def check_dictionary(src: dict[str, list[Path]]) -> pd.DataFrame:
    """Every exported column must have a data dictionary row; returns the rows of the exported datasets."""
    dictionary = io.read_dictionary()
    documented = dictionary.groupby("dataset")["column"].apply(set).to_dict()
    problems = []
    for target, dataset, _ in CONTENTS:
        columns = {c for p in src[target] for c in pipeline.file_columns(p)}
        undocumented = sorted(columns - documented.get(dataset, set()))
        if undocumented:
            problems.append(f"{target} ({dataset}): {', '.join(undocumented)}")
    if problems:
        raise PipelineError(
            "Export stopped: exported columns without a data dictionary row:\n  "
            + "\n  ".join(problems)
        )
    datasets = {d for _, d, _ in CONTENTS}
    return dictionary[dictionary["dataset"].isin(datasets)]


def _clean_export_dir() -> Path:
    out = config.export_dir()
    if out.parts[-2:] != ("export", "phase-1") or config.ROOT not in out.parents:
        raise PipelineError(f"Unexpected export directory: {out}")
    if out.exists():
        shutil.rmtree(out)
    (out / "points").mkdir(parents=True)
    (out / "model").mkdir()
    return out


def build(src: dict[str, list[Path]], dictionary: pd.DataFrame) -> Path:
    out = _clean_export_dir()
    for path in src["points/"]:
        shutil.copy2(path, out / "points" / path.name)
    for target in (
        "trips.parquet",
        "model/trips_model.parquet",
        "model/vehicle_day_model.parquet",
    ):
        shutil.copy2(src[target][0], out / target)
    for target in ("trips.csv", "vehicle_day.csv", "cell_hour.csv", "fleet_hour.csv"):
        df = pd.read_parquet(src[target][0])
        io.count_rows(rows_in=len(df), rows_out=len(df))
        df.to_csv(out / target, index=False, encoding="utf-8")
    shutil.copy2(src["points_sample.csv"][0], out / "points_sample.csv")
    dictionary.to_csv(out / "data_dictionary.csv", index=False, encoding="utf-8")
    return out


def manifest(out: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(
        p for p in out.rglob("*") if p.is_file() and p.name != "README_data.md"
    ):
        rel = path.relative_to(out).as_posix()
        entry = "points/" if rel.startswith("points/") else rel
        dataset = {t: d for t, d, _ in CONTENTS}.get(entry, "")
        if path.suffix in (".csv", ".parquet") and rel != "data_dictionary.csv":
            columns, n_rows = _columns_and_rows(path)
        else:
            columns, n_rows = pipeline.file_columns(path), None
        rows.append(
            {
                "file": rel,
                "dataset": dataset,
                "format": path.suffix.lstrip("."),
                "rows": n_rows,
                "columns": len(columns),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return pd.DataFrame(rows).astype({"rows": "Int64"})


def readme(table: pd.DataFrame, total_bytes: int, dev: bool) -> str:
    n_vehicles = int(table["file"].str.startswith("points/").sum())
    lines = [
        "# Prishtina Taxi Mobility: Phase I datasets",
        "",
        f"Prepared GPS traces of {n_vehicles} taxis in Prishtina, June 1 to August 31, 2023, from the",
        "project's preprocessing pipeline (`python -m taxi.pipeline run`). Every column of",
        "every file is described in `data_dictionary.csv` (dataset, column, dtype, unit,",
        "attribute type, values, description, source).",
        "",
        f"Built by run `{io.current_run_id()}` from commit `{pipeline.git_commit()}`"
        + (" (dev run: only the 3 development vehicles)." if dev else "."),
        "",
        "## Files",
        "",
        "| File | Content | Rows | Columns | Size (MB) |",
        "|---|---|---:|---:|---:|",
    ]
    for target, _, what in CONTENTS:
        part = (
            table[table["file"].str.startswith(target)]
            if target.endswith("/")
            else table[table["file"] == target]
        )
        rows = int(part["rows"].sum())
        columns = int(part["columns"].max())
        mb = part["bytes"].sum() / 1e6
        files = f" ({len(part)} files)" if target.endswith("/") else ""
        lines.append(
            f"| `{target}` | {what}{files} | {rows:,} | {columns} | {mb:,.1f} |"
        )
    d = table[table["file"] == "data_dictionary.csv"]
    lines.append(
        f"| `data_dictionary.csv` | one row per column of every dataset above | "
        f"{len(pd.read_csv(config.export_dir() / 'data_dictionary.csv')):,} | "
        f"{int(d['columns'].iloc[0])} | {d['bytes'].iloc[0] / 1e6:,.1f} |"
    )
    lines += [
        "",
        f"Total size: {total_bytes / 1e6:,.1f} MB ({total_bytes:,} bytes) in {len(table)} data files,",
        "plus this README.",
        "`export_manifest.csv` in the repository's `reports/tables/` lists the size and SHA-256",
        "of every file.",
        "",
        "## Conventions",
        "",
        "- Times: `ts_utc` in UTC, `*_local` in Kosovo local time (Europe/Skopje, UTC+2 in the",
        "  whole period). The raw clock is treated as local time; the source does not confirm it.",
        "- Each point stores the step that starts at it (`dt_s`, `dist_m`, `gap_after`). Durations,",
        "  distances, and shares are sums over steps, never point counts. A gap (`gap_after`,",
        "  longer than 15 minutes) is unknown time and is left out.",
        (
            f"- Trips: rule `{config.TRIP_RULE}` (column `rule` of the trip table); "
            "a trip that crosses midnight belongs to the day it starts."
        ),
        "- Missing values are empty in CSV and null in Parquet. `flag_*` columns mark rows kept",
        "  for outlier checks in Phase II.",
        "",
        "## Sources and licences",
        "",
        SOURCES.rstrip(),
        f"- **City centre** (`dist_center_m`). {config.CITY_CENTER_SOURCE}.",
        "",
    ]
    return "\n".join(lines)


def run(dev: bool) -> None:
    """Build data/export/phase-1/ after the dictionary check; write the manifest and README_data.md."""
    src = _sources()
    dictionary = check_dictionary(src)
    out = build(src, dictionary)
    table = manifest(out)
    io.write_report_table(table, "export_manifest")
    total = int(table["bytes"].sum())
    (out / "README_data.md").write_text(readme(table, total, dev), encoding="utf-8")
    total += (out / "README_data.md").stat().st_size
    print(
        f"{out.relative_to(config.ROOT)}: {len(table) + 1} files, {total / 1e6:,.1f} MB; "
        "every exported column has a data dictionary row"
    )

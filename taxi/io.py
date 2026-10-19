"""Read and write helpers shared by every stage, and the cleaning log.

Every path comes from taxi/config.py, so the same calls read and write the dev locations
during a --dev run.
"""

import csv
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal

import pandas as pd

from taxi import config

Stage = config.PointsStage

LOG_COLUMNS = [
    "run_id",
    "stage",
    "rule_id",
    "action",
    "vehicle_id",
    "rows_in",
    "rows_affected",
    "note",
]
LOG_ACTIONS = {"drop", "set_na", "impute", "flag"}

_run_id: str | None = None
_row_counts = {"rows_in": 0, "rows_out": 0}


# --- Run id and row counters (used by the pipeline for run_log.csv) -------------------


def new_run_id() -> str:
    """UTC start time of a run, for example 20261021T183000Z."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def set_run_id(run_id: str) -> None:
    global _run_id
    _run_id = run_id


def current_run_id() -> str:
    """The id of the running pipeline run, or a fresh one when a stage runs outside it."""
    global _run_id
    if _run_id is None:
        _run_id = new_run_id()
    return _run_id


def reset_row_counts() -> None:
    _row_counts["rows_in"] = 0
    _row_counts["rows_out"] = 0


def row_counts() -> dict[str, int]:
    """Rows read and written through this module since the last reset."""
    return dict(_row_counts)


def count_rows(rows_in: int = 0, rows_out: int = 0) -> None:
    """Add rows a stage read or wrote without going through read_*/write_* (raw CSVs)."""
    _row_counts["rows_in"] += rows_in
    _row_counts["rows_out"] += rows_out


# --- Points: one Parquet file per vehicle ---------------------------------------------


def points_path(stage: Stage, vehicle_id: int) -> Path:
    return config.points_dir(stage) / f"{int(vehicle_id)}.parquet"


def raw_path(vehicle_id: int) -> Path:
    return config.RAW_DIR / f"{int(vehicle_id)}.csv"


def vehicle_ids(stage: Literal["raw", "interim", "clean", "final"]) -> list[int]:
    """Vehicles available at a stage, sorted.

    "raw" lists the non-empty raw files; during a dev run it is limited to DEV_VEHICLES.
    The other stages list the Parquet files present in that stage's (dev) folder.
    """
    if stage == "raw":
        ids = [
            int(p.stem) for p in config.RAW_DIR.glob("*.csv") if p.stat().st_size > 0
        ]
        if config.is_dev():
            ids = [v for v in ids if v in config.DEV_VEHICLES]
    else:
        ids = [int(p.stem) for p in config.points_dir(stage).glob("*.parquet")]
    return sorted(ids)


def read_points(
    stage: Stage, vehicle_ids: list[int] | None = None, columns: list[str] | None = None
) -> pd.DataFrame:
    """Read one Parquet file per requested vehicle (all available vehicles when None).

    Reading every vehicle of a full run loads tens of millions of rows. Stages loop over
    vehicles instead, and fleet-wide counts use DuckDB over the files.
    """
    ids = _available(stage) if vehicle_ids is None else [int(v) for v in vehicle_ids]
    if not ids:
        raise FileNotFoundError(
            f"No {stage} points in {config.points_dir(stage)}. Run the stage that writes them."
        )
    frames = []
    for v in ids:
        path = points_path(stage, v)
        if not path.exists():
            raise FileNotFoundError(f"No {stage} points for vehicle {v}: {path}")
        frames.append(pd.read_parquet(path, columns=columns))
    df = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    # concat turns categoricals with different categories per vehicle into object.
    for col in frames[0].columns:
        if isinstance(frames[0][col].dtype, pd.CategoricalDtype) and not isinstance(
            df[col].dtype, pd.CategoricalDtype
        ):
            df[col] = df[col].astype("category")
    _row_counts["rows_in"] += len(df)
    return df


def write_points(df: pd.DataFrame, stage: Stage) -> None:
    """Write one file per vehicle in df, replacing that vehicle's file.

    Not to_parquet(partition_cols=...): on a rerun that adds files next to the old ones.
    """
    if "vehicle_id" not in df.columns or df["vehicle_id"].isna().any():
        raise ValueError("write_points needs a vehicle_id on every row")
    out_dir = config.points_dir(stage)
    out_dir.mkdir(parents=True, exist_ok=True)
    for v, part in df.groupby("vehicle_id", sort=True, observed=True):
        _write_parquet(part, points_path(stage, int(v)))
        _row_counts["rows_out"] += len(part)


def _available(stage: Stage) -> list[int]:
    return vehicle_ids(stage)


# --- Tables: one Parquet file each under data/processed/ ------------------------------


def table_path(name: str) -> Path:
    """Path of a processed table: "trips", "vehicle_day", "samples/srs_0.01", "model/trips_model"."""
    parts = PurePosixPath(name).parts
    if not parts or name.startswith("/") or ".." in parts or name.endswith(".parquet"):
        raise ValueError(f"Invalid table name: {name!r}")
    return config.processed_dir().joinpath(*parts[:-1], f"{parts[-1]}.parquet")


def read_table(name: str) -> pd.DataFrame:
    df = pd.read_parquet(table_path(name))
    _row_counts["rows_in"] += len(df)
    return df


def write_table(df: pd.DataFrame, name: str) -> None:
    path = table_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet(df, path)
    _row_counts["rows_out"] += len(df)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    # Write next to the target and rename, so an interrupted run never leaves half a file.
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


# --- Cleaning log ---------------------------------------------------------------------


def log_step(
    stage: str,
    rule_id: str,
    action: str,
    vehicle_id: int | None,
    rows_in: int,
    rows_affected: int,
    note: str = "",
) -> None:
    """Append one row to cleaning_log.csv. Call it for every rule, also when 0 rows are affected."""
    if action not in LOG_ACTIONS:
        raise ValueError(f"action must be one of {sorted(LOG_ACTIONS)}, got {action!r}")
    if not 0 <= rows_affected <= rows_in:
        raise ValueError(
            f"rows_affected ({rows_affected}) must be between 0 and rows_in ({rows_in})"
        )
    path = config.cleaning_log_csv()
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(LOG_COLUMNS)
        vid = "" if vehicle_id is None else int(vehicle_id)
        writer.writerow(
            [
                current_run_id(),
                stage,
                rule_id,
                action,
                vid,
                int(rows_in),
                int(rows_affected),
                note,
            ]
        )


def read_log() -> pd.DataFrame:
    """The cleaning log, with an empty frame of the right columns when there is none yet."""
    path = config.cleaning_log_csv()
    if not path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(
        path,
        dtype={"run_id": str, "stage": str, "rule_id": str, "action": str, "note": str},
        keep_default_na=False,
        na_values={"vehicle_id": [""]},
    ).astype({"vehicle_id": "Int16", "rows_in": "int64", "rows_affected": "int64"})


def clear_log(stage: str | None = None) -> None:
    """Remove one stage's rows from the cleaning log (all rows when stage is None).

    The pipeline calls this before a stage runs, so a rerun replaces the stage's rows
    instead of counting its drops twice.
    """
    path = config.cleaning_log_csv()
    if not path.exists():
        return
    if stage is None:
        path.unlink()
        return
    log = read_log()
    log[log["stage"] != stage].to_csv(path, index=False)

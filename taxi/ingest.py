"""Data collection and typed ingestion.

verify-raw: the first run writes reports/tables/raw_inventory.csv (T1). Every later run
compares data/raw/ against it and stops on any difference, so everyone on the team can
show they work from the same bytes.

ingest: every raw row becomes one typed row of interim/points, one Parquet file per
vehicle. Nothing is dropped here. A value that does not parse becomes NA, is counted in
ingest_summary.csv, and is logged; the cleaning rules decide what to drop.
"""

import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from taxi import PipelineError, config, io

INVENTORY_COLUMNS = [
    "file",
    "vehicle_id",
    "bytes",
    "rows",
    "header",
    "first_ts_raw",
    "last_ts_raw",
    "sha256",
]
_CHUNK_BYTES = 1 << 24


def inspect_raw_file(path: Path) -> dict:
    """One inventory row: size, SHA-256, data rows, header, and the first and last DeviceDateTime as text."""
    sha = hashlib.sha256()
    newlines = 0
    last_byte = b""
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK_BYTES):
            sha.update(chunk)
            newlines += chunk.count(b"\n")
            last_byte = chunk[-1:]
    size = path.stat().st_size
    row = {
        "file": path.name,
        "vehicle_id": int(path.stem),
        "bytes": size,
        "rows": 0,
        "header": "",
        "first_ts_raw": "",
        "last_ts_raw": "",
        "sha256": sha.hexdigest(),
    }
    if size == 0:
        return row
    lines = newlines if last_byte == b"\n" else newlines + 1
    row["rows"] = lines - 1
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        row["header"] = ",".join(next(reader))
        first = next(reader, None)
    if first:
        row["first_ts_raw"] = first[0]
        row["last_ts_raw"] = _last_record(path)[0]
    return row


def _last_record(path: Path) -> list[str]:
    with path.open("rb") as fh:
        fh.seek(max(0, path.stat().st_size - 4096))
        tail = fh.read().decode("utf-8")
    last_line = [line for line in tail.splitlines() if line.strip()][-1]
    return next(csv.reader([last_line]))


def build_inventory() -> pd.DataFrame:
    files = sorted(config.RAW_DIR.glob("*.csv"))
    if not files:
        raise PipelineError(
            f"No CSV files in {config.RAW_DIR}. Download the dataset first (see README.md)."
        )
    bad_names = [p.name for p in files if not p.stem.isdigit()]
    if bad_names:
        raise PipelineError(
            f"Raw file names must be vehicle ids, found: {', '.join(bad_names)}"
        )
    if len(files) != config.RAW_FILES_EXPECTED:
        raise PipelineError(
            f"Expected {config.RAW_FILES_EXPECTED} raw files in {config.RAW_DIR}, found {len(files)}."
        )
    inv = pd.DataFrame([inspect_raw_file(p) for p in files], columns=INVENTORY_COLUMNS)
    return inv.sort_values("vehicle_id", ignore_index=True)


def verify_raw(dev: bool = False) -> None:
    """Check the raw files: count, headers, empty files, and the bytes against raw_inventory.csv.

    The raw files are the input of every run, so dev runs check all 85 of them too.
    """
    inv = build_inventory()
    io.count_rows(rows_in=int(inv["rows"].sum()))

    expected_header = ",".join(config.RAW_COLUMNS)
    non_empty = inv[inv["bytes"] > 0]
    wrong = non_empty[non_empty["header"] != expected_header]
    if not wrong.empty:
        details = "\n".join(f"  {r.file}: {r.header}" for r in wrong.itertuples())
        raise PipelineError(
            f"Unexpected header (expected {expected_header}):\n{details}"
        )

    empty = sorted(inv.loc[inv["bytes"] == 0, "vehicle_id"].tolist())
    if empty != sorted(config.EMPTY_FILES_EXPECTED):
        raise PipelineError(
            f"Empty raw files {empty}, expected {sorted(config.EMPTY_FILES_EXPECTED)}."
        )
    small = non_empty[non_empty["bytes"] < config.SMALL_FILE_BYTES]

    print(
        f"{len(inv)} raw files, {len(non_empty)} with data, {int(inv['rows'].sum()):,} data rows, "
        f"{int(inv['bytes'].sum()):,} bytes."
    )
    print(
        f"Empty files (expected, kept in the inventory): {', '.join(f'{v}.csv' for v in empty)}"
    )
    print(
        f"Files under {config.SMALL_FILE_BYTES:,} bytes: "
        + ", ".join(f"{r.file} ({r.rows:,} rows)" for r in small.itertuples())
    )

    if not config.RAW_INVENTORY_CSV.exists():
        config.RAW_INVENTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
        inv.to_csv(config.RAW_INVENTORY_CSV, index=False)
        print(
            f"Wrote {config.RAW_INVENTORY_CSV.relative_to(config.ROOT)}. Commit it: it pins the input."
        )
        return

    pinned = pd.read_csv(config.RAW_INVENTORY_CSV, dtype=str, keep_default_na=False)
    problems = _compare_inventories(pinned, inv.astype(str))
    if problems:
        shown = "\n".join(f"  {p}" for p in problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise PipelineError(
            f"data/raw/ differs from {config.RAW_INVENTORY_CSV.relative_to(config.ROOT)}:\n{shown}{more}"
        )
    print(
        f"All {len(inv)} files match {config.RAW_INVENTORY_CSV.relative_to(config.ROOT)}."
    )


def _compare_inventories(pinned: pd.DataFrame, current: pd.DataFrame) -> list[str]:
    problems = []
    if list(pinned.columns) != INVENTORY_COLUMNS:
        return [
            f"inventory columns are {list(pinned.columns)}, expected {INVENTORY_COLUMNS}"
        ]
    merged = pinned.merge(
        current,
        on="file",
        how="outer",
        suffixes=("_pinned", "_now"),
        indicator="presence",
    )
    for f in merged.loc[merged["presence"] == "left_only", "file"]:
        problems.append(f"{f}: in the inventory but not in data/raw/")
    for f in merged.loc[merged["presence"] == "right_only", "file"]:
        problems.append(f"{f}: in data/raw/ but not in the inventory")
    both = merged[merged["presence"] == "both"]
    for col in INVENTORY_COLUMNS[1:]:
        diff = both[both[f"{col}_pinned"] != both[f"{col}_now"]]
        for f, now, was in zip(
            diff["file"], diff[f"{col}_now"], diff[f"{col}_pinned"], strict=True
        ):
            problems.append(f"{f}: {col} is {now}, inventory has {was}")
    return problems


# --- ingest: raw CSV to typed Parquet --------------------------------------------------

INTERIM_DATASET = "interim/points"
# raw name -> (interim name, dtype). Coordinates stay float64: float32 loses the sixth decimal.
NUMERIC_COLUMNS = {
    "Longitute": ("lon", "float64"),
    "Latitude": ("lat", "float64"),
    "Speed": ("speed_kmh", "float32"),
    "Direction": ("heading_deg", "float32"),
}
BOOL_COLUMNS = {"Di1": "di1", "Di2": "di2", "Di3": "di3"}

INTERIM_DICTIONARY = [
    {
        "column": "vehicle_id",
        "attribute_type": "nominal",
        "values": "ids between 1 and 119, with gaps",
        "description": "Taxi id, from the raw file name. Stored as a number, but only a label.",
        "source": "file name",
    },
    {
        "column": "source_row",
        "attribute_type": "ordinal",
        "values": "0 to rows - 1",
        "description": "0-based data row in the raw file. With vehicle_id it points back to one raw line.",
        "source": "row position in the raw file",
    },
    {
        "column": "ts_utc",
        "attribute_type": "interval",
        "values": "June 1 to August 31, 2023",
        "description": "Time of the point in UTC, used for storage, ordering, and time differences. "
        "Differences make sense, ratios do not.",
        "source": "DeviceDateTime, read on the device clock (DEVICE_TZ)",
    },
    {
        "column": "ts_local",
        "attribute_type": "interval",
        "values": "June 1 to August 31, 2023, UTC+2",
        "description": "The same instant in Kosovo local time, used for everything calendar-related "
        "(date, hour, weekday).",
        "source": "ts_utc",
    },
    {
        "column": "ts_has_seconds",
        "attribute_type": "nominal",
        "values": "true / false / NA",
        "description": "False when the raw timestamp has minutes only: vehicles 1 and 100 on days 1-12 "
        "of each month, where the files were re-saved in a spreadsheet. Several points can then "
        "share a timestamp. NA when the timestamp did not parse.",
        "source": "format of DeviceDateTime",
    },
    {
        "column": "lon",
        "unit": "degrees (WGS 84)",
        "attribute_type": "interval",
        "values": "-180 to 180",
        "description": "Longitude.",
        "source": "Longitute (sic)",
    },
    {
        "column": "lat",
        "unit": "degrees (WGS 84)",
        "attribute_type": "interval",
        "values": "-90 to 90",
        "description": "Latitude.",
        "source": "Latitude",
    },
    {
        "column": "speed_kmh",
        "unit": "km/h",
        "attribute_type": "ratio",
        "values": "0 or more",
        "description": "Speed reported by the device. 0 means standing still and 60 km/h is twice "
        "30 km/h, so ratios make sense.",
        "source": "Speed",
    },
    {
        "column": "heading_deg",
        "unit": "degrees",
        "attribute_type": "interval, circular",
        "values": "0 to 360",
        "description": "Direction of travel. Circular: the mean of 350 and 10 degrees is 0, not 180.",
        "source": "Direction",
    },
    {
        "column": "di1",
        "attribute_type": "nominal",
        "values": "true / false / NA",
        "description": "Digital input 1, binary: the ignition is on (dataset author).",
        "source": "Di1",
    },
    {
        "column": "di2",
        "attribute_type": "nominal",
        "values": "true / false / NA",
        "description": "Digital input 2, binary: the taximeter is on (dataset author, from memory; "
        "the data agrees). The signal drops out for a few seconds at a time.",
        "source": "Di2",
    },
    {
        "column": "di3",
        "attribute_type": "nominal",
        "values": "true / false / NA",
        "description": "Digital input 3, binary: a passenger is in the vehicle (dataset author, "
        "from memory). Stuck true most of the time in 25 of the 83 vehicles.",
        "source": "Di3",
    },
]


def read_raw(vehicle_id: int) -> tuple[pd.DataFrame, dict]:
    """Parse one raw CSV into the interim schema. Returns the points and a summary row.

    Every raw row is kept, in file order. A value that does not parse becomes NA and is
    counted in the summary.
    """
    raw = pd.read_csv(
        io.raw_path(vehicle_id), dtype=str, keep_default_na=False, encoding="utf-8"
    )
    if list(raw.columns) != config.RAW_COLUMNS:
        raise PipelineError(f"{vehicle_id}.csv has columns {list(raw.columns)}")
    n = len(raw)

    text = raw["DeviceDateTime"]
    with_seconds = pd.to_datetime(
        text, format=config.RAW_TS_FORMAT_SECONDS, errors="coerce"
    )
    minutes_only = pd.to_datetime(
        text.where(with_seconds.isna()),
        format=config.RAW_TS_FORMAT_MINUTES,
        errors="coerce",
    )
    ts_device = with_seconds.fillna(minutes_only)
    ts_utc = ts_device.dt.tz_localize(
        config.DEVICE_TZ, ambiguous="raise", nonexistent="raise"
    ).dt.tz_convert("UTC")
    has_seconds = with_seconds.notna().astype("boolean")
    has_seconds[ts_device.isna()] = pd.NA

    df = pd.DataFrame(
        {
            "vehicle_id": np.full(n, vehicle_id, dtype="int16"),
            "source_row": np.arange(n, dtype="int32"),
            "ts_utc": ts_utc,
            "ts_local": ts_utc.dt.tz_convert(config.TZ_LOCAL),
            "ts_has_seconds": has_seconds,
        }
    )
    outside = (ts_device < config.STUDY_START) | (ts_device >= config.STUDY_END)
    summary = {
        "vehicle_id": vehicle_id,
        "rows": n,
        "ts_with_seconds": int(with_seconds.notna().sum()),
        "ts_minutes_only": int((with_seconds.isna() & minutes_only.notna()).sum()),
        "ts_unparsed": int(ts_device.isna().sum()),
        "ts_outside_window": int(outside.sum()),
    }

    numeric_failed = pd.Series(False, index=raw.index)
    for raw_name, (name, dtype) in NUMERIC_COLUMNS.items():
        values = pd.to_numeric(raw[raw_name], errors="coerce").astype(dtype)
        empty = raw[raw_name].str.strip() == ""
        failed = values.isna() & ~empty
        df[name] = values
        summary[f"{name}_empty"] = int(empty.sum())
        summary[f"{name}_unparsed"] = int(failed.sum())
        numeric_failed |= failed

    bool_failed = pd.Series(False, index=raw.index)
    for raw_name, name in BOOL_COLUMNS.items():
        values = raw[raw_name].map(config.RAW_BOOL_VALUES).astype("boolean")
        empty = raw[raw_name].str.strip() == ""
        failed = values.isna() & ~empty
        df[name] = values
        summary[f"{name}_empty"] = int(empty.sum())
        summary[f"{name}_unparsed"] = int(failed.sum())
        bool_failed |= failed

    summary["rows_numeric_unparsed"] = int(numeric_failed.sum())
    summary["rows_bool_unparsed"] = int(bool_failed.sum())
    return df, summary


def run(dev: bool) -> None:
    """Raw CSV to typed Parquet (interim/points), one file per vehicle, every row kept."""
    inventory_rows = pd.read_csv(config.RAW_INVENTORY_CSV).set_index("vehicle_id")[
        "rows"
    ]
    ids = io.vehicle_ids("raw")
    for stale in sorted(set(io.vehicle_ids("interim")) - set(ids)):
        io.points_path("interim", stale).unlink()

    summaries = []
    for v in ids:
        df, summary = read_raw(v)
        io.count_rows(rows_in=len(df))
        if len(df) != inventory_rows[v]:
            raise PipelineError(
                f"Vehicle {v}: {len(df):,} rows parsed, raw_inventory.csv has {inventory_rows[v]:,}."
            )
        _log_parse_failures(v, summary)
        io.write_points(df, "interim")
        summaries.append(summary)

    table = pd.DataFrame(summaries)
    io.write_report_table(table, "ingest_summary")
    io.write_dictionary(INTERIM_DATASET, INTERIM_DICTIONARY, df, owner="ingest")
    largest = max(ids, key=lambda v: inventory_rows[v])
    io.write_report_table(memory_comparison(largest), "dtypes_memory")

    print(
        f"{len(ids)} vehicles, {int(table['rows'].sum()):,} rows: "
        f"{int(table['ts_minutes_only'].sum()):,} timestamps without seconds, "
        f"{int(table['ts_unparsed'].sum()):,} unparsed, "
        f"{int(table['ts_outside_window'].sum()):,} outside the study window, "
        f"{int(table['rows_numeric_unparsed'].sum()):,} rows with an unparsed number, "
        f"{int(table['rows_bool_unparsed'].sum()):,} with an unparsed boolean."
    )


def _log_parse_failures(vehicle_id: int, summary: dict) -> None:
    rows = summary["rows"]
    io.log_step(
        "ingest",
        "I01",
        "set_na",
        vehicle_id,
        rows,
        summary["ts_unparsed"],
        "DeviceDateTime matches neither raw format",
    )
    io.log_step(
        "ingest",
        "I02",
        "set_na",
        vehicle_id,
        rows,
        summary["rows_numeric_unparsed"],
        "Longitute, Latitude, Speed, or Direction is not a number",
    )
    io.log_step(
        "ingest",
        "I03",
        "set_na",
        vehicle_id,
        rows,
        summary["rows_bool_unparsed"],
        "Di1, Di2, or Di3 is not True/False/TRUE/FALSE (empty fields stay NA, not logged)",
    )


def memory_comparison(vehicle_id: int) -> pd.DataFrame:
    """Size of one vehicle's data as CSV, as pandas guesses it, typed, and as Parquet."""
    path = io.raw_path(vehicle_id)
    guessed = pd.read_csv(path)
    typed, _ = read_raw(vehicle_id)
    rows = [
        ("CSV file on disk", path.stat().st_size, "text"),
        (
            "pandas, read_csv with guessed dtypes",
            int(guessed.memory_usage(deep=True).sum()),
            ", ".join(f"{c}: {t}" for c, t in guessed.dtypes.astype(str).items()),
        ),
        (
            "pandas, typed interim schema",
            int(typed.memory_usage(deep=True).sum()),
            ", ".join(f"{c}: {t}" for c, t in typed.dtypes.astype(str).items()),
        ),
        (
            "Parquet file on disk (interim)",
            io.points_path("interim", vehicle_id).stat().st_size,
            "typed interim schema, snappy compression",
        ),
    ]
    out = pd.DataFrame(rows, columns=["representation", "bytes", "dtypes"])
    out.insert(0, "vehicle_id", vehicle_id)
    out.insert(1, "rows", len(typed))
    out.insert(4, "mb", (out["bytes"] / 1_000_000).round(2))
    return out

"""Data collection: verify the raw files against the committed inventory.

The first run of verify-raw writes reports/tables/raw_inventory.csv (T1). Every later run
compares data/raw/ against it and stops on any difference, so everyone on the team can
show they work from the same bytes.
"""

import csv
import hashlib
from pathlib import Path

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

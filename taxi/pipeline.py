"""Command line for the pipeline.

    python -m taxi.pipeline verify-raw                  check data/raw/ against raw_inventory.csv
    python -m taxi.pipeline run [--stage NAME] [--dev]  run every stage in order, or one stage
    python -m taxi.pipeline check [--dev]               reconcile rows, planned outputs, dictionary

Each stage is one function `(dev: bool) -> None` in its module.
A stage whose module or function does not exist yet stops the run with a clear message.
"""

import argparse
import csv
import importlib
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from taxi import PipelineError, config, io

# Stage name -> (module, function), in run order.
STAGES: dict[str, tuple[str, str]] = {
    "verify-raw": ("taxi.ingest", "verify_raw"),
    "ingest": ("taxi.ingest", "run"),
    "profile-before": ("taxi.quality", "run_before"),
    "clean": ("taxi.clean", "run"),
    "profile-after": ("taxi.quality", "run_after"),
    "aggregate": ("taxi.aggregate", "run"),
    "sample": ("taxi.sample", "run"),
    "features": ("taxi.features", "run"),
    "trips": ("taxi.trips", "run"),
    "transform": ("taxi.transform", "run"),
    "reduce": ("taxi.reduce", "run"),
    "export": ("taxi.export", "run"),
}

# The last stage that writes each points dataset. Rows dropped up to that stage, as logged
# in cleaning_log.csv, plus the rows kept must add up to the raw rows.
POINTS_WRITERS = {"interim": "ingest", "clean": "clean", "final": "trips"}

RUN_LOG_COLUMNS = [
    "run_id",
    "stage",
    "dev",
    "started_utc",
    "finished_utc",
    "rows_in",
    "rows_out",
    "git_commit",
]

# Tables and figures of the Phase I report. File names are fixed here so check can find them.
TABLES = [
    ("T2", "data_dictionary.csv"),
    ("T3", "quality_before.csv"),
    ("T3", "quality_after.csv"),
    ("T4", "cleaning_log.csv"),
    ("T5", "missing_strategy.csv"),
    ("T5", "imputation_eval.csv"),
    ("T6", "integration_summary.csv"),
    ("T7", "aggregation_summary.csv"),
    ("T8", "sampling_evaluation.csv"),
    ("T9", "feature_catalog.csv"),
    ("T10", "discretization_bins.csv"),
    ("T11", "feature_selection.csv"),
    ("T12", "pca_explained_variance.csv"),
    ("T12", "pca_loadings.csv"),
]
FIGURES = [
    ("F1", "f01_rows_per_vehicle.png"),
    ("F2", "f02_coverage_vehicle_day.png"),
    ("F3", "f03_sampling_interval.png"),
    ("F4", "f04_raw_points_map.png"),
    ("F5", "f05_reported_vs_implied_speed.png"),
    ("F6", "f06_cleaning_funnel.png"),
    ("F7", "f07_hourly_activity.png"),
    ("F8", "f08_samples_vs_full.png"),
    ("F9", "f09_transformations.png"),
    ("F10", "f10_discretization.png"),
    ("F11", "f11_feature_correlation.png"),
    ("F12", "f12_pca_scree.png"),
]
EXPORT_FILES = [
    "points",
    "trips.parquet",
    "trips.csv",
    "vehicle_day.csv",
    "cell_hour.csv",
    "fleet_hour.csv",
    "model/trips_model.parquet",
    "model/vehicle_day_model.parquet",
    "points_sample.csv",
    "data_dictionary.csv",
    "README_data.md",
]


def sample_names() -> list[str]:
    """Planned sample tables, named <method>_<size>."""
    names = [
        f"srs_{0.01:g}",
        f"systematic_{config.THIN_S}s",
        f"cluster_{config.CLUSTER_N_VEHICLES}veh",
    ]
    names += [f"stratified_{size:g}" for size in config.SAMPLE_SIZES]
    return names


# --- run ------------------------------------------------------------------------------


def stage_function(name: str) -> Callable[[bool], None] | None:
    """The function behind a stage, or None when its module or function is not written yet."""
    module_name, func_name = STAGES[name]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return None
        raise
    return getattr(module, func_name, None)


def run(stage: str | None, dev: bool) -> None:
    """Run one stage, or every stage in order. A full run rewrites cleaning_log.csv from scratch."""
    config.set_dev(dev)
    run_id = io.new_run_id()
    io.set_run_id(run_id)
    commit = git_commit()
    names = [stage] if stage else list(STAGES)
    io.clear_log(stage)
    print(
        f"Run {run_id} ({'dev: vehicles ' + str(config.DEV_VEHICLES) if dev else 'full'}), commit {commit}"
    )

    for name in names:
        fn = stage_function(name)
        if fn is None:
            module_name, func_name = STAGES[name]
            raise PipelineError(
                f"Stage {name} is not implemented yet ({module_name}.{func_name}). Stopped here."
            )
        print(f"[{name}] started")
        io.reset_row_counts()
        started = datetime.now(UTC)
        try:
            fn(dev)
        except PipelineError as exc:
            raise PipelineError(f"[{name}] {exc}") from exc
        finished = datetime.now(UTC)
        counts = io.row_counts()
        _append_run_log(
            {
                "run_id": run_id,
                "stage": name,
                "dev": dev,
                "started_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "finished_utc": finished.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "rows_in": counts["rows_in"],
                "rows_out": counts["rows_out"],
                "git_commit": commit,
            }
        )
        seconds = (finished - started).total_seconds()
        print(
            f"[{name}] done in {seconds:.1f} s: {counts['rows_in']:,} rows in, {counts['rows_out']:,} rows out"
        )


def git_commit() -> str:
    """HEAD of the repository, with -dirty when the code (taxi/, requirements.txt) has uncommitted changes."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=config.ROOT, capture_output=True, text=True, check=True
        ).stdout

    try:
        head = git("rev-parse", "HEAD").strip()
    except (OSError, subprocess.CalledProcessError):
        return "none"
    dirty = git("status", "--porcelain", "--", "taxi", "requirements.txt").strip()
    return f"{head}-dirty" if dirty else head


def _append_run_log(row: dict) -> None:
    path = config.run_log_csv()
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RUN_LOG_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


# --- check ----------------------------------------------------------------------------


def check(dev: bool) -> int:
    """Verify the outputs of a run. Prints every problem and returns 1 when there is any."""
    config.set_dev(dev)
    sections = {
        "Row reconciliation": _check_reconciliation(dev),
        "Planned outputs": _check_planned_outputs(),
        "Schemas and data dictionary": _check_schemas_and_dictionary(),
    }
    failed = False
    for title, problems in sections.items():
        print(f"{title}: {'OK' if not problems else f'{len(problems)} problem(s)'}")
        for p in problems:
            print(f"  - {p}")
        failed = failed or bool(problems)
    print("check FAILED" if failed else "check passed")
    return 1 if failed else 0


def _check_reconciliation(dev: bool) -> list[str]:
    if not config.RAW_INVENTORY_CSV.exists():
        return ["raw_inventory.csv is missing: run verify-raw first"]
    inv = pd.read_csv(config.RAW_INVENTORY_CSV)
    raw_rows = inv[inv["bytes"] > 0].set_index("vehicle_id")["rows"]
    expected = sorted(v for v in raw_rows.index if not dev or v in config.DEV_VEHICLES)
    drops = io.read_log().query("action == 'drop'")
    order = list(STAGES)

    problems = []
    for points_stage, writer in POINTS_WRITERS.items():
        present = io.vehicle_ids(points_stage)
        if not present:
            problems.append(f"no {points_stage} points yet (written by stage {writer})")
            continue
        upto = order[: order.index(writer) + 1]
        dropped = (
            drops[drops["stage"].isin(upto)]
            .groupby("vehicle_id")["rows_affected"]
            .sum()
        )
        missing = [v for v in expected if v not in present]
        if missing:
            problems.append(f"{points_stage}: no file for vehicles {missing}")
        extra = [v for v in present if v not in expected]
        if extra:
            problems.append(f"{points_stage}: unexpected files for vehicles {extra}")
        for v in expected:
            if v not in present:
                continue
            kept = pq.ParquetFile(io.points_path(points_stage, v)).metadata.num_rows
            gone = int(dropped.get(v, 0))
            if kept + gone != raw_rows[v]:
                problems.append(
                    f"{points_stage}, vehicle {v}: {kept:,} kept + {gone:,} dropped = {kept + gone:,}, "
                    f"raw has {raw_rows[v]:,}"
                )
    return problems


def planned_outputs() -> list[tuple[str, str, Path]]:
    """(kind, label, path) of every planned output, at the current (dev) paths."""
    out = [("tables", "T1", config.RAW_INVENTORY_CSV)]
    out += [("tables", tid, config.tables_dir() / name) for tid, name in TABLES]
    out += [("figures", fid, config.figures_dir() / name) for fid, name in FIGURES]
    out.append(("logs", "run_log", config.run_log_csv()))
    for name in ["trips", "vehicle_day", "cell_hour", "fleet_hour"]:
        out.append(("datasets", name, io.table_path(name)))
    out += [
        ("datasets", "samples", io.table_path(f"samples/{name}"))
        for name in sample_names()
    ]
    out += [
        ("datasets", "model", io.table_path(f"model/{name}"))
        for name in ("trips_model", "vehicle_day_model")
    ]
    out.append(("datasets", "points_sample", config.sample_csv()))
    out += [("export", "export", config.export_dir() / name) for name in EXPORT_FILES]
    return out


def _check_planned_outputs() -> list[str]:
    missing: dict[str, list[str]] = {}
    for kind, label, path in planned_outputs():
        if not path.exists():
            missing.setdefault(kind, []).append(
                f"{label} {path.relative_to(config.ROOT)}"
            )
    return [f"missing {kind}: {', '.join(items)}" for kind, items in missing.items()]


def dataset_files() -> dict[str, list[Path]]:
    """Files of every dataset described by the data dictionary, keyed by the dictionary's dataset name.

    The dataset name is the path under data/ without the extension.
    Exported files count under the dataset they copy.
    """
    files: dict[str, list[Path]] = {}
    for stage, sub in config.POINTS_SUBDIRS.items():
        files[sub] = sorted(config.points_dir(stage).glob("*.parquet"))
    for name in [
        "trips",
        "vehicle_day",
        "cell_hour",
        "fleet_hour",
        "model/trips_model",
        "model/vehicle_day_model",
    ]:
        files[f"processed/{name}"] = [io.table_path(name)]
    files["sample/points_sample"] = [config.sample_csv()]

    exp = config.export_dir()
    files["processed/points"] += sorted((exp / "points").glob("*.parquet"))
    files["processed/trips"] += [exp / "trips.parquet", exp / "trips.csv"]
    for name in ["vehicle_day", "cell_hour", "fleet_hour"]:
        files[f"processed/{name}"].append(exp / f"{name}.csv")
    for name in ["trips_model", "vehicle_day_model"]:
        files[f"processed/model/{name}"].append(exp / "model" / f"{name}.parquet")
    files["sample/points_sample"].append(exp / "points_sample.csv")
    return {k: [p for p in v if p.exists()] for k, v in files.items()}


def file_columns(path: Path) -> list[str]:
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as fh:
            return next(csv.reader(fh), [])
    return [c for c in pq.read_schema(path).names if not c.startswith("__index_level_")]


def _check_schemas_and_dictionary() -> list[str]:
    problems = []
    datasets = {k: v for k, v in dataset_files().items() if v}

    # Every per-vehicle file of a points dataset has the same columns and types.
    for stage, sub in config.POINTS_SUBDIRS.items():
        paths = sorted(config.points_dir(stage).glob("*.parquet"))
        if len(paths) > 1:
            first = pq.read_schema(paths[0]).remove_metadata()
            odd = [
                p.stem
                for p in paths[1:]
                if not pq.read_schema(p).remove_metadata().equals(first)
            ]
            if odd:
                problems.append(
                    f"{sub}: schema differs from vehicle {paths[0].stem} in vehicles {odd}"
                )

    if not datasets:
        return problems
    dict_path = config.data_dictionary_csv()
    if not dict_path.exists():
        return [
            *problems,
            f"{dict_path.relative_to(config.ROOT)} is missing, so no column is documented",
        ]
    dictionary = pd.read_csv(dict_path, dtype=str, keep_default_na=False)
    documented = dictionary.groupby("dataset")["column"].apply(set).to_dict()
    for dataset, paths in sorted(datasets.items()):
        columns = {c for p in paths for c in file_columns(p)}
        undocumented = sorted(columns - documented.get(dataset, set()))
        if undocumented:
            problems.append(
                f"{dataset}: no dictionary row for {', '.join(undocumented)}"
            )
    return problems


# --- command line ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m taxi.pipeline", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "verify-raw", help="check data/raw/ against reports/tables/raw_inventory.csv"
    )
    run_parser = sub.add_parser(
        "run", help="run every stage in order, or one with --stage"
    )
    run_parser.add_argument("--stage", choices=list(STAGES), help="run only this stage")
    run_parser.add_argument(
        "--dev",
        action="store_true",
        help="only DEV_VEHICLES, outputs under data/dev, reports/dev",
    )
    check_parser = sub.add_parser(
        "check", help="row reconciliation, planned outputs, data dictionary"
    )
    check_parser.add_argument(
        "--dev", action="store_true", help="check the outputs of a dev run"
    )
    args = parser.parse_args(argv)

    try:
        if args.command == "verify-raw":
            stage_function("verify-raw")(False)
        elif args.command == "run":
            run(args.stage, args.dev)
        else:
            return check(args.dev)
    except PipelineError as exc:
        sys.stdout.flush()
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

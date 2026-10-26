"""Sampling methods, their evaluation against the full data (T8, F8), and the reviewer sample.

Samples are drawn from processed/points_clean one vehicle at a time, with SEED, and keep
its columns. Methods:

- srs: simple random sample, every point equally likely, exact size round(frac * N);
- stratified: the same fraction from every stratum (vehicle x local hour), at least one
  row per stratum (groupby().sample(frac=...) would round small strata down to zero);
- systematic: the first point of every THIN_S-second window, per vehicle;
- cluster: all points of min(CLUSTER_N_VEHICLES, available) random vehicles.

The evaluation compares each sample with the full data on what Phases II and III use most:
the speed distribution (Kolmogorov-Smirnov statistic), the share of points per hour of
day, the moving share, and the vehicles covered.

The reviewer sample (data/sample/points_sample.csv) is public. Night-time rows with the
ignition off show where a taxi parks overnight, so that file withholds their position
(rule S01); the samples under data/processed/samples/ keep every value.
"""

from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from taxi import PipelineError, config, figures, io

SAMPLE_CSV_MAX_BYTES = 5_000_000
SRS_FRACTION = 0.01
# Withheld (NA) in the reviewer sample on night-time rows with the ignition off (rule S01).
WITHHELD_COLUMNS = ["lat", "lon", "cell_id"]


def _vehicles(stage: io.Stage) -> list[int]:
    ids = io.vehicle_ids(stage)
    if not ids:
        raise PipelineError(
            f"No {stage} points in {config.points_dir(stage)}: run the stage that writes them."
        )
    return ids


def _each_vehicle(stage: io.Stage) -> Iterator[tuple[int, pd.DataFrame]]:
    for v in _vehicles(stage):
        yield v, io.read_points(stage, [v])


def srs(frac: float, stage: io.Stage = "clean") -> pd.DataFrame:
    """Simple random sample of exactly round(frac * N) points of the whole fleet."""
    ids = _vehicles(stage)
    sizes = np.array(
        [pq.ParquetFile(io.points_path(stage, v)).metadata.num_rows for v in ids]
    )
    total = int(sizes.sum())
    rng = np.random.default_rng(config.SEED)
    picked = np.sort(rng.choice(total, size=round(frac * total), replace=False))
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    parts = []
    for v, start, size in zip(ids, starts, sizes, strict=True):
        local = picked[(picked >= start) & (picked < start + size)] - start
        if len(local):
            parts.append(io.read_points(stage, [v]).iloc[local])
    return pd.concat(parts, ignore_index=True)


def stratified(frac: float, stage: io.Stage = "clean") -> pd.DataFrame:
    """The same fraction from every vehicle x local-hour stratum, at least one row each."""
    rng = np.random.default_rng(config.SEED)
    parts = []
    for _, df in _each_vehicle(stage):
        hour = df["ts_local"].dt.hour
        size = hour.map(hour.value_counts())
        wanted = np.maximum(1, np.round(frac * size)).astype("int64")
        key = pd.Series(rng.random(len(df)), index=df.index)
        rank = key.groupby(hour).rank(method="first")
        parts.append(df[rank <= wanted])
    return pd.concat(parts, ignore_index=True)


def systematic(thin_s: int, stage: io.Stage = "clean") -> pd.DataFrame:
    """The first point of every thin_s-second window of each vehicle (no randomness)."""
    parts = []
    for _, df in _each_vehicle(stage):
        d = df.sort_values(["ts_utc", "source_row"], kind="stable")
        window = d["ts_utc"].astype("int64") // (thin_s * 1_000_000_000)
        parts.append(d[~window.duplicated()])
    return pd.concat(parts, ignore_index=True)


def cluster(n_vehicles: int, stage: io.Stage = "clean") -> pd.DataFrame:
    """All points of min(n_vehicles, available) vehicles chosen at random."""
    ids = _vehicles(stage)
    rng = np.random.default_rng(config.SEED)
    chosen = sorted(
        rng.choice(ids, size=min(n_vehicles, len(ids)), replace=False).tolist()
    )
    return io.read_points(stage, chosen)


# --- evaluation -----------------------------------------------------------------------


def _summary(df: pd.DataFrame) -> dict:
    """What a sample is compared on, in a form that adds up across vehicles."""
    speed = df["speed_kmh"].dropna()
    return {
        "rows": len(df),
        "vehicles": set(df["vehicle_id"].unique().tolist()),
        "speed_counts": speed.round(1).value_counts(),
        "hour_counts": df["ts_local"]
        .dt.hour.value_counts()
        .reindex(range(24), fill_value=0),
        "moving": int((speed >= config.V_MOVE_KMH).sum()),
        "speed_n": len(speed),
        "speed_sum": float(speed.astype("float64").sum()),
    }


def _merge(a: dict, b: dict) -> dict:
    return {
        "rows": a["rows"] + b["rows"],
        "vehicles": a["vehicles"] | b["vehicles"],
        "speed_counts": a["speed_counts"].add(b["speed_counts"], fill_value=0),
        "hour_counts": a["hour_counts"] + b["hour_counts"],
        "moving": a["moving"] + b["moving"],
        "speed_n": a["speed_n"] + b["speed_n"],
        "speed_sum": a["speed_sum"] + b["speed_sum"],
    }


def _ks(sample_counts: pd.Series, full_counts: pd.Series) -> float:
    """Two-sample Kolmogorov-Smirnov statistic from value counts: the largest ECDF gap."""
    values = sample_counts.index.union(full_counts.index).sort_values()
    s = sample_counts.reindex(values, fill_value=0).cumsum() / sample_counts.sum()
    f = full_counts.reindex(values, fill_value=0).cumsum() / full_counts.sum()
    return float((s - f).abs().max())


def evaluate(name: str, method: str, size: str, sample: dict, full: dict) -> dict:
    hour_s = sample["hour_counts"] / sample["rows"]
    hour_f = full["hour_counts"] / full["rows"]
    moving_s = sample["moving"] / sample["speed_n"]
    moving_f = full["moving"] / full["speed_n"]
    return {
        "sample": name,
        "method": method,
        "size": size,
        "rows": sample["rows"],
        "share_of_full": round(sample["rows"] / full["rows"], 6),
        "vehicles": len(sample["vehicles"]),
        "vehicles_full": len(full["vehicles"]),
        "ks_speed": round(_ks(sample["speed_counts"], full["speed_counts"]), 5),
        "hour_share_max_diff_pp": round(100 * float((hour_s - hour_f).abs().max()), 3),
        "hour_share_tvd": round(0.5 * float((hour_s - hour_f).abs().sum()), 5),
        "moving_share": round(moving_s, 5),
        "moving_share_full": round(moving_f, 5),
        "mean_speed_kmh": round(sample["speed_sum"] / sample["speed_n"], 3),
        "mean_speed_kmh_full": round(full["speed_sum"] / full["speed_n"], 3),
    }


def sample_plan() -> list[tuple[str, str, str, Callable[[io.Stage], pd.DataFrame]]]:
    """(file name, method, size label, draw function) for every planned sample."""
    plan = [
        (
            f"srs_{SRS_FRACTION:g}",
            "srs",
            f"{SRS_FRACTION:g}",
            lambda st: srs(SRS_FRACTION, st),
        ),
        (
            f"systematic_{config.THIN_S}s",
            "systematic",
            f"{config.THIN_S}s",
            lambda st: systematic(config.THIN_S, st),
        ),
        (
            f"cluster_{config.CLUSTER_N_VEHICLES}veh",
            "cluster",
            f"{config.CLUSTER_N_VEHICLES}veh",
            lambda st: cluster(config.CLUSTER_N_VEHICLES, st),
        ),
    ]
    for frac in config.SAMPLE_SIZES:
        plan.append(
            (
                f"stratified_{frac:g}",
                "stratified",
                f"{frac:g}",
                lambda st, f=frac: stratified(f, st),
            )
        )
    return plan


# --- stage ----------------------------------------------------------------------------


def run(dev: bool) -> None:
    """Draw every planned sample from points_clean, evaluate it (T8, F8), write the reviewer CSV."""
    build(stage="clean", dictionary_dataset="processed/points_clean")


def build(stage: io.Stage, dictionary_dataset: str) -> pd.DataFrame:
    """The sample stage on any points dataset; run() uses points_clean."""
    for old in config.samples_dir().glob("*.parquet"):
        old.unlink()

    full = None
    for _, df in _each_vehicle(stage):
        part = _summary(df)
        full = part if full is None else _merge(full, part)

    rows, hour_profiles = [], {"full data": full["hour_counts"] / full["rows"]}
    for name, method, size, draw in sample_plan():
        sample = draw(stage)
        io.write_table(sample, f"samples/{name}")
        summary = _summary(sample)
        rows.append(evaluate(name, method, size, summary, full))
        if method != "stratified" or size == f"{SRS_FRACTION:g}":
            hour_profiles[name] = summary["hour_counts"] / summary["rows"]
    table = pd.DataFrame(rows)
    io.write_report_table(table, "sampling_evaluation")
    _figure_samples_vs_full(table, hour_profiles)

    reviewer = _reviewer_sample(stage, full["rows"])
    rows_by_column = io.read_dictionary()
    described = rows_by_column[rows_by_column["dataset"] == dictionary_dataset]
    if described.empty:
        raise PipelineError(
            f"No data dictionary rows for {dictionary_dataset}; its stage writes them."
        )
    rows = described.drop(columns=["dataset", "dtype"]).to_dict("records")
    for row in rows:
        if row["column"] in WITHHELD_COLUMNS:
            row["description"] += " NA in this file when coords_withheld is true."
    start, end = config.NIGHT_HOURS
    rows.append(
        {
            "column": "coords_withheld",
            "unit": "",
            "attribute_type": "nominal",
            "values": "true / false",
            "description": (
                f"lat, lon, and cell_id are withheld (NA): the row is at night ({start:02d}:00 to "
                f"{end:02d}:00 local) with the ignition off, so it would show where the taxi "
                "parks overnight (rule S01). Only in this public file."
            ),
            "source": "hour, di1",
            "owner": "sample",
        }
    )
    io.write_dictionary("sample/points_sample", rows, reviewer)
    return table


def withhold_overnight(df: pd.DataFrame) -> pd.DataFrame:
    """A copy of df with lat, lon, and cell_id set to NA on night-time rows with the ignition off.

    Adds coords_withheld. A row with an unknown ignition (di1 NA) keeps its position.
    """
    start, end = config.NIGHT_HOURS
    night = ((df["hour"] >= start) & (df["hour"] < end)).to_numpy(bool)
    parked = df["di1"].eq(False).fillna(False).to_numpy(bool)
    withheld = night & parked
    out = df.copy()
    out.loc[withheld, ["lat", "lon"]] = np.nan
    out.loc[withheld, "cell_id"] = pd.NA
    out["coords_withheld"] = withheld
    return out


def _reviewer_sample(stage: io.Stage, total_rows: int) -> pd.DataFrame:
    """A separate stratified draw sized so the CSV stays under 5 MB (never a cut of a larger sample)."""
    path = config.sample_csv()
    path.parent.mkdir(parents=True, exist_ok=True)
    first = io.read_points(stage, [_vehicles(stage)[0]])
    probe = first.head(2000).to_csv(index=False).encode("utf-8")
    bytes_per_row = len(probe) / min(2000, len(first))
    strata = 24 * len(_vehicles(stage))
    target_rows = 0.9 * SAMPLE_CSV_MAX_BYTES / bytes_per_row
    frac = max(1e-6, (target_rows - strata) / total_rows)
    for _ in range(6):
        sample = withhold_overnight(stratified(frac, stage))
        sample.to_csv(path, index=False, encoding="utf-8")
        size = path.stat().st_size
        if size < SAMPLE_CSV_MAX_BYTES:
            withheld = int(sample["coords_withheld"].sum())
            io.log_step(
                "sample",
                "S01",
                "set_na",
                None,
                len(sample),
                withheld,
                "points_sample.csv: lat, lon, cell_id withheld on night-time rows with the ignition off",
            )
            print(
                f"points_sample.csv: {len(sample):,} rows, {size:,} bytes, fraction {frac:.6f}, "
                f"position withheld on {withheld:,} rows"
            )
            return sample
        frac *= 0.8 * SAMPLE_CSV_MAX_BYTES / size
    raise PipelineError("Could not draw a reviewer sample under 5 MB.")


def _figure_samples_vs_full(
    table: pd.DataFrame, hour_profiles: dict[str, pd.Series]
) -> None:
    """F8: hour-of-day profile of each sample against the full data, and progressive sampling."""
    fig, (ax_hour, ax_ks, ax_tvd) = figures.subplots(
        13, 3.9, ncols=3, width_ratios=[2.2, 1, 1]
    )

    full = hour_profiles.pop("full data")
    ax_hour.plot(
        full.index,
        100 * full.to_numpy(),
        color=figures.INK,
        linewidth=3,
        label="full data",
    )
    for color, (name, profile) in zip(
        figures.SERIES, hour_profiles.items(), strict=False
    ):
        ax_hour.plot(
            profile.index,
            100 * profile.to_numpy(),
            color=color,
            linewidth=1.6,
            label=name,
        )
    ax_hour.set_xticks(range(0, 24, 3))
    ax_hour.set_xlim(-0.5, 23.5)
    ax_hour.set_xlabel("local hour of day")
    ax_hour.set_ylabel("share of points (%)")
    ax_hour.set_title("Points per hour of day")
    ax_hour.legend(loc="lower center", ncols=2)

    prog = (
        table[table["method"] == "stratified"]
        .assign(frac=lambda t: t["size"].astype(float))
        .sort_values("frac")
    )
    others = table[table["method"] != "stratified"]
    for ax, col, label in [
        (ax_ks, "ks_speed", "KS statistic, speed"),
        (ax_tvd, "hour_share_tvd", "total variation, hour shares"),
    ]:
        ax.plot(
            prog["frac"],
            prog[col],
            color=figures.SERIES[3],
            marker="o",
            markersize=5,
            label="stratified",
        )
        for _, row in others.iterrows():
            ax.axhline(row[col], color=figures.INK_MUTED, linewidth=1)
            ax.text(
                prog["frac"].max(),
                row[col],
                f" {row['sample']}",
                fontsize=6,
                color=figures.INK_SECONDARY,
                va="bottom",
                ha="right",
            )
        ax.set_xscale("log")
        ax.set_xticks(prog["frac"], [f"{f:g}" for f in prog["frac"]])
        ax.set_xlabel("stratified sample fraction (log scale)")
        ax.set_ylabel(f"{label} (0 = same as full)")
        ax.set_ylim(bottom=0)
        ax.set_title("Progressive sampling" if col == "ks_speed" else " ")
    fig.suptitle(
        "F8. Samples compared with the full data",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=11,
    )
    figures.save_figure(fig, "f08_samples_vs_full")

"""Data quality profile of the points, before and after cleaning (T3), and figures F1-F3.

profile-before reads interim/points and profile-after reads processed/points_clean. Both
go one vehicle at a time and add the counts up for the fleet, so memory stays small.

Each row of a profile is one check on one column, grouped by quality dimension. Count
rows have `n` (rows checked), `count` (rows failing or flagged), and `share`. Value rows
(interval quantiles, speed comparison) have `value` and `unit` instead.
"""

import numpy as np
import pandas as pd

from taxi import PipelineError, config, figures, io

PROFILE_COLUMNS = [
    "dimension",
    "check",
    "column",
    "n",
    "count",
    "share",
    "value",
    "unit",
]
RAW_FIELDS = [
    "vehicle_id",
    "ts_utc",
    "lon",
    "lat",
    "speed_kmh",
    "heading_deg",
    "di1",
    "di2",
    "di3",
]
STUDY_DAYS = (config.STUDY_END - config.STUDY_START).days
# The state of a step is the state of the point it starts at.
STATES = ["moving", "standing, ignition on", "ignition off", "speed unknown"]
QUANTILES = [0.1, 0.5, 0.9, 0.99]


def _haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (
        np.radians(np.asarray(x, dtype="float64")) for x in (lat1, lon1, lat2, lon2)
    )
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6_371_000.0 * np.arcsin(np.sqrt(a))


def step_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One row per point with the step that starts at it, to the next point of the vehicle.

    Uses the cleaning stage's dt_s and dist_m when df has them; otherwise computes them
    from consecutive points sorted by ts_utc, then source_row. `implied_kmh` is set only
    on steps where comparing it with the reported speed makes sense.
    """
    d = df.sort_values(["vehicle_id", "ts_utc", "source_row"], kind="stable")
    nxt = d.groupby("vehicle_id", sort=False)[["ts_utc", "lat", "lon"]].shift(-1)
    if "dt_s" in d.columns and "dist_m" in d.columns:
        dt = d["dt_s"].astype("float64")
        dist = d["dist_m"].astype("float64")
    else:
        dt = (nxt["ts_utc"] - d["ts_utc"]).dt.total_seconds()
        dist = pd.Series(
            _haversine_m(d["lat"], d["lon"], nxt["lat"], nxt["lon"]), index=d.index
        )

    speed = d["speed_kmh"].astype("float64")
    ignition_off = d["di1"].eq(False).fillna(False).to_numpy(dtype=bool)
    state = np.select(
        [
            ignition_off,
            speed.isna().to_numpy(),
            (speed >= config.V_MOVE_KMH).to_numpy(),
        ],
        ["ignition off", "speed unknown", "moving"],
        default="standing, ignition on",
    )

    lo, hi = config.SPEED_CHECK_DT_S
    usable = (
        dt.between(lo, hi)
        & (dist >= config.SPEED_CHECK_MIN_DIST_M)
        & (speed >= config.V_MOVE_KMH)
    )
    if "ts_has_seconds" in d.columns:
        has = d["ts_has_seconds"].fillna(False).astype(bool)
        next_has = has.groupby(d["vehicle_id"], sort=False).shift(-1, fill_value=False)
        usable &= has & next_has
    implied = (dist / dt * 3.6).where(usable)

    return pd.DataFrame(
        {
            "vehicle_id": d["vehicle_id"],
            "dt_s": dt.astype("float32"),
            "state": pd.Categorical(state, categories=STATES),
            "speed_kmh": speed.astype("float32"),
            "implied_kmh": implied.astype("float32"),
        },
        index=d.index,
    )


def _count_rows(df: pd.DataFrame, steps: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []

    def add(dimension: str, check: str, column: str, n, count) -> None:
        rows.append(
            {
                "dimension": dimension,
                "check": check,
                "column": column,
                "n": int(n),
                "count": int(count),
            }
        )

    # Completeness
    for col in df.columns:
        add("completeness", "missing values", col, len(df), df[col].isna().sum())
    local_day = df["ts_local"].dt.tz_localize(None).dt.normalize()
    in_window = local_day.between(
        config.STUDY_START, config.STUDY_END, inclusive="left"
    )
    days_with_data = local_day[in_window].groupby(df["vehicle_id"]).nunique()
    n_vehicle_days = STUDY_DAYS * df["vehicle_id"].nunique()
    add(
        "completeness",
        "vehicle-days without any point",
        "ts_local",
        n_vehicle_days,
        n_vehicle_days - days_with_data.sum(),
    )

    # Validity
    ts = df["ts_utc"].dropna()
    ts_device = ts.dt.tz_convert(config.DEVICE_TZ).dt.tz_localize(None)
    outside = (ts_device < config.STUDY_START) | (ts_device >= config.STUDY_END)
    add("validity", "outside the study window", "ts_utc", len(ts), outside.sum())
    lat, lon = df["lat"].dropna(), df["lon"].dropna()
    add("validity", "outside -90 to 90", "lat", len(lat), (~lat.between(-90, 90)).sum())
    add(
        "validity",
        "outside -180 to 180",
        "lon",
        len(lon),
        (~lon.between(-180, 180)).sum(),
    )
    both = df[["lat", "lon"]].dropna()
    add(
        "validity",
        "latitude and longitude both 0",
        "lat, lon",
        len(both),
        ((both["lat"] == 0) & (both["lon"] == 0)).sum(),
    )
    speed = df["speed_kmh"].dropna()
    add(
        "validity",
        f"below 0 or above {config.SPEED_MAX_KMH} km/h",
        "speed_kmh",
        len(speed),
        ((speed < 0) | (speed > config.SPEED_MAX_KMH)).sum(),
    )
    heading = df["heading_deg"].dropna()
    add(
        "validity",
        "outside 0 to 360",
        "heading_deg",
        len(heading),
        (~heading.between(0, 360)).sum(),
    )

    # Uniqueness
    duplicate = df.duplicated(
        subset=[c for c in RAW_FIELDS if c in df.columns], keep="first"
    )
    same_ts = df.duplicated(subset=["vehicle_id", "ts_utc"], keep=False)
    if "ts_has_seconds" in df.columns:
        groups = [
            ("timestamp with seconds", df["ts_has_seconds"].eq(True).fillna(False)),
            ("timestamp without seconds", df["ts_has_seconds"].eq(False).fillna(False)),
        ]
    else:
        groups = [("all rows", pd.Series(True, index=df.index))]
    for label, mask in groups:
        add(
            "uniqueness",
            f"exact duplicate of an earlier row ({label})",
            "all raw fields",
            mask.sum(),
            (duplicate & mask).sum(),
        )
        add(
            "uniqueness",
            f"another row has the same vehicle and time ({label})",
            "vehicle_id, ts_utc",
            mask.sum(),
            (same_ts & mask).sum(),
        )

    # Consistency between the inputs: with the ignition off, nothing should move or run.
    ignition_known = df["di1"].notna()
    ignition_off = df["di1"].eq(False).fillna(False)
    moving = (df["speed_kmh"] >= config.V_MOVE_KMH).fillna(False) & ignition_known
    add(
        "consistency",
        "moving with the ignition off",
        "speed_kmh, di1",
        moving.sum(),
        (moving & ignition_off).sum(),
    )
    for col, what in [("di2", "taximeter on"), ("di3", "passenger present")]:
        on = df[col].eq(True).fillna(False) & ignition_known
        add(
            "consistency",
            f"{what} with the ignition off",
            f"{col}, di1",
            on.sum(),
            (on & ignition_off).sum(),
        )

    # Timeliness
    dt = steps["dt_s"].dropna()
    add(
        "timeliness",
        f"interval longer than GAP_S ({config.GAP_S} s)",
        "interval to next point",
        len(dt),
        (dt > config.GAP_S).sum(),
    )
    add(
        "timeliness",
        "interval of 0 s (same timestamp)",
        "interval to next point",
        len(dt),
        (dt == 0).sum(),
    )

    # Plausibility
    if "in_kosovo" in df.columns:
        known = df["in_kosovo"].dropna()
        add(
            "plausibility",
            "outside Kosovo (boundary polygon)",
            "in_kosovo",
            len(known),
            (~known.astype(bool)).sum(),
        )
    else:
        lon_min, lat_min, lon_max, lat_max = config.KOSOVO_BBOX
        inside = both["lon"].between(lon_min, lon_max) & both["lat"].between(
            lat_min, lat_max
        )
        add(
            "plausibility",
            "outside Kosovo (bounding box)",
            "lat, lon",
            len(both),
            (~inside).sum(),
        )

    # Flags set by the cleaning stage (profile after cleaning only)
    for col in df.columns:
        if col.startswith("flag_") or col == "speed_imputed":
            flagged = df[col].eq(True).fillna(False)
            add("flags", "flag set", col, len(df), flagged.sum())
    return rows


def _value_rows(steps: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []

    def add(dimension: str, check: str, column: str, n, value, unit: str) -> None:
        rows.append(
            {
                "dimension": dimension,
                "check": check,
                "column": column,
                "n": int(n),
                "value": float(value),
                "unit": unit,
            }
        )

    positive = steps[steps["dt_s"] > 0]
    observed = steps[steps["dt_s"].between(0, config.GAP_S)]
    total_time = observed["dt_s"].astype("float64").sum()
    for state in ["all", *STATES]:
        part = positive if state == "all" else positive[positive["state"] == state]
        if len(part) == 0:
            continue
        for q in QUANTILES:
            add(
                "timeliness",
                f"interval to next point, p{round(q * 100)}",
                state,
                len(part),
                part["dt_s"].quantile(q),
                "s",
            )
        if state != "all" and total_time > 0:
            share = (
                observed.loc[observed["state"] == state, "dt_s"].astype("float64").sum()
                / total_time
            )
            add(
                "timeliness",
                "share of observed time (steps up to GAP_S)",
                state,
                len(part),
                share,
                "share",
            )

    pairs = steps[steps["implied_kmh"].notna()]
    if len(pairs):
        ratio = pairs["speed_kmh"] / pairs["implied_kmh"]
        diff = (pairs["speed_kmh"] - pairs["implied_kmh"]).abs()
        label = "speed_kmh vs implied speed"
        add(
            "consistency",
            "reported / implied speed, median",
            label,
            len(pairs),
            ratio.median(),
            "ratio",
        )
        add(
            "consistency",
            "absolute difference, median",
            label,
            len(pairs),
            diff.median(),
            "km/h",
        )
        add(
            "consistency",
            "absolute difference, p90",
            label,
            len(pairs),
            diff.quantile(0.9),
            "km/h",
        )
    return rows


def _table(count_rows: list[dict], value_rows: list[dict]) -> pd.DataFrame:
    counts = pd.DataFrame(
        count_rows, columns=["dimension", "check", "column", "n", "count"]
    )
    counts["share"] = (counts["count"] / counts["n"].where(counts["n"] > 0)).round(6)
    values = pd.DataFrame(
        value_rows, columns=["dimension", "check", "column", "n", "value", "unit"]
    )
    out = pd.concat([counts, values], ignore_index=True).reindex(
        columns=PROFILE_COLUMNS
    )
    return out.astype({"n": "Int64", "count": "Int64"})


def profile(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Quality checks of a points DataFrame (one or more vehicles), one row per check and column."""
    steps = step_frame(df)
    out = _table(_count_rows(df, steps), _value_rows(steps))
    out.insert(0, "label", label)
    return out


# --- stages ---------------------------------------------------------------------------


def run_before(dev: bool) -> None:
    """Profile interim/points: quality_before.csv, quality_before_by_vehicle.csv, F1-F3."""
    daily = _run("interim", "before cleaning", "quality_before", ingest_rows=True)
    _figure_rows_per_vehicle()
    _figure_coverage(daily)


def run_after(dev: bool) -> None:
    """Profile processed/points_clean: quality_after.csv and quality_after_by_vehicle.csv."""
    _run("clean", "after cleaning", "quality_after", ingest_rows=False)


def _run(stage: str, label: str, table: str, ingest_rows: bool) -> dict[int, pd.Series]:
    ids = io.vehicle_ids(stage)
    if not ids:
        raise PipelineError(
            f"No {stage} points in {config.points_dir(stage)}: run the stage that writes them."
        )

    by_vehicle, fleet_counts, pooled, daily = [], [], [], {}
    for v in ids:
        df = io.read_points(stage, [v])
        steps = step_frame(df)
        counts = _count_rows(df, steps)
        vehicle_table = _table(counts, _value_rows(steps))
        vehicle_table.insert(0, "vehicle_id", v)
        by_vehicle.append(vehicle_table)
        fleet_counts.extend(counts)
        pooled.append(steps.drop(columns="vehicle_id"))
        day = df["ts_local"].dt.tz_localize(None).dt.normalize()
        daily[v] = day.value_counts().sort_index()

    summed = (
        pd.DataFrame(fleet_counts)
        .groupby(["dimension", "check", "column"], sort=False, as_index=False)[
            ["n", "count"]
        ]
        .sum()
        .to_dict("records")
    )
    summed += _fleet_only_rows(ingest_rows)
    steps = pd.concat(pooled, ignore_index=True)
    fleet = _table(summed, _value_rows(steps))
    fleet.insert(0, "label", label)

    io.write_report_table(fleet, table)
    io.write_report_table(
        pd.concat(by_vehicle, ignore_index=True), f"{table}_by_vehicle"
    )
    if stage == "interim":
        _figure_sampling_interval(steps)
    return daily


def _fleet_only_rows(ingest_rows: bool) -> list[dict]:
    """Checks that only make sense for the whole dataset: empty raw files, parse failures."""
    inventory = pd.read_csv(config.RAW_INVENTORY_CSV)
    if config.is_dev():
        inventory = inventory[inventory["vehicle_id"].isin(config.DEV_VEHICLES)]
    rows = [
        {
            "dimension": "completeness",
            "check": "empty raw file",
            "column": "file",
            "n": len(inventory),
            "count": int((inventory["bytes"] == 0).sum()),
        }
    ]
    if ingest_rows and io.report_table_path("ingest_summary").exists():
        summary = io.read_report_table("ingest_summary")
        total = int(summary["rows"].sum())
        for col in [
            "ts",
            "lon",
            "lat",
            "speed_kmh",
            "heading_deg",
            "di1",
            "di2",
            "di3",
        ]:
            rows.append(
                {
                    "dimension": "validity",
                    "check": "did not parse at ingestion",
                    "column": col if col != "ts" else "ts_utc",
                    "n": total,
                    "count": int(summary[f"{col}_unparsed"].sum()),
                }
            )
    return rows


# --- figures --------------------------------------------------------------------------


def _figure_rows_per_vehicle() -> None:
    """F1: data rows per raw file, sorted, with the empty and small files marked."""
    figures.apply_style()
    inv = pd.read_csv(config.RAW_INVENTORY_CSV).sort_values(
        "rows", ascending=False, kind="stable"
    )
    small = (inv["bytes"] > 0) & (inv["bytes"] < config.SMALL_FILE_BYTES)
    empty = inv["bytes"] == 0
    x = np.arange(len(inv))
    fig, ax = figures.subplots(10, 3.8)
    ax.bar(
        x[~small & ~empty],
        inv.loc[~small & ~empty, "rows"] / 1000,
        width=0.75,
        color=figures.SERIES[0],
        label="files of 2 MB or more",
    )
    ax.bar(
        x[small],
        inv.loc[small, "rows"] / 1000,
        width=0.75,
        color=figures.SERIES[1],
        label="files under 2 MB",
    )
    for xi in x[empty.to_numpy()]:
        ax.text(
            xi,
            8,
            "empty",
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=7,
            color=figures.INK_SECONDARY,
        )
    ax.set_xticks(x, inv["vehicle_id"].astype(str), rotation=90, fontsize=6)
    ax.set_xlim(-0.8, len(inv) - 0.2)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("vehicle id (raw file name), sorted by rows")
    ax.set_ylabel("data rows (thousands)")
    ax.set_title(
        f"F1. Data rows per raw file ({len(inv)} files, {int(empty.sum())} empty)"
    )
    ax.legend(loc="upper right")
    figures.save_figure(fig, "f01_rows_per_vehicle")


def _figure_coverage(daily: dict[int, pd.Series]) -> None:
    """F2: points per vehicle and local day; blank where a vehicle has no point that day."""
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    figures.apply_style()
    days = pd.date_range(
        config.STUDY_START, config.STUDY_END, freq="D", inclusive="left"
    )
    ids = sorted(daily)
    matrix = pd.DataFrame({v: daily[v].reindex(days) for v in ids}).T.to_numpy(
        dtype="float64"
    )
    masked = np.ma.masked_invalid(np.where(matrix > 0, matrix, np.nan))

    cmap = LinearSegmentedColormap.from_list("blue", figures.SEQUENTIAL_BLUE)
    cmap.set_bad(figures.SURFACE)
    fig, ax = figures.subplots(10, max(2.5, 0.11 * len(ids) + 1.2))
    image = ax.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=LogNorm(vmin=max(1, np.nanmin(masked)), vmax=np.nanmax(masked)),
    )
    ax.grid(False)
    ax.set_yticks(
        np.arange(len(ids)), [str(v) for v in ids], fontsize=5 if len(ids) > 30 else 7
    )
    week_starts = [i for i, d in enumerate(days) if d.dayofweek == 0]
    ax.set_xticks(
        week_starts, [f"{days[i]:%b} {days[i].day}" for i in week_starts], fontsize=7
    )
    ax.set_xlabel("local date, 2023 (ticks on Mondays)")
    ax.set_ylabel("vehicle id")
    missing = int(np.isnan(np.where(matrix > 0, matrix, np.nan)).sum())
    ax.set_title(
        f"F2. Points per vehicle and day ({missing:,} of {matrix.size:,} vehicle-days without data, blank)"
    )
    bar = fig.colorbar(image, ax=ax, pad=0.01, fraction=0.03)
    bar.set_label("points per day (log scale)", color=figures.INK_SECONDARY, fontsize=8)
    low, high = image.norm.vmin, image.norm.vmax
    nice = [
        t
        for t in [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000]
        if low <= t <= high
    ]
    bar.set_ticks(nice, labels=[f"{t:,}" for t in nice])
    bar.ax.minorticks_off()
    bar.outline.set_visible(False)
    bar.ax.tick_params(
        labelsize=7, colors=figures.INK_MUTED, labelcolor=figures.INK_SECONDARY
    )
    figures.save_figure(fig, "f02_coverage_vehicle_day")


def _figure_sampling_interval(steps: pd.DataFrame) -> None:
    """F3: distribution of the time to the next point, by the state of the vehicle."""
    figures.apply_style()
    # Log-spaced bins with edges on half seconds: timestamps have 1 s resolution, so a bin
    # narrower than a second would show empty gaps between whole seconds.
    edges = np.unique(np.round(np.logspace(0, np.log10(7 * 86400), 70))) + 0.5
    edges = np.concatenate([[0.5], edges])
    fig, ax = figures.subplots(9, 4.2)
    positive = steps[steps["dt_s"] > 0]
    for color, state in zip(
        figures.SERIES,
        ["moving", "standing, ignition on", "ignition off"],
        strict=False,
    ):
        dt = positive.loc[positive["state"] == state, "dt_s"].to_numpy()
        if len(dt) == 0:
            continue
        counts, _ = np.histogram(np.clip(dt, edges[0], edges[-1] - 1), bins=edges)
        ax.stairs(
            100 * counts / len(dt),
            edges,
            color=color,
            linewidth=2,
            label=f"{state} ({len(dt):,} intervals)",
        )
    ax.axvline(config.GAP_S, color=figures.INK_MUTED, linewidth=1)
    ax.text(
        config.GAP_S * 1.08,
        ax.get_ylim()[1] * 0.95,
        f"gap threshold\n({config.GAP_S // 60} min)",
        fontsize=7,
        color=figures.INK_SECONDARY,
        va="top",
    )
    ax.set_xscale("log")
    ticks = [1, 10, 60, 600, 3600, 86400, 7 * 86400]
    ax.set_xticks(ticks, ["1 s", "10 s", "1 min", "10 min", "1 h", "1 day", "1 week"])
    ax.set_xlabel("time to the next point of the same vehicle (log scale)")
    ax.set_ylabel("share of the state's intervals per bin (%)")
    ax.set_title(
        "F3. Sampling interval by vehicle state (state of the point the interval starts at)"
    )
    ax.legend(loc="upper right")
    figures.save_figure(fig, "f03_sampling_interval")

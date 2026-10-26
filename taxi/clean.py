"""Pastrimi i pikave: nga points ne points_clean.

Per cdo taksi zbatohen rregullat e validitetit dhe te vlerave qe mungojne.
Cdo rregull shenohet ne cleaning_log.csv.

Rreshtat pa sekonda (taksite 1 dhe 100) nuk fshihen si duplikate, dhe hapat qe i prekin
nuk kane shpejtesi te llogaritur.

Figurat: F4 harta e pikave, F5 shpejtesia e raportuar kundrejt asaj te llogaritur
(kontrolli i njesise km/h), F6 sa rreshta mbeten pas cdo rregulli.
"""

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from taxi import PipelineError, config, figures, geo, integrate, io, missing
from taxi.ingest import INTERIM_DICTIONARY

STAGE = "clean"
CLEAN_DATASET = "processed/points_clean"
DROP_RULES = ["R01", "R02", "R03", "R04", "R08"]
MAP_LON = np.arange(19.6, 22.2001, 0.01)
MAP_LAT = np.arange(41.45, 43.6701, 0.01)
RAW_FIELDS = [
    "ts_utc",
    "lon",
    "lat",
    "speed_kmh",
    "heading_deg",
    "di1",
    "di2",
    "di3",
]

CLEAN_DICTIONARY = [
    {
        "column": "dt_s",
        "unit": "s",
        "attribute_type": "ratio",
        "values": "0 or more, NA for the last point",
        "description": "Duration of the step that starts at this point: time to the vehicle's next "
        "kept point. 0 when both share a timestamp.",
        "source": "ts_utc",
    },
    {
        "column": "dist_m",
        "unit": "m",
        "attribute_type": "ratio",
        "values": "0 or more, NA for the last point",
        "description": "Haversine distance of the step that starts at this point.",
        "source": "lat, lon",
    },
    {
        "column": "implied_speed_kmh",
        "unit": "km/h",
        "attribute_type": "ratio",
        "values": "0 or more, NA",
        "description": "dist_m / dt_s * 3.6. NA when dt_s is NA or 0, and when the step starts or "
        "ends at a timestamp without seconds.",
        "source": "dt_s, dist_m",
    },
    {
        "column": "gap_after",
        "attribute_type": "nominal",
        "values": "true / false",
        "description": "The step is longer than GAP_S (900 s): the time is unknown, not parked. "
        "Left out of durations and distances (M4).",
        "source": "dt_s",
    },
    {
        "column": "speed_imputed",
        "attribute_type": "nominal",
        "values": "true / false",
        "description": "speed_kmh was missing (or set to NA by R06) and filled by rule M2 with "
        "IMPUTE_METHOD.",
        "source": "rule M2",
    },
    {
        "column": "flag_dup_ts",
        "attribute_type": "nominal",
        "values": "true / false",
        "description": "Another kept row of the vehicle has the same ts_utc with different values "
        "(R09). Mostly the timestamps without seconds of vehicles 1 and 100.",
        "source": "rule R09",
    },
    {
        "column": "in_kosovo",
        "attribute_type": "nominal",
        "values": "true / false",
        "description": "The point lies inside the Kosovo boundary (R05). Points outside can be real "
        "trips abroad; they are kept.",
        "source": "WOF country polygon (HDX, CC BY)",
    },
    {
        "column": "municipality",
        "attribute_type": "nominal",
        "values": "38 municipality names; NA outside Kosovo",
        "description": "Municipality the point lies in, labelled by WOF id: polygon 1108809573, "
        "named 'Fushe Kosova' in WOF, is labelled 'Mitrovice e Veriut' (North Mitrovica); every "
        "other polygon keeps its WOF name.",
        "source": "WOF localadmin polygons (HDX, CC BY)",
    },
    {
        "column": "cell_id",
        "attribute_type": "nominal",
        "values": "<col>_<row>",
        "description": "Square grid cell of GRID_SIZE_M (500 m) in EPSG:32634: col = floor(x / 500), "
        "row = floor(y / 500).",
        "source": "lat, lon",
    },
    {
        "column": "date_local",
        "attribute_type": "interval",
        "values": "2023-06-01 to 2023-08-31",
        "description": "Local date of the point.",
        "source": "ts_local",
    },
    {
        "column": "hour",
        "unit": "h",
        "attribute_type": "interval, circular",
        "values": "0 to 23",
        "description": "Local hour of the point.",
        "source": "ts_local",
    },
    {
        "column": "is_weekend",
        "attribute_type": "nominal",
        "values": "true / false",
        "description": "The local date is a Saturday or Sunday.",
        "source": "calendar_2023.csv",
    },
    {
        "column": "is_holiday",
        "attribute_type": "nominal",
        "values": "true / false",
        "description": "The local date is an official holiday (Kurban Bajram, June 28, is the only "
        "one in the window).",
        "source": "calendar_2023.csv (Ministry of Internal Affairs)",
    },
]


def run(dev: bool) -> None:
    ids = io.vehicle_ids("interim")
    if not ids:
        raise PipelineError("No interim points: run the ingest stage first.")
    for stale in sorted(set(io.vehicle_ids("clean")) - set(ids)):
        io.points_path("clean", stale).unlink()

    rng = np.random.default_rng(config.SEED)
    pairs, summaries = [], []
    joined = 0
    inside = np.zeros((len(MAP_LON) - 1, len(MAP_LAT) - 1))
    outside = np.zeros_like(inside)
    for v in ids:
        df, raw_missing, raw_rows = clean_vehicle(v)
        io.write_points(df, "clean")
        summaries.append(missing.vehicle_summary(df, raw_missing, raw_rows, rng))
        pairs.append(speed_pairs(df))
        joined = joined + integrate.join_counts(df)
        inside += map_counts(df[df["in_kosovo"]])
        outside += map_counts(df[~df["in_kosovo"]])

    rows = [{**r, "owner": "ingest"} for r in INTERIM_DICTIONARY] + CLEAN_DICTIONARY
    io.write_dictionary(CLEAN_DATASET, rows, df, owner=STAGE)

    missing.write_tables(summaries)
    integrate.write_summary(joined, len(ids))
    figure_points_map(inside, outside)
    figure_speed_check(pd.concat(pairs, ignore_index=True))
    figure_funnel()


def clean_vehicle(vehicle_id: int) -> tuple[pd.DataFrame, pd.Series, int]:
    df = io.read_points("interim", [vehicle_id]).sort_values("source_row")
    raw_rows = len(df)
    raw_missing = df[list(missing.COLUMN_RULES)].isna().sum()
    dropped = 0

    def log(rule_id: str, action: str, n: int, note: str) -> None:
        io.log_step(STAGE, rule_id, action, vehicle_id, len(df), int(n), note)

    def drop(rule_id: str, mask: pd.Series, note: str) -> None:
        nonlocal df, dropped
        n = int(mask.sum())
        over = n > config.MAX_DROP_SHARE * raw_rows
        reviewed = rule_id == "R08" and vehicle_id in config.R08_REVIEWED_VEHICLES
        if over and reviewed:
            note += "; above MAX_DROP_SHARE, rows reviewed (T2b)"
        log(rule_id, "drop", n, note)
        if over and not reviewed:
            raise PipelineError(
                f"Vehicle {vehicle_id}: {rule_id} drops {n:,} of {raw_rows:,} rows "
                f"({n / raw_rows:.2%}), above MAX_DROP_SHARE ({config.MAX_DROP_SHARE:.0%}). "
                "Look at the rows before changing anything."
            )
        df = df[~mask]
        dropped += n

    drop("R01", df["ts_utc"].isna(), "timestamp missing (did not parse)")
    ts_device = df["ts_utc"].dt.tz_convert(config.DEVICE_TZ).dt.tz_localize(None)
    drop(
        "R02",
        (ts_device < config.STUDY_START) | (ts_device >= config.STUDY_END),
        "outside the study window on the device clock",
    )
    drop("R03", df["lat"].isna() | df["lon"].isna(), "lat or lon missing")
    drop(
        "R04",
        ~df["lat"].between(-90, 90)
        | ~df["lon"].between(-180, 180)
        | ((df["lat"] == 0) & (df["lon"] == 0)),
        "lat or lon out of range, or lat = lon = 0",
    )

    has_seconds = df["ts_has_seconds"].astype(bool)
    same_as_earlier = df.duplicated(RAW_FIELDS + ["ts_has_seconds"])
    drop(
        "R08",
        same_as_earlier & has_seconds,
        "exact duplicate of a lower source_row, rows with seconds only; "
        f"{int((same_as_earlier & ~has_seconds).sum()):,} identical-looking minute-only rows kept",
    )

    df["flag_dup_ts"] = df["ts_utc"].duplicated(keep=False)
    log(
        "R09",
        "flag",
        df["flag_dup_ts"].sum(),
        f"same ts_utc as another kept row; "
        f"{int((df['flag_dup_ts'] & ~df['ts_has_seconds']).sum()):,} of them without seconds",
    )

    df = df.sort_values(["ts_utc", "source_row"], kind="stable").reset_index(drop=True)
    add_steps(df)
    log(
        "M4",
        "flag",
        df["gap_after"].sum(),
        f"step longer than GAP_S ({config.GAP_S} s): gap_after, no points created",
    )

    bad_speed = (df["speed_kmh"] < 0) | (df["speed_kmh"] > config.SPEED_MAX_KMH)
    df.loc[bad_speed, "speed_kmh"] = np.nan
    log(
        "R06",
        "set_na",
        bad_speed.sum(),
        f"speed_kmh < 0 or > SPEED_MAX_KMH ({config.SPEED_MAX_KMH})",
    )
    bad_heading = (df["heading_deg"] < 0) | (df["heading_deg"] > 360)
    df.loc[bad_heading, "heading_deg"] = np.nan
    log("R07", "set_na", bad_heading.sum(), "heading_deg outside 0-360")

    missing_speed = df["speed_kmh"].isna()
    estimate = missing.speed_estimate(df, config.IMPUTE_METHOD)
    df["speed_imputed"] = (
        missing_speed
        & missing.impute_eligible(df, config.IMPUTE_MAX_DT_S)
        & estimate.notna()
    )
    df["speed_kmh"] = df["speed_kmh"].mask(df["speed_imputed"], estimate)
    log(
        "M2",
        "impute",
        df["speed_imputed"].sum(),
        f"{int(missing_speed.sum()):,} speeds missing, "
        f"{int(df['speed_imputed'].sum()):,} filled with {config.IMPUTE_METHOD}",
    )

    df["in_kosovo"] = integrate.in_kosovo(df["lon"], df["lat"])
    log(
        "R05",
        "flag",
        (~df["in_kosovo"]).sum(),
        "outside the Kosovo boundary: kept, in_kosovo = false",
    )
    integrate.add_point_context(df)
    if len(df) + dropped != raw_rows:
        raise PipelineError(
            f"Vehicle {vehicle_id}: {len(df):,} kept + {dropped:,} dropped != {raw_rows:,} rows in"
        )
    columns = [r["column"] for r in INTERIM_DICTIONARY + CLEAN_DICTIONARY]
    return df[columns], raw_missing, raw_rows


def add_steps(df: pd.DataFrame) -> None:
    nxt = df.shift(-1)
    dt = (nxt["ts_utc"] - df["ts_utc"]).dt.total_seconds()
    df["dt_s"] = dt.astype("float32")
    df["dist_m"] = geo.haversine_m(df["lat"], df["lon"], nxt["lat"], nxt["lon"]).astype(
        "float32"
    )
    no_seconds = ~df["ts_has_seconds"] | ~nxt["ts_has_seconds"].fillna(True)
    implied = df["dist_m"] / dt.where(dt > 0) * 3.6
    df["implied_speed_kmh"] = implied.mask(no_seconds).astype("float32")
    df["gap_after"] = dt > config.GAP_S


def speed_pairs(df: pd.DataFrame) -> pd.DataFrame:
    lo, hi = config.SPEED_CHECK_DT_S
    usable = (
        df["dt_s"].between(lo, hi)
        & (df["dist_m"] >= config.SPEED_CHECK_MIN_DIST_M)
        & (df["speed_kmh"] >= config.V_MOVE_KMH)
        & ~df["speed_imputed"]
        & df["implied_speed_kmh"].notna()
    )
    return df.loc[usable, ["speed_kmh", "implied_speed_kmh"]]


def map_counts(df: pd.DataFrame) -> np.ndarray:
    return np.histogram2d(df["lon"], df["lat"], bins=[MAP_LON, MAP_LAT])[0]


def figure_points_map(inside: np.ndarray, outside: np.ndarray) -> None:
    log = io.read_log().query("stage == @STAGE")
    n_r04 = int(log.loc[log["rule_id"] == "R04", "rows_affected"].sum())
    n_outside = int(log.loc[log["rule_id"] == "R05", "rows_affected"].sum())
    n_off_map = n_outside - int(outside.sum())

    fig, ax = figures.subplots(7, 6.2)
    blues = mcolors.ListedColormap(figures.SEQUENTIAL_BLUE)
    mesh = ax.pcolormesh(
        MAP_LON,
        MAP_LAT,
        np.ma.masked_equal(inside, 0).T,
        cmap=blues,
        norm=mcolors.LogNorm(),
    )
    ax.pcolormesh(
        MAP_LON,
        MAP_LAT,
        np.ma.masked_equal(outside, 0).T,
        cmap=mcolors.ListedColormap([figures.SERIES[1]]),
    )
    boundary = integrate.kosovo_boundary()
    for part in getattr(boundary, "geoms", [boundary]):
        x, y = part.exterior.xy
        ax.plot(x, y, color=figures.INK_SECONDARY, linewidth=0.8)
    ax.set_aspect(1 / np.cos(np.radians(42.6)))
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.set_title("F4. Points per cell of 0.01 degrees, after the drop rules")
    fig.colorbar(mesh, ax=ax, shrink=0.7, label="Points per cell (log scale)")
    ax.text(
        0.01,
        0.01,
        f"Orange: outside Kosovo, kept with in_kosovo = false ({n_outside:,} points, "
        f"{n_off_map:,} beyond this map)\n"
        f"R04: {n_r04:,} points dropped for impossible coordinates, not shown",
        transform=ax.transAxes,
        fontsize=8,
        color=figures.INK_SECONDARY,
        bbox={"facecolor": figures.SURFACE, "edgecolor": figures.AXIS, "pad": 4},
    )
    figures.save_figure(fig, "f04_raw_points_map")


def figure_speed_check(pairs: pd.DataFrame) -> None:
    ratio = (pairs["implied_speed_kmh"] / pairs["speed_kmh"]).median()
    print(
        f"Speed unit check: {len(pairs):,} steps, median implied / reported speed {ratio:.3f}"
    )
    top = 150
    fig, ax = figures.subplots(6.5, 5.5)
    hb = ax.hexbin(
        pairs["speed_kmh"],
        pairs["implied_speed_kmh"],
        gridsize=60,
        extent=(0, top, 0, top),
        mincnt=1,
        bins="log",
        cmap=mcolors.ListedColormap(figures.SEQUENTIAL_BLUE),
    )
    ax.plot(
        [0, top], [0, top], color=figures.INK, linewidth=1, label="implied = reported"
    )
    ax.plot(
        [0, top / 3.6],
        [0, top],
        color=figures.SERIES[1],
        linewidth=1,
        linestyle="--",
        label="implied = 3.6 x reported (speed in m/s)",
    )
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("Reported speed (km/h)")
    ax.set_ylabel("Implied speed, distance / time (km/h)")
    ax.set_title(
        f"F5. Reported vs implied speed, median ratio {ratio:.2f} ({len(pairs):,} steps)"
    )
    ax.legend(loc="upper left")
    fig.colorbar(hb, ax=ax, label="Steps per hexagon (log scale)")
    figures.save_figure(fig, "f05_reported_vs_implied_speed")


def figure_funnel() -> None:
    log = io.read_log().query("stage == @STAGE")
    dropped = log.groupby("rule_id")["rows_affected"].sum()
    remaining = [int(log.loc[log["rule_id"] == "R01", "rows_in"].sum())]
    for rule in DROP_RULES:
        remaining.append(remaining[-1] - int(dropped.get(rule, 0)))
    labels = ["Raw rows"] + [f"after {rule}" for rule in DROP_RULES]

    fig, ax = figures.subplots(7, 3.5)
    y = np.arange(len(labels))[::-1]
    ax.barh(y, remaining, color=figures.SERIES[0], height=0.6)
    for yi, rule, n in zip(y, [None, *DROP_RULES], remaining):
        text = f"{n:,}" if rule is None else f"{n:,} (-{int(dropped.get(rule, 0)):,})"
        ax.text(n, yi, f" {text}", va="center", fontsize=8, color=figures.INK_SECONDARY)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, remaining[0] * 1.35)
    ax.xaxis.set_major_formatter("{x:,.0f}")
    ax.set_xlabel("Rows")
    ax.grid(axis="y", visible=False)
    ax.set_title("F6. Rows remaining after each drop rule")
    figures.save_figure(fig, "f06_cleaning_funnel")

"""Pastrimi i pikave: nga points ne points_clean.

Per cdo taksi zbatohen rregullat e validitetit dhe te vlerave qe mungojne.
Cdo rregull shenohet ne cleaning_log.csv.

Rreshtat pa sekonda (taksite 1 dhe 100) nuk fshihen si duplikate, dhe hapat qe i prekin
nuk kane shpejtesi te llogaritur.
"""

import numpy as np
import pandas as pd

from taxi import PipelineError, config, geo, integrate, io, missing
from taxi.ingest import INTERIM_DICTIONARY

STAGE = "clean"
CLEAN_DATASET = "processed/points_clean"
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
        "column": "flag_jump",
        "attribute_type": "nominal",
        "values": "true / false",
        "description": "The step's implied speed is above JUMP_KMH (250 km/h) over more than "
        "JUMP_MIN_DIST_M (100 m): a possible GPS jump, kept for Phase II (R10).",
        "source": "rule R10",
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

    for v in ids:
        df = clean_vehicle(v)
        io.write_points(df, "clean")

    rows = [{**r, "owner": "ingest"} for r in INTERIM_DICTIONARY] + CLEAN_DICTIONARY
    io.write_dictionary(CLEAN_DATASET, rows, df, owner=STAGE)


def clean_vehicle(vehicle_id: int) -> pd.DataFrame:
    df = io.read_points("interim", [vehicle_id]).sort_values("source_row")
    raw_rows = len(df)
    dropped = 0

    def log(rule_id: str, action: str, n: int, note: str) -> None:
        io.log_step(STAGE, rule_id, action, vehicle_id, len(df), int(n), note)

    def drop(rule_id: str, mask: pd.Series, note: str) -> None:
        nonlocal df, dropped
        n = int(mask.sum())
        log(rule_id, "drop", n, note)
        if n > config.MAX_DROP_SHARE * raw_rows:
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
    df["flag_jump"] = (df["implied_speed_kmh"] > config.JUMP_KMH) & (
        df["dist_m"] > config.JUMP_MIN_DIST_M
    )
    log(
        "R10",
        "flag",
        df["flag_jump"].sum(),
        f"implied speed > JUMP_KMH ({config.JUMP_KMH}) over > JUMP_MIN_DIST_M "
        f"({config.JUMP_MIN_DIST_M} m): kept for Phase II",
    )

    integrate.add_point_context(df)
    if len(df) + dropped != raw_rows:
        raise PipelineError(
            f"Vehicle {vehicle_id}: {len(df):,} kept + {dropped:,} dropped != {raw_rows:,} rows in"
        )
    columns = [r["column"] for r in INTERIM_DICTIONARY + CLEAN_DICTIONARY]
    return df[columns]


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

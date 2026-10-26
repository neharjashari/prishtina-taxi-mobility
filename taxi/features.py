"""Point-level feature creation: processed/points_clean to processed/points (the final points)."""

import numpy as np
import pandas as pd

from taxi import PipelineError, config, geo, io, transform

STAGE = "features"
CLEAN_DATASET = "processed/points_clean"
FINAL_DATASET = "processed/points"

QUESTIONS = {
    "Q1": "when and where the fleet is most active",
    "Q2": "how speeds change across the day and across the city",
    "Q3": "how much time taxis spend moving, standing, or switched off",
    "Q4": "what a typical trip looks like",
    "Q5": "how weekends differ from working days",
}

FEATURE_COLUMNS = [
    "dow",
    "minute_of_week",
    "tod_bucket",
    "is_night",
    "is_moving",
    "accel_ms2",
    "heading_change_deg",
    "compass8",
    "speed_bin",
    "x_m",
    "y_m",
    "dist_center_m",
    "hour_sin",
    "hour_cos",
    "heading_sin",
    "heading_cos",
]

FEATURE_DICTIONARY = [
    {
        "column": "dow",
        "attribute_type": "ordinal, circular",
        "values": "0 = Monday ... 6 = Sunday",
        "description": "Local day of the week.",
        "source": "ts_local",
    },
    {
        "column": "minute_of_week",
        "unit": "min",
        "attribute_type": "interval, circular",
        "values": "0 to 10079",
        "description": "Minutes since Monday 00:00 local time: dow * 1440 + hour * 60 + minute.",
        "source": "ts_local",
    },
    {
        "column": "tod_bucket",
        "attribute_type": "ordinal, circular",
        "values": "night [0, 6), morning [6, 10), midday [10, 15), afternoon [15, 19), "
        "evening [19, 24) local hours",
        "description": "Time of day bucket of the local hour (TOD_BUCKETS, left-closed).",
        "source": "hour",
    },
    {
        "column": "is_night",
        "attribute_type": "nominal",
        "values": "true / false",
        "description": "The local hour is in NIGHT_HOURS, [0, 6).",
        "source": "hour",
    },
    {
        "column": "is_moving",
        "attribute_type": "nominal",
        "values": "true / false",
        "description": "speed_kmh >= V_MOVE_KMH (5 km/h). False when speed_kmh is NA. The step "
        "that starts at the point counts as moving time when this is true.",
        "source": "speed_kmh",
    },
    {
        "column": "accel_ms2",
        "unit": "m/s²",
        "attribute_type": "ratio",
        "values": "negative when slowing down, NA",
        "description": "Acceleration over the step that starts at the point: (next point's "
        "speed - this point's speed) / 3.6 / dt_s. NA for the last point, when gap_after is "
        "true, when dt_s is 0, and when the step starts or ends at a timestamp without "
        "seconds (their dt_s is not exact).",
        "source": "speed_kmh, dt_s, gap_after, ts_has_seconds",
    },
    {
        "column": "heading_change_deg",
        "unit": "degrees",
        "attribute_type": "ratio",
        "values": "0 to 180, NA",
        "description": "Circular difference to the next point's heading, "
        "min(|a - b| mod 360, 360 - |a - b| mod 360). NA for the last point and across a gap. "
        "A standing vehicle keeps its last heading or reports noise.",
        "source": "heading_deg, gap_after",
    },
    {
        "column": "compass8",
        "attribute_type": "nominal, circular",
        "values": "N, NE, E, SE, S, SW, W, NW",
        "description": "Heading in 45-degree sectors centred on north: N covers [337.5, 22.5), "
        "NE [22.5, 67.5), and so on. NA when heading_deg is NA.",
        "source": "heading_deg",
    },
    {
        "column": "speed_bin",
        "attribute_type": "ordinal",
        "values": "stopped [0, 5), slow [5, 20), urban [20, 50), fast [50, 80), highway "
        "[80, inf) km/h",
        "description": "Domain-based speed bin (SPEED_BINS_KMH, left-closed). 'stopped' ends at "
        "V_MOVE_KMH, so it is exactly the points with is_moving false. NA when speed_kmh is NA.",
        "source": "speed_kmh",
    },
    {
        "column": "x_m",
        "unit": "m (EPSG:32634)",
        "attribute_type": "interval",
        "values": "easting, WGS 84 / UTM zone 34N",
        "description": "Projected x coordinate, so distances can be measured in metres.",
        "source": "lon, lat",
    },
    {
        "column": "y_m",
        "unit": "m (EPSG:32634)",
        "attribute_type": "interval",
        "values": "northing, WGS 84 / UTM zone 34N",
        "description": "Projected y coordinate.",
        "source": "lon, lat",
    },
    {
        "column": "dist_center_m",
        "unit": "m",
        "attribute_type": "ratio",
        "values": "0 or more",
        "description": f"Haversine distance to CITY_CENTER {config.CITY_CENTER}: "
        f"{config.CITY_CENTER_SOURCE}.",
        "source": "lat, lon, CITY_CENTER",
    },
    {
        "column": "hour_sin",
        "attribute_type": "interval",
        "values": "-1 to 1",
        "description": "sin(2 pi (hour + minute / 60) / 24): the clock time as a point on a "
        "circle, with hour_cos, so that 23:59 lies next to 00:00.",
        "source": "ts_local",
    },
    {
        "column": "hour_cos",
        "attribute_type": "interval",
        "values": "-1 to 1",
        "description": "cos(2 pi (hour + minute / 60) / 24).",
        "source": "ts_local",
    },
    {
        "column": "heading_sin",
        "attribute_type": "interval",
        "values": "-1 to 1, NA",
        "description": "sin(heading in radians): the east component of the direction of travel, "
        "with heading_cos (the north component), so that 359 degrees lies next to 1 degree.",
        "source": "heading_deg",
    },
    {
        "column": "heading_cos",
        "attribute_type": "interval",
        "values": "-1 to 1, NA",
        "description": "cos(heading in radians).",
        "source": "heading_deg",
    },
]

FEATURE_CATALOG = [
    {
        "feature": "dow",
        "formula": "ts_local day of week, 0 = Monday",
        "attribute_type": "ordinal, circular",
        "purpose": "Q1 (day of week), Q5 (weekends vs working days)",
    },
    {
        "feature": "minute_of_week",
        "formula": "dow * 1440 + hour * 60 + minute",
        "unit": "min",
        "attribute_type": "interval, circular",
        "purpose": "Q1, Q5: weekly activity profile (hour-of-week heatmaps)",
    },
    {
        "feature": "tod_bucket",
        "formula": "hour in [0, 6) night, [6, 10) morning, [10, 15) midday, [15, 19) "
        "afternoon, [19, 24) evening",
        "attribute_type": "ordinal, circular",
        "purpose": "Q1, Q2: activity and speed by part of the day",
    },
    {
        "feature": "is_night",
        "formula": "hour in [0, 6)",
        "attribute_type": "nominal (binary, symmetric)",
        "purpose": "Q1: night-time activity",
    },
    {
        "feature": "is_moving",
        "formula": "speed_kmh >= 5",
        "attribute_type": "nominal (binary, symmetric)",
        "purpose": "Q3: moving vs standing time (time-weighted over steps)",
    },
    {
        "feature": "accel_ms2",
        "formula": "(speed_next - speed) / 3.6 / dt_s, NA across gaps, dt_s = 0, or "
        "timestamps without seconds",
        "unit": "m/s²",
        "attribute_type": "ratio",
        "purpose": "Q2: stop-and-go driving across the day and the city; Phase II: speed spikes",
    },
    {
        "feature": "heading_change_deg",
        "formula": "circular |heading_next - heading|, 0-180, NA across gaps",
        "unit": "degrees",
        "attribute_type": "ratio",
        "purpose": "No guiding question directly (turning, GPS zig-zag for Phase II "
        "outliers): candidate to drop",
    },
    {
        "feature": "compass8",
        "formula": "45-degree sectors of heading centred on N",
        "attribute_type": "nominal, circular",
        "purpose": "Q1, Q4: direction of flows, for example towards or away from the centre by hour",
    },
    {
        "feature": "speed_bin",
        "formula": "speed_kmh in [0, 5) stopped, [5, 20) slow, [20, 50) urban, [50, 80) fast, "
        "[80, inf) highway",
        "attribute_type": "ordinal",
        "purpose": "Q2, Q3: time spent in each speed regime",
    },
    {
        "feature": "x_m",
        "formula": "EPSG:32634 easting of (lon, lat)",
        "unit": "m",
        "attribute_type": "interval",
        "purpose": "Q1 (where): distances and grid cells in metres",
    },
    {
        "feature": "y_m",
        "formula": "EPSG:32634 northing of (lon, lat)",
        "unit": "m",
        "attribute_type": "interval",
        "purpose": "Q1 (where): distances and grid cells in metres",
    },
    {
        "feature": "dist_center_m",
        "formula": "haversine((lat, lon), CITY_CENTER)",
        "unit": "m",
        "attribute_type": "ratio",
        "purpose": "Q1 (centre vs outskirts), Q2 (speed by distance from the centre)",
    },
    {
        "feature": "hour_sin",
        "formula": "sin(2 pi (hour + minute / 60) / 24)",
        "attribute_type": "interval",
        "purpose": "Q1: clock time as a circle for distance-based methods",
    },
    {
        "feature": "hour_cos",
        "formula": "cos(2 pi (hour + minute / 60) / 24)",
        "attribute_type": "interval",
        "purpose": "Q1: clock time as a circle for distance-based methods",
    },
    {
        "feature": "heading_sin",
        "formula": "sin(radians(heading_deg))",
        "attribute_type": "interval",
        "purpose": "Q4: direction of travel as a circle (359 next to 1 degree)",
    },
    {
        "feature": "heading_cos",
        "formula": "cos(radians(heading_deg))",
        "attribute_type": "interval",
        "purpose": "Q4: direction of travel as a circle (359 next to 1 degree)",
    },
]


# --- Feature catalogue (T9) -----------------------------------------------------------

CATALOG_NAME = "feature_catalog"
CATALOG_COLUMNS = [
    "feature",
    "dataset",
    "formula",
    "unit",
    "attribute_type",
    "purpose",
    "owner",
]
CATALOG_ORDER = [
    FINAL_DATASET,
    "processed/trips",
    "processed/model/trips_model",
    "processed/model/vehicle_day_model",
]


def write_catalog(dataset: str, rows: list[dict], owner: str) -> None:
    """Replace one dataset's rows in feature_catalog.csv (T9), keeping the other datasets."""
    new = pd.DataFrame(rows)
    for col in CATALOG_COLUMNS:
        if col not in new.columns:
            new[col] = ""
    new["dataset"] = dataset
    new = new.fillna("")
    new.loc[new["owner"] == "", "owner"] = owner
    path = io.report_table_path(CATALOG_NAME)
    if path.exists():
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        new = pd.concat([old[old["dataset"] != dataset], new], ignore_index=True)
    rank = {d: i for i, d in enumerate(CATALOG_ORDER)}
    new = new.sort_values(
        "dataset", key=lambda s: s.map(rank).fillna(len(rank)), kind="stable"
    )
    io.write_report_table(new[CATALOG_COLUMNS], CATALOG_NAME)


def dataset_dictionary(dataset: str) -> list[dict]:
    """The data dictionary rows of one dataset, as written by its stage, for reuse downstream."""
    rows = io.read_dictionary()
    rows = rows[rows["dataset"] == dataset]
    if rows.empty:
        raise PipelineError(
            f"No data dictionary rows for {dataset}; its stage writes them."
        )
    return rows.drop(columns=["dataset", "dtype"]).to_dict("records")


# --- Features -------------------------------------------------------------------------


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """The final-point feature columns for one vehicle's clean points, in step order."""
    if config.CITY_CENTER is None:
        raise PipelineError("CITY_CENTER is not set in taxi/config.py.")
    df = df.sort_values(["ts_utc", "source_row"], kind="stable").reset_index(drop=True)
    local = df["ts_local"]
    minute = local.dt.minute.astype("int16")
    hour = df["hour"].astype("int16")

    df["dow"] = local.dt.dayofweek.astype("int8")
    df["minute_of_week"] = (
        df["dow"].astype("int16") * 1440 + hour * 60 + minute
    ).astype("int16")
    df["tod_bucket"] = transform.tod_bucket(hour)
    night_from, night_to = config.NIGHT_HOURS
    df["is_night"] = (hour >= night_from) & (hour < night_to)

    speed = df["speed_kmh"].astype("float64")
    df["is_moving"] = (speed >= config.V_MOVE_KMH).astype(bool)
    dt = df["dt_s"].astype("float64")
    has_seconds = df["ts_has_seconds"].fillna(False).astype(bool)
    exact_step = has_seconds & has_seconds.shift(-1, fill_value=False)
    accel = (speed.shift(-1) - speed) / 3.6 / dt.where(dt > 0)
    df["accel_ms2"] = accel.where(~df["gap_after"] & exact_step).astype("float32")

    heading = df["heading_deg"].astype("float64")
    diff = (heading.shift(-1) - heading).abs() % 360
    change = np.minimum(diff, 360 - diff)
    df["heading_change_deg"] = change.where(~df["gap_after"]).astype("float32")
    df["compass8"] = transform.compass8(heading)
    df["speed_bin"] = transform.speed_bin(speed)

    x_m, y_m = geo.to_metric(df["lon"], df["lat"])
    df["x_m"] = x_m
    df["y_m"] = y_m
    center_lat, center_lon = config.CITY_CENTER
    df["dist_center_m"] = geo.haversine_m(
        df["lat"], df["lon"], center_lat, center_lon
    ).astype("float32")

    clock = 2 * np.pi * (hour + minute / 60) / 24
    df["hour_sin"] = np.sin(clock).astype("float32")
    df["hour_cos"] = np.cos(clock).astype("float32")
    rad = np.radians(heading)
    df["heading_sin"] = np.sin(rad).astype("float32")
    df["heading_cos"] = np.cos(rad).astype("float32")
    return df


def run(dev: bool) -> None:
    """Add the point features to every vehicle's clean points and write the final points."""
    ids = io.vehicle_ids("clean")
    if not ids:
        raise PipelineError("No clean points: run the clean stage first.")
    for stale in sorted(set(io.vehicle_ids("final")) - set(ids)):
        io.points_path("final", stale).unlink()

    clean_rows = dataset_dictionary(CLEAN_DATASET)
    clean_columns = [r["column"] for r in clean_rows]
    rows_total = 0
    for v in ids:
        df = io.read_points("clean", [v])
        rows_in = len(df)
        out = add_features(df)[clean_columns + FEATURE_COLUMNS]
        if len(out) != rows_in:
            raise PipelineError(
                f"Vehicle {v}: features changed the row count ({rows_in:,} -> {len(out):,})."
            )
        io.write_points(out, "final")
        rows_total += rows_in

    io.write_dictionary(
        FINAL_DATASET, clean_rows + FEATURE_DICTIONARY, out, owner=STAGE
    )
    write_catalog(FINAL_DATASET, FEATURE_CATALOG, owner=STAGE)
    print(
        f"{len(ids)} vehicles, {rows_total:,} points: {len(FEATURE_COLUMNS)} feature columns "
        f"added; CITY_CENTER = {config.CITY_CENTER}"
    )

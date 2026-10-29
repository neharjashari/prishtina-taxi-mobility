"""Discretization, binarization, and variable transformation."""

import re

import numpy as np
import pandas as pd

from taxi import PipelineError, config, figures, io

BIN_METHODS = ("equal_width", "equal_frequency", "domain")


# --- Discretization -------------------------------------------------------------------


def _left_closed(values: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(
        pd.Series(values).astype("float64"),
        bins=edges,
        labels=labels,
        right=False,
        ordered=True,
    )


def speed_bin(speed_kmh: pd.Series) -> pd.Series:
    """Domain-based speed bins (SPEED_BINS_KMH): stopped, slow, urban, fast, highway."""
    return _left_closed(speed_kmh, config.SPEED_BINS_KMH, config.SPEED_BIN_LABELS)


def tod_bucket(hour: pd.Series) -> pd.Series:
    """Time-of-day bucket of a local hour (TOD_BUCKETS): night, morning, midday, afternoon, evening."""
    return _left_closed(hour, config.TOD_BUCKETS, config.TOD_LABELS)


def compass8(heading_deg: pd.Series) -> pd.Series:
    """Heading in eight 45-degree sectors centred on north: N covers [337.5, 22.5), NE [22.5, 67.5), ..."""
    h = pd.Series(heading_deg).astype("float64")
    sector = np.floor(((h % 360) + 22.5) % 360 / 45)
    codes = sector.fillna(-1).astype("int8").to_numpy()
    return pd.Series(
        pd.Categorical.from_codes(codes, categories=config.COMPASS_LABELS),
        index=h.index,
    )


def distance_bin(distance_m: pd.Series) -> pd.Series:
    """Domain-based trip distance bins (DISTANCE_BINS_M): under 2 km, 2-5, 5-10, 10-20, 20 km or more."""
    return _left_closed(distance_m, config.DISTANCE_BINS_M, config.DISTANCE_BIN_LABELS)


def bin_edges(
    values: pd.Series, method: str, n_bins: int = config.N_COMPARE_BINS
) -> np.ndarray:
    """Edges of n_bins bins by one method, for left-closed binning of `values`."""
    v = pd.Series(values).dropna().to_numpy(dtype="float64")
    if len(v) == 0:
        raise ValueError("bin_edges needs at least one value")
    if method == "equal_width":
        edges = np.linspace(v.min(), v.max(), n_bins + 1)
    elif method == "equal_frequency":
        edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
    elif method == "domain":
        edges = np.array(config.DISTANCE_BINS_M, dtype="float64")
    else:
        raise ValueError(f"method must be one of {BIN_METHODS}, got {method!r}")
    if edges[-1] <= v.max():
        edges[-1] = np.nextafter(v.max(), np.inf)
    if len(edges) < 2:
        edges = np.array([v.min(), np.nextafter(v.min(), np.inf)])
    return edges


def apply_bins(values: pd.Series, edges: np.ndarray) -> pd.Series:
    """Left-closed bins [a, b) with the given edges, as an ordered categorical of intervals."""
    return pd.cut(pd.Series(values).astype("float64"), bins=edges, right=False)


# --- Binarization ---------------------------------------------------------------------


def to_binary(values: pd.Series) -> pd.Series:
    """0/1 as Int8 from a boolean column. NA stays NA (never read as 0)."""
    return pd.Series(values).astype("boolean").astype("Int8")


def one_hot(values: pd.Series, prefix: str) -> pd.DataFrame:
    """One 0/1 (int8) column per category, named <prefix>_<category>, in category order."""
    s = pd.Series(values)
    categories = (
        list(s.cat.categories)
        if isinstance(s.dtype, pd.CategoricalDtype)
        else sorted(s.dropna().unique())
    )
    return pd.DataFrame(
        {
            f"{prefix}_{str(c).replace(' ', '_')}": (s == c).astype("int8")
            for c in categories
        },
        index=s.index,
    )


BINARY_ATTRIBUTES: dict[str, tuple[bool, str]] = {
    "is_moving": (False, "moving and standing are both informative states"),
    "is_weekend": (False, "weekend and working day are both informative"),
    "is_night": (False, "night and day are both informative"),
    "is_holiday": (
        True,
        (
            "only the one holiday is informative; two ordinary days are not alike because "
            "neither is a holiday"
        ),
    ),
    "di1": (False, "ignition on and off are both informative states"),
    "di2": (True, "taximeter on marks a paid ride; off is the default state"),
    "di3": (
        True,
        "a passenger on board is the event of interest; stuck in 25 vehicles",
    ),
    "in_kosovo": (True, "the rare value (outside Kosovo) is the informative one"),
    "gap_after": (True, "quality flag: only a gap carries information"),
    "speed_imputed": (True, "quality flag"),
    "flag_dup_ts": (True, "quality flag"),
}


# --- Stage: transform -----------------------------------------------------------------

STAGE = "transform"
TRIPS_MODEL = "model/trips_model"
VEHICLE_DAY_MODEL = "model/vehicle_day_model"
TRIPS_MODEL_DATASET = "processed/model/trips_model"
VEHICLE_DAY_MODEL_DATASET = "processed/model/vehicle_day_model"

LOG_CANDIDATES = {
    "trips_model": [
        "distance_m",
        "duration_s",
        "straight_m",
        "n_points",
        "start_dist_center_m",
        "end_dist_center_m",
    ],
    "vehicle_day_model": [
        "km",
        "n_trips",
        "trip_median_distance_m",
        "trip_median_duration_s",
    ],
}
F9_VARIABLES = [
    ("trips_model", "distance_m", "trip distance (m)"),
    ("trips_model", "duration_s", "trip duration (s)"),
    ("vehicle_day_model", "km", "vehicle-day distance (km)"),
]
ID_COLUMNS = {
    "trips_model": ["trip_id", "vehicle_id"],
    "vehicle_day_model": ["vehicle_id", "date_local"],
}
POINT_BIN_COLUMNS = ["speed_bin", "tod_bucket", "compass8"]
POINT_BIN_LABELS = {
    "speed_bin": config.SPEED_BIN_LABELS,
    "tod_bucket": config.TOD_LABELS,
    "compass8": config.COMPASS_LABELS,
}
PC_COLUMN = re.compile(r"^pc\d+$")

MODEL_SPEC: dict[str, dict[str, tuple[str, str, str, str]]] = {
    "trips_model": {
        "trip_id": ("", "nominal", "Trip id, as in processed/trips.", "identifier"),
        "vehicle_id": ("", "nominal", "Taxi id.", "identifier"),
        "distance_m": (
            "m",
            "ratio",
            "Trip distance (processed/trips).",
            "Q4: trip distance",
        ),
        "duration_s": (
            "s",
            "ratio",
            "Trip duration (processed/trips).",
            "Q4: trip duration",
        ),
        "straight_m": (
            "m",
            "ratio",
            "Distance from origin to destination.",
            "Q4: reach of a trip",
        ),
        "straightness": (
            "",
            "ratio",
            "straight_m / distance_m, NA when distance_m = 0.",
            "Q4: direct vs roundabout trips",
        ),
        "avg_speed_kmh": (
            "km/h",
            "ratio",
            "distance_m / duration_s * 3.6.",
            "Q2: trip speed",
        ),
        "max_speed_kmh": (
            "km/h",
            "ratio",
            "Highest reported speed in the trip.",
            "Q2; Phase II: speed outliers",
        ),
        "n_points": (
            "",
            "ratio",
            "Points in the trip.",
            "none: point counts depend on the reporting rate, not on time",
        ),
        "start_dist_center_m": (
            "m",
            "ratio",
            "dist_center_m of the trip's first point.",
            "Q4: origin centrality",
        ),
        "end_dist_center_m": (
            "m",
            "ratio",
            "dist_center_m of the trip's last point.",
            "Q4: destination centrality",
        ),
        "start_hour_sin": (
            "",
            "interval",
            "sin(2 pi h / 24) of the local start time h (hours with minutes).",
            "Q1: start time as a circle",
        ),
        "start_hour_cos": (
            "",
            "interval",
            "cos(2 pi h / 24) of the local start time.",
            "Q1: start time as a circle",
        ),
        "di2_share": (
            "",
            "ratio",
            "Share of the trip's time (its steps with a known di2) with the raw taximeter signal on; NA when unknown. Under the di2 rule every trip is a taximeter episode, so this only measures the signal's short dropouts; under stop_gap it is the paid share of the trip.",
            "Q3, Q4: paid share of the trip (stop_gap rule only)",
        ),
        "is_weekend": (
            "",
            "nominal (binary, symmetric)",
            "1 when the trip starts on a Saturday or Sunday.",
            "Q5: weekends vs working days",
        ),
        "is_holiday": (
            "",
            "nominal (binary, asymmetric)",
            "1 when the trip starts on the official holiday (June 28).",
            "Q5: single-day case study",
        ),
        "tod_night": (
            "",
            "nominal (binary, asymmetric)",
            "1 when tod_bucket at the start is night.",
            "Q1: when trips start",
        ),
        "tod_morning": (
            "",
            "nominal (binary, asymmetric)",
            "1 when tod_bucket at the start is morning.",
            "Q1: when trips start",
        ),
        "tod_midday": (
            "",
            "nominal (binary, asymmetric)",
            "1 when tod_bucket at the start is midday.",
            "Q1: when trips start",
        ),
        "tod_afternoon": (
            "",
            "nominal (binary, asymmetric)",
            "1 when tod_bucket at the start is afternoon.",
            "Q1: when trips start",
        ),
        "tod_evening": (
            "",
            "nominal (binary, asymmetric)",
            "1 when tod_bucket at the start is evening.",
            "Q1: when trips start",
        ),
    },
    "vehicle_day_model": {
        "vehicle_id": ("", "nominal", "Taxi id.", "identifier"),
        "date_local": ("", "interval", "Local date.", "identifier"),
        "n_points": (
            "",
            "ratio",
            "Points of the vehicle-day (vehicle_day).",
            "none: point counts depend on the reporting rate, not on time",
        ),
        "observed_h": (
            "h",
            "ratio",
            "Observed hours, steps without a gap (vehicle_day).",
            "none: coverage of the day (about 24 h unless the device was off)",
        ),
        "moving_h": (
            "h",
            "ratio",
            "Hours of moving steps (vehicle_day).",
            "Q3: time moving",
        ),
        "km": (
            "km",
            "ratio",
            "Distance of the day, steps without a gap (vehicle_day).",
            "Q1, Q3: how much the taxi works",
        ),
        "avg_speed_moving_kmh": (
            "km/h",
            "ratio",
            (
                "Distance over time of the moving steps with seconds at both ends "
                "(vehicle_day); NA on a day without such steps."
            ),
            "Q2: speed",
        ),
        "max_speed_kmh": (
            "km/h",
            "ratio",
            "Highest reported speed (vehicle_day).",
            "Q2; Phase II: speed outliers",
        ),
        "di1_share": (
            "",
            "ratio",
            "Share of observed time with the ignition on (vehicle_day).",
            "Q3: switched on vs off",
        ),
        "di2_share": (
            "",
            "ratio",
            "Share of observed time with the taximeter on (vehicle_day).",
            "Q3: paid time",
        ),
        "di3_share": (
            "",
            "ratio",
            "Share of observed time with a passenger (vehicle_day); NA when Di3 is unreliable for this vehicle.",
            "Q3: occupied time",
        ),
        "di3_reliable": (
            "",
            "nominal (binary)",
            "Whether Di3 is usable for this vehicle: true when it is true for no more than DI3_STUCK_SHARE of observed ignition-off time; false above the threshold, and NA when that time is unavailable.",
            "none: data-quality flag for interpreting di3_share",
        ),
        "n_gaps": (
            "",
            "ratio",
            "Steps longer than GAP_S (vehicle_day).",
            "none: data-quality indicator",
        ),
        "moving_share": (
            "",
            "ratio",
            "moving_h / observed_h.",
            "Q3: share of time moving",
        ),
        "n_trips": (
            "",
            "ratio",
            (
                "Trips that start on this vehicle-day: 0 when none, NA for a vehicle "
                "without any trip under the di2 rule (its taximeter is never on)."
            ),
            "Q1, Q4: how many trips",
        ),
        "trip_median_distance_m": (
            "m",
            "ratio",
            "Median distance of the day's trips; NA without trips.",
            "Q4: typical trip distance",
        ),
        "trip_median_duration_s": (
            "s",
            "ratio",
            "Median duration of the day's trips; NA without trips.",
            "Q4: typical trip duration",
        ),
        "night_trip_share": (
            "",
            "ratio",
            "Share of the day's trips that start in NIGHT_HOURS; NA without trips.",
            "Q1: night work",
        ),
        "is_weekend": (
            "",
            "nominal (binary, symmetric)",
            "1 on Saturdays and Sundays.",
            "Q5: weekends vs working days",
        ),
        "is_holiday": (
            "",
            "nominal (binary, asymmetric)",
            "1 on the official holiday (June 28).",
            "Q5: single-day case study",
        ),
    },
}


def skewness(values: pd.Series) -> float:
    from scipy.stats import skew

    v = pd.Series(values).dropna().astype("float64")
    return float(skew(v)) if len(v) > 2 and v.std() > 0 else float("nan")


def choose_log(df: pd.DataFrame, table: str) -> tuple[pd.DataFrame, list[dict]]:
    """log1p for each LOG_CANDIDATES column where it lowers |skewness|; the column becomes <name>_log."""
    rows = []
    for col in LOG_CANDIDATES[table]:
        raw = df[col].astype("float64")
        logged = np.log1p(raw)
        before, after = skewness(raw), skewness(logged)
        use_log = bool(np.isfinite(after) and abs(after) < abs(before))
        rows.append(
            {
                "table": table,
                "feature": col,
                "n": int(raw.notna().sum()),
                "skew_raw": round(before, 3),
                "skew_log1p": round(after, 3),
                "kept": "log1p" if use_log else "raw",
                "column": f"{col}_log" if use_log else col,
            }
        )
        if use_log:
            df[col] = logged
            df = df.rename(columns={col: f"{col}_log"})
    return df, rows


def feature_columns(df: pd.DataFrame, table: str) -> list[str]:
    """Candidate features of a model table: everything except ids, one-hot columns and z-scores."""
    return [
        c
        for c in df.columns
        if c not in ID_COLUMNS[table]
        and not c.startswith("tod_")
        and not c.endswith("_z")
        and not PC_COLUMN.match(c)
    ]


def add_zscores(df: pd.DataFrame, table: str) -> tuple[pd.DataFrame, list[dict]]:
    """<feature>_z for every candidate feature, fitted on the whole table (population std)."""
    rows, z = [], {}
    for col in feature_columns(df, table):
        x = df[col].astype("float64")
        mean, std = float(x.mean()), float(x.std(ddof=0))
        z[f"{col}_z"] = ((x - mean) / std if std > 0 else x * np.nan).astype("float32")
        rows.append(
            {
                "table": table,
                "feature": col,
                "n": int(x.notna().sum()),
                "mean": mean,
                "std": std,
                "min": float(x.min()),
                "max": float(x.max()),
            }
        )
    return pd.concat([df, pd.DataFrame(z, index=df.index)], axis=1), rows


def base_name(column: str) -> tuple[str, str]:
    """(spec name, variant) of a model column: variant is '', 'log', 'z', or 'log_z'."""
    name, variant = column, ""
    if name.endswith("_z"):
        name, variant = name[:-2], "z"
    if name.endswith("_log"):
        name, variant = name[:-4], "log" + ("_z" if variant else "")
    return name, variant


def model_dictionary(df: pd.DataFrame, table: str) -> tuple[list[dict], list[dict]]:
    """Data dictionary rows and feature catalogue rows for every column of a model table."""
    spec = MODEL_SPEC[table]
    dictionary, catalog = [], []
    for col in df.columns:
        if PC_COLUMN.match(col):
            continue
        name, variant = base_name(col)
        unit, kind, description, purpose = spec[name]
        if variant.startswith("log"):
            unit, description = (
                f"log1p({unit or 'value'})",
                f"log1p of {name}: {description}",
            )
            kind = "interval"
        if variant.endswith("z"):
            unit = "standard deviations"
            description = f"z-score of {name}{' (log1p)' if 'log' in variant else ''}: (x - mean) / std over the whole table (scaling_params.csv)."
            kind = "interval"
        dictionary.append(
            {
                "column": col,
                "unit": unit,
                "attribute_type": kind,
                "description": description,
                "source": "processed/trips"
                if table == "trips_model"
                else "processed/vehicle_day, processed/trips",
                "owner": STAGE,
            }
        )
        if purpose != "identifier" and not variant.endswith("z"):
            formula = {
                "": "as described",
                "log": f"log1p({name})",
                "z": f"({name} - mean) / std",
                "log_z": f"(log1p({name}) - mean) / std",
            }[variant]
            catalog.append(
                {
                    "feature": col,
                    "formula": formula if variant else description,
                    "unit": unit,
                    "attribute_type": kind,
                    "purpose": purpose,
                    "owner": STAGE,
                }
            )
    if any(c.endswith("_z") for c in df.columns):
        catalog.append(
            {
                "feature": "<feature>_z",
                "formula": "(x - mean) / std of every candidate feature, fitted on the whole "
                "table; parameters in scaling_params.csv",
                "unit": "standard deviations",
                "attribute_type": "interval",
                "purpose": "PCA input and distance-based methods in Phase II: features on one scale",
                "owner": STAGE,
            }
        )
    return dictionary, catalog


def read_point_summaries(trip_ids: pd.Index) -> tuple[pd.DataFrame, dict]:
    """Per-trip values from the final points, and fleet-wide bin and binary counts."""
    binaries = list(BINARY_ATTRIBUTES)
    columns = [
        "vehicle_id",
        "ts_utc",
        "source_row",
        "trip_id",
        "di1",
        "di3",
        "dist_center_m",
        "dt_s",
        "gap_after",
        *POINT_BIN_COLUMNS,
        *binaries,
    ]
    per_trip, bins, binary, reliability = [], {}, {}, []
    for v in io.vehicle_ids("final"):
        df = io.read_points("final", [v], columns=list(dict.fromkeys(columns)))
        df = df.sort_values(["ts_utc", "source_row"], kind="stable").reset_index(
            drop=True
        )
        step_h = (
            df["dt_s"].astype("float64").where(df["dt_s"].notna() & ~df["gap_after"], 0)
            / 3600
        )
        ignition_off = df["di1"].eq(False)
        ignition_off_known = ignition_off & df["di3"].notna()
        off_h = float(step_h[ignition_off].sum())
        di3_off_h = float(step_h[ignition_off_known & df["di3"].fillna(False)].sum())
        di3_off_share = di3_off_h / off_h if off_h > 0 else np.nan
        reliability.append(
            {
                "vehicle_id": int(v),
                "di3_off_share": di3_off_share,
                "di3_reliable": (
                    di3_off_share <= config.DI3_STUCK_SHARE
                    if np.isfinite(di3_off_share)
                    else pd.NA
                ),
            }
        )
        for col in POINT_BIN_COLUMNS:
            sel = (
                df["is_moving"]
                if col == "compass8"
                else pd.Series(True, index=df.index)
            )
            counts = df.loc[sel, col].value_counts(sort=False)
            hours = step_h[sel].groupby(df.loc[sel, col], observed=False).sum()
            prev = bins.get(col)
            part = pd.DataFrame({"count": counts, "observed_h": hours})
            bins[col] = part if prev is None else prev.add(part, fill_value=0)
        for col in binaries:
            s = df[col]
            known = s.notna()
            ones = known & s.fillna(False).astype(bool)
            b = binary.setdefault(
                col, {"n": 0, "ones": 0, "na": 0, "h_ones": 0.0, "h_known": 0.0}
            )
            b["n"] += len(s)
            b["ones"] += int(ones.sum())
            b["na"] += int((~known).sum())
            b["h_ones"] += float(step_h[ones].sum())
            b["h_known"] += float(step_h[known].sum())

        tid = df["trip_id"]
        inside = tid.notna()
        if not inside.any():
            continue
        g = df[inside].groupby("trip_id", sort=False)
        is_last = inside & tid.ne(tid.shift(-1))
        step = inside & ~is_last & df["dt_s"].notna()
        known = step & df["di2"].notna()
        dt = df["dt_s"].astype("float64")
        on = dt.where(known & df["di2"].fillna(False), 0.0).groupby(tid).sum()
        total = dt.where(known, 0.0).groupby(tid).sum()
        per_trip.append(
            pd.DataFrame(
                {
                    "start_dist_center_m": g["dist_center_m"].first().astype("float64"),
                    "end_dist_center_m": g["dist_center_m"].last().astype("float64"),
                    "di2_share": on / total.where(total > 0),
                }
            )
        )
    endpoints = pd.concat(per_trip) if per_trip else pd.DataFrame()
    missing = trip_ids.difference(endpoints.index)
    if len(missing):
        raise PipelineError(
            f"{len(missing):,} trips have no points with their trip_id (first: {missing[0]}); "
            "rerun the trips stage."
        )
    reliability_df = pd.DataFrame(reliability).astype({"di3_reliable": "boolean"})
    return endpoints.reindex(trip_ids), {
        "bins": bins,
        "binary": binary,
        "di3_reliability": reliability_df,
    }


def hourly_profile() -> pd.DataFrame:
    """Share of observed time moving and with the taximeter on, per local hour, working days vs weekends."""
    parts = []
    columns = ["hour", "is_weekend", "dt_s", "gap_after", "is_moving", "di2"]
    for v in io.vehicle_ids("final"):
        df = io.read_points("final", [v], columns=columns)
        ok = df["dt_s"].notna() & ~df["gap_after"]
        h = df["dt_s"].astype("float64").where(ok, 0.0) / 3600
        known = df["di2"].notna()
        part = pd.DataFrame(
            {
                "is_weekend": df["is_weekend"].to_numpy(),
                "hour": df["hour"].to_numpy(),
                "observed_h": h,
                "moving_h": h.where(df["is_moving"], 0.0),
                "di2_known_h": h.where(known, 0.0),
                "di2_h": h.where(known & df["di2"].fillna(False), 0.0),
            }
        )
        parts.append(part.groupby(["is_weekend", "hour"]).sum())
    if not parts:
        raise PipelineError("No final points: run the features stage first.")
    t = pd.concat(parts).groupby(level=[0, 1]).sum().reset_index()
    t["moving_share"] = t["moving_h"] / t["observed_h"].where(t["observed_h"] > 0)
    t["di2_share"] = t["di2_h"] / t["di2_known_h"].where(t["di2_known_h"] > 0)
    t["tod_bucket"] = tod_bucket(t["hour"])
    return t


def build_trips_model(trips: pd.DataFrame, endpoints: pd.DataFrame) -> pd.DataFrame:
    start = trips["start_ts_local"]
    h = start.dt.hour + start.dt.minute / 60 + start.dt.second / 3600
    clock = 2 * np.pi * h / 24
    ep = endpoints.reset_index(drop=True)
    tm = pd.DataFrame(
        {
            "trip_id": trips["trip_id"].to_numpy(),
            "vehicle_id": trips["vehicle_id"].to_numpy(),
            "distance_m": trips["distance_m"].astype("float64"),
            "duration_s": trips["duration_s"].astype("float64"),
            "straight_m": trips["straight_m"].astype("float64"),
            "straightness": trips["straightness"].astype("float64"),
            "avg_speed_kmh": trips["avg_speed_kmh"].astype("float64"),
            "max_speed_kmh": trips["max_speed_kmh"].astype("float64"),
            "n_points": trips["n_points"].astype("float64"),
            "start_dist_center_m": ep["start_dist_center_m"].to_numpy(),
            "end_dist_center_m": ep["end_dist_center_m"].to_numpy(),
            "start_hour_sin": np.sin(clock),
            "start_hour_cos": np.cos(clock),
            "di2_share": ep["di2_share"].to_numpy(),
            "is_weekend": to_binary(trips["is_weekend"]).to_numpy(),
            "is_holiday": to_binary(trips["is_holiday"]).to_numpy(),
        }
    )
    tm["trip_id"] = tm["trip_id"].astype("string")
    return pd.concat([tm, one_hot(trips["tod_bucket"], "tod")], axis=1)


def vehicles_without_trips(trips: pd.DataFrame, vd: pd.DataFrame) -> list[int]:
    """Vehicles with vehicle-days but no trip at all under the di2 rule (taximeter never on)."""
    if config.TRIP_RULE != "di2":
        return []
    with_days = set(vd["vehicle_id"].astype(int))
    return sorted(with_days - set(trips["vehicle_id"].astype(int)))


def build_vehicle_day_model(
    vd: pd.DataFrame, trips: pd.DataFrame, di3_reliability: pd.DataFrame
) -> pd.DataFrame:
    t = trips.assign(
        date_local=trips["start_ts_local"].dt.date,
        night=trips["start_ts_local"].dt.hour.between(
            config.NIGHT_HOURS[0], config.NIGHT_HOURS[1] - 1
        ),
    )
    stats = t.groupby(["vehicle_id", "date_local"]).agg(
        n_trips=("trip_id", "size"),
        trip_median_distance_m=("distance_m", "median"),
        trip_median_duration_s=("duration_s", "median"),
        night_trip_share=("night", "mean"),
    )
    vd = vd.sort_values(["vehicle_id", "date_local"]).reset_index(drop=True)
    key = pd.MultiIndex.from_arrays(
        [vd["vehicle_id"].astype("int64"), pd.to_datetime(vd["date_local"]).dt.date],
        names=["vehicle_id", "date_local"],
    )
    stats.index = stats.index.set_levels(
        stats.index.levels[0].astype("int64"), level="vehicle_id"
    )
    unmatched = stats.index.difference(key)
    if len(unmatched):
        raise PipelineError(
            f"{len(unmatched):,} vehicle-days with trips have no vehicle_day row "
            f"(first: {unmatched[0]})."
        )
    s = stats.reindex(key)
    no_signal = vd["vehicle_id"].isin(vehicles_without_trips(trips, vd)).to_numpy()
    reliability = di3_reliability.set_index("vehicle_id")["di3_reliable"]
    di3_reliable = vd["vehicle_id"].map(reliability).astype("boolean")
    di3_usable = di3_reliable.fillna(False).to_numpy(dtype=bool)
    observed = vd["observed_h"].astype("float64")
    vm = pd.DataFrame(
        {
            "vehicle_id": vd["vehicle_id"].astype("int16"),
            "date_local": vd["date_local"],
            **{
                c: vd[c].astype("float64")
                for c in [
                    "n_points",
                    "observed_h",
                    "moving_h",
                    "km",
                    "avg_speed_moving_kmh",
                    "max_speed_kmh",
                    "di1_share",
                    "di2_share",
                    "n_gaps",
                ]
            },
            "di3_share": vd["di3_share"].where(di3_usable).astype("float64"),
            "di3_reliable": di3_reliable,
            "moving_share": vd["moving_h"].astype("float64")
            / observed.where(observed > 0),
            "n_trips": s["n_trips"].fillna(0).mask(no_signal).to_numpy(dtype="float64"),
            "trip_median_distance_m": s["trip_median_distance_m"].to_numpy(
                dtype="float64"
            ),
            "trip_median_duration_s": s["trip_median_duration_s"].to_numpy(
                dtype="float64"
            ),
            "night_trip_share": s["night_trip_share"].to_numpy(dtype="float64"),
            "is_weekend": to_binary(vd["is_weekend"]).to_numpy(),
            "is_holiday": to_binary(vd["is_holiday"]).to_numpy(),
        }
    )
    return vm


def discretization_table(trips: pd.DataFrame, point_bins: dict) -> pd.DataFrame:
    """T10: trip distance by three methods, and the point bins with points and hours per bin."""
    rows = []
    distance = trips["distance_m"].astype("float64")
    for method in BIN_METHODS if len(distance) else []:
        edges = bin_edges(distance, method)
        binned = apply_bins(distance, edges)
        counts = binned.value_counts(sort=False)
        for i, (interval, count) in enumerate(counts.items()):
            label = (
                config.DISTANCE_BIN_LABELS[i]
                if method == "domain"
                else f"[{interval.left / 1000:.2f}, {interval.right / 1000:.2f}) km"
            )
            rows.append(
                {
                    "variable": "trip distance_m",
                    "method": method,
                    "bin": label,
                    "left": float(interval.left),
                    "right": float(interval.right),
                    "unit": "m",
                    "count": int(count),
                    "share": round(count / len(distance), 4)
                    if len(distance)
                    else np.nan,
                    "observed_h": np.nan,
                    "observed_share": np.nan,
                }
            )
    edges = {
        "speed_bin": (config.SPEED_BINS_KMH, config.SPEED_BIN_LABELS, "km/h"),
        "tod_bucket": (config.TOD_BUCKETS, config.TOD_LABELS, "local hour"),
    }
    for col, part in point_bins.items():
        part = part.reindex(POINT_BIN_LABELS[col], fill_value=0)
        total_n, total_h = part["count"].sum(), part["observed_h"].sum()
        for i, (label, r) in enumerate(part.iterrows()):
            if col == "compass8":
                left, right, unit = (
                    (i * 45 - 22.5) % 360,
                    (i * 45 + 22.5) % 360,
                    "degrees",
                )
            else:
                e, _, unit = edges[col]
                left, right = e[i], e[i + 1]
            rows.append(
                {
                    "variable": col + (" (moving points)" if col == "compass8" else ""),
                    "method": "domain",
                    "bin": str(label),
                    "left": float(left),
                    "right": float(right),
                    "unit": unit,
                    "count": int(r["count"]),
                    "share": round(r["count"] / total_n, 4) if total_n else np.nan,
                    "observed_h": round(float(r["observed_h"]), 2),
                    "observed_share": round(r["observed_h"] / total_h, 4)
                    if total_h
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def binarization_table(binary: dict) -> pd.DataFrame:
    rows = []
    for col, b in binary.items():
        asym, note = BINARY_ATTRIBUTES[col]
        known = b["n"] - b["na"]
        rows.append(
            {
                "column": col,
                "dataset": "processed/points",
                "n": b["n"],
                "ones": b["ones"],
                "zeros": known - b["ones"],
                "na": b["na"],
                "share_ones_points": round(b["ones"] / known, 4) if known else np.nan,
                "share_ones_time": round(b["h_ones"] / b["h_known"], 4)
                if b["h_known"] and col != "gap_after"
                else np.nan,
                "asymmetric": asym,
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def figure_transformations(
    models: dict[str, pd.DataFrame], skew_rows: list[dict]
) -> None:
    """F9: the three size variables before and after log1p, with their skewness."""
    kept = {(r["table"], r["feature"]): r for r in skew_rows}
    fig, axes = figures.subplots(12, 6, nrows=2, ncols=len(F9_VARIABLES))
    for j, (table, col, label) in enumerate(F9_VARIABLES):
        r = kept[(table, col)]
        df = models[table]
        values = df[r["column"]].astype("float64").dropna()
        raw = np.expm1(values) if r["kept"] == "log1p" else values
        for i, (data, kind, sk) in enumerate(
            [(raw, "raw", r["skew_raw"]), (np.log1p(raw), "log1p", r["skew_log1p"])]
        ):
            ax = axes[i, j]
            color = figures.SERIES[0] if r["kept"] == kind else figures.INK_MUTED
            ax.hist(data, bins=50, color=color)
            ax.set_title(
                f"{label}, {kind}: skew {sk:.2f}"
                + (" (kept)" if r["kept"] == kind else "")
            )
            ax.set_xlabel(f"log1p({label})" if kind == "log1p" else label)
            ax.set_ylabel("trips" if table == "trips_model" else "vehicle-days")
    fig.suptitle(
        "F9. Distributions before and after the log transform",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=11,
    )
    figures.save_figure(fig, "f09_transformations")


def figure_discretization(table: pd.DataFrame) -> None:
    """F10: trip distance in equal-width, equal-frequency, and domain-based bins."""
    rows = table[table["variable"] == "trip distance_m"]
    fig, axes = figures.subplots(13, 4.2, ncols=len(BIN_METHODS), sharey=True)
    titles = {
        "equal_width": "Equal width",
        "equal_frequency": "Equal frequency (quantiles)",
        "domain": "Domain-based",
    }
    for ax, method, color in zip(axes, BIN_METHODS, figures.SERIES, strict=False):
        part = rows[rows["method"] == method]
        ax.bar(range(len(part)), part["count"], color=color)
        ax.set_xticks(range(len(part)), part["bin"], rotation=30, ha="right")
        for x, (c, s) in enumerate(zip(part["count"], part["share"], strict=True)):
            ax.text(
                x,
                c,
                f"{s:.0%}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=figures.INK_SECONDARY,
            )
        ax.set_title(titles[method])
        ax.set_xlabel("trip distance bin, left-closed")
    axes[0].set_ylabel("trips")
    fig.suptitle(
        "F10. Discretization of trip distance: three methods",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=11,
    )
    figures.save_figure(fig, "f10_discretization")


def run(dev: bool) -> None:
    """Model tables (trips_model, vehicle_day_model), T10, F9, F10, scaling and skewness tables."""
    from taxi import features

    if not io.table_path("trips").exists():
        raise PipelineError("No trips.parquet: run the trips stage first.")
    if not io.table_path("vehicle_day").exists():
        raise PipelineError(
            "No vehicle_day.parquet: the aggregate stage writes it (needed for vehicle_day_model)."
        )
    trips = io.read_table("trips")
    vd = io.read_table("vehicle_day")
    endpoints, summaries = read_point_summaries(pd.Index(trips["trip_id"]))

    models, skew_rows, scale_rows = {}, [], []
    for table, frame in [
        ("trips_model", build_trips_model(trips, endpoints)),
        (
            "vehicle_day_model",
            build_vehicle_day_model(vd, trips, summaries["di3_reliability"]),
        ),
    ]:
        frame, rows = choose_log(frame, table)
        skew_rows += rows
        frame, rows = add_zscores(frame, table)
        scale_rows += rows
        models[table] = frame

    io.write_report_table(pd.DataFrame(skew_rows), "transformation_skewness")
    io.write_report_table(pd.DataFrame(scale_rows), "scaling_params")
    io.write_report_table(summaries["di3_reliability"], "di3_reliability")
    bins = discretization_table(trips, summaries["bins"])
    io.write_report_table(bins, "discretization_bins")
    io.write_report_table(
        binarization_table(summaries["binary"]), "binarization_summary"
    )
    figure_transformations(models, skew_rows)
    figure_discretization(bins)

    for table, name, dataset in [
        ("trips_model", TRIPS_MODEL, TRIPS_MODEL_DATASET),
        ("vehicle_day_model", VEHICLE_DAY_MODEL, VEHICLE_DAY_MODEL_DATASET),
    ]:
        frame = models[table]
        io.write_table(frame, name)
        dictionary, catalog = model_dictionary(frame, table)
        io.write_dictionary(dataset, dictionary, frame, owner=STAGE)
        features.write_catalog(dataset, catalog, owner=STAGE)

    logged = [r["column"] for r in skew_rows if r["kept"] == "log1p"]
    print(
        f"trips_model {models['trips_model'].shape}, vehicle_day_model "
        f"{models['vehicle_day_model'].shape}; log1p kept for {', '.join(logged) or 'none'}"
    )

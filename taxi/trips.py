"""Stops and trips: a new level of detail, one row per trip."""

import numpy as np
import pandas as pd

from taxi import PipelineError, config, features, geo, io, transform

STAGE = "trips"
TRIPS_DATASET = "processed/trips"
SHORT_FLAG_NOTE = "fewer than TRIP_SHORT_POINTS (3) points or distance_m = 0"

TRIP_COLUMNS = [
    "trip_id",
    "vehicle_id",
    "rule",
    "start_ts_local",
    "end_ts_local",
    "duration_s",
    "distance_m",
    "straight_m",
    "straightness",
    "avg_speed_kmh",
    "max_speed_kmh",
    "n_points",
    "start_cell",
    "end_cell",
    "start_municipality",
    "end_municipality",
    "tod_bucket",
    "is_weekend",
    "is_holiday",
    "flag_short",
]

POINT_ID_DICTIONARY = [
    {
        "column": "stop_id",
        "attribute_type": "nominal",
        "values": "<vehicle_id>-<YYYYMMDD>-S<seq>, NA when not in a stop",
        "description": "Stop (stay point) the point belongs to: consecutive points within "
        "STOP_RADIUS_M (100 m) of the stop's first point for at least STOP_MIN_S (300 s), "
        "never across a gap. Local date of the stop's first point.",
        "source": "x_m, y_m, ts_utc, gap_after",
    },
    {
        "column": "trip_id",
        "attribute_type": "nominal",
        "values": "<vehicle_id>-<YYYYMMDD>-<seq>, NA when not in a trip",
        "description": "Trip the point belongs to, by the rule TRIP_RULE (see processed/trips, "
        "column rule). Local date of the trip start. Under stop_gap, a trip's first and last "
        "points are the last point of the stop before and the first point of the stop "
        "after, so they also have a stop_id.",
        "source": "stop_id, gap_after (stop_gap) or di2, gap_after (di2)",
    },
]

TRIPS_DICTIONARY = [
    {
        "column": "trip_id",
        "attribute_type": "nominal",
        "values": "<vehicle_id>-<YYYYMMDD>-<seq>",
        "description": "Trip id: vehicle, local date of the trip start (a trip that crosses "
        "midnight belongs to the day it starts), and a sequence number from 001.",
        "source": "trips stage",
    },
    {
        "column": "vehicle_id",
        "attribute_type": "nominal",
        "values": "ids between 1 and 119, with gaps",
        "description": "Taxi id.",
        "source": "points",
    },
    {
        "column": "rule",
        "attribute_type": "nominal",
        "values": "stop_gap / di2",
        "description": "Segmentation rule that produced the trip (TRIP_RULE).",
        "source": "config.TRIP_RULE",
    },
    {
        "column": "start_ts_local",
        "attribute_type": "interval",
        "values": "June 1 to August 31, 2023",
        "description": "Local time of the trip's first point.",
        "source": "ts_local",
    },
    {
        "column": "end_ts_local",
        "attribute_type": "interval",
        "values": "June 1 to September 1, 2023",
        "description": "Local time of the trip's last point.",
        "source": "ts_local",
    },
    {
        "column": "duration_s",
        "unit": "s",
        "attribute_type": "ratio",
        "values": "more than 0",
        "description": "end - start: the sum of dt_s over the trip's steps (no gap inside a trip).",
        "source": "dt_s",
    },
    {
        "column": "distance_m",
        "unit": "m",
        "attribute_type": "ratio",
        "values": "0 or more",
        "description": "Sum of dist_m (haversine) over the trip's steps.",
        "source": "dist_m",
    },
    {
        "column": "straight_m",
        "unit": "m",
        "attribute_type": "ratio",
        "values": "0 or more",
        "description": "Haversine distance from the first to the last point.",
        "source": "lat, lon",
    },
    {
        "column": "straightness",
        "attribute_type": "ratio",
        "values": "0 to 1, NA when distance_m = 0",
        "description": "straight_m / distance_m: 1 for a straight trip, near 0 for a round trip.",
        "source": "straight_m, distance_m",
    },
    {
        "column": "avg_speed_kmh",
        "unit": "km/h",
        "attribute_type": "ratio",
        "values": "0 or more",
        "description": "distance_m / duration_s * 3.6 (distance over time, not a mean of points).",
        "source": "distance_m, duration_s",
    },
    {
        "column": "max_speed_kmh",
        "unit": "km/h",
        "attribute_type": "ratio",
        "values": "0 to 200, NA",
        "description": "Highest reported speed_kmh among the trip's points.",
        "source": "speed_kmh",
    },
    {
        "column": "n_points",
        "attribute_type": "ratio",
        "values": "2 or more",
        "description": "Points in the trip, including its first and last point.",
        "source": "points",
    },
    {
        "column": "start_cell",
        "attribute_type": "nominal",
        "values": "<col>_<row>",
        "description": "Grid cell (500 m) of the first point.",
        "source": "cell_id",
    },
    {
        "column": "end_cell",
        "attribute_type": "nominal",
        "values": "<col>_<row>",
        "description": "Grid cell (500 m) of the last point.",
        "source": "cell_id",
    },
    {
        "column": "start_municipality",
        "attribute_type": "nominal",
        "values": "municipality names; NA outside Kosovo",
        "description": "Municipality of the first point.",
        "source": "municipality",
    },
    {
        "column": "end_municipality",
        "attribute_type": "nominal",
        "values": "municipality names; NA outside Kosovo",
        "description": "Municipality of the last point.",
        "source": "municipality",
    },
    {
        "column": "tod_bucket",
        "attribute_type": "ordinal, circular",
        "values": "night, morning, midday, afternoon, evening",
        "description": "Time-of-day bucket of the local start hour.",
        "source": "hour",
    },
    {
        "column": "is_weekend",
        "attribute_type": "nominal",
        "values": "true / false",
        "description": "The trip starts on a Saturday or Sunday.",
        "source": "is_weekend",
    },
    {
        "column": "is_holiday",
        "attribute_type": "nominal",
        "values": "true / false",
        "description": "The trip starts on an official holiday (Kurban Bajram, June 28).",
        "source": "is_holiday",
    },
    {
        "column": "flag_short",
        "attribute_type": "nominal",
        "values": "true / false",
        "description": f"The trip has {SHORT_FLAG_NOTE}. Kept, for Phase II.",
        "source": "n_points, distance_m",
    },
]

POINT_ID_CATALOG = [
    {
        "feature": "stop_id",
        "formula": "stay point: points within 100 m of the first point for at least 300 s, "
        "not across a gap",
        "attribute_type": "nominal",
        "purpose": "Q3 (standing time), Q4 (trip origins and destinations are stops)",
    },
    {
        "feature": "trip_id",
        "formula": "run of trip steps under TRIP_RULE, at least 2 points and a positive duration",
        "attribute_type": "nominal",
        "purpose": "Q4: links every point to its trip",
    },
]

TRIPS_CATALOG = [
    {
        "feature": "duration_s",
        "formula": "end_ts - start_ts (sum of dt_s over the trip's steps)",
        "unit": "s",
        "attribute_type": "ratio",
        "purpose": "Q4: typical trip duration",
    },
    {
        "feature": "distance_m",
        "formula": "sum of dist_m over the trip's steps",
        "unit": "m",
        "attribute_type": "ratio",
        "purpose": "Q4: typical trip distance",
    },
    {
        "feature": "straight_m",
        "formula": "haversine(first point, last point)",
        "unit": "m",
        "attribute_type": "ratio",
        "purpose": "Q4: how far the destination is from the origin",
    },
    {
        "feature": "straightness",
        "formula": "straight_m / distance_m",
        "attribute_type": "ratio",
        "purpose": "Q4: direct rides vs round trips and cruising",
    },
    {
        "feature": "avg_speed_kmh",
        "formula": "distance_m / duration_s * 3.6",
        "unit": "km/h",
        "attribute_type": "ratio",
        "purpose": "Q2: trip speed by time of day and area",
    },
    {
        "feature": "max_speed_kmh",
        "formula": "max(speed_kmh) over the trip's points",
        "unit": "km/h",
        "attribute_type": "ratio",
        "purpose": "Q2; Phase II: speed outliers",
    },
    {
        "feature": "n_points",
        "formula": "count of the trip's points",
        "attribute_type": "ratio",
        "purpose": "No guiding question (depends on the reporting rate): quality only",
    },
    {
        "feature": "start_cell",
        "formula": "cell_id of the first point",
        "attribute_type": "nominal",
        "purpose": "Q4: origin areas (500 m cells)",
    },
    {
        "feature": "end_cell",
        "formula": "cell_id of the last point",
        "attribute_type": "nominal",
        "purpose": "Q4: destination areas (500 m cells)",
    },
    {
        "feature": "start_municipality",
        "formula": "municipality of the first point",
        "attribute_type": "nominal",
        "purpose": "Q4: origin municipality",
    },
    {
        "feature": "end_municipality",
        "formula": "municipality of the last point",
        "attribute_type": "nominal",
        "purpose": "Q4: destination municipality",
    },
    {
        "feature": "tod_bucket",
        "formula": "tod_bucket of the start hour",
        "attribute_type": "ordinal, circular",
        "purpose": "Q1, Q4: when trips start",
    },
    {
        "feature": "flag_short",
        "formula": "n_points < 3 or distance_m = 0",
        "attribute_type": "nominal (binary, asymmetric)",
        "purpose": "Phase II: implausible or trivial trips",
    },
]


# --- Segmentation ---------------------------------------------------------------------


def runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and last index of every run of True in a 1-D boolean array."""
    m = np.concatenate([[0], np.asarray(mask, dtype=np.int8), [0]])
    d = np.diff(m)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1


def detect_stops(
    x_m: np.ndarray,
    y_m: np.ndarray,
    t_s: np.ndarray,
    gap_after: np.ndarray,
    radius_m: float = config.STOP_RADIUS_M,
    min_s: float = config.STOP_MIN_S,
) -> np.ndarray:
    """Stop number per point (0, 1, ... in time order; -1 outside a stop)."""
    n = len(t_s)
    stop = np.full(n, -1, dtype=np.int32)
    xs, ys, ts, gaps = x_m.tolist(), y_m.tolist(), t_s.tolist(), gap_after.tolist()
    r2 = float(radius_m) ** 2
    i, k = 0, 0
    while i < n - 1:
        xi, yi = xs[i], ys[i]
        j = i
        while j + 1 < n and not gaps[j]:
            dx, dy = xs[j + 1] - xi, ys[j + 1] - yi
            if not dx * dx + dy * dy <= r2:
                break
            j += 1
        if j > i and ts[j] - ts[i] >= min_s:
            stop[i : j + 1] = k
            k += 1
            i = j + 1
        else:
            i += 1
    return stop


def bridge_off_runs(
    on: np.ndarray,
    gap_after: np.ndarray,
    t_s: np.ndarray,
    bridge_s: float,
    exact: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """The taximeter signal with short "off" interruptions bridged, and how many were bridged."""
    bridged = on.copy()
    if bridge_s <= 0 or not on.any():
        return bridged, 0
    starts, ends, duration = off_interruptions(on, gap_after, t_s)
    short = duration < bridge_s
    if exact is not None:
        coarse = ~(exact[starts - 1] & exact[ends + 1])
        short |= coarse & (duration <= bridge_s)
    for a, b in zip(starts[short], ends[short], strict=True):
        bridged[a : b + 1] = True
    return bridged, int(short.sum())


def off_interruptions(
    on: np.ndarray, gap_after: np.ndarray, t_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First point, last point, and duration of every gap-free "off" run between two "on" points."""
    starts, ends = runs(~on)
    inner = (starts > 0) & (ends < len(on) - 1)
    starts, ends = starts[inner], ends[inner]
    gap_cum = np.concatenate([[0], np.cumsum(gap_after.astype(np.int64))])
    no_gap = (gap_cum[ends + 1] - gap_cum[starts - 1]) == 0
    starts, ends = starts[no_gap], ends[no_gap]
    return starts, ends, t_s[ends + 1] - t_s[starts]


def trip_steps(
    rule: str,
    stop: np.ndarray,
    di2_on: np.ndarray,
    gap_after: np.ndarray,
    has_step: np.ndarray,
) -> np.ndarray:
    """Which steps (one per point, the step that starts at it) are inside a trip."""
    usable = has_step & ~gap_after
    if rule == "stop_gap":
        same_stop = np.zeros(len(stop), dtype=bool)
        same_stop[:-1] = (stop[:-1] >= 0) & (stop[:-1] == stop[1:])
        return usable & ~same_stop
    if rule == "di2":
        return usable & di2_on
    raise ValueError(f"rule must be one of {config.TRIP_RULES}, got {rule!r}")


def segment(steps: np.ndarray, t_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """First and last point of every trip from the trip steps, and the zero-duration runs left out."""
    s, e = runs(steps)
    first, last = s, e + 1
    positive = t_s[last] - t_s[first] > 0
    return first[positive], last[positive], int((~positive).sum())


def _ids(vehicle_id: int, start_local: pd.Series, sep: str = "") -> np.ndarray:
    day = start_local.dt.strftime("%Y%m%d").reset_index(drop=True)
    seq = day.groupby(day, sort=False).cumcount() + 1
    return (
        f"{int(vehicle_id)}-" + day + "-" + sep + seq.map("{:03d}".format)
    ).to_numpy()


def trip_table(
    df: pd.DataFrame, first: np.ndarray, last: np.ndarray, rule: str
) -> pd.DataFrame:
    """One row per trip of one vehicle (columns of TRIP_COLUMNS)."""
    vehicle_id = int(df["vehicle_id"].iloc[0])
    t_s = df["ts_utc"].astype("int64").to_numpy() / 1e9
    dist = df["dist_m"].astype("float64").fillna(0).to_numpy()
    cum = np.concatenate([[0.0], np.cumsum(dist)])
    distance = cum[last] - cum[first]
    duration = t_s[last] - t_s[first]
    lat, lon = df["lat"].to_numpy(), df["lon"].to_numpy()
    straight = geo.haversine_m(lat[first], lon[first], lat[last], lon[last])

    member = point_trip_index(len(df), first, last)
    inside = member >= 0
    speed = df["speed_kmh"].astype("float64").to_numpy()
    max_speed = (
        pd.Series(speed[inside])
        .groupby(member[inside])
        .max()
        .reindex(range(len(first)))
        .to_numpy()
    )
    start_local = df["ts_local"].iloc[first].reset_index(drop=True)
    n_points = last - first + 1
    with np.errstate(divide="ignore", invalid="ignore"):
        straightness = np.where(
            distance > 0, np.minimum(straight / distance, 1.0), np.nan
        )
        avg_speed = distance / duration * 3.6
    return pd.DataFrame(
        {
            "trip_id": pd.array(_ids(vehicle_id, start_local), dtype="string"),
            "vehicle_id": np.full(len(first), vehicle_id, dtype="int16"),
            "rule": pd.Categorical(
                [rule] * len(first), categories=list(config.TRIP_RULES)
            ),
            "start_ts_local": start_local,
            "end_ts_local": df["ts_local"].iloc[last].reset_index(drop=True),
            "duration_s": duration.astype("float32"),
            "distance_m": distance.astype("float32"),
            "straight_m": straight.astype("float32"),
            "straightness": straightness.astype("float32"),
            "avg_speed_kmh": avg_speed.astype("float32"),
            "max_speed_kmh": max_speed.astype("float32"),
            "n_points": n_points.astype("int32"),
            "start_cell": df["cell_id"].iloc[first].astype("string").to_numpy(),
            "end_cell": df["cell_id"].iloc[last].astype("string").to_numpy(),
            "start_municipality": df["municipality"].iloc[first].to_numpy(),
            "end_municipality": df["municipality"].iloc[last].to_numpy(),
            "tod_bucket": transform.tod_bucket(
                df["hour"].iloc[first].reset_index(drop=True)
            ),
            "is_weekend": df["is_weekend"].iloc[first].to_numpy(dtype=bool),
            "is_holiday": df["is_holiday"].iloc[first].to_numpy(dtype=bool),
            "flag_short": (n_points < config.TRIP_SHORT_POINTS) | (distance == 0),
        }
    )


def point_trip_index(n: int, first: np.ndarray, last: np.ndarray) -> np.ndarray:
    """Trip number of every point (-1 outside a trip). Trips never share a point."""
    starts = np.zeros(n + 1, dtype=np.int64)
    ends = np.zeros(n + 1, dtype=np.int64)
    np.add.at(starts, first, 1)
    np.add.at(ends, last + 1, 1)
    inside = np.cumsum(starts - ends)[:n] > 0
    number = np.cumsum(starts)[:n] - 1
    return np.where(inside, number, -1)


# --- Per vehicle ----------------------------------------------------------------------


def vehicle_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "t_s": df["ts_utc"].astype("int64").to_numpy() / 1e9,
        "gap": df["gap_after"].to_numpy(dtype=bool),
        "has_step": df["dt_s"].notna().to_numpy(),
        "dt": df["dt_s"].astype("float64").fillna(0).to_numpy(),
        "dist": df["dist_m"].astype("float64").fillna(0).to_numpy(),
        "di2": df["di2"].fillna(False).to_numpy(dtype=bool),
        "exact": df["ts_has_seconds"].fillna(False).to_numpy(dtype=bool),
    }


def process_vehicle(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Stops, both trip rules, and the bridge sensitivity for one vehicle."""
    df = df.drop(columns=["stop_id", "trip_id"], errors="ignore")
    df = df.sort_values(["ts_utc", "source_row"], kind="stable").reset_index(drop=True)
    vehicle_id = int(df["vehicle_id"].iloc[0])
    a = vehicle_arrays(df)
    usable = a["has_step"] & ~a["gap"]
    observed_s = float(a["dt"][usable].sum())
    total_m = float(a["dist"][usable].sum())
    n_days = int(df["date_local"].nunique())

    stop = detect_stops(df["x_m"].to_numpy(), df["y_m"].to_numpy(), a["t_s"], a["gap"])
    n_stops = int(stop.max()) + 1 if len(stop) else 0
    stop_first = np.flatnonzero((stop >= 0) & (np.r_[-1, stop[:-1]] != stop))
    stop_last = np.flatnonzero((stop >= 0) & (np.r_[stop[1:], -1] != stop))
    stop_dur = a["t_s"][stop_last] - a["t_s"][stop_first]
    stop_ids = np.full(len(df), None, dtype=object)
    if n_stops:
        ids = _ids(vehicle_id, df["ts_local"].iloc[stop_first], sep="S")
        stop_ids[stop >= 0] = ids[stop[stop >= 0]]
    df["stop_id"] = pd.array(stop_ids, dtype="string")

    di2_on, n_bridged = bridge_off_runs(
        a["di2"], a["gap"], a["t_s"], config.DI2_BRIDGE_S, a["exact"]
    )
    stats = {
        "vehicle_id": vehicle_id,
        "points": len(df),
        "vehicle_days": n_days,
        "observed_s": observed_s,
        "distance_m": total_m,
        "n_stops": n_stops,
        "stop_durations": stop_dur,
        "stop_time_s": float(stop_dur.sum()),
        "di2_bridged": n_bridged,
        "rules": {},
    }
    chosen = None
    for rule in config.TRIP_RULES:
        steps = trip_steps(rule, stop, di2_on, a["gap"], a["has_step"])
        first, last, zero_runs = segment(steps, a["t_s"])
        trips = trip_table(df, first, last, rule)
        per_day = trips["start_ts_local"].dt.date.value_counts()
        stats["rules"][rule] = {
            "trips": trips,
            "zero_duration_runs": zero_runs,
            "trips_per_day": per_day.reindex(
                df["date_local"].unique(), fill_value=0
            ).to_numpy(),
        }
        stats["rules"][rule]["spans"] = (first, last)
        if rule == config.TRIP_RULE:
            chosen = (trips, first, last)

    stats["has_di2"] = bool(a["di2"].any())
    stats["minute_control"] = minute_bridge_control(a, config.DI2_BRIDGE_S)
    stats["overlap"] = rule_overlap(a, stats["rules"])
    trips, first, last = chosen
    member = point_trip_index(len(df), first, last)
    trip_ids = np.full(len(df), None, dtype=object)
    trip_ids[member >= 0] = trips["trip_id"].to_numpy()[member[member >= 0]]
    df["trip_id"] = pd.array(trip_ids, dtype="string")

    stats["sensitivity"] = bridge_sensitivity(a)
    starts, ends, durations = off_interruptions(a["di2"], a["gap"], a["t_s"])
    stats["interruptions_s"] = durations
    stats["interruptions_exact"] = a["exact"][starts - 1] & a["exact"][ends + 1]
    return df, trips, stats


def minute_bridge_control(a: dict[str, np.ndarray], bridge_s: float) -> dict | None:
    """Control test of the bridging rule for timestamps without seconds, on one vehicle."""
    exact = a["exact"]
    if exact.all() or not exact.any():
        return None
    t = a["t_s"][exact]
    gap = np.zeros(len(t), dtype=bool)
    gap[:-1] = np.diff(t) > config.GAP_S
    starts, ends, true_s = off_interruptions(a["di2"][exact], gap, t)
    minute = np.floor(t / 60) * 60
    measured = minute[ends + 1] - minute[starts]
    truth = true_s < bridge_s
    less_than, at_most = measured < bridge_s, measured <= bridge_s
    return {
        "interruptions": len(truth),
        "bridged_exact": int(truth.sum()),
        "agree_less_than": int((less_than == truth).sum()),
        "agree_at_most": int((at_most == truth).sum()),
        "missed_less_than": int((truth & ~less_than).sum()),
        "wrongly_bridged_at_most": int((at_most & ~truth).sum()),
    }


def minute_bridge_table(stats: list[dict]) -> pd.DataFrame:
    """di2_minute_bridge_control.csv: the control test of minute_bridge_control, per vehicle and in total."""
    rows = [
        {"vehicle_id": str(g["vehicle_id"]), **g["minute_control"]}
        for g in stats
        if g["minute_control"]
    ]
    columns = [
        "vehicle_id",
        "interruptions",
        "bridged_exact",
        "agree_less_than",
        "agree_at_most",
        "missed_less_than",
        "wrongly_bridged_at_most",
    ]
    out = pd.DataFrame(rows, columns=columns)
    if len(out):
        total = out[columns[1:]].sum()
        out = pd.concat(
            [out, pd.DataFrame([{"vehicle_id": "all", **total}])], ignore_index=True
        )
    n = out["interruptions"].where(out["interruptions"] > 0)
    out["share_agree_less_than"] = (out["agree_less_than"] / n).round(4)
    out["share_agree_at_most"] = (out["agree_at_most"] / n).round(4)
    out.insert(1, "bridge_s", config.DI2_BRIDGE_S)
    return out


def rule_overlap(a: dict[str, np.ndarray], rules: dict) -> dict[str, np.ndarray]:
    """How the two rules relate on one vehicle."""
    sg_first, sg_last = rules["stop_gap"]["spans"]
    d2_first, _ = rules["di2"]["spans"]
    pos = np.searchsorted(sg_first, d2_first, side="right") - 1
    inside = (pos >= 0) & (d2_first < sg_last[np.maximum(pos, 0)])
    rides = np.bincount(pos[inside], minlength=len(sg_first))
    if not a["di2"].any():
        return {
            "rides_per_stop_gap_trip": rides,
            "metered_share": np.full(len(sg_first), np.nan),
        }
    on, _ = bridge_off_runs(
        a["di2"], a["gap"], a["t_s"], config.DI2_BRIDGE_S, a["exact"]
    )
    dt_on = np.concatenate([[0.0], np.cumsum(np.where(on & ~a["gap"], a["dt"], 0.0))])
    total = a["t_s"][sg_last] - a["t_s"][sg_first]
    metered = (dt_on[sg_last] - dt_on[sg_first]) / np.where(total > 0, total, np.nan)
    return {"rides_per_stop_gap_trip": rides, "metered_share": metered}


def bridge_sensitivity(a: dict[str, np.ndarray]) -> dict[float, dict]:
    """di2 episodes of one vehicle for every bridge threshold in DI2_BRIDGE_COMPARE_S."""
    out = {}
    for bridge in config.DI2_BRIDGE_COMPARE_S:
        on, n_bridged = bridge_off_runs(
            a["di2"], a["gap"], a["t_s"], bridge, a["exact"]
        )
        steps = a["has_step"] & ~a["gap"] & on
        first, last, _ = segment(steps, a["t_s"])
        cum = np.concatenate([[0.0], np.cumsum(a["dist"])])
        out[bridge] = {
            "durations": a["t_s"][last] - a["t_s"][first],
            "distances": cum[last] - cum[first],
            "bridged": n_bridged,
        }
    return out


# --- Checks and comparison tables -----------------------------------------------------


def _vehicle_day_km() -> pd.Series | None:
    path = io.table_path("vehicle_day")
    if not path.exists():
        return None
    vd = pd.read_parquet(path, columns=["vehicle_id", "km"])
    return vd.groupby("vehicle_id")["km"].sum().astype("float64") * 1000


def check_trips(
    trips: pd.DataFrame,
    points_m: dict[int, float],
    points_n: dict[int, tuple[int, int]],
) -> pd.DataFrame:
    """The trip checks as a table. Raises PipelineError when one fails."""
    rows = []

    def add(check: str, failing: int, total: int, note: str = "", status=None) -> None:
        rows.append(
            {
                "check": check,
                "status": status or ("pass" if failing == 0 else "FAIL"),
                "failing": failing,
                "total": total,
                "note": note,
            }
        )

    add(
        "every trip has at least 2 points",
        int((trips["n_points"] < 2).sum()),
        len(trips),
    )
    add(
        "every trip has a positive duration",
        int((trips["duration_s"] <= 0).sum()),
        len(trips),
    )
    t = trips.sort_values(["vehicle_id", "start_ts_local"])
    prev_end = t.groupby("vehicle_id")["end_ts_local"].shift(1)
    add(
        "trips of one vehicle do not overlap",
        int((t["start_ts_local"] < prev_end).sum()),
        len(trips),
        "the next trip starts at or after the end of the previous one",
    )
    trip_m = trips.groupby("vehicle_id")["distance_m"].sum().astype("float64")
    over = [v for v, m in trip_m.items() if m > points_m[v] * (1 + 1e-9) + 1e-6]
    add(
        "per vehicle, sum of trip distances <= sum of step distances without a gap",
        len(over),
        len(points_m),
        f"vehicles over: {over}" if over else "",
    )
    vd_m = _vehicle_day_km()
    if vd_m is None:
        add(
            "per vehicle, sum of trip distances <= sum of vehicle_day.km",
            0,
            len(points_m),
            "vehicle_day.parquet not found (aggregate stage): not checked",
            status="skipped",
        )
    else:
        over = [v for v, m in trip_m.items() if m > vd_m.get(v, 0.0) * (1 + 1e-4) + 1.0]
        add(
            "per vehicle, sum of trip distances <= sum of vehicle_day.km",
            len(over),
            len(trip_m),
            f"vehicles over: {over}"
            if over
            else "km is float32: tolerance 0.01% + 1 m",
        )
    bad_rows = [v for v, (n_in, n_out) in points_n.items() if n_in != n_out]
    add(
        "points: rows written = rows read (nothing dropped)",
        len(bad_rows),
        len(points_n),
    )
    table = pd.DataFrame(rows)
    io.write_report_table(table, "trip_checks")
    failed = table[table["status"] == "FAIL"]
    if not failed.empty:
        raise PipelineError(
            "Trip checks failed (see trip_checks.csv): " + "; ".join(failed["check"])
        )
    return table


def _summary_row(label: str, rule: str, group: list[dict]) -> dict:
    trips = pd.concat([g["rules"][rule]["trips"] for g in group], ignore_index=True)
    per_day = np.concatenate([g["rules"][rule]["trips_per_day"] for g in group])
    observed = sum(g["observed_s"] for g in group)
    distance = sum(g["distance_m"] for g in group)
    row = {
        "rule": rule,
        "vehicles": label,
        "vehicle_days": int(sum(g["vehicle_days"] for g in group)),
        "vehicles_with_trips": sum(1 for g in group if len(g["rules"][rule]["trips"])),
        "trips": len(trips),
        "trips_per_vehicle_day_mean": round(float(per_day.mean()), 2)
        if len(per_day)
        else np.nan,
        "trips_per_vehicle_day_median": float(np.median(per_day))
        if len(per_day)
        else np.nan,
        "duration_min_median": round(float(trips["duration_s"].median() / 60), 2),
        "duration_min_p25": round(float(trips["duration_s"].quantile(0.25) / 60), 2),
        "duration_min_p75": round(float(trips["duration_s"].quantile(0.75) / 60), 2),
        "distance_km_median": round(float(trips["distance_m"].median() / 1000), 3),
        "distance_km_p25": round(float(trips["distance_m"].quantile(0.25) / 1000), 3),
        "distance_km_p75": round(float(trips["distance_m"].quantile(0.75) / 1000), 3),
        "share_time_in_trips": round(float(trips["duration_s"].sum()) / observed, 4)
        if observed
        else np.nan,
        "share_distance_in_trips": round(float(trips["distance_m"].sum()) / distance, 4)
        if distance
        else np.nan,
        "share_flag_short": round(float(trips["flag_short"].mean()), 4)
        if len(trips)
        else np.nan,
        "straightness_median": round(float(trips["straightness"].median()), 3)
        if trips["straightness"].notna().any()
        else np.nan,
        "zero_duration_runs": int(
            sum(g["rules"][rule]["zero_duration_runs"] for g in group)
        ),
    }
    row["vehicles_without_di2"] = sum(1 for g in group if not g["has_di2"])
    if rule == "stop_gap":
        rides = np.concatenate([g["overlap"]["rides_per_stop_gap_trip"] for g in group])
        metered = np.concatenate([g["overlap"]["metered_share"] for g in group])
        metered = metered[np.isfinite(metered)]
        row["di2_rides_per_trip_mean"] = (
            round(float(rides.mean()), 2) if len(rides) else np.nan
        )
        row["share_trips_with_2plus_rides"] = (
            round(float((rides >= 2).mean()), 4) if len(rides) else np.nan
        )
        row["metered_time_share_median"] = (
            round(float(np.median(metered)), 3) if len(metered) else np.nan
        )
        row["share_trips_unmetered"] = (
            round(float((metered == 0).mean()), 4) if len(metered) else np.nan
        )
        durations = np.concatenate([g["stop_durations"] for g in group])
        row["stops"] = int(sum(g["n_stops"] for g in group))
        row["stop_min_median"] = (
            round(float(np.median(durations) / 60), 2) if len(durations) else np.nan
        )
        row["share_time_in_stops"] = (
            round(sum(g["stop_time_s"] for g in group) / observed, 4)
            if observed
            else np.nan
        )
    else:
        row["di2_interruptions_bridged"] = int(sum(g["di2_bridged"] for g in group))
    return row


def rule_comparison(stats: list[dict]) -> pd.DataFrame:
    """trip_rule_comparison.csv: both rules, per vehicle and for all vehicles together."""
    rows = []
    for rule in config.TRIP_RULES:
        rows.append(_summary_row("all", rule, stats))
        for g in stats:
            rows.append(_summary_row(str(g["vehicle_id"]), rule, [g]))
    return pd.DataFrame(rows)


def sensitivity_table(stats: list[dict]) -> pd.DataFrame:
    """di2_bridge_sensitivity.csv: di2 episodes by bridge threshold, all vehicles together."""
    rows = []
    for bridge in config.DI2_BRIDGE_COMPARE_S:
        dur = np.concatenate([g["sensitivity"][bridge]["durations"] for g in stats])
        dist = np.concatenate([g["sensitivity"][bridge]["distances"] for g in stats])
        rows.append(
            {
                "bridge_s": bridge,
                "chosen": bridge == config.DI2_BRIDGE_S,
                "interruptions_bridged": int(
                    sum(g["sensitivity"][bridge]["bridged"] for g in stats)
                ),
                "episodes": len(dur),
                "duration_min_median": round(float(np.median(dur) / 60), 2)
                if len(dur)
                else np.nan,
                "duration_min_p25": round(float(np.quantile(dur, 0.25) / 60), 2)
                if len(dur)
                else np.nan,
                "duration_min_p75": round(float(np.quantile(dur, 0.75) / 60), 2)
                if len(dur)
                else np.nan,
                "distance_km_median": round(float(np.median(dist) / 1000), 3)
                if len(dur)
                else np.nan,
                "share_under_1_min": round(float((dur < 60).mean()), 4)
                if len(dur)
                else np.nan,
                "taximeter_on_h": round(float(dur.sum() / 3600), 1),
            }
        )
    out = pd.DataFrame(rows)
    out["episodes_change_vs_previous"] = out["episodes"].pct_change().round(4)
    out["episodes_change_per_10_s"] = (
        out["episodes"].pct_change() / out["bridge_s"].diff() * 10
    ).round(4)
    return out


INTERRUPTION_BINS_S = [
    0,
    5,
    10,
    15,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    120,
    180,
    300,
    600,
    1800,
    np.inf,
]


def interruption_table(stats: list[dict]) -> pd.DataFrame:
    """di2_interruptions.csv: how long the taximeter "off" interruptions between two "on" points last."""
    durations = np.concatenate([g["interruptions_s"] for g in stats])
    exact = np.concatenate([g["interruptions_exact"] for g in stats])
    edges = INTERRUPTION_BINS_S

    def count(values: np.ndarray) -> np.ndarray:
        binned = pd.cut(pd.Series(values, dtype="float64"), bins=edges, right=False)
        return binned.value_counts(sort=False).to_numpy()

    n_exact, n_minute = count(durations[exact]), count(durations[~exact])
    total = max(int(n_exact.sum()), 1)
    width = np.diff(np.array(edges, dtype="float64"))
    return pd.DataFrame(
        {
            "from_s": edges[:-1],
            "to_s": edges[1:],
            "interruptions": n_exact,
            "share": np.round(n_exact / total, 4),
            "cumulative_share": np.round(np.cumsum(n_exact) / total, 4),
            "per_second": np.where(
                np.isfinite(width), np.round(n_exact / width, 2), np.nan
            ),
            "interruptions_minute_only": n_minute,
            "bridged": [b <= config.DI2_BRIDGE_S for b in edges[1:]],
        }
    )


# --- Stage ----------------------------------------------------------------------------


def run(dev: bool) -> None:
    """Stops and trips for every vehicle: final points with stop_id and trip_id, trips.parquet."""
    ids = io.vehicle_ids("final")
    if not ids:
        raise PipelineError("No final points: run the features stage first.")
    if config.TRIP_RULE not in config.TRIP_RULES:
        raise PipelineError(f"TRIP_RULE must be one of {config.TRIP_RULES}.")

    all_trips, stats, municipalities = [], [], []
    points_m, points_n = {}, {}
    for v in ids:
        df = io.read_points("final", [v])
        n_in = len(df)
        points, trips, s = process_vehicle(df)
        io.write_points(points, "final")
        points_n[v] = (n_in, len(points))
        points_m[v] = s["distance_m"]
        all_trips.append(trips)
        stats.append(s)
        municipalities += [
            m for m in points["municipality"].cat.categories if m not in municipalities
        ]

    trips = pd.concat(all_trips, ignore_index=True)
    for col in ("start_cell", "end_cell"):
        trips[col] = trips[col].astype("category")
    for col in ("start_municipality", "end_municipality"):
        trips[col] = pd.Categorical(trips[col], categories=municipalities)
    trips = trips[TRIP_COLUMNS]

    check_trips(trips, points_m, points_n)
    io.write_table(trips, "trips")
    io.write_report_table(rule_comparison(stats), "trip_rule_comparison")
    io.write_report_table(sensitivity_table(stats), "di2_bridge_sensitivity")
    io.write_report_table(interruption_table(stats), "di2_interruptions")
    io.write_report_table(minute_bridge_table(stats), "di2_minute_bridge_control")

    point_rows = features.dataset_dictionary(features.FINAL_DATASET)
    point_rows = [r for r in point_rows if r["column"] not in ("stop_id", "trip_id")]
    io.write_dictionary(
        features.FINAL_DATASET, point_rows + POINT_ID_DICTIONARY, points, owner=STAGE
    )
    io.write_dictionary(TRIPS_DATASET, TRIPS_DICTIONARY, trips, owner=STAGE)
    features.write_catalog(
        features.FINAL_DATASET,
        [{**r, "owner": features.STAGE} for r in features.FEATURE_CATALOG]
        + POINT_ID_CATALOG,
        owner=STAGE,
    )
    features.write_catalog(TRIPS_DATASET, TRIPS_CATALOG, owner=STAGE)

    short = int(trips["flag_short"].sum())
    print(
        f"{len(ids)} vehicles: {sum(s['n_stops'] for s in stats):,} stops, {len(trips):,} trips "
        f"(rule {config.TRIP_RULE}), {short:,} flagged short"
    )

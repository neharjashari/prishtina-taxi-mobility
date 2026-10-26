"""Agregimi i pikave te pastra ne tri tabela: vehicle_day, cell_hour dhe fleet_hour.

Kohezgjatjet, distancat dhe pjeset llogariten si shuma mbi hapat pa boshllak, dhe cdo
shpejtesi mesatare eshte distanca e hapave ne levizje pjesetuar me kohen e tyre, vetem per
hapat me sekonda ne te dy skajet (T2: koha pa sekonda e prish shpejtesine). Nje hap
i takon dates, ores dhe qelizes se pikes ku fillon.

aggregation_summary.csv (T7) krahason numrin e rreshtave dhe koeficientin e variacionit
te shpejtesise ne nivel pike, qelize-ore dhe taksi-dite. F7 tregon sa taksi jane aktive ne
cdo ore te dites, me kufijte e TOD_BUCKETS.
"""

import numpy as np
import pandas as pd

from taxi import PipelineError, config, figures, io

STEP_SUMS = ["t", "t_moving", "d", "d_moving", "t_speed", "d_speed"]
DI_COLUMNS = ["di1", "di2", "di3"]

VEHICLE_DAY_DICTIONARY = [
    {"column": "vehicle_id", "attribute_type": "nominal", "description": "Vehicle."},
    {
        "column": "date_local",
        "attribute_type": "interval",
        "description": "Local date. Days without data have no row (M6).",
    },
    {
        "column": "n_points",
        "attribute_type": "ratio",
        "description": "Points of the vehicle on the day.",
    },
    {
        "column": "first_ts_local",
        "attribute_type": "interval",
        "description": "First point of the day.",
    },
    {
        "column": "last_ts_local",
        "attribute_type": "interval",
        "description": "Last point of the day.",
    },
    {
        "column": "observed_h",
        "unit": "h",
        "attribute_type": "ratio",
        "description": "Sum of dt_s over the day's steps without a gap.",
    },
    {
        "column": "moving_h",
        "unit": "h",
        "attribute_type": "ratio",
        "description": "Sum of dt_s over the day's moving steps (speed_kmh >= V_MOVE_KMH) without a gap.",
    },
    {
        "column": "km",
        "unit": "km",
        "attribute_type": "ratio",
        "description": "Sum of dist_m over the day's steps without a gap, in km.",
    },
    {
        "column": "avg_speed_moving_kmh",
        "unit": "km/h",
        "attribute_type": "ratio",
        "description": "Distance over time of the moving steps with seconds at both ends (T2). NA without such steps.",
    },
    {
        "column": "max_speed_kmh",
        "unit": "km/h",
        "attribute_type": "ratio",
        "description": "Highest speed_kmh of the day.",
    },
    {
        "column": "di1_share",
        "attribute_type": "ratio",
        "values": "0 to 1",
        "description": "Share of observed time with di1 (ignition) true; steps with di1 NA left out.",
    },
    {
        "column": "di2_share",
        "attribute_type": "ratio",
        "values": "0 to 1",
        "description": "Share of observed time with di2 (taximeter) true; steps with di2 NA left out.",
    },
    {
        "column": "di3_share",
        "attribute_type": "ratio",
        "values": "0 to 1",
        "description": "Share of observed time with di3 (passenger) true; steps with di3 NA left out.",
    },
    {
        "column": "n_gaps",
        "attribute_type": "ratio",
        "description": "Steps of the day with gap_after true.",
    },
    {
        "column": "is_weekend",
        "attribute_type": "nominal",
        "values": "true / false",
        "description": "Saturday or Sunday.",
    },
    {
        "column": "is_holiday",
        "attribute_type": "nominal",
        "values": "true / false",
        "description": "Official holiday (calendar_2023.csv).",
    },
]
CELL_HOUR_DICTIONARY = [
    {
        "column": "cell_id",
        "attribute_type": "nominal",
        "values": "<col>_<row>",
        "description": "Grid cell of GRID_SIZE_M (500 m) in EPSG:32634.",
    },
    {
        "column": "date_local",
        "attribute_type": "interval",
        "description": "Local date.",
    },
    {
        "column": "hour",
        "unit": "h",
        "attribute_type": "interval, circular",
        "values": "0 to 23",
        "description": "Local hour.",
    },
    {
        "column": "n_points",
        "attribute_type": "ratio",
        "description": "Points in the cell and hour.",
    },
    {
        "column": "n_vehicles",
        "attribute_type": "ratio",
        "description": "Distinct vehicles with a point in the cell and hour.",
    },
    {
        "column": "observed_min",
        "unit": "min",
        "attribute_type": "ratio",
        "description": "Sum of dt_s over steps without a gap that start in the cell and hour.",
    },
    {
        "column": "moving_min",
        "unit": "min",
        "attribute_type": "ratio",
        "description": "The same over moving steps.",
    },
    {
        "column": "avg_speed_moving_kmh",
        "unit": "km/h",
        "attribute_type": "ratio",
        "description": "Distance over time of those moving steps with seconds at both ends (T2). NA without such steps.",
    },
    {
        "column": "municipality",
        "attribute_type": "nominal",
        "description": "Municipality with the most points of the cell over the whole run; "
        "NA outside Kosovo.",
    },
]
FLEET_HOUR_DICTIONARY = [
    {
        "column": "hour_start_local",
        "attribute_type": "interval",
        "description": "Start of the local hour; every hour of the study window.",
    },
    {
        "column": "n_vehicles_active",
        "attribute_type": "ratio",
        "description": "Vehicles with at least one point in the hour.",
    },
    {
        "column": "n_vehicles_moving",
        "attribute_type": "ratio",
        "description": "Vehicles with at least one moving step without a gap in the hour.",
    },
    {
        "column": "n_points",
        "attribute_type": "ratio",
        "description": "Points in the hour.",
    },
    {
        "column": "avg_speed_moving_kmh",
        "unit": "km/h",
        "attribute_type": "ratio",
        "description": "Distance over time of the moving steps with seconds at both ends (T2). NA without such steps.",
    },
    {
        "column": "di2_share",
        "attribute_type": "ratio",
        "values": "0 to 1",
        "description": "Share of observed time with di2 (taximeter) true; steps with di2 NA "
        "left out. NA without observed time.",
    },
]


def steps(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["dt_s"].astype("float64")
    dist = df["dist_m"].astype("float64")
    observed = dt.notna() & ~df["gap_after"]
    moving = observed & (df["speed_kmh"] >= config.V_MOVE_KMH)
    seconds = df["ts_has_seconds"] & df["ts_has_seconds"].shift(-1, fill_value=True)
    out = pd.DataFrame(
        {
            "t": dt.where(observed, 0),
            "t_moving": dt.where(moving, 0),
            "d": dist.where(observed, 0),
            "d_moving": dist.where(moving, 0),
            "t_speed": dt.where(moving & seconds, 0),
            "d_speed": dist.where(moving & seconds, 0),
            "moving": moving,
        }
    )
    for col in DI_COLUMNS:
        out[f"{col}_on"] = out["t"].where(df[col].eq(True).fillna(False), 0)
        out[f"{col}_known"] = out["t"].where(df[col].notna(), 0)
    return out


def speed(distance_m: pd.Series, time_s: pd.Series) -> pd.Series:
    return (distance_m / time_s.where(time_s > 0) * 3.6).astype("float32")


def share(on: pd.Series, known: pd.Series) -> pd.Series:
    return (on / known.where(known > 0)).astype("float32")


def vehicle_day(df: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    g = pd.concat([df, s], axis=1).groupby("date_local", sort=True)
    sums = g[
        [*STEP_SUMS, *[f"{c}_{k}" for c in DI_COLUMNS for k in ("on", "known")]]
    ].sum()
    out = pd.DataFrame(
        {
            "vehicle_id": df["vehicle_id"].iloc[0],
            "n_points": g.size().astype("int32"),
            "first_ts_local": g["ts_local"].min(),
            "last_ts_local": g["ts_local"].max(),
            "observed_h": (sums["t"] / 3600).astype("float32"),
            "moving_h": (sums["t_moving"] / 3600).astype("float32"),
            "km": (sums["d"] / 1000).astype("float32"),
            "avg_speed_moving_kmh": speed(sums["d_speed"], sums["t_speed"]),
            "max_speed_kmh": g["speed_kmh"].max().astype("float32"),
            **{
                f"{c}_share": share(sums[f"{c}_on"], sums[f"{c}_known"])
                for c in DI_COLUMNS
            },
            "n_gaps": g["gap_after"].sum().astype("int16"),
            "is_weekend": g["is_weekend"].first(),
            "is_holiday": g["is_holiday"].first(),
        }
    )
    out["vehicle_id"] = out["vehicle_id"].astype("int16")
    return out.reset_index()[[r["column"] for r in VEHICLE_DAY_DICTIONARY]]


def cell_hour_part(df: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    keys = ["cell_id", "date_local", "hour"]
    g = pd.concat([df[keys], s], axis=1).groupby(keys, sort=False)
    out = g[STEP_SUMS].sum()
    out["n_points"] = g.size()
    return out


def fleet_hour_part(df: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    hour_start = df["ts_local"].dt.floor("h").rename("hour_start_local")
    g = s.groupby(hour_start, sort=False)
    out = g[["t_speed", "d_speed", "di2_on", "di2_known"]].sum()
    out["n_points"] = g.size()
    out["moving"] = g["moving"].any()
    return out


def cell_municipality(counts: pd.Series) -> pd.Series:
    counts = counts[counts > 0].reset_index(name="n")
    top = counts.sort_values("n", ascending=False).drop_duplicates("cell_id")
    return top.set_index("cell_id")["municipality"]


def build_cell_hour(parts: list[pd.DataFrame], municipality: pd.Series) -> pd.DataFrame:
    all_parts = pd.concat(parts)
    g = all_parts.groupby(level=[0, 1, 2], sort=True)
    sums = g.sum()
    out = pd.DataFrame(
        {
            "n_points": sums["n_points"].astype("int32"),
            "n_vehicles": g.size().astype("int16"),
            "observed_min": (sums["t"] / 60).astype("float32"),
            "moving_min": (sums["t_moving"] / 60).astype("float32"),
            "avg_speed_moving_kmh": speed(sums["d_speed"], sums["t_speed"]),
        }
    ).reset_index()
    out["cell_id"] = out["cell_id"].astype("string")
    out["hour"] = out["hour"].astype("int8")
    out["municipality"] = pd.Categorical(
        out["cell_id"].map(municipality), categories=municipality.cat.categories
    )
    return out[[r["column"] for r in CELL_HOUR_DICTIONARY]]


def build_fleet_hour(parts: list[pd.DataFrame]) -> pd.DataFrame:
    hours = pd.date_range(
        config.STUDY_START,
        config.STUDY_END,
        freq="h",
        inclusive="left",
        tz=config.TZ_LOCAL,
    )
    all_parts = pd.concat(parts)
    outside = all_parts.index.difference(hours)
    if len(outside):
        raise PipelineError(
            f"Points in hours outside the study window: {outside[:3].tolist()}"
        )
    g = all_parts.groupby(level=0)
    sums = g.sum().reindex(hours, fill_value=0)
    active = g.size().reindex(hours, fill_value=0)
    out = pd.DataFrame(
        {
            "hour_start_local": hours,
            "n_vehicles_active": active.to_numpy().astype("int16"),
            "n_vehicles_moving": sums["moving"].to_numpy().astype("int16"),
            "n_points": sums["n_points"].to_numpy().astype("int32"),
            "avg_speed_moving_kmh": speed(sums["d_speed"], sums["t_speed"]).to_numpy(),
            "di2_share": share(sums["di2_on"], sums["di2_known"]).to_numpy(),
        }
    )
    return out


def variation_row(
    level: str, rows: int, n: float, total: float, squares: float
) -> dict:
    mean = total / n
    std = np.sqrt(max(squares / n - mean**2, 0) * n / (n - 1))
    return {
        "level": level,
        "rows": rows,
        "n_with_speed": int(n),
        "mean_speed_kmh": mean,
        "std_speed_kmh": std,
        "cv": std / mean,
    }


def table_row(level: str, table: pd.DataFrame) -> dict:
    values = table["avg_speed_moving_kmh"].dropna().astype("float64")
    return variation_row(
        level, len(table), len(values), values.sum(), (values**2).sum()
    )


def run(dev: bool) -> None:
    ids = io.vehicle_ids("clean")
    if not ids:
        raise PipelineError("No clean points: run the clean stage first.")

    days, cells, hours = [], [], []
    muni_counts = []
    point_n, point_sum, point_sq, step_km = 0, 0.0, 0.0, 0.0
    for v in ids:
        df = io.read_points("clean", [v])
        s = steps(df)
        days.append(vehicle_day(df, s))
        cells.append(cell_hour_part(df, s))
        hours.append(fleet_hour_part(df, s))
        muni_counts.append(
            df.groupby(["cell_id", "municipality"], observed=True).size()
        )
        moving_speed = df.loc[df["speed_kmh"] >= config.V_MOVE_KMH, "speed_kmh"]
        moving_speed = moving_speed.astype("float64")
        point_n += len(moving_speed)
        point_sum += moving_speed.sum()
        point_sq += (moving_speed**2).sum()
        step_km += s["d"].sum() / 1000

    vd = pd.concat(days, ignore_index=True)
    municipality = cell_municipality(
        pd.concat(muni_counts).groupby(level=[0, 1], observed=True).sum()
    )
    ch = build_cell_hour(cells, municipality)
    fh = build_fleet_hour(hours)

    if not np.isclose(vd["km"].astype("float64").sum(), step_km, rtol=1e-5):
        raise PipelineError(
            f"vehicle_day km {vd['km'].sum():,.1f} != points_clean km {step_km:,.1f}"
        )
    expected_hours = (config.STUDY_END - config.STUDY_START).days * 24
    if len(fh) != expected_hours:
        raise PipelineError(f"fleet_hour has {len(fh)} rows, expected {expected_hours}")
    if not np.isclose(
        ch["observed_min"].astype("float64").sum() / 60,
        vd["observed_h"].astype("float64").sum(),
        rtol=1e-5,
    ):
        raise PipelineError("cell_hour observed time does not match vehicle_day")

    for name, table, rows in [
        ("vehicle_day", vd, VEHICLE_DAY_DICTIONARY),
        ("cell_hour", ch, CELL_HOUR_DICTIONARY),
        ("fleet_hour", fh, FLEET_HOUR_DICTIONARY),
    ]:
        io.write_table(table, name)
        io.write_dictionary(f"processed/{name}", rows, table, owner="aggregate")

    summary = pd.DataFrame(
        [
            variation_row(
                "point (moving points)", point_n, point_n, point_sum, point_sq
            ),
            table_row("cell_hour", ch),
            table_row("vehicle_day", vd),
        ]
    )
    io.write_report_table(summary.round(4), "aggregation_summary")
    print(
        f"vehicle_day {len(vd):,} rows, {step_km:,.1f} km; cell_hour {len(ch):,} rows; "
        f"fleet_hour {len(fh):,} rows"
    )
    figure_hourly_activity(fh)


def figure_hourly_activity(fh: pd.DataFrame) -> None:
    hour = fh["hour_start_local"].dt.hour
    profile = fh.groupby(hour)[["n_vehicles_active", "n_vehicles_moving"]].mean()

    fig, ax = figures.subplots(7.5, 4)
    ax.plot(
        profile.index + 0.5,
        profile["n_vehicles_active"],
        color=figures.SERIES[0],
        marker="o",
        markersize=3,
        label="active (at least one point)",
    )
    ax.plot(
        profile.index + 0.5,
        profile["n_vehicles_moving"],
        color=figures.SERIES[1],
        marker="o",
        markersize=3,
        label="moving (at least one moving step)",
    )
    top = profile.to_numpy().max() * 1.15
    for edge in config.TOD_BUCKETS[1:-1]:
        ax.axvline(edge, color=figures.INK_MUTED, linewidth=0.8, linestyle="--")
    for lo, hi, label in zip(
        config.TOD_BUCKETS[:-1], config.TOD_BUCKETS[1:], config.TOD_LABELS
    ):
        ax.text(
            (lo + hi) / 2,
            top * 0.97,
            label,
            ha="center",
            va="top",
            fontsize=8,
            color=figures.INK_SECONDARY,
        )
    ax.set_xlim(0, 24)
    ax.set_ylim(0, top)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("Local hour of day (h)")
    ax.set_ylabel("Vehicles per hour, mean over days")
    ax.set_title("F7. Hourly activity of the fleet, with the time-of-day bucket edges")
    ax.legend(loc="lower right")
    figures.save_figure(fig, "f07_hourly_activity")

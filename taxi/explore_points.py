"""Point-level exploration: summary statistics of the final points.

The final points are built from the screened points, so the errors the outlier rules
remove are gone. Every statistic is given twice, weighted by points and weighted by time
(dt_s of the step that starts at the point, steps without a gap), because a parked taxi
reports far less often than a moving one: point counts are not time.

Fleet-wide over all vehicles with DuckDB, without loading the points into pandas: each
variable is grouped into its values rounded to a fixed resolution, with the number of
points and the time at each value, and stats.summarize runs on that compressed table.
n, n_missing, min, and max are exact; the other statistics are exact at the resolution.

Outputs:
- T16 summary_points.csv: the numeric variables, overall and by group;
- summary_points_categorical.csv: frequencies and mode of the categorical variables;
- summary_points_heading.csv: circular summary of the heading, never an arithmetic mean;
- outliers_kept_points.csv: the unusual but possible points that stay (speed at or above
  SPEED_FAST_KMH, points outside Kosovo): how many, when, and in which municipality;
- F15: the speed distribution by points against by time.
"""

import duckdb
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from taxi import PipelineError, config, figures, io, stats

# Time weight of a point: the duration of the step that starts at it, 0 across a gap.
TIME_WEIGHT = "CASE WHEN NOT gap_after AND dt_s IS NOT NULL THEN dt_s ELSE 0 END"
KNOWN_DIST = "CASE WHEN NOT gap_after AND dist_m IS NOT NULL THEN dist_m ELSE 0 END"
# Variable: (unit, resolution of the grouped values).
NUMERIC = {
    "speed_kmh": ("km/h", 0.1),
    "implied_speed_kmh": ("km/h", 0.1),
    "accel_ms2": ("m/s²", 0.01),
    "dt_s": ("s", 1.0),
    "dist_m": ("m", 0.1),
    "dist_center_m": ("m", 1.0),
}
GROUPS = ["tod_bucket", "is_weekend", "di1", "di2", "in_kosovo"]
# di3 is left out: it is stuck true in 25 vehicles, so its fleet frequencies mislead.
CATEGORICAL = [
    "speed_bin",
    "tod_bucket",
    "compass8",
    "municipality",
    "di1",
    "di2",
    "is_moving",
    "in_kosovo",
]
ORDER = {
    "speed_bin": config.SPEED_BIN_LABELS,
    "tod_bucket": config.TOD_LABELS,
    "compass8": config.COMPASS_LABELS,
}
TOP_MUNICIPALITIES = 10
F15_BIN_KMH = 5
F15_TOP_KMH = 150


def _connect() -> tuple[duckdb.DuckDBPyConnection, int]:
    """A DuckDB view p over the final points, with the time weight w; and the row count."""
    if not io.vehicle_ids("final"):
        raise PipelineError("No final points: run the features and trips stages first.")
    files = str(config.points_dir("final") / "*.parquet")
    con = duckdb.connect()
    con.execute(
        f"CREATE VIEW p AS SELECT *, {TIME_WEIGHT} AS w FROM read_parquet('{files}')"
    )
    n = con.execute("SELECT count(*) FROM p").fetchone()[0]
    io.count_rows(rows_in=n)
    return con, n


def _group_expr(group: str) -> str:
    return (
        "'all'" if group == "all" else f"COALESCE(CAST({group} AS VARCHAR), 'missing')"
    )


def _decimals(resolution: float) -> int:
    return max(0, -int(np.floor(np.log10(resolution))))


def binned(
    con: duckdb.DuckDBPyConnection, var: str, group: str, where: str = "TRUE"
) -> pd.DataFrame:
    """Points (n) and time (t, s) at each value of var rounded to its resolution, per group."""
    res = NUMERIC[var][1]
    df = con.execute(
        f"""SELECT {_group_expr(group)} AS gv, round({var} / {res}) * {res} AS v,
                   count(*) AS n, sum(w) AS t
            FROM p WHERE {var} IS NOT NULL AND {where} GROUP BY ALL"""
    ).df()
    df["v"] = df["v"].round(_decimals(res))
    return df


def numeric_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """T16: every numeric variable, overall and per group, by points and by time."""
    rows = []
    for var, (unit, res) in NUMERIC.items():
        for group in ["all", *GROUPS]:
            values = binned(con, var, group)
            exact = (
                con.execute(
                    f"""SELECT {_group_expr(group)} AS gv,
                           count(*) - count({var}) AS missing_p,
                           count(*) FILTER (w > 0 AND {var} IS NULL) AS missing_t,
                           count({var}) FILTER (w > 0) AS n_t,
                           min({var}) AS min_p, max({var}) AS max_p,
                           min({var}) FILTER (w > 0) AS min_t,
                           max({var}) FILTER (w > 0) AS max_t
                    FROM p GROUP BY ALL"""
                )
                .df()
                .set_index("gv")
            )
            for gv, part in values.groupby("gv", sort=True):
                e = exact.loc[gv]
                for weighting, col in (("points", "n"), ("time", "t")):
                    s = stats.summarize(part["v"], part[col])
                    if weighting == "points":
                        s.update(
                            n=int(part["n"].sum()),
                            n_missing=int(e["missing_p"]),
                            min=float(e["min_p"]),
                            max=float(e["max_p"]),
                        )
                    else:
                        s.update(
                            n=int(e["n_t"]),
                            n_missing=int(e["missing_t"]),
                            min=float(e["min_t"]) if e["n_t"] else np.nan,
                            max=float(e["max_t"]) if e["n_t"] else np.nan,
                        )
                    rows.append(
                        {
                            "table": "points",
                            "variable": var,
                            "unit": unit,
                            "weighting": weighting,
                            "resolution": res,
                            "group": group,
                            "group_value": gv,
                            **s,
                        }
                    )
    return pd.DataFrame(rows)


def categorical_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Frequencies of each categorical variable, as shares of points and of time, with the mode."""
    parts = []
    for var in CATEGORICAL:
        df = con.execute(
            f"""SELECT COALESCE(CAST({var} AS VARCHAR), 'missing') AS value,
                       count(*) AS points, sum(w) AS time_s
                FROM p GROUP BY ALL"""
        ).df()
        if var == "municipality":
            named = df[df["value"] != "missing"].sort_values("time_s", ascending=False)
            top, rest = named.head(TOP_MUNICIPALITIES), named.iloc[TOP_MUNICIPALITIES:]
            other = pd.DataFrame(
                [
                    {
                        "value": f"other ({len(rest)} municipalities)",
                        "points": rest["points"].sum(),
                        "time_s": rest["time_s"].sum(),
                    }
                ]
            )
            df = pd.concat(
                [
                    top,
                    other,
                    df[df["value"] == "missing"].assign(value="outside Kosovo"),
                ],
                ignore_index=True,
            )
        elif var in ORDER:
            rank = {label: i for i, label in enumerate(ORDER[var])}
            df = df.sort_values("value", key=lambda s: s.map(rank).fillna(len(rank)))
        else:
            df = df.sort_values("points", ascending=False)
        df["point_share"] = df["points"] / df["points"].sum()
        df["time_share"] = df["time_s"] / df["time_s"].sum()
        df["is_point_mode"] = df["points"] == df["points"].max()
        df["is_time_mode"] = df["time_s"] == df["time_s"].max()
        parts.append(df.assign(variable=var))
    out = pd.concat(parts, ignore_index=True)
    return out[
        [
            "variable",
            "value",
            "points",
            "point_share",
            "is_point_mode",
            "time_s",
            "time_share",
            "is_time_mode",
        ]
    ]


def heading_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Circular mean direction and spread of the heading, for all points and moving points."""
    rows = []
    for subset, where in (("all points", "TRUE"), ("moving points", "is_moving")):
        df = con.execute(
            f"""SELECT CAST(round(heading_deg) AS INTEGER) % 360 AS v,
                       count(*) AS n, sum(w) AS t
                FROM p WHERE heading_deg IS NOT NULL AND {where} GROUP BY ALL"""
        ).df()
        for weighting, col in (("points", "n"), ("time", "t")):
            circ = stats.circular_summary(df["v"], df[col])
            circ["n"] = int(df.loc[df[col] > 0, "n"].sum())
            rows.append(
                {"subset": subset, "weighting": weighting, "resolution_deg": 1, **circ}
            )
    return pd.DataFrame(rows)


def kept_outliers(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """The unusual but possible points that stay: how many, when, and where (areas only)."""
    total_n, total_t, total_d = con.execute(
        f"SELECT count(*), sum(w), sum({KNOWN_DIST}) FROM p"
    ).fetchone()
    kinds = {
        f"speed {config.SPEED_FAST_KMH} km/h or more": f"speed_kmh >= {config.SPEED_FAST_KMH}",
        "outside Kosovo": "NOT in_kosovo",
    }
    rows = []
    for kind, where in kinds.items():
        r = con.execute(
            f"""SELECT count(*) AS points, COALESCE(sum(w), 0) / 3600 AS time_h,
                       count(DISTINCT vehicle_id) AS vehicles,
                       count(DISTINCT CAST(vehicle_id AS VARCHAR) || '-' || CAST(date_local AS VARCHAR))
                           AS vehicle_days,
                       count(DISTINCT trip_id) AS trips,
                       COALESCE(sum({KNOWN_DIST}), 0) / 1000 AS km
                FROM p WHERE {where}"""
        ).df()
        r = r.assign(
            kind=kind,
            dimension="all",
            value="all",
            share_of_points=r["points"] / total_n,
            share_of_time=r["time_h"] * 3600 / total_t,
            share_of_km=r["km"] * 1000 / total_d,
        )
        rows.append(r)
        by_hour = con.execute(
            f"""SELECT CAST(hour AS VARCHAR) AS value, count(*) AS points,
                       sum(w) / 3600 AS time_h, count(DISTINCT vehicle_id) AS vehicles
                FROM p WHERE {where} GROUP BY hour ORDER BY hour"""
        ).df()
        rows.append(by_hour.assign(kind=kind, dimension="local hour"))
        if kind.startswith("speed"):
            by_muni = con.execute(
                f"""SELECT COALESCE(CAST(municipality AS VARCHAR), 'outside Kosovo') AS value,
                           count(*) AS points, sum(w) / 3600 AS time_h,
                           count(DISTINCT vehicle_id) AS vehicles
                    FROM p WHERE {where} GROUP BY ALL ORDER BY time_h DESC
                    LIMIT {TOP_MUNICIPALITIES}"""
            ).df()
            rows.append(by_muni.assign(kind=kind, dimension="municipality"))
    out = pd.concat(rows, ignore_index=True)
    columns = [
        "kind",
        "dimension",
        "value",
        "points",
        "time_h",
        "vehicles",
        "vehicle_days",
        "trips",
        "km",
        "share_of_points",
        "share_of_time",
        "share_of_km",
    ]
    return out[columns]


def figure_speed(con: duckdb.DuckDBPyConnection) -> tuple[float, float]:
    """F15: share of points and share of time per speed bin; returns the stopped shares."""
    df = binned(con, "speed_kmh", "all")
    last = F15_TOP_KMH // F15_BIN_KMH
    df["bin"] = np.minimum(np.floor(df["v"] / F15_BIN_KMH), last).astype(int)
    per_bin = df.groupby("bin")[["n", "t"]].sum().reindex(range(last + 1), fill_value=0)
    points = per_bin["n"] / per_bin["n"].sum()
    time = per_bin["t"] / per_bin["t"].sum()
    stopped = df["v"] < config.V_MOVE_KMH
    stopped_p = df.loc[stopped, "n"].sum() / df["n"].sum()
    stopped_t = df.loc[stopped, "t"].sum() / df["t"].sum()

    fig, ax = figures.subplots(9, 4.8)
    x = per_bin.index.to_numpy() * F15_BIN_KMH
    width = F15_BIN_KMH * 0.4
    ax.bar(
        x + F15_BIN_KMH / 2 - width / 2,
        points * 100,
        width,
        color=figures.SERIES[0],
        label="share of points",
    )
    ax.bar(
        x + F15_BIN_KMH / 2 + width / 2,
        time * 100,
        width,
        color=figures.SERIES[1],
        label="share of observed time",
    )
    ax.set_xlim(0, F15_TOP_KMH + F15_BIN_KMH)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.set_xlabel(
        f"Reported speed (km/h, {F15_BIN_KMH} km/h bins; the last bin is {F15_TOP_KMH} or more)"
    )
    ax.set_ylabel("Share (%)")
    ax.set_title(
        f"F15. Reported speed: share of points vs share of time ({int(df['n'].sum()):,} points)"
    )
    ax.text(
        0.98,
        0.95,
        f"Stopped (under {config.V_MOVE_KMH} km/h): {stopped_p:.1%} of points, "
        f"{stopped_t:.1%} of observed time",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=figures.INK_SECONDARY,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.88))
    figures.save_figure(fig, "f15_speed_point_vs_time_weighted")
    return stopped_p, stopped_t


def run(dev: bool) -> None:
    con, n = _connect()
    tables = {
        "summary_points": numeric_summary(con),
        "summary_points_categorical": categorical_summary(con),
        "summary_points_heading": heading_summary(con),
        "outliers_kept_points": kept_outliers(con),
    }
    for name, table in tables.items():
        io.write_report_table(table, name)
        io.count_rows(rows_out=len(table))
    stopped_p, stopped_t = figure_speed(con)

    kept = tables["outliers_kept_points"].query("dimension == 'all'").set_index("kind")
    print(
        f"{n:,} final points; stopped: {stopped_p:.1%} of points, {stopped_t:.1%} of time"
    )
    for kind, r in kept.iterrows():
        print(
            f"{kind}: {int(r['points']):,} points ({r['share_of_points']:.3%}), "
            f"{r['time_h']:,.1f} h, {int(r['vehicles'])} vehicles, {int(r['trips']):,} trips"
        )

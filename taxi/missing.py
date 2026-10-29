"""Missing values: identification, the speed estimates, and the imputation test.

missing_strategy.csv: what is missing, how much, the likely mechanism, and how it is
handled (rules M1-M6). missing_gaps.csv: the hour and the ignition state (di1) at which
gaps start, which shows they are not missing completely at random.

imputation_eval.csv: 5% of the known speeds that M2 could fill are hidden, filled by each
method, and compared on the mean absolute error.
"""

import numpy as np
import pandas as pd

from taxi import config, io

IMPUTE_METHODS = ("implied_speed", "interpolate", "previous")
HIDE_SHARE = 0.05
STUDY_DAYS = (config.STUDY_END - config.STUDY_START).days
COLUMN_RULES = {
    "ts_utc": ("M1", "drop (R01)", "a time cannot be guessed reliably"),
    "lon": ("M1", "drop (R03)", "a position cannot be guessed reliably"),
    "lat": ("M1", "drop (R03)", "a position cannot be guessed reliably"),
    "speed_kmh": (
        "M2",
        "fill with IMPUTE_METHOD when both steps are short, else leave NA",
        "between two short steps the speed changes little",
    ),
    "heading_deg": ("-", "leave NA", "not needed for durations or distances"),
    "di1": ("M3", "leave NA", "an on/off state cannot be inferred from neighbours"),
    "di2": ("M3", "leave NA", "an on/off state cannot be inferred from neighbours"),
    "di3": ("M3", "leave NA", "an on/off state cannot be inferred from neighbours"),
}


def speed_estimate(df: pd.DataFrame, method: str) -> pd.Series:
    if method == "implied_speed":
        return df["implied_speed_kmh"].astype("float32")

    speed = df["speed_kmh"].astype("float64")
    prev_speed = speed.shift(1)
    if method == "previous":
        return prev_speed.astype("float32")
    if method == "interpolate":
        next_speed = speed.shift(-1)
        dt_before = df["dt_s"].shift(1).astype("float64")
        dt_after = df["dt_s"].astype("float64")
        span = dt_before + dt_after
        weight = (dt_before / span.where(span > 0)).fillna(0.5)
        return (prev_speed + (next_speed - prev_speed) * weight).astype("float32")
    raise ValueError(f"IMPUTE_METHOD must be one of {IMPUTE_METHODS}, got {method!r}")


def impute_eligible(df: pd.DataFrame, max_dt_s: float) -> np.ndarray:
    dt_after = df["dt_s"].astype("float64")
    dt_before = dt_after.shift(1)
    gap_before = df["gap_after"].shift(1, fill_value=True)
    return (
        (dt_before <= max_dt_s)
        & (dt_after <= max_dt_s)
        & ~gap_before
        & ~df["gap_after"]
    ).to_numpy()


def value_rows(n_missing: pd.Series, n_rows: int) -> list[dict]:
    rows = []
    for col, (rule, strategy, why) in COLUMN_RULES.items():
        count = int(n_missing[col])
        rows.append(
            {
                "kind": "value",
                "column": col,
                "n": n_rows,
                "count": count,
                "mechanism": "parse failure or empty field"
                if count
                else "none in the data",
                "rule": rule,
                "strategy": strategy,
                "why": why,
            }
        )
    return rows


def file_row() -> dict:
    inventory = pd.read_csv(config.RAW_INVENTORY_CSV)
    empty = inventory.loc[inventory["bytes"] == 0, "vehicle_id"].tolist()
    return {
        "kind": "file",
        "column": "all",
        "n": len(inventory),
        "count": len(empty),
        "mechanism": f"raw files of vehicles {empty} are empty (0 bytes); the source gives no reason",
        "rule": "M5",
        "strategy": "exclude, list in the inventory and the report",
        "why": "there is nothing to repair",
    }


def day_row(days: int, vehicles: int) -> dict:
    n = STUDY_DAYS * vehicles
    return {
        "kind": "vehicle-day",
        "column": "date_local",
        "n": n,
        "count": n - days,
        "mechanism": "taxi not in service that day (argued from the devices, not provable)",
        "rule": "M6",
        "strategy": "no row in vehicle_day; coverage shown in F2",
        "why": "a day without points has no measurements to aggregate",
    }


def gap_counts(df: pd.DataFrame) -> pd.DataFrame:
    steps = df[df["dt_s"].notna()]
    di1 = steps["di1"].astype("string").fillna("NA")
    parts = {
        "hour": steps.groupby("hour")["gap_after"].agg(["size", "sum"]),
        "di1 before the gap": steps.groupby(di1)["gap_after"].agg(["size", "sum"]),
    }
    table = pd.concat(parts, names=["group", "value"])
    return table.rename(columns={"size": "steps", "sum": "gaps"})


def gap_row(counts: pd.DataFrame, minutes: pd.Series) -> dict:
    by_di1 = counts.loc["di1 before the gap"]
    rate = by_di1["gaps"] / by_di1["steps"] * 10_000
    hours = counts.loc["hour"]
    return {
        "kind": "gap",
        "column": "dt_s",
        "n": int(hours["steps"].sum()),
        "count": int(hours["gaps"].sum()),
        "mechanism": (
            f"gaps per 10,000 steps: {rate.get('False', 0):.2f} after the ignition is off, "
            f"{rate.get('True', 0):.2f} after it is on, so not MCAR; length median "
            f"{minutes.median():.0f} min, p90 {minutes.quantile(0.9):.0f} min "
            "(missing_gaps.csv)"
        ),
        "rule": "M4",
        "strategy": "set gap_after, leave the step out of durations and distances",
        "why": "positions between two distant points are unknown; no points are created",
    }


def hidden_speeds(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    known = (
        df["speed_kmh"].notna()
        & ~df["speed_imputed"]
        & impute_eligible(df, config.IMPUTE_MAX_DT_S)
    )
    candidates = np.flatnonzero(known.to_numpy())
    hide = rng.choice(
        candidates, size=round(HIDE_SHARE * len(candidates)), replace=False
    )
    masked = df.copy()
    masked.loc[masked.index[hide], "speed_kmh"] = np.nan
    out = pd.DataFrame({"truth": df["speed_kmh"].to_numpy()[hide]})
    for method in IMPUTE_METHODS:
        out[method] = speed_estimate(masked, method).to_numpy()[hide]
    return out


def evaluate(hidden: pd.DataFrame) -> pd.DataFrame:
    compared = hidden.dropna()
    rows = []
    for method in IMPUTE_METHODS:
        error = (compared[method] - compared["truth"]).abs()
        rows.append(
            {
                "method": method,
                "n_hidden": len(hidden),
                "n_with_estimate": int(hidden[method].notna().sum()),
                "n_compared": len(compared),
                "mae_kmh": error.mean(),
                "median_ae_kmh": error.median(),
                "p90_ae_kmh": error.quantile(0.9),
            }
        )
    table = pd.DataFrame(rows)
    table["chosen"] = table["mae_kmh"] == table["mae_kmh"].min()
    return table.round(3)


def vehicle_summary(
    df: pd.DataFrame, raw_missing: pd.Series, raw_rows: int, rng: np.random.Generator
) -> dict:
    return {
        "raw_missing": raw_missing,
        "raw_rows": raw_rows,
        "days": df["date_local"].nunique(),
        "gap_counts": gap_counts(df),
        "gap_minutes": df.loc[df["gap_after"], "dt_s"] / 60,
        "hidden": hidden_speeds(df, rng),
    }


def write_tables(summaries: list[dict]) -> None:
    n_missing = sum(s["raw_missing"] for s in summaries)
    n_rows = sum(s["raw_rows"] for s in summaries)
    days = sum(s["days"] for s in summaries)
    counts = pd.concat([s["gap_counts"] for s in summaries]).groupby(level=[0, 1]).sum()
    minutes = pd.concat([s["gap_minutes"] for s in summaries])

    strategy = pd.DataFrame(
        value_rows(n_missing, n_rows)
        + [file_row(), gap_row(counts, minutes), day_row(days, len(summaries))]
    )
    strategy.insert(4, "share", (strategy["count"] / strategy["n"]).round(6))
    io.write_report_table(strategy, "missing_strategy")

    gaps = counts.reset_index()
    gaps["gaps_per_10000_steps"] = (gaps["gaps"] / gaps["steps"] * 10_000).round(3)
    io.write_report_table(gaps, "missing_gaps")

    evaluation = evaluate(
        pd.concat([s["hidden"] for s in summaries], ignore_index=True)
    )
    io.write_report_table(evaluation, "imputation_eval")
    best = evaluation.loc[evaluation["chosen"], "method"].iloc[0]
    print(
        f"Imputation test: lowest MAE with {best}; IMPUTE_METHOD is {config.IMPUTE_METHOD}"
    )

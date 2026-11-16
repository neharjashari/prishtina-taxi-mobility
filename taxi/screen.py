"""Heqja e gabimeve: points_clean ne points_screened.

Per cdo automjet zbatohen rregullat O01-O04 qe i gjen faza detect (point_flags.parquet),
ne rendin qe ato i supozojne: O01 heq pikat, pastaj hapat rillogariten me clean.add_steps;
O02 dhe O04 vendosin NA ne dist_m dhe implied_speed_kmh te hapit (dt_s mbetet); O03 vendos
NA ne speed_kmh dhe M2 e mbush perseri si ne fazen e pare. Cdo rregull shenohet ne
cleaning_log.csv, edhe kur nuk prek asnje rresht.
"""

import numpy as np
import pandas as pd

from taxi import PipelineError, clean, config, io, missing, outliers
from taxi.ingest import INTERIM_DICTIONARY

STAGE = "screen"
SCREENED_DATASET = "processed/points_screened"
STEP_COLUMNS = ["dist_m", "implied_speed_kmh"]
NOTES = {
    "O01": "spike: the position jumps away and back",
    "O02": f"step faster than JUMP_KMH ({config.JUMP_KMH} km/h): dist_m and "
    "implied_speed_kmh",
    "O03": f"reported speed its neighbours cannot explain (ACCEL_MAX_MS2 "
    f"{config.ACCEL_MAX_MS2} m/s²): speed_kmh",
    "O04": f"parked drift of DRIFT_MIN_M ({config.DRIFT_MIN_M} m) or more: dist_m and "
    "implied_speed_kmh",
}

SCREENED_DICTIONARY = [
    {
        "column": "outlier_rule",
        "attribute_type": "nominal",
        "values": "O02, O03, O04, or several joined with a comma; NA",
        "description": "The outlier rules that changed this row. O02 and O04 set dist_m and "
        "implied_speed_kmh of the step that starts here to NA (dt_s is kept); O03 sets "
        "speed_kmh to NA, and M2 may fill it again (speed_imputed). Points of O01 are dropped.",
        "source": "outliers/point_flags",
        "owner": STAGE,
    },
]


def run(dev: bool) -> None:
    ids = io.vehicle_ids("clean")
    if not ids:
        raise PipelineError("No clean points: run the clean stage first.")
    for stale in sorted(set(io.vehicle_ids("screened")) - set(ids)):
        io.points_path("screened", stale).unlink()

    flags = io.read_table(outliers.FLAGS_TABLE)
    for v in ids:
        df = screen_vehicle(v, flags[flags["vehicle_id"] == v])
        io.write_points(df, "screened")

    rows = (
        [{**r, "owner": "ingest"} for r in INTERIM_DICTIONARY]
        + [{**r, "owner": "clean"} for r in clean.CLEAN_DICTIONARY]
        + SCREENED_DICTIONARY
    )
    io.write_dictionary(SCREENED_DATASET, rows, df, owner=STAGE)


def screen_vehicle(vehicle_id: int, flags: pd.DataFrame) -> pd.DataFrame:
    df = io.read_points("clean", [vehicle_id])
    df = df.sort_values(["ts_utc", "source_row"], kind="stable").reset_index(drop=True)
    rows_in = len(df)
    dropped = 0

    def flagged(rule_id: str) -> pd.Series:
        return df["source_row"].isin(
            flags.loc[flags["rule_id"] == rule_id, "source_row"]
        )

    def apply(rule_id: str, columns: list[str]) -> None:
        nonlocal df, dropped
        mask = flagged(rule_id)
        n = int(mask.sum())
        action = config.OUTLIER_ACTIONS[rule_id]
        io.log_step(STAGE, rule_id, action, vehicle_id, len(df), n, NOTES[rule_id])
        if action == "drop" and n > config.MAX_DROP_SHARE * rows_in:
            raise PipelineError(
                f"Vehicle {vehicle_id}: {rule_id} drops {n:,} of {rows_in:,} rows "
                f"({n / rows_in:.2%}), above MAX_DROP_SHARE ({config.MAX_DROP_SHARE:.0%}). "
                "Look at the rows before changing anything."
            )
        if action == "drop":
            df = df[~mask].reset_index(drop=True)
            dropped += n
        if action == "set_na":
            df.loc[mask, columns] = np.nan

    apply("O01", [])
    clean.add_steps(df)
    apply("O02", STEP_COLUMNS)
    apply("O04", STEP_COLUMNS)
    apply("O03", ["speed_kmh"])

    missing_speed = df["speed_kmh"].isna()
    estimate = missing.speed_estimate(df, config.IMPUTE_METHOD)
    filled = (
        missing_speed
        & missing.impute_eligible(df, config.IMPUTE_MAX_DT_S)
        & estimate.notna()
    )
    df["speed_kmh"] = df["speed_kmh"].mask(filled, estimate)
    df["speed_imputed"] = df["speed_imputed"] | filled
    io.log_step(
        STAGE,
        "M2",
        "impute",
        vehicle_id,
        len(df),
        int(filled.sum()),
        f"{int(missing_speed.sum()):,} speeds missing after O03, "
        f"{int(filled.sum()):,} filled with {config.IMPUTE_METHOD}",
    )

    changed = flags[
        flags["rule_id"].map(config.OUTLIER_ACTIONS) == "set_na"
    ].sort_values("rule_id")
    labels = changed.groupby("source_row")["rule_id"].agg(",".join)
    df["outlier_rule"] = df["source_row"].map(labels).astype("string")

    if len(df) + dropped != rows_in:
        raise PipelineError(
            f"Vehicle {vehicle_id}: {len(df):,} kept + {dropped:,} dropped != {rows_in:,} rows in"
        )
    return df

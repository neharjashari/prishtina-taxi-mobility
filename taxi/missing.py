"""Vlerat qe mungojne: menyrat per te vleresuar shpejtesine"""

import numpy as np
import pandas as pd

IMPUTE_METHODS = ("implied_speed", "interpolate", "previous")


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

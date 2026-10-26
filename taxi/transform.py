"""Discretization, binarization, and variable transformation."""

import numpy as np
import pandas as pd

from taxi import config

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
    "flag_jump": (True, "quality flag, for Phase II"),
}

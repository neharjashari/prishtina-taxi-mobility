"""Statistics shared by the outlier and exploration stages.

Robust scores, fences, intervals, multiple-testing control, weighted quantiles, circular
summaries, and one summary layout for every summary table (points, aggregates, trips).
Quantiles use the inverted-CDF definition, weighted or not, so a weighted summary with
equal weights gives the same numbers as an unweighted one.
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats as sps

# Columns of every summary table, after table, variable, group, group_value.
SUMMARY_COLUMNS = [
    "n",
    "n_missing",
    "weight",
    "mean",
    "trimmed_mean_10",
    "median",
    "sd",
    "mad",
    "iqr",
    "min",
    "p01",
    "p05",
    "p25",
    "p75",
    "p95",
    "p99",
    "max",
    "skewness",
]
_PERCENTILES = {
    "p01": 0.01,
    "p05": 0.05,
    "p25": 0.25,
    "p75": 0.75,
    "p95": 0.95,
    "p99": 0.99,
}
# Scales the MAD to the standard deviation of a normal distribution.
_MAD_NORMAL = 0.6745


def _as_float(values: ArrayLike) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def robust_z(x: ArrayLike) -> np.ndarray:
    """Modified z-score, 0.6745 * (x - median) / MAD. NA everywhere when the MAD is 0."""
    x = _as_float(x)
    median = np.nanmedian(x) if np.isfinite(x).any() else np.nan
    mad = np.nanmedian(np.abs(x - median)) if np.isfinite(x).any() else np.nan
    if not np.isfinite(mad) or mad == 0:
        return np.full(x.shape, np.nan)
    return _MAD_NORMAL * (x - median) / mad


def tukey_fences(x: ArrayLike, k: float = 1.5) -> tuple[float, float]:
    """Lower and upper fence: Q1 - k * IQR and Q3 + k * IQR."""
    x = _as_float(x)
    x = x[np.isfinite(x)]
    q1, q3 = np.quantile(x, [0.25, 0.75], method="inverted_cdf")
    return float(q1 - k * (q3 - q1)), float(q3 + k * (q3 - q1))


def wilson_ci(successes: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval of a proportion; NA when n is 0."""
    if n == 0:
        return np.nan, np.nan
    ci = sps.binomtest(int(successes), int(n)).proportion_ci(
        confidence_level=level, method="wilson"
    )
    return float(ci.low), float(ci.high)


def bh_adjust(p: ArrayLike) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values). NA stays NA."""
    p = _as_float(p)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q[ok] = sps.false_discovery_control(p[ok], method="bh")
    return q


def _weighted_arrays(
    x: ArrayLike, w: ArrayLike | None
) -> tuple[np.ndarray, np.ndarray]:
    """x and w without missing values or non-positive weights."""
    x = _as_float(x)
    w = np.ones_like(x) if w is None else _as_float(w)
    if w.shape != x.shape:
        raise ValueError(f"x and w differ in shape: {x.shape} and {w.shape}")
    keep = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return x[keep], w[keep]


def weighted_quantile(x: ArrayLike, w: ArrayLike, q: ArrayLike) -> np.ndarray:
    """Smallest x whose cumulative weight reaches q of the total (inverted CDF)."""
    x, w = _weighted_arrays(x, w)
    q = np.atleast_1d(_as_float(q))
    if len(x) == 0:
        return np.full(q.shape, np.nan)
    order = np.argsort(x, kind="stable")
    x, cum = x[order], np.cumsum(w[order])
    idx = np.searchsorted(cum, q * cum[-1], side="left")
    return x[np.clip(idx, 0, len(x) - 1)]


def circular_summary(deg: ArrayLike, w: ArrayLike | None = None) -> dict:
    """Mean direction (0-360), mean resultant length R (0-1), and circular sd in degrees."""
    deg, w = _weighted_arrays(deg, w)
    if len(deg) == 0:
        return {
            "n": 0,
            "mean_deg": np.nan,
            "resultant_length": np.nan,
            "circular_sd_deg": np.nan,
        }
    rad = np.radians(deg)
    s, c = np.sum(w * np.sin(rad)), np.sum(w * np.cos(rad))
    r = float(np.hypot(s, c) / np.sum(w))
    return {
        "n": len(deg),
        "mean_deg": float(np.degrees(np.arctan2(s, c)) % 360) if r > 0 else np.nan,
        "resultant_length": r,
        "circular_sd_deg": float(np.degrees(np.sqrt(-2 * np.log(r))))
        if r > 0
        else np.inf,
    }


def summarize(x: ArrayLike, w: ArrayLike | None = None) -> dict:
    """Location, spread, percentiles, and skewness, with the keys of SUMMARY_COLUMNS.

    With weights (for example dt_s, for time-weighted statistics), every statistic is
    weighted: the mean, the quantiles, the standard deviation and skewness (population
    versions), the MAD around the weighted median, and the trimmed mean over the values
    between the weighted 10th and 90th percentiles.
    """
    raw = _as_float(x)
    n_missing = int(np.sum(~np.isfinite(raw)))
    x, w = _weighted_arrays(raw, w)
    out = dict.fromkeys(SUMMARY_COLUMNS, np.nan)
    out.update(n=len(x), n_missing=n_missing, weight=float(np.sum(w)))
    if len(x) == 0:
        return out

    qs = weighted_quantile(x, w, [0.10, 0.25, 0.5, 0.75, 0.90])
    p10, q1, median, q3, p90 = (float(v) for v in qs)
    mean = float(np.average(x, weights=w))
    var = float(np.average((x - mean) ** 2, weights=w))
    sd = np.sqrt(var)
    inner = (x >= p10) & (x <= p90)
    out.update(
        mean=mean,
        trimmed_mean_10=float(np.average(x[inner], weights=w[inner])),
        median=median,
        sd=float(sd),
        mad=float(weighted_quantile(np.abs(x - median), w, 0.5)[0]),
        iqr=q3 - q1,
        min=float(x.min()),
        max=float(x.max()),
        skewness=float(np.average((x - mean) ** 3, weights=w) / sd**3)
        if sd > 0
        else np.nan,
    )
    pct = weighted_quantile(x, w, list(_PERCENTILES.values()))
    out.update({k: float(v) for k, v in zip(_PERCENTILES, pct, strict=True)})
    return out


def _check() -> None:
    lo, hi = wilson_ci(45, 50)
    assert abs(lo - 0.786) < 0.001 and abs(hi - 0.957) < 0.001, (lo, hi)
    assert np.isnan(wilson_ci(0, 0)[0])

    assert np.isnan(robust_z([5.0, 5.0, 5.0])).all()
    z = robust_z([1.0, 2.0, 3.0, 4.0, 100.0])
    assert abs(z[2]) < 1e-12 and z[4] > 3.5, z

    lo, hi = tukey_fences([1, 2, 3, 4, 5, 6, 7, 8])
    assert (lo, hi) == (2 - 1.5 * 4, 6 + 1.5 * 4), (lo, hi)

    q = bh_adjust([0.01, 0.04, 0.03, np.nan])
    assert np.allclose(q[:3], [0.03, 0.04, 0.04]) and np.isnan(q[3]), q

    assert weighted_quantile([1, 2, 3], [1, 1, 10], 0.5)[0] == 3
    x = np.arange(1.0, 101.0)
    assert weighted_quantile(x, np.ones_like(x), 0.25)[0] == np.quantile(
        x, 0.25, method="inverted_cdf"
    )

    circ = circular_summary([350.0, 10.0])
    assert min(circ["mean_deg"], 360 - circ["mean_deg"]) < 1e-9, circ
    assert circular_summary([0.0, 180.0])["resultant_length"] < 1e-12

    s = summarize([1.0, 2.0, 3.0, 4.0, np.nan])
    assert s["n"] == 4 and s["n_missing"] == 1 and s["mean"] == 2.5, s
    assert s["median"] == 2 and s["min"] == 1 and s["max"] == 4, s
    sw = summarize([1.0, 2.0, 3.0], [1.0, 1.0, 10.0])
    assert sw["median"] == 3 and abs(sw["mean"] - 33 / 12) < 1e-12, sw
    assert summarize([])["n"] == 0
    assert list(summarize([1.0, 2.0])) == SUMMARY_COLUMNS
    print("stats checks passed")


if __name__ == "__main__":
    _check()

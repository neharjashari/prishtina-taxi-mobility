"""Haversine distances, projection to metres, and grid cells."""

import numpy as np
from numpy.typing import ArrayLike
from pyproj import Transformer

from taxi import config

EARTH_RADIUS_M = 6_371_008.8

_TO_METRIC = Transformer.from_crs("EPSG:4326", config.CRS_METRIC, always_xy=True)


def _as_float(values: ArrayLike) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def haversine_m(
    lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike
) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (
        np.radians(_as_float(v)) for v in (lat1, lon1, lat2, lon2)
    )
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def to_metric(lon: ArrayLike, lat: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    x_m, y_m = _TO_METRIC.transform(_as_float(lon), _as_float(lat))
    x_m, y_m = np.asarray(x_m, dtype=np.float64), np.asarray(y_m, dtype=np.float64)
    missing = ~(np.isfinite(x_m) & np.isfinite(y_m))
    x_m[missing] = np.nan
    y_m[missing] = np.nan
    return x_m, y_m


def cell_id(
    x_m: ArrayLike, y_m: ArrayLike, size_m: int = config.GRID_SIZE_M
) -> np.ndarray:
    x_m, y_m = np.atleast_1d(_as_float(x_m)), np.atleast_1d(_as_float(y_m))
    valid = np.isfinite(x_m) & np.isfinite(y_m)
    col = np.floor(x_m[valid] / size_m).astype(np.int64).astype(str)
    row = np.floor(y_m[valid] / size_m).astype(np.int64).astype(str)
    cells = np.full(x_m.shape, None, dtype=object)
    cells[valid] = np.char.add(np.char.add(col, "_"), row)
    return cells


def _check() -> None:
    one_deg = haversine_m(42.0, 21.0, 43.0, 21.0)
    assert abs(one_deg - 111_195) < 10, one_deg
    assert haversine_m(42.66, 21.16, 42.66, 21.16) == 0
    assert np.isnan(haversine_m(np.nan, 21.16, 42.66, 21.16))
    assert haversine_m(
        [42.0, 42.0], [21.0, 21.0], [43.0, 42.0], [21.0, 21.0]
    ).shape == (2,)

    x, y = to_metric(21.1655, 42.6629)
    assert 500_000 < x < 520_000, x
    assert 4_700_000 < y < 4_740_000, y
    x_swapped, y_swapped = to_metric(42.6629, 21.1655)
    assert not (500_000 < x_swapped < 520_000 and 4_700_000 < y_swapped < 4_740_000)
    x2, y2 = to_metric(21.1655, 42.6719)
    metric = np.hypot(x2 - x, y2 - y)
    sphere = haversine_m(42.6629, 21.1655, 42.6719, 21.1655)
    assert abs(metric - sphere) / sphere < 0.005, (metric, sphere)
    x_na, _ = to_metric(np.nan, 42.66)
    assert np.isnan(x_na)

    assert list(cell_id([509_000.0, 509_499.9, 509_500.0], [4_723_000.0] * 3)) == [
        "1018_9446",
        "1018_9446",
        "1019_9446",
    ]
    assert cell_id(-1.0, -1.0)[0] == "-1_-1"
    assert cell_id([np.nan], [4_723_000.0])[0] is None
    assert cell_id(1_500.0, 1_500.0, size_m=1_000)[0] == "1_1"
    print("geo checks passed")


if __name__ == "__main__":
    _check()

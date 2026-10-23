"""Bashkimi i pikave me burimet e tjera: kufijte, rrjeti dhe kalendari.
Burimet: kufijte e Kosoves dhe komunat nga Who's On First (HDX, CC BY), dhe
data/external/calendar_2023.csv (festat zyrtare nga MPB).
"""

from functools import cache

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry.base import BaseGeometry

from taxi import PipelineError, config, geo

COUNTRY_SHP = config.KOSOVO_ADMIN_DIR / "whosonfirst-data-admin-xk-country-polygon.shp"
MUNICIPALITY_SHP = (
    config.KOSOVO_ADMIN_DIR / "whosonfirst-data-admin-xk-localadmin-polygon.shp"
)
MUNICIPALITY_RELABEL = {1108809573: "Mitrovice e Veriut"}


@cache
def kosovo_boundary() -> BaseGeometry:
    country = gpd.read_file(COUNTRY_SHP).to_crs("EPSG:4326")
    shape = country.geometry.union_all()
    shapely.prepare(shape)
    return shape


@cache
def municipalities() -> gpd.GeoDataFrame:
    m = gpd.read_file(MUNICIPALITY_SHP).to_crs("EPSG:4326")
    m["municipality"] = (
        m["id"].astype("int64").map(MUNICIPALITY_RELABEL).fillna(m["name"])
    )
    return m[["municipality", "geometry"]]


def municipality_categories() -> list[str]:
    return sorted(municipalities()["municipality"].unique())


@cache
def calendar() -> pd.DataFrame:
    cal = pd.read_csv(
        config.CALENDAR_CSV,
        usecols=["date", "is_weekend", "is_holiday"],
        dtype={"date": str, "is_weekend": bool, "is_holiday": bool},
        true_values=["true"],
        false_values=["false"],
    )
    cal["date"] = pd.to_datetime(cal["date"], format="%Y-%m-%d").dt.date
    return cal.set_index("date")


def in_kosovo(lon: pd.Series, lat: pd.Series) -> np.ndarray:
    lon_min, lat_min, lon_max, lat_max = config.KOSOVO_BBOX
    lon, lat = lon.to_numpy(np.float64), lat.to_numpy(np.float64)
    inside = np.zeros(len(lon), dtype=bool)
    in_box = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
    inside[in_box] = shapely.contains_xy(kosovo_boundary(), lon[in_box], lat[in_box])
    return inside


def municipality(lon: pd.Series, lat: pd.Series, inside: np.ndarray) -> pd.Categorical:
    labels = np.full(len(lon), None, dtype=object)
    idx = np.flatnonzero(inside)
    if len(idx):
        points = gpd.GeoDataFrame(
            {"row": idx},
            geometry=gpd.points_from_xy(lon.to_numpy()[idx], lat.to_numpy()[idx]),
            crs="EPSG:4326",
        )
        joined = gpd.sjoin(
            points, municipalities(), how="inner", predicate="within"
        ).drop_duplicates("row")
        labels[joined["row"].to_numpy()] = joined["municipality"].to_numpy()
    return pd.Categorical(labels, categories=municipality_categories())


def add_point_context(df: pd.DataFrame) -> pd.DataFrame:
    df["municipality"] = municipality(df["lon"], df["lat"], df["in_kosovo"].to_numpy())
    x_m, y_m = geo.to_metric(df["lon"], df["lat"])
    df["cell_id"] = pd.array(geo.cell_id(x_m, y_m), dtype="string")

    df["date_local"] = df["ts_local"].dt.date
    df["hour"] = df["ts_local"].dt.hour.astype("int8")
    cal = calendar()
    missing = set(df["date_local"]) - set(cal.index)
    if missing:
        raise PipelineError(
            f"{config.CALENDAR_CSV.name} has no row for {sorted(missing)[:5]}"
        )
    df["is_weekend"] = df["date_local"].map(cal["is_weekend"]).astype(bool)
    df["is_holiday"] = df["date_local"].map(cal["is_holiday"]).astype(bool)
    return df

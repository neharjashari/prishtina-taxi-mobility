"""Bashkimi i pikave me burimet e tjera: kufijte, rrjeti dhe kalendari.
Burimet: kufijte e Kosoves dhe komunat nga Who's On First (HDX, CC BY), dhe
data/external/calendar_2023.csv (festat zyrtare nga MPB).

integration_summary.csv tregon per cdo burim celesin e bashkimit, sa rreshta u bashkuan,
sa jo dhe pse.
"""

from functools import cache

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry.base import BaseGeometry

from taxi import PipelineError, config, geo, io

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


def join_counts(df: pd.DataFrame) -> pd.Series:
    inside = df["in_kosovo"]
    return pd.Series(
        {
            "points": len(df),
            "in_kosovo": int(inside.sum()),
            "municipality": int(df["municipality"].notna().sum()),
            "no_municipality_inside": int((inside & df["municipality"].isna()).sum()),
            "cell_id": int(df["cell_id"].notna().sum()),
            "holiday": int(df["is_holiday"].sum()),
            "weekend": int(df["is_weekend"].sum()),
        }
    )


def write_summary(counts: pd.Series, vehicles: int) -> None:
    inventory = pd.read_csv(config.RAW_INVENTORY_CSV)
    if inventory["vehicle_id"].duplicated().any():
        raise PipelineError("A vehicle_id appears twice in raw_inventory.csv")
    headers = inventory.loc[inventory["bytes"] > 0, "header"]
    if (headers != ",".join(config.RAW_COLUMNS)).any():
        raise PipelineError("Not every raw file with data has the expected header")
    files = len(inventory)
    empty = int((inventory["bytes"] == 0).sum())
    points = int(counts["points"])
    outside = points - int(counts["in_kosovo"])
    no_muni = points - int(counts["municipality"])

    rows = [
        {
            "source": "vehicle files (Kaggle, 85 CSV)",
            "join_key": "file name -> vehicle_id",
            "time_base": "DeviceDateTime on the device clock (DEVICE_TZ), stored as ts_utc",
            "unit": "files",
            "n": files,
            "matched": files - empty,
            "unmatched": empty,
            "why_unmatched": "empty files (0 bytes), excluded by M5; every vehicle_id is "
            f"unique and every file with data has the same header; {vehicles} vehicles in this run",
        },
        {
            "source": "Kosovo boundary (WOF country polygon, HDX)",
            "join_key": "lon, lat inside the polygon",
            "time_base": "none",
            "unit": "points",
            "n": points,
            "matched": int(counts["in_kosovo"]),
            "unmatched": outside,
            "why_unmatched": "points outside Kosovo, kept with in_kosovo = false (R05)",
        },
        {
            "source": "municipalities (WOF localadmin polygons, HDX)",
            "join_key": "lon, lat inside the polygon",
            "time_base": "none",
            "unit": "points",
            "n": points,
            "matched": int(counts["municipality"]),
            "unmatched": no_muni,
            "why_unmatched": f"{outside:,} outside Kosovo; "
            f"{int(counts['no_municipality_inside']):,} inside the country polygon but in "
            "no municipality polygon",
        },
        {
            "source": f"grid ({config.GRID_SIZE_M} m cells, {config.CRS_METRIC})",
            "join_key": "lon, lat projected to metres",
            "time_base": "none",
            "unit": "points",
            "n": points,
            "matched": int(counts["cell_id"]),
            "unmatched": points - int(counts["cell_id"]),
            "why_unmatched": "every point with coordinates gets a cell",
        },
        {
            "source": "calendar_2023.csv (Ministry of Internal Affairs)",
            "join_key": "date_local",
            "time_base": f"local date from ts_local ({config.TZ_LOCAL})",
            "unit": "points",
            "n": points,
            "matched": points,
            "unmatched": 0,
            "why_unmatched": "the stage stops if a date is missing; "
            f"{int(counts['weekend']):,} points on weekends, "
            f"{int(counts['holiday']):,} on the holiday (June 28)",
        },
    ]
    io.write_report_table(pd.DataFrame(rows), "integration_summary")

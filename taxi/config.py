"""Paths, study window, thresholds, and seed for the whole pipeline.

Change a value only with team agreement, and record when and why it changed.

Paths that depend on --dev are functions: the pipeline calls `set_dev(True)` before a dev
run, and every output under data/ and reports/ then moves to data/dev/ and reports/dev/.
data/raw/, data/external/ and the raw inventory are always read from their normal location.
"""

from pathlib import Path
from typing import Literal

import pandas as pd

SEED = 42

# --- Time -----------------------------------------------------------------------------

# Kosovo local time. The IANA time zone database has no zone for Kosovo; Europe/Skopje
# has the same rules: UTC+2 for the whole study window, with no clock change.
TZ_LOCAL = "Europe/Skopje"
# Clock of DeviceDateTime. The source does not say whether it is UTC or local time; the
# team treats it as local time and revisits this if new evidence appears.
DEVICE_TZ = "Europe/Skopje"
# Study window on the device clock, end exclusive.
STUDY_START = pd.Timestamp("2023-06-01 00:00")
STUDY_END = pd.Timestamp("2023-09-01 00:00")

# --- Raw files ------------------------------------------------------------------------

RAW_COLUMNS = [
    "DeviceDateTime",
    "Longitute",
    "Latitude",
    "Speed",
    "Direction",
    "Di1",
    "Di2",
    "Di3",
]
RAW_FILES_EXPECTED = 85
EMPTY_FILES_EXPECTED = [8, 27]
# DeviceDateTime formats, tried in this order. Every file uses the first. 1.csv and 100.csv
# also use the second, without seconds, for days 1-12 of each month (re-saved in a
# spreadsheet). Both are month-first.
RAW_TS_FORMAT_SECONDS = "%m/%d/%Y %I:%M:%S %p"
RAW_TS_FORMAT_MINUTES = "%m/%d/%Y %H:%M"
# Boolean text in Di1-Di3: True/False in most files, TRUE/FALSE in 1.csv and 100.csv.
# Anything else, including an empty field, becomes NA.
RAW_BOOL_VALUES = {"True": True, "False": False, "TRUE": True, "FALSE": False}
# Bounding box of the Who's On First Kosovo country polygon (lon_min, lat_min, lon_max,
# lat_max), rounded to 4 decimals. Used by the quality profile before cleaning; the
# cleaning stage tests the polygon itself.
KOSOVO_BBOX = (20.0143, 41.8576, 21.7900, 43.2680)
# Files below this size are reported by verify-raw (18, 36, 64 and 66 at download time).
SMALL_FILE_BYTES = 2_000_000
# One near the median (1), one large (56), and one small (66) vehicle. Vehicle 1 also
# has both raw timestamp formats, so dev runs exercise them.
DEV_VEHICLES = [1, 56, 66]

# --- Sampling -------------------------------------------------------------------------

SAMPLE_SIZES = [0.001, 0.005, 0.01, 0.05]
THIN_S = 60
CLUSTER_N_VEHICLES = 10

# --- Validity and missing-value rules -------------------------------------------------

SPEED_MAX_KMH = 200
JUMP_KMH = 250
JUMP_MIN_DIST_M = 100
MAX_DROP_SHARE = 0.01
# Vehicles whose exact duplicates (R08) are above MAX_DROP_SHARE and were reviewed by the
# team: retransmitted records, so R08 may drop them. Every other vehicle and rule keeps
# the guard.
R08_REVIEWED_VEHICLES = [3, 5, 17, 30]
GAP_S = 900
IMPUTE_MAX_DT_S = 60
IMPUTE_METHOD = "implied_speed"
V_MOVE_KMH = 5
# Steps where an implied speed (distance / time) is meaningful enough to compare with the
# reported speed: 5-60 s long and at least 10 m, so GPS jitter on short steps stays out.
SPEED_CHECK_DT_S = (5, 60)
SPEED_CHECK_MIN_DIST_M = 10
GRID_SIZE_M = 500
CRS_METRIC = "EPSG:32634"

# --- Features -------------------------------------------------------------------------

STOP_RADIUS_M = 100
STOP_MIN_S = 300
# (lat, lon), to be set with a cited source.
CITY_CENTER: tuple[float, float] | None = None
# Bins and buckets are left-closed, [a, b): use pd.cut(..., right=False).
SPEED_BINS_KMH = [0, 5, 20, 50, 80, float("inf")]
SPEED_BIN_LABELS = ["stopped", "slow", "urban", "fast", "highway"]
TOD_BUCKETS = [0, 6, 10, 15, 19, 24]
TOD_LABELS = ["night", "morning", "midday", "afternoon", "evening"]
NIGHT_HOURS = (0, 6)

# --- Paths ----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
EXTERNAL_DIR = ROOT / "data" / "external"
KOSOVO_ADMIN_DIR = EXTERNAL_DIR / "kosovo_admin"
CALENDAR_CSV = EXTERNAL_DIR / "calendar_2023.csv"
WEATHER_CSV = EXTERNAL_DIR / "weather_prishtina.csv"
# Pins the input bytes. It describes data/raw/, so it is not switched by --dev.
RAW_INVENTORY_CSV = ROOT / "reports" / "tables" / "raw_inventory.csv"

PointsStage = Literal["interim", "clean", "final"]
POINTS_SUBDIRS: dict[str, str] = {
    "interim": "interim/points",
    "clean": "processed/points_clean",
    "final": "processed/points",
}

_dev = False


def set_dev(dev: bool) -> None:
    """Switch every output path to the dev locations (True) or the normal ones (False)."""
    global _dev
    _dev = dev


def is_dev() -> bool:
    return _dev


def data_dir() -> Path:
    return ROOT / "data" / "dev" if _dev else ROOT / "data"


def reports_dir() -> Path:
    return ROOT / "reports" / "dev" if _dev else ROOT / "reports"


def points_dir(stage: PointsStage) -> Path:
    return data_dir() / POINTS_SUBDIRS[stage]


def processed_dir() -> Path:
    return data_dir() / "processed"


def samples_dir() -> Path:
    return processed_dir() / "samples"


def model_dir() -> Path:
    return processed_dir() / "model"


def sample_csv() -> Path:
    return data_dir() / "sample" / "points_sample.csv"


def export_dir() -> Path:
    return data_dir() / "export" / "phase-1"


def tables_dir() -> Path:
    return reports_dir() / "tables"


def figures_dir() -> Path:
    return reports_dir() / "figures"


def run_log_csv() -> Path:
    return reports_dir() / "run_log.csv"


def cleaning_log_csv() -> Path:
    return tables_dir() / "cleaning_log.csv"


def data_dictionary_csv() -> Path:
    return tables_dir() / "data_dictionary.csv"

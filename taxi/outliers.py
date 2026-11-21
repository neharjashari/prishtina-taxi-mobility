"""Point and step outliers: rules O01-O04, the detect stage, and the review of the detections.

The rules, per vehicle, in the order the screen stage applies them:

- O01 spike: a point whose position jumps away and back. Both of its steps are faster than
  JUMP_KMH, and its two neighbours are close to each other. Evaluated on points_clean.
- O02 impossible step: implied speed above JUMP_KMH (seconds at both ends, no gap).
- O03 speed spike: the reported speed is so far from the line between its neighbours'
  speeds that reaching it needs more than ACCEL_MAX_MS2, while the neighbours agree.
- O04 parked drift: a step of at least DRIFT_MIN_M within DRIFT_MAX_DT_S while the
  ignition is off and the reported speed is 0 at both ends.

O02-O04 are evaluated on the points without the O01 spikes, with the steps recomputed
exactly as the clean stage computes them. A step rule flags the point the step starts at.

The detect stage changes no data. It writes the flags (data/processed/outliers/
point_flags.parquet), T13 (flags per rule and vehicle), T14 (distribution-based detectors
against the rules), the review sample and, once reviewers have labelled it, the validation
of each rule (T15), and figures F13 and F14. Example tracks are drawn in metres relative to
the flagged point, without a base map, so no figure shows where a vehicle was.
"""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import cohen_kappa_score

from taxi import PipelineError, clean, config, figures, geo, io, stats

STAGE = "detect"
FLAGS_TABLE = "outliers/point_flags"
FLAGS_DATASET = "processed/outliers/point_flags"
RULES = ["O01", "O02", "O03", "O04"]
STEP_RULES = {"O02", "O04"}
RULE_NAMES = {
    "O01": "spike",
    "O02": "impossible step",
    "O03": "speed spike",
    "O04": "parked drift",
}
FLAG_COLUMNS = ["vehicle_id", "source_row", "rule_id", "score"]
LABELS = {"error", "real", "unsure"}
COLUMNS = [
    "vehicle_id",
    "source_row",
    "ts_utc",
    "ts_has_seconds",
    "lat",
    "lon",
    "speed_kmh",
    "di1",
    "dt_s",
    "dist_m",
    "implied_speed_kmh",
    "gap_after",
    "speed_imputed",
]
# F13 bins, on log10(1 + speed in km/h).
F13_X_EDGES = np.linspace(0, 2.35, 95)
F13_Y_EDGES = np.linspace(0, 5, 201)

FLAGS_DICTIONARY = [
    {
        "column": "vehicle_id",
        "attribute_type": "nominal",
        "values": "1 to 119",
        "description": "Vehicle id, the raw file name.",
        "source": "points_clean",
    },
    {
        "column": "source_row",
        "attribute_type": "nominal",
        "values": "0 or more",
        "description": "0-based data row of the flagged point in the raw file. For a step rule "
        "(O02, O04), the point the step starts at.",
        "source": "points_clean",
    },
    {
        "column": "rule_id",
        "attribute_type": "nominal",
        "values": "O01, O02, O03, O04",
        "description": "O01 spike (position jumps away and back), O02 impossible step (implied "
        "speed above JUMP_KMH), O03 speed spike (reported speed off the line between its "
        "neighbours by more than ACCEL_MAX_MS2 can explain), O04 parked drift (moves DRIFT_MIN_M or more with the "
        "ignition off and speed 0). One row per point and rule.",
        "source": "detect stage",
    },
    {
        "column": "score",
        "unit": "km/h (O01, O02), m/s² (O03), m (O04)",
        "attribute_type": "ratio",
        "values": "above the rule's threshold",
        "description": "The measure the rule tests: the slower of the two implied speeds (O01), "
        "the implied speed (O02), the acceleration needed to reach the reported speed from the "
        "line between the neighbours' speeds over the shorter step (O03), "
        "the step distance (O04).",
        "source": "detect stage",
    },
]


# --- The rules ------------------------------------------------------------------------


def read_vehicle(vehicle_id: int) -> pd.DataFrame:
    """One vehicle's clean points, sorted by ts_utc and source_row, with Phase I steps."""
    df = io.read_points("clean", [vehicle_id], columns=COLUMNS)
    return df.sort_values(["ts_utc", "source_row"], kind="stable").reset_index(
        drop=True
    )


def _arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    def num(col: str) -> np.ndarray:
        return df[col].to_numpy(dtype=np.float64, na_value=np.nan)

    return {
        "implied": num("implied_speed_kmh"),
        "dist": num("dist_m"),
        "dt": num("dt_s"),
        "speed": num("speed_kmh"),
        "lat": num("lat"),
        "lon": num("lon"),
        "gap": df["gap_after"].to_numpy(dtype=bool, na_value=False),
        "secs": df["ts_has_seconds"].to_numpy(dtype=bool, na_value=False),
        "parked": df["di1"].eq(False).fillna(False).to_numpy(dtype=bool),
    }


Result = tuple[np.ndarray, np.ndarray, np.ndarray]  # flagged, score, evaluable


def _empty(n: int) -> Result:
    return np.zeros(n, bool), np.full(n, np.nan), np.zeros(n, bool)


def spike_points(a: dict[str, np.ndarray], jump_kmh: float = config.JUMP_KMH) -> Result:
    """O01: both steps of the point faster than jump_kmh, and its neighbours close together."""
    n = len(a["implied"])
    flagged, score, evaluable = _empty(n)
    if n < 3:
        return flagged, score, evaluable
    inc, out = a["implied"][:-2], a["implied"][1:-1]
    known = np.isfinite(inc) & np.isfinite(out) & ~a["gap"][:-2] & ~a["gap"][1:-1]
    back = geo.haversine_m(a["lat"][:-2], a["lon"][:-2], a["lat"][2:], a["lon"][2:])
    shorter = np.fmin(a["dist"][:-2], a["dist"][1:-1])
    hit = (
        known
        & (inc > jump_kmh)
        & (out > jump_kmh)
        & (back < config.SPIKE_RETURN_SHARE * shorter)
    )
    evaluable[1:-1] = known
    flagged[1:-1] = hit
    score[1:-1] = np.where(hit, np.fmin(inc, out), np.nan)
    return flagged, score, evaluable


def impossible_steps(
    a: dict[str, np.ndarray], jump_kmh: float = config.JUMP_KMH
) -> Result:
    """O02: the step's implied speed is above jump_kmh (seconds at both ends, no gap)."""
    evaluable = np.isfinite(a["implied"]) & ~a["gap"]
    flagged = evaluable & (a["implied"] > jump_kmh)
    return flagged, np.where(flagged, a["implied"], np.nan), evaluable


def speed_spikes(
    a: dict[str, np.ndarray], accel_max: float = config.ACCEL_MAX_MS2
) -> Result:
    """O03: a reported speed its neighbours cannot explain.

    For a point with usable steps on both sides (positive duration, no gap, seconds at
    both ends, as for accel_ms2), the speed its neighbours imply is their linear
    interpolation in time. The point is a candidate when reaching its reported speed from
    that line needs more than accel_max (m/s²) over the shorter of its two steps, while
    the neighbours agree with each other: the change between them needs at most
    accel_max. Of two adjacent candidates only the larger deviation is flagged, because
    one wrong speed makes both points of the shared step look wrong.

    The typical case is an event point about 1 s before a regular one whose speed reads
    0 between two neighbours that agree (56, 0, 36 km/h). Its acceleration into the point
    is spread over 10 s and looks harmless; only the deviation from the line shows it.
    Score: that acceleration.
    """
    n = len(a["speed"])
    flagged, score, evaluable = _empty(n)
    if n < 3:
        return flagged, score, evaluable
    v = a["speed"] / 3.6
    dt = a["dt"][:-1]
    ok = (dt > 0) & ~a["gap"][:-1] & a["secs"][:-1] & a["secs"][1:]
    d_in = np.where(ok[:-1], dt[:-1], np.nan)
    d_out = np.where(ok[1:], dt[1:], np.nan)
    v_prev, v_mid, v_next = v[:-2], v[1:-1], v[2:]
    known = (
        np.isfinite(d_in)
        & np.isfinite(d_out)
        & np.isfinite(v_prev)
        & np.isfinite(v_mid)
        & np.isfinite(v_next)
    )
    total = d_in + d_out
    expected = v_prev + (v_next - v_prev) * d_in / total
    need = np.abs(v_mid - expected) / np.fmin(d_in, d_out)
    across = np.abs(v_next - v_prev) / total
    candidate = known & (need > accel_max) & (across <= accel_max)
    ranked = np.where(candidate, need, -np.inf)
    left = np.r_[-np.inf, ranked[:-1]]
    right = np.r_[ranked[1:], -np.inf]
    hit = candidate & (ranked >= left) & (ranked > right)
    evaluable[1:-1] = known
    flagged[1:-1] = hit
    score[1:-1] = np.where(hit, need, np.nan)
    return flagged, score, evaluable


def parked_drift(a: dict[str, np.ndarray], min_m: float = config.DRIFT_MIN_M) -> Result:
    """O04: a step of at least min_m within DRIFT_MAX_DT_S, parked at both ends.

    Parked: the ignition (di1) is off and the reported speed is 0.
    """
    n = len(a["dist"])
    flagged, score, evaluable = _empty(n)
    if n < 2:
        return flagged, score, evaluable
    still = a["parked"] & (a["speed"] == 0)
    known = (
        still[:-1]
        & still[1:]
        & ~a["gap"][:-1]
        & (a["dt"][:-1] <= config.DRIFT_MAX_DT_S)
        & np.isfinite(a["dist"][:-1])
    )
    hit = known & (a["dist"][:-1] >= min_m)
    evaluable[:-1] = known
    flagged[:-1] = hit
    score[:-1] = np.where(hit, a["dist"][:-1], np.nan)
    return flagged, score, evaluable


def without_spikes(points: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """The points without the O01 spikes, with the steps recomputed by clean.add_steps."""
    spikes = spike_points(_arrays(points))[0]
    if not spikes.any():
        return points, spikes
    kept = points.loc[~spikes].reset_index(drop=True)
    clean.add_steps(kept)
    return kept, spikes


def rule_results(
    points: pd.DataFrame, kept: pd.DataFrame, factor: float = 1.0
) -> dict[str, Result]:
    """Every rule with its threshold times factor: O01 on points, O02-O04 on kept."""
    a0, a1 = _arrays(points), _arrays(kept)
    return {
        "O01": spike_points(a0, config.JUMP_KMH * factor),
        "O02": impossible_steps(a1, config.JUMP_KMH * factor),
        "O03": speed_spikes(a1, config.ACCEL_MAX_MS2 * factor),
        "O04": parked_drift(a1, config.DRIFT_MIN_M * factor),
    }


def _flag_rows(
    frame: pd.DataFrame, rule: str, mask: np.ndarray, score: np.ndarray
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vehicle_id": frame["vehicle_id"].to_numpy()[mask],
            "source_row": frame["source_row"].to_numpy()[mask],
            "rule_id": rule,
            "score": score[mask],
        }
    )


def _typed_flags(parts: list[pd.DataFrame]) -> pd.DataFrame:
    df = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=FLAG_COLUMNS)
    )
    return df.astype(
        {
            "vehicle_id": "int16",
            "source_row": "int32",
            "rule_id": "string",
            "score": "float32",
        }
    )[FLAG_COLUMNS]


def comparable_steps(kept: pd.DataFrame) -> np.ndarray:
    """Steps where implied and reported speed can be compared (the F5 filter)."""
    lo, hi = config.SPEED_CHECK_DT_S
    return (
        kept["dt_s"].between(lo, hi)
        & (kept["dist_m"] >= config.SPEED_CHECK_MIN_DIST_M)
        & (kept["speed_kmh"] >= config.V_MOVE_KMH)
        & ~kept["speed_imputed"].astype(bool)
        & kept["implied_speed_kmh"].notna()
    ).to_numpy()


# --- The detect stage -----------------------------------------------------------------


def run(dev: bool) -> None:
    ids = io.vehicle_ids("clean")
    if not ids:
        raise PipelineError("No clean points: run the clean stage first.")

    flags, near, counts, ratios, ratio_o02 = [], [], [], [], []
    hist = np.zeros((len(F13_X_EDGES) - 1, len(F13_Y_EDGES) - 1))
    for v in ids:
        points = read_vehicle(v)
        kept, _ = without_spikes(points)
        res = rule_results(points, kept)
        res_near = rule_results(points, kept, config.REVIEW_NEAR_FACTOR)
        no_seconds = int((~_arrays(points)["secs"]).sum())
        for rule in RULES:
            frame = points if rule == "O01" else kept
            hit, score, evaluable = res[rule]
            flags.append(_flag_rows(frame, rule, hit, score))
            near_hit = res_near[rule][0] & ~hit
            near.append(_flag_rows(frame, rule, near_hit, res_near[rule][1]))
            counts.append(
                {
                    "rule_id": rule,
                    "vehicle_id": v,
                    "rows_clean": len(points),
                    "evaluated": int(evaluable.sum()),
                    "flagged": int(hit.sum()),
                    "rows_without_seconds": no_seconds,
                }
            )
        usable = comparable_steps(kept)
        ratios.append(
            np.log(
                kept["implied_speed_kmh"].to_numpy(np.float64)[usable]
                / kept["speed_kmh"].to_numpy(np.float64)[usable]
            ).astype(np.float32)
        )
        ratio_o02.append(res["O02"][0][usable])
        hist += speed_histogram(kept)

    flags_df = _typed_flags(flags)
    near_df = _typed_flags(near)
    io.write_table(flags_df, FLAGS_TABLE)
    io.write_dictionary(FLAGS_DATASET, FLAGS_DICTIONARY, flags_df, owner=STAGE)
    rules_table = rules_points_table(counts)
    io.write_report_table(rules_table, "outlier_rules_points")
    io.write_report_table(
        methods_table(np.concatenate(ratios), np.concatenate(ratio_o02)),
        "outlier_methods_points",
    )

    sample = draw_review_sample(flags_df, near_df, np.random.default_rng(config.SEED))
    io.write_report_table(sample, "outlier_review_sample")
    write_validation(sample)

    figure_speeds(hist, rules_table)
    figure_examples(sample)
    fleet = rules_table[rules_table["vehicle_id"].isna()].set_index("rule_id")
    for rule in RULES:
        r = fleet.loc[rule]
        print(
            f"{rule} {RULE_NAMES[rule]}: {int(r['flagged']):,} flagged in "
            f"{int(r['vehicles_flagged'])} vehicles, largest vehicle share {r['max_vehicle_share']:.3%}"
        )


def rules_points_table(counts: list[dict]) -> pd.DataFrame:
    """T13: per rule and vehicle, and per rule over the fleet (vehicle_id empty)."""
    per = pd.DataFrame(counts)
    per["share"] = per["flagged"] / per["rows_clean"]
    per["over_max_drop_share"] = per["share"] > config.MAX_DROP_SHARE
    fleet = per.groupby("rule_id", as_index=False).agg(
        rows_clean=("rows_clean", "sum"),
        evaluated=("evaluated", "sum"),
        flagged=("flagged", "sum"),
        rows_without_seconds=("rows_without_seconds", "sum"),
        vehicles_flagged=("flagged", lambda s: int((s > 0).sum())),
        max_vehicle_share=("share", "max"),
        over_max_drop_share=("over_max_drop_share", "any"),
    )
    fleet["share"] = fleet["flagged"] / fleet["rows_clean"]
    fleet["vehicle_id"] = pd.NA
    table = pd.concat([per, fleet], ignore_index=True)
    table["vehicle_id"] = table["vehicle_id"].astype("Int16")
    columns = [
        "rule_id",
        "vehicle_id",
        "rows_clean",
        "rows_without_seconds",
        "evaluated",
        "flagged",
        "share",
        "over_max_drop_share",
        "vehicles_flagged",
        "max_vehicle_share",
    ]
    return table[columns].sort_values(["rule_id", "vehicle_id"], na_position="last")


def methods_table(log_ratio: np.ndarray, o02: np.ndarray) -> pd.DataFrame:
    """T14: distribution-based detectors on log(implied / reported speed), against O02.

    The same comparable steps for every method. expected_if_normal is what each rule flags
    on error-free normal data of the same size: the share it flags by construction.
    """
    x = log_ratio.astype(np.float64)
    n = len(x)
    z = (x - x.mean()) / x.std() if n else x
    rz = stats.robust_z(x)
    lo, hi = stats.tukey_fences(x) if n else (np.nan, np.nan)
    tukey_tail = 2 * norm.sf(4 * norm.ppf(0.75))
    methods = [
        ("z-score", "|z| > 3 (mean and sd)", np.abs(z) > 3, 2 * norm.sf(3)),
        (
            "modified z-score",
            f"|modified z| > {config.ROBUST_Z_MAX} (median and MAD)",
            np.abs(rz) > config.ROBUST_Z_MAX,
            2 * norm.sf(config.ROBUST_Z_MAX),
        ),
        (
            "Tukey fences",
            "outside Q1 - 1.5 IQR, Q3 + 1.5 IQR",
            (x < lo) | (x > hi),
            tukey_tail,
        ),
        (
            "rule O02",
            f"implied speed above JUMP_KMH ({config.JUMP_KMH} km/h)",
            o02,
            np.nan,
        ),
    ]
    rows = []
    for name, rule, hit, tail in methods:
        rows.append(
            {
                "method": name,
                "rule": rule,
                "steps": n,
                "flagged": int(hit.sum()),
                "share": hit.mean() if n else np.nan,
                "flagged_implied_higher": int((hit & (x > 0)).sum()),
                "flagged_implied_lower": int((hit & (x < 0)).sum()),
                "expected_if_normal": tail * n,
                "also_rule_O02": int((hit & o02).sum()),
            }
        )
    return pd.DataFrame(rows)


# --- Review of the detections (T15) ---------------------------------------------------


def draw_review_sample(
    flags: pd.DataFrame, near: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Per rule: flagged items and near-misses at random, at most REVIEW_MAX_PER_VEHICLE each.

    Rules are interleaved, so the first REVIEW_CALIBRATION_N items (labelled by every
    reviewer) cover every rule; the rest go round-robin to the reviewers. The label files
    show neither the kind (flagged or near-miss) nor the score, so labels are blind.
    """
    parts = []
    for rule in RULES:
        for kind, pool, n in (
            ("flagged", flags, config.REVIEW_N_PER_RULE),
            ("near", near, config.REVIEW_NEAR_N),
        ):
            p = pool[pool["rule_id"] == rule]
            p = p.iloc[rng.permutation(len(p))]
            p = p[p.groupby("vehicle_id").cumcount() < config.REVIEW_MAX_PER_VEHICLE]
            parts.append(p.head(n).assign(kind=kind))
    sample = pd.concat(parts, ignore_index=True)
    sample = sample.iloc[rng.permutation(len(sample))]
    sample["rank"] = sample.groupby("rule_id").cumcount()
    sample = sample.sort_values(["rank", "rule_id"], kind="stable").reset_index(
        drop=True
    )
    sample["item"] = np.arange(1, len(sample) + 1)
    calibration = sample["item"] <= config.REVIEW_CALIBRATION_N
    reviewers = config.REVIEWERS
    sample["reviewer"] = "all"
    sample.loc[~calibration, "reviewer"] = [
        reviewers[i % len(reviewers)] for i in range(int((~calibration).sum()))
    ]
    return sample[
        ["item", "vehicle_id", "source_row", "rule_id", "kind", "score", "reviewer"]
    ]


def label_path(reviewer: str) -> Path:
    return config.OUTLIER_LABELS_DIR / f"{reviewer}.csv"


def start_labels(reviewer: str) -> Path:
    """Create a reviewer's label file: their items and the calibration items, labels empty.

    Labels are then written by hand: error, real, or unsure, with an optional note. The
    file never shows the score or whether an item was flagged. Never overwrites a file.
    """
    if reviewer not in config.REVIEWERS:
        raise ValueError(
            f"reviewer must be one of {config.REVIEWERS}, got {reviewer!r}"
        )
    path = label_path(reviewer)
    if path.exists():
        raise FileExistsError(
            f"{path} exists; it holds labels, so it is never overwritten"
        )
    sample = io.read_report_table("outlier_review_sample")
    mine = sample[sample["reviewer"].isin([reviewer, "all"])]
    path.parent.mkdir(parents=True, exist_ok=True)
    mine[["item", "vehicle_id", "source_row", "rule_id"]].assign(
        label="", note=""
    ).to_csv(path, index=False, encoding="utf-8")
    return path


def read_labels() -> pd.DataFrame:
    """Every reviewer's labelled rows; rows still without a label are left out."""
    parts = []
    for path in sorted(config.OUTLIER_LABELS_DIR.glob("*.csv")):
        df = pd.read_csv(
            path,
            dtype={
                "item": "int64",
                "vehicle_id": "int64",
                "source_row": "int64",
                "rule_id": str,
            },
            keep_default_na=False,
        )
        parts.append(df.assign(reviewer=path.stem))
    columns = ["item", "vehicle_id", "source_row", "rule_id", "reviewer", "label"]
    if not parts:
        return pd.DataFrame(columns=columns)
    labels = pd.concat(parts, ignore_index=True)
    labels["label"] = labels["label"].str.strip().str.lower()
    labels = labels[labels["label"] != ""]
    unknown = sorted(set(labels["label"]) - LABELS)
    if unknown:
        raise PipelineError(
            f"Unknown labels {unknown} in {config.OUTLIER_LABELS_DIR}: use {sorted(LABELS)}."
        )
    return labels[columns]


def _consensus(labels: pd.Series) -> str:
    counts = labels.value_counts()
    if len(counts) == 1 or counts.iloc[0] > counts.iloc[1]:
        return counts.index[0]
    return "unsure"


def validation_table(sample: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """T15: per rule the precision of the flagged items and the errors among the near-misses.

    An item's label is the majority of its reviewers' labels (a tie counts as unsure).
    Precision = errors / (errors + real), with its Wilson 95% interval. Agreement between
    reviewers (Cohen's kappa per pair, on the calibration items) is in the row "all".
    """
    keys = ["item", "vehicle_id", "source_row", "rule_id"]
    merged = labels.merge(sample[[*keys, "kind"]], on=keys, how="inner")
    items = merged.groupby(["item", "rule_id", "kind"], as_index=False)["label"].agg(
        _consensus
    )
    rows = []
    for rule in [*RULES, "all"]:
        sel = items if rule == "all" else items[items["rule_id"] == rule]
        sampled = sample if rule == "all" else sample[sample["rule_id"] == rule]
        flagged, near_ = sel[sel["kind"] == "flagged"], sel[sel["kind"] == "near"]
        errors = int((flagged["label"] == "error").sum())
        real = int((flagged["label"] == "real").sum())
        near_errors = int((near_["label"] == "error").sum())
        near_real = int((near_["label"] == "real").sum())
        precision = errors / (errors + real) if errors + real else np.nan
        low, high = stats.wilson_ci(errors, errors + real)
        if rule == "all":
            decision = ""
        elif errors + real == 0:
            decision = "not reviewed"
        else:
            decision = (
                "remove" if precision >= config.REVIEW_MIN_PRECISION else "flag only"
            )
        rows.append(
            {
                "rule_id": rule,
                "sampled_flagged": int((sampled["kind"] == "flagged").sum()),
                "labelled_flagged": len(flagged),
                "errors": errors,
                "real": real,
                "unsure": int((flagged["label"] == "unsure").sum()),
                "precision": precision,
                "precision_ci_low": low,
                "precision_ci_high": high,
                "sampled_near": int((sampled["kind"] == "near").sum()),
                "labelled_near": len(near_),
                "near_errors": near_errors,
                "near_real": near_real,
                "near_error_share": near_errors / (near_errors + near_real)
                if near_errors + near_real
                else np.nan,
                "decision": decision,
            }
        )
    table = pd.DataFrame(rows)

    calibration = sample.loc[sample["reviewer"] == "all", "item"]
    wide = merged[merged["item"].isin(calibration)].pivot_table(
        index="item", columns="reviewer", values="label", aggfunc="first"
    )
    kappas = {}
    names = list(wide.columns)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            both = wide[[a, b]].dropna()
            if len(both) >= 2:
                with np.errstate(divide="ignore", invalid="ignore"):
                    kappas[f"{a}-{b}"] = cohen_kappa_score(both[a], both[b])
    table["calibration_items"] = pd.NA
    table["kappa_pairs"] = ""
    table["kappa_mean"] = np.nan
    last = table.index[-1]
    table.loc[last, "calibration_items"] = int(wide.notna().all(axis=1).sum())
    table.loc[last, "kappa_pairs"] = "; ".join(
        f"{k} {v:.2f}" for k, v in kappas.items()
    )
    if kappas:
        table.loc[last, "kappa_mean"] = float(np.nanmean(list(kappas.values())))
    return table


def write_validation(sample: pd.DataFrame) -> None:
    """Write T15's validation when there are labels; otherwise remove a stale table."""
    path = io.report_table_path("outlier_validation")
    labels = read_labels()
    if labels.empty:
        path.unlink(missing_ok=True)
        print(
            f"No review labels in {config.OUTLIER_LABELS_DIR}: outlier_validation not written."
        )
        return
    keys = ["item", "vehicle_id", "source_row", "rule_id"]
    matched = labels.merge(sample[keys], on=keys, how="inner")
    if len(matched) < len(labels):
        print(
            f"{len(labels) - len(matched):,} of {len(labels):,} labels match no item of this "
            "review sample and are left out."
        )
    io.write_report_table(validation_table(sample, labels), "outlier_validation")


# --- Figures --------------------------------------------------------------------------


def _log1p10(v: np.ndarray) -> np.ndarray:
    return np.log10(1 + v)


def speed_histogram(kept: pd.DataFrame) -> np.ndarray:
    """F13 counts: steps with an implied speed and no gap, on log10(1 + km/h) bins."""
    use = (
        kept["implied_speed_kmh"].notna()
        & ~kept["gap_after"]
        & kept["speed_kmh"].notna()
    )
    x = _log1p10(kept.loc[use, "speed_kmh"].to_numpy(np.float64))
    y = _log1p10(kept.loc[use, "implied_speed_kmh"].to_numpy(np.float64))
    x = np.clip(x, 0, F13_X_EDGES[-1] - 1e-9)
    y = np.clip(y, 0, F13_Y_EDGES[-1] - 1e-9)
    return np.histogram2d(x, y, bins=[F13_X_EDGES, F13_Y_EDGES])[0]


def _speed_ticks(ax: plt.Axes, axis: str, values: list[int]) -> None:
    ticks = _log1p10(np.array(values, dtype=np.float64))
    labels = [f"{v:,}" for v in values]
    if axis == "x":
        ax.set_xticks(ticks, labels)
    else:
        ax.set_yticks(ticks, labels)


def figure_speeds(hist: np.ndarray, rules_table: pd.DataFrame) -> None:
    """F13: implied against reported speed per step, with the O02 region."""
    fleet = rules_table[rules_table["vehicle_id"].isna()].set_index("rule_id")[
        "flagged"
    ]
    fig, ax = figures.subplots(7, 5.5)
    counts = np.ma.masked_equal(hist.T, 0)
    mesh = ax.pcolormesh(
        F13_X_EDGES,
        F13_Y_EDGES,
        counts,
        norm=mcolors.LogNorm(vmin=1, vmax=max(1, counts.max())),
        cmap=mcolors.ListedColormap(figures.SEQUENTIAL_BLUE),
        shading="flat",
    )
    diag = _log1p10(np.array([0.0, 200.0]))
    ax.plot(diag, diag, color=figures.INK, linewidth=1, label="implied = reported")
    jump = _log1p10(np.array([config.JUMP_KMH]))[0]
    ax.axhline(
        jump,
        color=figures.SERIES[1],
        linewidth=1.2,
        linestyle="--",
        label=f"JUMP_KMH ({config.JUMP_KMH} km/h): above it, rule O02",
    )
    ax.text(
        0.05,
        jump + 0.12,
        f"O02: {int(fleet['O02']):,} steps (after removing {int(fleet['O01']):,} O01 spikes)",
        color=figures.INK_SECONDARY,
        fontsize=8,
    )
    _speed_ticks(ax, "x", [0, 5, 20, 50, 100, 200])
    _speed_ticks(ax, "y", [0, 10, 100, 1_000, 10_000, 100_000])
    ax.set_xlim(F13_X_EDGES[0], F13_X_EDGES[-1])
    ax.set_ylim(F13_Y_EDGES[0], F13_Y_EDGES[-1])
    ax.set_xlabel("Reported speed (km/h, log scale of 1 + v)")
    ax.set_ylabel("Implied speed, distance / time (km/h, log scale of 1 + v)")
    ax.set_title(f"F13. Reported vs implied speed per step ({int(hist.sum()):,} steps)")
    ax.legend(loc="upper left")
    fig.colorbar(mesh, ax=ax, label="Steps per cell (log scale)")
    figures.save_figure(fig, "f13_implied_vs_reported_speed")


def context(vehicle_id: int, source_row: int, rule_id: str) -> tuple[pd.DataFrame, int]:
    """The points a rule saw (with the spike for O01, without the spikes for the others)
    and the position of the flagged point in them."""
    points = read_vehicle(vehicle_id)
    frame = points if rule_id == "O01" else without_spikes(points)[0]
    hit = np.flatnonzero(frame["source_row"].to_numpy() == source_row)
    if len(hit) == 0:
        raise ValueError(
            f"source_row {source_row} not among vehicle {vehicle_id}'s points"
        )
    return frame, int(hit[0])


def draw_context(
    ax_track: plt.Axes,
    ax_speed: plt.Axes,
    frame: pd.DataFrame,
    idx: int,
    rule_id: str,
    window: int = 10,
) -> None:
    """Track and speeds of `window` points either side of the flagged point.

    The track is in metres relative to the flagged point (EPSG:32634, no base map) and
    the time in seconds relative to it, so the drawing shows no place and no time of day.
    """
    lo, hi = max(0, idx - window), min(len(frame), idx + window + 1)
    w = frame.iloc[lo:hi]
    k = idx - lo
    x, y = geo.to_metric(w["lon"], w["lat"])
    x, y = x - x[k], y - y[k]
    ax_track.plot(x, y, color=figures.INK_MUTED, linewidth=1, zorder=1)
    ax_track.scatter(x, y, s=12, color=figures.SERIES[0], zorder=2, label="points")
    if rule_id in STEP_RULES and k + 1 < len(w):
        ax_track.plot(
            x[k : k + 2],
            y[k : k + 2],
            color=figures.SERIES[1],
            linewidth=2.5,
            zorder=3,
            label="flagged step",
        )
    ax_track.scatter(
        x[k],
        y[k],
        s=70,
        facecolors="none",
        edgecolors=figures.SERIES[1],
        linewidths=1.5,
        zorder=4,
        label="flagged point" if rule_id not in STEP_RULES else "start of the step",
    )
    ax_track.set_aspect("equal", adjustable="datalim")
    ax_track.set_xlabel("East (m, from the flagged point)")
    ax_track.set_ylabel("North (m, from the flagged point)")
    ax_track.legend(loc="best")

    t = (w["ts_utc"] - w["ts_utc"].iloc[k]).dt.total_seconds()
    ax_speed.plot(
        t,
        w["speed_kmh"],
        marker="o",
        markersize=3,
        color=figures.SERIES[0],
        linewidth=1.2,
        label="reported speed",
    )
    ax_speed.scatter(
        t,
        w["implied_speed_kmh"],
        marker="x",
        color=figures.SERIES[1],
        s=20,
        label="implied speed of the step",
        zorder=3,
    )
    if rule_id in ("O01", "O02"):
        ax_speed.axhline(
            config.JUMP_KMH, color=figures.INK_MUTED, linestyle="--", linewidth=1
        )
    ax_speed.axvline(0, color=figures.INK_MUTED, linewidth=0.8)
    ax_speed.set_yscale("symlog", linthresh=10)
    ax_speed.set_ylim(bottom=0)
    ax_speed.set_xlabel("Time from the flagged point (s)")
    ax_speed.set_ylabel("Speed (km/h, symmetric log)")
    ax_speed.legend(loc="best")


def review_plot(
    vehicle_id: int, source_row: int, rule_id: str, window: int = 10
) -> plt.Figure:
    """The plot reviewers label from. Not saved: review plots stay on the reviewer's machine."""
    fig, (ax_track, ax_speed) = figures.subplots(10, 4.2, ncols=2)
    frame, idx = context(vehicle_id, source_row, rule_id)
    draw_context(ax_track, ax_speed, frame, idx, rule_id, window)
    fig.suptitle(
        f"{rule_id} {RULE_NAMES[rule_id]}", x=0.01, ha="left", fontweight="bold"
    )
    return fig


def render_review(reviewer: str) -> Path:
    """Draw every review plot of a reviewer: their items and the calibration items.

    One PNG per item in REVIEW_PLOTS_DIR/<reviewer>/, named by item number and rule only,
    so the file shows neither the score nor whether the item was flagged. The folder is
    under reports/dev/, which git ignores: review plots stay on the reviewer's machine.
    Reads the full clean points, so `clean` must have run on all vehicles on this machine.
    """
    if reviewer not in config.REVIEWERS:
        raise ValueError(
            f"reviewer must be one of {config.REVIEWERS}, got {reviewer!r}"
        )
    sample = io.read_report_table("outlier_review_sample")
    mine = sample[sample["reviewer"].isin([reviewer, "all"])]
    out = config.REVIEW_PLOTS_DIR / reviewer
    out.mkdir(parents=True, exist_ok=True)
    figures.apply_style()
    for vehicle_id, items in mine.groupby("vehicle_id"):
        points = read_vehicle(int(vehicle_id))
        kept = without_spikes(points)[0]
        for item in items.itertuples():
            frame = points if item.rule_id == "O01" else kept
            idx = int(
                np.flatnonzero(frame["source_row"].to_numpy() == item.source_row)[0]
            )
            fig, (ax_track, ax_speed) = figures.subplots(10, 4.2, ncols=2)
            draw_context(ax_track, ax_speed, frame, idx, item.rule_id)
            fig.suptitle(
                f"Item {item.item}: {item.rule_id} {RULE_NAMES[item.rule_id]}",
                x=0.01,
                ha="left",
                fontweight="bold",
            )
            fig.savefig(out / f"item_{item.item:03d}_{item.rule_id}.png", dpi=110)
            plt.close(fig)
    return out


def figure_examples(sample: pd.DataFrame) -> None:
    """F14: the first flagged item of each rule in the review sample, in context."""
    fig, axes = figures.subplots(16, 7, nrows=2, ncols=len(RULES))
    for j, rule in enumerate(RULES):
        ax_track, ax_speed = axes[0, j], axes[1, j]
        items = sample[(sample["rule_id"] == rule) & (sample["kind"] == "flagged")]
        if items.empty:
            ax_track.set_title(f"{rule} {RULE_NAMES[rule]}: no detections")
            ax_track.axis("off")
            ax_speed.axis("off")
            continue
        item = items.iloc[0]
        frame, idx = context(int(item["vehicle_id"]), int(item["source_row"]), rule)
        draw_context(ax_track, ax_speed, frame, idx, rule)
        unit = {"O01": "km/h", "O02": "km/h", "O03": "m/s²", "O04": "m"}[rule]
        ax_track.set_title(
            f"{rule} {RULE_NAMES[rule]} (score {item['score']:,.1f} {unit})"
        )
    fig.suptitle(
        "F14. One flagged example per rule, 10 points either side, relative position and time",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    figures.save_figure(fig, "f14_point_rule_examples")


def _track(
    lat: list[float],
    lon: list[float],
    t_s: list[float],
    speed: list[float],
    di1: list[bool] | None = None,
) -> dict[str, np.ndarray]:
    """Arrays of a small hand-made track, with steps computed as clean.add_steps does."""
    df = pd.DataFrame(
        {
            "ts_utc": pd.Timestamp("2023-06-01", tz="UTC")
            + pd.to_timedelta(t_s, unit="s"),
            "lat": lat,
            "lon": lon,
            "speed_kmh": speed,
            "di1": pd.array(di1 or [True] * len(lat), dtype="boolean"),
            "ts_has_seconds": pd.array([True] * len(lat), dtype="boolean"),
        }
    )
    clean.add_steps(df)
    return _arrays(df)


def _check() -> None:
    step = 0.001  # about 111 m of latitude
    # O01: the third point jumps 2.2 km away and back within 20 s.
    a = _track(
        [0, step, 20 * step, 2 * step, 3 * step],
        [21.0] * 5,
        [0, 10, 20, 30, 40],
        [40] * 5,
    )
    hit, score, _ = spike_points(a)
    assert hit.tolist() == [False, False, True, False, False], hit
    assert score[2] > config.JUMP_KMH
    # O02: one step of 2.2 km in 10 s that does not come back.
    a = _track([0, step, 21 * step, 22 * step], [21.0] * 4, [0, 10, 20, 30], [40] * 4)
    assert impossible_steps(a)[0].tolist() == [False, True, False, False]
    assert not spike_points(a)[0].any()
    # O03: an event point reads 0 between 56 and 36 km/h, 10 s and 1 s away.
    lat = [0, step, 2 * step, 2.1 * step, 3 * step]
    a = _track(lat, [21.0] * 5, [0, 10, 20, 21, 31], [56, 56, 0, 36, 36])
    assert speed_spikes(a)[0].tolist() == [False, False, True, False, False]
    # O03: of two adjacent candidates (84 then 15 km/h, 1 s apart) only the larger deviation.
    a = _track(lat, [21.0] * 5, [0, 10, 13, 14, 24], [83, 83, 84, 15, 80])
    assert speed_spikes(a)[0].tolist() == [False, False, False, True, False]
    # O03: a steady start from rest is not a spike.
    a = _track(lat, [21.0] * 5, [0, 10, 20, 30, 40], [0, 10, 20, 30, 40])
    assert not speed_spikes(a)[0].any()
    # O04: parked at both ends, 220 m apart within 360 s; the last step drives off.
    parked = [False, False, False, True]
    a = _track(
        [0, 2 * step, 2 * step, 3 * step],
        [21.0] * 4,
        [0, 360, 720, 730],
        [0, 0, 0, 30],
        parked,
    )
    assert parked_drift(a)[0].tolist() == [True, False, False, False]
    print("outlier rule checks passed")


if __name__ == "__main__":
    _check()

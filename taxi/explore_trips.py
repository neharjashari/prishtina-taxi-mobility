"""Trip summaries and multivariate outlier analysis."""

from __future__ import annotations

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, chi2_contingency, spearmanr
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from taxi import PipelineError, config, figures, io, stats

STAGE = "explore-trips"
METHODS = ["robust_mahalanobis", "lof", "isolation_forest", "pca_reconstruction"]


def _kept_columns(table: str, frame: pd.DataFrame) -> list[str]:
    path = config.tables_dir() / "feature_selection.csv"
    if not path.exists():
        raise PipelineError("feature_selection.csv is missing; run reduce first.")
    selection = pd.read_csv(path)
    kept = selection.loc[(selection["table"] == table) & (selection["decision"] == "kept"), "feature"].tolist()
    return [c for c in kept if f"{c}_z" in frame and frame[f"{c}_z"].notna().any()]


def _bh_flags(scores: np.ndarray, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    p = chi2.sf(np.maximum(scores, 0), n_features)
    q = stats.bh_adjust(p)
    return q <= config.FDR_Q, q


def detect(frame: pd.DataFrame, table: str) -> tuple[pd.DataFrame, dict]:
    """Score rows with robust distance, LOF, isolation forest, and PCA reconstruction."""
    columns = _kept_columns(table, frame)
    if not columns:
        raise PipelineError(f"No selected features available for {table}.")
    zcols = [f"{c}_z" for c in columns]
    complete = frame[zcols].notna().all(axis=1).to_numpy()
    x = frame.loc[complete, zcols].astype("float64").to_numpy()
    if len(x) <= len(columns) + 1:
        raise PipelineError(f"Too few complete rows for {table}: {len(x)}.")

    rng = np.random.default_rng(config.SEED)
    fit_n = len(x)
    fit = x
    valid_rows = np.flatnonzero(complete)

    robust = MinCovDet(random_state=config.SEED).fit(fit)
    robust_score = robust.mahalanobis(x)
    robust_flag, adjusted = _bh_flags(robust_score, len(columns))

    center = fit.mean(axis=0)
    covariance = np.cov(fit, rowvar=False)
    centered = x - center
    classical_score = np.einsum("ij,jk,ik->i", centered, np.linalg.pinv(covariance), centered)
    classical_flag, _ = _bh_flags(classical_score, len(columns))
    classical_full = np.zeros(len(frame), dtype=bool)
    classical_full[valid_rows] = classical_flag

    lof_n = min(len(x), config.LOF_MAX_N)
    lof_idx = np.sort(rng.choice(len(x), lof_n, replace=False)) if lof_n < len(x) else np.arange(len(x))
    lof = LocalOutlierFactor(n_neighbors=min(config.LOF_K, max(2, lof_n - 1)))
    lof.fit_predict(x[lof_idx])
    lof_score = -lof.negative_outlier_factor_
    lof_z = stats.robust_z(lof_score)
    lof_flag = np.isfinite(lof_z) & (lof_z > config.ROBUST_Z_MAX)

    forest = IsolationForest(n_estimators=config.IFOREST_TREES, random_state=config.SEED, n_jobs=-1).fit(fit)
    if_score = -forest.score_samples(x)
    if_z = stats.robust_z(if_score)
    if_flag = np.isfinite(if_z) & (if_z > config.ROBUST_Z_MAX)

    pca = PCA(n_components=min(5, x.shape[1]), svd_solver="full").fit(fit)
    reconstructed = pca.inverse_transform(pca.transform(x))
    pca_score = np.mean((x - reconstructed) ** 2, axis=1)
    pca_z = stats.robust_z(pca_score)
    pca_flag = np.isfinite(pca_z) & (pca_z > config.ROBUST_Z_MAX)

    score_values = np.full((len(frame), len(METHODS)), np.nan)
    flag_values = np.zeros((len(frame), len(METHODS)), dtype=bool)
    for j, (scores, flags) in enumerate(zip([robust_score, lof_score, if_score, pca_score], [robust_flag, lof_flag, if_flag, pca_flag], strict=True)):
        positions = valid_rows[lof_idx] if j == 1 else valid_rows
        score_values[positions, j] = scores
        flag_values[positions, j] = flags
    result = frame.reset_index(drop=True).copy()
    for j, method in enumerate(METHODS):
        result[f"score_{method}"] = score_values[:, j]
        result[f"flag_{method}"] = flag_values[:, j]
    result["methods_flagged"] = flag_values.sum(axis=1).astype("int8")
    result["class"] = np.where(complete, "unreviewed", "low_coverage")
    return result, {
        "features": columns,
        "complete_rows": int(complete.sum()),
        "fit_rows": int(fit_n),
        "lof_rows": int(lof_n),
        "adjusted_p": adjusted,
        "classical_flags": classical_full,
    }


def _comparison_rows(results: dict[str, pd.DataFrame], metadata: dict[str, dict]) -> list[dict]:
    rows = []
    for table, frame in results.items():
        meta = metadata[table]
        for method in METHODS:
            score = frame[f"score_{method}"].to_numpy(dtype="float64")
            flags = frame[f"flag_{method}"].to_numpy(dtype=bool)
            rows.append({"dataset": table, "row_type": "method", "method": method, "method_b": "", "rows": int(np.isfinite(score).sum()), "flagged": int(flags.sum()), "fit_rows": meta["fit_rows"], "lof_rows": meta["lof_rows"] if method == "lof" else pd.NA, "score_correlation": np.nan, "jaccard": np.nan, "classical_only_misses": pd.NA})
        robust = frame["flag_robust_mahalanobis"].to_numpy(dtype=bool)
        classical = meta["classical_flags"]
        rows.append({"dataset": table, "row_type": "classical_comparison", "method": "classical_mahalanobis", "method_b": "robust_mahalanobis", "rows": len(frame), "flagged": int(classical.sum()), "fit_rows": meta["fit_rows"], "lof_rows": pd.NA, "score_correlation": np.nan, "jaccard": np.nan, "classical_only_misses": int((robust & ~classical).sum())})
        for a, b in combinations(METHODS, 2):
            fa, fb = frame[f"flag_{a}"].to_numpy(dtype=bool), frame[f"flag_{b}"].to_numpy(dtype=bool)
            sa, sb = frame[f"score_{a}"].to_numpy(dtype="float64"), frame[f"score_{b}"].to_numpy(dtype="float64")
            valid = np.isfinite(sa) & np.isfinite(sb)
            union = np.sum(fa | fb)
            rho = spearmanr(sa[valid], sb[valid]).statistic if valid.sum() > 2 else np.nan
            rows.append({"dataset": table, "row_type": "method_overlap", "method": a, "method_b": b, "rows": int(valid.sum()), "flagged": pd.NA, "fit_rows": meta["fit_rows"], "lof_rows": pd.NA, "score_correlation": rho, "jaccard": float(np.sum(fa & fb) / union) if union else np.nan, "classical_only_misses": pd.NA})
    return rows


def _top_anomalies(frame: pd.DataFrame, table: str) -> pd.DataFrame:
    top = frame[frame["methods_flagged"] > 0].nlargest(20, "methods_flagged")
    rows = []
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        rows.append({"dataset": table, "index": rank, "trip_id": row.get("trip_id", ""), "vehicle_id": row.get("vehicle_id", pd.NA), "date_local": str(row.get("date_local", "")), "methods_flagged": int(row["methods_flagged"]), "robust_mahalanobis": bool(row["flag_robust_mahalanobis"]), "lof": bool(row["flag_lof"]), "isolation_forest": bool(row["flag_isolation_forest"]), "pca_reconstruction": bool(row["flag_pca_reconstruction"]), "class": row["class"], "review_note": ""})
    return pd.DataFrame(rows)


def _trip_summary(trips: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    center = model.set_index("trip_id")[["start_dist_center_m", "end_dist_center_m"]]
    t = trips.copy()
    for col in center:
        t[col] = t["trip_id"].map(center[col])
    measures = ["duration_s", "distance_m", "avg_speed_kmh", "max_speed_kmh", "straightness", "start_dist_center_m", "end_dist_center_m"]
    rows = []
    def emit(part: pd.DataFrame, group: str, value: str, cls: str, columns: list[str]) -> None:
        for col in columns:
            item = {"table": "trips", "variable": col, "group": group, "group_value": value, "trip_set": cls}
            item.update(stats.summarize(part[col]))
            rows.append(item)
    for cls, subset in [("ride", t[t["trip_class"].eq("ride")]), ("stationary", t[t["trip_class"].eq("stationary")])]:
        columns = measures if cls == "ride" else measures[:-2]
        emit(subset, "all", "all", cls, columns)
    rides = t[t["trip_class"].eq("ride")]
    for col, group in [("tod_bucket", "tod_bucket"), ("is_weekend", "day_type"), ("start_municipality", "start_municipality")]:
        counts = rides[col].value_counts(dropna=True)
        values = counts.index[:5] if group == "start_municipality" else counts.index
        for value in values:
            emit(rides[rides[col].eq(value)], group, str(value), "ride", measures)
    return pd.DataFrame(rows)


def _trip_rules(trips: pd.DataFrame, baseline: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    for label, part in [("before_screening", baseline), ("after_screening", trips)]:
        if part is None:
            continue
        current = part.copy()
        if "trip_class" not in current:
            current["trip_class"] = np.where(current["distance_m"] < config.STATIONARY_MAX_M, "stationary", "ride")
        for cls, group in current.groupby("trip_class", observed=False):
            rows.append({"dataset": label, "row_type": "class", "rule": "trip_class", "value": str(cls), "count": len(group), "median_duration_s": group["duration_s"].median(), "median_distance_m": group["distance_m"].median(), "median_speed_kmh": group["avg_speed_kmh"].median(), "review_status": "descriptive"})
        checks = [
            ("too_fast", current["avg_speed_kmh"] > config.TRIP_MAX_AVG_KMH, "Review every remaining trip; trace errors to source points."),
            ("over_100_km", current["distance_m"] > config.TRIP_LONG_DISTANCE_M, "Long trips are retained and described."),
            ("over_3_hours", current["duration_s"] > config.TRIP_LONG_DURATION_S, "Inspect the longest meter-on episodes."),
        ]
        for rule, mask, note in checks:
            rows.append({"dataset": label, "row_type": "rule", "rule": rule, "value": note, "count": int(mask.sum()), "median_duration_s": np.nan, "median_distance_m": np.nan, "median_speed_kmh": np.nan, "review_status": "review_required" if rule == "too_fast" else "descriptive"})
        rides = current["trip_class"].eq("ride")
        rows.append({"dataset": label, "row_type": "model_inclusion", "rule": "ride_only_model", "value": "stationary trips omitted from model tables", "count": int((~rides).sum()), "median_duration_s": np.nan, "median_distance_m": np.nan, "median_speed_kmh": np.nan, "review_status": "descriptive"})
    return pd.DataFrame(rows)


def _trip_figure(trips: pd.DataFrame) -> None:
    fig, ax = figures.subplots(11, 6)
    positive = trips[(trips["distance_m"] > 0) & (trips["duration_s"] > 0)]
    for cls, color in zip(["ride", "stationary"], figures.SERIES, strict=True):
        part = positive[positive["trip_class"].eq(cls)]
        ax.scatter(part["duration_s"], part["distance_m"], s=7, alpha=.35, label=cls, color=color)
    flagged = positive[(positive["avg_speed_kmh"] > config.TRIP_MAX_AVG_KMH) | (positive["distance_m"] > config.TRIP_LONG_DISTANCE_M) | (positive["duration_s"] > config.TRIP_LONG_DURATION_S)]
    if len(flagged):
        ax.scatter(flagged["duration_s"], flagged["distance_m"], marker="x", s=20, color="#d62728", label="trip rule")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("duration (s, log scale)"); ax.set_ylabel("distance (m, log scale)"); ax.legend()
    ax.set_title("F19. Trip distance and duration")
    figures.save_figure(fig, "f19_trip_classes")


def _outlier_figure(frame: pd.DataFrame) -> None:
    part = frame.dropna(subset=["pc1", "pc2"])
    fig, ax = figures.subplots(10, 7)
    scatter = ax.scatter(part["pc1"], part["pc2"], c=part["methods_flagged"].clip(upper=4), cmap="Blues", s=13, alpha=.65)
    for rank, (_, row) in enumerate(part[part["methods_flagged"] > 0].nlargest(20, "methods_flagged").iterrows(), 1):
        ax.annotate(str(rank), (row["pc1"], row["pc2"]), fontsize=7)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("F20. Vehicle-day anomalies by number of methods")
    fig.colorbar(scatter, ax=ax, label="methods flagging row (capped at 4)")
    figures.save_figure(fig, "f20_multivariate_outliers")


def _feature_matrices(table: str, frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    x = frame[columns].astype("float64")
    rows = []
    for method in ("pearson", "spearman", "covariance"):
        matrix = x.cov() if method == "covariance" else x.corr(method=method)
        for a in columns:
            for b in columns:
                rows.append({"section": "feature_matrix", "dataset": table, "statistic": method, "feature": a, "feature_b": b, "group": "all", "group_value": "all", "value": matrix.loc[a, b], "n": int(x[[a, b]].dropna().shape[0])})
    return rows


def _multivariate_summary(trips: pd.DataFrame, trip_model: pd.DataFrame, vd_model: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for table, frame in [("trips_model", trip_model), ("vehicle_day_model", vd_model)]:
        columns = _kept_columns(table, frame)
        rows += _feature_matrices(table, frame, columns)
        if table == "vehicle_day_model":
            z = frame[[f"{c}_z" for c in columns]].dropna()
            pca = PCA(n_components=min(5, len(columns), len(z) - 1), svd_solver="full").fit(z)
            for i, ratio in enumerate(pca.explained_variance_ratio_, 1):
                rows.append({"section": "pca_after_screening", "dataset": table, "statistic": "explained_variance_ratio", "feature": f"pc{i}", "feature_b": "", "group": "all", "group_value": "all", "value": float(ratio), "n": len(z)})
            components = pca.components_.copy()
            signs = np.sign(components[np.arange(len(components)), np.abs(components).argmax(axis=1)])
            components *= signs[:, None]
            for i, component in enumerate(components, 1):
                for feature, loading in zip(columns, component, strict=True):
                    rows.append({"section": "pca_after_screening", "dataset": table, "statistic": "loading", "feature": feature, "feature_b": f"pc{i}", "group": "all", "group_value": "all", "value": float(loading), "n": len(z)})
            variance_reference_path = config.tables_dir() / "pca_explained_variance_phase1.csv"
            if variance_reference_path.exists():
                variance_reference = pd.read_csv(variance_reference_path)
                variance_reference = variance_reference[variance_reference["scaling"] == "zscore"].set_index("component")
                for i, ratio in enumerate(pca.explained_variance_ratio_, 1):
                    name = f"pc{i}"
                    if name in variance_reference.index:
                        rows.append({"section": "pca_variance_change", "dataset": table, "statistic": "current_minus_phase1", "feature": name, "feature_b": "", "group": "all", "group_value": "all", "value": float(ratio - variance_reference.loc[name, "explained_variance_ratio"]), "n": len(z)})
            reference_path = config.tables_dir() / "pca_loadings_phase1.csv"
            if reference_path.exists():
                reference = pd.read_csv(reference_path).set_index("feature")
                for i, component in enumerate(components, 1):
                    name = f"pc{i}"
                    if name not in reference:
                        continue
                    for feature, loading in zip(columns, component, strict=True):
                        if feature in reference.index:
                            rows.append({"section": "pca_loading_change", "dataset": table, "statistic": "current_minus_phase1", "feature": feature, "feature_b": name, "group": "all", "group_value": "all", "value": float(loading - reference.loc[feature, name]), "n": len(z)})
    joined = trips[trips["trip_class"].eq("ride")].merge(trip_model, on=["trip_id", "vehicle_id"], how="inner", suffixes=("", "_model"))
    measures = [c for c in ["duration_s", "distance_m", "avg_speed_kmh", "max_speed_kmh", "straightness"] if c in joined]
    joined["day_type"] = np.where(joined["is_weekend"], "weekend", "weekday")
    for group in ["tod_bucket", "day_type"]:
        for value, part in joined.groupby(group, dropna=False, observed=False):
            x = part[measures].astype("float64")
            for col in measures:
                rows.append({"section": "group_mean", "dataset": "trips_model", "statistic": "mean", "feature": col, "feature_b": "", "group": group, "group_value": str(value), "value": x[col].mean(), "n": int(x[col].notna().sum())})
            cov = x.cov()
            for a in measures:
                for b in measures:
                    rows.append({"section": "group_covariance", "dataset": "trips_model", "statistic": "covariance", "feature": a, "feature_b": b, "group": group, "group_value": str(value), "value": cov.loc[a, b], "n": int(x[[a, b]].dropna().shape[0])})
    t = trips[trips["trip_class"].eq("ride")].copy()
    municipalities = t["start_municipality"].value_counts().head(5).index
    t["origin"] = t["start_municipality"].where(t["start_municipality"].isin(municipalities), "other").fillna("other")
    t["destination"] = t["end_municipality"].where(t["end_municipality"].isin(municipalities), "other").fillna("other")
    matrices = []
    for name, a, b in [("origin_destination", "origin", "destination"), ("time_day_type", "tod_bucket", "is_weekend")]:
        cross = pd.crosstab(t[a], t[b], dropna=False)
        statistic, p, dof, _ = chi2_contingency(cross)
        n = int(cross.to_numpy().sum()); denom = min(cross.shape[0]-1, cross.shape[1]-1)
        cramers = float(np.sqrt(statistic / (n * denom))) if n and denom > 0 else np.nan
        rows.append({"section": "contingency_test", "dataset": "trips", "statistic": "chi_square", "feature": a, "feature_b": b, "group": name, "group_value": "all", "value": float(statistic), "p_value": float(p), "degrees_freedom": int(dof), "cramers_v": cramers, "n": n})
        for origin in cross.index:
            for destination in cross.columns:
                matrices.append({"matrix": name, "origin": str(origin), "destination": str(destination), "count": int(cross.loc[origin, destination])})
    return pd.DataFrame(rows), pd.DataFrame(matrices)


def _correlation_figure(summary: pd.DataFrame, od: pd.DataFrame) -> None:
    corr = summary[(summary.section == "feature_matrix") & (summary.dataset == "vehicle_day_model") & (summary.statistic == "spearman")]
    matrix = corr.pivot(index="feature", columns="feature_b", values="value")
    od_matrix = od[od.matrix == "origin_destination"].pivot(index="origin", columns="destination", values="count").fillna(0)
    fig, axes = figures.subplots(14, 6, nrows=1, ncols=2)
    im = axes[0].imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    axes[0].set_xticks(range(len(matrix.columns)), matrix.columns, rotation=90); axes[0].set_yticks(range(len(matrix.index)), matrix.index); axes[0].set_title("Spearman correlations")
    fig.colorbar(im, ax=axes[0], fraction=.046)
    im2 = axes[1].imshow(od_matrix, cmap="Blues", aspect="auto")
    axes[1].set_xticks(range(len(od_matrix.columns)), od_matrix.columns, rotation=90); axes[1].set_yticks(range(len(od_matrix.index)), od_matrix.index); axes[1].set_title("Trips between municipalities")
    fig.colorbar(im2, ax=axes[1], fraction=.046, label="trips")
    fig.suptitle("F21. Correlation and municipality movement")
    figures.save_figure(fig, "f21_correlation_od_matrix")


def run(dev: bool) -> None:
    """Write trip summaries, detector comparisons, multivariate tables, and figures."""
    paths = [io.table_path("trips"), io.table_path("model/trips_model"), io.table_path("model/vehicle_day_model")]
    if any(not p.exists() for p in paths):
        raise PipelineError("Run trips, transform, and reduce before explore-trips.")
    trips, trip_model, vd_model = io.read_table("trips"), io.read_table("model/trips_model"), io.read_table("model/vehicle_day_model")
    baseline_path = config.processed_dir() / "trips_phase1.parquet"
    baseline = pd.read_parquet(baseline_path) if baseline_path.exists() else None
    io.write_report_table(_trip_rules(trips, baseline), "outliers_trips")
    io.write_report_table(_trip_summary(trips, trip_model), "summary_trips")
    _trip_figure(trips)

    results, metadata = {}, {}
    for name, frame in [("vehicle_day_model", vd_model), ("trips_model", trip_model)]:
        results[name], metadata[name] = detect(frame, name)
    rows = _comparison_rows(results, metadata)
    rows.extend(_top_anomalies(results["vehicle_day_model"], "vehicle_day_model").to_dict("records"))
    rows.extend(_top_anomalies(results["trips_model"], "trips_model").to_dict("records"))
    io.write_report_table(pd.DataFrame(rows), "outliers_multivariate")
    _outlier_figure(results["vehicle_day_model"])

    summary, od = _multivariate_summary(trips, trip_model, vd_model)
    io.write_report_table(summary, "multivariate_summary")
    io.write_report_table(od, "od_matrix")
    _correlation_figure(summary, od)
    for table, meta in metadata.items():
        print(f"{table}: {meta['complete_rows']:,} complete rows; LOF scored {meta['lof_rows']:,}; top cases await review")

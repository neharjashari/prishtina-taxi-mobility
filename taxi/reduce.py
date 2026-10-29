"""Feature subset selection and dimensionality reduction."""

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA

from taxi import PipelineError, config, features, figures, io, transform

STAGE = "reduce"
TABLES = {
    "vehicle_day_model": transform.VEHICLE_DAY_MODEL,
    "trips_model": transform.TRIPS_MODEL,
}
DATASETS = {
    "vehicle_day_model": transform.VEHICLE_DAY_MODEL_DATASET,
    "trips_model": transform.TRIPS_MODEL_DATASET,
}

INTERPRETABILITY = {
    "vehicle_day_model": [
        "km",
        "moving_h",
        "moving_share",
        "n_trips",
        "trip_median_distance_m",
        "trip_median_duration_s",
        "avg_speed_moving_kmh",
        "max_speed_kmh",
        "di1_share",
        "di2_share",
        "night_trip_share",
        "is_weekend",
        "is_holiday",
        "di3_reliable",
        "di3_share",
        "observed_h",
        "n_gaps",
        "n_points",
    ],
    "trips_model": [
        "distance_m",
        "duration_s",
        "avg_speed_kmh",
        "straight_m",
        "straightness",
        "max_speed_kmh",
        "start_dist_center_m",
        "end_dist_center_m",
        "start_hour_sin",
        "start_hour_cos",
        "di2_share",
        "is_weekend",
        "is_holiday",
        "n_points",
    ],
}
UNRELIABLE = {
    "di3_share": "unreliable: di3_share is NA for vehicles where Di3 is stuck or reliability is unknown",
}


def _base(column: str) -> str:
    return transform.base_name(column)[0]


def _priority(table: str, column: str) -> int:
    order = INTERPRETABILITY[table]
    base = _base(column)
    return order.index(base) if base in order else len(order)


def _purpose(table: str, column: str) -> str:
    """The purpose of a feature, as in the catalogue; di2_share of a trip depends on the trip rule."""
    if (
        table == "trips_model"
        and _base(column) == "di2_share"
        and config.TRIP_RULE == "di2"
    ):
        return (
            "none: with the di2 trip rule every trip is a taximeter episode, so the share "
            "only measures short signal dropouts"
        )
    return transform.MODEL_SPEC[table][_base(column)][3]


def _serves_question(table: str, column: str) -> bool:
    return any(q in _purpose(table, column) for q in features.QUESTIONS)


def correlations(
    df: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pearson and Spearman matrices over pairwise complete rows."""
    x = df[columns].astype("float64")
    return x.corr(method="pearson"), x.corr(method="spearman")


def select(df: pd.DataFrame, table: str) -> tuple[list[str], pd.DataFrame]:
    """The kept features of one model table and one T11 row per candidate."""
    rows = [
        {
            "table": table,
            "feature": c,
            "step": 1,
            "decision": "dropped",
            "reason": "identifier or timestamp, not a feature",
        }
        for c in transform.ID_COLUMNS[table]
    ]
    rows += [
        {
            "table": table,
            "feature": c,
            "step": 4,
            "decision": "dropped",
            "reason": (
                "one-hot column of tod_bucket at the start: the same start time as "
                "start_hour_sin and start_hour_cos, in 5 bins; the cyclic pair keeps the "
                "order and puts 23:59 next to 00:00. Stays in the table for methods on "
                "categories"
            ),
        }
        for c in df.columns
        if c.startswith("tod_")
    ]
    candidates = transform.feature_columns(df, table)
    stats = {}
    for c in candidates:
        x = df[c].astype("float64").dropna()
        top = float(x.value_counts(normalize=True).iloc[0]) if len(x) else 1.0
        binary = bool(len(x)) and set(x.unique()) <= {0.0, 1.0}
        stats[c] = {
            "top_value_share": round(top, 4),
            "variance_unscaled": float(x.var(ddof=0)) if len(x) else 0.0,
            "binary": binary,
            "n": len(x),
        }

    kept, info = [], {}
    for c in candidates:
        s = stats[c]
        reason = None
        if _base(c) in UNRELIABLE:
            reason = UNRELIABLE[_base(c)]
        elif s["n"] == 0 or s["variance_unscaled"] <= 1e-12 * max(
            1.0, float(df[c].abs().max())
        ):
            reason = "near-constant: unscaled variance is essentially zero"
        elif s["binary"] and s["variance_unscaled"] < config.BINARY_MIN_VARIANCE:
            reason = (
                f"near-constant 0/1 feature: p(1 - p) = {s['variance_unscaled']:.4f} "
                f"< {config.BINARY_MIN_VARIANCE}"
            )
        elif s["top_value_share"] >= config.NEAR_CONSTANT_SHARE:
            reason = (
                f"near-constant: one value covers {s['top_value_share']:.1%} of the rows "
                f"(>= {config.NEAR_CONSTANT_SHARE:.0%})"
            )
        if reason:
            info[c] = (2, reason)
        else:
            kept.append(c)

    for c in kept:
        if not _serves_question(table, c):
            purpose = _purpose(table, c).removeprefix("none: ").removeprefix("none ")
            info[c] = (3, f"serves no guiding question: {purpose}")
    candidates_4 = [c for c in kept if c not in info]

    pearson, spearman = correlations(df, kept)
    stayed: list[str] = []
    for c in sorted(candidates_4, key=lambda c: (_priority(table, c), kept.index(c))):
        partners = []
        for k in stayed:
            rp, rs = pearson.loc[c, k], spearman.loc[c, k]
            strength = np.nanmax([abs(rp), abs(rs)])
            if np.isfinite(strength) and strength >= config.CORR_MAX_ABS:
                partners.append((strength, k, rp, rs))
        if partners:
            _, k, rp, rs = max(partners, key=lambda p: p[0])
            info[c] = (
                4,
                (
                    f"redundant with {k} (Pearson {rp:+.2f}, Spearman {rs:+.2f}); "
                    f"{k} is easier to interpret"
                ),
            )
        else:
            stayed.append(c)

    kept_final = [c for c in kept if c not in info]
    for c in candidates:
        others = [o for o in kept if o != c] if c in pearson.index else []
        if others:
            r = pd.concat(
                [pearson.loc[c, others].abs(), spearman.loc[c, others].abs()], axis=1
            ).max(axis=1)
            max_r, with_ = round(float(r.max()), 3), str(r.idxmax())
        else:
            max_r, with_ = np.nan, ""
        step, reason = info.get(
            c, (4, "kept: informative, not redundant, serves a guiding question")
        )
        rows.append(
            {
                "table": table,
                "feature": c,
                "step": step,
                "decision": "kept" if c in kept_final else "dropped",
                "reason": reason,
                "top_value_share": stats[c]["top_value_share"],
                "variance_unscaled": stats[c]["variance_unscaled"],
                "max_abs_r": max_r,
                "most_correlated_with": with_,
            }
        )
    return kept_final, pd.DataFrame(rows)


def elbow(ratios: np.ndarray) -> int:
    """Component count at the elbow of the scree curve: the point farthest from the chord."""
    n = len(ratios)
    if n < 3:
        return n
    x = np.arange(1, n + 1, dtype="float64")
    x1, y1, x2, y2 = x[0], ratios[0], x[-1], ratios[-1]
    dist = np.abs((y2 - y1) * x - (x2 - x1) * ratios + x2 * y1 - y2 * x1)
    return int(np.argmax(dist)) + 1


def run_pca(df: pd.DataFrame, kept: list[str], scaling_params: pd.DataFrame) -> dict:
    """PCA on the z-scores of the kept features; the same on min-max and unscaled for comparison."""
    complete = df[kept].notna().all(axis=1)
    if complete.sum() <= len(kept):
        raise PipelineError(
            f"PCA needs more complete rows than features: {int(complete.sum())} rows, {len(kept)} features."
        )
    params = scaling_params.set_index("feature")
    raw = df.loc[complete, kept].astype("float64")
    span = (params.loc[kept, "max"] - params.loc[kept, "min"]).replace(0, np.nan)
    inputs = {
        "zscore": df.loc[complete, [f"{c}_z" for c in kept]]
        .astype("float64")
        .to_numpy(),
        "minmax": ((raw - params.loc[kept, "min"]) / span).to_numpy(),
        "none": raw.to_numpy(),
    }
    out = {"complete": complete, "rows": int(complete.sum())}
    for scaling, x in inputs.items():
        pca = PCA(svd_solver="full", random_state=config.SEED).fit(x)
        out[scaling] = pca
    ratios = out["zscore"].explained_variance_ratio_
    cumulative = np.cumsum(ratios)
    out["k"] = int(np.searchsorted(cumulative, config.PCA_VARIANCE_TARGET - 1e-12) + 1)
    out["elbow"] = elbow(ratios)
    return out


def pca_tables(result: dict, kept: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for scaling in ("zscore", "minmax", "none"):
        pca = result[scaling]
        cum = np.cumsum(pca.explained_variance_ratio_)
        for i, (var, ratio) in enumerate(
            zip(pca.explained_variance_, pca.explained_variance_ratio_, strict=True)
        ):
            rows.append(
                {
                    "scaling": scaling,
                    "component": f"pc{i + 1}",
                    "explained_variance": round(float(var), 6),
                    "explained_variance_ratio": round(float(ratio), 6),
                    "cumulative_ratio": round(float(cum[i]), 6),
                    "kept": scaling == "zscore" and i < result["k"],
                    "elbow": scaling == "zscore" and i + 1 == result["elbow"],
                    "rows": result["rows"],
                }
            )
    components = oriented_components(result["zscore"])
    loadings = pd.DataFrame(
        components.T,
        index=kept,
        columns=[f"pc{i + 1}" for i in range(components.shape[0])],
    ).round(6)
    loadings.index.name = "feature"
    return pd.DataFrame(rows), loadings.reset_index()


def component_signs(pca: PCA) -> np.ndarray:
    """+1 or -1 per component, so that its largest absolute weight is positive (a fixed sign)."""
    comp = pca.components_
    return np.sign(comp[np.arange(len(comp)), np.abs(comp).argmax(axis=1)])


def oriented_components(pca: PCA) -> np.ndarray:
    """PCA weights with the sign of component_signs."""
    return pca.components_ * component_signs(pca)[:, None]


def figure_correlation(pearson: pd.DataFrame, spearman: pd.DataFrame) -> None:
    """F11: Pearson and Spearman correlation of the candidate vehicle-day features."""
    cmap = LinearSegmentedColormap.from_list(
        "diverging", [figures.SERIES[0], figures.SURFACE, figures.SERIES[1]]
    )
    n = len(pearson)
    fig, axes = figures.subplots(14, min(7.5, 0.3 * n + 2.4), ncols=2)
    for ax, matrix, title in [
        (axes[0], pearson, "Pearson r"),
        (axes[1], spearman, "Spearman rho"),
    ]:
        im = ax.imshow(matrix.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
        ax.set_xticks(range(n), matrix.columns, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(n), matrix.index, fontsize=7)
        ax.grid(False)
        for i in range(n):
            for j in range(n):
                v = matrix.iat[i, j]
                if np.isfinite(v) and i != j and abs(v) >= config.CORR_MAX_ABS:
                    ax.text(
                        j,
                        i,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color=figures.INK,
                    )
        ax.set_title(f"{title} (|r| >= {config.CORR_MAX_ABS} labelled)")
    fig.colorbar(im, ax=axes, shrink=0.6, label="correlation")
    fig.suptitle(
        "F11. Correlation of the candidate vehicle-day features",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=11,
    )
    figures.save_figure(fig, "f11_feature_correlation")


def figure_scree(variance: pd.DataFrame, loadings: pd.DataFrame, k: int) -> None:
    """F12: scree plot with the cumulative share, the PC1 and PC2 weights, and the effect of scaling."""
    z = variance[variance["scaling"] == "zscore"]
    n = len(z)
    fig, (ax, ax_w, ax_s) = figures.subplots(
        14, 4.6, ncols=3, width_ratios=[1.4, 1.2, 0.8]
    )
    x = np.arange(1, n + 1)
    ax.bar(
        x,
        z["explained_variance_ratio"],
        color=[figures.SERIES[0] if i <= k else figures.INK_MUTED for i in x],
    )
    ax.plot(
        x,
        z["cumulative_ratio"],
        color=figures.SERIES[1],
        marker="o",
        markersize=4,
        label="cumulative",
    )
    ax.axhline(
        config.PCA_VARIANCE_TARGET,
        color=figures.INK_SECONDARY,
        linewidth=1,
        linestyle="--",
    )
    ax.text(
        n,
        config.PCA_VARIANCE_TARGET,
        f"{config.PCA_VARIANCE_TARGET:.0%}",
        va="bottom",
        ha="right",
        fontsize=7,
    )
    ax.set_xticks(x)
    ax.set_xlabel("principal component")
    ax.set_ylabel("share of variance")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Scree plot: {k} components kept")
    ax.legend(loc="center right")

    w = (
        loadings.set_index("feature")[["pc1", "pc2"]]
        if "pc2" in loadings
        else loadings.set_index("feature")[["pc1"]]
    )
    pos = np.arange(len(w))
    height = 0.4
    for i, (col, color) in enumerate(zip(w.columns, figures.SERIES, strict=False)):
        ax_w.barh(
            pos + (i - 0.5) * height, w[col], height=height, color=color, label=col
        )
    ax_w.set_yticks(pos, w.index, fontsize=7)
    ax_w.invert_yaxis()
    ax_w.axvline(0, color=figures.AXIS, linewidth=0.8)
    ax_w.set_xlabel("weight in the component")
    ax_w.set_title("PC1 and PC2 weights")
    ax_w.legend(loc="lower right")

    first = variance[variance["component"] == "pc1"].set_index("scaling")[
        "explained_variance_ratio"
    ]
    order = ["none", "minmax", "zscore"]
    ax_s.bar(
        range(3),
        [first[s] for s in order],
        color=[figures.INK_MUTED, figures.SERIES[2], figures.SERIES[0]],
    )
    ax_s.set_xticks(range(3), ["unscaled", "min-max", "z-score"])
    ax_s.set_ylim(0, 1.02)
    ax_s.set_ylabel("share of variance in PC1")
    ax_s.set_title("Effect of scaling")
    fig.suptitle(
        "F12. PCA of the vehicle-day features",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=11,
    )
    figures.save_figure(fig, "f12_pca_scree")


def run(dev: bool) -> None:
    """Feature selection (T11, F11) on both model tables; PCA (T12, F12) on vehicle_day_model."""
    for name in TABLES.values():
        if not io.table_path(name).exists():
            raise PipelineError(f"No {name}.parquet: run the transform stage first.")
    scaling = io.read_report_table("scaling_params")

    selections, kept_by_table = [], {}
    for table, name in TABLES.items():
        df = io.read_table(name)
        df = df[[c for c in df.columns if not transform.PC_COLUMN.match(c)]]
        kept, rows = select(df, table)
        selections.append(rows)
        kept_by_table[table] = (df, kept)
    io.write_report_table(pd.concat(selections, ignore_index=True), "feature_selection")

    table = "vehicle_day_model"
    df, kept = kept_by_table[table]
    candidates = [
        c
        for c in transform.feature_columns(df, table)
        if df[c].astype("float64").var() > 0
    ]
    pearson, spearman = correlations(df, candidates)
    figure_correlation(pearson, spearman)

    params = scaling[scaling["table"] == table]
    result = run_pca(df, kept, params)
    variance, loadings = pca_tables(result, kept)
    io.write_report_table(variance, "pca_explained_variance")
    io.write_report_table(loadings, "pca_loadings")
    figure_scree(variance, loadings, result["k"])

    k = result["k"]
    pca = result["zscore"]
    x = (
        df.loc[result["complete"], [f"{c}_z" for c in kept]]
        .astype("float64")
        .to_numpy()
    )
    scores = pca.transform(x) * component_signs(pca)
    for i in range(k):
        col = np.full(len(df), np.nan, dtype="float32")
        col[result["complete"].to_numpy()] = scores[:, i]
        df[f"pc{i + 1}"] = col
    io.write_table(df, TABLES[table])

    dictionary, catalog = transform.model_dictionary(df, table)
    cum = variance.loc[variance["scaling"] == "zscore", "cumulative_ratio"].to_numpy()
    ratio = variance.loc[
        variance["scaling"] == "zscore", "explained_variance_ratio"
    ].to_numpy()
    for i in range(k):
        top = loadings.set_index("feature")[f"pc{i + 1}"]
        top = top.reindex(top.abs().sort_values(ascending=False).index)[:3]
        weights = ", ".join(f"{f} {w:+.2f}" for f, w in top.items())
        dictionary.append(
            {
                "column": f"pc{i + 1}",
                "unit": "",
                "attribute_type": "interval",
                "description": f"Score on principal component {i + 1} of the z-scored kept features "
                f"({ratio[i]:.1%} of the variance, {cum[i]:.1%} cumulative); largest weights: {weights}. "
                "NA when a kept feature is NA.",
                "source": "pca_loadings.csv",
            }
        )
        catalog.append(
            {
                "feature": f"pc{i + 1}",
                "formula": f"z-scored kept features x weights of component {i + 1}",
                "attribute_type": "interval",
                "purpose": "Phases II and III: compact summary of the vehicle-day (Q1, Q3)",
            }
        )
    io.write_dictionary(DATASETS[table], dictionary, df, owner=STAGE)
    features.write_catalog(DATASETS[table], catalog, owner=STAGE)

    print(
        f"kept {len(kept)} of {len(transform.feature_columns(df, table))} vehicle-day features "
        f"and {len(kept_by_table['trips_model'][1])} trip features; PCA on {result['rows']} "
        f"vehicle-days: {k} components reach {config.PCA_VARIANCE_TARGET:.0%} "
        f"({cum[k - 1]:.1%}), elbow at {result['elbow']}"
    )

"""
Agentic ECT signature distribution analysis and ensemble collapse strategies.

Provides:
1. ``COLLAPSE_STRATEGIES``: Declarative catalog of 6 ECT collapse methods
   (``barycenter``, ``trimmed_barycenter``, ``frechet_medoid``,
   ``euclidean_medoid``, ``mode_medoid``, ``cluster_barycenter``).
2. ``analyze_ect_distribution``: Characterizes the empirical distribution of
   ECT signatures {Psi_1, ..., Psi_M} (Weiszfeld geometric median vs. Euclidean
   mean shift, MAD outlier detection, hierarchical/spectral regime clustering,
   method-family separation ANOVA, and a decision matrix recommending the best
   collapse strategy).
3. ``collapse_ect_ensemble``: Collapses the candidate ensemble into a single
   representation under the chosen strategy (or compares all strategies side by
   side when ``compare_all=True``), verifying topological fidelity via
   ``manifold_convexity_ratio`` and computing ECT-weighted Rubin's rules
   per-column representational variability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

COLLAPSE_STRATEGIES: dict[str, dict[str, Any]] = {
    "barycenter": {
        "name": "barycenter",
        "kind": "pooling",
        "mechanism": (
            "Kernel-weighted pooling X_bar = sum_m w_m X^(m) with "
            "w_m proportional to exp(-||psi_m - psi_med||_2^2 / (2 * tau^2)) "
            "(weighted vote for categorical columns)."
        ),
        "ideal_regime": "unimodal_concentrated",
        "when_to_choose": (
            "Choose when the ECT signature distribution is unimodal and the data "
            "manifold is locally convex (manifold_convexity_ratio <= 1.05). "
            "Pooling across candidates reduces imputation variance (Rubin-style "
            "topological pooling)."
        ),
        "parameters": ["temperature"],
    },
    "trimmed_barycenter": {
        "name": "trimmed_barycenter",
        "kind": "pooling",
        "mechanism": (
            "Drops the alpha-fraction of candidates farthest from the Weiszfeld "
            "geometric median psi_med and computes the kernel-weighted barycenter "
            "on the surviving core (1 - alpha) * M candidates."
        ),
        "ideal_regime": "unimodal_heavy_tailed",
        "when_to_choose": (
            "Choose when the ECT signature distribution has outlier candidates or "
            "heavy tails (e.g. a diverging iterative imputer or SimpleImputer "
            "baseline) that would distort an untrimmed barycenter."
        ),
        "parameters": ["temperature", "trim_quantile"],
    },
    "frechet_medoid": {
        "name": "frechet_medoid",
        "kind": "selection",
        "mechanism": (
            "Selects the single candidate realization X^(m*) minimizing L2 "
            "distance to the L1 Weiszfeld geometric median psi_med: "
            "m* = argmin_m ||psi_m - psi_med||_2."
        ),
        "ideal_regime": "non_convex_or_heavy_tailed",
        "when_to_choose": (
            "Choose when averaging across datasets crosses a hole or curvature in "
            "the manifold (manifold_convexity_ratio > 1.05) or when strict "
            "single-realization row coherence is required."
        ),
        "parameters": [],
    },
    "euclidean_medoid": {
        "name": "euclidean_medoid",
        "kind": "selection",
        "mechanism": (
            "Classic Phil selector: picks the single candidate realization X^(m*) "
            "closest to the arithmetic mean ECT signature psi_bar: "
            "m* = argmin_m ||psi_m - psi_bar||_2."
        ),
        "ideal_regime": "unimodal_symmetric",
        "when_to_choose": (
            "Baseline selector for symmetric, outlier-free ensembles where a "
            "single realization is desired."
        ),
        "parameters": [],
    },
    "mode_medoid": {
        "name": "mode_medoid",
        "kind": "selection",
        "mechanism": (
            "Selects the single candidate realization X^(m*) at the maximum local "
            "Gaussian KDE density in ECT signature space: "
            "m* = argmax_m sum_k exp(-||psi_m - psi_k||_2^2 / (2 * h^2))."
        ),
        "ideal_regime": "multimodal_or_skewed",
        "when_to_choose": (
            "Choose when the ECT distribution is multimodal or strongly skewed "
            "and you want a single realization from the highest-density topological "
            "mode without averaging across incompatible regimes."
        ),
        "parameters": ["temperature"],
    },
    "cluster_barycenter": {
        "name": "cluster_barycenter",
        "kind": "pooling",
        "mechanism": (
            "Identifies topological clusters in ECT space, selects the dominant "
            "(or agent-specified cluster_id) regime C_k, and computes the "
            "kernel-weighted barycenter strictly over candidates m in C_k."
        ),
        "ideal_regime": "multimodal_clustered",
        "when_to_choose": (
            "Choose when candidates split into distinct topological regimes "
            "(e.g. domain-guided MaskedIterativeImputer vs. marginal baselines) "
            "and you want variance reduction within a specific regime without "
            "cross-cluster blurring."
        ),
        "parameters": ["temperature", "cluster_id"],
    },
}


@dataclass
class ECTCollapseResult:
    strategy: str
    collapsed_df: pd.DataFrame
    selected_index: int
    weights: list[float]
    effective_ensemble_size: float
    ect_distance_to_median: float
    medoid_ect_distance_to_median: float
    manifold_convexity_ratio: float
    non_convex_manifold_warning: bool
    distribution_summary: dict[str, Any]
    column_variability: dict[str, dict[str, Any]]
    strategy_comparison: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("collapsed_df", None)
        return payload

    def to_report_dict(self) -> dict[str, Any]:
        return self.to_dict()


def list_collapse_strategies_metadata() -> list[dict[str, Any]]:
    """Return declarative metadata for all supported ECT collapse strategies."""
    return list(COLLAPSE_STRATEGIES.values())


def compute_weiszfeld_median(
    vectors: np.ndarray, max_iter: int = 100, tol: float = 1e-6
) -> np.ndarray:
    """Compute the L1 geometric median of row vectors via Weiszfeld's algorithm."""
    if len(vectors) == 0:
        raise ValueError("Cannot compute geometric median of empty array.")
    if len(vectors) == 1:
        return vectors[0].copy()

    y = np.mean(vectors, axis=0)
    for _ in range(max_iter):
        dists = np.linalg.norm(vectors - y, axis=1)
        zero_mask = dists < 1e-12
        if np.any(zero_mask):
            # If current iterate coincides with a sample point, nudge or check optimality
            inv_dists = np.where(zero_mask, 0.0, 1.0 / np.maximum(dists, 1e-12))
            if np.sum(inv_dists) == 0.0:
                break
        else:
            inv_dists = 1.0 / dists
        weights = inv_dists / np.sum(inv_dists)
        y_next = np.sum(vectors * weights[:, None], axis=0)
        if np.linalg.norm(y_next - y) < tol:
            y = y_next
            break
        y = y_next
    return y


def _flatten_descriptors(descriptors: list[np.ndarray]) -> np.ndarray:
    if not descriptors:
        return np.empty((0, 0), dtype=float)
    return np.stack([np.asarray(d, dtype=float).reshape(-1) for d in descriptors])


def analyze_ect_distribution(
    descriptors: list[np.ndarray],
    candidate_metadata: list[dict[str, Any]] | None = None,
    candidate_methods: list[str] | None = None,
) -> dict[str, Any]:
    """
    Analyze the empirical distribution of ECT signatures {Psi_1, ..., Psi_M}
    to guide agentic selection of the collapse strategy.
    """
    if candidate_metadata is None and candidate_methods is not None:
        candidate_metadata = [
            {"index": idx, "method": str(m), "estimator": str(m)}
            for idx, m in enumerate(candidate_methods)
        ]
    n = len(descriptors)
    if n == 0:
        return {
            "n_candidates": 0,
            "distribution_regime": "empty",
            "recommended_strategy": "barycenter",
        }

    V = _flatten_descriptors(descriptors)
    mean_vec = np.mean(V, axis=0)
    median_vec = compute_weiszfeld_median(V)

    dists_to_mean = np.linalg.norm(V - mean_vec, axis=1)
    dists_to_median = np.linalg.norm(V - median_vec, axis=1)

    if n > 1:
        pw_dists = pdist(V, metric="euclidean")
        dist_matrix = squareform(pw_dists)
        median_pw = float(np.median(pw_dists))
        mean_pw = float(np.mean(pw_dists))
        max_pw = float(np.max(pw_dists))
    else:
        pw_dists = np.array([0.0])
        dist_matrix = np.zeros((1, 1), dtype=float)
        median_pw = 0.0
        mean_pw = 0.0
        max_pw = 0.0

    scale_denom = median_pw if median_pw > 1e-9 else 1.0
    center_shift_norm = float(np.linalg.norm(mean_vec - median_vec) / scale_denom)
    frechet_variance = float(np.mean(dists_to_median**2))
    iqr_dist = (
        float(np.percentile(dists_to_median, 75) - np.percentile(dists_to_median, 25))
        if n >= 2
        else 0.0
    )

    # Outlier detection via robust MAD modified Z-score on distances to geometric median
    med_d = float(np.median(dists_to_median))
    mad_d = float(np.median(np.abs(dists_to_median - med_d)))
    if mad_d > 1e-9:
        mod_z = 0.6745 * (dists_to_median - med_d) / mad_d
    else:
        std_d = float(np.std(dists_to_median))
        mod_z = (
            (dists_to_median - med_d) / std_d
            if std_d > 1e-9
            else np.zeros_like(dists_to_median)
        )

    outlier_candidates: list[dict[str, Any]] = []
    for idx in range(n):
        if float(mod_z[idx]) > 2.5 and dists_to_median[idx] > med_d:
            meta = (
                candidate_metadata[idx]
                if candidate_metadata and idx < len(candidate_metadata)
                else {}
            )
            outlier_candidates.append(
                {
                    "index": int(idx),
                    "distance_to_median": round(float(dists_to_median[idx]), 6),
                    "modified_z_score": round(float(mod_z[idx]), 4),
                    "method": meta.get("method", "unknown"),
                    "estimator": meta.get("estimator", "unknown"),
                }
            )

    # Modality & hierarchical clustering on pairwise ECT distances
    best_k = 1
    best_sil = 0.0
    best_labels = np.zeros(n, dtype=int)
    if n >= 4 and max_pw > 1e-9:
        Z_link = linkage(pw_dists, method="average")
        for k_try in range(2, min(5, n)):
            labels = fcluster(Z_link, t=k_try, criterion="maxclust") - 1
            n_unique_labels = len(np.unique(labels))
            if 2 <= n_unique_labels < n:
                try:
                    sil = float(
                        silhouette_score(dist_matrix, labels, metric="precomputed")
                    )
                except ValueError:
                    sil = 0.0
                if sil > best_sil:
                    best_sil = sil
                    best_k = n_unique_labels
                    best_labels = labels

    is_multimodal = bool(best_k >= 2 and best_sil >= 0.35)

    clusters_info: list[dict[str, Any]] = []
    if is_multimodal:
        unique_clusters = sorted(np.unique(best_labels).tolist())
    else:
        best_labels = np.zeros(n, dtype=int)
        unique_clusters = [0]

    for cid in unique_clusters:
        member_idxs = [int(i) for i in np.where(best_labels == cid)[0]]
        sub_V = V[member_idxs]
        sub_med = compute_weiszfeld_median(sub_V)
        sub_dists = np.linalg.norm(sub_V - sub_med, axis=1)
        medoid_local = int(np.argmin(sub_dists))
        medoid_global = member_idxs[medoid_local]

        methods_in_cluster: dict[str, int] = {}
        for m_idx in member_idxs:
            meta = (
                candidate_metadata[m_idx]
                if candidate_metadata and m_idx < len(candidate_metadata)
                else {}
            )
            label = meta.get("estimator") or meta.get("method") or "unknown"
            methods_in_cluster[str(label)] = methods_in_cluster.get(str(label), 0) + 1

        clusters_info.append(
            {
                "cluster_id": int(cid),
                "size": len(member_idxs),
                "weight_fraction": round(len(member_idxs) / n, 4),
                "medoid_index": int(medoid_global),
                "member_indices": member_idxs,
                "intra_cluster_mean_dist": round(float(np.mean(sub_dists)), 6),
                "method_composition": methods_in_cluster,
            }
        )

    # Sort clusters by size descending so cluster 0 is dominant
    clusters_info.sort(key=lambda c: (-c["size"], c["intra_cluster_mean_dist"]))
    for new_id, c_dict in enumerate(clusters_info):
        c_dict["cluster_id"] = new_id

    # Method-family ANOVA F-statistic in ECT space
    method_separation_f_stat = 0.0
    if candidate_metadata and len(candidate_metadata) == n and n >= 3:
        families = [
            str(m.get("estimator") or m.get("method") or "unknown")
            for m in candidate_metadata
        ]
        unique_fams = sorted(set(families))
        if 1 < len(unique_fams) < n:
            ss_between = 0.0
            ss_within = 0.0
            for fam in unique_fams:
                idxs = [i for i, f in enumerate(families) if f == fam]
                fam_V = V[idxs]
                fam_mean = np.mean(fam_V, axis=0)
                ss_between += len(idxs) * float(np.sum((fam_mean - mean_vec) ** 2))
                ss_within += float(np.sum((fam_V - fam_mean) ** 2))
            df_between = len(unique_fams) - 1
            df_within = n - len(unique_fams)
            if ss_within > 1e-12 and df_between > 0 and df_within > 0:
                method_separation_f_stat = float(
                    (ss_between / df_between) / (ss_within / df_within)
                )

    # Determine topological regime & recommended strategy
    if is_multimodal:
        regime = "multimodal_clustered"
        recommended_strategy = "cluster_barycenter"
        rationale = (
            f"ECT signature distribution is multimodal (best_k={best_k}, "
            f"silhouette={best_sil:.3f}). Global averaging across clusters risks "
            "blurring distinct topological regimes; prefer 'cluster_barycenter' "
            "or 'mode_medoid'."
        )
    elif len(outlier_candidates) > 0 or center_shift_norm > 0.15:
        regime = "unimodal_heavy_tailed"
        recommended_strategy = "trimmed_barycenter"
        rationale = (
            f"Detected {len(outlier_candidates)} outlier candidate(s) and "
            f"mean-vs-median shift={center_shift_norm:.3f}. Prefer "
            "'trimmed_barycenter' or 'frechet_medoid' to resist outlier pull."
        )
    else:
        regime = "unimodal_concentrated"
        recommended_strategy = "barycenter"
        rationale = (
            "ECT signatures form a unimodal, concentrated distribution without "
            "extreme outliers. Kernel-weighted 'barycenter' pooling reduces "
            "imputation variance while preserving global topology."
        )

    # Rank candidates by distance to geometric median and Euclidean mean
    frechet_medoid_idx = int(np.argmin(dists_to_median))
    euclidean_medoid_idx = int(np.argmin(dists_to_mean))

    # Mode medoid via Gaussian KDE in ECT space
    bandwidth = median_pw if median_pw > 1e-9 else 1.0
    kde_densities = np.mean(np.exp(-(dist_matrix**2) / (2.0 * (bandwidth**2))), axis=1)
    mode_medoid_idx = int(np.argmax(kde_densities))

    return {
        "n_candidates": n,
        "distribution_regime": regime,
        "recommended_strategy": recommended_strategy,
        "recommendation_rationale": rationale,
        "center_and_dispersion": {
            "mean_pairwise_l2": round(mean_pw, 6),
            "median_pairwise_l2": round(median_pw, 6),
            "max_pairwise_l2": round(max_pw, 6),
            "frechet_variance": round(frechet_variance, 6),
            "iqr_distance_to_median": round(iqr_dist, 6),
            "center_shift_norm": round(center_shift_norm, 6),
        },
        "key_candidate_indices": {
            "frechet_medoid_index": frechet_medoid_idx,
            "euclidean_medoid_index": euclidean_medoid_idx,
            "mode_medoid_index": mode_medoid_idx,
        },
        "outliers": {
            "n_outliers": len(outlier_candidates),
            "outlier_candidates": outlier_candidates,
        },
        "modality_and_clustering": {
            "is_multimodal": is_multimodal,
            "best_k": int(best_k),
            "silhouette_score": round(best_sil, 4),
            "method_separation_f_stat": round(method_separation_f_stat, 4),
            "clusters": clusters_info,
        },
        "strategy_comparison_guide": COLLAPSE_STRATEGIES,
    }


def _compute_strategy_weights(
    strategy: str,
    V: np.ndarray,
    dist_info: dict[str, Any],
    *,
    temperature: float | None = 1.0,
    trim_quantile: float = 0.2,
    cluster_id: int | None = None,
) -> tuple[np.ndarray, int]:
    """Compute candidate weights w in Delta^{M-1} and representative index for ``strategy``."""
    temp_val = 1.0 if temperature is None else float(temperature)
    n = len(V)
    if n == 1:
        return np.array([1.0], dtype=float), 0

    median_vec = compute_weiszfeld_median(V)
    mean_vec = np.mean(V, axis=0)
    dists_to_median = np.linalg.norm(V - median_vec, axis=1)
    dists_to_mean = np.linalg.norm(V - mean_vec, axis=1)

    frechet_idx = int(np.argmin(dists_to_median))
    euclidean_idx = int(np.argmin(dists_to_mean))

    pw_dists = pdist(V, metric="euclidean")
    median_pw = float(np.median(pw_dists)) if len(pw_dists) else 1.0
    scale = max(median_pw, float(np.median(dists_to_median)), 1e-6) * max(
        temp_val, 1e-3
    )

    if strategy == "euclidean_medoid":
        w = np.zeros(n, dtype=float)
        w[euclidean_idx] = 1.0
        return w, euclidean_idx

    if strategy == "frechet_medoid":
        w = np.zeros(n, dtype=float)
        w[frechet_idx] = 1.0
        return w, frechet_idx

    if strategy == "mode_medoid":
        mode_idx = int(
            dist_info.get("key_candidate_indices", {}).get(
                "mode_medoid_index", frechet_idx
            )
        )
        w = np.zeros(n, dtype=float)
        w[mode_idx] = 1.0
        return w, mode_idx

    if strategy == "barycenter":
        logits = -(dists_to_median**2) / (2.0 * (scale**2))
        logits -= np.max(logits)
        w = np.exp(logits)
        w /= np.sum(w)
        return w, frechet_idx

    if strategy == "trimmed_barycenter":
        alpha = float(np.clip(trim_quantile, 0.0, 0.8))
        n_keep = max(1, int(np.ceil((1.0 - alpha) * n)))
        keep_idxs = np.argsort(dists_to_median)[:n_keep]
        w = np.zeros(n, dtype=float)
        sub_dists = dists_to_median[keep_idxs]
        logits = -(sub_dists**2) / (2.0 * (scale**2))
        logits -= np.max(logits)
        sub_w = np.exp(logits)
        sub_w /= np.sum(sub_w)
        w[keep_idxs] = sub_w
        return w, int(keep_idxs[0])

    if strategy == "cluster_barycenter":
        clusters = dist_info.get("modality_and_clustering", {}).get("clusters", [])
        target_cid = 0 if cluster_id is None else int(cluster_id)
        chosen_cluster = None
        for c in clusters:
            if c["cluster_id"] == target_cid:
                chosen_cluster = c
                break
        if chosen_cluster is None and clusters:
            chosen_cluster = clusters[0]

        if chosen_cluster is None:
            return _compute_strategy_weights(
                "barycenter", V, dist_info, temperature=temperature
            )

        member_idxs = np.array(chosen_cluster["member_indices"], dtype=int)
        sub_V = V[member_idxs]
        sub_med = compute_weiszfeld_median(sub_V)
        sub_dists = np.linalg.norm(sub_V - sub_med, axis=1)
        sub_scale = max(float(np.median(sub_dists)), scale, 1e-6)
        logits = -(sub_dists**2) / (2.0 * (sub_scale**2))
        logits -= np.max(logits)
        sub_w = np.exp(logits)
        sub_w /= np.sum(sub_w)
        w = np.zeros(n, dtype=float)
        w[member_idxs] = sub_w
        rep_idx = int(chosen_cluster.get("medoid_index", member_idxs[0]))
        return w, rep_idx

    raise ValueError(
        f"Unknown collapse strategy '{strategy}'. Supported: {sorted(COLLAPSE_STRATEGIES)}"
    )


def _pool_candidate_dataframes(
    candidates: list[pd.DataFrame],
    weights: np.ndarray,
    *,
    original_df: pd.DataFrame | None = None,
    variable_bounds: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Combine candidate DataFrames using probability weights ``weights``."""
    if not candidates:
        raise ValueError("No candidate DataFrames to pool.")

    active_idxs = [i for i, w in enumerate(weights) if w > 1e-12]
    if len(active_idxs) == 1:
        out = candidates[active_idxs[0]].copy()
    else:
        template = candidates[0]
        out_data: dict[str, Any] = {}
        for col in template.columns:
            col_series = template[col]
            if pd.api.types.is_numeric_dtype(
                col_series
            ) and not pd.api.types.is_bool_dtype(col_series):
                stacked = np.column_stack(
                    [candidates[i][col].to_numpy(dtype=float) for i in active_idxs]
                )
                active_w = weights[active_idxs]
                active_w = active_w / np.sum(active_w)
                pooled = np.sum(stacked * active_w[None, :], axis=1)
                if original_df is not None and col in original_df.columns:
                    orig_dtype = original_df[col].dtype
                    if pd.api.types.is_integer_dtype(orig_dtype):
                        pooled = np.rint(pooled).astype(orig_dtype)
                out_data[col] = pooled
            else:
                # Weighted mode across active candidates for categorical/boolean columns
                n_rows = len(template)
                vals_matrix = [candidates[i][col].tolist() for i in active_idxs]
                active_w = weights[active_idxs]
                chosen_col = []
                for r in range(n_rows):
                    scores: dict[Any, float] = {}
                    for m_pos, w_val in enumerate(active_w):
                        v = vals_matrix[m_pos][r]
                        scores[v] = scores.get(v, 0.0) + float(w_val)
                    best_val = max(scores.items(), key=lambda kv: kv[1])[0]
                    chosen_col.append(best_val)
                out_data[col] = pd.Series(
                    chosen_col, index=template.index, dtype=col_series.dtype
                )
        out = pd.DataFrame(out_data, index=template.index)

    # Enforce variable_bounds if provided
    if variable_bounds:
        for col, b_cfg in variable_bounds.items():
            if col not in out.columns:
                continue
            if not pd.api.types.is_numeric_dtype(out[col]):
                continue
            if isinstance(b_cfg, dict):
                lo = b_cfg.get("min_value")
                hi = b_cfg.get("max_value")
            else:
                lo = getattr(b_cfg, "min_value", None)
                hi = getattr(b_cfg, "max_value", None)
            if lo is not None or hi is not None:
                out[col] = out[col].clip(lower=lo, upper=hi)

    return out


def _compute_rubin_column_variability(
    candidates: list[pd.DataFrame],
    weights: np.ndarray,
    original_df: pd.DataFrame | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Compute per-column between-imputation uncertainty (ECT-weighted Rubin's rules)
    across the ensemble at missing cells.
    """
    if not candidates:
        return {}

    w = np.asarray(weights, dtype=float)
    w = w / np.sum(w) if np.sum(w) > 0 else np.ones(len(candidates)) / len(candidates)
    n_active = int(np.sum(w > 1e-12))
    # When a single-realization selector is used (1-hot weight vector), compute
    # between-imputation dispersion across the full ensemble so Rubin uncertainty
    # still reflects how much the candidate imputations disagree!
    w_var = (
        np.ones(len(candidates), dtype=float) / len(candidates)
        if n_active <= 1 and len(candidates) > 1
        else w
    )

    template = candidates[0]
    report: dict[str, dict[str, Any]] = {}

    for col in template.columns:
        if original_df is not None and col in original_df.columns:
            miss_mask = original_df[col].isna().to_numpy()
        else:
            miss_mask = np.ones(len(template), dtype=bool)

        n_missing = int(np.sum(miss_mask))
        if n_missing == 0:
            continue

        series0 = template[col]
        if pd.api.types.is_numeric_dtype(series0) and not pd.api.types.is_bool_dtype(
            series0
        ):
            # Shape: (n_missing, M)
            mat = np.column_stack(
                [c[col].to_numpy(dtype=float)[miss_mask] for c in candidates]
            )
            cell_means = np.sum(mat * w_var[None, :], axis=1)
            cell_vars = np.sum(
                ((mat - cell_means[:, None]) ** 2) * w_var[None, :], axis=1
            )
            cell_stds = np.sqrt(np.maximum(cell_vars, 0.0))
            cell_iqrs = np.percentile(mat, 75, axis=1) - np.percentile(mat, 25, axis=1)

            col_scale = float(np.nanstd(template[col].to_numpy(dtype=float)))
            rel_std = float(np.mean(cell_stds) / col_scale) if col_scale > 1e-9 else 0.0
            report[str(col)] = {
                "n_imputed_cells": n_missing,
                "type": "numeric",
                "mean_between_imputation_std": round(float(np.mean(cell_stds)), 6),
                "max_between_imputation_std": round(float(np.max(cell_stds)), 6),
                "mean_between_imputation_iqr": round(float(np.mean(cell_iqrs)), 6),
                "relative_uncertainty_ratio": round(rel_std, 4),
                "uncertainty_tier": (
                    "high"
                    if rel_std >= 0.5
                    else ("moderate" if rel_std >= 0.2 else "low")
                ),
            }
        else:
            # Categorical agreement across candidates on missing cells
            mat_obj = [c[col].loc[miss_mask].tolist() for c in candidates]
            agreements: list[float] = []
            for r_idx in range(n_missing):
                counts: dict[Any, float] = {}
                for m_idx, w_val in enumerate(w):
                    val = mat_obj[m_idx][r_idx]
                    counts[val] = counts.get(val, 0.0) + float(w_val)
                agreements.append(max(counts.values()) if counts else 1.0)
            mean_agree = float(np.mean(agreements))
            disagreement = 1.0 - mean_agree
            report[str(col)] = {
                "n_imputed_cells": n_missing,
                "type": "categorical",
                "modal_agreement_rate": round(mean_agree, 4),
                "between_imputation_disagreement": round(disagreement, 4),
                "uncertainty_tier": (
                    "high"
                    if disagreement >= 0.4
                    else ("moderate" if disagreement >= 0.15 else "low")
                ),
            }

    return report


def _evaluate_collapsed_ect_fidelity(
    collapsed_df: pd.DataFrame,
    standardized_candidates: list[np.ndarray],
    weights: np.ndarray,
    magic_estimator: Any,
    median_vec: np.ndarray,
    medoid_dist: float,
) -> tuple[float, float, bool]:
    """
    Compute the ECT signature of the collapsed representation and check whether
    linear pooling stayed on the manifold (manifold_convexity_ratio <= 1.05).
    """
    active_idxs = [i for i, w in enumerate(weights) if w > 1e-12]
    if len(active_idxs) == 1 or magic_estimator is None or not standardized_candidates:
        return medoid_dist, 1.0, False

    try:
        active_w = weights[active_idxs]
        active_w = active_w / np.sum(active_w)
        Z_collapsed = np.zeros_like(standardized_candidates[0], dtype=float)
        for idx, w_val in zip(active_idxs, active_w):
            Z_collapsed += float(w_val) * np.asarray(
                standardized_candidates[idx], dtype=float
            )
        psi_collapsed_list = magic_estimator.generate([Z_collapsed])
        psi_collapsed = np.asarray(psi_collapsed_list[0], dtype=float).reshape(-1)
        collapsed_dist = float(np.linalg.norm(psi_collapsed - median_vec))
        denom = max(medoid_dist, 1e-6)
        ratio = float(collapsed_dist / denom)
        return collapsed_dist, ratio, bool(ratio > 1.05)
    except (ValueError, RuntimeError, TypeError, AttributeError):
        return medoid_dist, 1.0, False


def collapse_ect_ensemble(
    candidates: list[pd.DataFrame],
    descriptors: list[np.ndarray],
    *,
    strategy: str = "auto",
    temperature: float = 1.0,
    trim_quantile: float = 0.2,
    cluster_id: int | None = None,
    compare_all: bool = False,
    original_df: pd.DataFrame | None = None,
    standardized_candidates: list[np.ndarray] | None = None,
    candidate_metadata: list[dict[str, Any]] | None = None,
    magic_estimator: Any = None,
    variable_bounds: dict[str, Any] | None = None,
) -> ECTCollapseResult:
    """
    Collapse an ensemble of candidate imputations {X^(1), ..., X^(M)} using their
    ECT signatures {Psi_1, ..., Psi_M} and return the collapsed DataFrame plus
    topological fidelity and Rubin representational variability diagnostics.
    """
    if not candidates or not descriptors:
        raise ValueError("candidates and descriptors must be non-empty.")

    dist_info = analyze_ect_distribution(
        descriptors, candidate_metadata=candidate_metadata
    )
    V = _flatten_descriptors(descriptors)
    median_vec = compute_weiszfeld_median(V)
    dists_to_median = np.linalg.norm(V - median_vec, axis=1)
    medoid_dist = float(np.min(dists_to_median))

    effective_strategy = (
        dist_info["recommended_strategy"] if strategy == "auto" else strategy
    )
    if effective_strategy not in COLLAPSE_STRATEGIES:
        raise ValueError(
            f"Unsupported strategy '{effective_strategy}'. "
            f"Choose 'auto' or one of {sorted(COLLAPSE_STRATEGIES)}."
        )

    def _run_single(
        strat_name: str,
    ) -> tuple[pd.DataFrame, np.ndarray, int, float, float, bool]:
        w, rep_idx = _compute_strategy_weights(
            strat_name,
            V,
            dist_info,
            temperature=temperature,
            trim_quantile=trim_quantile,
            cluster_id=cluster_id,
        )
        df_c = _pool_candidate_dataframes(
            candidates,
            w,
            original_df=original_df,
            variable_bounds=variable_bounds,
        )
        active_idxs = [i for i, val in enumerate(w) if val > 1e-12]
        if len(active_idxs) == 1:
            c_dist = float(dists_to_median[active_idxs[0]])
            c_ratio = float(c_dist / max(medoid_dist, 1e-6))
            warn = False
        else:
            c_dist, c_ratio, warn = _evaluate_collapsed_ect_fidelity(
                df_c,
                standardized_candidates or [],
                w,
                magic_estimator,
                median_vec,
                medoid_dist,
            )
        return df_c, w, rep_idx, c_dist, c_ratio, warn

    collapsed_df, weights, selected_index, ect_dist, conv_ratio, non_convex_warn = (
        _run_single(effective_strategy)
    )
    eff_n = float(1.0 / np.sum(weights**2)) if np.sum(weights**2) > 0 else 1.0

    # If auto strategy picked a pooling method but the manifold is non-convex,
    # fall back to frechet_medoid automatically
    if strategy == "auto" and non_convex_warn:
        effective_strategy = "frechet_medoid"
        collapsed_df, weights, selected_index, ect_dist, conv_ratio, non_convex_warn = (
            _run_single(effective_strategy)
        )
        eff_n = 1.0

    comparison_table: list[dict[str, Any]] = []
    if compare_all:
        for strat_name in COLLAPSE_STRATEGIES:
            _, w_s, idx_s, dist_s, ratio_s, warn_s = _run_single(strat_name)
            eff_s = float(1.0 / np.sum(w_s**2)) if np.sum(w_s**2) > 0 else 1.0
            comparison_table.append(
                {
                    "strategy": strat_name,
                    "kind": COLLAPSE_STRATEGIES[strat_name]["kind"],
                    "representative_index": int(idx_s),
                    "effective_ensemble_size": round(eff_s, 3),
                    "ect_distance_to_median": round(dist_s, 6),
                    "manifold_convexity_ratio": round(ratio_s, 4),
                    "non_convex_manifold_warning": warn_s,
                }
            )

    col_variability = _compute_rubin_column_variability(
        candidates, weights, original_df=original_df
    )

    return ECTCollapseResult(
        strategy=effective_strategy,
        collapsed_df=collapsed_df,
        selected_index=int(selected_index),
        weights=[round(float(x), 6) for x in weights.tolist()],
        effective_ensemble_size=round(eff_n, 4),
        ect_distance_to_median=round(ect_dist, 6),
        medoid_ect_distance_to_median=round(medoid_dist, 6),
        manifold_convexity_ratio=round(conv_ratio, 4),
        non_convex_manifold_warning=non_convex_warn,
        distribution_summary=dist_info,
        column_variability=col_variability,
        strategy_comparison=comparison_table,
    )

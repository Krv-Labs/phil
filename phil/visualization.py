"""
Visualization utilities for Phil imputation analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.manifold import MDS

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_mds(
    descriptors: list[np.ndarray],
    selected_index: int,
    ax=None,
    figsize: tuple[int, int] = (8, 6),
    random_state: int | None = None,
) -> tuple[Figure, np.ndarray]:
    """
    Visualize the ECT descriptor space via Multi-Dimensional Scaling (MDS).
    """
    try:
        import matplotlib.patheffects as pe
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        ) from exc

    if len(descriptors) < 2:
        raise ValueError("plot_mds requires at least two descriptors.")
    if not 0 <= selected_index < len(descriptors):
        raise ValueError("selected_index is out of range for descriptors.")

    flat = [np.asarray(d).ravel() for d in descriptors]
    if len({f.shape[0] for f in flat}) != 1:
        raise ValueError("All descriptors must have the same flattened length.")
    flat_array = np.asarray(flat)

    dist_matrix = cdist(flat_array, flat_array, metric="euclidean")

    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=random_state,
        normalized_stress="auto",
    )
    embedding = mds.fit_transform(dist_matrix)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    mask = np.ones(len(descriptors), dtype=bool)
    mask[selected_index] = False
    ax.scatter(
        embedding[mask, 0],
        embedding[mask, 1],
        c="#adb5bd",
        s=80,
        zorder=2,
        label="Candidates",
    )

    mean_pt = embedding.mean(axis=0)
    ax.scatter(
        mean_pt[0],
        mean_pt[1],
        c="#4dabf7",
        s=120,
        marker="D",
        zorder=3,
        label="Mean",
    )

    sel = embedding[selected_index]
    ax.scatter(
        sel[0],
        sel[1],
        c="#f03e3e",
        s=200,
        marker="*",
        zorder=4,
        label=f"Selected (#{selected_index})",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
    )

    for i, (x, y) in enumerate(embedding):
        ax.annotate(
            str(i),
            (x, y),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
            color="#495057",
        )

    ax.set_title("ECT Descriptor Space (MDS)", fontsize=13)
    ax.set_xlabel("MDS dimension 1")
    ax.set_ylabel("MDS dimension 2")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    return fig, embedding

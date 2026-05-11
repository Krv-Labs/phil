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
) -> tuple["Figure", np.ndarray]:
    """
    Visualize the ECT descriptor space via Multi-Dimensional Scaling (MDS).

    Computes pairwise L2 distances between ECT descriptors, projects them
    into 2D via MDS, and plots which candidate was selected as the
    representative imputation.

    Parameters
    ----------
    descriptors:
        List of ECT descriptor arrays, one per candidate imputation.
        Each array has shape ``(num_thetas, resolution)``.
    selected_index:
        Index into ``descriptors`` identifying the selected representative.
    ax:
        Existing matplotlib ``Axes`` to draw on. If ``None``, a new figure
        is created.
    figsize:
        Figure size in inches when creating a new figure.
    random_state:
        Seed for MDS random initialisation.

    Returns
    -------
    fig : matplotlib.figure.Figure
    embedding : np.ndarray, shape ``(n, 2)``
        2D MDS coordinates, one row per descriptor.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        ) from exc

    flat = np.array([d.ravel() for d in descriptors])
    dist_matrix = cdist(flat, flat, metric="euclidean")

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

    # Plot candidates
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

    # Mean position in embedding space
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

    # Selected representative
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

    # Annotate each point with its index
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

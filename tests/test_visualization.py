import numpy as np
import pytest

pytest.importorskip("matplotlib", reason="matplotlib required for visualization tests")

from phil.visualization import plot_mds


class TestPlotMds:
    def _make_descriptors(self, n=5, num_thetas=8, resolution=10):
        rng = np.random.RandomState(0)
        return [rng.rand(num_thetas, resolution) for _ in range(n)]

    def test_returns_figure_and_embedding(self):
        import matplotlib.figure

        descriptors = self._make_descriptors()
        fig, embedding = plot_mds(descriptors, selected_index=2, random_state=0)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert embedding.shape == (5, 2)

    def test_embedding_shape_matches_n_descriptors(self):
        for n in [3, 7, 15]:
            descriptors = self._make_descriptors(n=n)
            _, embedding = plot_mds(descriptors, selected_index=0, random_state=0)
            assert embedding.shape == (n, 2)

    def test_accepts_existing_axes(self):
        import matplotlib.pyplot as plt

        descriptors = self._make_descriptors()
        fig_outer, ax = plt.subplots()
        fig_returned, _ = plot_mds(descriptors, selected_index=1, ax=ax, random_state=0)
        assert fig_returned is fig_outer
        plt.close("all")

    def test_single_descriptor_raises(self):
        descriptors = self._make_descriptors(n=1)
        with pytest.raises(Exception):
            plot_mds(descriptors, selected_index=0, random_state=0)

    def test_deterministic_with_random_state(self):
        descriptors = self._make_descriptors()
        _, emb1 = plot_mds(descriptors, selected_index=0, random_state=42)
        _, emb2 = plot_mds(descriptors, selected_index=0, random_state=42)
        np.testing.assert_allclose(emb1, emb2)

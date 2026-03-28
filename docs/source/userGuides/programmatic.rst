.. _programmatic:

============
Programmatic
============

Construct custom configs directly:

.. code-block:: python

   from phil import Phil, ImputationConfig, ECTConfig
   import pandas as pd

   config = ImputationConfig(
     methods=["SimpleImputer", "KNNImputer"],
     modules=["sklearn.impute", "sklearn.impute"],
     grids=[{"strategy": ["mean", "median"]}, {"n_neighbors": [3, 5]}],
   )

   magic_config = ECTConfig(
     num_thetas=32,
     radius=1.0,
     resolution=64,
     scale=256,
     normalize=True,
     seed=42,
   )

   model = Phil(samples=20, param_grid=config, config=magic_config)
   model.fit(pd.DataFrame({"x": [1, None, 3], "y": [1.0, 2.0, None]}))

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

Predictability Profiling & Domain Priors
----------------------------------------

Before fitting an ensemble, profile which incomplete columns are empirically
predictable from observed features and which require domain/literature priors:

.. code-block:: python

   from phil import (
       Phil,
       DomainKnowledge,
       CovariateSubset,
       VariableBounds,
       profile_predictability,
       build_research_worksheet,
   )

   profile = profile_predictability(df)
   worksheet = build_research_worksheet(df, domain_hint="cardiology")

   domain = DomainKnowledge(
       predictable_columns=profile["predictable_columns"],
       bounds={"age": VariableBounds(lower=18.0, upper=100.0)},
       covariate_priors=[
           CovariateSubset(
               covariates={"sex": "F"},
               means={"cholesterol": 195.0},
               stds={"cholesterol": 25.0},
           )
       ],
   )

   phil = Phil(samples=20, domain_knowledge=domain, collapse_strategy="auto")
   completed_df = phil.fit(df)

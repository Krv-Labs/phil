.. _programmatic:

============
Programmatic
============

Construct custom configs directly:

.. code-block:: python

   from phil import Phil, ImputationConfig, ECTConfig
   import pandas as pd
   from sklearn.model_selection import ParameterGrid

   config = ImputationConfig(
     methods=["SimpleImputer", "KNNImputer"],
     modules=["sklearn.impute", "sklearn.impute"],
     grids=[ParameterGrid({"strategy": ["mean", "median"]}),
            ParameterGrid({"n_neighbors": [3, 5]})],
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
       ImputationConfig,
       DomainKnowledge,
       CovariateSubset,
       VariableBounds,
       profile_predictability,
       build_research_worksheet,
   )
   import pandas as pd
   from sklearn.model_selection import ParameterGrid

   # Synthetic measurements; the bounds below describe this example only.
   df = pd.DataFrame({
       "signal": [10.0, 20.0, None, 40.0, 50.0, 60.0, 70.0, 80.0],
       "reference": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
   })
   profile = profile_predictability(df)
   worksheet = build_research_worksheet(
       profile, domain_context="synthetic measurements"
   )
   print(profile["feature_profiles"]["signal"]["predictability_tier"])
   print(worksheet["feature_research_cards"])

   domain = DomainKnowledge(
       variable_bounds={
           "signal": VariableBounds(
               min_value=0.0, max_value=100.0,
               citations=["Synthetic example: signal range is 0 to 100"],
           ),
       },
       covariate_subsets={
           "signal": CovariateSubset(
               predictors=["reference"],
               citations=["Synthetic example: signal = 10 * reference"],
           ),
       },
   )

   config = ImputationConfig(
       methods=["IterativeImputer"],
       modules=["sklearn.impute"],
       grids=[ParameterGrid({})],
       domain_knowledge=domain,
   )
   phil = Phil(samples=20, param_grid=config, collapse_strategy="auto")
   completed_df = phil.fit(df)

Replace the example priors and citations with verified sources for your data.
Bounds constrain imputed cells; observed values remain unchanged. Both ``fit()``
and ``transform()`` return the original column names and units, including
categorical columns. Numeric covariance priors use original units and cover all
numeric columns; specify ``variables`` to identify the matrix order. Covariance
priors currently require numeric-only input.

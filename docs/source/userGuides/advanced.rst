.. _advanced:

========
Advanced
========

- Swap default descriptor strategy by passing a custom ``magic`` backend.
- Control sklearn transform output with ``verbose_feature_names_out`` handling in
  ``PhilTransformer``.
- Integrate ``PhilTransformer`` at the beginning of a predictive pipeline.

ECT Signature Distribution Collapse
-----------------------------------

After fitting a ``Phil`` ensemble, you can inspect the geometry of the ECT
signature cloud and compare all 6 collapse strategies without re-running the
imputation grid:

.. code-block:: python

   from phil import Phil
   import pandas as pd

   df = pd.DataFrame({
       "x": [10.0, 20.0, None, 40.0, 50.0],
       "y": [1.0, 2.0, 3.0, 4.0, 5.0],
   })

   phil = Phil(samples=25, collapse_strategy="auto", random_state=42)
   phil.fit(df)

   # Compare barycenter, trimmed_barycenter, frechet_medoid,
   # euclidean_medoid, mode_medoid, and cluster_barycenter
   comparison = phil.collapse(strategy="auto", compare_all=True)
   print(comparison.distribution_summary)
   print(comparison.strategy_comparison)
   print(comparison.column_variability)
   completed_df = comparison.collapsed_df

``collapse()`` returns an ``ECTCollapseResult``. Use its attributes to access
the completed data and diagnostics, or ``to_report_dict()`` for a serializable
report without the DataFrame. Pooling fidelity is measured by preprocessing
the returned DataFrame and computing its ECT signature, including categorical
votes and numeric bounds. Direct ``collapse_ect_ensemble()`` callers must supply
the fitted ``preprocessor`` and ``magic_estimator`` to measure pooling fidelity;
unavailable distances and ratios are reported as ``None``.

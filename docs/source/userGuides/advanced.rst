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

   phil = Phil(samples=25, collapse_strategy="auto", random_state=42)
   phil.fit(df)

   # Compare barycenter, trimmed_barycenter, frechet_medoid,
   # euclidean_medoid, mode_medoid, and cluster_barycenter
   comparison = phil.collapse(strategy="auto", compare_all=True)
   print(comparison["diagnostics"])
   print(comparison["agreement_ratio"], comparison["strategies"])

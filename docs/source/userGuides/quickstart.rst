.. _quickstart:

==========
Quickstart
==========

This guide gets you from zero to imputed dataset in under 5 minutes.

Prerequisites
-------------

- Python 3.10+
- Phil installed (``pip install phil``)

Basic Usage
-----------

Phil generates multiple candidate imputations and selects the most representative one using topological descriptors.

**Step 1: Create a DataFrame with missing values**

.. code-block:: python

   import pandas as pd
   import numpy as np

   df = pd.DataFrame({
       "age": [25.0, 30.0, np.nan, 45.0, np.nan],
       "income": [50000, np.nan, 75000, 80000, 65000],
       "category": ["A", "B", "A", np.nan, "B"],
   })
   print(f"Missing values: {df.isna().sum().sum()}")

**Step 2: Fit Phil and get the representative imputation**

.. code-block:: python

   from phil import Phil

   imputer = Phil(samples=25, random_state=42)
   completed = imputer.fit(df)

   print(completed)
   print(f"Missing values after: {completed.isna().sum().sum()}")

That's it! Phil generates 25 candidate imputations, computes ECT descriptors for each, and returns the candidate closest to the ensemble's center.

Understanding the Output
------------------------

Phil returns a fully imputed DataFrame. The imputation strategy is chosen automatically based on which candidate best represents the ensemble:

.. code-block:: python

   # Access metadata about the selection
   print(f"Number of candidates generated: {imputer.n_candidates_}")
   print(f"Selected candidate index: {imputer.selected_idx_}")

Pipeline Integration
--------------------

Use ``PhilTransformer`` for sklearn pipelines:

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.ensemble import RandomForestClassifier
   from phil.transformers import PhilTransformer

   pipe = Pipeline([
       ("impute", PhilTransformer(samples=25, random_state=42)),
       ("clf", RandomForestClassifier()),
   ])

   # Fit on data with missing values
   pipe.fit(X_train, y_train)
   predictions = pipe.predict(X_test)

Configuring Imputation Strategies
---------------------------------

By default, Phil explores multiple imputation strategies. You can customize which strategies to include:

.. code-block:: python

   imputer = Phil(
       samples=50,
       strategies=["mean", "median", "knn", "iterative"],
       random_state=42,
   )

Next Steps
----------

- :doc:`programmatic` - Detailed API usage
- :doc:`intermediate` - Tuning and configuration options
- :doc:`advanced` - Custom strategies and descriptor options
- :ref:`API Reference <api-reference>` - Full documentation

.. _overview:

========
Overview
========

Phil provides **representation-guided imputation**: instead of choosing a single imputation strategy, Phil explores many possibilities and uses topological descriptors to select the most representative result.

The Problem
-----------

Imputation is a critical step in data preprocessing, but the choice of strategy significantly impacts downstream analysis:

- **Mean imputation** may distort variance
- **KNN imputation** depends on distance metrics and neighborhood size
- **Iterative imputation** depends on model choice and convergence criteria

How do you know which imputation is "correct"? Phil answers this by treating imputation as an **ensemble selection problem**.

Phil's Approach
---------------

.. mermaid::

   graph TB
      subgraph "1. Generate Candidates"
         A[DataFrame with missing values]
         B1["Mean imputation"]
         B2["KNN imputation"]
         B3["Iterative imputation"]
         B4["Custom strategy"]
      end

      subgraph "2. Compute Descriptors"
         C["ECT computation"]
         D1["Descriptor 1"]
         D2["Descriptor 2"]
         D3["Descriptor 3"]
         D4["Descriptor 4"]
      end

      subgraph "3. Select Representative"
         E["Distance matrix"]
         F["Centroid selection"]
         G["Representative dataset"]
      end

      A --> B1
      A --> B2
      A --> B3
      A --> B4

      B1 --> C
      B2 --> C
      B3 --> C
      B4 --> C

      C --> D1
      C --> D2
      C --> D3
      C --> D4

      D1 --> E
      D2 --> E
      D3 --> E
      D4 --> E

      E --> F
      F --> G

      style A fill:#f9f9f9,stroke:#999
      style C fill:#D9EDF7,stroke:#31708F,stroke-width:2px
      style F fill:#D9EDF7,stroke:#31708F,stroke-width:2px
      style G fill:#DFF0D8,stroke:#3C763D,stroke-width:2px

Key Concepts
------------

**Candidate Generation**
   Phil uses sklearn's imputation methods to generate multiple completed datasets from the same input. Each candidate represents a different "version" of the data.

**ECT Descriptors**
   The Euler Characteristic Transform captures topological structure of each candidate. This provides a principled way to compare imputations based on their geometric properties rather than arbitrary metrics.

**Representative Selection**
   Phil computes pairwise distances between candidate descriptors and selects the one closest to the ensemble mean. This is analogous to selecting a "medoid" in clustering.

Why Topological Descriptors?
----------------------------

Traditional comparison methods (e.g., comparing filled values directly) are sensitive to:

- Scale of individual features
- Arbitrary ordering of rows
- Local perturbations

ECT descriptors are:

- **Invariant** to permutations of data points
- **Robust** to small perturbations
- **Informative** about global structure

Integration Options
-------------------

Phil supports two integration patterns:

**Standalone Usage**

.. code-block:: python

   from phil import Phil

   completed_df = Phil(samples=25).fit(df_with_missing)

**sklearn Pipeline**

.. code-block:: python

   from phil.transformers import PhilTransformer

   pipe = Pipeline([
       ("impute", PhilTransformer(samples=25)),
       ("model", YourModel()),
   ])

Next Steps
----------

- :ref:`Quickstart <quickstart>` - Get started quickly
- :ref:`User Guide <user_guide>` - Detailed configuration
- :ref:`API Reference <api-reference>` - Full documentation

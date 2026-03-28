.. _index:

====
Phil
====

**Topological Imputation with Representative Selection**

Phil generates multiple candidate imputations using configurable sklearn pipelines, scores them with ECT (Euler Characteristic Transform) descriptors, and selects the most representative imputation. Instead of picking a single imputation strategy, explore the space of possibilities and let topological methods guide the selection.

Quick Links
-----------

.. grid:: 1 2 3 3
   :gutter: 3
   :padding: 2 2 0 0

   .. grid-item-card:: :octicon:`rocket` Quickstart
      :link: quickstart
      :link-type: ref
      :class-card: intro-card
      :shadow: md

      Impute a dataset and select a representative in minutes.

   .. grid-item-card:: :octicon:`book` User Guide
      :link: user_guide
      :link-type: ref
      :class-card: intro-card
      :shadow: md

      Installation, configuration, and advanced workflows.

   .. grid-item-card:: :octicon:`code` API Reference
      :link: api-reference
      :link-type: ref
      :class-card: intro-card
      :shadow: md

      Full API documentation for :mod:`phil`.

What is Phil?
-------------

Phil addresses a common challenge in data science: **how do you choose the right imputation strategy?** Different methods (mean, median, KNN, iterative) produce different completed datasets, each with distinct downstream effects on analysis.

Phil's approach:

1. **Generate** multiple candidate imputations across configurable sklearn grids
2. **Score** each candidate using topological descriptors (ECT)
3. **Select** the representative closest to the mean descriptor profile
4. **Deploy** directly or integrate via ``PhilTransformer`` in sklearn pipelines

Typical Workflow
----------------

.. mermaid::

   graph LR
      subgraph Input
         A[DataFrame with missing values]
      end

      subgraph "Stage 1: Impute"
         B["Generate candidates"]
         C1["KNN imputation"]
         C2["Iterative imputation"]
         C3["Mean/Median imputation"]
      end

      subgraph "Stage 2: Score"
         D["ECT descriptors"]
         E["Distance matrix"]
      end

      subgraph "Stage 3: Select"
         F["Representative"]
      end

      A --> B
      B --> C1
      B --> C2
      B --> C3
      C1 --> D
      C2 --> D
      C3 --> D
      D --> E
      E --> F

      style Input fill:#f9f9f9,stroke:#999
      style B fill:#D9EDF7,stroke:#31708F,stroke-width:2px
      style D fill:#D9EDF7,stroke:#31708F,stroke-width:2px
      style F fill:#DFF0D8,stroke:#3C763D,stroke-width:2px

**YAML-Driven (Recommended)**

.. code-block:: python

   from phil import Phil

   imputer = Phil(samples=25, random_state=42)
   completed = imputer.fit(df_with_missing)

**Pipeline Integration**

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from phil.transformers import PhilTransformer

   pipe = Pipeline([
       ("impute", PhilTransformer(samples=25)),
       ("model", YourModel()),
   ])

Key Features
------------

**Multi-Strategy Exploration**
   Generate imputations across mean, median, KNN, iterative, and custom methods.

**Topological Scoring**
   Use ECT descriptors to capture structural properties of each imputation candidate.

**Representative Selection**
   Automatically choose the candidate closest to the ensemble's central tendency.

**sklearn Integration**
   Drop-in transformer for seamless pipeline compatibility.

Installation
------------

.. code-block:: bash

   pip install phil

For development:

.. code-block:: bash

   git clone https://github.com/Krv-Analytics/phil.git
   cd phil
   uv sync --extra dev --extra docs

Supports Python 3.10, 3.11, 3.12.

Next Steps
----------

- :ref:`Quickstart <quickstart>` - Impute your first dataset
- :ref:`User Guide <user_guide>` - Complete installation and configuration
- :ref:`API Reference <api-reference>` - Detailed class and function docs
- :ref:`Transformers <transformers>` - sklearn pipeline integration

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api

.. toctree::
   :maxdepth: 2
   :caption: Guides
   :hidden:

   overview
   user_guide
   configuration

References
----------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

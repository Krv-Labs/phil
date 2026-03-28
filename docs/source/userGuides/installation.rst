.. _installation:

============
Installation
============

.. code-block:: bash

   pip install philler

   # or
   git clone https://github.com/Krv-Analytics/phil.git
   cd phil
   uv sync --group dev

Phil uses `trailed` internally for ECT, so the runtime dependency is managed via
`uv` in ``pyproject.toml``.

.. _mcp:

==========
MCP Server
==========

**No Code. Just a DataFrame and a Question.**

The Phil MCP (Model Context Protocol) server lets AI clients — Claude,
Gemini, Cursor, and others — run topology-guided imputation sweeps on your
pandas or polars dataframes without you writing any Python. Point the agent
at a CSV or Parquet file, ask for the best imputation, and let it pick the
candidate that best represents the ensemble.

This guide is for **practitioners** who know their data has missing values
and want a reproducible imputation rather than a single hand-picked
strategy.

Workflow Comparison
-------------------

.. list-table::
   :widths: 30 25 25 20
   :header-rows: 1

   * - Approach
     - You Do
     - AI Does
     - Speed
   * - **Programmatic** (Python)
     - Write Python, configure grids
     - (nothing)
     - Depends on grid size
   * - **PhilTransformer** (sklearn)
     - Wire into a Pipeline
     - (nothing)
     - Depends on grid size
   * - **MCP + Agent** (recommended)
     - Hand AI a file path, ask question
     - Entire sweep workflow
     - ~5–60s (automated tuning)

The Value Prop
--------------

**Traditional imputation**:
- You guess a strategy (mean, KNN, iterative regressors)
- You run it once, hope it's "good enough"
- No principled way to compare alternatives

**Phil via the MCP**:
- The agent generates a *grid* of candidate imputations
- Each candidate is scored with an ECT topological descriptor
- The candidate closest to the ensemble centroid is selected
- The agent can iterate on the grid if descriptor spread looks degenerate
- You read a structured summary, not a chart of distance metrics

Setup
-----

Phil ships an MCP server entry point (``phil-mcp``) via the ``mcp`` extra
of the published ``philler`` package. You do **not** need to clone the
repo — `uvx <https://docs.astral.sh/uv/guides/tools/>`_ (or ``pipx``) can
launch it directly from PyPI.

.. note::
   Phil works with any MCP-capable client, including Cursor and Gemini CLI,
   where you can add Phil as an MCP server/tool.

.. tab-set::

   .. tab-item:: Claude Desktop

      Open ``~/Library/Application Support/Claude/claude_desktop_config.json``
      (macOS) or ``%APPDATA%\Claude\claude_desktop_config.json`` (Windows)
      and add:

      .. code-block:: json

         {
           "mcpServers": {
             "phil": {
               "command": "uv tool run",
               "args": ["--from", "philler[mcp]", "phil-mcp"]
             }
           }
         }

      Restart Claude Desktop. A hammer icon in new chats confirms the
      tools loaded.

      .. note::
         GUI-launched apps on macOS often don't inherit your shell
         ``PATH``. If Claude can't find ``uvx``, replace
         ``"command": "uv tool run"`` with its absolute path (find it
         with ``which uvx``, e.g. ``/Users/yourname/.local/bin/uvx``).

   .. tab-item:: Gemini CLI

      .. code-block:: bash

         gemini mcp add phil uv tool run --from "philler[mcp]" phil-mcp

   .. tab-item:: Claude Code

      .. code-block:: bash

         claude mcp add phil -- uv tool run --from "philler[mcp]" phil-mcp

   .. tab-item:: Cursor / Windsurf

      Open **Settings → Features → MCP → Add new MCP server**:

      - Name: ``phil``
      - Type: ``command``
      - Command: ``uv tool run --from "philler[mcp]" phil-mcp``

Alternative install methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you prefer a persistent install over ephemeral ``uv`` invocations:

.. code-block:: bash

   pipx install "philler[mcp]"      # then use command: phil-mcp
   # or
   pip install "philler[mcp]"       # in any venv

Developing against a local clone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Contributors working on Phil's source can launch the server from a
checkout instead:

.. code-block:: bash

   uv sync --group mcp
   uv run phil-mcp

Point your MCP client at ``uv run --group mcp phil-mcp`` (with ``cwd``
set to the clone) for live-edit development.

Workflow
--------

Once connected, give the agent a goal rather than instructions. The
agent already knows the technical steps.

**The recommended prompt:**

   *"I have a dataset with missing values at* ``path/to/data.csv``\ *.
   Use Phil to run an imputation sweep and pick the most representative
   completed dataset. Export it to* ``path/to/imputed.csv``\ *."*

Under the hood the agent will:

1. **Ingest** — register the file as a stable ``dataset_id`` handle
2. **Characterize** — summarize missingness, dtypes, and unique counts
3. **List grids** — pick the named ``GridGallery`` strategy that fits
   the data (``default``, ``finance``, ``healthcare``, ...)
4. **Create a config** — generate canonical YAML with sensible defaults
5. **Validate** — confirm the config parses and the dataset is reachable
6. **Run the sweep** — fit each candidate imputation and score with ECT
7. **Diagnose** — inspect descriptor spread and method counts; iterate
   if the grid collapsed
8. **Export** — write the selected imputed DataFrame to disk

Pandas and polars
~~~~~~~~~~~~~~~~~

Phil's pipeline runs on pandas internally, but the MCP server accepts
any file format pandas or pyarrow can read. To use a polars frame, write
it to Parquet first and ingest the path:

.. code-block:: python

   import polars as pl

   df = pl.read_csv("raw.csv")
   df.write_parquet("for_phil.parquet")
   # then in your agent chat:
   #   "Run a Phil sweep on /abs/path/to/for_phil.parquet"

Available MCP Tools
-------------------

The server exposes these tools to the AI client. The agent automatically
chains them together:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Tool
     - What It Does
   * - **get_workflow_guide**
     - Returns the opinionated, phase-by-phase Phil workflow as markdown. Opt-in — agents that prefer their own plan can ignore it.
   * - **get_runtime_context**
     - Reports the server cache directory, session id, and path-visibility guidance — useful when the agent needs to ferry sandboxed files into a host-readable location.
   * - **ingest_dataset**
     - Registers a CSV or Parquet path and returns a stable ``dataset_id`` handle. Pass that handle to every downstream tool.
   * - **begin_dataset_upload / append_dataset_chunk / finalize_dataset_upload**
     - Chunked base64 upload pipeline for clients that cannot share a filesystem with the server. Use only when path-based ingest is impossible.
   * - **characterize_dataset**
     - Sparse per-column schema: dtype, n_unique, missing percent, plus aggregate row/column counts. Cheap and safe to call on wide datasets.
   * - **probe_columns**
     - Deep per-column inspection for up to 20 columns at a time: sample values, top frequencies, basic numeric statistics.
   * - **list_grids**
     - Enumerates the named ``GridGallery`` entries (``default``, ``sampling``, ``finance``, ``healthcare``, ``marketing``, ``engineering``) with method lists and intent blurbs.
   * - **create_config**
     - Materializes a canonical YAML config tailored to a ``dataset_id`` and grid choice. Stores it on the session so subsequent ``refine_active_config`` / ``run_imputation_sweep`` calls can omit it.
   * - **validate_config**
     - Validates and normalizes a config YAML, returning structured issues if anything is off. Rejects fenced Markdown blocks to prevent silent parse failures.
   * - **refine_config / refine_active_config**
     - Apply dotted-path overrides (e.g. ``imputation.samples=50``) to an explicit or session config. Unknown keys raise structured errors with the valid path list.
   * - **get_active_config**
     - Returns the in-session config YAML, useful for inspection before running a sweep.
   * - **run_imputation_sweep**
     - The headline tool: fits the candidate grid, scores each with the ECT magic method, selects the representative, and persists a ``RunRecord`` plus a markdown diff against the previous run.
   * - **diagnose_sweep**
     - Inspects a saved run's descriptor spread, selected index, and per-method candidate counts. Use to decide whether to broaden the grid or raise ``samples``.
   * - **get_candidate_descriptors**
     - Returns the top-k candidates ranked by closeness to the mean descriptor, including the selected index.
   * - **compare_sweeps**
     - Side-by-side comparison of two persisted runs by config and descriptor statistics.
   * - **get_experiment_history**
     - Markdown table of every sweep run in the current session — handy for telling the story of how the agent iterated.
   * - **get_sweep_summary**
     - Returns the full persisted ``RunRecord`` for a given ``run_id``.
   * - **export_imputed_data**
     - Writes the selected imputed DataFrame to disk; CSV / Parquet / Feather inferred from the extension.

Example: A Mixed-Type Frame with Missing Values
-----------------------------------------------

Suppose ``demo.csv`` has 10 rows, two numeric columns, and one categorical
column, with about 20% missingness:

.. code-block:: text

   age,income,category
   25,50000,A
   30,,B
   ,75000,A
   45,80000,
   ...

A successful agent dialog looks like:

1. ``ingest_dataset("/data/demo.csv")`` → ``dataset_id="ds_abc123"``
2. ``characterize_dataset("ds_abc123")`` → reports 4 missing in ``income``,
   1 missing in ``category``, etc.
3. ``list_grids()`` → agent picks ``default``
4. ``create_config("ds_abc123", grid="default", samples=20)``
5. ``run_imputation_sweep`` → returns ``selected_index=7``,
   ``descriptor_stats.mean_pairwise_l2=0.14``
6. ``export_imputed_data("/data/demo_imputed.csv")``

The resulting CSV is the candidate Phil considered most representative of
the imputation ensemble — not the highest-likelihood, not the lowest-loss,
but the one closest to the centroid of the ECT descriptor cloud.

Bringing Your Own Data
----------------------

1. Ensure your CSV or Parquet file is accessible on the machine running
   the MCP server.
2. Connect the server using the setup steps above.
3. Ask: *"Run a Phil imputation sweep on* ``my_data.csv`` *and export
   the chosen imputation. Use the* ``finance`` *grid."*

The agent handles missingness analysis, grid selection, descriptor
scoring, and selection. Your job is to interpret the resulting imputed
dataset using domain knowledge.

.. seealso::

   - :doc:`/configuration` — programmatic YAML/config reference
   - :doc:`programmatic` — equivalent Python workflow

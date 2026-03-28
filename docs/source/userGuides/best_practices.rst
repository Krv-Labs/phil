.. _best_practices:

===============
Best Practices
===============

- Keep missingness patterns explicit in schema before fitting.
- Start with ``samples=5-20``.
- Prefer stable random seeds for reproducibility.
- Use ``assert``s in tests for downstream numeric invariants.

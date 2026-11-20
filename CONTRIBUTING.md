# Contributing

Scientific changes need a stated estimand, model assumptions, and validation against known
ground truth. A heatmap or a file-existence assertion is not sufficient evidence.

```bash
uv sync --all-groups
uv run ruff format .
uv run ruff check .
uv run pytest
uv run zensical build
uv build
```

Add deterministic tests for numerical changes. For stochastic recovery tests, use error
metrics and tolerances justified by sample size rather than exact floating-point equality.
Public APIs require type hints and NumPy-style docstrings. New dependencies need a concrete
reason and should remain optional when they serve only one integration.

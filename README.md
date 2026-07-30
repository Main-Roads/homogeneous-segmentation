# Homogeneous Segmentation

Homogeneous segmentation of linear spatial data such as pavement condition
indicators and traffic volumes.

This package ports two methods from the R package
[HS](https://cran.r-project.org/web/packages/HS/index.html):

- Spatial Heterogeneity Segmentation (SHS)
- Minimum Coefficient of Variation (MCV)

The cumulative difference approach (CDA) from the original R package is not
implemented in this repository.

## Installation

Install directly from the optimization branch:

```bash
pip install "git+https://github.com/Main-Roads/homogeneous-segmentation.git@feat/optimize-numpy-numba"
```

Or add it to a `uv`-managed project:

```bash
uv add "homogeneous-segmentation @ git+https://github.com/Main-Roads/homogeneous-segmentation.git@feat/optimize-numpy-numba"
```

## Development

Install the development tooling with `uv`:

```bash
uv sync --extra dev
```

Run linting, formatting, and type checking manually:

```bash
uv run ruff format src tests benchmarks
uv run ruff check src tests benchmarks
uv run python -m pyright
```

This repository uses an 88-character line length.

Install the local pre-commit hooks:

```bash
uv run pre-commit install
```

## Recommended Usage

For new code, prefer the optimized path by setting `legacy=False`. That keeps
the public API stable while allowing the package to choose the execution
strategy internally.

```python
from homogeneous_segmentation import (
    segment_ids_to_maximize_spatial_heterogeneity,
    segment_ids_to_minimize_coefficient_of_variation,
)
import pandas as pd
from io import StringIO

df = pd.read_csv(StringIO("""road,slk_from,slk_to,cwy,deflection,dirn
H001,0.00,0.01,L,179.37,L
H001,0.01,0.02,L,177.12,L
H001,0.02,0.03,L,179.06,L
H001,0.03,0.04,L,212.65,L
H001,0.04,0.05,L,175.35,L
H001,0.05,0.06,L,188.66,L
H001,0.06,0.07,L,188.31,L
H001,0.07,0.08,L,174.48,L
H001,0.08,0.09,L,210.28,L
H001,0.09,0.10,L,260.05,L
H001,0.10,0.11,L,228.83,L
H001,0.11,0.12,L,226.33,L
H001,0.12,0.13,L,245.53,L
H001,0.13,0.14,L,315.77,L
H001,0.14,0.15,L,373.86,L
H001,0.15,0.16,L,333.56,L"""))

df["seg.shs"] = segment_ids_to_maximize_spatial_heterogeneity(
    data=df,
    measure=("slk_from", "slk_to"),
    variable_column_names=["deflection"],
    allowed_segment_length_range=(0.030, 0.080),
    legacy=False,
)

df["seg.mcv"] = segment_ids_to_minimize_coefficient_of_variation(
    data=df,
    measure=("slk_from", "slk_to"),
    variable_column_names=["deflection"],
    allowed_segment_length_range=(0.030, 0.080),
    legacy=False,
)
```

Expected result:

```python
expected_result = pd.read_csv(StringIO("""road,slk_from,slk_to,cwy,deflection,dirn,seg.shs,seg.mcv
H001,0.00,0.01,L,179.37,L,1,1
H001,0.01,0.02,L,177.12,L,1,1
H001,0.02,0.03,L,179.06,L,1,1
H001,0.03,0.04,L,212.65,L,1,1
H001,0.04,0.05,L,175.35,L,2,2
H001,0.05,0.06,L,188.66,L,2,2
H001,0.06,0.07,L,188.31,L,2,2
H001,0.07,0.08,L,174.48,L,2,2
H001,0.08,0.09,L,210.28,L,2,2
H001,0.09,0.10,L,260.05,L,3,3
H001,0.10,0.11,L,228.83,L,3,3
H001,0.11,0.12,L,226.33,L,3,3
H001,0.12,0.13,L,245.53,L,3,3
H001,0.13,0.14,L,315.77,L,3,3
H001,0.14,0.15,L,373.86,L,3,3
H001,0.15,0.16,L,333.56,L,3,3
"""))

pd.testing.assert_frame_equal(
    left=df,
    right=expected_result,
    check_like=True,
)
```

## NaN And Index Behavior

Rows with missing values in `variable_column_names` are excluded before
segmentation.

The returned `pandas.Series`:

- is indexed by the original labels of the retained rows
- is sorted by the start measure column
- can be assigned back to the original DataFrame, leaving dropped rows as
  missing segment IDs

## Benchmarking

This repository includes a tracked benchmark script in
`benchmarks/bench.py`.

Run the full large-dataset benchmark:

```bash
uv run python3 benchmarks/bench.py --only-full-df2 --full-df2-repeats 3 --include-full-df2-legacy
```

The benchmark prints live progress for each repeat and reports segment counts
for each run.

Current full `df2` snapshot on the development machine:

- SHS legacy: about `116.3s`
- SHS optimized: about `1.90s` (`61x` speedup)
- MCV legacy: about `133.9s`
- MCV optimized: about `2.52s` (`53x` speedup)

These numbers are environment-dependent, but they show the intended usage:
for real workloads, prefer `legacy=False` and let the optimized path handle the
execution strategy in the background.

## Background And References

The original R package author is Yongze Song. The related paper is:

> Song, Yongze, Peng Wu, Daniel Gilmore, and Qindong Li.
> "A spatial heterogeneity-based segmentation model for analyzing road
> deterioration network data in multi-scale infrastructure systems."
> IEEE Transactions on Intelligent Transportation Systems (2020).

This project is not a complete port of the original R version. Only SHS and
MCV are implemented here.
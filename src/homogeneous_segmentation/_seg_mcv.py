"""
Implementation of the 'Minimum Coefficient of Variation' (MCV) homogeneous segmentation algorithm.
"""

from typing import Literal, Optional

import numpy as np
import pandas as pd

from ._cumulative_p import cumulative_p
from ._optimal_bisections import optimal_bisections

Backend = Literal["legacy", "auto", "numpy", "numba"]


def segment_ids_to_minimize_coefficient_of_variation(
    data: pd.DataFrame,
    measure: tuple[str, str],
    variable_column_names: list[str],
    allowed_segment_length_range: Optional[tuple[float, float]] = None,
    legacy: bool = True,
    backend: Optional[Backend] = None,
    memory_budget_bytes: Optional[int] = None,
) -> pd.Series:
    """Segment contiguous rows using the Minimum Coefficient of Variation criterion.

    Rows with missing values in `variable_column_names` are excluded before
    segmentation. The returned Series is indexed by the original labels of the
    retained rows, ordered by `measure[0]`.
    """

    if backend == "legacy":
        legacy = True
    elif backend is not None:
        legacy = False

    if not legacy:
        from ._opt_fast import segment_ids_to_minimize_coefficient_of_variation_fast

        return segment_ids_to_minimize_coefficient_of_variation_fast(
            data=data,
            measure=measure,
            variable_column_names=variable_column_names,
            allowed_segment_length_range=allowed_segment_length_range,
            backend="auto" if backend is None else backend,
            memory_budget_bytes=memory_budget_bytes,
        )

    LENGTH_COLUMN_NAME = "___length___"

    measure_start, measure_end = measure

    # preprocessing
    data = data.dropna(subset=variable_column_names).sort_values(by=measure_start)
    output_index = data.index.copy()
    if data.empty:
        return pd.Series(dtype=np.int64, index=output_index)

    # add length # remove system errors of small data
    data[LENGTH_COLUMN_NAME] = (
        (data[measure_end] - data[measure_start]).round(decimals=10).values
    )

    if allowed_segment_length_range is None:
        allowed_segment_length_range = (
            data[LENGTH_COLUMN_NAME].min(),
            data[LENGTH_COLUMN_NAME].sum(),
        )
    assert allowed_segment_length_range is not None

    if data[LENGTH_COLUMN_NAME].sum() <= allowed_segment_length_range[1]:
        return pd.Series(
            data=np.ones(len(data.index), dtype=np.int64),
            index=output_index,
        )
    ## Start original mcv function

    # n_var = len(var)
    min_allowed_length, max_allowed_length = allowed_segment_length_range

    # first segmentation
    ss = optimal_bisections(
        variables=data.loc[:, variable_column_names].values.transpose(),
        length=data[LENGTH_COLUMN_NAME].values,
        minimum_segment_length=min_allowed_length,
        cumulative_split_statistic=cumulative_p,
        goal="min",
    )
    # ss = np.array([3,5])
    # following segmentation
    k1 = np.array([0, *ss])
    k2 = np.array([*ss, len(data.index)])
    ll = k2 - k1  # integer length of arrays on each side of split
    split_boundaries = np.append(
        0, np.cumsum(ll)
    )  # split boundaries, including 0 and len(data.index)
    segment_id = np.repeat(np.arange(0, len(ll)), ll)  # segment id for each row of data
    # segdatalist = np.split(data, segid)

    # NOTE: We are grouping the data wherever _seg2 found a maximum Q value.
    # Generally there should be only a single maximum, but if there
    # is an equal maximum at two indices, then there will be more than 2 groups.
    # in practice this should be vanishingly rare...
    # but in future we should come back and prevent this.
    data_grouped_by_seg = data.groupby(segment_id)
    data_grouped_by_segment_id: list[pd.DataFrame] = [
        group for _, group in data.groupby(segment_id)
    ]

    # lengthdata        = pandas.DataFrame([data.loc[:,length], segid], columns = ["length","seg.id"]))
    segment_length_summary = data_grouped_by_seg[LENGTH_COLUMN_NAME].sum()
    segment_length = segment_length_summary.round(10).to_numpy(
        dtype=np.float64,
        copy=False,
    )
    k = np.flatnonzero(segment_length > max_allowed_length)

    while len(k) > 0:
        sa = [
            optimal_bisections(
                variables=data_grouped_by_segment_id[x]
                .loc[:, variable_column_names]
                .values.transpose(),
                length=data_grouped_by_segment_id[x][LENGTH_COLUMN_NAME].values,
                minimum_segment_length=min_allowed_length,
                cumulative_split_statistic=cumulative_p,
                goal="min",
            )
            + split_boundaries[x]
            for x in k
        ]
        ss = np.sort(np.concatenate([ss, *sa]))
        k1 = np.array([0, *ss])
        k2 = np.array([*ss, len(data.index)])
        ll = k2 - k1
        split_boundaries = np.append(np.array([0]), np.cumsum(ll))
        segment_id = np.repeat(np.arange(0, len(ll)), ll)
        # segdatalist = np.split(data, segid)

        # we are grouping the data wherever _seg2 found a maximum. generally there should be only a single maximum,
        # but if there is an equal maximum at two indices, then there will be 3 groups overall.
        data_grouped_by_segment_id: list[pd.DataFrame] = [
            group for _, group in data.groupby(segment_id)
        ]

        # lengthdata        = pandas.DataFrame([data.loc[:,length], segid], columns = ["length","seg.id"]))
        segment_length_summary = data[LENGTH_COLUMN_NAME].groupby(segment_id).sum()
        segment_length = segment_length_summary.round(10).to_numpy(
            dtype=np.float64,
            copy=False,
        )
        k = np.flatnonzero(segment_length > max_allowed_length)

    # # add seg.id
    k1 = np.array([0, *ss])
    k2 = np.array([*ss, len(data.index)])
    ll = k2 - k1
    segment_id = np.repeat(np.arange(0, len(ll), dtype=np.int64), ll) + 1
    ## End original shs function

    return pd.Series(index=output_index, data=segment_id)

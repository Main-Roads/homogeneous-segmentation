"""
NumPy-first optimized implementations for segmentation functions.

These keep the public API but avoid pandas group copies by operating on
NumPy arrays and index boundaries. They are intended to be used as an
optional fast path when `legacy=False` is passed to the public functions.
"""
from typing import Optional
import numpy as np
import pandas as pd
from ._optimal_bisections import optimal_bisections
from ._cumulative_q import cumulative_q
from ._cumulative_p import cumulative_p


def _prepare_data(data: pd.DataFrame, measure: tuple[str, str], variable_column_names: list[str]):
    measure_start, measure_end = measure
    original_index = data.index
    data = (
        data
        .dropna(subset=variable_column_names)
        .sort_values(by=measure_start)
        .reset_index(drop=True)
    )
    LENGTH_COLUMN_NAME = "___length___"
    data[LENGTH_COLUMN_NAME] = (data[measure_end] - data[measure_start]).round(decimals=10).values
    return data, original_index, LENGTH_COLUMN_NAME


def segment_ids_to_maximize_spatial_heterogeneity_fast(
    data: pd.DataFrame,
    measure: tuple[str, str],
    variable_column_names: list[str],
    allowed_segment_length_range: Optional[tuple[float, float]] = None,
):
    data, original_index, LENGTH_COLUMN_NAME = _prepare_data(data, measure, variable_column_names)

    if allowed_segment_length_range is None:
        allowed_segment_length_range = (
            data[LENGTH_COLUMN_NAME].min(),
            data[LENGTH_COLUMN_NAME].sum(),
        )

    if data[LENGTH_COLUMN_NAME].sum() <= allowed_segment_length_range[1]:
        return pd.Series(data=np.ones(len(data.index), dtype=np.int64), index=data.index)

    min_allowed_length, max_allowed_length = allowed_segment_length_range

    variables = data.loc[:, variable_column_names].values.transpose()  # shape: (n_vars, n_rows)
    lengths = data[LENGTH_COLUMN_NAME].values

    ss = optimal_bisections(
        variables=variables,
        length=lengths,
        minimum_segment_length=min_allowed_length,
        cumulative_split_statistic=cumulative_q,
        goal="max",
    )

    # build initial partition boundaries
    k1 = np.array([0, *ss])
    k2 = np.array([*ss, len(data.index)])
    ll = k2 - k1
    split_boundaries = np.append(0, np.cumsum(ll))
    segment_id = np.repeat(np.arange(0, len(ll)), ll)

    segment_length = np.round(np.add.reduceat(lengths, split_boundaries[:-1]), 10)
    k = np.flatnonzero(segment_length > max_allowed_length)

    while len(k) > 0:
        sa_list = []
        for x in k:
            start = int(split_boundaries[x])
            end = int(split_boundaries[x + 1])
            sub_ss = optimal_bisections(
                variables=variables[:, start:end],
                length=lengths[start:end],
                minimum_segment_length=min_allowed_length,
                cumulative_split_statistic=cumulative_q,
                goal="max",
            )
            if sub_ss.size:
                sa_list.append(sub_ss + split_boundaries[x])

        if sa_list:
            ss = np.sort(np.concatenate([ss, *sa_list]))

        k1 = np.array([0, *ss])
        k2 = np.array([*ss, len(data.index)])
        ll = k2 - k1
        split_boundaries = np.append(np.array([0]), np.cumsum(ll))
        segment_id = np.repeat(np.arange(0, len(ll)), ll)

        segment_length = np.round(np.add.reduceat(lengths, split_boundaries[:-1]), 10)
        k = np.flatnonzero(segment_length > max_allowed_length)

    k1 = np.array([0, *ss])
    k2 = np.array([*ss, len(data.index)])
    ll = k2 - k1
    segment_id = np.repeat(np.arange(0, len(ll), dtype=np.int64), ll) + 1

    return pd.Series(index=original_index, data=segment_id)


def segment_ids_to_minimize_coefficient_of_variation_fast(
    data: pd.DataFrame,
    measure: tuple[str, str],
    variable_column_names: list[str],
    allowed_segment_length_range: Optional[tuple[float, float]] = None,
):
    data, original_index, LENGTH_COLUMN_NAME = _prepare_data(data, measure, variable_column_names)

    if allowed_segment_length_range is None:
        allowed_segment_length_range = (
            data[LENGTH_COLUMN_NAME].min(),
            data[LENGTH_COLUMN_NAME].sum(),
        )

    if data[LENGTH_COLUMN_NAME].sum() <= allowed_segment_length_range[1]:
        return pd.Series(data=np.ones(len(data.index), dtype=np.int64), index=data.index)

    min_allowed_length, max_allowed_length = allowed_segment_length_range

    variables = data.loc[:, variable_column_names].values.transpose()  # shape: (n_vars, n_rows)
    lengths = data[LENGTH_COLUMN_NAME].values

    ss = optimal_bisections(
        variables=variables,
        length=lengths,
        minimum_segment_length=min_allowed_length,
        cumulative_split_statistic=cumulative_p,
        goal="min",
    )

    # build initial partition boundaries
    k1 = np.array([0, *ss])
    k2 = np.array([*ss, len(data.index)])
    ll = k2 - k1
    split_boundaries = np.append(0, np.cumsum(ll))
    segment_id = np.repeat(np.arange(0, len(ll)), ll)

    segment_length = np.round(np.add.reduceat(lengths, split_boundaries[:-1]), 10)
    k = np.flatnonzero(segment_length > max_allowed_length)

    while len(k) > 0:
        sa_list = []
        for x in k:
            start = int(split_boundaries[x])
            end = int(split_boundaries[x + 1])
            sub_ss = optimal_bisections(
                variables=variables[:, start:end],
                length=lengths[start:end],
                minimum_segment_length=min_allowed_length,
                cumulative_split_statistic=cumulative_p,
                goal="min",
            )
            if sub_ss.size:
                sa_list.append(sub_ss + split_boundaries[x])

        if sa_list:
            ss = np.sort(np.concatenate([ss, *sa_list]))

        k1 = np.array([0, *ss])
        k2 = np.array([*ss, len(data.index)])
        ll = k2 - k1
        split_boundaries = np.append(np.array([0]), np.cumsum(ll))
        segment_id = np.repeat(np.arange(0, len(ll)), ll)

        segment_length = np.round(np.add.reduceat(lengths, split_boundaries[:-1]), 10)
        k = np.flatnonzero(segment_length > max_allowed_length)

    k1 = np.array([0, *ss])
    k2 = np.array([*ss, len(data.index)])
    ll = k2 - k1
    segment_id = np.repeat(np.arange(0, len(ll), dtype=np.int64), ll) + 1

    return pd.Series(index=original_index, data=segment_id)

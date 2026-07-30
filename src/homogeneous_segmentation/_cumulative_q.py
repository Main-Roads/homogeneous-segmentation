"""
This is a private module containing the implementation of the cumulative `Q statistic`
which supports Spatial Heterogeneity-based Segmentation (SHS) method.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from numba import njit

Backend = Literal["numpy", "numba"]


def _as_2d_float64(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    array = np.asarray(data, dtype=np.float64)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    return np.ascontiguousarray(array)


def cumulative_q_legacy(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Original cumulative Q-statistic implementation for a single variable."""

    array = np.asarray(data, dtype=np.float64)
    n_rows = array.shape[0]
    if n_rows < 2:
        return np.empty(0, dtype=np.float64)

    cum_n = np.arange(1, n_rows, dtype=np.float64)
    data_left = array[:-1]
    data_right = array[:0:-1]
    cum_data_left = np.cumsum(data_left, dtype=np.float64)
    cum_data_right = np.cumsum(data_right, dtype=np.float64)[::-1]
    sumd = np.sum(array, dtype=np.float64)
    cum_datasquare_left = np.cumsum(data_left * data_left, dtype=np.float64)
    cum_datasquare_right = np.cumsum(data_right * data_right, dtype=np.float64)[::-1]

    with np.errstate(invalid="ignore", divide="ignore"):
        result = 1.0 - (
            (
                (cum_datasquare_left - cum_data_left * cum_data_left / cum_n)
                + (cum_datasquare_right - cum_data_right * cum_data_right / cum_n[::-1])
            )
            / (np.sum(array * array, dtype=np.float64) - sumd * sumd / n_rows)
        )
    return result


def cumulative_q_numpy_optimized(
    data: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Optimized NumPy implementation of the cumulative Q-statistic."""

    array = np.asarray(data, dtype=np.float64)
    n_rows = array.shape[0]
    if n_rows < 2:
        return np.empty(0, dtype=np.float64)

    squared = array * array
    cum_n = np.arange(1, n_rows, dtype=np.float64)
    cum_n_rev = n_rows - cum_n
    cum_sum = np.cumsum(array[:-1], dtype=np.float64)
    cum_sq = np.cumsum(squared[:-1], dtype=np.float64)
    total_sum = cum_sum[-1] + array[-1]
    total_sq = cum_sq[-1] + squared[-1]
    right_sum = total_sum - cum_sum
    right_sq = total_sq - cum_sq
    denominator = total_sq - total_sum * total_sum / n_rows

    with np.errstate(invalid="ignore", divide="ignore"):
        result = 1.0 - (
            (
                (cum_sq - cum_sum * cum_sum / cum_n)
                + (right_sq - right_sum * right_sum / cum_n_rev)
            )
            / denominator
        )
    return result


@njit(cache=True, fastmath=False)
def cumulative_q_numba_optimized(
    data: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Numba implementation of the cumulative Q-statistic for a single variable."""

    n_rows = data.shape[0]
    if n_rows < 2:
        return np.empty(0, dtype=np.float64)

    output = np.empty(n_rows - 1, dtype=np.float64)
    prefix_sum = np.empty(n_rows - 1, dtype=np.float64)
    prefix_sq_sum = np.empty(n_rows - 1, dtype=np.float64)

    total_sum = 0.0
    total_sq_sum = 0.0
    running_sum = 0.0
    running_sq_sum = 0.0

    for index in range(n_rows):
        value = data[index]
        total_sum += value
        total_sq_sum += value * value
        if index < n_rows - 1:
            running_sum += value
            running_sq_sum += value * value
            prefix_sum[index] = running_sum
            prefix_sq_sum[index] = running_sq_sum

    denominator = total_sq_sum - total_sum * total_sum / n_rows
    for index in range(n_rows - 1):
        left_n = index + 1.0
        right_n = n_rows - left_n
        left_sum = prefix_sum[index]
        left_sq_sum = prefix_sq_sum[index]
        right_sum = total_sum - left_sum
        right_sq_sum = total_sq_sum - left_sq_sum

        output[index] = 1.0 - (
            (
                (left_sq_sum - left_sum * left_sum / left_n)
                + (right_sq_sum - right_sum * right_sum / right_n)
            )
            / denominator
        )

    return output


def cumulative_q_matrix_legacy(
    data: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Original cumulative Q-statistic implementation for one or more variables."""

    array = _as_2d_float64(data)
    n_rows = array.shape[1]
    if n_rows < 2:
        return np.empty((array.shape[0], 0), dtype=np.float64)

    cum_n = np.arange(1, n_rows, dtype=np.float64)
    data_left = array[:, :-1]
    data_right = array[:, :0:-1]

    cum_data_left = np.cumsum(data_left, axis=1, dtype=np.float64)
    cum_data_right = np.cumsum(data_right, axis=1, dtype=np.float64)[:, ::-1]
    sumd = np.sum(array, axis=1)
    cum_datasquare_left = np.cumsum(data_left * data_left, axis=1, dtype=np.float64)
    cum_datasquare_right = np.cumsum(data_right * data_right, axis=1, dtype=np.float64)[
        :, ::-1
    ]
    denominator = np.sum(array * array, axis=1) - sumd * sumd / float(n_rows)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = 1.0 - (
            (
                (cum_datasquare_left - cum_data_left * cum_data_left / cum_n)
                + (cum_datasquare_right - cum_data_right * cum_data_right / cum_n[::-1])
            )
            / denominator[:, np.newaxis]
        )
    return result


def cumulative_q_matrix_numpy_optimized(
    data: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Optimized NumPy implementation of cumulative Q-statistics for one or more variables."""

    array = _as_2d_float64(data)
    n_rows = array.shape[1]
    if n_rows < 2:
        return np.empty((array.shape[0], 0), dtype=np.float64)

    squared = array * array
    cum_n = np.arange(1, n_rows, dtype=np.float64)
    cum_n_rev = n_rows - cum_n
    cum_sum = np.cumsum(array[:, :-1], axis=1, dtype=np.float64)
    cum_sq = np.cumsum(squared[:, :-1], axis=1, dtype=np.float64)
    total_sum = cum_sum[:, -1:] + array[:, -1:]
    total_sq = cum_sq[:, -1:] + squared[:, -1:]
    right_sum = total_sum - cum_sum
    right_sq = total_sq - cum_sq
    denominator = total_sq - total_sum * total_sum / float(n_rows)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = 1.0 - (
            (
                (cum_sq - cum_sum * cum_sum / cum_n)
                + (right_sq - right_sum * right_sum / cum_n_rev)
            )
            / denominator
        )
    return result


def cumulative_q_matrix_numba_optimized(
    data: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Numba implementation of cumulative Q-statistics for one or more variables."""

    array = _as_2d_float64(data)
    n_vars, n_rows = array.shape
    if n_rows < 2:
        return np.empty((n_vars, 0), dtype=np.float64)

    result = np.empty((n_vars, n_rows - 1), dtype=np.float64)
    for row_index in range(n_vars):
        result[row_index, :] = cumulative_q_numba_optimized(array[row_index])
    return result


def cumulative_q_matrix(
    data: npt.NDArray[np.float64],
    *,
    legacy: bool = True,
    backend: Backend = "numpy",
) -> npt.NDArray[np.float64]:
    """Compute cumulative Q-statistics for one or more variables at once.

    Args:
        data: Array of shape ``(n_vars, n_rows)`` or ``(n_rows,)``.

    Returns:
        Array of shape ``(n_vars, n_rows - 1)`` containing Q-statistics for each
        potential split position.
    """

    if legacy:
        return cumulative_q_matrix_legacy(data)
    if backend == "numba":
        return cumulative_q_matrix_numba_optimized(data)
    return cumulative_q_matrix_numpy_optimized(data)


def cumulative_q(
    data: npt.NDArray[np.float64],
    *,
    legacy: bool = True,
    backend: Backend = "numpy",
) -> npt.NDArray[np.float64]:
    """Computes the cumulative Q-statistic for each potential split index in an array.

    The Q-statistic is a measure that quantifies the importance of each variable by
    comparing the variance of the target variable across different spatial regions
    against the overall variance in the entire study area. It is used to identify
    optimal segmentation points in the dataset where the variance within segments is
    minimized, and the variance between segments is maximized.

    Formula:

    ```
    Q = 1 - Σ(N_v, j * σ_v, j ^ 2) / (N_v * σ_v ^ 2)
    ```

    Where:

    - `N_v` and `σ_v^2` are the number and population variance of observations within the whole study area.
    - `N_v,j` and `σ_v,j^2` are the number and population variance of observations within the j-th sub-region.
    - `M` is the total number of sub-regions.

    Args:
        data (npt.ArrayLike): An array-like object containing the data points for which the Q values are to be computed.

    Returns:
        npt.ArrayLike: An array of the same length as 'data', containing the computed Q values for each potential split
        point.
    """
    array = np.asarray(data, dtype=np.float64)
    if legacy:
        return cumulative_q_legacy(array)
    if backend == "numba":
        return cumulative_q_numba_optimized(array)
    return cumulative_q_numpy_optimized(array)

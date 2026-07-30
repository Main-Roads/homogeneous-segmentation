"""
This is a private module containing the implementation of the cumulative `P statistic`
which supports the Minimise Coefficient of Variation (MCV) method.
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


def cumulative_p_legacy(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Original cumulative P-statistic implementation for a single variable."""

    array = np.asarray(data, dtype=np.float64)
    n_rows = array.shape[0]
    if n_rows < 2:
        return np.empty(0, dtype=np.float64)

    cum_n = np.arange(1, n_rows, dtype=np.float64)
    cum_n_rev = cum_n[::-1]
    data_left = array[:-1]
    data_right = array[:0:-1]
    cum_data_left = np.cumsum(data_left, dtype=np.float64)
    cum_data_right = np.cumsum(data_right, dtype=np.float64)[::-1]
    cum_datasquare_left = np.cumsum(data_left * data_left, dtype=np.float64)
    cum_datasquare_right = np.cumsum(data_right * data_right, dtype=np.float64)[::-1]

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (
            np.sqrt(
                ((cum_n * cum_datasquare_left / (cum_data_left * cum_data_left)) - 1.0)
                * cum_n
                / (cum_n - 1.0)
            )
            + np.sqrt(
                (
                    (
                        cum_n_rev
                        * cum_datasquare_right
                        / (cum_data_right * cum_data_right)
                    )
                    - 1.0
                )
                * cum_n_rev
                / (cum_n_rev - 1.0)
            )
        ) / 2.0
    return result


def cumulative_p_numpy_optimized(
    data: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Optimized NumPy implementation of the cumulative P-statistic."""

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

    with np.errstate(invalid="ignore", divide="ignore"):
        left = np.sqrt(
            ((cum_n * cum_sq / (cum_sum * cum_sum)) - 1.0) * cum_n / (cum_n - 1.0)
        )
        right = np.sqrt(
            ((cum_n_rev * right_sq / (right_sum * right_sum)) - 1.0)
            * cum_n_rev
            / (cum_n_rev - 1.0)
        )
    return 0.5 * (left + right)


@njit(cache=True, fastmath=False)
def cumulative_p_numba_optimized(
    data: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Numba implementation of the cumulative P-statistic for a single variable."""

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

    for index in range(n_rows - 1):
        left_n = index + 1.0
        right_n = n_rows - left_n
        left_sum = prefix_sum[index]
        left_sq_sum = prefix_sq_sum[index]
        right_sum = total_sum - left_sum
        right_sq_sum = total_sq_sum - left_sq_sum

        if left_n <= 1.0:
            left_p = np.nan
        else:
            left_term = (
                ((left_n * left_sq_sum / (left_sum * left_sum)) - 1.0)
                * left_n
                / (left_n - 1.0)
            )
            left_p = np.sqrt(left_term)

        if right_n <= 1.0:
            right_p = np.nan
        else:
            right_term = (
                ((right_n * right_sq_sum / (right_sum * right_sum)) - 1.0)
                * right_n
                / (right_n - 1.0)
            )
            right_p = np.sqrt(right_term)

        output[index] = 0.5 * (left_p + right_p)

    return output


def cumulative_p_matrix_legacy(
    data: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Original cumulative P-statistic implementation for one or more variables."""

    array = _as_2d_float64(data)
    n_rows = array.shape[1]
    if n_rows < 2:
        return np.empty((array.shape[0], 0), dtype=np.float64)

    cum_n = np.arange(1, n_rows, dtype=np.float64)
    cum_n_rev = cum_n[::-1]
    data_left = array[:, :-1]
    data_right = array[:, :0:-1]
    cum_data_left = np.cumsum(data_left, axis=1, dtype=np.float64)
    cum_data_right = np.cumsum(data_right, axis=1, dtype=np.float64)[:, ::-1]
    cum_datasquare_left = np.cumsum(data_left * data_left, axis=1, dtype=np.float64)
    cum_datasquare_right = np.cumsum(data_right * data_right, axis=1, dtype=np.float64)[
        :, ::-1
    ]

    with np.errstate(invalid="ignore", divide="ignore"):
        left_term = np.sqrt(
            ((cum_n * cum_datasquare_left / (cum_data_left * cum_data_left)) - 1.0)
            * cum_n
            / (cum_n - 1.0)
        )
        right_term = np.sqrt(
            (
                (cum_n_rev * cum_datasquare_right / (cum_data_right * cum_data_right))
                - 1.0
            )
            * cum_n_rev
            / (cum_n_rev - 1.0)
        )
        result = (left_term + right_term) / 2.0
    return result


def cumulative_p_matrix_numpy_optimized(
    data: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Optimized NumPy implementation of cumulative P-statistics for one or more variables."""

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

    with np.errstate(invalid="ignore", divide="ignore"):
        left = np.sqrt(
            ((cum_n * cum_sq / (cum_sum * cum_sum)) - 1.0) * cum_n / (cum_n - 1.0)
        )
        right = np.sqrt(
            ((cum_n_rev * right_sq / (right_sum * right_sum)) - 1.0)
            * cum_n_rev
            / (cum_n_rev - 1.0)
        )
    return 0.5 * (left + right)


def cumulative_p_matrix_numba_optimized(
    data: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Numba implementation of cumulative P-statistics for one or more variables."""

    array = _as_2d_float64(data)
    n_vars, n_rows = array.shape
    if n_rows < 2:
        return np.empty((n_vars, 0), dtype=np.float64)

    result = np.empty((n_vars, n_rows - 1), dtype=np.float64)
    for row_index in range(n_vars):
        result[row_index, :] = cumulative_p_numba_optimized(array[row_index])
    return result


def cumulative_p_matrix(
    data: npt.NDArray[np.float64],
    *,
    legacy: bool = True,
    backend: Backend = "numpy",
) -> npt.NDArray[np.float64]:
    """Compute cumulative P-statistics for one or more variables at once."""

    if legacy:
        return cumulative_p_matrix_legacy(data)
    if backend == "numba":
        return cumulative_p_matrix_numba_optimized(data)
    return cumulative_p_matrix_numpy_optimized(data)


def cumulative_p(
    data: npt.NDArray[np.float64],
    *,
    legacy: bool = True,
    backend: Backend = "numpy",
) -> npt.NDArray[np.float64]:
    """Compute the cumulative P-statistic for each potential split index in an array."""

    array = np.asarray(data, dtype=np.float64)
    if legacy:
        return cumulative_p_legacy(array)
    if backend == "numba":
        return cumulative_p_numba_optimized(array)
    return cumulative_p_numpy_optimized(array)

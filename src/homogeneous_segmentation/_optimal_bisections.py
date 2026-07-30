from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import numpy.typing as npt
from numba import njit

from ._cumulative_p import (
    cumulative_p,
    cumulative_p_legacy,
    cumulative_p_matrix,
    cumulative_p_numba_optimized,
    cumulative_p_numpy_optimized,
)
from ._cumulative_q import (
    cumulative_q,
    cumulative_q_legacy,
    cumulative_q_matrix,
    cumulative_q_numba_optimized,
    cumulative_q_numpy_optimized,
)

_goal_functions = {"min": np.min, "max": np.max}

Backend = Literal["numpy", "numba"]

_Q_STATISTICS = (
    cumulative_q,
    cumulative_q_legacy,
    cumulative_q_numpy_optimized,
    cumulative_q_numba_optimized,
)

_P_STATISTICS = (
    cumulative_p,
    cumulative_p_legacy,
    cumulative_p_numpy_optimized,
    cumulative_p_numba_optimized,
)


def _select_best_candidate_positions(
    candidate_indices: npt.NDArray[np.int64],
    mean_objective: npt.NDArray[np.float64],
    goal: Literal["min", "max"],
) -> npt.NDArray[np.int64]:
    """Return all best split positions to preserve legacy tie behavior."""

    valid = ~np.isnan(mean_objective)
    if not np.any(valid):
        return np.empty(0, dtype=np.int64)

    objective_values = mean_objective[valid]
    best_value = (
        np.nanmin(objective_values) if goal == "min" else np.nanmax(objective_values)
    )
    return candidate_indices[mean_objective == best_value].astype(np.int64) + 1


def _as_2d_float64(
    variables: list[npt.NDArray[np.float64]] | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    array = np.asarray(variables, dtype=np.float64)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    return np.ascontiguousarray(array)


def _valid_split_mask(
    length: npt.NDArray[np.float64], minimum_segment_length: float
) -> npt.NDArray[np.bool_]:
    cumulative_length_left = np.cumsum(length)
    cumulative_length_right = np.cumsum(length[::-1])[::-1]

    k_mask = ~(
        (cumulative_length_left <= minimum_segment_length)
        | (cumulative_length_right <= minimum_segment_length)
    )
    expanded_false = (
        np.convolve((~k_mask).astype(np.int8), np.ones(3, dtype=np.int8), mode="same")
        > 0
    )
    return ~expanded_false


@njit(cache=True)
def _mean_objective_q_numba(
    variables: np.ndarray, candidate_mask: np.ndarray
) -> np.ndarray:
    n_vars, n_rows = variables.shape
    candidate_count = int(candidate_mask.sum())
    mean_objective = np.zeros(candidate_count, dtype=np.float64)

    for row in range(n_vars):
        total_sum = 0.0
        total_square_sum = 0.0
        for col in range(n_rows):
            value = variables[row, col]
            total_sum += value
            total_square_sum += value * value

        denominator = total_square_sum - (total_sum * total_sum / n_rows)
        left_sum = 0.0
        left_square_sum = 0.0
        out_index = 0

        for split_index in range(n_rows - 1):
            value = variables[row, split_index]
            left_sum += value
            left_square_sum += value * value

            if candidate_mask[split_index]:
                n_left = split_index + 1
                n_right = n_rows - n_left
                right_sum = total_sum - left_sum
                right_square_sum = total_square_sum - left_square_sum

                if denominator == 0.0:
                    score = np.nan
                else:
                    score = 1.0 - (
                        (
                            (left_square_sum - (left_sum * left_sum / n_left))
                            + (right_square_sum - (right_sum * right_sum / n_right))
                        )
                        / denominator
                    )
                mean_objective[out_index] += score
                out_index += 1

    return mean_objective / n_vars


@njit(cache=True)
def _mean_objective_p_numba(
    variables: np.ndarray, candidate_mask: np.ndarray
) -> np.ndarray:
    n_vars, n_rows = variables.shape
    candidate_count = int(candidate_mask.sum())
    mean_objective = np.zeros(candidate_count, dtype=np.float64)

    for row in range(n_vars):
        total_sum = 0.0
        total_square_sum = 0.0
        for col in range(n_rows):
            value = variables[row, col]
            total_sum += value
            total_square_sum += value * value

        left_sum = 0.0
        left_square_sum = 0.0
        out_index = 0

        for split_index in range(n_rows - 1):
            value = variables[row, split_index]
            left_sum += value
            left_square_sum += value * value

            if candidate_mask[split_index]:
                n_left = split_index + 1
                n_right = n_rows - n_left
                right_sum = total_sum - left_sum
                right_square_sum = total_square_sum - left_square_sum

                if n_left <= 1 or n_right <= 1 or left_sum == 0.0 or right_sum == 0.0:
                    score = np.nan
                else:
                    left_inner = (
                        ((n_left * left_square_sum / (left_sum * left_sum)) - 1.0)
                        * n_left
                        / (n_left - 1.0)
                    )
                    right_inner = (
                        ((n_right * right_square_sum / (right_sum * right_sum)) - 1.0)
                        * n_right
                        / (n_right - 1.0)
                    )
                    score = (np.sqrt(left_inner) + np.sqrt(right_inner)) / 2.0
                mean_objective[out_index] += score
                out_index += 1

    return mean_objective / n_vars


def _mean_objective_numpy(
    variables: npt.NDArray[np.float64],
    cumulative_split_statistic: Callable[
        [npt.NDArray[np.float64]], npt.NDArray[np.float64]
    ],
    candidate_mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    if cumulative_split_statistic in _Q_STATISTICS:
        return np.mean(
            cumulative_q_matrix(variables, legacy=False, backend="numpy")[
                :, candidate_mask
            ],
            axis=0,
        )
    if cumulative_split_statistic in _P_STATISTICS:
        return np.mean(
            cumulative_p_matrix(variables, legacy=False, backend="numpy")[
                :, candidate_mask
            ],
            axis=0,
        )

    candidate_count = int(candidate_mask.sum())
    objective_matrix = np.empty((variables.shape[0], candidate_count), dtype=np.float64)
    for row_index, variable in enumerate(variables):
        objective_matrix[row_index, :] = cumulative_split_statistic(variable)[
            candidate_mask
        ]
    return np.mean(objective_matrix, axis=0)


def _mean_objective_numba(
    variables: npt.NDArray[np.float64],
    cumulative_split_statistic: Callable[
        [npt.NDArray[np.float64]], npt.NDArray[np.float64]
    ],
    candidate_mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    if cumulative_split_statistic in _Q_STATISTICS:
        return _mean_objective_q_numba(variables, candidate_mask)
    if cumulative_split_statistic in _P_STATISTICS:
        return _mean_objective_p_numba(variables, candidate_mask)
    return _mean_objective_numpy(variables, cumulative_split_statistic, candidate_mask)


def optimal_bisections(
    variables: list[npt.NDArray[np.float64]],
    length: npt.NDArray[np.float64],
    minimum_segment_length: float,
    cumulative_split_statistic: Callable[
        [npt.NDArray[np.float64]], npt.NDArray[np.float64]
    ],
    goal: Literal["min", "max"] = "max",
    backend: Backend = "numpy",
) -> npt.NDArray[np.int64]:
    """
    Bisects the given data at either the minimum and maximum of the
    'cumulative_split_statistic' function.

    This function enforces a minimum

    Args:
        data (pandas.DataFrame): The DataFrame containing the data to be segmented.
        var (list[str]): A list of column names in 'data' that are to be used for segmentation.
        length_column (str): The name of the column that represents the length of each segment.
        min_length (float): Minimum allowed lengths for each segment.

    Returns:
        list[int]: A list of indices in 'data' where the maximum Q values are found, indicating optimal split points.
    """

    goal_function = _goal_functions.get(goal)
    if goal_function is None:
        raise ValueError(f"goal must be one of {list(_goal_functions.keys())}")

    variables_array = _as_2d_float64(variables)
    length_array = np.asarray(length, dtype=np.float64)

    if length_array.size < 2:
        return np.empty(0, dtype=np.int64)

    k_mask = _valid_split_mask(length_array, minimum_segment_length)
    candidate_mask = k_mask[1:]
    candidate_indices = np.flatnonzero(candidate_mask)
    if candidate_indices.size == 0:
        return np.empty(0, dtype=np.int64)

    if backend == "numba":
        mean_objective = _mean_objective_numba(
            variables_array,
            cumulative_split_statistic,
            candidate_mask,
        )
    else:
        mean_objective = _mean_objective_numpy(
            variables_array,
            cumulative_split_statistic,
            candidate_mask,
        )

    return _select_best_candidate_positions(candidate_indices, mean_objective, goal)

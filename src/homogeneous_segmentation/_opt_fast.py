"""Numba-first optimized implementations for segmentation functions."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from numba import njit

Backend = Literal["auto", "numpy", "numba"]
SelectedBackend = Literal["numpy", "numba"]
ObjectiveKind = Literal["p", "q"]
StorageMode = Literal["memory", "memmap"]

_CANDIDATE_EXPANSION_KERNEL = np.ones(3, dtype=np.int8)


@dataclass(frozen=True)
class BackendPlan:
    requested_backend: Backend
    selected_backend: SelectedBackend
    memory_budget_bytes: int
    estimated_prefix_bytes: int
    estimated_numpy_bytes: int
    storage_mode: StorageMode
    candidate_batch_size: int
    n_rows: int
    n_vars: int


@dataclass
class PrefixWorkspace:
    prefix_sum: np.ndarray
    prefix_sq: np.ndarray
    prefix_length: np.ndarray
    storage_mode: StorageMode
    temp_dir: tempfile.TemporaryDirectory | None = None

    def cleanup(self) -> None:
        if self.temp_dir is not None:
            self.temp_dir.cleanup()


def _default_memory_budget_bytes() -> int:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return 512 * 1024 * 1024
    total_memory = int(page_size) * int(phys_pages)
    return max(512 * 1024 * 1024, total_memory // 4)


def _estimate_prefix_bytes(n_rows: int, n_vars: int) -> int:
    return 8 * ((2 * max(1, n_vars) + 1) * (n_rows + 1))


def _estimate_numpy_backend_bytes(n_rows: int, n_vars: int) -> int:
    candidate_count = max(1, n_rows - 1)
    return (
        _estimate_prefix_bytes(n_rows, n_vars)
        + 8 * max(1, n_vars) * candidate_count * 6
    )


def _candidate_batch_size(n_rows: int, n_vars: int, memory_budget_bytes: int) -> int:
    candidate_count = max(1, n_rows - 1)
    per_candidate_bytes = max(1, n_vars) * 8 * 6
    return int(max(1, min(candidate_count, memory_budget_bytes // per_candidate_bytes)))


def _build_backend_plan(
    n_rows: int,
    n_vars: int,
    backend: Backend,
    memory_budget_bytes: Optional[int],
) -> BackendPlan:
    budget = memory_budget_bytes or _default_memory_budget_bytes()
    estimated_prefix_bytes = _estimate_prefix_bytes(n_rows, n_vars)
    estimated_numpy_bytes = _estimate_numpy_backend_bytes(n_rows, n_vars)

    if backend == "auto":
        selected_backend: SelectedBackend = "numba"
    else:
        selected_backend = backend

    storage_mode: StorageMode = (
        "memmap" if estimated_prefix_bytes > budget else "memory"
    )
    return BackendPlan(
        requested_backend=backend,
        selected_backend=selected_backend,
        memory_budget_bytes=budget,
        estimated_prefix_bytes=estimated_prefix_bytes,
        estimated_numpy_bytes=estimated_numpy_bytes,
        storage_mode=storage_mode,
        candidate_batch_size=_candidate_batch_size(n_rows, n_vars, budget),
        n_rows=n_rows,
        n_vars=n_vars,
    )


def plan_backend_for_data(
    data: pd.DataFrame,
    measure: tuple[str, str],
    variable_column_names: list[str],
    backend: Backend = "auto",
    memory_budget_bytes: Optional[int] = None,
) -> BackendPlan:
    prepared, _, _ = _prepare_data(data, measure, variable_column_names)
    return _build_backend_plan(
        n_rows=len(prepared.index),
        n_vars=len(variable_column_names),
        backend=backend,
        memory_budget_bytes=memory_budget_bytes,
    )


def _prepare_data(
    data: pd.DataFrame,
    measure: tuple[str, str],
    variable_column_names: list[str],
) -> tuple[pd.DataFrame, np.ndarray, pd.Index]:
    measure_start, measure_end = measure
    prepared = data.dropna(subset=variable_column_names).sort_values(by=measure_start)
    output_index = prepared.index.copy()
    if prepared.empty:
        return prepared, np.empty(0, dtype=np.float64), output_index

    lengths = (
        prepared[measure_end].to_numpy(dtype=np.float64, copy=False)
        - prepared[measure_start].to_numpy(dtype=np.float64, copy=False)
    ).round(decimals=10)
    return prepared, lengths, output_index


def _allocate_buffer(
    shape: tuple[int, ...],
    storage_mode: StorageMode,
    temp_dir: tempfile.TemporaryDirectory | None,
    name: str,
) -> np.ndarray:
    if storage_mode == "memory":
        return np.empty(shape, dtype=np.float64)

    if temp_dir is None:
        raise ValueError("temp_dir is required for memmap storage")
    path = os.path.join(temp_dir.name, f"{name}.dat")
    return np.memmap(path, dtype=np.float64, mode="w+", shape=shape)


def _build_prefix_workspace(
    prepared: pd.DataFrame,
    variable_column_names: list[str],
    lengths: np.ndarray,
    plan: BackendPlan,
) -> PrefixWorkspace:
    temp_dir = (
        tempfile.TemporaryDirectory(prefix="homogeneous-segmentation-")
        if plan.storage_mode == "memmap"
        else None
    )
    prefix_sum = _allocate_buffer(
        (plan.n_vars, plan.n_rows + 1), plan.storage_mode, temp_dir, "prefix_sum"
    )
    prefix_sq = _allocate_buffer(
        (plan.n_vars, plan.n_rows + 1), plan.storage_mode, temp_dir, "prefix_sq"
    )
    prefix_length = _allocate_buffer(
        (plan.n_rows + 1,), plan.storage_mode, temp_dir, "prefix_length"
    )

    prefix_length[0] = 0.0
    np.cumsum(lengths, dtype=np.float64, out=prefix_length[1:])

    for row_index, column_name in enumerate(variable_column_names):
        values = prepared[column_name].to_numpy(dtype=np.float64, copy=False)
        prefix_sum[row_index, 0] = 0.0
        prefix_sq[row_index, 0] = 0.0
        np.cumsum(values, dtype=np.float64, out=prefix_sum[row_index, 1:])
        squared = np.multiply(values, values, dtype=np.float64)
        np.cumsum(squared, dtype=np.float64, out=prefix_sq[row_index, 1:])

    return PrefixWorkspace(
        prefix_sum=prefix_sum,
        prefix_sq=prefix_sq,
        prefix_length=prefix_length,
        storage_mode=plan.storage_mode,
        temp_dir=temp_dir,
    )


def _candidate_positions(
    lengths: np.ndarray,
    start: int,
    end: int,
    minimum_segment_length: float,
) -> np.ndarray:
    n_rows = end - start
    if n_rows < 2:
        return np.empty(0, dtype=np.int64)

    segment_lengths = lengths[start:end]
    left_lengths = np.cumsum(segment_lengths, dtype=np.float64)
    right_lengths = np.cumsum(segment_lengths[::-1], dtype=np.float64)[::-1]
    k_mask = (left_lengths > minimum_segment_length) & (
        right_lengths > minimum_segment_length
    )
    expanded_false = (
        np.convolve((~k_mask).astype(np.int8), _CANDIDATE_EXPANSION_KERNEL, mode="same")
        > 0
    )
    candidate_mask = (~expanded_false)[1:]
    return np.flatnonzero(candidate_mask).astype(np.int64) + start + 1


@njit(cache=True)
def _mean_objective_q_numba_prefix(
    prefix_sum: np.ndarray,
    prefix_sq: np.ndarray,
    candidate_positions: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    n_vars = prefix_sum.shape[0]
    candidate_count = candidate_positions.shape[0]
    mean_objective = np.empty(candidate_count, dtype=np.float64)
    segment_n = float(end - start)

    for candidate_index in range(candidate_count):
        position = int(candidate_positions[candidate_index])
        left_n = float(position - start)
        right_n = float(end - position)
        score_sum = 0.0
        has_nan = False

        for row_index in range(n_vars):
            total_sum = prefix_sum[row_index, end] - prefix_sum[row_index, start]
            total_sq = prefix_sq[row_index, end] - prefix_sq[row_index, start]
            left_sum = prefix_sum[row_index, position] - prefix_sum[row_index, start]
            left_sq = prefix_sq[row_index, position] - prefix_sq[row_index, start]
            right_sum = total_sum - left_sum
            right_sq = total_sq - left_sq
            denominator = total_sq - total_sum * total_sum / segment_n

            if denominator == 0.0:
                has_nan = True
                break

            score = 1.0 - (
                (
                    (left_sq - left_sum * left_sum / left_n)
                    + (right_sq - right_sum * right_sum / right_n)
                )
                / denominator
            )
            if np.isnan(score):
                has_nan = True
                break
            score_sum += score

        mean_objective[candidate_index] = np.nan if has_nan else score_sum / n_vars

    return mean_objective


@njit(cache=True)
def _mean_objective_p_numba_prefix(
    prefix_sum: np.ndarray,
    prefix_sq: np.ndarray,
    candidate_positions: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    n_vars = prefix_sum.shape[0]
    candidate_count = candidate_positions.shape[0]
    mean_objective = np.empty(candidate_count, dtype=np.float64)

    for candidate_index in range(candidate_count):
        position = int(candidate_positions[candidate_index])
        left_n = float(position - start)
        right_n = float(end - position)
        score_sum = 0.0
        has_nan = False

        for row_index in range(n_vars):
            total_sum = prefix_sum[row_index, end] - prefix_sum[row_index, start]
            total_sq = prefix_sq[row_index, end] - prefix_sq[row_index, start]
            left_sum = prefix_sum[row_index, position] - prefix_sum[row_index, start]
            left_sq = prefix_sq[row_index, position] - prefix_sq[row_index, start]
            right_sum = total_sum - left_sum
            right_sq = total_sq - left_sq

            if left_n <= 1.0 or right_n <= 1.0 or left_sum == 0.0 or right_sum == 0.0:
                has_nan = True
                break

            left_inner = (
                ((left_n * left_sq / (left_sum * left_sum)) - 1.0)
                * left_n
                / (left_n - 1.0)
            )
            right_inner = (
                ((right_n * right_sq / (right_sum * right_sum)) - 1.0)
                * right_n
                / (right_n - 1.0)
            )
            if left_inner < 0.0 or right_inner < 0.0:
                has_nan = True
                break

            score = 0.5 * (np.sqrt(left_inner) + np.sqrt(right_inner))
            if np.isnan(score):
                has_nan = True
                break
            score_sum += score

        mean_objective[candidate_index] = np.nan if has_nan else score_sum / n_vars

    return mean_objective


def _mean_objective_numpy_prefix(
    workspace: PrefixWorkspace,
    candidate_positions: np.ndarray,
    start: int,
    end: int,
    objective: ObjectiveKind,
    plan: BackendPlan,
) -> np.ndarray:
    candidate_count = candidate_positions.size
    mean_objective = np.empty(candidate_count, dtype=np.float64)
    segment_n = float(end - start)
    total_sum = (
        workspace.prefix_sum[:, end : end + 1]
        - workspace.prefix_sum[:, start : start + 1]
    )
    total_sq = (
        workspace.prefix_sq[:, end : end + 1]
        - workspace.prefix_sq[:, start : start + 1]
    )
    denominator = None
    if objective == "q":
        denominator = total_sq - total_sum * total_sum / segment_n

    for batch_start in range(0, candidate_count, plan.candidate_batch_size):
        batch_stop = min(candidate_count, batch_start + plan.candidate_batch_size)
        positions = candidate_positions[batch_start:batch_stop]
        left_n = (positions - start).astype(np.float64)
        right_n = (end - positions).astype(np.float64)
        left_sum = (
            workspace.prefix_sum[:, positions]
            - workspace.prefix_sum[:, start : start + 1]
        )
        left_sq = (
            workspace.prefix_sq[:, positions]
            - workspace.prefix_sq[:, start : start + 1]
        )
        right_sum = total_sum - left_sum
        right_sq = total_sq - left_sq

        with np.errstate(invalid="ignore", divide="ignore"):
            if objective == "q":
                assert denominator is not None
                scores = 1.0 - (
                    (
                        (left_sq - left_sum * left_sum / left_n)
                        + (right_sq - right_sum * right_sum / right_n)
                    )
                    / denominator
                )
            else:
                left = np.sqrt(
                    ((left_n * left_sq / (left_sum * left_sum)) - 1.0)
                    * left_n
                    / (left_n - 1.0)
                )
                right = np.sqrt(
                    ((right_n * right_sq / (right_sum * right_sum)) - 1.0)
                    * right_n
                    / (right_n - 1.0)
                )
                scores = 0.5 * (left + right)

        mean_objective[batch_start:batch_stop] = np.mean(scores, axis=0)

    return mean_objective


def _optimal_bisections_from_prefix(
    workspace: PrefixWorkspace,
    lengths: np.ndarray,
    start: int,
    end: int,
    minimum_segment_length: float,
    objective: ObjectiveKind,
    goal: Literal["min", "max"],
    plan: BackendPlan,
) -> np.ndarray:
    candidate_positions = _candidate_positions(
        lengths, start, end, minimum_segment_length
    )
    if candidate_positions.size == 0:
        return np.empty(0, dtype=np.int64)

    if plan.selected_backend == "numba":
        if objective == "q":
            mean_objective = _mean_objective_q_numba_prefix(
                workspace.prefix_sum,
                workspace.prefix_sq,
                candidate_positions,
                start,
                end,
            )
        else:
            mean_objective = _mean_objective_p_numba_prefix(
                workspace.prefix_sum,
                workspace.prefix_sq,
                candidate_positions,
                start,
                end,
            )
    else:
        mean_objective = _mean_objective_numpy_prefix(
            workspace,
            candidate_positions,
            start,
            end,
            objective,
            plan,
        )

    valid = ~np.isnan(mean_objective)
    if not np.any(valid):
        return np.empty(0, dtype=np.int64)

    objective_values = mean_objective[valid]
    best_value = (
        np.nanmin(objective_values) if goal == "min" else np.nanmax(objective_values)
    )
    return candidate_positions[mean_objective == best_value]


def _segment_ids_fast(
    *,
    prepared: pd.DataFrame,
    variable_column_names: list[str],
    lengths: np.ndarray,
    output_index: pd.Index,
    allowed_segment_length_range: Optional[tuple[float, float]],
    objective: ObjectiveKind,
    goal: Literal["min", "max"],
    backend: Backend,
    memory_budget_bytes: Optional[int],
) -> pd.Series:
    n_rows = lengths.size
    if n_rows == 0:
        return pd.Series(dtype=np.int64, index=output_index)

    if allowed_segment_length_range is None:
        allowed_segment_length_range = (
            float(lengths.min()),
            float(lengths.sum()),
        )

    total_length = float(lengths.sum())
    if total_length <= allowed_segment_length_range[1]:
        return pd.Series(data=np.ones(n_rows, dtype=np.int64), index=output_index)

    min_allowed_length, max_allowed_length = allowed_segment_length_range
    plan = _build_backend_plan(
        n_rows=n_rows,
        n_vars=len(variable_column_names),
        backend=backend,
        memory_budget_bytes=memory_budget_bytes,
    )
    workspace = _build_prefix_workspace(prepared, variable_column_names, lengths, plan)

    try:
        pending_segments: list[tuple[int, int]] = [(0, n_rows)]
        split_positions: list[int] = []

        while pending_segments:
            start, end = pending_segments.pop()
            segment_length = float(np.round(lengths[start:end].sum(), 10))
            if segment_length <= max_allowed_length:
                continue

            best_positions = _optimal_bisections_from_prefix(
                workspace=workspace,
                lengths=lengths,
                start=start,
                end=end,
                minimum_segment_length=min_allowed_length,
                objective=objective,
                goal=goal,
                plan=plan,
            )
            if best_positions.size == 0:
                continue

            split_positions.extend(int(position) for position in best_positions)
            child_boundaries = np.concatenate(
                (
                    np.array([start], dtype=np.int64),
                    best_positions,
                    np.array([end], dtype=np.int64),
                )
            )
            for boundary_index in range(child_boundaries.size - 1):
                child_start = int(child_boundaries[boundary_index])
                child_end = int(child_boundaries[boundary_index + 1])
                child_length = float(np.round(lengths[child_start:child_end].sum(), 10))
                if child_length > max_allowed_length:
                    pending_segments.append((child_start, child_end))

        if split_positions:
            final_splits = np.array(sorted(split_positions), dtype=np.int64)
            final_boundaries = np.concatenate(
                (
                    np.array([0], dtype=np.int64),
                    final_splits,
                    np.array([n_rows], dtype=np.int64),
                )
            )
        else:
            final_boundaries = np.array([0, n_rows], dtype=np.int64)

        segment_lengths = np.diff(final_boundaries)
        segment_id = np.repeat(
            np.arange(1, len(segment_lengths) + 1, dtype=np.int64), segment_lengths
        )
        return pd.Series(index=output_index, data=segment_id)
    finally:
        workspace.cleanup()


def segment_ids_to_maximize_spatial_heterogeneity_fast(
    data: pd.DataFrame,
    measure: tuple[str, str],
    variable_column_names: list[str],
    allowed_segment_length_range: Optional[tuple[float, float]] = None,
    backend: Backend = "auto",
    memory_budget_bytes: Optional[int] = None,
):
    prepared, lengths, output_index = _prepare_data(
        data, measure, variable_column_names
    )
    return _segment_ids_fast(
        prepared=prepared,
        variable_column_names=variable_column_names,
        lengths=lengths,
        output_index=output_index,
        allowed_segment_length_range=allowed_segment_length_range,
        objective="q",
        goal="max",
        backend=backend,
        memory_budget_bytes=memory_budget_bytes,
    )


def segment_ids_to_minimize_coefficient_of_variation_fast(
    data: pd.DataFrame,
    measure: tuple[str, str],
    variable_column_names: list[str],
    allowed_segment_length_range: Optional[tuple[float, float]] = None,
    backend: Backend = "auto",
    memory_budget_bytes: Optional[int] = None,
):
    prepared, lengths, output_index = _prepare_data(
        data, measure, variable_column_names
    )
    return _segment_ids_fast(
        prepared=prepared,
        variable_column_names=variable_column_names,
        lengths=lengths,
        output_index=output_index,
        allowed_segment_length_range=allowed_segment_length_range,
        objective="p",
        goal="min",
        backend=backend,
        memory_budget_bytes=memory_budget_bytes,
    )

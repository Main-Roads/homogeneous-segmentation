#!/usr/bin/env python3
"""Benchmark legacy and optimized segmentation paths on synthetic and real data.

Usage: python benchmarks/bench.py [-n ROWS] [-r REPEATS]

This script does not modify the repository or commit changes.
"""

import argparse
import gc
import time
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd

from homogeneous_segmentation import (
    segment_ids_to_maximize_spatial_heterogeneity,
    segment_ids_to_minimize_coefficient_of_variation,
)
from homogeneous_segmentation._opt_fast import plan_backend_for_data

DF2_PATH = Path("tests/test_data/df2.csv")
SYNTHETIC_SEGMENT_RANGE = (0.030, 0.080)
REAL_SEGMENT_RANGE = (0.050, 0.200)
MEASURE = ("slk_from", "slk_to")
VARIABLES = ["deflection"]


class BenchmarkResult(TypedDict):
    mean_seconds: float
    segment_count: int
    selected_backend: str


def make_df(n_rows: int):
    slk_from = np.linspace(0.0, float(n_rows) * 0.01, n_rows, endpoint=False)
    slk_to = slk_from + 0.01
    rng = np.random.default_rng(12345)
    deflection = rng.normal(loc=200.0, scale=50.0, size=n_rows)
    return pd.DataFrame(
        {
            "road": "H001",
            "slk_from": slk_from,
            "slk_to": slk_to,
            "deflection": deflection,
        }
    )


def load_df2_subset() -> pd.DataFrame:
    df = pd.read_csv(DF2_PATH)
    return df.loc[
        (df["road"] == "H001")
        & (df["cwy"] == "S")
        & (df["dirn"] == "L")
        & (df["slk_from"] > 50)
        & (df["slk_from"] < 60)
    ].copy()


def load_df2_full() -> pd.DataFrame:
    return pd.read_csv(DF2_PATH)


def time_func(func, *args, repeats=3, **kwargs):
    times = []
    last_result = None
    progress_label = kwargs.pop("_progress_label", None)
    for repeat_index in range(repeats):
        gc.collect()
        if progress_label is not None:
            print(
                f"Starting {progress_label} repeat {repeat_index + 1}/{repeats}...",
                flush=True,
            )
        t0 = time.perf_counter()
        last_result = func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if progress_label is not None:
            print(
                f"Finished {progress_label} repeat {repeat_index + 1}/{repeats} in {times[-1]:.4f}s",
                flush=True,
            )
    return times, last_result


def segment_count(result: pd.Series) -> int:
    if len(result.index) == 0:
        return 0
    return int(result.max())


def describe_backend(
    data: pd.DataFrame,
    *,
    backend: str,
    memory_budget_bytes: int | None,
) -> str:
    if backend == "legacy":
        return "legacy"

    plan = plan_backend_for_data(
        data=data,
        measure=MEASURE,
        variable_column_names=VARIABLES,
        backend=backend,
        memory_budget_bytes=memory_budget_bytes,
    )
    prefix_mib = plan.estimated_prefix_bytes / (1024 * 1024)
    return f"{plan.selected_backend}/{plan.storage_mode} ({prefix_mib:.1f} MiB prefix)"


def benchmark_variant(
    func,
    data: pd.DataFrame,
    *,
    case_name: str,
    algorithm_name: str,
    label: str,
    segment_range: tuple[float, float],
    repeats: int,
    memory_budget_bytes: int | None,
):
    kwargs = {
        "data": data,
        "measure": MEASURE,
        "variable_column_names": VARIABLES,
        "allowed_segment_length_range": segment_range,
    }
    if label == "legacy":
        kwargs["legacy"] = True
    else:
        kwargs["legacy"] = False
        kwargs["backend"] = label
        kwargs["memory_budget_bytes"] = memory_budget_bytes

    selected_backend = describe_backend(
        data,
        backend=label,
        memory_budget_bytes=memory_budget_bytes,
    )
    progress_label = f"{case_name} {algorithm_name} {label} [{selected_backend}]"
    times, result = time_func(
        func, repeats=repeats, _progress_label=progress_label, **kwargs
    )
    return {
        "mean_seconds": float(np.mean(times)),
        "segment_count": segment_count(result),
        "selected_backend": selected_backend,
    }


def warm_backends(
    data: pd.DataFrame,
    segment_range: tuple[float, float],
    memory_budget_bytes: int | None,
) -> None:
    segment_ids_to_maximize_spatial_heterogeneity(
        data,
        MEASURE,
        VARIABLES,
        allowed_segment_length_range=segment_range,
        legacy=False,
        backend="numba",
        memory_budget_bytes=memory_budget_bytes,
    )
    segment_ids_to_minimize_coefficient_of_variation(
        data,
        MEASURE,
        VARIABLES,
        allowed_segment_length_range=segment_range,
        legacy=False,
        backend="numba",
        memory_budget_bytes=memory_budget_bytes,
    )


def summarize_case(
    name: str,
    rows: int,
    repeats: int,
    segment_range: tuple[float, float],
    shs_results: dict[str, BenchmarkResult],
    mcv_results: dict[str, BenchmarkResult],
) -> None:
    print(f"\n{name} ({rows} rows, {repeats} repeats, range={segment_range}):")

    for algorithm_name, results in (("SHS", shs_results), ("MCV", mcv_results)):
        legacy_mean = results.get("legacy", {}).get("mean_seconds")
        print(f"  {algorithm_name}:")
        for label in results:
            mean_seconds = results[label]["mean_seconds"]
            selected_backend = results[label]["selected_backend"]
            segments = results[label]["segment_count"]
            if label == "legacy":
                print(
                    f"    {label:<6} selected={selected_backend:<26} segments={segments:<6} mean={mean_seconds:.4f}s"
                )
            else:
                line = f"    {label:<6} selected={selected_backend:<26} segments={segments:<6} mean={mean_seconds:.4f}s"
                if legacy_mean is not None:
                    speedup = legacy_mean / mean_seconds
                    line += f" speedup={speedup:.2f}x"
                print(line)


def run_case(
    name: str,
    data: pd.DataFrame,
    *,
    segment_range: tuple[float, float],
    repeats: int,
    memory_budget_bytes: int | None,
    include_legacy: bool,
) -> None:
    print(f"\nPreparing {name} benchmark...")
    warm_backends(data, segment_range, memory_budget_bytes)

    labels = ["auto", "numpy", "numba"]
    if include_legacy:
        labels.insert(0, "legacy")

    shs_results: dict[str, BenchmarkResult] = {
        label: benchmark_variant(
            segment_ids_to_maximize_spatial_heterogeneity,
            data,
            case_name=name,
            algorithm_name="SHS",
            label=label,
            segment_range=segment_range,
            repeats=repeats,
            memory_budget_bytes=memory_budget_bytes,
        )
        for label in labels
    }
    mcv_results: dict[str, BenchmarkResult] = {
        label: benchmark_variant(
            segment_ids_to_minimize_coefficient_of_variation,
            data,
            case_name=name,
            algorithm_name="MCV",
            label=label,
            segment_range=segment_range,
            repeats=repeats,
            memory_budget_bytes=memory_budget_bytes,
        )
        for label in labels
    }

    summarize_case(
        name, len(data.index), repeats, segment_range, shs_results, mcv_results
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--rows", type=int, default=20000)
    parser.add_argument("-r", "--repeats", type=int, default=3)
    parser.add_argument("--real-repeats", type=int, default=3)
    parser.add_argument("--full-df2-repeats", type=int, default=1)
    parser.add_argument("--memory-budget-mib", type=int, default=None)
    parser.add_argument("--skip-full-df2", action="store_true")
    parser.add_argument("--include-full-df2-legacy", action="store_true")
    parser.add_argument("--only-full-df2", action="store_true")
    args = parser.parse_args()

    memory_budget_bytes = None
    if args.memory_budget_mib is not None:
        memory_budget_bytes = args.memory_budget_mib * 1024 * 1024

    if not args.only_full_df2:
        print(f"Generating synthetic dataframe with {args.rows} rows...")
        run_case(
            "Synthetic",
            make_df(args.rows),
            segment_range=SYNTHETIC_SEGMENT_RANGE,
            repeats=args.repeats,
            memory_budget_bytes=memory_budget_bytes,
            include_legacy=True,
        )

        print(f"Loading filtered subset from {DF2_PATH}...")
        run_case(
            "df2 subset",
            load_df2_subset(),
            segment_range=REAL_SEGMENT_RANGE,
            repeats=args.real_repeats,
            memory_budget_bytes=memory_budget_bytes,
            include_legacy=True,
        )

    if not args.skip_full_df2:
        print(f"Loading full dataset from {DF2_PATH}...")
        run_case(
            "df2 full",
            load_df2_full(),
            segment_range=REAL_SEGMENT_RANGE,
            repeats=args.full_df2_repeats,
            memory_budget_bytes=memory_budget_bytes,
            include_legacy=args.include_full_df2_legacy,
        )


if __name__ == "__main__":
    main()

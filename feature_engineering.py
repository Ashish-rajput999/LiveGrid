#!/usr/bin/env python3
"""
LiveGrid Phase 2 — Feature Engineering

Reads the raw training data from generate_dataset.py and computes
time-series features and the binary prediction label for ML training.

Features computed:
    - load_ratio_rolling_mean_5: 5-tick rolling mean of load_ratio
    - load_ratio_rolling_std_5: 5-tick rolling std of load_ratio
    - load_ratio_delta: tick-over-tick change in load_ratio
    - neighbor_avg_load: average load_ratio of neighbor nodes at same tick
    - ticks_since_warning: ticks since last WARNING state

Label:
    - will_fail_within_10_ticks: 1 if node fails within next 10 ticks

Usage:
    python feature_engineering.py

Input:
    output/training_data.csv
    output/neighbor_map.json

Output:
    output/features.csv
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd


# ── Constants ────────────────────────────────────────────────────────

INPUT_CSV = os.path.join("output", "training_data.csv")
NEIGHBOR_MAP_JSON = os.path.join("output", "neighbor_map.json")
OUTPUT_CSV = os.path.join("output", "features.csv")

ROLLING_WINDOW = 5
LABEL_HORIZON = 10


def load_data() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Load the raw training data and neighbor adjacency map.

    Returns:
        Tuple of (DataFrame with training data, neighbor map dict).

    Raises:
        FileNotFoundError: If input files don't exist.
    """
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Training data not found at '{INPUT_CSV}'. "
            f"Run generate_dataset.py first."
        )
    if not os.path.exists(NEIGHBOR_MAP_JSON):
        raise FileNotFoundError(
            f"Neighbor map not found at '{NEIGHBOR_MAP_JSON}'. "
            f"Run generate_dataset.py first."
        )

    print(f"  Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    with open(NEIGHBOR_MAP_JSON) as f:
        neighbor_map = json.load(f)
    print(f"  Loaded neighbor map ({len(neighbor_map)} nodes)")

    return df, neighbor_map


def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling statistical features for load_ratio.

    For each (run_id, node_id) group, computes:
    - load_ratio_rolling_mean_5: 5-tick rolling mean
    - load_ratio_rolling_std_5: 5-tick rolling std
    - load_ratio_delta: tick-over-tick change

    Args:
        df: DataFrame sorted by (run_id, node_id, tick).

    Returns:
        DataFrame with new feature columns added.
    """
    print("  Computing rolling features...")

    # Sort to ensure correct time ordering within groups
    df = df.sort_values(["run_id", "node_id", "tick"]).reset_index(drop=True)

    # Group by (run_id, node_id) for per-node per-run rolling calculations
    grouped = df.groupby(["run_id", "node_id"])["load_ratio"]

    # Rolling mean and std (window=5, min_periods=1 to avoid NaN at start)
    df["load_ratio_rolling_mean_5"] = grouped.transform(
        lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    ).round(4)

    df["load_ratio_rolling_std_5"] = grouped.transform(
        lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).std().fillna(0)
    ).round(4)

    # Delta: change from previous tick
    df["load_ratio_delta"] = grouped.transform(
        lambda x: x.diff().fillna(0)
    ).round(4)

    return df


def compute_neighbor_features(
    df: pd.DataFrame,
    neighbor_map: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Compute the average load_ratio of each node's neighbors at each tick.

    For each (run_id, tick, node_id), looks up the node's neighbors
    from the adjacency map and computes the mean of their load_ratios
    at the same tick.

    Args:
        df: DataFrame with load_ratio column.
        neighbor_map: Adjacency dict {node_id: [neighbor_ids]}.

    Returns:
        DataFrame with 'neighbor_avg_load' column added.
    """
    print("  Computing neighbor features...")

    # Build a fast lookup: (run_id, tick, node_id) -> load_ratio
    # Use a pivot approach for efficiency
    load_pivot = df.pivot_table(
        index=["run_id", "tick"],
        columns="node_id",
        values="load_ratio",
        aggfunc="first",
    )

    # For each node, compute the mean load of its neighbors
    neighbor_avg = {}
    for node_id, neighbors in neighbor_map.items():
        valid_neighbors = [n for n in neighbors if n in load_pivot.columns]
        if valid_neighbors:
            neighbor_avg[node_id] = load_pivot[valid_neighbors].mean(axis=1)
        else:
            neighbor_avg[node_id] = pd.Series(0.0, index=load_pivot.index)

    # Map back to original DataFrame rows
    neighbor_avg_values = []
    for _, row in df.iterrows():
        key = (row["run_id"], row["tick"])
        node_id = row["node_id"]
        if node_id in neighbor_avg and key in neighbor_avg[node_id].index:
            neighbor_avg_values.append(round(neighbor_avg[node_id][key], 4))
        else:
            neighbor_avg_values.append(0.0)

    df["neighbor_avg_load"] = neighbor_avg_values
    return df


def compute_neighbor_features_fast(
    df: pd.DataFrame,
    neighbor_map: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Compute neighbor_avg_load efficiently using vectorized merge operations.

    This is a faster alternative to the row-by-row approach, using
    pandas merge and groupby operations.

    Args:
        df: DataFrame with load_ratio column.
        neighbor_map: Adjacency dict {node_id: [neighbor_ids]}.

    Returns:
        DataFrame with 'neighbor_avg_load' column added.
    """
    print("  Computing neighbor features (vectorized)...")

    # Build edges DataFrame from neighbor map
    edges = []
    for node_id, neighbors in neighbor_map.items():
        for neighbor_id in neighbors:
            edges.append({"node_id": node_id, "neighbor_id": neighbor_id})
    edges_df = pd.DataFrame(edges)

    # Get load_ratio lookup: (run_id, tick, node_id) -> load_ratio
    load_lookup = df[["run_id", "tick", "node_id", "load_ratio"]].copy()
    load_lookup = load_lookup.rename(columns={
        "node_id": "neighbor_id",
        "load_ratio": "neighbor_load_ratio",
    })

    # Join: for each row, find neighbor's load ratios at the same (run_id, tick)
    # First, merge edges with original df to get (run_id, tick, node_id, neighbor_id)
    expanded = df[["run_id", "tick", "node_id"]].merge(edges_df, on="node_id")

    # Then merge with load lookup to get neighbor's load_ratio
    expanded = expanded.merge(
        load_lookup,
        on=["run_id", "tick", "neighbor_id"],
    )

    # Compute mean neighbor load per (run_id, tick, node_id)
    neighbor_means = (
        expanded
        .groupby(["run_id", "tick", "node_id"])["neighbor_load_ratio"]
        .mean()
        .round(4)
        .reset_index()
        .rename(columns={"neighbor_load_ratio": "neighbor_avg_load"})
    )

    # Merge back to original DataFrame
    df = df.merge(neighbor_means, on=["run_id", "tick", "node_id"], how="left")
    df["neighbor_avg_load"] = df["neighbor_avg_load"].fillna(0.0)

    return df


def compute_ticks_since_warning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute how many ticks since each node was last in WARNING state.

    For each (run_id, node_id) group, tracks when the node was last
    in WARNING status and computes the elapsed ticks. Nodes that
    have never been in WARNING get a value of 999.

    Args:
        df: DataFrame sorted by (run_id, node_id, tick).

    Returns:
        DataFrame with 'ticks_since_warning' column added.
    """
    print("  Computing ticks_since_warning...")

    df = df.sort_values(["run_id", "node_id", "tick"]).reset_index(drop=True)

    results = []
    for (run_id, node_id), group in df.groupby(["run_id", "node_id"]):
        last_warning_tick = -999  # Sentinel: never warned
        ticks_since = []

        for _, row in group.iterrows():
            tick = row["tick"]
            status = row["status"]

            if status == "WARNING":
                last_warning_tick = tick
                ticks_since.append(0)
            elif last_warning_tick >= 0:
                ticks_since.append(tick - last_warning_tick)
            else:
                ticks_since.append(999)  # Never warned

        results.extend(ticks_since)

    df["ticks_since_warning"] = results
    return df


def compute_failure_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the binary label: will_fail_within_10_ticks.

    For each row, looks ahead up to 10 ticks within the same
    (run_id, node_id) group. If any future tick has status == 'FAILED',
    the label is 1, otherwise 0.

    Rows in the last 10 ticks of each run are dropped since the label
    cannot be computed (insufficient lookahead).

    Args:
        df: DataFrame sorted by (run_id, node_id, tick).

    Returns:
        DataFrame with 'will_fail_within_10_ticks' column, with
        trailing rows removed.
    """
    print("  Computing failure labels (will_fail_within_10_ticks)...")

    df = df.sort_values(["run_id", "node_id", "tick"]).reset_index(drop=True)

    # For each (run_id, node_id), create a binary series: 1 if FAILED
    df["_is_failed"] = (df["status"] == "FAILED").astype(int)

    # For each group, compute a reverse rolling max over the next 10 ticks
    # This means: "is there a 1 (failure) in the next 10 ticks?"
    labels = []
    max_ticks_per_group = []

    for (run_id, node_id), group in df.groupby(["run_id", "node_id"]):
        failed_series = group["_is_failed"].values
        ticks = group["tick"].values
        max_tick = ticks.max()
        n = len(failed_series)

        group_labels = []
        for i in range(n):
            # Look ahead up to 10 ticks
            end_idx = min(i + LABEL_HORIZON + 1, n)
            future = failed_series[i + 1:end_idx]
            if len(future) < LABEL_HORIZON:
                # Not enough lookahead — will be dropped
                group_labels.append(-1)
            else:
                group_labels.append(int(future.max()))

        labels.extend(group_labels)

    df["will_fail_within_10_ticks"] = labels
    df = df.drop(columns=["_is_failed"])

    # Drop rows where label couldn't be computed
    before = len(df)
    df = df[df["will_fail_within_10_ticks"] >= 0].reset_index(drop=True)
    after = len(df)
    print(f"  Dropped {before - after:,} rows (insufficient lookahead)")

    return df


def main() -> None:
    """
    Run the full feature engineering pipeline.

    Loads raw training data, computes 5 new features and the binary
    failure prediction label, then saves the enriched dataset.
    """
    print("=" * 60)
    print("  ⚡ LiveGrid — Feature Engineering")
    print("=" * 60 + "\n")

    # Step 1: Load data
    df, neighbor_map = load_data()

    # Step 2: Rolling features
    df = compute_rolling_features(df)

    # Step 3: Neighbor features (use fast vectorized version)
    df = compute_neighbor_features_fast(df, neighbor_map)

    # Step 4: Ticks since warning
    df = compute_ticks_since_warning(df)

    # Step 5: Failure label
    df = compute_failure_label(df)

    # Step 6: Save
    print(f"\n  Saving to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)
    abs_path = os.path.abspath(OUTPUT_CSV)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  ✅ Feature engineering complete")
    print(f"{'=' * 60}")
    print(f"  Output: {abs_path}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\n  Feature stats:")
    feature_cols = [
        "load_ratio_rolling_mean_5", "load_ratio_rolling_std_5",
        "load_ratio_delta", "neighbor_avg_load", "ticks_since_warning",
    ]
    for col in feature_cols:
        print(f"    {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")

    # Label distribution
    pos = df["will_fail_within_10_ticks"].sum()
    neg = len(df) - pos
    print(f"\n  Label distribution:")
    print(f"    Positive (will fail): {pos:,} ({pos / len(df) * 100:.1f}%)")
    print(f"    Negative (won't fail): {neg:,} ({neg / len(df) * 100:.1f}%)")
    print(f"    Class ratio (neg/pos): {neg / max(pos, 1):.1f}:1")


if __name__ == "__main__":
    main()

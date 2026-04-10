#!/usr/bin/env python3
"""
LiveGrid Phase 2 — Dataset Generation

Runs the LiveGrid simulator 500 times with different random seeds and
randomized scenarios, then concatenates all results into a single
training dataset with run_id tracking.

Usage:
    python generate_dataset.py

Output:
    output/training_data.csv   — ~500K rows of node time-series data
    output/neighbor_map.json   — adjacency graph for feature engineering
"""

from __future__ import annotations

import json
import os
import random
import sys
import io
from typing import Any

from livegrid.config import SimulationConfig
from livegrid.engine.engine import SimulationEngine
from livegrid.grid.grid import Grid
from livegrid.logging.logger import DataLogger
from livegrid.scenarios.heatwave import HeatWaveScenario
from livegrid.scenarios.sudden_failure import SuddenFailureScenario
from livegrid.scenarios.base import BaseScenario


# ── Constants ────────────────────────────────────────────────────────

NUM_RUNS = 500
TICKS_PER_RUN = 100
OUTPUT_DIR = "output"
TRAINING_CSV = os.path.join(OUTPUT_DIR, "training_data.csv")
NEIGHBOR_MAP_JSON = os.path.join(OUTPUT_DIR, "neighbor_map.json")

# Node IDs in the sample grid (for random scenario targeting)
ALL_NODE_IDS = [
    "GEN-1", "GEN-2", "SUB-1", "SUB-2", "SUB-3",
    "DIST-1", "DIST-2", "DIST-3", "DIST-4", "DIST-5",
]


def build_random_scenarios(rng: random.Random) -> list[BaseScenario]:
    """
    Build a randomized set of scenarios for one simulation run.

    Randomly selects one of three scenario configurations:
    - Heatwave only (33%)
    - Sudden failure only (33%)
    - Both combined (34%)

    Scenario parameters (timing, intensity, target) are randomized
    to ensure diverse training data.

    Args:
        rng: Random number generator for reproducible randomization.

    Returns:
        List of scenario instances for this run.
    """
    choice = rng.randint(0, 2)  # 0=heatwave, 1=sudden_failure, 2=both
    scenarios: list[BaseScenario] = []

    if choice in (0, 2):
        # Heatwave with randomized parameters
        scenarios.append(HeatWaveScenario(
            start_tick=rng.randint(10, 50),
            duration=rng.randint(10, 30),
            load_increase_pct=rng.uniform(0.15, 0.50),
        ))

    if choice in (1, 2):
        # Sudden failure at a random node and tick
        scenarios.append(SuddenFailureScenario(
            target_node_id=rng.choice(ALL_NODE_IDS),
            trigger_tick=rng.randint(15, 75),
        ))

    return scenarios


def run_single_simulation(
    run_id: int,
    seed: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """
    Run a single simulation and return all node snapshot rows with run_id.

    Suppresses all stdout output from the engine for batch operation.

    Args:
        run_id: Unique identifier for this simulation run.
        seed: Random seed for the simulation engine.
        rng: RNG for scenario randomization (separate from engine RNG).

    Returns:
        List of dictionaries, each representing one node snapshot row
        with an added 'run_id' field.
    """
    # Build fresh grid
    grid = Grid.build_sample_grid()

    # Build random scenarios
    scenarios = build_random_scenarios(rng)

    # Configure engine
    config = SimulationConfig(
        total_ticks=TICKS_PER_RUN,
        random_seed=seed,
        load_fluctuation_pct=0.03,
    )

    logger = DataLogger()
    engine = SimulationEngine(
        grid=grid,
        logger=logger,
        scenarios=scenarios,
        config=config,
    )

    # Suppress stdout during simulation
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        engine.run()
    finally:
        sys.stdout = old_stdout

    # Get data and add run_id
    rows = logger.get_node_data()
    for row in rows:
        row["run_id"] = run_id

    return rows


def extract_neighbor_map(grid: Grid) -> dict[str, list[str]]:
    """
    Extract the adjacency map from a grid instance.

    Args:
        grid: A Grid instance with nodes and edges.

    Returns:
        Dictionary mapping each node_id to its list of neighbor node_ids.
    """
    neighbor_map: dict[str, list[str]] = {}
    for node in grid.all_nodes():
        neighbor_map[node.id] = node.neighbors.copy()
    return neighbor_map


def main() -> None:
    """
    Generate the full training dataset from 500 simulation runs.

    Each run uses a different random seed and randomized scenario
    configuration. Results are concatenated into a single CSV with
    a run_id column for tracking.
    """
    print("=" * 60)
    print("  ⚡ LiveGrid — Dataset Generation")
    print(f"  Runs: {NUM_RUNS} | Ticks/run: {TICKS_PER_RUN}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Master RNG for scenario randomization (separate from per-run seeds)
    master_rng = random.Random(12345)

    # Save neighbor map from a sample grid
    sample_grid = Grid.build_sample_grid()
    neighbor_map = extract_neighbor_map(sample_grid)
    with open(NEIGHBOR_MAP_JSON, "w") as f:
        json.dump(neighbor_map, f, indent=2)
    print(f"\n📊 Neighbor map saved to {NEIGHBOR_MAP_JSON}")

    # Collect all rows
    all_rows: list[dict[str, Any]] = []
    failures_seen = 0

    for run_id in range(NUM_RUNS):
        seed = run_id  # Each run gets a unique seed
        rows = run_single_simulation(run_id, seed, master_rng)
        all_rows.extend(rows)

        # Count failures in this run
        run_failures = sum(1 for r in rows if r["status"] == "FAILED")
        failures_seen += run_failures

        # Progress bar
        if (run_id + 1) % 25 == 0 or run_id == 0:
            pct = (run_id + 1) / NUM_RUNS * 100
            sys.stdout.write(
                f"\r  Progress: {run_id + 1}/{NUM_RUNS} runs "
                f"({pct:.0f}%) | "
                f"Rows: {len(all_rows):,} | "
                f"Failure rows: {failures_seen:,}"
            )
            sys.stdout.flush()

    print(f"\n\n  ✅ All {NUM_RUNS} runs complete.")
    print(f"  Total rows: {len(all_rows):,}")
    print(f"  Failure rows: {failures_seen:,} "
          f"({failures_seen / len(all_rows) * 100:.1f}%)")

    # Write CSV using the existing logger's column order + run_id
    import csv
    columns = [
        "run_id", "tick", "node_id", "node_type", "capacity",
        "current_load", "load_ratio", "voltage", "frequency", "status",
    ]

    with open(TRAINING_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    abs_path = os.path.abspath(TRAINING_CSV)
    print(f"\n💾 Training data saved to: {abs_path}")
    print(f"   Columns: {', '.join(columns)}")
    print(f"   Rows: {len(all_rows):,}")


if __name__ == "__main__":
    main()

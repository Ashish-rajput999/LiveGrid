#!/usr/bin/env python3
"""
LiveGrid — Real-Time Power Grid Failure Simulation

Entry point that wires together all components and runs a sample simulation
with heat wave and sudden failure scenarios.

Usage:
    python main.py
"""

import os
import sys

from livegrid.config import SimulationConfig
from livegrid.engine.engine import SimulationEngine
from livegrid.grid.grid import Grid
from livegrid.logging.logger import DataLogger
from livegrid.scenarios.heatwave import HeatWaveScenario
from livegrid.scenarios.sudden_failure import SuddenFailureScenario


def main() -> None:
    """Run the LiveGrid simulation."""

    # --- Configuration ---
    config = SimulationConfig(
        total_ticks=100,
        nominal_voltage=230.0,
        nominal_frequency=50.0,
        warning_threshold=0.9,
        failure_threshold=1.0,
        max_cascade_depth=10,
        load_fluctuation_pct=0.03,
        random_seed=42,
        output_dir="output",
    )

    # --- Build the grid ---
    print("\n📐 Building sample power grid...")
    grid = Grid.build_sample_grid()
    print(grid.summary())

    # --- Configure scenarios ---
    scenarios = [
        # Heat wave: starts at tick 30, lasts 20 ticks, +30% load
        HeatWaveScenario(
            start_tick=30,
            duration=20,
            load_increase_pct=0.30,
        ),
        # Sudden failure: SUB-1 fails at tick 50
        SuddenFailureScenario(
            target_node_id="SUB-1",
            trigger_tick=50,
        ),
    ]

    print(f"\n🎬 Scenarios loaded:")
    for s in scenarios:
        print(f"   • {s}")

    # --- Initialize logger and engine ---
    logger = DataLogger()
    engine = SimulationEngine(
        grid=grid,
        logger=logger,
        scenarios=scenarios,
        config=config,
    )

    # --- Run simulation ---
    engine.run()

    # --- Export results ---
    os.makedirs(config.output_dir, exist_ok=True)

    node_csv = os.path.join(config.output_dir, "node_timeseries.csv")
    events_csv = os.path.join(config.output_dir, "events.csv")

    node_path = logger.to_csv(node_csv)
    events_path = logger.events_to_csv(events_csv)

    print(f"\n💾 Results exported:")
    print(f"   Node data:  {node_path}")
    print(f"   Events:     {events_path}")
    print(f"\n{logger.summary()}")

    # --- Final statistics ---
    print("\n" + "=" * 70)
    print("  📈 Simulation Summary")
    print("=" * 70)

    all_nodes = grid.all_nodes()
    failed = [n for n in all_nodes if n.status.value == "FAILED"]
    warned = [n for n in all_nodes if n.status.value == "WARNING"]

    print(f"  Total nodes:      {len(all_nodes)}")
    print(f"  Final OK:         {len(all_nodes) - len(failed) - len(warned)}")
    print(f"  Final WARNING:    {len(warned)}")
    print(f"  Final FAILED:     {len(failed)}")

    if failed:
        print(f"\n  Failed nodes:")
        for n in failed:
            print(f"    🔴 {n.id} ({n.node_type})")

    if warned:
        print(f"\n  Warning nodes:")
        for n in warned:
            print(
                f"    🟡 {n.id} — load: {n.current_load:.1f}/{n.capacity:.0f}MW "
                f"({n.load_ratio:.0%})"
            )

    print("\n" + "=" * 70)
    print("  ✅ LiveGrid simulation complete.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

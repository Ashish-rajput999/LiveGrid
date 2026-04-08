"""
Simulation engine — orchestrates the tick-based simulation loop.

Manages the flow: scenarios → load fluctuation → electrical update →
failure detection → cascade propagation → logging. Each tick represents
one second of simulated grid operation.
"""

from __future__ import annotations

import random
import sys
from typing import Optional

from livegrid.config import SimulationConfig
from livegrid.grid.grid import Grid
from livegrid.logging.logger import DataLogger
from livegrid.models.node import Node, NodeStatus
from livegrid.scenarios.base import BaseScenario


class SimulationEngine:
    """
    Core simulation engine for the power grid.

    Runs a tick-based loop where each tick:
    1. Applies active scenarios (external events)
    2. Applies random load fluctuations (demand noise)
    3. Updates electrical state (voltage, frequency)
    4. Detects overloads and triggers failures
    5. Propagates cascading failures
    6. Logs all node states and events

    Usage:
        grid = Grid.build_sample_grid()
        logger = DataLogger()
        config = SimulationConfig(total_ticks=100)
        scenarios = [HeatWaveScenario(), SuddenFailureScenario()]

        engine = SimulationEngine(grid, logger, scenarios, config)
        engine.run()
    """

    def __init__(
        self,
        grid: Grid,
        logger: DataLogger,
        scenarios: Optional[list[BaseScenario]] = None,
        config: Optional[SimulationConfig] = None,
    ) -> None:
        self.grid = grid
        self.logger = logger
        self.scenarios = scenarios or []
        self.config = config or SimulationConfig()

        # Initialize RNG
        self._rng = random.Random(self.config.random_seed)

        # Cascade tracking for the current tick
        self._cascade_log: list[str] = []

    # --- Main loop ---

    def run(self, ticks: Optional[int] = None) -> None:
        """
        Run the simulation for the configured number of ticks.

        Args:
            ticks: Override the number of ticks (uses config.total_ticks if None).
        """
        total = ticks or self.config.total_ticks

        print("\n" + "=" * 70)
        print("  ⚡ LiveGrid Simulation Starting")
        print(f"  Ticks: {total} | Nodes: {self.grid.node_count} | "
              f"Scenarios: {len(self.scenarios)}")
        print("=" * 70)

        # Log initial state (tick -1 isn't logged, tick 0 is the first state)
        self.logger.log_tick(tick=0, nodes=self.grid.all_nodes())
        self.logger.log_event(
            tick=0, event_type="SYSTEM", description="Simulation started"
        )

        for tick in range(1, total + 1):
            self._run_tick(tick)

            # Print progress every 10 ticks
            if tick % 10 == 0 or tick == total:
                failed = len(self.grid.get_failed_nodes())
                operational = len(self.grid.get_operational_nodes())
                sys.stdout.write(
                    f"\r  Tick {tick:>4}/{total} | "
                    f"Operational: {operational} | Failed: {failed}"
                )
                sys.stdout.flush()

        print("\n")
        self.logger.log_event(
            tick=total, event_type="SYSTEM", description="Simulation completed"
        )

        print(self.grid.summary())

    def _run_tick(self, tick: int) -> None:
        """Execute a single simulation tick."""

        # Step 1: Apply scenarios
        self._apply_scenarios(tick)

        # Step 2: Apply random load fluctuations to operational nodes
        for node in self.grid.get_operational_nodes():
            self._apply_load_fluctuation(node)

        # Step 3: Update electrical state for all operational nodes
        for node in self.grid.get_operational_nodes():
            node.update_electrical_state(
                nominal_voltage=self.config.nominal_voltage,
                nominal_frequency=self.config.nominal_frequency,
                warning_threshold=self.config.warning_threshold,
                failure_threshold=self.config.failure_threshold,
            )

        # Step 4: Detect newly failed nodes (from overload in step 3)
        newly_failed = [
            n for n in self.grid.all_nodes()
            if n.status == NodeStatus.FAILED and n.current_load > 0
        ]

        # Step 5: Cascade propagation
        for node in newly_failed:
            self.logger.log_event(
                tick=tick,
                event_type="OVERLOAD",
                node_id=node.id,
                description=(
                    f"Node {node.id} FAILED from overload "
                    f"(load={node.current_load:.1f}MW, "
                    f"capacity={node.capacity:.0f}MW, "
                    f"ratio={node.load_ratio:.2%})"
                ),
            )
            self._cascade(node.id, tick, depth=0)

        # Step 6: Log all node states
        self.logger.log_tick(tick=tick, nodes=self.grid.all_nodes())

    # --- Scenarios ---

    def _apply_scenarios(self, tick: int) -> None:
        """Apply all active scenarios for this tick."""
        for scenario in self.scenarios:
            if scenario.is_active(tick):
                events = scenario.apply(self.grid, tick)
                for event_desc in events:
                    self.logger.log_event(
                        tick=tick,
                        event_type="SCENARIO",
                        description=event_desc,
                    )
                    # Print scenario events in real-time
                    print(f"\n  [Tick {tick}] {event_desc}")

    # --- Load dynamics ---

    def _apply_load_fluctuation(self, node: Node) -> None:
        """
        Apply random load fluctuation to a node.

        Simulates demand noise: small random walk bounded by ±fluctuation_pct.
        Load is clamped to [0, +inf) — it can exceed capacity (that's how
        overloads happen organically).
        """
        pct = self.config.load_fluctuation_pct
        delta = self._rng.uniform(-pct, pct) * node.current_load
        node.current_load = max(0.0, node.current_load + delta)

    # --- Cascade logic ---

    def _cascade(self, failed_node_id: str, tick: int, depth: int) -> None:
        """
        Propagate cascading failure from a failed node.

        Redistributes the failed node's load equally among its active
        neighbors. If any neighbor exceeds capacity as a result, it
        fails and the cascade recurses.

        Args:
            failed_node_id: ID of the node that just failed.
            tick: Current simulation tick.
            depth: Current cascade recursion depth.
        """
        if depth >= self.config.max_cascade_depth:
            self.logger.log_event(
                tick=tick,
                event_type="CASCADE_LIMIT",
                node_id=failed_node_id,
                description=(
                    f"Cascade depth limit ({self.config.max_cascade_depth}) "
                    f"reached at node {failed_node_id}"
                ),
            )
            return

        failed_node = self.grid.get_node(failed_node_id)
        if failed_node is None:
            return

        # Load to redistribute
        stranded_load = failed_node.current_load
        if stranded_load <= 0:
            return

        # Zero out the failed node's load (it's been shed)
        failed_node.current_load = 0.0

        # Find active neighbors to absorb the load
        active_neighbors = self.grid.get_active_neighbors(failed_node_id)

        if not active_neighbors:
            self.logger.log_event(
                tick=tick,
                event_type="CASCADE",
                node_id=failed_node_id,
                description=(
                    f"Node {failed_node_id}: {stranded_load:.1f}MW load LOST "
                    f"(no active neighbors to absorb)"
                ),
            )
            return

        # Distribute load equally
        load_per_neighbor = stranded_load / len(active_neighbors)

        self.logger.log_event(
            tick=tick,
            event_type="CASCADE",
            node_id=failed_node_id,
            description=(
                f"Node {failed_node_id}: redistributing {stranded_load:.1f}MW "
                f"to {len(active_neighbors)} neighbors "
                f"({load_per_neighbor:.1f}MW each)"
            ),
        )

        # Apply load and check for secondary failures
        secondary_failures: list[str] = []

        for neighbor in active_neighbors:
            neighbor.current_load += load_per_neighbor

            # Recalculate electrical state
            neighbor.update_electrical_state(
                nominal_voltage=self.config.nominal_voltage,
                nominal_frequency=self.config.nominal_frequency,
                warning_threshold=self.config.warning_threshold,
                failure_threshold=self.config.failure_threshold,
            )

            if neighbor.status == NodeStatus.FAILED:
                secondary_failures.append(neighbor.id)
                self.logger.log_event(
                    tick=tick,
                    event_type="CASCADE",
                    node_id=neighbor.id,
                    description=(
                        f"Node {neighbor.id} FAILED from cascade "
                        f"(received {load_per_neighbor:.1f}MW from {failed_node_id}, "
                        f"total load={neighbor.current_load:.1f}MW, "
                        f"capacity={neighbor.capacity:.0f}MW)"
                    ),
                )
                print(
                    f"\n  [Tick {tick}] 🔴 CASCADE: {neighbor.id} failed "
                    f"(overloaded by cascade from {failed_node_id})"
                )

        # Recurse for secondary failures
        for node_id in secondary_failures:
            self._cascade(node_id, tick, depth + 1)

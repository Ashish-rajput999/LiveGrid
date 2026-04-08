"""
Data logging layer for simulation output.

Collects structured time-series data from every simulation tick and
scenario events. Stores data in memory with optional CSV export.
Designed to produce ML-friendly tabular data.
"""

from __future__ import annotations

import csv
import os
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from livegrid.models.node import Node


class DataLogger:
    """
    Structured data logger for grid simulation.

    Maintains two collections:
    - Node snapshots: one row per node per tick (time-series data)
    - Events: discrete events like failures, cascades, scenario triggers

    Both can be exported to CSV for analysis or ML pipeline ingestion.

    Usage:
        logger = DataLogger()
        logger.log_tick(tick=0, nodes=grid.all_nodes())
        logger.log_event(tick=5, event_type="CASCADE", node_id="SUB-1",
                        description="Node failed due to overload")
        logger.to_csv("output/node_data.csv")
        logger.events_to_csv("output/events.csv")
    """

    # Column order for node snapshot CSV
    NODE_COLUMNS = [
        "tick", "node_id", "node_type", "capacity", "current_load",
        "load_ratio", "voltage", "frequency", "status",
    ]

    # Column order for events CSV
    EVENT_COLUMNS = [
        "tick", "event_type", "node_id", "description",
    ]

    def __init__(self) -> None:
        self._node_data: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []

    # --- Recording ---

    def log_tick(self, tick: int, nodes: list["Node"]) -> None:
        """
        Record a snapshot of all nodes at the given tick.

        Args:
            tick: Current simulation tick number.
            nodes: List of all nodes to snapshot.
        """
        for node in nodes:
            self._node_data.append(node.snapshot(tick))

    def log_event(
        self,
        tick: int,
        event_type: str,
        description: str,
        node_id: str = "",
    ) -> None:
        """
        Record a discrete event.

        Args:
            tick: Tick when the event occurred.
            event_type: Category (e.g. "SCENARIO", "CASCADE", "FAILURE", "OVERLOAD").
            description: Human-readable description.
            node_id: Related node ID, if applicable.
        """
        self._events.append({
            "tick": tick,
            "event_type": event_type,
            "node_id": node_id,
            "description": description,
        })

    # --- Queries ---

    @property
    def node_row_count(self) -> int:
        """Total number of node snapshot rows recorded."""
        return len(self._node_data)

    @property
    def event_count(self) -> int:
        """Total number of events recorded."""
        return len(self._events)

    def get_node_data(self) -> list[dict[str, Any]]:
        """Return all node snapshot rows (for programmatic access / future Pandas)."""
        return self._node_data.copy()

    def get_events(self) -> list[dict[str, Any]]:
        """Return all event rows."""
        return self._events.copy()

    # --- CSV export ---

    def to_csv(self, filepath: str) -> str:
        """
        Write node time-series data to a CSV file.

        Creates parent directories if they don't exist.

        Args:
            filepath: Output file path.

        Returns:
            Absolute path of the written file.
        """
        return self._write_csv(filepath, self.NODE_COLUMNS, self._node_data)

    def events_to_csv(self, filepath: str) -> str:
        """
        Write event data to a CSV file.

        Args:
            filepath: Output file path.

        Returns:
            Absolute path of the written file.
        """
        return self._write_csv(filepath, self.EVENT_COLUMNS, self._events)

    @staticmethod
    def _write_csv(
        filepath: str,
        columns: list[str],
        rows: list[dict[str, Any]],
    ) -> str:
        """Write rows to CSV with given columns."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        abs_path = os.path.abspath(filepath)
        return abs_path

    # --- Summary ---

    def summary(self) -> str:
        """Generate a summary of logged data."""
        lines = [
            f"📊 DataLogger Summary:",
            f"   Node data rows: {self.node_row_count:,}",
            f"   Event records:  {self.event_count:,}",
        ]

        if self._events:
            # Count events by type
            type_counts: dict[str, int] = {}
            for event in self._events:
                etype = event["event_type"]
                type_counts[etype] = type_counts.get(etype, 0) + 1
            lines.append("   Events by type:")
            for etype, count in sorted(type_counts.items()):
                lines.append(f"     {etype}: {count}")

        return "\n".join(lines)

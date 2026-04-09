"""
Sudden failure scenario — simulates unexpected equipment failure.

Immediately fails a specific node at a given tick, representing
equipment malfunction, cyber attack, or natural disaster.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from livegrid.scenarios.base import BaseScenario

if TYPE_CHECKING:
    from livegrid.grid.grid import Grid


class SuddenFailureScenario(BaseScenario):
    """
    Simulates a sudden, unexpected failure of a specific node.

    At the trigger tick, the target node is immediately set to FAILED status.
    Its load remains as-is for the cascade engine to redistribute.

    Args:
        target_node_id: ID of the node to fail.
        trigger_tick: Tick at which the failure occurs.
    """

    def __init__(
        self,
        target_node_id: str = "SUB-1",
        trigger_tick: int = 50,
    ) -> None:
        self._target_node_id = target_node_id
        self._trigger_tick = trigger_tick
        self._triggered = False

    @property
    def name(self) -> str:
        return "sudden_failure"

    def is_active(self, tick: int) -> bool:
        return tick == self._trigger_tick and not self._triggered

    def apply(self, grid: "Grid", tick: int) -> list[str]:
        """
        Immediately fail the target node.

        Returns event descriptions for logging.
        """
        events: list[str] = []
        node = grid.get_node(self._target_node_id)

        if node is None:
            events.append(
                f"⚠️  SUDDEN FAILURE: Target node '{self._target_node_id}' "
                f"not found in grid — skipping"
            )
            self._triggered = True
            return events

        if node.status.value == "FAILED":
            events.append(
                f"⚠️  SUDDEN FAILURE: Node '{self._target_node_id}' "
                f"is already FAILED — no effect"
            )
            self._triggered = True
            return events

        # Store load before failure for cascade
        load_before = node.current_load
        node.fail()
        self._triggered = True

        events.append(
            f"💥 SUDDEN FAILURE at tick {tick}: Node {self._target_node_id} "
            f"forced to FAILED state (was carrying {load_before:.1f}MW)"
        )

        return events

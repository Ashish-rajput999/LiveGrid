"""
Heat wave scenario — simulates peak summer demand.

Increases load across all operational nodes for a sustained period,
representing air conditioning surges, cooling demand, etc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from livegrid.models.node import NodeStatus
from livegrid.scenarios.base import BaseScenario

if TYPE_CHECKING:
    from livegrid.grid.grid import Grid


class HeatWaveScenario(BaseScenario):
    """
    Simulates a heat wave event that increases power demand grid-wide.

    During the active period (start_tick to start_tick + duration),
    all operational nodes see their load increase by `load_increase_pct`.
    The increase is applied additively each tick — it's the scenario's
    responsibility to model sustained high demand.

    Note: The increase is applied as a one-time bump on the first active
    tick, then load naturally fluctuates around the new higher baseline
    via the engine's random walk.

    Args:
        start_tick: Tick when the heat wave begins.
        duration: Number of ticks the heat wave lasts.
        load_increase_pct: Fractional load increase (e.g. 0.30 = 30%).
    """

    def __init__(
        self,
        start_tick: int = 30,
        duration: int = 20,
        load_increase_pct: float = 0.30,
    ) -> None:
        self._start_tick = start_tick
        self._duration = duration
        self._load_increase_pct = load_increase_pct
        self._applied = False  # Track whether the initial surge was applied

    @property
    def name(self) -> str:
        return "heat_wave"

    def is_active(self, tick: int) -> bool:
        return self._start_tick <= tick < self._start_tick + self._duration

    def apply(self, grid: "Grid", tick: int) -> list[str]:
        """
        Apply heat wave load increase.

        On the first active tick, bumps all operational node loads.
        Returns event descriptions for logging.
        """
        events: list[str] = []

        if not self._applied:
            # First active tick: apply the surge
            for node in grid.get_operational_nodes():
                increase = node.current_load * self._load_increase_pct
                node.current_load += increase
                events.append(
                    f"HEAT_WAVE: {node.id} load increased by {increase:.1f}MW "
                    f"({self._load_increase_pct:.0%}) → {node.current_load:.1f}MW"
                )
            self._applied = True

            events.insert(0, (
                f"🌡️  HEAT WAVE STARTED at tick {tick} "
                f"(duration={self._duration} ticks, "
                f"+{self._load_increase_pct:.0%} load)"
            ))

        # Log ongoing status
        if tick == self._start_tick + self._duration - 1:
            events.append(f"🌡️  HEAT WAVE ENDING at tick {tick}")
            # Note: we don't reduce load back — it stays elevated,
            # representing lasting demand. The engine's fluctuation
            # will naturally adjust.

        return events

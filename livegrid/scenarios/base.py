"""
Base class for simulation scenarios.

Scenarios are injectable events that modify grid state during simulation.
They follow the Strategy pattern — each scenario implements `apply()` and
the engine calls all active scenarios on every tick.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from livegrid.grid.grid import Grid


class BaseScenario(ABC):
    """
    Abstract base for all simulation scenarios.

    Subclasses must implement:
        - apply(grid, tick): modify grid state and return event descriptions
        - is_active(tick): whether this scenario fires at the given tick

    Example:
        class MyScenario(BaseScenario):
            name = "my_scenario"

            def is_active(self, tick: int) -> bool:
                return tick == 50

            def apply(self, grid: Grid, tick: int) -> list[str]:
                node = grid.get_node("GEN-1")
                node.fail()
                return ["GEN-1 manually failed by MyScenario"]
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this scenario."""
        ...

    @abstractmethod
    def is_active(self, tick: int) -> bool:
        """
        Whether this scenario should fire at the given tick.

        Args:
            tick: Current simulation tick.

        Returns:
            True if the scenario should apply its effects at this tick.
        """
        ...

    @abstractmethod
    def apply(self, grid: "Grid", tick: int) -> list[str]:
        """
        Apply scenario effects to the grid.

        This method is only called when `is_active(tick)` returns True.
        Implementations should modify the grid state directly and return
        a list of human-readable event descriptions for logging.

        Args:
            grid: The grid to modify.
            tick: Current simulation tick.

        Returns:
            List of event description strings.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

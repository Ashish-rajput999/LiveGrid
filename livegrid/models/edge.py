"""
Edge model for power grid transmission lines.

Each edge represents a transmission line connecting two nodes (substations)
in the power grid graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Edge:
    """
    A transmission line connecting two nodes in the power grid.

    Attributes:
        source: ID of the source node.
        target: ID of the target node.
        capacity: Maximum power transfer capacity in MW.
        impedance: Line impedance in ohms (for future power flow calculations).
        active: Whether this transmission line is currently operational.
    """

    source: str
    target: str
    capacity: float = 500.0  # MW transfer limit
    impedance: float = 0.01  # ohms (placeholder for future use)
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert edge to a serializable dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "capacity": self.capacity,
            "impedance": self.impedance,
            "active": self.active,
        }

    def __repr__(self) -> str:
        status = "ACTIVE" if self.active else "INACTIVE"
        return (
            f"Edge({self.source} → {self.target}, "
            f"cap={self.capacity:.0f}MW, {status})"
        )

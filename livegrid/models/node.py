"""
Node model for power grid substations.

Each node represents a substation or generator in the power grid.
Tracks electrical state (voltage, frequency, load) and computes
derived status based on configurable thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeStatus(Enum):
    """Operational status of a grid node."""

    OK = "OK"
    WARNING = "WARNING"
    FAILED = "FAILED"

    def __str__(self) -> str:
        return self.value


class NodeType(Enum):
    """Classification of node role in the grid."""

    GENERATOR = "GENERATOR"
    SUBSTATION = "SUBSTATION"
    DISTRIBUTION = "DISTRIBUTION"

    def __str__(self) -> str:
        return self.value


@dataclass
class Node:
    """
    A single node in the power grid graph.

    Represents a generator, substation, or distribution point with
    electrical properties and operational state. Voltage and frequency
    are derived from the load-to-capacity ratio using simplified but
    physically meaningful models.

    Attributes:
        id: Unique identifier for this node.
        node_type: Role in the grid (GENERATOR, SUBSTATION, DISTRIBUTION).
        capacity: Maximum rated power capacity in MW.
        current_load: Current power demand/throughput in MW.
        voltage: Current voltage in kV (derived from load ratio).
        frequency: Current frequency in Hz (derived from load ratio).
        status: Operational status (OK, WARNING, FAILED).
        neighbors: List of adjacent node IDs in the grid graph.
    """

    id: str
    node_type: NodeType
    capacity: float  # MW
    current_load: float  # MW
    voltage: float = 230.0  # kV (nominal)
    frequency: float = 50.0  # Hz (nominal)
    status: NodeStatus = NodeStatus.OK
    neighbors: list[str] = field(default_factory=list)

    # --- Derived properties ---

    @property
    def load_ratio(self) -> float:
        """Ratio of current load to capacity (0.0 to unbounded)."""
        if self.capacity <= 0:
            return float("inf") if self.current_load > 0 else 0.0
        return self.current_load / self.capacity

    def is_overloaded(self) -> bool:
        """Check if current load exceeds capacity."""
        return self.current_load > self.capacity

    # --- State updates ---

    def update_electrical_state(
        self,
        nominal_voltage: float = 230.0,
        nominal_frequency: float = 50.0,
        warning_threshold: float = 0.9,
        failure_threshold: float = 1.0,
    ) -> None:
        """
        Recalculate voltage, frequency, and status from current load.

        Uses simplified droop models:
        - Voltage drops linearly with load ratio (5% droop at full load).
        - Frequency drops based on load-generation imbalance (governor response).
        - Status transitions: OK → WARNING at warning_threshold, FAILED at failure_threshold.

        Args:
            nominal_voltage: Base voltage in kV (default 230).
            nominal_frequency: Base frequency in Hz (default 50).
            warning_threshold: Load ratio triggering WARNING status.
            failure_threshold: Load ratio triggering FAILED status.
        """
        if self.status == NodeStatus.FAILED:
            # Failed nodes stay dead until explicitly restored
            return

        ratio = self.load_ratio

        # Voltage droop: 5% drop at full load, linear
        # V = V_nominal * (1 - 0.05 * ratio)
        # Clamp to prevent negative voltage at extreme overloads
        voltage_factor = max(0.0, 1.0 - 0.05 * ratio)
        self.voltage = round(nominal_voltage * voltage_factor, 2)

        # Frequency response: governor droop characteristic
        # f = f_nominal * (1 - 0.02 * (ratio - 0.5)) for ratio > 0.5
        # At 50% load, frequency is nominal. Drops ~1Hz per 100% overload above 50%.
        if ratio <= 0.5:
            self.frequency = nominal_frequency
        else:
            freq_deviation = 0.02 * (ratio - 0.5)
            self.frequency = round(nominal_frequency * max(0.0, 1.0 - freq_deviation), 2)

        # Status transitions
        if ratio >= failure_threshold:
            self.fail()
        elif ratio >= warning_threshold:
            self.status = NodeStatus.WARNING
        else:
            self.status = NodeStatus.OK

    def fail(self) -> None:
        """
        Mark this node as failed.

        Sets status to FAILED and zeros out voltage/frequency to represent
        a complete loss of power at this node.
        """
        self.status = NodeStatus.FAILED
        self.voltage = 0.0
        self.frequency = 0.0

    def restore(self) -> None:
        """
        Restore a failed node to operational state.

        Resets load to zero and status to OK. Voltage and frequency
        will be recalculated on the next update cycle.
        """
        self.status = NodeStatus.OK
        self.current_load = 0.0
        self.voltage = 230.0
        self.frequency = 50.0

    # --- Serialization ---

    def snapshot(self, tick: int) -> dict[str, Any]:
        """
        Create a complete snapshot of this node's state at a given tick.

        Returns a flat dictionary suitable for CSV/DataFrame rows.

        Args:
            tick: The simulation tick number.

        Returns:
            Dictionary with all node fields plus tick and derived load_ratio.
        """
        return {
            "tick": tick,
            "node_id": self.id,
            "node_type": str(self.node_type),
            "capacity": self.capacity,
            "current_load": round(self.current_load, 2),
            "load_ratio": round(self.load_ratio, 4),
            "voltage": self.voltage,
            "frequency": self.frequency,
            "status": str(self.status),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert node to a serializable dictionary."""
        return {
            "id": self.id,
            "node_type": str(self.node_type),
            "capacity": self.capacity,
            "current_load": round(self.current_load, 2),
            "load_ratio": round(self.load_ratio, 4),
            "voltage": self.voltage,
            "frequency": self.frequency,
            "status": str(self.status),
            "neighbors": self.neighbors.copy(),
        }

    def __repr__(self) -> str:
        return (
            f"Node(id={self.id!r}, type={self.node_type}, "
            f"load={self.current_load:.1f}/{self.capacity:.1f}MW, "
            f"V={self.voltage:.1f}kV, f={self.frequency:.2f}Hz, "
            f"status={self.status})"
        )

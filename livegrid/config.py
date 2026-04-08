"""
Simulation configuration constants and defaults.

Centralizes all tunable parameters so they can be adjusted without
modifying any simulation logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SimulationConfig:
    """
    Configuration for the simulation engine.

    All thresholds, physical constants, and simulation parameters
    are defined here for easy tuning and reproducibility.

    Attributes:
        total_ticks: Number of simulation ticks to run.
        nominal_voltage: Base voltage in kV.
        nominal_frequency: Base frequency in Hz.
        warning_threshold: Load ratio (0-1) that triggers WARNING status.
        failure_threshold: Load ratio (0-1) that triggers FAILED status.
        max_cascade_depth: Maximum recursion depth for cascade propagation.
        load_fluctuation_pct: Max random load change per tick (as fraction, e.g. 0.03 = 3%).
        random_seed: Seed for random number generator. None for non-deterministic.
        output_dir: Directory for CSV output files.
    """

    total_ticks: int = 100
    nominal_voltage: float = 230.0  # kV
    nominal_frequency: float = 50.0  # Hz
    warning_threshold: float = 0.9  # 90% load ratio → WARNING
    failure_threshold: float = 1.0  # 100% load ratio → FAILED
    max_cascade_depth: int = 10
    load_fluctuation_pct: float = 0.03  # ±3% per tick
    random_seed: Optional[int] = 42
    output_dir: str = "output"

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.total_ticks <= 0:
            raise ValueError(f"total_ticks must be positive, got {self.total_ticks}")
        if self.warning_threshold <= 0 or self.warning_threshold > 1:
            raise ValueError(
                f"warning_threshold must be in (0, 1], got {self.warning_threshold}"
            )
        if self.failure_threshold <= 0:
            raise ValueError(
                f"failure_threshold must be positive, got {self.failure_threshold}"
            )
        if self.warning_threshold >= self.failure_threshold:
            raise ValueError(
                f"warning_threshold ({self.warning_threshold}) must be less than "
                f"failure_threshold ({self.failure_threshold})"
            )
        if self.max_cascade_depth <= 0:
            raise ValueError(
                f"max_cascade_depth must be positive, got {self.max_cascade_depth}"
            )
        if self.load_fluctuation_pct < 0 or self.load_fluctuation_pct > 1:
            raise ValueError(
                f"load_fluctuation_pct must be in [0, 1], got {self.load_fluctuation_pct}"
            )

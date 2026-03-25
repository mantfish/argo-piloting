"""Shared type definitions for the profiling float simulator.

This module is the single source of truth for all dataclasses and type aliases
used across the simulator. All other modules should import from here — no type
definitions should live elsewhere.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Phase = Literal["ascending", "descending", "drift_on_surface", "on_seabed", "parking","communicating"]
"""Current vertical phase of the float.

"parking" means the float is drifting horizontally at its target depth,
neither ascending nor descending.
"""

ParkMode = Literal["parking_depth", "park_on_bottom", "drift_on_surface"]
"""How the float behaves between profiles.

- "parking_depth": drift at a fixed target depth.
- "park_on_bottom": rest on the seabed until the next ascent.
"""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ProfilerState:
    """Full state of the profiler at a single instant in time."""

    time: datetime
    lat: float
    lon: float
    depth: float        # Positive metres below the surface.
    phase: Phase
    x: float = 0.0     # Eastward displacement from simulation start (metres).
    y: float = 0.0     # Northward displacement from simulation start (metres).


@dataclass
class ControlAction:
    """Instruction issued to the profiler at each surfacing."""

    park_mode: ParkMode
    cycle_hours: float
    transmission_duration_minutes: float
    target_depth: float | None = None # Metres; only meaningful when park_mode is "parking_depth".
    ascent_speed_ms: float | None = None      # Metres per second, positive.
    descent_speed_ms: float | None = None   # Metres per second, positive.
        # Time in hours from reaching parking depth or seabed until the profiler begins ascending.


@dataclass
class TrajectoryRecord:
    """One row in the recorded trajectory — maps directly to a DataFrame column."""

    time: datetime
    lat: float
    lon: float
    x: float                    # Eastward displacement from simulation start (metres).
    y: float                    # Northward displacement from simulation start (metres).
    depth: float                # Positive metres below the surface.
    phase: Phase
    u: float                    # Eastward current at this point (m/s); float("nan") if unavailable.
    v: float                    # Northward current at this point (m/s); float("nan") if unavailable.
    bathymetry_depth: float     # Local seabed depth in metres, positive down; float("nan") if unavailable.
    on_seabed: bool


class ControlStrategy(ABC):
    """Abstract base class for all control strategies."""

    @abstractmethod
    def get_action(self, **kwargs) -> "ControlAction":
        """Return the ControlAction for the next cycle."""

    @abstractmethod
    def get_log(self) -> dict:
        """Return a JSON-serialisable dict of strategy parameters for logging."""


@dataclass
class SimConfig:
    """Top-level configuration for a single simulation run."""

    start_state: ProfilerState
    end_time: datetime
    control_strategy: ControlStrategy       # E.g. "no_control"; used in plot titles and output filenames.
    forecast_noise_std: float   # Std dev of Gaussian noise added to forecast velocity fields; 0.0 = perfect forecast.
    forecast_noise_seed: int    # Random seed for reproducibility of noise.
    forecast_horizon_hours: float # Forecast horizon in hours.
    data_dir: Path              # Directory containing the NetCDF tiles from data_getter.
    output_dir: Path            # Destination for trajectory parquet files and plots.

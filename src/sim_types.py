"""Shared type definitions for the profiling float simulator."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal

import numpy as np

Phase = Literal["ascending", "descending", "parking", "on_seabed", "communicating"]

@dataclass
class GeoLocation:
    lat: float
    lon: float


@dataclass
class ProfilerState:
    time: datetime
    location: GeoLocation
    depth: float
    phase: Phase
    x: float = 0.0   # metres east of simulation start
    y: float = 0.0   # metres north of simulation start


@dataclass
class EstimatedState:
    time: datetime
    location: GeoLocation
    depth: float
    phase: Phase
    x: float = 0.0
    y: float = 0.0
    bx: float = 0.0  # estimated current bias east (m/s)
    by: float = 0.0  # estimated current bias north (m/s)
    P: np.ndarray = field(default_factory=lambda: np.eye(4) * 1e6)  # 4x4 for [x, y, bx, by]


@dataclass
class ControlAction:
    parking_depth: float        # metres
    duration_hours: float       # hours
    science_cost: float         # 0 (no science) to 1 (full science)
    ascent_speed_ms: float = 0.01    # m/s, positive upward
    descent_speed_ms: float = 0.01  # m/s, positive downward


@dataclass
class SimConfig:
    start_time: datetime
    start_location: GeoLocation
    noise_seed: int
    bias_function: Callable          # f(t: datetime) -> [bx, by] in m/s
    process_noise: np.ndarray        # 4x4 Q matrix
    debug: bool
    data_dir: Path
    time_end: datetime
    control: object                  # KFMPC instance
    dt: float = 3600.0               # simulation timestep in seconds

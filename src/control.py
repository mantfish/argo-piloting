"""Control strategies for profiling float mission planning.

Each strategy encapsulates the decision logic that runs at the surface between
dive cycles. A strategy receives the current profiler state and a forecast
window, then returns a :class:`~sim_types.ControlAction` describing the next
cycle (park mode, depths, speeds, duration).

To add a new strategy, subclass :class:`ControlStrategy` and implement
:meth:`~ControlStrategy.get_action` and :meth:`~ControlStrategy.get_log`.
"""
from __future__ import annotations

import logging

import numpy as np
import xarray as xr

from sim_types import ControlAction, ControlStrategy, ProfilerState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------


class NoControlParkOnBottom(ControlStrategy):
    """Passive strategy: always park on the seabed, no active steering.

    The float descends until it touches the bottom, waits for the configured
    cycle duration, then ascends. No forecast information is used.
    """

    def __init__(self) -> None:
        self.cycle_hours = 120
        self.name = "no_control"
        self.default_control_action = ControlAction(
            park_mode="park_on_bottom",
            transmission_duration_minutes=30,
            ascent_speed_ms=0.01,
            descent_speed_ms=0.01,
            cycle_hours=self.cycle_hours,
        )

    def get_action(self, **kwargs) -> ControlAction:
        return self.default_control_action

    def get_log(self) -> dict:
        return {
            "control_strategy": self.name,
            "cycle_hours": self.default_control_action.cycle_hours,
            "transmission_duration_minutes": self.default_control_action.transmission_duration_minutes,
            "ascent_speed": self.default_control_action.ascent_speed_ms,
            "descent_speed": self.default_control_action.descent_speed_ms,
            "park_mode": self.default_control_action.park_mode,
        }


class DriftTowardsPoint(ControlStrategy):
    """Opportunistic steering: surface-drift when current points toward target.

    At each cycle the strategy samples the surface velocity from the forecast
    and compares its bearing to the bearing toward ``target_location``. If the
    angle between them is within ``theta_tolerance_deg``, the float drifts on
    the surface for the cycle; otherwise it parks at depth to avoid drifting
    the wrong way.

    Parameters
    ----------
    target_location:
        ``[lat, lon]`` of the destination point in decimal degrees.
    """

    def __init__(self, target_location: list[float]) -> None:
        self.target_location = target_location
        self.cycle_hours = 120
        self.transmission_duration_minutes = 30
        self.theta_tolerance_deg = 60.0
        self.ascent_speed_ms = 0.01
        self.descent_speed_ms = 0.01
        self.name = "drift_towards_point"

        self.parking_action = ControlAction(
            park_mode="parking_depth",
            transmission_duration_minutes=self.transmission_duration_minutes,
            ascent_speed_ms=self.ascent_speed_ms,
            descent_speed_ms=self.descent_speed_ms,
            cycle_hours=self.cycle_hours,
        )
        self.drifting_action = ControlAction(
            park_mode="drift_on_surface",
            transmission_duration_minutes=self.transmission_duration_minutes,
            cycle_hours=self.cycle_hours,
        )

    def get_action(self, profiler_state: ProfilerState, forecast: xr.Dataset, **kwargs) -> ControlAction:
        """Choose park or surface-drift based on current forecast bearing.

        Parameters
        ----------
        profiler_state:
            Current float state (position and time used for velocity lookup).
        forecast:
            Forecast dataset covering at least the current position and time.

        Returns
        -------
        ControlAction
            ``drifting_action`` if the surface current points within
            ``theta_tolerance_deg`` of the target; ``parking_action`` otherwise.
        """
        surface = forecast.isel(depth=0)
        t = np.clip(
            np.datetime64(profiler_state.time, "ns"),
            surface.time.values[0],
            surface.time.values[-1],
        )
        east_uo = float(surface["uo"].interp(time=t, latitude=profiler_state.lat, longitude=profiler_state.lon))
        north_vo = float(surface["vo"].interp(time=t, latitude=profiler_state.lat, longitude=profiler_state.lon))

        theta_current = np.arctan2(east_uo, north_vo)

        theta_positions = np.arctan2(
            (self.target_location[1] - profiler_state.lon) * np.cos(np.radians(profiler_state.lat)),
            self.target_location[0] - profiler_state.lat,
        )

        theta_diff = theta_positions - theta_current
        theta_diff = (theta_diff + np.pi) % (2 * np.pi) - np.pi  # wrap to [−π, π]

        if np.abs(theta_diff) <= self.theta_tolerance_deg * np.pi / 180:
            logger.info("Angle is %s, therefore drifting towards location for %d hours", theta_diff, self.cycle_hours)
            return self.drifting_action
        else:
            logger.info("Angle is %s, therefore parking on bottom for %d hours", theta_diff, self.cycle_hours)
            return self.parking_action

    def get_log(self) -> dict:
        return {
            "control_strategy": self.name,
            "cycle_hours": self.cycle_hours,
            "target_location": self.target_location,
            "theta_tolerance_deg": self.theta_tolerance_deg,
            "transmission_duration_minutes": self.transmission_duration_minutes,
            "ascent_speed": self.ascent_speed_ms,
            "descent_speed": self.descent_speed_ms,
        }

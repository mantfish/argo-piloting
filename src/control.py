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
import matplotlib.pyplot as plt

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

    def __init__(self, debug = False) -> None:
        self.cycle_hours = 120
        self.name = "no_control"
        self.default_control_action = ControlAction(
            park_mode="park_on_bottom",
            transmission_duration_minutes=30,
            ascent_speed_ms=0.01,
            descent_speed_ms=0.01,
            cycle_hours=self.cycle_hours,
        )
        self.debug = debug
        if debug:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_xlabel("Longitude")
            self.ax.set_ylabel("Latitude")
            self.ax.set_title("No Control Park on Bottom")
            self.ax.legend(loc="upper left")

    def get_action(self, **kwargs) -> ControlAction:
        if self.debug:
            profiler_state = kwargs["profiler_state"]
            self.update_debug_plot([profiler_state.lat, profiler_state.lon])
        return self.default_control_action

    def update_debug_plot(self, current_location: list[float]) -> None:
        """Redraw the debug plot for the current cycle decision.

        Parameters
        ----------
        current_location:
            ``[lat, lon]`` of the float's current position.
        """
        lat, lon = current_location
        self.ax.scatter(lon, lat, color="blue", s=80, zorder=5)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        plt.pause(0.05)

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

    def __init__(self, target_location: list[float], debug: bool = False) -> None:
        self.target_location = target_location
        self.cycle_hours = 120
        self.transmission_duration_minutes = 30
        self.theta_tolerance_deg = 60.0
        self.ascent_speed_ms = 0.01
        self.descent_speed_ms = 0.01
        self.name = "drift_towards_point"
        self.debug = debug

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

        if debug:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_xlabel("Longitude")
            self.ax.set_ylabel("Latitude")
            self.ax.set_title("DriftTowardsPoint — cycle decisions")
            self.ax.scatter(
                self.target_location[1], self.target_location[0],
                color="red", s=120, zorder=5, label="Target",
            )
            self.ax.legend(loc="upper left")

    def get_action(self, profiler_state: ProfilerState, forecast: xr.Dataset, **kwargs) -> ControlAction:
        """Choose park or surface-drift based on current forecast bearing.

        Parameters
        ----------
        profiler_state:
            Current float state (position and time used for velocity lookup).
        forecast:
            Forecast dataset covering at least the current position and time.
        debug:
            Plots map at each action to show why it's doing something

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

        if east_uo is np.nan or north_vo is np.nan:
            raise ValueError(f"NaN velocity at lat={profiler_state.lat} lon={profiler_state.lon} time={profiler_state.time}")

        theta_current = np.arctan2(east_uo, north_vo)

        theta_positions = np.arctan2(
            (self.target_location[1] - profiler_state.lon) * np.cos(np.radians(profiler_state.lat)),
            self.target_location[0] - profiler_state.lat,
        )

        theta_diff = theta_positions - theta_current
        theta_diff = (theta_diff + np.pi) % (2 * np.pi) - np.pi  # wrap to [−π, π]

        if self.debug:
            self.update_debug_plot(theta_current, theta_positions, [profiler_state.lat, profiler_state.lon])

        if np.abs(theta_diff) <= self.theta_tolerance_deg * np.pi / 180:
            logger.info("Angle is %s, therefore drifting towards location for %d hours", theta_diff, self.cycle_hours)
            return self.drifting_action
        else:
            logger.info("Angle is %s, therefore parking on bottom for %d hours", theta_diff, self.cycle_hours)
            return self.parking_action

    def update_debug_plot(self, theta_current: float, theta_target: float, current_location: list[float]) -> None:
        """Redraw the debug plot for the current cycle decision.

        Parameters
        ----------
        theta_current:
            Bearing of the surface current (radians, north=0, east=π/2).
        theta_target:
            Bearing from float to target (radians, same convention).
        current_location:
            ``[lat, lon]`` of the float's current position.
        """
        lat, lon = current_location
        arrow_len = 0.3  # degrees — fixed visual scale regardless of current speed

        self.ax.scatter(lon, lat, color="blue", s=80, zorder=5)

        # Current velocity direction (blue arrow)
        self.ax.quiver(
            lon, lat,
            np.sin(theta_current) * arrow_len,
            np.cos(theta_current) * arrow_len,
            color="blue", scale=1, scale_units="xy", angles="xy",
            zorder=4,
        )

        # Bearing to target (green arrow)
        self.ax.quiver(
            lon, lat,
            np.sin(theta_target) * arrow_len,
            np.cos(theta_target) * arrow_len,
            color="green", scale=1, scale_units="xy", angles="xy",
            zorder=4,
        )

        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.set_title(
            f"Cycle @ ({lat:.3f}°N, {lon:.3f}°E)  "
            f"Δθ={np.degrees(theta_current - theta_target):.1f}°"
        )
        self.fig.canvas.draw()
        plt.pause(0.05)


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

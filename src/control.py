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
import math
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from itertools import product

from sim_types import ControlAction, ControlStrategy, ProfilerState, GeoLocation
from particle_mover import quickly_estimate_next_surface_position
from data_loader import build_velocity_interpolator

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------


class NoControlParkOnBottom(ControlStrategy):
    """Passive strategy: always park on the seabed, no active steering.

    The float descends until it touches the bottom, waits for the configured
    cycle duration, then ascends. No forecast information is used.
    """

    def __init__(self, default_action: ControlAction, debug = False) -> None:
        self.cycle_hours = default_action.cycle_hours
        self.name = "no_control"
        self.default_control_action = default_action
        self.debug = debug
        self._cycle = 0
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
            self.update_debug_plot([profiler_state.location.lat, profiler_state.location.lon], profiler_state.time)
        self._cycle += 1
        return self.default_control_action

    def update_debug_plot(self, current_location: list[float], time: datetime) -> None:
        """Redraw the debug plot for the current cycle decision.

        Parameters
        ----------
        current_location:
            ``[lat, lon]`` of the float's current position.
        time:
            Current simulation time.
        """
        lat, lon = current_location
        self.ax.plot(lon, lat, "o", color="blue", markersize=8, zorder=5)
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.set_title(
            f"No Control  |  Cycle {self._cycle}  |  {time.strftime('%Y-%m-%d %H:%M')}"
        )
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

    def __init__(self, default_action: ControlAction, target_location: list[float], debug: bool = False) -> None:
        self.target_location = target_location
        self.cycle_hours = default_action.cycle_hours
        self.transmission_duration_minutes = default_action.transmission_duration_minutes
        self.theta_tolerance_deg = 60.0
        self.ascent_speed_ms = default_action.ascent_speed_ms
        self.descent_speed_ms = default_action.descent_speed_ms
        self.name = "drift_towards_point"
        self.debug = debug

        self.parking_action = ControlAction(
            park_mode="parking_on_bottom",
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

        self._cycle = 0

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
        east_uo = float(surface["uo"].interp(time=t, latitude=profiler_state.location.lat, longitude=profiler_state.location.lon))
        north_vo = float(surface["vo"].interp(time=t, latitude=profiler_state.location.lat, longitude=profiler_state.location.lon))

        if east_uo is np.nan or north_vo is np.nan:
            raise ValueError(f"NaN velocity at lat={profiler_state.location.lat} lon={profiler_state.location.lon} time={profiler_state.time}")

        theta_current = np.arctan2(east_uo, north_vo)

        theta_positions = np.arctan2(
            (self.target_location[1] - profiler_state.location.lon) * np.cos(np.radians(profiler_state.location.lat)),
            self.target_location[0] - profiler_state.location.lat,
        )

        theta_diff = theta_positions - theta_current
        theta_diff = (theta_diff + np.pi) % (2 * np.pi) - np.pi  # wrap to [−π, π]

        if self.debug:
            self.update_debug_plot(theta_current, theta_positions, [profiler_state.location.lat, profiler_state.location.lon], profiler_state.time)

        if np.abs(theta_diff) <= self.theta_tolerance_deg * np.pi / 180:
            logger.info("Angle is %s, therefore drifting towards location for %d hours", theta_diff, self.cycle_hours)
            self._cycle += 1
            return self.drifting_action
        else:
            logger.info("Angle is %s, therefore parking on bottom for %d hours", theta_diff, self.cycle_hours)
            self._cycle += 1
            return self.parking_action

    def update_debug_plot(self, theta_current: float, theta_target: float, current_location: list[float], time: datetime) -> None:
        """Redraw the debug plot for the current cycle decision.

        Parameters
        ----------
        theta_current:
            Bearing of the surface current (radians, north=0, east=π/2).
        theta_target:
            Bearing from float to target (radians, same convention).
        current_location:
            ``[lat, lon]`` of the float's current position.
        time:
            Current simulation time.
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
            f"DriftTowardsPoint  |  Cycle {self._cycle}  |  {time.strftime('%Y-%m-%d %H:%M')}\n"
            f"({lat:.3f}°N, {lon:.3f}°E)  Δθ={np.degrees(theta_current - theta_target):.1f}°"
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

class CircleDrift(ControlStrategy):
    """Geofencing strategy: keep the float within a target circle.

    At each surfacing the strategy simulates where the float would end up
    under the last-used action, then chooses between two actions:

    - ``measurement_action``: the caller-supplied default (typically a full
      deep dive for scientific data collection).
    - ``quick_action``: a short, shallow parking cycle (1 h at 10 m) used to
      keep the float from drifting too far from the target.

    Decision logic:
    - **Inside circle, next position also inside**: take the full
      ``measurement_action`` — conditions are favourable, collect data.
    - **Inside circle, next position outside**: take ``quick_action`` to
      avoid drifting out.
    - **Outside circle, next position closer**: take ``quick_action`` to
      check again sooner as the float approaches.
    - **Outside circle, next position not closer**: take ``measurement_action``
      (float is drifting away; no benefit in short cycles).

    Parameters
    ----------
    default_action:
        Full-depth measurement action returned when conditions allow.
    target_location:
        ``[lat, lon]`` of the circle centre in decimal degrees.
    radius_km:
        Radius of the target circle in kilometres.
    debug:
        If ``True``, opens a live map updated at each cycle. Current position
        is plotted in green (measurement action) or orange (quick action);
        the predicted next position is shown as a smaller marker of the same
        colour. The target circle is drawn as a dashed boundary.
    """

    def __init__(self, default_action: ControlAction, target_location: list[float], radius_km: float, debug: bool = False) -> None:
        self.target_location = GeoLocation(lat=target_location[0], lon=target_location[1])
        self.radius_km = radius_km
        self.default_action = default_action
        self.measurement_action = default_action
        self.name = "circle_drift"
        self.debug = debug
        self.last_action = self.measurement_action
        self._cycle = 0

        self.quick_action = ControlAction(
            park_mode="parking_depth",
            target_depth=10,
            transmission_duration_minutes=self.measurement_action.transmission_duration_minutes,
            ascent_speed_ms=self.measurement_action.ascent_speed_ms,
            descent_speed_ms=self.measurement_action.descent_speed_ms,
            cycle_hours=1,
        )

        if debug:
            import matplotlib.lines as mlines
            plt.ion()
            if HAS_CARTOPY:
                self.fig = plt.figure(figsize=(8, 8))
                self.ax = self.fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                self.ax.add_feature(cfeature.LAND, facecolor="lightgrey", zorder=1)
                self.ax.coastlines(resolution="10m", linewidth=0.6, zorder=2)
                gl = self.ax.gridlines(draw_labels=True, linewidth=0.4, color="grey", alpha=0.5, linestyle="--")
                gl.top_labels = False
                gl.right_labels = False
                self._transform = ccrs.PlateCarree()
            else:
                self.fig, self.ax = plt.subplots(figsize=(8, 8))
                self.ax.set_xlabel("Longitude")
                self.ax.set_ylabel("Latitude")
                self._transform = None

            self.ax.set_title("circle_drift — cycle decisions")

            # Draw target centre and circle boundary
            _scatter_kw = dict(transform=self._transform) if self._transform else {}
            _plot_kw = dict(transform=self._transform) if self._transform else {}
            self.ax.scatter(
                self.target_location.lon, self.target_location.lat,
                color="red", s=120, zorder=5, marker="*", **_scatter_kw,
            )
            radius_deg_lat = self.radius_km / 111.32
            radius_deg_lon = self.radius_km / (111.32 * math.cos(math.radians(self.target_location.lat)))
            theta = np.linspace(0, 2 * np.pi, 360)
            self.ax.plot(
                self.target_location.lon + radius_deg_lon * np.cos(theta),
                self.target_location.lat + radius_deg_lat * np.sin(theta),
                color="red", linestyle="--", linewidth=1.0, **_plot_kw,
            )

            self.ax.legend(handles=[
                mlines.Line2D([], [], marker="o", color="green",  linestyle="None", markersize=8, label="Measurement"),
                mlines.Line2D([], [], marker="o", color="orange", linestyle="None", markersize=8, label="Quick"),
                mlines.Line2D([], [], marker="*", color="red",    linestyle="None", markersize=10, label="Target"),
                mlines.Line2D([], [], color="red", linestyle="--", linewidth=1, label=f"r={radius_km} km"),
            ], loc="upper left")

            # Seed the point list with the target so the initial extent is centred on it
            self._plot_lats = [self.target_location.lat]
            self._plot_lons = [self.target_location.lon]

    def update_debug_plot(self, current: GeoLocation, next_pos: GeoLocation, action: ControlAction, time: datetime) -> None:
        """Redraw the debug plot for the current cycle decision.

        Parameters
        ----------
        current:
            Current float position.
        next_pos:
            Predicted next surface position.
        action:
            The action chosen this cycle (determines point colour).
        time:
            Current simulation time.
        """
        colour = "green" if action is self.measurement_action else "orange"
        kw = dict(transform=self._transform) if self._transform else {}

        self.ax.scatter(current.lon, current.lat, color=colour, s=80, zorder=5, **kw)
        self.ax.scatter(next_pos.lon, next_pos.lat, color=colour, s=30, zorder=4, alpha=0.6, **kw)
        self.ax.plot(
            [current.lon, next_pos.lon], [current.lat, next_pos.lat],
            color=colour, linewidth=0.8, zorder=3, **kw,
        )

        # Update dynamic extent
        self._plot_lats.extend([current.lat, next_pos.lat])
        self._plot_lons.extend([current.lon, next_pos.lon])
        pad = max(self.radius_km / 111.32, 0.5)
        lat_min = min(self._plot_lats) - pad
        lat_max = max(self._plot_lats) + pad
        lon_min = min(self._plot_lons) - pad
        lon_max = max(self._plot_lons) + pad
        if self._transform:
            self.ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=self._transform)
        else:
            self.ax.set_xlim(lon_min, lon_max)
            self.ax.set_ylim(lat_min, lat_max)

        action_name = "measurement" if action is self.measurement_action else "quick"
        self.ax.set_title(
            f"CircleDrift  |  Cycle {self._cycle}  |  {time.strftime('%Y-%m-%d %H:%M')}\n"
            f"({current.lat:.3f}°N, {current.lon:.3f}°E)  → {action_name}"
        )
        self.fig.canvas.draw()
        plt.pause(0.05)

    def predict_next_surface_position(self, profiler_state: ProfilerState, forecast: xr.Dataset, action: ControlAction) -> ProfilerState:
        """Estimate where the float will surface after executing *action*."""
        return quickly_estimate_next_surface_position(profiler_state, action, forecast)

    def is_position_within_circle(self, position: GeoLocation) -> bool:
        """Return True if *position* is within ``radius_km`` of the target."""
        dlat_m = (position.lat - self.target_location.lat) * 111320.0
        dlon_m = (position.lon - self.target_location.lon) * 111320.0 * math.cos(math.radians(self.target_location.lat))
        distance_km = math.sqrt(dlat_m ** 2 + dlon_m ** 2) / 1000.0
        return distance_km <= self.radius_km

    def is_next_position_closer(self, next_position: GeoLocation, current_position: GeoLocation) -> bool:
        """Return True if *next_position* is closer to the target than *current_position*."""
        dlat_m = (next_position.lat - self.target_location.lat) * 111320.0
        dlon_m = (next_position.lon - self.target_location.lon) * 111320.0 * math.cos(math.radians(self.target_location.lat))
        distance_km_next = math.sqrt(dlat_m ** 2 + dlon_m ** 2) / 1000.0

        dlat_m = (current_position.lat - self.target_location.lat) * 111320.0
        dlon_m = (current_position.lon - self.target_location.lon) * 111320.0 * math.cos(math.radians(self.target_location.lat))
        distance_km_current = math.sqrt(dlat_m ** 2 + dlon_m ** 2) / 1000.0

        return distance_km_next < distance_km_current

    def get_action(self, profiler_state: ProfilerState, forecast: xr.Dataset, **kwargs) -> ControlAction:
        """Choose the next action based on current and predicted position.

        Parameters
        ----------
        profiler_state:
            Current float state.
        forecast:
            Forecast dataset used to simulate the next surface position.
        """
        next_position = self.predict_next_surface_position(profiler_state, forecast, self.last_action)
        currently_inside = self.is_position_within_circle(profiler_state.location)
        next_inside = self.is_position_within_circle(next_position.location)

        if currently_inside:
            if next_inside:
                logger.info("Inside circle, next position also inside — taking measurement action")
                action = self.measurement_action
            else:
                logger.info("Inside circle but predicted to drift outside — taking quick action to stay in")
                action = self.quick_action
        else:
            if self.is_next_position_closer(next_position.location, profiler_state.location):
                logger.info("Outside circle, drifting closer — taking quick action to recheck sooner")
                action = self.quick_action
            else:
                # NOTE: float is outside the circle and drifting further away.
                # Returning measurement_action here — consider whether quick_action
                # would be more appropriate to avoid compounding the drift.
                logger.info("Outside circle, drifting away — taking measurement action")
                action = self.measurement_action

        self.last_action = action
        self._cycle += 1
        if self.debug:
            self.update_debug_plot(profiler_state.location, next_position.location, action, profiler_state.time)
        return action

    def get_log(self) -> dict:
        return {
            "control_strategy": self.name,
            "target_location": [self.target_location.lat, self.target_location.lon],
            "radius_km": self.radius_km,
            "measurement_cycle_hours": self.measurement_action.cycle_hours,
            "quick_cycle_hours": self.quick_action.cycle_hours,
            "park_mode": self.measurement_action.park_mode,
            "ascent_speed": self.measurement_action.ascent_speed_ms,
            "descent_speed": self.measurement_action.descent_speed_ms,
        }

    
class CircleMPC(ControlStrategy):
    """Model-predictive geofencing: keep the float within a target circle.

    When inside the circle, behaviour mirrors :class:`circle_drift` — take a
    full measurement dive while conditions allow, switch to a short shallow
    cycle when the float is predicted to drift outside.

    When outside the circle, a finite-horizon search is run over all
    combinations of ``possible_actions`` up to ``action_horizon`` steps deep.
    The action sequence whose terminal state is closest to the circle centre
    is selected, and the **first** action in that sequence is executed.

    This is a receding-horizon (MPC) approach: only the immediate next action
    is taken, and the search is repeated at each surfacing.

    The candidate action set contains three fixed actions plus the caller's
    measurement action:

    - ``ten_quick_profiler``: 1-hour cycle parking at 10 m — stays near the
      surface to exploit surface currents.
    - ``mid_6h_profile``: 6-hour cycle parking at 20 m — medium-depth drift.
    - ``bottom_1_day``: 25-hour cycle parking on the seabed — deep slow drift.
    - ``measurement_action``: the caller-supplied default.

    Parameters
    ----------
    default_action:
        Full measurement action; included in the candidate set and used as
        the fallback inside the circle when conditions allow.
    target_location:
        ``[lat, lon]`` of the circle centre in decimal degrees.
    radius_km:
        Radius of the target circle in kilometres.
    debug:
        Reserved for future visualisation.
    """

    def __init__(self, default_action: ControlAction, target_location: list[float], radius_km: float, debug: bool = False) -> None:
        self.target_location = GeoLocation(lat=target_location[0], lon=target_location[1])
        self.radius_km = radius_km
        self.measurement_action = default_action
        self.name = "circle_mpc"
        self.debug = debug
        self.last_action = self.measurement_action
        self.default_action = default_action
        self._cycle = 0

        self.ten_quick_profiler = ControlAction(
            park_mode="parking_depth",
            target_depth=10,
            transmission_duration_minutes=self.measurement_action.transmission_duration_minutes,
            ascent_speed_ms=self.measurement_action.ascent_speed_ms,
            descent_speed_ms=self.measurement_action.descent_speed_ms,
            cycle_hours=1,
        )

        self.mid_6h_profile = ControlAction(
            park_mode="parking_depth",
            target_depth=20,
            transmission_duration_minutes=self.measurement_action.transmission_duration_minutes,
            ascent_speed_ms=self.measurement_action.ascent_speed_ms,
            descent_speed_ms=self.measurement_action.descent_speed_ms,
            cycle_hours=6,
        )

        self.bottom_1_day = ControlAction(
            park_mode="park_on_bottom",
            transmission_duration_minutes=self.measurement_action.transmission_duration_minutes,
            ascent_speed_ms=self.measurement_action.ascent_speed_ms,
            descent_speed_ms=self.measurement_action.descent_speed_ms,
            cycle_hours=25,
        )

        self.time_horizon_hours = 120
        self.action_horizon = 2
        self.possible_actions = [self.ten_quick_profiler, self.mid_6h_profile, self.bottom_1_day, self.measurement_action]

        # Map each candidate action to a fixed display colour
        self._action_colours = {
            id(self.ten_quick_profiler): "orange",
            id(self.mid_6h_profile):     "mediumpurple",
            id(self.bottom_1_day):       "saddlebrown",
            id(self.measurement_action): "green",
        }

        if debug:
            import matplotlib.lines as mlines
            plt.ion()
            if HAS_CARTOPY:
                self.fig = plt.figure(figsize=(8, 8))
                self.ax = self.fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                self.ax.add_feature(cfeature.LAND, facecolor="lightgrey", zorder=1)
                self.ax.coastlines(resolution="10m", linewidth=0.6, zorder=2)
                gl = self.ax.gridlines(draw_labels=True, linewidth=0.4, color="grey", alpha=0.5, linestyle="--")
                gl.top_labels = False
                gl.right_labels = False
                self._transform = ccrs.PlateCarree()
            else:
                self.fig, self.ax = plt.subplots(figsize=(8, 8))
                self.ax.set_xlabel("Longitude")
                self.ax.set_ylabel("Latitude")
                self._transform = None

            self.ax.set_title("circle_mpc — cycle decisions")

            _scatter_kw = dict(transform=self._transform) if self._transform else {}
            _plot_kw    = dict(transform=self._transform) if self._transform else {}

            # Target centre and circle boundary
            self.ax.scatter(
                self.target_location.lon, self.target_location.lat,
                color="red", s=120, zorder=5, marker="*", **_scatter_kw,
            )
            radius_deg_lat = self.radius_km / 111.32
            radius_deg_lon = self.radius_km / (111.32 * math.cos(math.radians(self.target_location.lat)))
            theta = np.linspace(0, 2 * np.pi, 360)
            self.ax.plot(
                self.target_location.lon + radius_deg_lon * np.cos(theta),
                self.target_location.lat + radius_deg_lat * np.sin(theta),
                color="red", linestyle="--", linewidth=1.0, **_plot_kw,
            )

            self.ax.legend(handles=[
                mlines.Line2D([], [], marker="o", color="green",        linestyle="None", markersize=8,  label="Measurement"),
                mlines.Line2D([], [], marker="o", color="orange",       linestyle="None", markersize=8,  label="Quick (10 m, 1 h)"),
                mlines.Line2D([], [], marker="o", color="mediumpurple", linestyle="None", markersize=8,  label="Mid (20 m, 6 h)"),
                mlines.Line2D([], [], marker="o", color="saddlebrown",  linestyle="None", markersize=8,  label="Bottom (1 day)"),
                mlines.Line2D([], [], marker="*", color="red",          linestyle="None", markersize=10, label="Target"),
                mlines.Line2D([], [], color="red", linestyle="--", linewidth=1, label=f"r={radius_km} km"),
            ], loc="upper left")

            self._plot_lats = [self.target_location.lat]
            self._plot_lons = [self.target_location.lon]

    def update_debug_plot(
        self,
        current: GeoLocation,
        next_pos: GeoLocation,
        action: ControlAction,
        time: datetime,
        mpc_path: list[ProfilerState] | None = None,
    ) -> None:
        """Redraw the debug plot for the current cycle decision.

        Parameters
        ----------
        current:
            Current float position.
        next_pos:
            Predicted position after the chosen first action.
        action:
            The action chosen this cycle (determines point colour).
        time:
            Current simulation time.
        mpc_path:
            When outside the circle, the full sequence of predicted states for
            the best MPC path.  Drawn as a faded chain so you can see how far
            ahead the planner looked.
        """
        colour = self._action_colours.get(id(action), "blue")
        kw = dict(transform=self._transform) if self._transform else {}

        # Current position (large dot)
        self.ax.scatter(current.lon, current.lat, color=colour, s=80, zorder=5, **kw)
        # Predicted next position (smaller, semi-transparent)
        self.ax.scatter(next_pos.lon, next_pos.lat, color=colour, s=30, zorder=4, alpha=0.6, **kw)
        self.ax.plot(
            [current.lon, next_pos.lon], [current.lat, next_pos.lat],
            color=colour, linewidth=0.8, zorder=3, **kw,
        )

        self._plot_lats.extend([current.lat, next_pos.lat])
        self._plot_lons.extend([current.lon, next_pos.lon])

        # MPC lookahead chain (faded grey dashed line)
        if mpc_path and len(mpc_path) > 1:
            path_lons = [s.location.lon for s in mpc_path]
            path_lats = [s.location.lat for s in mpc_path]
            self.ax.plot(path_lons, path_lats, color="grey", linewidth=0.6,
                         linestyle=":", alpha=0.5, zorder=2, **kw)
            self._plot_lats.extend(path_lats)
            self._plot_lons.extend(path_lons)

        # Dynamic extent with padding
        pad = max(self.radius_km / 111.32, 0.5)
        lat_min = min(self._plot_lats) - pad
        lat_max = max(self._plot_lats) + pad
        lon_min = min(self._plot_lons) - pad
        lon_max = max(self._plot_lons) + pad
        if self._transform:
            self.ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=self._transform)
        else:
            self.ax.set_xlim(lon_min, lon_max)
            self.ax.set_ylim(lat_min, lat_max)

        action_name = {
            id(self.ten_quick_profiler): "quick 10 m",
            id(self.mid_6h_profile):     "mid 20 m",
            id(self.bottom_1_day):       "bottom 1 day",
            id(self.measurement_action): "measurement",
        }.get(id(action), "unknown")
        self.ax.set_title(
            f"CircleMPC  |  Cycle {self._cycle}  |  {time.strftime('%Y-%m-%d %H:%M')}\n"
            f"({current.lat:.3f}°N, {current.lon:.3f}°E)  → {action_name}"
        )
        self.fig.canvas.draw()
        plt.pause(0.05)

    def predict_next_surface_position(self, profiler_state: ProfilerState, forecast: xr.Dataset,
                                      action: ControlAction) -> ProfilerState:
        """Estimate where the float will surface after executing *action*."""
        return quickly_estimate_next_surface_position(profiler_state, action, forecast)

    def distance_from_center(self, location: GeoLocation) -> float:
        """Return the great-circle distance in km from *location* to the circle centre."""
        dlat_m = (location.lat - self.target_location.lat) * 111320.0
        dlon_m = (location.lon - self.target_location.lon) * 111320.0 * math.cos(
            math.radians(self.target_location.lat))
        return math.sqrt(dlat_m ** 2 + dlon_m ** 2) / 1000.0

    def is_position_within_circle(self, location: GeoLocation) -> bool:
        """Return True if *location* is within ``radius_km`` of the target."""
        return self.distance_from_center(location) <= self.radius_km

    def get_action(self, profiler_state: ProfilerState, forecast: xr.Dataset, **kwargs) -> ControlAction:
        """Choose the next action, using MPC when outside the target circle.

        When inside, mirrors :class:`circle_drift` logic. When outside,
        searches all action sequences of length ``action_horizon`` and returns
        the first action from the sequence that minimises the terminal distance
        from the circle centre, subject to a ``time_horizon_hours`` budget.

        Parameters
        ----------
        profiler_state:
            Current float state.
        forecast:
            Forecast dataset used to simulate candidate action sequences.
        """
        first_state = self.predict_next_surface_position(profiler_state, forecast, self.last_action)
        currently_inside = self.is_position_within_circle(profiler_state.location)
        next_inside = self.is_position_within_circle(first_state.location)
        mpc_path: list[ProfilerState] | None = None

        if currently_inside:
            if next_inside:
                logger.info("Inside circle, next position also inside — taking measurement action")
                action = self.measurement_action
            else:
                logger.info("Inside circle but predicted to drift outside — taking quick action")
                action = self.ten_quick_profiler
        else:
            # MPC: enumerate all action sequences of length action_horizon,
            # simulate each by chaining predicted states, then pick the sequence
            # whose terminal position is closest to the circle centre.
            closest_final_distance = self.distance_from_center(self.predict_next_surface_position(profiler_state, forecast, self.measurement_action).location)
            best_first_action = self.measurement_action
            best_path_states: list[ProfilerState] = []

            for path in product(self.possible_actions, repeat=self.action_horizon):
                elapsed_hours = 0
                state = first_state
                path_states = [first_state]
                for step_action in path:
                    state = self.predict_next_surface_position(state, forecast, step_action)
                    path_states.append(state)
                    elapsed_hours += step_action.cycle_hours
                    if elapsed_hours > self.time_horizon_hours:
                        break
                final_distance = self.distance_from_center(state.location)
                if final_distance < closest_final_distance:
                    closest_final_distance = final_distance
                    best_first_action = path[0]
                    best_path_states = path_states

            logger.info(
                "Outside circle — MPC selected first action with park_mode=%s "
                "(predicted terminal distance %.2f km)",
                best_first_action.park_mode, closest_final_distance,
            )
            action = best_first_action
            mpc_path = best_path_states

        self.last_action = action
        self._cycle += 1
        if self.debug:
            self.update_debug_plot(profiler_state.location, first_state.location, action, profiler_state.time, mpc_path)
        return action

    def get_log(self) -> dict:
        return {
            "control_strategy": self.name,
            "target_location": [self.target_location.lat, self.target_location.lon],
            "radius_km": self.radius_km,
            "time_horizon_hours": self.time_horizon_hours,
            "action_horizon": self.action_horizon,
            "measurement_cycle_hours": self.measurement_action.cycle_hours,
            "park_mode": self.measurement_action.park_mode,
            "ascent_speed": self.measurement_action.ascent_speed_ms,
            "descent_speed": self.measurement_action.descent_speed_ms,
        }


class MPCwithFavourable(ControlStrategy):
    """Cost-function MPC: minimise distance to target while rewarding favourable flow.

    At each surfacing every candidate action (park mode × depth × duration) is
    evaluated by predicting the resulting surface position and scoring it with:

        cost = distance_weight × distance_km
             − final_state_weight × Σ_t (v_forecast(t) · û_toward_target)

    The first term penalises landing far from ``target_location``.
    The second term rewards positions where the forecast currents over the next
    ``time_horizon_final_state_hours`` hours point toward the target — i.e. the
    float will be carried closer even without further active control.

    The action with the lowest cost is executed.

    Parameters
    ----------
    default_action:
        Reference action; provides ascent/descent speeds and transmission
        duration inherited by all candidate actions.
    target_location:
        ``[lat, lon]`` of the target in decimal degrees.
    radius_km:
        Informational only — stored for logging; not used in cost function.
    debug:
        If ``True``, draws a live map of cycle decisions (requires matplotlib)
        and emits per-action cost breakdowns via the logger at DEBUG level.
    """

    def __init__(
        self,
        default_action: ControlAction,
        target_location: list[float],
        debug: bool = False,
    ) -> None:
        self.target_location = GeoLocation(lat=target_location[0], lon=target_location[1])
        self.measurement_action = default_action
        self.default_action = default_action
        self.name = "mpc_with_favourable"
        self.debug = debug
        self.last_action = self.measurement_action
        self._cycle = 0

        # How many hours of forecast flow to sum at the predicted landing position
        self.time_horizon_final_state_hours = 6
        # Cost-function weights
        self.distance_weight = 1
        self.flow_weight = -0.8*1/self.time_horizon_final_state_hours


        mid_depth = 30.0   # metres — mid-water between near-surface (10 m) and bottom

        # Candidate actions: list of dicts with keys park_mode, target_depth, cycle_hours.
        # Three depth regimes × varying durations to give the optimiser real choice.
        self._action_configs = [
            # ── Near-surface: short cycles to exploit surface currents ──────
            {"park_mode": "parking_depth", "target_depth": 10.0, "cycle_hours":   1},
            {"park_mode": "parking_depth", "target_depth": 10.0, "cycle_hours":   2},
            {"park_mode": "parking_depth", "target_depth": 10.0, "cycle_hours":   4},
            {"park_mode": "parking_depth", "target_depth": 10.0, "cycle_hours":   6},
            # ── Bottom: slow deep drift, long intervals ───────────────────
            {"park_mode": "park_on_bottom", "target_depth": None, "cycle_hours":  24},
            {"park_mode": "park_on_bottom", "target_depth": None, "cycle_hours":  36},
            {"park_mode": "park_on_bottom", "target_depth": None, "cycle_hours":  48},
            {"park_mode": "park_on_bottom", "target_depth": None, "cycle_hours":  72},
            {"park_mode": "park_on_bottom", "target_depth": None, "cycle_hours": 120},
            # ── Mid-depth: medium drift at ~40 m ──────────────────────────
            {"park_mode": "parking_depth", "target_depth": mid_depth, "cycle_hours":  8},
            {"park_mode": "parking_depth", "target_depth": mid_depth, "cycle_hours": 12},
            {"park_mode": "parking_depth", "target_depth": mid_depth, "cycle_hours": 18},
            {"park_mode": "parking_depth", "target_depth": mid_depth, "cycle_hours": 36},
            {"park_mode": "parking_depth", "target_depth": mid_depth, "cycle_hours": 72},
        ]

        if debug:
            import matplotlib.lines as mlines
            import matplotlib.patches as mpatches
            plt.ion()
            if HAS_CARTOPY:
                self.fig = plt.figure(figsize=(14, 10))
                self.ax = self.fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                self._transform = ccrs.PlateCarree()
            else:
                self.fig, self.ax = plt.subplots(figsize=(14, 10))
                self._transform = None

    @staticmethod
    def _action_color(cfg: dict) -> str:
        if cfg["park_mode"] == "park_on_bottom":
            return "saddlebrown"
        if cfg["target_depth"] is not None and cfg["target_depth"] <= 15:
            return "orange"
        return "mediumpurple"

    def _build_action(self, cfg: dict) -> ControlAction:
        return ControlAction(
            cycle_hours=float(cfg["cycle_hours"]),
            park_mode=cfg["park_mode"],
            target_depth=cfg["target_depth"],
            transmission_duration_minutes=self.measurement_action.transmission_duration_minutes,
            ascent_speed_ms=self.measurement_action.ascent_speed_ms,
            descent_speed_ms=self.measurement_action.descent_speed_ms,
        )

    def predict_next_surface_position(
        self,
        profiler_state: ProfilerState,
        forecast: xr.Dataset,
        action: ControlAction,
    ) -> ProfilerState:
        """Estimate where the float will surface after executing *action*."""
        return quickly_estimate_next_surface_position(profiler_state, action, forecast)

    def distance_from_center(self, location: GeoLocation) -> float:
        """Return the distance in km from *location* to ``target_location``."""
        dlat_m = (location.lat - self.target_location.lat) * 111320.0
        dlon_m = (location.lon - self.target_location.lon) * 111320.0 * math.cos(
            math.radians(self.target_location.lat)
        )
        return math.sqrt(dlat_m ** 2 + dlon_m ** 2) / 1000.0

    def distance_term(self, action_location, current_location, action: ControlAction) -> float:
        """Return the cost term for distance to the target."""
        delta_d = self.distance_from_center(action_location) - self.distance_from_center(current_location)
        velocity_term = delta_d / action.cycle_hours
        return self.distance_weight * velocity_term

    def flow_term(
        self,
        forecast_interpolators: tuple,
        predicted_state: ProfilerState,
    ) -> float:
        """Sum of surface-current projections toward the target at the predicted position.

        For each hour in ``time_horizon_final_state_hours``, projects the
        forecast surface velocity onto the unit vector pointing from
        ``predicted_state.location`` toward ``target_location``. A positive
        return value means the currents will on average push the float toward
        the target — i.e. this is a favourable position to land in.

        Parameters
        ----------
        forecast_interpolators:
            ``(interp_u, interp_v)`` from :func:`build_velocity_interpolator`.
            Each callable takes ``[[t_s, depth_m, lat, lon]]`` and returns an
            array of velocities in m/s.
        predicted_state:
            Float state at the end of the candidate dive (surface, phase=communicating).

        Returns
        -------
        float
            Cumulative dot-product sum (m/s × hours).  Positive = favourable.
        """
        from datetime import timedelta as _td

        location = predicted_state.location
        time     = predicted_state.time
        interp_u, interp_v = forecast_interpolators

        # Unit vector from predicted position toward target, scaled for metric space.
        dlat = self.target_location.lat - location.lat
        dlon = (self.target_location.lon - location.lon) * math.cos(math.radians(location.lat))
        norm = math.sqrt(dlat ** 2 + dlon ** 2)
        if norm == 0.0:
            return 0.0
        unit_north = dlat / norm   # northward (lat) component
        unit_east  = dlon / norm   # eastward  (lon) component

        depth_m = 0.5   # first CMEMS depth level — surface currents
        total = 0.0
        for i in range(self.time_horizon_final_state_hours):
            t_s = np.datetime64(time + _td(hours=i), "s").astype(np.float64)
            u = float(interp_u([[t_s, depth_m, location.lat, location.lon]])[0])
            v = float(interp_v([[t_s, depth_m, location.lat, location.lon]])[0])
            if math.isnan(u) or math.isnan(v):
                continue
            # u = eastward, v = northward — project onto unit vector toward target
            total += unit_east * u + unit_north * v

        return total

    def update_debug_plot(
        self,
        current: GeoLocation,
        best_next: GeoLocation,
        all_candidates: list[tuple[GeoLocation, float, str]],
        chosen_color: str,
        action: ControlAction,
        time: datetime,
    ) -> None:
        """Redraw the live map each cycle: future trajectories only, colored by action category."""
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches

        self.ax.cla()
        kw = dict(transform=self._transform) if self._transform else {}

        if HAS_CARTOPY:
            self.ax.add_feature(cfeature.LAND, facecolor="lightgrey", zorder=1)
            self.ax.coastlines(resolution="10m", linewidth=0.6, zorder=2)
            gl = self.ax.gridlines(draw_labels=True, linewidth=0.4, color="grey", alpha=0.5, linestyle="--")
            gl.top_labels = False
            gl.right_labels = False
        else:
            self.ax.set_xlabel("Longitude")
            self.ax.set_ylabel("Latitude")

        # Target
        self.ax.scatter(
            self.target_location.lon, self.target_location.lat,
            color="red", s=150, zorder=8, marker="*", **kw,
        )

        # Candidate trajectories: thin colored spokes
        all_lats = [current.lat, best_next.lat, self.target_location.lat]
        all_lons = [current.lon, best_next.lon, self.target_location.lon]
        for loc, _cost, color in all_candidates:
            self.ax.plot(
                [current.lon, loc.lon], [current.lat, loc.lat],
                color=color, linewidth=1.2, alpha=0.45, zorder=3, **kw,
            )
            self.ax.scatter(loc.lon, loc.lat, color=color, s=25, alpha=0.6, zorder=3, **kw)
            all_lats.append(loc.lat)
            all_lons.append(loc.lon)

        # Chosen trajectory: highlighted
        self.ax.plot(
            [current.lon, best_next.lon], [current.lat, best_next.lat],
            color=chosen_color, linewidth=3, zorder=6, **kw,
        )
        self.ax.scatter(best_next.lon, best_next.lat, color=chosen_color, s=120, zorder=6, **kw)

        # Current position
        self.ax.scatter(current.lon, current.lat, color="steelblue", s=80, zorder=7, **kw)

        # Extent
        pad = 0.3
        if self._transform:
            self.ax.set_extent(
                [min(all_lons) - pad, max(all_lons) + pad,
                 min(all_lats) - pad, max(all_lats) + pad],
                crs=self._transform,
            )
        else:
            self.ax.set_xlim(min(all_lons) - pad, max(all_lons) + pad)
            self.ax.set_ylim(min(all_lats) - pad, max(all_lats) + pad)

        # Legend
        handles = [
            mpatches.Patch(color="orange",       label="Surface ~10 m"),
            mpatches.Patch(color="saddlebrown",  label="Bottom park"),
            mpatches.Patch(color="mediumpurple", label="Mid ~40 m"),
            mlines.Line2D([], [], color="steelblue", marker="o", markersize=7,
                          linestyle="None", label="Current position"),
            mlines.Line2D([], [], color="red", marker="*", markersize=10,
                          linestyle="None", label="Target"),
        ]
        self.ax.legend(handles=handles, loc="upper left", fontsize=8)

        depth_str = f"{action.target_depth:.0f} m" if action.target_depth is not None else "bottom"
        self.ax.set_title(
            f"MPCwithFavourable  |  Cycle {self._cycle}  |  {time.strftime('%Y-%m-%d %H:%M')}\n"
            f"({current.lat:.3f}°N, {current.lon:.3f}°E)  → {action.park_mode} {depth_str} {action.cycle_hours:.0f}h"
        )
        self.fig.canvas.draw()
        plt.pause(0.05)

    def get_action(self, profiler_state: ProfilerState, forecast: xr.Dataset, **kwargs) -> ControlAction:
        """Evaluate all candidate actions and return the one with the lowest cost.

        Parameters
        ----------
        profiler_state:
            Current float state (position and time).
        forecast:
            Forecast dataset used to simulate candidate dives and evaluate
            the favourable-flow term.
        """
        forecast_interpolators = build_velocity_interpolator(forecast)
        next_state = self.predict_next_surface_position(profiler_state, forecast, self.last_action)
        current_distance = self.distance_from_center(next_state.location)

        best_cost    = math.inf
        best_cfg     = self._action_configs[0]
        best_state   = None
        best_dist    = 0.0
        best_flow    = 0.0
        all_candidates: list[tuple[GeoLocation, float, str]] = []

        logger.debug(
            "Cycle %d | lat=%.4f lon=%.4f | evaluating %d candidate actions",
            self._cycle,
            profiler_state.location.lat,
            profiler_state.location.lon,
            len(self._action_configs),
        )
        logger.debug(
            "  %-14s %-7s %5s | %8s  %-10s | %9s  %-12s | %s",
            "park_mode", "depth", "hours",
            "dist_km", "dist_term",
            "flow_sum", "flow_term",
            "total_cost",
        )

        for cfg in self._action_configs:
            action         = self._build_action(cfg)
            predicted      = self.predict_next_surface_position(next_state, forecast, action)
            distance_term  = self.distance_term(predicted.location, next_state.location, action)
            flow_term      = self.flow_term(forecast_interpolators, predicted)
            cost           = self.distance_weight * distance_term + self.flow_weight * flow_term

            depth_str = f"{cfg['target_depth']:.0f} m" if cfg["target_depth"] is not None else "bottom"
            logger.debug(
                "  %-14s %-7s %5.0fh | %8.2f  %+10.4f | %+9.4f  %+12.4f | %+.4f",
                cfg["park_mode"], depth_str, cfg["cycle_hours"],
                distance_term, flow_term,
                cost,
            )

            all_candidates.append((predicted.location, cost, self._action_color(cfg)))

            if cost < best_cost:
                best_cost  = cost
                best_cfg   = cfg
                best_state = predicted
                best_dist  = distance_term
                best_flow  = flow_term

        chosen_action = self._build_action(best_cfg)
        self.last_action = chosen_action
        self._cycle += 1

        depth_str = f"{best_cfg['target_depth']:.0f} m" if best_cfg["target_depth"] is not None else "bottom"
        logger.info(
            "Cycle %d | CHOSEN: %s %s %gh | "
            "dist=%.2f km (term=%+.4f) | "
            "flow_sum=%+.4f (term=%+.4f) | "
            "total_cost=%+.4f",
            self._cycle,
            best_cfg["park_mode"], depth_str, best_cfg["cycle_hours"],
            best_dist,  self.distance_weight * best_dist,
            best_flow, self.flow_weight * best_flow,
            best_cost,
        )

        if self.debug:
            self.update_debug_plot(
                profiler_state.location,
                best_state.location,
                all_candidates,
                self._action_color(best_cfg),
                chosen_action,
                profiler_state.time,
            )

        return chosen_action

    def get_log(self) -> dict:
        return {
            "control_strategy": self.name,
            "target_location": [self.target_location.lat, self.target_location.lon],
            "distance_weight": self.distance_weight,
            "flow_weight": self.flow_weight,
            "time_horizon_final_state_hours": self.time_horizon_final_state_hours,
            "n_candidate_actions": len(self._action_configs),
            "ascent_speed": self.measurement_action.ascent_speed_ms,
            "descent_speed": self.measurement_action.descent_speed_ms,
        }




class MPCwithFavourableMeasurement(ControlStrategy):
    """Cost-function MPC: minimise distance to target while rewarding favourable flow.

    At each surfacing every candidate action (park mode × depth × duration) is
    evaluated by predicting the resulting surface position and scoring it with:

        cost = distance_weight × distance_km
             − final_state_weight × Σ_t (v_forecast(t) · û_toward_target)

    The first term penalises landing far from ``target_location``.
    The second term rewards positions where the forecast currents over the next
    ``time_horizon_final_state_hours`` hours point toward the target — i.e. the
    float will be carried closer even without further active control.

    The action with the lowest cost is executed.

    Parameters
    ----------
    default_action:
        Reference action; provides ascent/descent speeds and transmission
        duration inherited by all candidate actions.
    target_location:
        ``[lat, lon]`` of the target in decimal degrees.
    radius_km:
        Informational only — stored for logging; not used in cost function.
    debug:
        If ``True``, draws a live map of cycle decisions (requires matplotlib)
        and emits per-action cost breakdowns via the logger at DEBUG level.
    """

    def __init__(
        self,
        default_action: ControlAction,
        target_location: list[float],
        radius_std_dv: float,
        debug: bool = False,
    ) -> None:
        self.target_location = GeoLocation(lat=target_location[0], lon=target_location[1])
        self.measurement_action = default_action
        self.default_action = default_action
        self.name = "mpc_with_favourable_measurement"
        self.debug = debug
        self.last_action = self.measurement_action
        self._cycle = 0
        self.radius_std_dv = radius_std_dv

        # How many hours of forecast flow to sum at the predicted landing position
        self.time_horizon_final_state_hours = 6
        # Cost-function weights
        self.distance_weight = 1
        self.measurement_weight = 1
        self.flow_weight = -0.8*1/self.time_horizon_final_state_hours


        mid_depth = 30.0   # metres — mid-water between near-surface (10 m) and bottom

        # Candidate actions: list of dicts with keys park_mode, target_depth, cycle_hours.
        # Three depth regimes × varying durations to give the optimiser real choice.
        self._action_configs = [
            # ── Near-surface: short cycles to exploit surface currents ──────
            {"park_mode": "parking_depth", "target_depth": 10.0, "cycle_hours":   1, "science_cost": 1},
            {"park_mode": "parking_depth", "target_depth": 10.0, "cycle_hours":   2, "science_cost": 1},
            {"park_mode": "parking_depth", "target_depth": 10.0, "cycle_hours":   4, "science_cost": 1},
            {"park_mode": "parking_depth", "target_depth": 10.0, "cycle_hours":   6, "science_cost": 1},
            # ── Bottom: slow deep drift, long intervals ───────────────────
            {"park_mode": "park_on_bottom", "target_depth": None, "cycle_hours":  24, "science_cost": 0.8},
            {"park_mode": "park_on_bottom", "target_depth": None, "cycle_hours":  36, "science_cost": 0.6},
            {"park_mode": "park_on_bottom", "target_depth": None, "cycle_hours":  48, "science_cost": 0.5},
            {"park_mode": "park_on_bottom", "target_depth": None, "cycle_hours":  72, "science_cost": 0.25},
            {"park_mode": "park_on_bottom", "target_depth": None, "cycle_hours": 120, "science_cost": 0},
            # ── Mid-depth: medium drift at ~40 m ──────────────────────────
            {"park_mode": "parking_depth", "target_depth": mid_depth, "cycle_hours":  8, "science_cost": 1},
            {"park_mode": "parking_depth", "target_depth": mid_depth, "cycle_hours": 12, "science_cost": 1},
            {"park_mode": "parking_depth", "target_depth": mid_depth, "cycle_hours": 18, "science_cost": 1},
            {"park_mode": "parking_depth", "target_depth": mid_depth, "cycle_hours": 36, "science_cost": 1},
            {"park_mode": "parking_depth", "target_depth": mid_depth, "cycle_hours": 72, "science_cost": 1},
        ]

        if debug:
            import matplotlib.lines as mlines
            import matplotlib.patches as mpatches
            plt.ion()
            if HAS_CARTOPY:
                self.fig = plt.figure(figsize=(14, 10))
                self.ax = self.fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                self._transform = ccrs.PlateCarree()
            else:
                self.fig, self.ax = plt.subplots(figsize=(14, 10))
                self._transform = None

    @staticmethod
    def _action_color(cfg: dict) -> str:
        if cfg["park_mode"] == "park_on_bottom":
            return "saddlebrown"
        if cfg["target_depth"] is not None and cfg["target_depth"] <= 15:
            return "orange"
        return "mediumpurple"

    def _build_action(self, cfg: dict) -> ControlAction:
        return ControlAction(
            cycle_hours=float(cfg["cycle_hours"]),
            park_mode=cfg["park_mode"],
            target_depth=cfg["target_depth"],
            transmission_duration_minutes=self.measurement_action.transmission_duration_minutes,
            ascent_speed_ms=self.measurement_action.ascent_speed_ms,
            descent_speed_ms=self.measurement_action.descent_speed_ms,
        )

    def predict_next_surface_position(
        self,
        profiler_state: ProfilerState,
        forecast: xr.Dataset,
        action: ControlAction,
    ) -> ProfilerState:
        """Estimate where the float will surface after executing *action*."""
        return quickly_estimate_next_surface_position(profiler_state, action, forecast)

    def distance_from_center(self, location: GeoLocation) -> float:
        """Return the distance in km from *location* to ``target_location``."""
        dlat_m = (location.lat - self.target_location.lat) * 111320.0
        dlon_m = (location.lon - self.target_location.lon) * 111320.0 * math.cos(
            math.radians(self.target_location.lat)
        )
        return math.sqrt(dlat_m ** 2 + dlon_m ** 2) / 1000.0

    def distance_term(self, action_location, current_location, action: ControlAction) -> float:
        """Return the cost term for distance to the target."""
        delta_d = self.distance_from_center(action_location) - self.distance_from_center(current_location)
        velocity_term = delta_d / action.cycle_hours
        return self.distance_weight * velocity_term

    def flow_term(
        self,
        forecast_interpolators: tuple,
        predicted_state: ProfilerState,
    ) -> float:
        """Sum of surface-current projections toward the target at the predicted position.

        For each hour in ``time_horizon_final_state_hours``, projects the
        forecast surface velocity onto the unit vector pointing from
        ``predicted_state.location`` toward ``target_location``. A positive
        return value means the currents will on average push the float toward
        the target — i.e. this is a favourable position to land in.

        Parameters
        ----------
        forecast_interpolators:
            ``(interp_u, interp_v)`` from :func:`build_velocity_interpolator`.
            Each callable takes ``[[t_s, depth_m, lat, lon]]`` and returns an
            array of velocities in m/s.
        predicted_state:
            Float state at the end of the candidate dive (surface, phase=communicating).

        Returns
        -------
        float
            Cumulative dot-product sum (m/s × hours).  Positive = favourable.
        """
        from datetime import timedelta as _td

        location = predicted_state.location
        time     = predicted_state.time
        interp_u, interp_v = forecast_interpolators

        # Unit vector from predicted position toward target, scaled for metric space.
        dlat = self.target_location.lat - location.lat
        dlon = (self.target_location.lon - location.lon) * math.cos(math.radians(location.lat))
        norm = math.sqrt(dlat ** 2 + dlon ** 2)
        if norm == 0.0:
            return 0.0
        unit_north = dlat / norm   # northward (lat) component
        unit_east  = dlon / norm   # eastward  (lon) component

        depth_m = 0.5   # first CMEMS depth level — surface currents
        total = 0.0
        for i in range(self.time_horizon_final_state_hours):
            t_s = np.datetime64(time + _td(hours=i), "s").astype(np.float64)
            u = float(interp_u([[t_s, depth_m, location.lat, location.lon]])[0])
            v = float(interp_v([[t_s, depth_m, location.lat, location.lon]])[0])
            if math.isnan(u) or math.isnan(v):
                continue
            # u = eastward, v = northward — project onto unit vector toward target
            total += unit_east * u + unit_north * v

        return total

    def update_debug_plot(
        self,
        current: GeoLocation,
        best_next: GeoLocation,
        all_candidates: list[tuple[GeoLocation, float, str]],
        chosen_color: str,
        action: ControlAction,
        time: datetime,
    ) -> None:
        """Redraw the live map each cycle: future trajectories only, colored by action category."""
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches

        self.ax.cla()
        kw = dict(transform=self._transform) if self._transform else {}

        if HAS_CARTOPY:
            self.ax.add_feature(cfeature.LAND, facecolor="lightgrey", zorder=1)
            self.ax.coastlines(resolution="10m", linewidth=0.6, zorder=2)
            gl = self.ax.gridlines(draw_labels=True, linewidth=0.4, color="grey", alpha=0.5, linestyle="--")
            gl.top_labels = False
            gl.right_labels = False
        else:
            self.ax.set_xlabel("Longitude")
            self.ax.set_ylabel("Latitude")

        # Target
        self.ax.scatter(
            self.target_location.lon, self.target_location.lat,
            color="red", s=150, zorder=8, marker="*", **kw,
        )

        # Candidate trajectories: thin colored spokes
        all_lats = [current.lat, best_next.lat, self.target_location.lat]
        all_lons = [current.lon, best_next.lon, self.target_location.lon]
        for loc, _cost, color in all_candidates:
            self.ax.plot(
                [current.lon, loc.lon], [current.lat, loc.lat],
                color=color, linewidth=1.2, alpha=0.45, zorder=3, **kw,
            )
            self.ax.scatter(loc.lon, loc.lat, color=color, s=25, alpha=0.6, zorder=3, **kw)
            all_lats.append(loc.lat)
            all_lons.append(loc.lon)

        # Chosen trajectory: highlighted
        self.ax.plot(
            [current.lon, best_next.lon], [current.lat, best_next.lat],
            color=chosen_color, linewidth=3, zorder=6, **kw,
        )
        self.ax.scatter(best_next.lon, best_next.lat, color=chosen_color, s=120, zorder=6, **kw)

        # Current position
        self.ax.scatter(current.lon, current.lat, color="steelblue", s=80, zorder=7, **kw)

        # Extent
        pad = 0.3
        if self._transform:
            self.ax.set_extent(
                [min(all_lons) - pad, max(all_lons) + pad,
                 min(all_lats) - pad, max(all_lats) + pad],
                crs=self._transform,
            )
        else:
            self.ax.set_xlim(min(all_lons) - pad, max(all_lons) + pad)
            self.ax.set_ylim(min(all_lats) - pad, max(all_lats) + pad)

        # Legend
        handles = [
            mpatches.Patch(color="orange",       label="Surface ~10 m"),
            mpatches.Patch(color="saddlebrown",  label="Bottom park"),
            mpatches.Patch(color="mediumpurple", label="Mid ~40 m"),
            mlines.Line2D([], [], color="steelblue", marker="o", markersize=7,
                          linestyle="None", label="Current position"),
            mlines.Line2D([], [], color="red", marker="*", markersize=10,
                          linestyle="None", label="Target"),
        ]
        self.ax.legend(handles=handles, loc="upper left", fontsize=8)

        depth_str = f"{action.target_depth:.0f} m" if action.target_depth is not None else "bottom"
        self.ax.set_title(
            f"MPCwithFavourable  |  Cycle {self._cycle}  |  {time.strftime('%Y-%m-%d %H:%M')}\n"
            f"({current.lat:.3f}°N, {current.lon:.3f}°E)  → {action.park_mode} {depth_str} {action.cycle_hours:.0f}h"
        )
        self.fig.canvas.draw()
        plt.pause(0.05)

    def guassian_distance_term(self, location: GeoLocation) -> float:
        distance = self.distance_from_center(location)
        return np.e**(-distance**2/(2*self.radius_std_dv**2))

    def get_action(self, profiler_state: ProfilerState, forecast: xr.Dataset, **kwargs) -> ControlAction:
        """Evaluate all candidate actions and return the one with the lowest cost.

        Parameters
        ----------
        profiler_state:
            Current float state (position and time).
        forecast:
            Forecast dataset used to simulate candidate dives and evaluate
            the favourable-flow term.
        """
        forecast_interpolators = build_velocity_interpolator(forecast)
        next_state = self.predict_next_surface_position(profiler_state, forecast, self.last_action)
        current_distance = self.distance_from_center(next_state.location)

        best_cost      = math.inf
        best_cfg       = self._action_configs[0]
        best_state     = None
        best_dist      = 0.0
        best_flow      = 0.0
        best_gaussian  = 0.0
        all_candidates: list[tuple[GeoLocation, float, str]] = []

        logger.debug(
            "Cycle %d | lat=%.4f lon=%.4f | evaluating %d candidate actions",
            self._cycle,
            profiler_state.location.lat,
            profiler_state.location.lon,
            len(self._action_configs),
        )
        logger.debug(
            "  %-14s %-7s %5s | %+10s | %+10s | %+10s (%4s) | %s",
            "park_mode", "depth", "hours",
            "dist_term", "flow_term", "gaussian", "sci",
            "total_cost",
        )

        for cfg in self._action_configs:
            action         = self._build_action(cfg)
            predicted      = self.predict_next_surface_position(next_state, forecast, action)
            distance_term  = self.distance_term(predicted.location, next_state.location, action)
            flow_term      = self.flow_term(forecast_interpolators, predicted)
            guassian_distance_term = self.guassian_distance_term(predicted.location)
            science_cost   = cfg["science_cost"]
            gaussian_term  = guassian_distance_term * science_cost
            cost           = (self.distance_weight * distance_term
                              + self.flow_weight * flow_term
                              + self.measurement_weight * gaussian_term)

            depth_str = f"{cfg['target_depth']:.0f} m" if cfg["target_depth"] is not None else "bottom"
            logger.debug(
                "  %-14s %-7s %5.0fh | %+10.4f | %+10.4f | %+10.4f (%4.2f) | %+.4f",
                cfg["park_mode"], depth_str, cfg["cycle_hours"],
                distance_term, flow_term,
                guassian_distance_term, science_cost,
                cost,
            )

            all_candidates.append((predicted.location, cost, self._action_color(cfg)))

            if cost < best_cost:
                best_cost     = cost
                best_cfg      = cfg
                best_state    = predicted
                best_dist     = distance_term
                best_flow     = flow_term
                best_gaussian = guassian_distance_term

        chosen_action = self._build_action(best_cfg)
        self.last_action = chosen_action
        self._cycle += 1

        depth_str = f"{best_cfg['target_depth']:.0f} m" if best_cfg["target_depth"] is not None else "bottom"
        logger.info(
            "Cycle %d | CHOSEN: %s %s %gh | "
            "dist_term=%+.4f | flow_term=%+.4f | gaussian=%.4f (sci=%.2f, term=%+.4f) | "
            "total_cost=%+.4f",
            self._cycle,
            best_cfg["park_mode"], depth_str, best_cfg["cycle_hours"],
            self.distance_weight * best_dist,
            self.flow_weight * best_flow,
            best_gaussian, best_cfg["science_cost"],
            -self.measurement_weight * best_gaussian * best_cfg["science_cost"],
            best_cost,
        )

        if self.debug:
            self.update_debug_plot(
                profiler_state.location,
                best_state.location,
                all_candidates,
                self._action_color(best_cfg),
                chosen_action,
                profiler_state.time,
            )

        return chosen_action

    def get_log(self) -> dict:
        return {
            "control_strategy": self.name,
            "target_location": [self.target_location.lat, self.target_location.lon],
            "distance_weight": self.distance_weight,
            "flow_weight": self.flow_weight,
            "measurement_weight": self.measurement_weight,
            "radius_std_dv": self.radius_std_dv,
            "time_horizon_final_state_hours": self.time_horizon_final_state_hours,
            "n_candidate_actions": len(self._action_configs),
            "ascent_speed": self.measurement_action.ascent_speed_ms,
            "descent_speed": self.measurement_action.descent_speed_ms,
        }












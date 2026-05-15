"""MPC control strategy using EKF state estimates."""
from __future__ import annotations

import math
import logging
from datetime import timedelta
from typing import Callable

import numpy as np

from sim_types import ControlAction, EstimatedState, GeoLocation
from particle_mover import simulate_estimate_forward, xy_to_latlon

logger = logging.getLogger(__name__)


class KFMPC:

    def __init__(
        self,
        target_location: list[float],  # [lat, lon]
        flow_weight: float = -0.8,
        distance_weight: float = 1.0,
        science_weight: float = 1.0,
        variance_weight: float = 1.0,
        radius_std_m: float = 5000.0,
        time_horizon_hours: int = 6,
    ) -> None:
        self.target = GeoLocation(lat=target_location[0], lon=target_location[1])
        self.flow_weight = flow_weight
        self.distance_weight = distance_weight
        self.science_weight = science_weight
        self.variance_weight = variance_weight
        self.radius_std_m = radius_std_m
        self.time_horizon_hours = time_horizon_hours

        self.possible_actions: list[ControlAction] = [
            ControlAction(duration_hours=12,  parking_depth=10,  science_cost=1.0),
            ControlAction(duration_hours=12,  parking_depth=30,  science_cost=0.8),
            ControlAction(duration_hours=48,  parking_depth=500, science_cost=0.5),
            ControlAction(duration_hours=120, parking_depth=500, science_cost=0.0),
        ]

    def _target_xy(self, start_lat: float, start_lon: float) -> tuple[float, float]:
        target_x = (self.target.lon - start_lon) * 111_000.0 * math.cos(math.radians(start_lat))
        target_y = (self.target.lat - start_lat) * 111_000.0
        return target_x, target_y

    def _flow_term(self, state: EstimatedState, interp_u: Callable, interp_v: Callable,
                   start_lat: float, start_lon: float) -> float:
        from particle_mover import _query_uv
        target_x, target_y = self._target_xy(start_lat, start_lon)
        to_target = np.array([target_x - state.x, target_y - state.y])
        norm = np.linalg.norm(to_target)
        if norm < 1.0:
            return 0.0
        n = to_target / norm

        ux, uy = 0.0, 0.0
        for h in range(self.time_horizon_hours):
            t = state.time + timedelta(hours=h)
            u, v = _query_uv(state.x, state.y, 0.0, t, interp_u, interp_v, start_lat, start_lon)
            ux += u + state.bx
            uy += v + state.by
        return float(np.dot(np.array([ux, uy]), n))/(self.time_horizon_hours)

    def _distance_term(self, current_state, previous_state, action, start_lat, start_lon):
        target_x, target_y = self._target_xy(start_lat, start_lon)
        d_current = math.sqrt((current_state.x - target_x) ** 2 + (current_state.y - target_y) ** 2)
        d_prev = math.sqrt((previous_state.x - target_x) ** 2 + (previous_state.y - target_y) ** 2)
        return (d_current - d_prev) / (action.duration_hours*3600)

    def _science_term(self, state: EstimatedState, action: ControlAction,
                      start_lat: float, start_lon: float) -> float:
        target_x, target_y = self._target_xy(start_lat, start_lon)
        dist = math.sqrt((state.x - target_x) ** 2 + (state.y - target_y) ** 2)
        proximity = math.exp(-dist ** 2 / (2 * self.radius_std_m ** 2))
        return proximity * action.science_cost

    def _variance_term(self, state: EstimatedState) -> float:
        return math.sqrt(float(np.trace(state.P[:2, :2])))

    def evaluate_cost(
        self,
        final_state: EstimatedState,
        prev_real_state: EstimatedState,
        action: ControlAction,
        interp_u: Callable,
        interp_v: Callable,
        start_lat: float,
        start_lon: float,
    ) -> tuple[float, float, float, float, float]:
        """Returns (total, flow_term, distance_term, science_term, variance_term)."""
        flow = self._flow_term(final_state, interp_u, interp_v, start_lat, start_lon)
        distance = self._distance_term(current_state=final_state, previous_state=prev_real_state, action=action, start_lat=start_lat, start_lon=start_lon)
        science = self._science_term(final_state, action, start_lat, start_lon)
        variance = self._variance_term(final_state)

        total = (
            - self.flow_weight * flow
            + self.distance_weight * distance
            + self.science_weight * science
            + self.variance_weight * variance
        )
        print(final_state.P)
        return total, flow, distance, science, variance

    def export_parameters(self) -> dict:
        return {
            "flow_weight": self.flow_weight,
            "distance_weight": self.distance_weight,
            "science_weight": self.science_weight,
            "variance_weight": self.variance_weight,
            "radius_std_m": self.radius_std_m,
            "time_horizon_hours": self.time_horizon_hours,
            "target_lat": self.target.lat,
            "target_lon": self.target.lon,
        }

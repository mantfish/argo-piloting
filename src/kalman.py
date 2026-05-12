"""Extended Kalman Filter math for the profiling float simulator.

Pure-math module: no I/O, no simulation state, no xarray. All functions are
stateless — they accept arrays and return arrays. State management lives in
particle_mover and main.

Dependency DAG: sim_types ← kalman (no import from particle_mover or control).
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from sim_types import ProfilerState, StateVector


def estimate_forecast_gradient(
    interp_u,
    interp_v,
    lat: float,
    lon: float,
    t: datetime,
    depth: float,
    delta_deg: float = 0.01,
) -> np.ndarray:
    """Numerically estimate the 2×2 velocity Jacobian via finite differences.

    Returns [[∂u/∂x, ∂u/∂y],
             [∂v/∂x, ∂v/∂y]]

    where x is eastward and y is northward, both in metres. Units are (m/s)/m = 1/s.

    NaN velocities at any sample point are treated as 0 so that the Jacobian
    degrades gracefully near data boundaries rather than propagating NaNs into P.

    Parameters
    ----------
    interp_u, interp_v:
        Velocity interpolators from build_velocity_interpolator().
        Each callable takes [[t_s, depth_m, lat, lon]] and returns an array.
    lat, lon:
        Current position in decimal degrees.
    t:
        Current simulation time.
    depth:
        Depth in metres (positive down).
    delta_deg:
        Finite-difference step in degrees. Default 0.01° ≈ 1.1 km.
    """
    t_s = np.datetime64(t, "s").astype(np.float64)

    def _query(la: float, lo: float) -> tuple[float, float]:
        u = float(interp_u([[t_s, depth, la, lo]])[0])
        v = float(interp_v([[t_s, depth, la, lo]])[0])
        return (0.0 if math.isnan(u) else u), (0.0 if math.isnan(v) else v)

    dx_m = delta_deg * 111320.0 * math.cos(math.radians(lat))
    dy_m = delta_deg * 111320.0

    u0, v0 = _query(lat, lon)
    u_px, v_px = _query(lat, lon + delta_deg)   # eastward perturbation
    u_py, v_py = _query(lat + delta_deg, lon)   # northward perturbation

    return np.array([
        [(u_px - u0) / dx_m, (u_py - u0) / dy_m],
        [(v_px - v0) / dx_m, (v_py - v0) / dy_m],
    ])


def build_F(dt_hours: float, vel_jacobian: np.ndarray) -> np.ndarray:
    """Construct the 4×4 state-transition Jacobian for one timestep.

    State vector: [x, y, bx, by]
      x, y  — position in metres
      bx, by — bias in m/hr (random walk)

    Linearised transition:
        x_new  = x + (u(x,y) + bx) * dt
        y_new  = y + (v(x,y) + by) * dt
        bx_new = bx
        by_new = by

    Parameters
    ----------
    dt_hours:
        Timestep in hours.
    vel_jacobian:
        2×2 array [[∂u/∂x, ∂u/∂y], [∂v/∂x, ∂v/∂y]] in units of 1/s.
        Pre-multiply by 3600 to convert to 1/hr before building F — done
        internally here.
    """
    J = vel_jacobian * 3600.0  # (1/s) → (1/hr) to match dt_hours units
    F = np.eye(4)
    F[0, 0] = 1.0 + J[0, 0] * dt_hours
    F[0, 1] =       J[0, 1] * dt_hours
    F[1, 0] =       J[1, 0] * dt_hours
    F[1, 1] = 1.0 + J[1, 1] * dt_hours
    F[0, 2] = dt_hours   # bias drives position: x += bx * dt
    F[1, 3] = dt_hours   # y += by * dt
    return F


def propagate_P(P: np.ndarray, F: np.ndarray, Q: np.ndarray, dt_hours: float) -> np.ndarray:
    """EKF prediction: propagate covariance one timestep.

    P_new = F @ P @ F.T + Q * dt_hours

    Parameters
    ----------
    P : (4, 4) current covariance
    F : (4, 4) state-transition Jacobian from build_F()
    Q : (4, 4) process noise covariance per unit time (per hour)
    dt_hours : timestep in hours
    """
    return F @ P @ F.T + Q * dt_hours


def gps_update_P(P: np.ndarray) -> np.ndarray:
    """GPS measurement update for covariance when R = 0 (perfect GPS).

    With R = 0 the Kalman gain collapses to identity for the observed
    (position) states, making the posterior position covariance exactly 0.
    Position rows and columns of P are zeroed; the bias-bias block P[2:,2:]
    is left unchanged because bias is not directly observed.
    """
    P_new = P.copy()
    P_new[:2, :] = 0.0
    P_new[:, :2] = 0.0
    return P_new


def gps_update_X(X: StateVector, gps_x: float, gps_y: float) -> StateVector:
    """GPS measurement update for state when R = 0.

    Position is set directly from the GPS fix; bias estimate is unchanged.

    Parameters
    ----------
    X : current EKF state vector
    gps_x : GPS eastward displacement in metres
    gps_y : GPS northward displacement in metres
    """
    return StateVector(x=gps_x, y=gps_y, bx=X.bx, by=X.by)

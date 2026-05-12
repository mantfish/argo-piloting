"""Profiling float simulator — entry point.

Wires all modules together into a complete simulation run. Reading this
file top-to-bottom gives a full picture of the system:

  1. Load lazy ocean + bathymetry datasets from NetCDF tiles.
  2. Loop over dive cycles until the configured end time:
       a. Integrate the float through one descent → park → ascent,
          applying known bias and process noise to the real state.
       b. Apply GPS update to the EKF estimated state (R=0).
       c. Get a (possibly noisy) forecast window at the surface.
       d. Ask the control module for the next ControlAction.
  3. Collect all TrajectoryRecords into a DataFrame, save to parquet,
     write a JSON config snapshot, and produce a trajectory plot.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from control import MPCwithFavourable, MPCwithFavourableMeasurement, MPCwithKalman
from data_loader import build_bathymetry_interpolator, get_forecast_field, load_bathymetry, load_manifest, select_tiles
from kalman import gps_update_P, gps_update_X
from noise import Q_DEFAULT, bias
from particle_mover import run_until_next_action
from plotter import plot_ekf_statistics, plot_trajectory
from sim_types import ControlAction, EKFRecord, GeoLocation, ProfilerState, RealProfilerState, SimConfig, StateVector, TrajectoryRecord

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------

def run_simulation(config: SimConfig) -> pd.DataFrame:
    """Execute a full profiler simulation and return the trajectory as a DataFrame."""

    # ------------------------------------------------------------------
    # Step 1 — Load data
    # ------------------------------------------------------------------
    manifest = load_manifest(config.data_dir)
    bathy_ds = load_bathymetry(config.data_dir / "D6_2024.nc")
    bathy_interp = build_bathymetry_interpolator(bathy_ds)
    logger.info("Manifest loaded (%d tiles) and bathymetry loaded.", len(manifest))

    # ------------------------------------------------------------------
    # Step 2 — Initialise
    # ------------------------------------------------------------------
    state: RealProfilerState = config.start_state
    est_state: ProfilerState = config.est_state
    all_records: list[TrajectoryRecord] = []
    ekf_records: list[EKFRecord] = []
    cycle_number = 0
    control = config.control_strategy
    loaded_action = control.default_action

    # ------------------------------------------------------------------
    # Step 3 — Main simulation loop
    # ------------------------------------------------------------------
    while state.time < config.end_time:

        if state.time >= config.end_time:
            break

        # Get forecast window for the control decision.
        forecast_end = state.time + timedelta(hours=200)
        if 200 < control.default_action.cycle_hours:
            raise ValueError("Forecast window is too short for the configured control strategy.")
        forecast_ds = select_tiles(manifest, config.data_dir, state.location.lat, state.location.lon, state.time, forecast_end)

        forecast = get_forecast_field(
            forecast_ds,
            lat=state.location.lat,
            lon=state.location.lon,
            time=state.time,
            hours_ahead=config.forecast_horizon_hours,
            noise_std=config.forecast_noise_std,
            seed=config.forecast_noise_seed,
        )

        # Ask the control module what to do next.
        next_action = control.get_action(
            profiler_state=state,
            forecast=forecast,
            est_state=est_state,
        )

        logger.info(
            "Cycle %d | %s | lat=%.4f lon=%.4f depth=%.1fm | trace(P[:2,:2])=%.1f m²",
            cycle_number,
            state.time.strftime("%Y-%m-%d %H:%M"),
            state.location.lat,
            state.location.lon,
            state.depth,
            float(np.trace(est_state.P[:2, :2])),
        )

        # Run particle simulation for one cycle (real state + EKF propagation).
        try:
            records, state, est_state = run_until_next_action(
                state,
                config.data_dir,
                manifest,
                bathy_interp,
                loaded_action,
                bias_fn=config.bias_fn,
                Q=config.Q,
                est_state=est_state,
                use_rk4=config.use_rk4,
            )
        except RuntimeError as e:
            logger.warning("Stopping simulation early: %s", e)
            break

        all_records.extend(records)
        logger.info(
            "  Surfaced at %s | lat=%.4f lon=%.4f",
            state.time.strftime("%Y-%m-%d %H:%M"),
            state.location.lat,
            state.location.lon,
        )

        # GPS update at surface: R=0 — set estimated position exactly from GPS.
        if est_state is not None:
            # Record innovation (before update) for filter diagnostics.
            ekf_records.append(EKFRecord(
                time=state.time,
                cycle=cycle_number,
                innovation_x=state.x - est_state.X.x,
                innovation_y=state.y - est_state.X.y,
                P_xx=float(est_state.P[0, 0]),
                P_yy=float(est_state.P[1, 1]),
            ))
            updated_X = gps_update_X(est_state.X, state.x, state.y)
            updated_P = gps_update_P(est_state.P)
            est_state = ProfilerState(
                time=state.time,
                lat=state.location.lat,
                lon=state.location.lon,
                X=updated_X,
                P=updated_P,
            )

        # Reset real state phase to communicating for next cycle.
        state = RealProfilerState(
            time=state.time,
            location=state.location,
            depth=0.0,
            phase="communicating",
            x=state.x,
            y=state.y,
            bathymetry_depth=state.bathymetry_depth,
        )

        loaded_action = next_action
        cycle_number += 1

    # ------------------------------------------------------------------
    # Step 4 — Build DataFrame
    # ------------------------------------------------------------------
    df = pd.DataFrame([vars(r) for r in all_records])
    logger.info(
        "Simulation complete. %d records across %d cycles.",
        len(df),
        cycle_number,
    )

    # ------------------------------------------------------------------
    # Step 5 — Save outputs
    # ------------------------------------------------------------------
    run_dir = config.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = run_dir / f"{config.control_strategy.name}_trajectory.parquet"
    df.to_parquet(parquet_path)
    logger.info("Trajectory saved to %s", parquet_path)

    meta_path = run_dir / f"{config.control_strategy.name}_config.json"
    meta = {
        "forecast_noise_std": config.forecast_noise_std,
        "forecast_noise_seed": config.forecast_noise_seed,
        "start_lat": config.start_state.location.lat,
        "start_lon": config.start_state.location.lon,
        "start_time": config.start_state.time.isoformat(),
        "end_time": config.end_time.isoformat(),
    }
    meta = meta | control.get_log()
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Config saved to %s", meta_path)

    # ------------------------------------------------------------------
    # Step 6 — Plot
    # ------------------------------------------------------------------
    plot_trajectory(
        df=df,
        config=config,
        save_path=run_dir / f"{config.control_strategy.name}_trajectory.png",
        show=False,
        bathy_ds=bathy_ds,
    )
    logger.info("Plot saved.")

    if ekf_records:
        plot_ekf_statistics(
            ekf_records=ekf_records,
            Q=config.Q,
            save_path=run_dir / f"{config.control_strategy.name}_ekf_statistics.png",
            show=False,
        )
        logger.info("EKF statistics plot saved.")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    start_real = RealProfilerState(
        time=datetime(2023, 10, 2, 0, 0, 0),
        location=GeoLocation(lat=55.2, lon=15.5),
        depth=0.0,
        phase="communicating",
        x=0.0,
        y=0.0,
    )

    # EKF initial estimate: position known from GPS; bias initialised to true value.
    start_est = ProfilerState(
        time=datetime(2023, 10, 2, 0, 0, 0),
        lat=55.2,
        lon=15.5,
        X=StateVector(x=0.0, y=0.0, bx=0.1, by=0.1),
        P=np.diag([0.0, 0.0, 0.25, 0.25]),  # position known; bias std = 0.5 m/hr
    )

    standard_cycle = ControlAction(
        park_mode="park_on_bottom",
        target_depth=50.0,
        cycle_hours=120,
        transmission_duration_minutes=30,
        ascent_speed_ms=0.01,
        descent_speed_ms=0.01,
    )

    control_strategy = MPCwithKalman(
        default_action=standard_cycle,
        target_location=[55.2, 15.5],
        debug=True,
        uncertainty_weight=0.001,
    )

    config = SimConfig(
        start_state=start_real,
        est_state=start_est,
        end_time=datetime(2025, 1, 24, 0, 0, 0),
        control_strategy=control_strategy,
        Q=Q_DEFAULT,
        bias_fn=bias,
        forecast_noise_std=0.0,
        forecast_noise_seed=42,
        forecast_horizon_hours=200.0,
        data_dir=Path("data/raw"),
        output_dir=Path(f"data/processed/{control_strategy.name}"),
    )

    df = run_simulation(config)
    print(f"Done. {len(df)} trajectory records.")

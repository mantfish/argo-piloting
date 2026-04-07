"""Profiling float simulator — entry point.

Wires all modules together into a complete simulation run. Reading this
file top-to-bottom gives a full picture of the system:

  1. Load lazy ocean + bathymetry datasets from NetCDF tiles.
  2. Loop over dive cycles until the configured end time:
       a. Integrate the float through one descent → park → ascent.
       b. Get a (possibly noisy) forecast window at the surface.
       c. Ask the control module for the next ControlAction.
       d. Advance the clock by the surface waiting time.
  3. Collect all TrajectoryRecords into a DataFrame, save to parquet,
     write a JSON config snapshot, and produce a trajectory plot.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from control import *
from data_loader import build_bathymetry_interpolator, get_forecast_field, load_bathymetry, load_manifest, select_tiles
from particle_mover import run_until_next_action
from plotter import plot_trajectory
from sim_types import ControlAction, ProfilerState, SimConfig, TrajectoryRecord

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
    state: ProfilerState = config.start_state
    all_records: list[TrajectoryRecord] = []
    cycle_number = 0
    control = config.control_strategy

    # ------------------------------------------------------------------
    # Step 3 — Main simulation loop
    # ------------------------------------------------------------------
    while state.time < config.end_time:

        # Check if simulation is complete.
        if state.time >= config.end_time:
            break

        # Get forecast window for the control decision.
        forecast_end = state.time + timedelta(hours=200)
        if 200 < control.cycle_hours:
            raise ValueError("Forecast window is too short for the configured control strategy.")
        forecast_ds = select_tiles(manifest, config.data_dir, state.lat, state.lon, state.time, forecast_end)

        forecast = get_forecast_field(
            forecast_ds,
            lat=state.lat,
            lon=state.lon,
            time=state.time,
            hours_ahead=config.forecast_horizon_hours,
            noise_std=config.forecast_noise_std,
            seed=config.forecast_noise_seed,
        )

        # Ask the control module what to do next.
        action = control.get_action(
            profiler_state=state,
            forecast=forecast,
        )

        logger.info(
            "Cycle %d | %s | lat=%.4f lon=%.4f depth=%.1fm",
            cycle_number,
            state.time.strftime("%Y-%m-%d %H:%M"),
            state.lat,
            state.lon,
            state.depth,
        )

        # Run particle simulation for one cycle.
        try:
            records, state = run_until_next_action(state, config.data_dir, manifest, bathy_interp, action, use_rk4=config.use_rk4)
        except RuntimeError as e:
            logger.warning("Stopping simulation early: %s", e)
            break

        all_records.extend(records)
        logger.info(
            "  Surfaced at %s | lat=%.4f lon=%.4f",
            state.time.strftime("%Y-%m-%d %H:%M"),
            state.lat,
            state.lon,
        )

        # Simulate surface waiting time before the next action.
        state = ProfilerState(
            time=state.time,
            lat=state.lat,
            lon=state.lon,
            depth=0.0,
            phase="communicating",
            x=state.x,
            y=state.y,
        )

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

    parquet_path = run_dir / f"{config.control_strategy}_trajectory.parquet"
    df.to_parquet(parquet_path)
    logger.info("Trajectory saved to %s", parquet_path)

    meta_path = run_dir / f"{config.control_strategy}_config.json"
    meta = {
        "forecast_noise_std": config.forecast_noise_std,
        "forecast_noise_seed": config.forecast_noise_seed,
        "start_lat": config.start_state.lat,
        "start_lon": config.start_state.lon,
        "start_time": config.start_state.time.isoformat(),
        "end_time": config.end_time.isoformat(),
    }
    meta = meta | control_strategy.get_log()
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Config saved to %s", meta_path)

    # ------------------------------------------------------------------
    # Step 6 — Plot
    # ------------------------------------------------------------------
    plot_trajectory(
        df=df,
        config=config,
        save_path=run_dir / f"{config.control_strategy}_trajectory.png",
        show=False,
        bathy_ds=bathy_ds,
    )
    logger.info("Plot saved.")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    start_state = ProfilerState(
        time=datetime(2025, 1, 15, 0, 0, 0),
        lat=55.2,
        lon=15.5,
        depth=0.0,
        phase="communicating",
    )

    #control_strategy = DriftTowardsPoint(target_location=[54.8,14.4], debug=True)
    control_strategy = NoControlParkOnBottom(cycle_hours=12, debug = True)

    config = SimConfig(
        start_state=start_state,
        end_time=datetime(2025, 1, 24, 0, 0, 0),
        control_strategy=control_strategy,
        forecast_noise_std=0.0,
        forecast_noise_seed=42,
        forecast_horizon_hours=120,
        data_dir=Path("data/test_data"),
        output_dir=Path("data/processed/test_data"),
    )

    df = run_simulation(config)
    print(f"Done. {len(df)} trajectory records.")

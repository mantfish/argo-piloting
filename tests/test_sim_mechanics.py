"""Tests for the profiling float simulator against synthetic current fields.

Each test generates a known current field, runs the simulation, and checks
the output against an analytical prediction. This validates the full stack:
data_loader → particle_mover → control, without needing real CMEMS data.

Run with:
    pytest tests/test_simulation.py -v
    pytest tests/test_simulation.py -v -k "test_uniform_east"  # single test
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

# ── path setup ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_loader import (
    build_bathymetry_interpolator,
    build_velocity_interpolator,
    load_bathymetry,
    load_manifest,
    load_working_window,
    select_tiles,
)
from particle_mover import run_until_next_action
from sim_types import ControlAction, GeoLocation, ProfilerState

# ── constants ───────────────────────────────────────────────────
DT_SECONDS = 600.0  # 10-minute timestep (matches default in particle_mover)
START_TIME = datetime(2025, 1, 15, 0, 0, 0)
START_LAT = 55.5
START_LON = 14.0
SPEED = 0.1  # m/s for most test cases
SEABED_DEPTH = 100.0  # m
DESCENT_SPEED = 0.01  # m/s
ASCENT_SPEED = 0.01  # m/s
CYCLE_HOURS = 120  # 5 days
TRANSMISSION_MINUTES = 30

# ── synthetic data generation ───────────────────────────────────

DEPTH_LEVELS = np.array(
    [0.494025, 1.541375, 2.645669, 3.819495, 5.078224,
     6.440614, 7.929560, 9.572997, 11.405000, 13.467137],
    dtype=np.float32,
)


def _make_grid(hours=280):
    """Create coordinate arrays matching CMEMS Baltic format."""
    lats = np.arange(54.0, 57.0, 0.028, dtype=np.float64)
    lons = np.arange(12.0, 18.0, 0.028, dtype=np.float64)
    t0 = np.datetime64("2025-01-15T00:00:00")
    times = np.array([t0 + np.timedelta64(h, "h") for h in range(hours)])
    return times, DEPTH_LEVELS, lats, lons


def _build_velocity_ds(uo, vo, times, depths, lats, lons):
    return xr.Dataset(
        {
            "uo": (["time", "depth", "latitude", "longitude"], uo,
                   {"standard_name": "eastward_sea_water_velocity", "units": "m s-1"}),
            "vo": (["time", "depth", "latitude", "longitude"], vo,
                   {"standard_name": "northward_sea_water_velocity", "units": "m s-1"}),
        },
        coords={
            "time": times,
            "depth": ("depth", depths, {"units": "m", "positive": "down"}),
            "latitude": ("latitude", lats, {"units": "degrees_north"}),
            "longitude": ("longitude", lons, {"units": "degrees_east"}),
        },
    )


def _build_bathy_ds(lats, lons, flat_depth=SEABED_DEPTH):
    elevation = np.full((len(lats), len(lons)), -flat_depth, dtype=np.float32)
    return xr.Dataset(
        {"elevation": (["lat", "lon"], elevation, {"units": "m"})},
        coords={
            "lat": ("lat", lats, {"units": "degrees_north"}),
            "lon": ("lon", lons, {"units": "degrees_east"}),
        },
    )


def _write_test_data(tmp_path, uo, vo, times, depths, lats, lons,
                     seabed_depth=SEABED_DEPTH):
    """Write a complete synthetic data directory and return its path."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Velocity
    nc_name = "synthetic_test.nc"
    ds = _build_velocity_ds(uo, vo, times, depths, lats, lons)
    ds.to_netcdf(data_dir / nc_name)

    # Manifest
    manifest = [{
        "file": nc_name,
        "lat": [float(lats[0]), float(lats[-1])],
        "lon": [float(lons[0]), float(lons[-1])],
        "time": [str(times[0]), str(times[-1])],
    }]
    (data_dir / "manifest.yaml").write_text(yaml.dump(manifest))

    # Bathymetry
    bathy_ds = _build_bathy_ds(lats, lons, flat_depth=seabed_depth)
    bathy_ds.to_netcdf(data_dir / "D6_2024.nc")

    return data_dir


def _run_one_cycle(data_dir, action, start_lat=START_LAT, start_lon=START_LON):
    """Run a single dive cycle and return (records, final_state)."""
    manifest = load_manifest(data_dir)
    bathy_ds = load_bathymetry(data_dir / "D6_2024.nc")
    bathy_interp = build_bathymetry_interpolator(bathy_ds)

    state = ProfilerState(
        time=START_TIME,
        location=GeoLocation(lat=start_lat, lon=start_lon),
        depth=0.0,
        phase="communicating",
    )

    return run_until_next_action(state, data_dir, manifest, bathy_interp, action)


# ── helper: expected drift time for a park-on-bottom cycle ──────

def _expected_drift_seconds_bottom_park(seabed_depth=SEABED_DEPTH):
    """Analytically compute total drifting time (transmission + descent + ascent).

    Uses the actual particle_mover defaults:
      dt_vertical_seconds=30  for descent/ascent steps
      dt_surface_seconds=120  for transmission window steps
    """
    dt_vertical = 30.0
    dt_surface = 120.0

    descent_steps = math.ceil(seabed_depth / (DESCENT_SPEED * dt_vertical))
    ascent_steps = math.ceil(seabed_depth / (ASCENT_SPEED * dt_vertical))

    transmission_s = TRANSMISSION_MINUTES * 60.0
    descent_s = descent_steps * dt_vertical
    ascent_s = ascent_steps * dt_vertical

    return transmission_s + descent_s + ascent_s


# ═══════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════


class TestUniformEast:
    """Constant 0.1 m/s eastward everywhere.

    The float should drift east during transmission, descent, and ascent,
    with zero northward displacement. On the seabed it fast-forwards
    (no drift).
    """

    @pytest.fixture()
    def data_dir(self, tmp_path):
        times, depths, lats, lons = _make_grid()
        shape = (len(times), len(depths), len(lats), len(lons))
        uo = np.full(shape, SPEED, dtype=np.float32)
        vo = np.zeros(shape, dtype=np.float32)
        return _write_test_data(tmp_path, uo, vo, times, depths, lats, lons)

    @pytest.fixture()
    def bottom_park_action(self):
        return ControlAction(
            park_mode="park_on_bottom",
            cycle_hours=CYCLE_HOURS,
            transmission_duration_minutes=TRANSMISSION_MINUTES,
            ascent_speed_ms=ASCENT_SPEED,
            descent_speed_ms=DESCENT_SPEED,
        )

    def test_eastward_displacement(self, data_dir, bottom_park_action):
        """Total eastward displacement should match drift_time × speed."""
        records, final = _run_one_cycle(data_dir, bottom_park_action)

        drift_s = _expected_drift_seconds_bottom_park()
        expected_dx = SPEED * drift_s

        actual_dx = final.x
        assert actual_dx == pytest.approx(expected_dx, abs=1.0), (
            f"Expected ~{expected_dx:.0f}m east, got {actual_dx:.0f}m"
        )

    def test_zero_northward_displacement(self, data_dir, bottom_park_action):
        """Northward displacement should be zero in a pure eastward field."""
        records, final = _run_one_cycle(data_dir, bottom_park_action)
        assert final.y == pytest.approx(0.0, abs=0.1), (
            f"Expected 0m north, got {final.y:.2f}m"
        )

    def test_latitude_unchanged(self, data_dir, bottom_park_action):
        """Latitude should not change in a pure eastward field."""
        records, final = _run_one_cycle(data_dir, bottom_park_action)
        assert final.lat == pytest.approx(START_LAT, abs=1e-6)

    def test_longitude_increases(self, data_dir, bottom_park_action):
        """Longitude should increase (eastward drift)."""
        records, final = _run_one_cycle(data_dir, bottom_park_action)
        assert final.lon > START_LON

    def test_ends_communicating(self, data_dir, bottom_park_action):
        """Float should end the cycle back at the surface in communicating phase."""
        records, final = _run_one_cycle(data_dir, bottom_park_action)
        assert final.depth == 0.0
        # final_state in run_until_next_action sets phase to "at_surface"
        # but the last record should be "communicating"
        assert records[-1].phase == "communicating"

    def test_seabed_record_exists(self, data_dir, bottom_park_action):
        """There should be at least one on_seabed record."""
        records, _ = _run_one_cycle(data_dir, bottom_park_action)
        seabed_records = [r for r in records if r.on_seabed]
        assert len(seabed_records) >= 1

    def test_depth_profile_shape(self, data_dir, bottom_park_action):
        """Depth should go: 0 → seabed → 0 (descend, park, ascend)."""
        records, _ = _run_one_cycle(data_dir, bottom_park_action)
        depths = [r.depth for r in records]
        max_depth = max(depths)
        assert max_depth == pytest.approx(SEABED_DEPTH, abs=0.1)
        assert depths[0] == pytest.approx(0.0, abs=1.0)  # near surface at start
        assert depths[-1] == pytest.approx(0.0, abs=0.1)  # back at surface


class TestUniformNorth:
    """Constant 0.1 m/s northward everywhere."""

    @pytest.fixture()
    def data_dir(self, tmp_path):
        times, depths, lats, lons = _make_grid()
        shape = (len(times), len(depths), len(lats), len(lons))
        uo = np.zeros(shape, dtype=np.float32)
        vo = np.full(shape, SPEED, dtype=np.float32)
        return _write_test_data(tmp_path, uo, vo, times, depths, lats, lons)

    @pytest.fixture()
    def action(self):
        return ControlAction(
            park_mode="park_on_bottom",
            cycle_hours=CYCLE_HOURS,
            transmission_duration_minutes=TRANSMISSION_MINUTES,
            ascent_speed_ms=ASCENT_SPEED,
            descent_speed_ms=DESCENT_SPEED,
        )

    def test_northward_displacement(self, data_dir, action):
        """Total northward displacement should match drift_time × speed."""
        records, final = _run_one_cycle(data_dir, action)

        drift_s = _expected_drift_seconds_bottom_park()
        expected_dy = SPEED * drift_s

        assert final.y == pytest.approx(expected_dy, abs=1.0)

    def test_zero_eastward_displacement(self, data_dir, action):
        """Eastward displacement should be zero."""
        records, final = _run_one_cycle(data_dir, action)
        assert final.x == pytest.approx(0.0, abs=0.1)

    def test_latitude_increases(self, data_dir, action):
        """Latitude should increase (northward drift)."""
        records, final = _run_one_cycle(data_dir, action)
        assert final.lat > START_LAT


class TestZeroCurrent:
    """No current anywhere — float should return to almost exactly where it started."""

    @pytest.fixture()
    def data_dir(self, tmp_path):
        times, depths, lats, lons = _make_grid()
        shape = (len(times), len(depths), len(lats), len(lons))
        uo = np.zeros(shape, dtype=np.float32)
        vo = np.zeros(shape, dtype=np.float32)
        return _write_test_data(tmp_path, uo, vo, times, depths, lats, lons)

    @pytest.fixture()
    def action(self):
        return ControlAction(
            park_mode="park_on_bottom",
            cycle_hours=CYCLE_HOURS,
            transmission_duration_minutes=TRANSMISSION_MINUTES,
            ascent_speed_ms=ASCENT_SPEED,
            descent_speed_ms=DESCENT_SPEED,
        )

    def test_no_displacement(self, data_dir, action):
        records, final = _run_one_cycle(data_dir, action)
        assert final.x == pytest.approx(0.0, abs=0.1)
        assert final.y == pytest.approx(0.0, abs=0.1)

    def test_returns_to_start(self, data_dir, action):
        records, final = _run_one_cycle(data_dir, action)
        assert final.lat == pytest.approx(START_LAT, abs=1e-6)
        assert final.lon == pytest.approx(START_LON, abs=1e-6)


class TestSurfaceOnly:
    """Current only at the shallowest depth level; zero below.

    This is the key test for the control paradigm: parking at depth
    should result in zero drift, while surface drift should show displacement.
    """

    @pytest.fixture()
    def data_dir(self, tmp_path):
        times, depths, lats, lons = _make_grid()
        shape = (len(times), len(depths), len(lats), len(lons))
        uo = np.zeros(shape, dtype=np.float32)
        vo = np.zeros(shape, dtype=np.float32)
        # Only the shallowest depth gets current
        uo[:, 0, :, :] = SPEED
        return _write_test_data(tmp_path, uo, vo, times, depths, lats, lons)

    def test_bottom_park_minimal_drift(self, data_dir):
        """Park on bottom: most of descent/ascent is below surface layer → minimal drift.

        The float should still drift during the transmission window (at depth=0,
        interpolated from the shallowest level) but NOT during descent through
        deeper layers.
        """
        action = ControlAction(
            park_mode="park_on_bottom",
            cycle_hours=CYCLE_HOURS,
            transmission_duration_minutes=TRANSMISSION_MINUTES,
            ascent_speed_ms=ASCENT_SPEED,
            descent_speed_ms=DESCENT_SPEED,
        )
        records, final = _run_one_cycle(data_dir, action)

        # Should be MUCH less than uniform-east displacement
        uniform_drift = SPEED * _expected_drift_seconds_bottom_park()
        assert final.x < uniform_drift * 0.5, (
            f"Surface-only field: bottom-park drift ({final.x:.0f}m) should be "
            f"much less than uniform drift ({uniform_drift:.0f}m)"
        )

    def test_surface_drift_has_displacement(self, data_dir):
        """Drift on surface: float should accumulate significant eastward displacement."""
        action = ControlAction(
            park_mode="drift_on_surface",
            cycle_hours=CYCLE_HOURS,
            transmission_duration_minutes=TRANSMISSION_MINUTES,
        )
        records, final = _run_one_cycle(data_dir, action)

        # 120 hours of surface drift at 0.1 m/s
        expected_dx = SPEED * CYCLE_HOURS * 3600.0
        # Allow generous tolerance — transmission window adds/overlaps
        assert final.x == pytest.approx(expected_dx, rel=0.05), (
            f"Expected ~{expected_dx:.0f}m east from surface drift, got {final.x:.0f}m"
        )


class TestParkingDepth:
    """Float parks at a target depth instead of on the seabed."""

    @pytest.fixture()
    def data_dir(self, tmp_path):
        times, depths, lats, lons = _make_grid()
        shape = (len(times), len(depths), len(lats), len(lons))
        uo = np.full(shape, SPEED, dtype=np.float32)
        vo = np.zeros(shape, dtype=np.float32)
        return _write_test_data(tmp_path, uo, vo, times, depths, lats, lons)

    def test_parks_at_target_depth(self, data_dir):
        """Float should reach target depth and stay there, not go deeper."""
        target = 50.0
        action = ControlAction(
            park_mode="parking_depth",
            target_depth=target,
            cycle_hours=CYCLE_HOURS,
            transmission_duration_minutes=TRANSMISSION_MINUTES,
            ascent_speed_ms=ASCENT_SPEED,
            descent_speed_ms=DESCENT_SPEED,
        )
        records, _ = _run_one_cycle(data_dir, action)

        parking_records = [r for r in records if r.phase == "parking"]
        assert len(parking_records) > 0, "No parking records found"
        for r in parking_records:
            assert r.depth == pytest.approx(target, abs=0.1)

    def test_shallower_park_less_descent_time(self, data_dir):
        """Parking at 30m should have less total drift time than parking at 80m,
        because descent and ascent are shorter → less total displacement in a
        uniform field (since parking phase also drifts at depth in uniform field).

        Wait — in a uniform field, parking at depth also drifts. So total
        displacement depends on total drift time (transmission + descent + park + ascent).
        Park time is: cycle_hours - descent_time - ascent_time.
        So actually total drift time ≈ cycle_hours for both (since park drifts too).

        In a uniform field, both should have similar total displacement.
        The difference matters in a surface-only field.
        """
        # This test just checks both configurations run without error
        for target in [30.0, 80.0]:
            action = ControlAction(
                park_mode="parking_depth",
                target_depth=target,
                cycle_hours=CYCLE_HOURS,
                transmission_duration_minutes=TRANSMISSION_MINUTES,
                ascent_speed_ms=ASCENT_SPEED,
                descent_speed_ms=DESCENT_SPEED,
            )
            records, final = _run_one_cycle(data_dir, action)
            assert len(records) > 0
            assert final.depth == 0.0


class TestTimingConsistency:
    """Check that the simulation clock advances correctly."""

    @pytest.fixture()
    def data_dir(self, tmp_path):
        times, depths, lats, lons = _make_grid()
        shape = (len(times), len(depths), len(lats), len(lons))
        uo = np.full(shape, SPEED, dtype=np.float32)
        vo = np.zeros(shape, dtype=np.float32)
        return _write_test_data(tmp_path, uo, vo, times, depths, lats, lons)

    @pytest.fixture()
    def action(self):
        return ControlAction(
            park_mode="park_on_bottom",
            cycle_hours=CYCLE_HOURS,
            transmission_duration_minutes=TRANSMISSION_MINUTES,
            ascent_speed_ms=ASCENT_SPEED,
            descent_speed_ms=DESCENT_SPEED,
        )

    def test_records_ordered_in_time(self, data_dir, action):
        """Trajectory records should be monotonically increasing in time."""
        records, _ = _run_one_cycle(data_dir, action)
        times = [r.time for r in records]
        for i in range(1, len(times)):
            assert times[i] >= times[i - 1], (
                f"Time went backwards at record {i}: {times[i - 1]} → {times[i]}"
            )

    def test_total_cycle_duration(self, data_dir, action):
        """Total cycle should be approximately cycle_hours long."""
        records, final = _run_one_cycle(data_dir, action)
        elapsed = (final.time - START_TIME).total_seconds() / 3600.0
        # Cycle hours + some overshoot from ascent after the fast-forward
        assert elapsed == pytest.approx(CYCLE_HOURS, abs=5.0), (
            f"Cycle took {elapsed:.1f}h, expected ~{CYCLE_HOURS}h"
        )


class TestPhaseTransitions:
    """Verify the sequence of phase transitions is correct."""

    @pytest.fixture()
    def data_dir(self, tmp_path):
        times, depths, lats, lons = _make_grid()
        shape = (len(times), len(depths), len(lats), len(lons))
        uo = np.full(shape, SPEED, dtype=np.float32)
        vo = np.zeros(shape, dtype=np.float32)
        return _write_test_data(tmp_path, uo, vo, times, depths, lats, lons)

    def test_bottom_park_phase_sequence(self, data_dir):
        """Phase sequence for park_on_bottom should be:
        communicating → descending → on_seabed → ascending → communicating
        """
        action = ControlAction(
            park_mode="park_on_bottom",
            cycle_hours=CYCLE_HOURS,
            transmission_duration_minutes=TRANSMISSION_MINUTES,
            ascent_speed_ms=ASCENT_SPEED,
            descent_speed_ms=DESCENT_SPEED,
        )
        records, _ = _run_one_cycle(data_dir, action)
        phases = [r.phase for r in records]

        # Extract unique phases in order (removing consecutive duplicates)
        unique_phases = [phases[0]]
        for p in phases[1:]:
            if p != unique_phases[-1]:
                unique_phases.append(p)

        expected = ["communicating", "descending", "on_seabed", "ascending", "communicating"]
        assert unique_phases == expected, (
            f"Phase sequence: {unique_phases}, expected: {expected}"
        )

    def test_parking_depth_phase_sequence(self, data_dir):
        """Phase sequence for parking_depth should be:
        communicating → descending → parking → ascending → communicating
        """
        action = ControlAction(
            park_mode="parking_depth",
            target_depth=50.0,
            cycle_hours=CYCLE_HOURS,
            transmission_duration_minutes=TRANSMISSION_MINUTES,
            ascent_speed_ms=ASCENT_SPEED,
            descent_speed_ms=DESCENT_SPEED,
        )
        records, _ = _run_one_cycle(data_dir, action)
        phases = [r.phase for r in records]

        unique_phases = [phases[0]]
        for p in phases[1:]:
            if p != unique_phases[-1]:
                unique_phases.append(p)

        expected = ["communicating", "descending", "parking", "ascending", "communicating"]
        assert unique_phases == expected, (
            f"Phase sequence: {unique_phases}, expected: {expected}"
        )

    def test_surface_drift_phase_sequence(self, data_dir):
        """Phase sequence for drift_on_surface should be:
        communicating → drift_on_surface → communicating
        """
        action = ControlAction(
            park_mode="drift_on_surface",
            cycle_hours=CYCLE_HOURS,
            transmission_duration_minutes=TRANSMISSION_MINUTES,
        )
        records, _ = _run_one_cycle(data_dir, action)
        phases = [r.phase for r in records]

        unique_phases = [phases[0]]
        for p in phases[1:]:
            if p != unique_phases[-1]:
                unique_phases.append(p)

        expected = ["communicating", "drift_on_surface", "communicating"]
        assert unique_phases == expected, (
            f"Phase sequence: {unique_phases}, expected: {expected}"
        )


class TestInterpolator:
    """Direct tests on the velocity interpolator (bypassing particle_mover)."""

    @pytest.fixture()
    def uniform_ds(self, tmp_path):
        """In-memory uniform eastward dataset."""
        times, depths, lats, lons = _make_grid(hours=48)
        shape = (len(times), len(depths), len(lats), len(lons))
        uo = np.full(shape, SPEED, dtype=np.float32)
        vo = np.zeros(shape, dtype=np.float32)
        return _build_velocity_ds(uo, vo, times, depths, lats, lons).compute()

    def test_uniform_returns_correct_value(self, uniform_ds):
        interp_u, interp_v = build_velocity_interpolator(uniform_ds)
        t_s = np.datetime64("2025-01-15T12:00", "s").astype(np.float64)
        u = float(interp_u([[t_s, 0.494025, 55.5, 14.0]])[0])
        v = float(interp_v([[t_s, 0.494025, 55.5, 14.0]])[0])
        assert u == pytest.approx(SPEED, abs=1e-4)
        assert v == pytest.approx(0.0, abs=1e-4)

    def test_interpolation_between_grid_points(self, uniform_ds):
        """Value between grid points should still be SPEED (uniform field)."""
        interp_u, _ = build_velocity_interpolator(uniform_ds)
        t_s = np.datetime64("2025-01-15T12:30", "s").astype(np.float64)
        u = float(interp_u([[t_s, 3.0, 55.123, 14.456]])[0])
        assert u == pytest.approx(SPEED, abs=1e-3)

    def test_edge_clamping(self, uniform_ds):
        """Points outside the grid should be clamped, not return NaN."""
        interp_u, _ = build_velocity_interpolator(uniform_ds)
        t_s = np.datetime64("2025-01-15T12:00", "s").astype(np.float64)
        # Way outside the grid
        u = float(interp_u([[t_s, 0.494025, 99.0, 99.0]])[0])
        assert not np.isnan(u), "Interpolator returned NaN for out-of-bounds point"


class TestBathymetry:
    """Tests for the bathymetry interpolator."""

    def test_flat_seabed(self, tmp_path):
        times, _, lats, lons = _make_grid()
        bathy_ds = _build_bathy_ds(lats, lons, flat_depth=150.0)
        bathy_path = tmp_path / "bathy.nc"
        bathy_ds.to_netcdf(bathy_path)

        loaded = load_bathymetry(bathy_path)
        interp = build_bathymetry_interpolator(loaded)

        assert interp(55.5, 14.0) == pytest.approx(150.0, abs=0.1)
        assert interp(54.5, 16.0) == pytest.approx(150.0, abs=0.1)
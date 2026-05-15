"""Live debug plotter for the profiling float simulator."""
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from particle_mover import xy_to_latlon

if TYPE_CHECKING:
    from sim_types import ControlAction, EstimatedState, ProfilerState, SimConfig


class DebugPlotter:

    def __init__(self, config: SimConfig, start_lat: float, start_lon: float) -> None:
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.target_lat = config.control.target.lat
        self.target_lon = config.control.target.lon
        self.bias_truth_fn = config.bias_function

        # Per-cycle accumulators (one entry per GPS fix / cycle)
        self._cycles: list[int] = []
        self._P_diag_pre:  list[np.ndarray] = []  # diag(P) just BEFORE GPS fix (end of dive)
        self._P_diag_post: list[np.ndarray] = []  # diag(P) just AFTER  GPS fix
        self._bx_est_history: list[float] = []
        self._by_est_history: list[float] = []
        self._bx_truth_history: list[float] = []
        self._by_truth_history: list[float] = []
        self._pred_surf_lats: list[float] = []
        self._pred_surf_lons: list[float] = []
        self._real_surf_lats: list[float] = []
        self._real_surf_lons: list[float] = []
        self._surf_error_x: list[float] = []   # real - predicted east (m)
        self._surf_error_y: list[float] = []   # real - predicted north (m)

        plt.ion()
        self.fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        self.fig.suptitle("Float simulator — live debug", fontsize=12, fontweight="bold")

        self.ax_map, self.ax_cand, self.ax_cost = axes[0]
        self.ax_var, self.ax_bias, self.ax_surf_error = axes[1]

        self._setup_axes()
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.pause(0.01)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_axes(self) -> None:
        self.ax_map.set_title("Surfacings (map)")
        self.ax_map.set_xlabel("Longitude")
        self.ax_map.set_ylabel("Latitude")
        self.ax_map.plot(self.target_lon, self.target_lat, "r*", ms=14, label="target", zorder=5)
        self.ax_map.legend(loc="upper left", fontsize=7)

        self.ax_cand.set_title("Candidate actions (current cycle)")
        self.ax_cand.set_xlabel("Longitude")
        self.ax_cand.set_ylabel("Latitude")

        self.ax_cost.set_title("Action costs (current cycle)")
        self.ax_cost.set_xlabel("Action")
        self.ax_cost.set_ylabel("Cost (lower = better)")

        self.ax_var.set_title("Position variance trace(P_xx)")
        self.ax_var.set_xlabel("Cycle")
        self.ax_var.set_ylabel("Variance (m²)")

        self.ax_bias.set_title("Bias estimates vs truth")
        self.ax_bias.set_xlabel("Cycle")
        self.ax_bias.set_ylabel("Bias (m/s)")

        self.ax_surf_error.set_title("Surfacing prediction error")
        self.ax_surf_error.set_xlabel("Cycle")
        self.ax_surf_error.set_ylabel("Error (m)")
        self.ax_surf_error.axhline(0, color="black", lw=0.5, alpha=0.3)

    # ------------------------------------------------------------------
    # Public update — called once per cycle
    # ------------------------------------------------------------------

    def update(
        self,
        cycle: int,
        real_traj: list[ProfilerState],
        chosen_est_traj: list[EstimatedState],
        all_action_results: list[tuple[ControlAction, list[EstimatedState], float]],
        real_history: list[ProfilerState],
        estimated_history: list[EstimatedState],
        updated_est: EstimatedState,
    ) -> None:
        # Scalar accumulators
        self._cycles.append(cycle)
        self._P_diag_pre.append(np.diag(chosen_est_traj[-1].P).copy())
        self._P_diag_post.append(np.diag(updated_est.P).copy())
        self._bx_est_history.append(updated_est.bx)
        self._by_est_history.append(updated_est.by)
        truth = self.bias_truth_fn(updated_est.time)
        self._bx_truth_history.append(truth[0])
        self._by_truth_history.append(truth[1])

        # Surfacing positions: predicted vs actual
        pred_surf = chosen_est_traj[-1]
        real_surf = real_traj[-1]
        self._pred_surf_lats.append(pred_surf.location.lat)
        self._pred_surf_lons.append(pred_surf.location.lon)
        self._real_surf_lats.append(real_surf.location.lat)
        self._real_surf_lons.append(real_surf.location.lon)
        self._surf_error_x.append(real_surf.x - pred_surf.x)
        self._surf_error_y.append(real_surf.y - pred_surf.y)

        chosen_action = all_action_results[
            min(range(len(all_action_results)), key=lambda i: all_action_results[i][2])
        ][0]

        self._draw_map(updated_est)
        self._draw_candidates(all_action_results, chosen_action, updated_est)
        self._draw_costs(all_action_results, chosen_action)
        self._draw_variance()
        self._draw_bias()
        self._draw_surf_error()

        self.fig.canvas.draw_idle()
        plt.pause(0.01)

    # ------------------------------------------------------------------
    # Panel drawing helpers
    # ------------------------------------------------------------------

    def _draw_map(self, updated_est: EstimatedState) -> None:
        ax = self.ax_map
        ax.cla()
        ax.set_title("Surfacings (map)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Light gray line connecting real surfacings in chronological order
        if len(self._real_surf_lons) > 1:
            ax.plot(self._real_surf_lons, self._real_surf_lats,
                    color="#cccccc", lw=0.8, zorder=1)

        # Real surfacings: blue dots
        ax.scatter(self._real_surf_lons, self._real_surf_lats,
                   color="#4e9af1", s=50, zorder=3, label="real surface")

        # Predicted surfacings: orange x markers
        ax.scatter(self._pred_surf_lons, self._pred_surf_lats,
                   color="#f5a623", s=60, marker="x", linewidths=2,
                   zorder=4, label="predicted surface")

        # Target
        ax.plot(self.target_lon, self.target_lat, "r*", ms=14, label="target", zorder=5)

        # Uncertainty ellipse at current estimated position
        self._draw_ellipse(ax, updated_est)

        ax.legend(loc="upper left", fontsize=7)

        # Auto-scale to all points + target
        all_lons = self._real_surf_lons + self._pred_surf_lons + [self.target_lon]
        all_lats = self._real_surf_lats + self._pred_surf_lats + [self.target_lat]
        if all_lons:
            pad_lon = max((max(all_lons) - min(all_lons)) * 0.1, 0.05)
            pad_lat = max((max(all_lats) - min(all_lats)) * 0.1, 0.05)
            ax.set_xlim(min(all_lons) - pad_lon, max(all_lons) + pad_lon)
            ax.set_ylim(min(all_lats) - pad_lat, max(all_lats) + pad_lat)

    def _draw_ellipse(self, ax: plt.Axes, state: EstimatedState) -> None:
        P = state.P[:2, :2]
        lat, lon = state.location.lat, state.location.lon
        lat_rad = math.radians(lat)

        lon_std = math.sqrt(max(P[0, 0], 0.0)) / (111_000.0 * math.cos(lat_rad))
        lat_std = math.sqrt(max(P[1, 1], 0.0)) / 111_000.0

        ellipse = mpatches.Ellipse(
            (lon, lat), width=2 * lon_std, height=2 * lat_std,
            edgecolor="#f5a623", facecolor="#f5a623", alpha=0.2, zorder=4,
        )
        ax.add_patch(ellipse)
        ax.plot(lon, lat, "o", color="#f5a623", ms=5, zorder=5)

    def _draw_candidates(
        self,
        all_action_results: list[tuple[ControlAction, list[EstimatedState], float]],
        chosen_action: ControlAction,
        current_est: EstimatedState,
    ) -> None:
        ax = self.ax_cand
        ax.cla()
        ax.set_title("Candidate actions (current cycle)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        cur_lat, cur_lon = current_est.location.lat, current_est.location.lon

        for action, traj, cost in all_action_results:
            is_chosen = (action is chosen_action)
            lats = [s.location.lat for s in traj]
            lons = [s.location.lon for s in traj]
            colour = "#2ecc71" if is_chosen else "#aaaaaa"
            lw = 1.8 if is_chosen else 0.8
            alpha = 1.0 if is_chosen else 0.6
            ax.plot(lons, lats, color=colour, lw=lw, alpha=alpha, zorder=3 if is_chosen else 2)
            label = f"{action.parking_depth:.0f}m/{action.duration_hours:.0f}h\nc={cost:.1f}"
            ax.annotate(label, (lons[-1], lats[-1]), fontsize=6, color=colour,
                        textcoords="offset points", xytext=(3, 3))

        ax.plot(cur_lon, cur_lat, "ko", ms=6, zorder=5, label="current")
        ax.plot(self.target_lon, self.target_lat, "r*", ms=10, zorder=5, label="target")
        ax.legend(loc="upper left", fontsize=7)

        margin = 1.0
        ax.set_xlim(cur_lon - margin, cur_lon + margin)
        ax.set_ylim(cur_lat - margin, cur_lat + margin)

    def _draw_costs(
        self,
        all_action_results: list[tuple[ControlAction, list[EstimatedState], float]],
        chosen_action: ControlAction,
    ) -> None:
        ax = self.ax_cost
        ax.cla()
        ax.set_title("Action costs (current cycle)")
        ax.set_xlabel("Action")
        ax.set_ylabel("Cost (lower = better)")

        labels = [f"{a.parking_depth:.0f}m\n{a.duration_hours:.0f}h" for a, _, _ in all_action_results]
        costs = [c for _, _, c in all_action_results]
        colours = ["#2ecc71" if a is chosen_action else "#aaaaaa" for a, _, _ in all_action_results]

        bars = ax.bar(labels, costs, color=colours, edgecolor="black", linewidth=0.5)
        for bar, cost in zip(bars, costs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{cost:.2f}", ha="center", va="bottom", fontsize=7)

    def _draw_variance(self) -> None:
        ax = self.ax_var
        ax.cla()
        ax.set_title("P diagonal — pre/post GPS fix (log scale)")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Variance")

        if not self._cycles:
            return

        labels  = ["x pos (m²)", "y pos (m²)", "bx ((m/s)²)", "by ((m/s)²)"]
        colours = ["#4e9af1",     "#1a6dcc",     "#f5a623",      "#c47d00"]
        styles  = ["-",           "--",           "-",            "--"]

        pre  = np.array(self._P_diag_pre)   # (n, 4)
        post = np.array(self._P_diag_post)  # (n, 4)

        # Build interleaved x-axis: pre at c-0.35, post at c+0.35.
        # Connecting post(n) → pre(n+1) shows growth during the next dive;
        # connecting pre(n) → post(n) shows the collapse at the GPS fix.
        for j, (label, colour, style) in enumerate(zip(labels, colours, styles)):
            xs, ys = [], []
            for i, c in enumerate(self._cycles):
                xs.extend([c - 0.35, c + 0.35])
                ys.extend([pre[i, j], post[i, j]])
            ax.plot(xs, ys, color=colour, lw=1.4, linestyle=style,
                    marker="o", ms=4, label=label)

        ax.set_yscale("log")
        ax.set_xticks(self._cycles)

        # "pre" / "post" tick labels on a second x-axis so integers stay clean
        ax.set_xticklabels([str(c) for c in self._cycles], fontsize=7)

        # Vertical dashed lines between pre and post within each cycle
        for c in self._cycles:
            ax.axvline(c, color="gray", lw=0.4, linestyle=":", alpha=0.5)

        # Annotate first cycle so reader knows which is which
        if len(self._cycles) >= 1:
            ax.text(self._cycles[0], 1, "←pre  post→",
                    fontsize=6, color="gray", ha="center", va="bottom",
                    transform=ax.get_xaxis_transform())

        ax.legend(loc="upper right", fontsize=7)

    def _draw_bias(self) -> None:
        ax = self.ax_bias
        ax.cla()
        ax.set_title("Bias estimates vs truth")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Bias (m/s)")
        ax.plot(self._cycles, self._bx_est_history, color="#f5a623", lw=1.5, label="bx est")
        ax.plot(self._cycles, self._by_est_history, color="#2ecc71", lw=1.5, label="by est")
        ax.plot(self._cycles, self._bx_truth_history, color="#f5a623",
                lw=1.0, linestyle="--", label="bx truth", alpha=0.6)
        ax.plot(self._cycles, self._by_truth_history, color="#2ecc71",
                lw=1.0, linestyle="--", label="by truth", alpha=0.6)
        ax.axhline(0, color="black", lw=0.5, alpha=0.3)
        ax.legend(loc="upper left", fontsize=7)

    def _draw_surf_error(self) -> None:
        ax = self.ax_surf_error
        ax.cla()
        ax.set_title("Surfacing prediction error")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Error (m)")
        ax.axhline(0, color="black", lw=0.5, alpha=0.3)

        dist = [math.sqrt(dx ** 2 + dy ** 2)
                for dx, dy in zip(self._surf_error_x, self._surf_error_y)]

        ax.plot(self._cycles, self._surf_error_x, color="#4e9af1", lw=1.5, label="dx east")
        ax.plot(self._cycles, self._surf_error_y, color="#2ecc71", lw=1.5, label="dy north")
        ax.plot(self._cycles, dist, color="black", lw=1.5, linestyle="--", label="|error|")
        ax.legend(loc="upper left", fontsize=7)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        self.fig.savefig(path, dpi=150, bbox_inches="tight")

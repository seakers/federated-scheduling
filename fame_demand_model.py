"""
Dynamic acceptance-probability model for the FAME federated-scheduling simulator.

The model maintains a latent demand field D(constellation, lat, lon, t) that evolves
via Ornstein-Uhlenbeck mean-reversion + event-driven spikes.  D is mapped to an
acceptance probability through a sigmoid so the value stays in a configurable band.

Single source of truth: the same DemandField instance is used by both the simulator
(to draw Bernoulli outcomes) and the stochastic planner (to set MILP coefficients).

Two RNG streams are kept separate:
  _rng_demand  — drives the OU evolution of the demand field
  _rng_outcome — drives the individual accept/reject Bernoulli draws

This ensures that paired-seed comparisons between planners (greedy / det / stochastic)
remain valid: all three schedulers use the same _rng_outcome seed and therefore face
identical accept/reject outcomes.
"""

from __future__ import annotations

import datetime as dt
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RegionDef:
    """A rectangular geographic region with a baseline demand level."""
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    baseline_demand: float  # 0–1; higher → more competing demand → lower p_accept


@dataclass
class DemandFieldConfig:
    """All tunable parameters for the demand-field model.

    Acceptance probability band
    ---------------------------
    p_min / p_max: final probability is clamped to this range.
    At full demand (D→∞) probability saturates near p_min.
    At zero demand (D→−∞) probability saturates near p_max.

    Sigmoid scaling
    ---------------
    sigmoid_scale: controls how steeply demand maps to probability.
    A larger value compresses the response (less variation); smaller → sharper.

    OU dynamics
    -----------
    reversion_rate  : κ in dD = κ(μ−D)dt + σ dW.  Units: 1/second.
    noise_std       : σ (per sqrt-second, scaled internally to the timestep).

    Event spikes
    ------------
    spike_magnitude  : peak demand added by a single spike event.
    spike_half_life_s: exponential decay half-life in seconds.

    Constellation popularity
    ------------------------
    constellation_popularity: maps constellation name → float in (0, 1].
    Higher popularity → higher base demand → lower p_accept.
    Must create a cost–reliability tension: cheap-but-popular ≠ expensive-but-reliable.

    Precomputation
    --------------
    grid_lat_deg / grid_lon_deg : resolution of the demand grid in degrees.
    timestep_s : how many seconds between demand-field update steps.
    horizon_s  : total horizon to precompute (seconds).
    """
    # --- probability band ---
    p_min: float = 0.30
    p_max: float = 0.90

    # --- sigmoid ---
    sigmoid_scale: float = 1.5

    # --- OU dynamics ---
    reversion_rate: float = 1.0 / 3600.0   # mean-revert over ~1 hour
    noise_std: float = 0.10                 # per sqrt-second

    # --- event spikes ---
    spike_magnitude: float = 2.0
    spike_half_life_s: float = 3600.0 * 2  # 2-hour half-life

    # --- constellation popularity (must differ to create planner tension) ---
    constellation_popularity: dict = field(default_factory=lambda: {
        # cheap / popular (high demand → lower p_accept at baseline)
        "Planet":          0.85,
        "Ubotica":         0.75,
        # medium
        "LOFT":            0.55,
        "Mission Control": 0.50,
        "ICEYE":           0.45,
        "AC":              0.40,
        # expensive / reliable (low demand → higher p_accept at baseline)
        "Umbra":           0.30,
        "Capella":         0.20,
    })
    default_popularity: float = 0.50

    # --- regional baselines (0=quiet, 1=congested) ---
    regions: list = field(default_factory=lambda: [
        RegionDef("Middle East",   lat_min=15, lat_max=40,  lon_min=35,  lon_max=65,  baseline_demand=0.80),
        RegionDef("SE Asia",       lat_min=-10, lat_max=25, lon_min=95,  lon_max=140, baseline_demand=0.75),
        RegionDef("East Asia",     lat_min=20,  lat_max=50, lon_min=105, lon_max=145, baseline_demand=0.65),
        RegionDef("Europe",        lat_min=35,  lat_max=70, lon_min=-10, lon_max=40,  baseline_demand=0.60),
        RegionDef("N America",     lat_min=25,  lat_max=60, lon_min=-130, lon_max=-60, baseline_demand=0.55),
        RegionDef("Open Ocean",    lat_min=-60, lat_max=60, lon_min=-180, lon_max=180, baseline_demand=0.10),
    ])

    # --- grid & timing ---
    grid_lat_deg: float = 5.0
    grid_lon_deg: float = 5.0
    timestep_s: float = 900.0    # 15-minute steps

    # --- seeds ---
    demand_seed: int = 1001
    outcome_seed: int = 1002

    # --- legacy constant-probability fallback flag ---
    use_constant_probability: bool = False


# ---------------------------------------------------------------------------
# Demand field
# ---------------------------------------------------------------------------

class DemandField:
    """
    Precomputed time-varying demand field over (constellation, lat, lon, time).

    Usage
    -----
    1. Construct with a config, reference time, and horizon.
    2. Register event spikes via ``add_spike(lat, lon, t_event)``.
    3. Call ``precompute()`` to build the grid.
    4. Query acceptance probability via ``acceptance_probability(constellation, lat, lon, t)``.
    5. Sample an outcome via ``sample_accept(constellation, lat, lon, t)``.

    The planner adapters (``make_acceptance_prob_function``,
    ``make_simulator_acceptance_function``) return callables matching the
    existing FAME function signatures.
    """

    def __init__(
        self,
        config: DemandFieldConfig,
        reference_time: dt.datetime,
        horizon_s: float,
    ):
        self.config = config
        self.reference_time = reference_time
        self.horizon_s = horizon_s

        # Build spatial grid
        self._lats = np.arange(-90, 90 + 1e-9, config.grid_lat_deg)
        self._lons = np.arange(-180, 180 + 1e-9, config.grid_lon_deg)
        self._n_lat = len(self._lats)
        self._n_lon = len(self._lons)

        # Build time grid
        self._n_steps = max(1, int(math.ceil(horizon_s / config.timestep_s)) + 1)
        self._times_s = np.linspace(0.0, horizon_s, self._n_steps)

        # Pending event spikes: list of (lat, lon, t_s, applied_flag)
        self._spikes: list[tuple[float, float, float]] = []

        # Demand array: (n_constellations, n_lat, n_lon, n_steps)
        # Populated by precompute().
        self._demand: Optional[np.ndarray] = None
        self._constellations: list[str] = []
        self._const_idx: dict[str, int] = {}

        self._rng_demand = np.random.default_rng(config.demand_seed)
        self._rng_outcome = np.random.default_rng(config.outcome_seed)

        self._precomputed = False

    # ------------------------------------------------------------------
    # Event registration
    # ------------------------------------------------------------------

    def add_spike(self, lat: float, lon: float, t_event: dt.datetime) -> None:
        """Register a demand spike at (lat, lon) starting at t_event.

        Multiple calls accumulate (e.g. an eruption and a ship sighting).
        """
        t_s = (t_event - self.reference_time).total_seconds()
        self._spikes.append((lat, lon, t_s))
        self._precomputed = False  # force recompute

    # ------------------------------------------------------------------
    # Precompute
    # ------------------------------------------------------------------

    def precompute(self, constellations: list[str]) -> None:
        """Build the full demand trajectory for all constellations."""
        self._constellations = list(constellations)
        self._const_idx = {n: i for i, n in enumerate(self._constellations)}
        n_c = len(self._constellations)
        cfg = self.config

        # --- Baseline demand per (constellation, lat, lon) ---
        baseline = self._build_baseline(n_c)  # (n_c, n_lat, n_lon)

        # --- OU evolution ---
        demand = np.empty((n_c, self._n_lat, self._n_lon, self._n_steps))
        demand[..., 0] = baseline

        dt_s = cfg.timestep_s
        kappa = cfg.reversion_rate
        sigma = cfg.noise_std
        exp_k = math.exp(-kappa * dt_s)
        noise_scale = sigma * math.sqrt((1 - math.exp(-2 * kappa * dt_s)) / (2 * kappa)) if kappa > 0 else sigma * math.sqrt(dt_s)

        for ti in range(1, self._n_steps):
            noise = self._rng_demand.standard_normal((n_c, self._n_lat, self._n_lon)) * noise_scale
            demand[..., ti] = baseline + exp_k * (demand[..., ti - 1] - baseline) + noise

        # --- Event spikes ---
        for lat_s, lon_s, t_start_s in self._spikes:
            spike_array = self._spike_contribution(lat_s, lon_s, t_start_s)  # (n_lat, n_lon, n_steps)
            for ci in range(n_c):
                demand[ci] += cfg.spike_magnitude * spike_array

        self._demand = demand
        self._precomputed = True

    def _build_baseline(self, n_c: int) -> np.ndarray:
        """Return (n_c, n_lat, n_lon) array of baseline demand values."""
        cfg = self.config
        # Start from the global quiet ocean baseline
        geo_baseline = np.full((self._n_lat, self._n_lon), 0.10)

        # Paint regions in increasing-priority order (last wins per pixel)
        for region in cfg.regions:
            lat_mask = (self._lats >= region.lat_min) & (self._lats <= region.lat_max)
            lon_mask = (self._lons >= region.lon_min) & (self._lons <= region.lon_max)
            lat_idx = np.where(lat_mask)[0]
            lon_idx = np.where(lon_mask)[0]
            ii, jj = np.meshgrid(lat_idx, lon_idx, indexing='ij')
            geo_baseline[ii, jj] = region.baseline_demand

        # Scale per constellation by popularity
        baseline = np.empty((n_c, self._n_lat, self._n_lon))
        for ci, cname in enumerate(self._constellations):
            pop = cfg.constellation_popularity.get(cname, cfg.default_popularity)
            baseline[ci] = geo_baseline * pop

        return baseline

    def _spike_contribution(self, lat: float, lon: float, t_start_s: float) -> np.ndarray:
        """Return (n_lat, n_lon, n_steps) spike contribution (unnormalized; caller multiplies by magnitude)."""
        cfg = self.config
        decay_rate = math.log(2) / cfg.spike_half_life_s

        # Spatial Gaussian: ~500 km radius (5° at equator ≈ 555 km)
        sigma_deg = 5.0
        lat_diff = self._lats[:, None] - lat      # (n_lat, 1)
        lon_diff = self._lons[None, :] - lon       # (1, n_lon)
        dist_sq = lat_diff ** 2 + lon_diff ** 2
        spatial = np.exp(-dist_sq / (2 * sigma_deg ** 2))  # (n_lat, n_lon)

        # Temporal: exponential decay from t_start
        temporal = np.where(
            self._times_s >= t_start_s,
            np.exp(-decay_rate * (self._times_s - t_start_s)),
            0.0
        )  # (n_steps,)

        return spatial[:, :, None] * temporal[None, None, :]

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _demand_at(self, constellation: str, lat: float, lon: float, t: dt.datetime) -> float:
        """Interpolate the demand field at a given (constellation, lat, lon, t)."""
        if not self._precomputed:
            raise RuntimeError("Call precompute() before querying the demand field.")

        t_s = (t - self.reference_time).total_seconds()

        ci = self._const_idx.get(constellation)
        if ci is None:
            # Unknown constellation — use global average
            ci_arr = np.mean(self._demand, axis=0)  # (n_lat, n_lon, n_steps)
        else:
            ci_arr = self._demand[ci]  # (n_lat, n_lon, n_steps)

        # Clamp to grid extents
        lat_c = np.clip(lat, self._lats[0], self._lats[-1])
        lon_c = np.clip(lon, self._lons[0], self._lons[-1])
        t_s_c = np.clip(t_s, self._times_s[0], self._times_s[-1])

        # Nearest-neighbour in lat/lon; linear interpolation in time
        i_lat = int(np.argmin(np.abs(self._lats - lat_c)))
        i_lon = int(np.argmin(np.abs(self._lons - lon_c)))

        # Linear time interpolation
        ti_lo = int(np.searchsorted(self._times_s, t_s_c, side='right')) - 1
        ti_lo = max(0, min(ti_lo, self._n_steps - 2))
        ti_hi = ti_lo + 1
        t_lo = self._times_s[ti_lo]
        t_hi = self._times_s[ti_hi]
        alpha = (t_s_c - t_lo) / (t_hi - t_lo) if t_hi > t_lo else 0.0

        d_lo = ci_arr[i_lat, i_lon, ti_lo]
        d_hi = ci_arr[i_lat, i_lon, ti_hi]
        return float(d_lo + alpha * (d_hi - d_lo))

    def acceptance_probability(self, constellation: str, lat: float, lon: float, t: dt.datetime) -> float:
        """Return the (deterministic) acceptance probability for (constellation, lat, lon, t).

        This is the value both the planner and the simulator sample against.
        Probability = p_max - (p_max - p_min) * sigmoid(demand / scale)
        """
        if self.config.use_constant_probability:
            return self._constant_fallback(constellation)

        d = self._demand_at(constellation, lat, lon, t)
        cfg = self.config
        # logistic sigmoid maps demand to [0, 1]
        sig = 1.0 / (1.0 + math.exp(-d / cfg.sigmoid_scale))
        prob = cfg.p_max - (cfg.p_max - cfg.p_min) * sig
        return float(np.clip(prob, cfg.p_min, cfg.p_max))

    def sample_accept(self, constellation: str, lat: float, lon: float, t: dt.datetime) -> bool:
        """Draw a Bernoulli outcome using the _rng_outcome stream."""
        p = self.acceptance_probability(constellation, lat, lon, t)
        return bool(self._rng_outcome.random() <= p)

    def _constant_fallback(self, constellation: str) -> float:
        """Legacy per-constellation constant for regression comparison."""
        _LEGACY = {
            "Planet": 0.40, "Umbra": 0.60, "Capella": 0.90,
            "LOFT": 0.71, "Ubotica": 0.50, "Mission Control": 0.64,
            "AC": 0.67, "ICEYE": 0.74,
        }
        return _LEGACY.get(constellation, 0.70)

    # ------------------------------------------------------------------
    # Cost–reliability tension check
    # ------------------------------------------------------------------

    def check_cost_reliability_tension(
        self,
        cost_map: dict[str, float],
        reference_lat: float = 0.0,
        reference_lon: float = 0.0,
        reference_time: Optional[dt.datetime] = None,
    ) -> None:
        """Warn loudly if cheapest constellation is also most accepted.

        cost_map: maps constellation name → booking cost (lower = cheaper).
        """
        if reference_time is None:
            reference_time = self.reference_time

        probs = {
            c: self.acceptance_probability(c, reference_lat, reference_lon, reference_time)
            for c in self._constellations
        }
        cheapest = min(cost_map, key=cost_map.get)
        most_accepted = max(probs, key=probs.get)

        if cheapest == most_accepted:
            warnings.warn(
                f"\n[DemandField] CONFIGURATION WARNING: The cheapest constellation "
                f"('{cheapest}', cost={cost_map[cheapest]:.3f}) is also the most accepted "
                f"(p={probs[cheapest]:.3f}).  This configuration CANNOT differentiate "
                f"the stochastic from the deterministic planner.  "
                f"Adjust constellation_popularity so that cheap providers have lower "
                f"acceptance probability than expensive ones.",
                stacklevel=2,
            )
            print(
                "\n  *** COST-RELIABILITY TENSION CHECK FAILED ***\n"
                f"  Cheapest: {cheapest} (cost={cost_map[cheapest]:.3f}, p_accept={probs[cheapest]:.3f})\n"
                f"  Most accepted: {most_accepted} (p_accept={probs[most_accepted]:.3f})\n"
                "  The planners will behave identically.  Fix constellation_popularity.\n"
            )
        else:
            print(
                f"\n  [DemandField] Cost-reliability tension OK: "
                f"cheapest='{cheapest}' (cost={cost_map[cheapest]:.3f}, p_accept={probs[cheapest]:.3f}), "
                f"most_accepted='{most_accepted}' (p_accept={probs[most_accepted]:.3f})."
            )

    # ------------------------------------------------------------------
    # Adapter factories
    # ------------------------------------------------------------------

    def make_acceptance_prob_function(self):
        """Return a callable matching the planner signature:
        f(constrained_request, satellite, obs_pass) → float
        """
        demand_field = self

        def _f(constrained_request, satellite, obs_pass):
            req = constrained_request.observation_request
            lat = req.lat_deg
            lon = req.lon_deg
            t = obs_pass.highest.time
            constellation = _constellation_name_from_satellite(satellite.name)
            return demand_field.acceptance_probability(constellation, lat, lon, t)

        return _f

    def make_simulator_acceptance_function(self):
        """Return a callable matching the simulator signature:
        f(request, satellite, world_time) → float (probability; caller draws the coin)

        Note: the simulator in fame_constellation_scheduler.py draws its own
        random.random() against the returned probability, so we return p here,
        not a boolean.  The outcome RNG stream in DemandField is therefore NOT
        used for the simulator path (stdlib random handles that draw), which is
        intentional: the simulator's accept/reject draw must remain tied to the
        shared stdlib random seed so all planners face identical outcomes.
        """
        demand_field = self

        def _f(request, satellite, world_time):
            lat = request.lat_deg
            lon = request.lon_deg
            t = world_time
            constellation = _constellation_name_from_satellite(satellite.name)
            return demand_field.acceptance_probability(constellation, lat, lon, t)

        return _f

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_heatmaps(
        self,
        results_path: str,
        timestep_indices: Optional[list[int]] = None,
        figsize_per_panel: tuple[float, float] = (5.0, 3.5),
        dpi: int = 120,
    ) -> list[str]:
        """Save per-constellation acceptance-probability heatmaps.

        Parameters
        ----------
        results_path : directory where PNGs are saved.
        timestep_indices : which time-steps to render (defaults to 5 evenly spaced).
        figsize_per_panel : (width, height) in inches per map panel.
        dpi : figure resolution.

        Returns
        -------
        List of saved file paths.
        """
        import os
        os.makedirs(results_path, exist_ok=True)

        if not self._precomputed:
            raise RuntimeError("Call precompute() before plotting.")

        n_c = len(self._constellations)
        if n_c == 0:
            return []

        if timestep_indices is None:
            n_show = min(5, self._n_steps)
            timestep_indices = list(np.linspace(0, self._n_steps - 1, n_show, dtype=int))

        n_t = len(timestep_indices)
        cfg = self.config
        saved_files = []

        # --- Multi-panel: (n_t rows) × (n_c cols) ---
        fig_w = figsize_per_panel[0] * n_c
        fig_h = figsize_per_panel[1] * n_t
        fig, axes = plt.subplots(n_t, n_c, figsize=(fig_w, fig_h), squeeze=False)

        for ti_idx, ti in enumerate(timestep_indices):
            t_s = self._times_s[ti]
            t_abs = self.reference_time + dt.timedelta(seconds=float(t_s))

            for ci, cname in enumerate(self._constellations):
                ax = axes[ti_idx, ci]

                # Convert demand → probability
                d_slice = self._demand[ci, :, :, ti]  # (n_lat, n_lon)
                sig = 1.0 / (1.0 + np.exp(-d_slice / cfg.sigmoid_scale))
                prob = cfg.p_max - (cfg.p_max - cfg.p_min) * sig
                prob = np.clip(prob, cfg.p_min, cfg.p_max)

                im = ax.imshow(
                    prob,
                    origin='lower',
                    extent=[-180, 180, -90, 90],
                    vmin=cfg.p_min, vmax=cfg.p_max,
                    cmap='RdYlGn', aspect='auto',
                    interpolation='nearest',
                )
                ax.set_title(f"{cname}\n{t_abs.strftime('%Y-%m-%d %H:%M')} UTC", fontsize=8)
                ax.set_xlabel("Longitude (°)", fontsize=7)
                ax.set_ylabel("Latitude (°)", fontsize=7)
                ax.tick_params(labelsize=6)

        fig.colorbar(
            plt.cm.ScalarMappable(
                cmap='RdYlGn',
                norm=plt.Normalize(vmin=cfg.p_min, vmax=cfg.p_max)
            ),
            ax=axes, label='Acceptance probability', shrink=0.6,
        )
        fig.suptitle("Dynamic Acceptance Probability — per constellation × time", fontsize=10)
        plt.tight_layout()

        out_path = os.path.join(results_path, "demand_heatmaps_all.png")
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        saved_files.append(out_path)

        # --- Animated / multi-panel per-constellation ---
        for ci, cname in enumerate(self._constellations):
            n_panels = len(timestep_indices)
            cols = min(n_panels, 5)
            rows = math.ceil(n_panels / cols)
            fig2, axes2 = plt.subplots(rows, cols, figsize=(figsize_per_panel[0] * cols, figsize_per_panel[1] * rows), squeeze=False)

            for panel_idx, ti in enumerate(timestep_indices):
                row, col = divmod(panel_idx, cols)
                ax = axes2[row, col]
                t_s = self._times_s[ti]
                t_abs = self.reference_time + dt.timedelta(seconds=float(t_s))

                d_slice = self._demand[ci, :, :, ti]
                sig = 1.0 / (1.0 + np.exp(-d_slice / cfg.sigmoid_scale))
                prob = np.clip(cfg.p_max - (cfg.p_max - cfg.p_min) * sig, cfg.p_min, cfg.p_max)

                ax.imshow(
                    prob, origin='lower',
                    extent=[-180, 180, -90, 90],
                    vmin=cfg.p_min, vmax=cfg.p_max,
                    cmap='RdYlGn', aspect='auto',
                    interpolation='nearest',
                )
                ax.set_title(t_abs.strftime('%H:%M UTC'), fontsize=8)
                ax.set_xlabel("Lon (°)", fontsize=6)
                ax.set_ylabel("Lat (°)", fontsize=6)
                ax.tick_params(labelsize=5)

            # Hide unused panels
            for panel_idx in range(n_panels, rows * cols):
                row, col = divmod(panel_idx, cols)
                axes2[row, col].set_visible(False)

            fig2.suptitle(f"Acceptance Probability Evolution — {cname}", fontsize=10)
            fig2.colorbar(
                plt.cm.ScalarMappable(
                    cmap='RdYlGn',
                    norm=plt.Normalize(vmin=cfg.p_min, vmax=cfg.p_max)
                ),
                ax=axes2, label='p_accept', shrink=0.5,
            )
            plt.tight_layout()

            safe_name = cname.replace(" ", "_").replace("/", "_")
            out_path2 = os.path.join(results_path, f"demand_heatmap_{safe_name}.png")
            fig2.savefig(out_path2, dpi=dpi, bbox_inches='tight')
            plt.close(fig2)
            saved_files.append(out_path2)

        print(f"[DemandField] Saved {len(saved_files)} heatmap(s) to {results_path}")
        return saved_files


# ---------------------------------------------------------------------------
# Satellite → constellation name helper
# ---------------------------------------------------------------------------

def _constellation_name_from_satellite(sat_name: str) -> str:
    """Map a satellite name string to its constellation name.

    Matches the grouping logic in volcano_stochastic_comparison_real.py.
    Returns the constellation name used in DemandFieldConfig.constellation_popularity.
    """
    u = sat_name.upper()
    if any(x in u for x in ["SKYSAT", "PELICAN", "TANAGER"]):
        return "Planet"
    if "UMBRA" in u:
        return "Umbra"
    if "CAPELLA" in u or "ACADIA" in u:
        return "Capella"
    if "LOFT" in u or "YAM" in u:
        return "LOFT"
    if "UBOTICA" in u or "HAMMER" in u or "ACCENTURE" in u:
        return "Ubotica"
    if "PERSISTENCE" in u or "LEMUR" in u:
        return "Mission Control"
    if "AEROCUBE" in u:
        return "AC"
    if "ICEYE" in u:
        return "ICEYE"
    return "Unknown"

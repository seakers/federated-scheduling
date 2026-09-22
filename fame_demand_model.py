"""
Dynamic acceptance-probability model for the FAME federated-scheduling simulator.

The model maintains a latent demand field D(constellation, lat, lon, t) that evolves
via Ornstein-Uhlenbeck mean-reversion + event-driven spikes.  D is mapped to an
acceptance probability through a sigmoid so the value stays in a configurable band.

Single source of truth: the same DemandField instance is used by both the simulator
(to draw Bernoulli outcomes) and the stochastic planner (to set MILP coefficients).
Optionally, the planner can be given a noisy view of p_accept via
``install_planner_estimate_error`` / ``make_noisy_acceptance_prob_function`` while the
simulator continues to use the true field — used only by acceptance_sensitivity_sweep.py.

Two RNG streams are kept separate:
  _rng_demand  — drives the OU evolution of the demand field
  _rng_outcome — drives the individual accept/reject Bernoulli draws

This ensures that paired-seed comparisons between planners (greedy / det / stochastic)
remain valid: all three schedulers use the same _rng_outcome seed and therefore face
identical accept/reject outcomes.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math
import struct
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
    p_min: float = 0.70
    p_max: float = 0.95

    # --- sigmoid ---
    # The demand field lives on a scale of roughly [-3, +3].
    # sigmoid_scale controls the steepness: at d=+sigmoid_scale → ~73% of the way
    # from midpoint to p_min; at d=-sigmoid_scale → ~73% of the way to p_max.
    # A value of 1.0 gives a steep, well-differentiated map.
    sigmoid_scale: float = 1.0

    # --- OU dynamics ---
    # κ: mean-reversion rate (1/s).  1/7200 → half-life ~83 min, natural variation
    # over hours without being too slow or too jumpy.
    reversion_rate: float = 1.0 / 7200.0
    # σ: per-step noise std IN DEMAND UNITS (not per sqrt-second).
    # The OU stationary std = σ_step / sqrt(1 - exp(-2κΔt)).
    # With Δt=900s, κ=1/7200: exp(-2*900/7200)≈0.78, so stationary_std ≈ σ_step/0.47.
    # We want stationary_std ≈ 0.4 demand units → σ_step ≈ 0.19.
    noise_std: float = 0.19

    # --- event spikes ---
    # Spike magnitude in demand units.  +3 pushes a previously mid-range cell
    # to near-minimum p_accept; decays with a 3-hour half-life.
    spike_magnitude: float = 3.0
    spike_half_life_s: float = 3600.0 * 3

    # --- constellation popularity (must differ to create planner tension) ---
    # These drive the BASELINE demand mean (mu_i) for each constellation.
    # The mapping is:  mu = (popularity - 0.5) * 6
    # so popularity=0.85 → mu=+2.1 (congested, p≈0.35)
    #    popularity=0.20 → mu=-1.8 (quiet,     p≈0.85)
    constellation_popularity: dict = field(default_factory=lambda: {
        # cheap / popular — high demand → low p_accept
        "Planet":          0.85,
        "Ubotica":         0.78,
        # medium
        "LOFT":            0.60,
        "Mission Control": 0.55,
        "ICEYE":           0.48,
        "AC":              0.42,
        "OroraTech":       0.75,
        # expensive / reliable — low demand → high p_accept
        "Umbra":           0.32,
        "constellr":       0.24,
        "SatVu":           0.22,
        "Capella":         0.20,
    })
    default_popularity: float = 0.50

    # --- regional baselines (0=very quiet, 1=very congested) ---
    # Sorted roughly quiet→busy; later entries paint over earlier ones,
    # then the whole map is Gaussian-blurred for smooth transitions.
    regions: list = field(default_factory=lambda: [
        # Polar / remote — very quiet
        RegionDef("Arctic",        lat_min=70,  lat_max=90,  lon_min=-180, lon_max=180, baseline_demand=0.05),
        RegionDef("Antarctica",    lat_min=-90, lat_max=-60, lon_min=-180, lon_max=180, baseline_demand=0.05),
        RegionDef("S Pacific",     lat_min=-60, lat_max=-20, lon_min=140,  lon_max=-70, baseline_demand=0.08),
        RegionDef("S Atlantic",    lat_min=-55, lat_max=-15, lon_min=-50,  lon_max=15,  baseline_demand=0.10),
        RegionDef("Indian Ocean",  lat_min=-45, lat_max=10,  lon_min=40,   lon_max=100, baseline_demand=0.12),
        RegionDef("N Pacific",     lat_min=20,  lat_max=55,  lon_min=160,  lon_max=-130,baseline_demand=0.15),
        # Moderate demand — active but not peak
        RegionDef("S America",     lat_min=-55, lat_max=15,  lon_min=-82,  lon_max=-34, baseline_demand=0.38),
        RegionDef("Sub-Sah Africa",lat_min=-35, lat_max=15,  lon_min=-18,  lon_max=50,  baseline_demand=0.38),
        RegionDef("Central Asia",  lat_min=30,  lat_max=55,  lon_min=50,   lon_max=90,  baseline_demand=0.48),
        RegionDef("N Africa",      lat_min=15,  lat_max=38,  lon_min=-18,  lon_max=37,  baseline_demand=0.45),
        RegionDef("Australia",     lat_min=-40, lat_max=-10, lon_min=113,  lon_max=154, baseline_demand=0.42),
        # High demand — heavily tasked regions
        RegionDef("N America",     lat_min=25,  lat_max=60,  lon_min=-130, lon_max=-60, baseline_demand=0.58),
        RegionDef("Europe",        lat_min=35,  lat_max=70,  lon_min=-10,  lon_max=40,  baseline_demand=0.62),
        RegionDef("East Asia",     lat_min=20,  lat_max=50,  lon_min=105,  lon_max=145, baseline_demand=0.70),
        RegionDef("SE Asia",       lat_min=-10, lat_max=25,  lon_min=95,   lon_max=140, baseline_demand=0.72),
        # Peak demand — conflict zones, choke points, major ISR targets
        RegionDef("Middle East",   lat_min=15,  lat_max=40,  lon_min=35,   lon_max=65,  baseline_demand=0.98),
        RegionDef("Korea Strait",  lat_min=32,  lat_max=42,  lon_min=124,  lon_max=132, baseline_demand=0.85),
        RegionDef("Persian Gulf",  lat_min=23,  lat_max=30,  lon_min=48,   lon_max=60,  baseline_demand=0.92),
        RegionDef("S China Sea",   lat_min=5,   lat_max=25,  lon_min=108,  lon_max=122, baseline_demand=0.90),
        RegionDef("E Ukraine",     lat_min=46,  lat_max=52,  lon_min=30,   lon_max=40,  baseline_demand=0.95),
    ])

    # --- grid & timing ---
    grid_lat_deg: float = 2.0
    grid_lon_deg: float = 2.0
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
        exp_k = math.exp(-kappa * dt_s)
        noise_scale = cfg.noise_std

        # Spatially correlated noise: generate at a coarse ~20° grid then zoom
        # to full resolution so demand perturbations drift as coherent regional
        # blobs rather than pixel-level static.
        from scipy.ndimage import zoom as nd_zoom
        _coarse_lat = max(2, self._n_lat // 9)   # ~20° cells
        _coarse_lon = max(2, self._n_lon // 9)
        _zoom_lat = self._n_lat / _coarse_lat
        _zoom_lon = self._n_lon / _coarse_lon

        for ti in range(1, self._n_steps):
            # Draw at coarse resolution
            coarse = self._rng_demand.standard_normal((n_c, _coarse_lat, _coarse_lon)) * noise_scale
            # Zoom to full grid with smooth interpolation (order=2 = quadratic spline)
            noise = np.stack([
                nd_zoom(coarse[ci], (_zoom_lat, _zoom_lon), order=2)[:self._n_lat, :self._n_lon]
                for ci in range(n_c)
            ])
            demand[..., ti] = baseline + exp_k * (demand[..., ti - 1] - baseline) + noise

        # --- Event spikes ---
        for lat_s, lon_s, t_start_s in self._spikes:
            spike_array = self._spike_contribution(lat_s, lon_s, t_start_s)  # (n_lat, n_lon, n_steps)
            for ci in range(n_c):
                demand[ci] += cfg.spike_magnitude * spike_array

        self._demand = demand
        self._precomputed = True

    def _build_baseline(self, n_c: int) -> np.ndarray:
        """Return (n_c, n_lat, n_lon) array of OU mean-reversion targets (μ).

        The demand field lives on a scale of roughly [-3, +3] where:
          +3  → very congested → p_accept near p_min (~0.30)
          0   → neutral        → p_accept near midpoint (~0.60)
          -3  → very quiet     → p_accept near p_max (~0.90)

        The geographic baseline encodes region congestion as a value in [-3, +3].
        Per-constellation popularity linearly shifts that baseline up (popular,
        congested) or down (niche, reliable).
        """
        cfg = self.config

        # Geo-congestion: open ocean quiet (-2), busy regions positive
        # RegionDef.baseline_demand is [0,1] → remap to [-2, +3] demand units
        geo_demand = np.full((self._n_lat, self._n_lon), -2.0)  # ocean default: quiet

        for region in sorted(cfg.regions, key=lambda r: r.baseline_demand):
            lat_mask = (self._lats >= region.lat_min) & (self._lats <= region.lat_max)
            lon_mask = (self._lons >= region.lon_min) & (self._lons <= region.lon_max)
            lat_idx = np.where(lat_mask)[0]
            lon_idx = np.where(lon_mask)[0]
            if len(lat_idx) == 0 or len(lon_idx) == 0:
                continue
            ii, jj = np.meshgrid(lat_idx, lon_idx, indexing='ij')
            # Map [0, 1] baseline_demand linearly to [-2, +3] demand units
            d_val = -2.0 + region.baseline_demand * 5.0
            geo_demand[ii, jj] = d_val

        # Smooth the hard region boundaries with a wide Gaussian blur
        from scipy.ndimage import gaussian_filter
        geo_demand = gaussian_filter(geo_demand, sigma=3.0)

        # Per-constellation popularity shifts the mean: popularity=0.5 → no shift;
        # popularity=1.0 → +3 (fully congested); popularity=0.0 → -3 (very quiet)
        baseline = np.empty((n_c, self._n_lat, self._n_lon))
        for ci, cname in enumerate(self._constellations):
            pop = cfg.constellation_popularity.get(cname, cfg.default_popularity)
            pop_shift = (pop - 0.5) * 6.0   # maps [0,1] → [-3, +3]
            baseline[ci] = geo_demand + pop_shift

        return baseline

    def _spike_contribution(self, lat: float, lon: float, t_start_s: float) -> np.ndarray:
        """Return (n_lat, n_lon, n_steps) spike contribution (unnormalized; caller multiplies by magnitude)."""
        cfg = self.config
        decay_rate = math.log(2) / cfg.spike_half_life_s

        # Spatial Gaussian: ~1300 km radius (12° at equator ≈ 1335 km)
        sigma_deg = 12.0
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
            "OroraTech": 0.45, "constellr": 0.85, "SatVu": 0.85,
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

        If ``planner_relative_error`` is set on this instance (see
        ``install_planner_estimate_error``), the returned callable applies a
        multiplicative relative error to the *true* p_accept.  The simulator
        path (``make_simulator_acceptance_function``) is never affected.
        Default / unset → identical to the true demand field (no behaviour change).
        """
        demand_field = self
        rel_error = float(getattr(self, "planner_relative_error", 0.0) or 0.0)
        est_seed = int(getattr(self, "planner_estimate_seed", 0) or 0)

        if rel_error <= 0.0:
            def _f(constrained_request, satellite, obs_pass):
                req = constrained_request.observation_request
                lat = req.lat_deg
                lon = req.lon_deg
                t = obs_pass.highest.time
                constellation = _constellation_name_from_satellite(satellite.name)
                return demand_field.acceptance_probability(constellation, lat, lon, t)

            return _f

        return self.make_noisy_acceptance_prob_function(
            relative_error=rel_error, estimate_seed=est_seed,
        )

    def make_noisy_acceptance_prob_function(
        self,
        relative_error: float,
        estimate_seed: int = 0,
    ):
        """Planner-only approximate p_accept with a relative error margin.

        For each coarse (constellation, lat, lon, time) key the planner sees
        ``p_hat = clip(p_true * U(1-δ, 1+δ), 0, 1)`` where δ = ``relative_error``.
        Factors are deterministic given ``estimate_seed`` (hash-based), so query
        order does not matter and paired schedulers share the same estimate.

        The simulator must keep using ``make_simulator_acceptance_function()``
        (truth) so accept/reject outcomes stay calibrated to the true demand.
        """
        demand_field = self
        if relative_error <= 0.0:
            # Always return exact truth here (do not re-enter make_acceptance_prob_function,
            # which may have planner_relative_error set).
            def _exact(constrained_request, satellite, obs_pass):
                req = constrained_request.observation_request
                lat = req.lat_deg
                lon = req.lon_deg
                t = obs_pass.highest.time
                constellation = _constellation_name_from_satellite(satellite.name)
                return demand_field.acceptance_probability(constellation, lat, lon, t)

            return _exact

        delta = float(relative_error)
        seed = int(estimate_seed)

        def _factor(constellation: str, lat: float, lon: float, t: dt.datetime) -> float:
            # Coarse bins → coherent regional mis-estimate of demand / p_accept.
            t_h = (t - demand_field.reference_time).total_seconds() / 3600.0
            key = (
                f"{seed}|{constellation}|"
                f"{round(lat, 1):.1f}|{round(lon, 1):.1f}|{round(t_h, 1):.1f}"
            )
            digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
            u = struct.unpack("<Q", digest)[0] / 2**64  # [0, 1)
            return 1.0 - delta + 2.0 * delta * u

        def _f(constrained_request, satellite, obs_pass):
            req = constrained_request.observation_request
            lat = req.lat_deg
            lon = req.lon_deg
            t = obs_pass.highest.time
            constellation = _constellation_name_from_satellite(satellite.name)
            p_true = demand_field.acceptance_probability(constellation, lat, lon, t)
            p_hat = p_true * _factor(constellation, lat, lon, t)
            return float(np.clip(p_hat, 0.0, 1.0))

        return _f

    def install_planner_estimate_error(
        self,
        relative_error: float = 0.0,
        estimate_seed: int = 0,
    ) -> None:
        """Opt-in: make subsequent ``make_acceptance_prob_function()`` noisy.

        Leave unset / ``relative_error=0`` for exact planner knowledge (default).
        Does not affect ``make_simulator_acceptance_function``.
        """
        self.planner_relative_error = float(relative_error)
        self.planner_estimate_seed = int(estimate_seed)

    def clear_planner_estimate_error(self) -> None:
        """Restore exact planner knowledge of p_accept."""
        self.planner_relative_error = 0.0
        self.planner_estimate_seed = 0

    def make_simulator_acceptance_function(self):
        """Return a callable matching the simulator signature:
        f(request, satellite, world_time) → float (probability; caller draws the coin)

        Note: the simulator in fame_constellation_scheduler.py draws its own
        random.random() against the returned probability, so we return p here,
        not a boolean.  The outcome RNG stream in DemandField is therefore NOT
        used for the simulator path (stdlib random handles that draw), which is
        intentional: the simulator's accept/reject draw must remain tied to the
        shared stdlib random seed so all planners face identical outcomes.

        Always returns the *true* acceptance probability (never the planner's
        noisy estimate).
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

    def _prob_grid(self, ci: int, ti: int) -> np.ndarray:
        """Return smoothed acceptance-probability array (n_lat, n_lon) for one constellation/timestep."""
        from scipy.ndimage import gaussian_filter
        cfg = self.config
        d_slice = self._demand[ci, :, :, ti]
        sig = 1.0 / (1.0 + np.exp(-d_slice / cfg.sigmoid_scale))
        prob = np.clip(cfg.p_max - (cfg.p_max - cfg.p_min) * sig, cfg.p_min, cfg.p_max)
        # Smooth across ~3 grid cells so region boundaries blend naturally
        return gaussian_filter(prob, sigma=1.5)

    def plot_heatmaps(
        self,
        results_path: str,
        timestep_indices: Optional[list[int]] = None,
        dpi: int = 150,
    ) -> list[str]:
        """Save per-constellation acceptance-probability maps on a proper Earth projection.

        Produces two outputs per run:
        1. ``demand_heatmaps_all.png``  — grid of (constellation × timestep) world maps.
        2. ``demand_heatmap_<name>.png`` — per-constellation multi-panel evolution strip.

        Each panel uses a PlateCarree projection with coastlines drawn from the
        bundled Natural Earth 50m-land shapefile (no network access required),
        a smooth probability gradient, geographic region annotations, and a
        shared perceptually-uniform colorbar.
        """
        import os
        import cartopy.crs as ccrs
        import cartopy.io.shapereader as shpreader
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path
        import matplotlib.patches as mpatches
        import matplotlib.ticker as mticker
        import shapely.geometry as sgeom

        os.makedirs(results_path, exist_ok=True)

        if not self._precomputed:
            raise RuntimeError("Call precompute() before plotting.")

        n_c = len(self._constellations)
        if n_c == 0:
            return []

        if timestep_indices is None:
            n_show = min(4, self._n_steps)
            timestep_indices = list(np.linspace(0, self._n_steps - 1, n_show, dtype=int))

        cfg = self.config
        saved_files = []

        # --- Colormap: deep red (congested/low p) → amber → teal → dark blue (quiet/high p) ---
        _cmap = LinearSegmentedColormap.from_list(
            "demand",
            [
                "#c0392b",   # 0.0 — deep red:   congested, low p_accept
                "#e74c3c",   # 0.2
                "#f39c12",   # 0.4 — amber:       moderate demand
                "#f1c40f",   # 0.5 — yellow:      borderline
                "#2ecc71",   # 0.7 — green:       light demand
                "#1abc9c",   # 0.85— teal
                "#1a5276",   # 1.0 — dark blue:   quiet, high p_accept
            ],
        )
        norm = plt.Normalize(vmin=cfg.p_min, vmax=cfg.p_max)

        # --- Load cached land shapefile for coastlines (no download needed) ---
        _LAND_SHP = os.path.join(
            os.path.expanduser("~"), ".local", "share", "cartopy",
            "shapefiles", "natural_earth", "physical", "ne_50m_land.shp",
        )
        _land_geoms: list = []
        if os.path.exists(_LAND_SHP):
            reader = shpreader.Reader(_LAND_SHP)
            _land_geoms = list(reader.geometries())
        else:
            print("[DemandField] Warning: ne_50m_land.shp not found; coastlines will be skipped.")

        # --- Region label positions (lon, lat, text) ---
        _region_labels = [
            (  47.0,  26.0, "Middle East"),
            ( 115.0,   5.0, "SE Asia"),
            ( 127.0,  36.0, "E Asia"),
            (  15.0,  52.0, "Europe"),
            ( -97.0,  40.0, "N America"),
            ( -45.0, -35.0, "S Atlantic\n(quiet)"),
            ( 170.0, -38.0, "S Pacific\n(quiet)"),
            (  75.0,  20.0, "S Asia"),
            ( -65.0, -15.0, "S America"),
        ]

        proj = ccrs.PlateCarree()

        # ------------------------------------------------------------------
        # Helper: draw one probability map onto a Cartopy axes
        # ------------------------------------------------------------------
        def _draw_map(ax, prob_grid, title, fontsize_title=8):
            ax.set_extent([-180, 180, -90, 90], crs=proj)

            # Ocean background
            ax.set_facecolor("#d0e8f5")

            # Probability overlay (full globe incl. ocean)
            ax.pcolormesh(
                self._lons, self._lats, prob_grid,
                transform=proj,
                cmap=_cmap, norm=norm,
                alpha=0.88, zorder=1,
                shading="auto",
            )

            # Land outlines from cached shapefile
            for geom in _land_geoms:
                ax.add_geometries(
                    [geom], proj,
                    facecolor="none",
                    edgecolor="#2c3e50",
                    linewidth=0.4,
                    zorder=3,
                )

            # Lat/lon grid lines (manual — no download required)
            for lon_g in range(-180, 181, 60):
                ax.plot(
                    [lon_g, lon_g], [-90, 90],
                    transform=proj, color="white",
                    linewidth=0.25, alpha=0.4, zorder=2,
                )
                if -165 < lon_g < 180:
                    ax.text(
                        lon_g, -87, f"{lon_g}°",
                        transform=proj, fontsize=4, ha="center",
                        color="white", alpha=0.7, zorder=4,
                    )
            for lat_g in range(-60, 91, 30):
                ax.plot(
                    [-180, 180], [lat_g, lat_g],
                    transform=proj, color="white",
                    linewidth=0.25, alpha=0.4, zorder=2,
                )
                ax.text(
                    -178, lat_g, f"{lat_g}°",
                    transform=proj, fontsize=4, ha="left",
                    color="white", alpha=0.7, zorder=4,
                )

            # Region annotations — always shown on every panel
            for lon_l, lat_l, lbl in _region_labels:
                ax.text(
                    lon_l, lat_l, lbl,
                    transform=proj, fontsize=4.5,
                    ha="center", va="center",
                    color="white", fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="#00000068",
                        edgecolor="none",
                    ),
                    zorder=5,
                )

            ax.set_title(title, fontsize=fontsize_title, pad=3, color="white")

        # Shared colorbar label
        _cbar_label = (
            "Acceptance probability   "
            "▌ red = congested / low    ▌ yellow = moderate    ▌ blue = quiet / high"
        )

        # ------------------------------------------------------------------
        # Figure 1: all constellations × selected timesteps
        # ------------------------------------------------------------------
        n_t = len(timestep_indices)
        fig_w = max(4.5 * n_c, 12)
        fig_h = 2.8 * n_t + 0.8

        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#0d1117")

        axes_flat = []
        for row in range(n_t):
            for col in range(n_c):
                ax = fig.add_subplot(n_t, n_c, row * n_c + col + 1, projection=proj)
                axes_flat.append(ax)

        for ti_idx, ti in enumerate(timestep_indices):
            t_s = self._times_s[ti]
            t_abs = self.reference_time + dt.timedelta(seconds=float(t_s))
            t_label = t_abs.strftime("%Y-%m-%d %H:%M UTC")

            for ci, cname in enumerate(self._constellations):
                idx = ti_idx * n_c + ci
                pop = cfg.constellation_popularity.get(cname, cfg.default_popularity)
                title = f"{cname}  (pop={pop:.0%})  —  {t_label}"
                _draw_map(
                    axes_flat[idx],
                    self._prob_grid(ci, ti),
                    title,
                                    )

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=_cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(
            sm, ax=axes_flat,
            orientation="horizontal", fraction=0.018, pad=0.03, aspect=60,
        )
        cbar.set_label(_cbar_label, fontsize=7, color="white")
        cbar.ax.tick_params(labelsize=6, colors="white")
        cbar.outline.set_edgecolor("#555")

        fig.suptitle(
            "Dynamic Acceptance Probability — commercial constellation demand field",
            fontsize=11, color="white", y=1.002, fontweight="bold",
        )
        plt.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.06,
                            hspace=0.25, wspace=0.05)

        out_all = os.path.join(results_path, "demand_heatmaps_all.png")
        fig.savefig(out_all, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        saved_files.append(out_all)
        print(f"[DemandField] Saved overview: {out_all}")

        # ------------------------------------------------------------------
        # Figure 2: per-constellation evolution strip
        # ------------------------------------------------------------------
        for ci, cname in enumerate(self._constellations):
            n_panels = len(timestep_indices)
            cols = min(n_panels, 4)
            rows = math.ceil(n_panels / cols)

            pop = cfg.constellation_popularity.get(cname, cfg.default_popularity)
            fig2 = plt.figure(
                figsize=(4.5 * cols, 2.8 * rows + 0.6),
                facecolor="#0d1117",
            )
            fig2.suptitle(
                f"{cname}  —  acceptance probability evolution  "
                f"(popularity={pop:.0%})",
                fontsize=9, color="white", y=1.002, fontweight="bold",
            )

            panel_axes = []
            for panel_idx, ti in enumerate(timestep_indices):
                ax2 = fig2.add_subplot(rows, cols, panel_idx + 1, projection=proj)
                t_s = self._times_s[ti]
                t_abs = self.reference_time + dt.timedelta(seconds=float(t_s))
                hours_offset = t_s / 3600.0
                title = f"T+{hours_offset:.1f}h  ({t_abs.strftime('%H:%M UTC')})"
                _draw_map(
                    ax2,
                    self._prob_grid(ci, ti),
                    title,
                    fontsize_title=7,
                )
                panel_axes.append(ax2)

            sm2 = plt.cm.ScalarMappable(cmap=_cmap, norm=norm)
            sm2.set_array([])
            cbar2 = fig2.colorbar(
                sm2, ax=panel_axes,
                orientation="horizontal", fraction=0.025, pad=0.04, aspect=40,
            )
            cbar2.set_label("p_accept", fontsize=7, color="white")
            cbar2.ax.tick_params(labelsize=6, colors="white")
            cbar2.outline.set_edgecolor("#555")

            plt.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.08,
                                hspace=0.25, wspace=0.05)

            safe_name = cname.replace(" ", "_").replace("/", "_")
            out2 = os.path.join(results_path, f"demand_heatmap_{safe_name}.png")
            fig2.savefig(out2, dpi=dpi, bbox_inches="tight", facecolor=fig2.get_facecolor())
            plt.close(fig2)
            saved_files.append(out2)

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
    if any(x in u for x in ["SKYSAT", "PELICAN", "TANAGER", "FLOCK"]):
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
    if "FOREST" in u:
        return "OroraTech"
    if "SKYBEE" in u:
        return "constellr"
    if "HOTSAT" in u:
        return "SatVu"
    return "Unknown"

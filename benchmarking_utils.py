"""
Shared utilities for MSA and volcano scheduler benchmarking scripts.

Extracted from msa_stochastic_comparison_real.py and
volcano_stochastic_comparison_real.py to avoid duplication.
"""

import datetime as dt
import copy
import glob

import numpy as np
import pyorbital
import pyorbital.orbital
from pyorbital.orbital import Orbital

from fame_geometry import *
from fame_agents_base import *
from fame_constellation_scheduler import ConstellationGroundScheduler


# ===========================================================================
# GROUND STATIONS  (shared 11-site KSAT network)
# ===========================================================================

GROUND_STATIONS = [
    Location(-79.55,    8.9833,    0.028, "KSAT Panama"),
    Location(-51.73363, 64.182789, 0,     "KSAT Nuuk"),
    Location(2.53219,  -72.01243,  0,     "KSAT Troll"),
    Location(142.3689,  43.8,      0,     "KSAT Hokkaido"),
    Location(103.9915,  1.3661,    0,     "KSAT Singapore"),
    Location(-70.85021,-52.93279,  0,     "KSAT Punta Arenas"),
    Location(127.7766,  26.4055,   0,     "KSAT Okinawa"),
    Location(57.5565,  -20.1142,   0,     "KSAT Mauritius"),
    Location(22.62216,  37.84604,  0,     "KSAT Nemea"),
    Location(31.12509,  70.36779,  0,     "KSAT Vardo"),
    Location(15.39964,  78.22875,  0,     "KSAT Svalbard"),
]


# ===========================================================================
# SATELLITE LOADING
# ===========================================================================

def load_satellites_once(sim_start: dt.datetime, horizon_h: float) -> list:
    """
    Load the full 8-constellation LEO fleet from the most recent TLE file.

    Uses display names as the canonical key (e.g. "SKYSAT-C1", "ICEYE-X4"),
    resolves TLE names via tle_to_display where they differ, and filters out
    decayed or invalid orbits by propagating to both sim_start and sim_start+horizon_h.

    Returns a list of Satellite objects with instrument_fov_rad populated.
    """
    tle_files = glob.glob("tles/all_tles_*.txt")
    if not tle_files:
        raise FileNotFoundError("No TLE files found in tles/")
    tle_file = sorted(tle_files)[-1]
    print(f"[Init] Using TLE file: {tle_file}")

    min_time = sim_start
    max_time = sim_start + dt.timedelta(hours=horizon_h)

    # Dynamic FLOCK / ICEYE names from TLE file
    flock_names, iceye_names = [], []
    with open(tle_file) as f:
        for line in f:
            if line.startswith("FLOCK"):
                flock_names.append(line.strip())
            elif line.startswith("ICEYE"):
                iceye_names.append(line.strip())

    # Display-name → swath km
    swaths_at_nadir_km = {
        "SKYSAT-A": 8, "SKYSAT-B": 8,
        "SKYSAT-C1": 5.9, "SKYSAT-C2": 5.9, "SKYSAT-C3": 5.9, "SKYSAT-C4": 5.9,
        "SKYSAT-C5": 5.9, "SKYSAT-C6": 5.9, "SKYSAT-C7": 5.9, "SKYSAT-C8": 5.9,
        "SKYSAT-C9": 5.9, "SKYSAT-C10": 5.9, "SKYSAT-C11": 5.9, "SKYSAT-C12": 5.9,
        "SKYSAT-C13": 5.9,
        "PELICAN-3001": 8, "PELICAN-3009": 8, "PELICAN-300A": 8,
        "PELICAN-300B": 8, "PELICAN-5": 8, "PELICAN-6": 8,
        "TANAGER-4001": 18,
        "UMBRA-07": 8, "UMBRA-09": 8, "UMBRA-10": 8, "UMBRA-11": 8,
        "CAPELLA-11 (ACADIA-1)": 10, "CAPELLA-13 (ACADIA-3)": 10,
        "CAPELLA-14 (ACADIA-4)": 10, "CAPELLA-15 (ACADIA-5)": 10,
        "CAPELLA-16 (ACADIA-6)": 10, "CAPELLA-17 (ACADIA-7)": 10,
        "LOFT YAM-6": 19.8, "LOFT YAM-7": 70, "LOFT YAM-8": 19.8,
        "Ubotica CogniSat-6 HAMMER": 20, "Ubotica ACCENTURE-1 SUAC": 20,
        "Mission Control Persistence": 100,
        #"LEMUR 2 KRISH": 17.5,
        "AEROCUBE 18A": 80, "AEROCUBE 18B": 80,
    }
    for name in flock_names:
        swaths_at_nadir_km[name] = 16.4
    for name in iceye_names:
        swaths_at_nadir_km[name] = 100

    # TLE name → display name for satellites whose TLE name differs from display name
    tle_to_display = {
        "SKYSAT 1": "SKYSAT-A", "SKYSAT 2": "SKYSAT-B",
        "SKYSAT C1": "SKYSAT-C1",  "SKYSAT C2": "SKYSAT-C2",  "SKYSAT C3": "SKYSAT-C3",
        "SKYSAT C4": "SKYSAT-C4",  "SKYSAT C5": "SKYSAT-C5",  "SKYSAT C6": "SKYSAT-C6",
        "SKYSAT C7": "SKYSAT-C7",  "SKYSAT C8": "SKYSAT-C8",  "SKYSAT C9": "SKYSAT-C9",
        "SKYSAT C10": "SKYSAT-C10","SKYSAT C11": "SKYSAT-C11","SKYSAT C12": "SKYSAT-C12",
        "SKYSAT C13": "SKYSAT-C13",
        "PELICAN-1 3001": "PELICAN-3001", "PELICAN-2 3009": "PELICAN-3009",
        "PELICAN-3 300A": "PELICAN-300A", "PELICAN-4 300B": "PELICAN-300B",
        "PELICAN-5 300C": "PELICAN-5",   "PELICAN-6 300D": "PELICAN-6",
        "TANAGER-1 4001": "TANAGER-4001",
        "CAPELLA-11 (ACADIA)": "CAPELLA-11 (ACADIA-1)",
        "CAPELLA-13 (ACADIA)": "CAPELLA-13 (ACADIA-3)",
        "CAPELLA-14 (ACADIA)": "CAPELLA-14 (ACADIA-4)",
        "CAPELLA-15 (ACADIA)": "CAPELLA-15 (ACADIA-5)",
        "CAPELLA-16 (ACADIA)": "CAPELLA-16 (ACADIA-6)",
        "CAPELLA-17 (ACADIA)": "CAPELLA-17 (ACADIA-7)",
        "YAM-6": "LOFT YAM-6", "YAM-7": "LOFT YAM-7", "YAM-8": "LOFT YAM-8",
        "HAMMER": "Ubotica CogniSat-6 HAMMER",
        "ACCENTURE-1": "Ubotica ACCENTURE-1 SUAC",
        "LEMUR 2 KRISH": "Mission Control Persistence",
    }
    display_to_tle = {v: k for k, v in tle_to_display.items()}

    sat_constellation_map = {
        "SKYSAT":      ("Planet",          InstrumentType.RGB),
        "PELICAN":     ("Planet",          InstrumentType.RGB),
        "TANAGER":     ("Planet",          InstrumentType.HYPERSPECTRAL),
        "FLOCK":       ("Planet",          InstrumentType.RGB),
        "UMBRA":       ("Umbra",           InstrumentType.SAR),
        "CAPELLA":     ("Capella",         InstrumentType.SAR),
        "ACADIA":      ("Capella",         InstrumentType.SAR),
        "YAM":         ("LOFT",            InstrumentType.HYPERSPECTRAL),
        "LOFT":        ("LOFT",            InstrumentType.HYPERSPECTRAL),
        "HAMMER":      ("Ubotica",         InstrumentType.HYPERSPECTRAL),
        "ACCENTURE":   ("Ubotica",         InstrumentType.HYPERSPECTRAL),
        "UBOTICA":     ("Ubotica",         InstrumentType.HYPERSPECTRAL),
        "LEMUR":       ("Mission Control", InstrumentType.RGB),
        "PERSISTENCE": ("Mission Control", InstrumentType.RGB),
        "AEROCUBE":    ("Aerospace",       InstrumentType.RGB),
        "ICEYE":       ("ICEYE",           InstrumentType.SAR),
    }

    satellites = []
    skipped = 0
    for display_name, swath_km in swaths_at_nadir_km.items():
        tle_name = display_to_tle.get(display_name, display_name)
        constellation, instrument = "Unknown", InstrumentType.RGB
        for key, (const, inst) in sat_constellation_map.items():
            if key in display_name.upper():
                constellation, instrument = const, inst
                break
        try:
            orbit = Orbital(tle_name, tle_file=tle_file)
            _ = orbit.get_lonlatalt(min_time)
            _ = orbit.get_lonlatalt(max_time)
            sat = Satellite(display_name, orbit, instruments=[instrument],
                            has_continuous_isl_to_ground=True)
            semi_major = sat.orbit.orbit_elements.semi_major_axis * pyorbital.orbital.A
            altitude = semi_major - pyorbital.orbital.A
            fov = 2 * np.atan2(swath_km / 2, altitude)
            sat.instrument_fov_rad = {it: fov for it in sat.instruments}
            satellites.append(sat)
        except Exception:
            skipped += 1
            continue

    print(f"[Init] Loaded {len(satellites)} satellites (skipped {skipped} decayed/invalid).")
    return satellites


# ===========================================================================
# WORLD + CONSTELLATION FACTORY
# ===========================================================================

def create_world_and_constellations(
    cached_satellites: list,
    sim_start: dt.datetime,
    demand_field=None,
    execution_probability_function=None,
    acceptance_notification_delay_h: tuple = (0.0, 0.0),
):
    """
    Spin up a fresh World + 8 ConstellationGroundSchedulers from deep-copied satellites.

    demand_field: if provided, its make_simulator_acceptance_function() is wired as
                  the simulator-side acceptance draw (same model the planner uses).
    execution_probability_function: if provided, passed to every ConstellationGroundScheduler.
    acceptance_notification_delay_h: (min_h, max_h) uniform delay before accept/reject
                  notification fires before the pass. (0, 0) = synchronous (no delay).
    """
    local_sats = copy.deepcopy(cached_satellites)
    world = World(satellites=local_sats)
    world.time = sim_start

    planet_sats   = [s for s in local_sats if any(x in s.name.upper() for x in ["SKYSAT", "PELICAN", "TANAGER", "FLOCK"])]
    umbra_sats    = [s for s in local_sats if "UMBRA"       in s.name.upper()]
    capella_sats  = [s for s in local_sats if "CAPELLA"     in s.name.upper() or "ACADIA"       in s.name.upper()]
    loft_sats     = [s for s in local_sats if "LOFT"        in s.name.upper() or "YAM"          in s.name.upper()]
    ubotica_sats  = [s for s in local_sats if "UBOTICA"     in s.name.upper() or "HAMMER"       in s.name.upper() or "ACCENTURE" in s.name.upper()]
    mc_sats       = [s for s in local_sats if "PERSISTENCE" in s.name.upper() or "LEMUR"        in s.name.upper()]
    aero_sats     = [s for s in local_sats if "AEROCUBE"    in s.name.upper()]
    iceye_sats    = [s for s in local_sats if "ICEYE"       in s.name.upper()]

    sim_acc_fn = demand_field.make_simulator_acceptance_function() if demand_field is not None else None

    def _make(sats, name, legacy_p):
        return ConstellationGroundScheduler(
            satellites=sats, ground_stations=GROUND_STATIONS,
            world=world, name=name,
            acceptance_probability=legacy_p,
            acceptance_probability_function=sim_acc_fn,
            execution_probability_function=execution_probability_function,
            acceptance_notification_delay_h=acceptance_notification_delay_h,
        )

    sched_planet  = _make(planet_sats,  "Planet",          0.60)
    sched_umbra   = _make(umbra_sats,   "Umbra",           0.80)
    sched_capella = _make(capella_sats, "Capella",         0.85)
    sched_loft    = _make(loft_sats,    "LOFT",            0.81)
    sched_ubotica = _make(ubotica_sats, "Ubotica",         0.80)
    sched_mc      = _make(mc_sats,      "Mission Control", 0.74)
    sched_aero    = _make(aero_sats,    "AC",              0.77)
    sched_iceye   = _make(iceye_sats,   "ICEYE",           0.74)

    all_constellations = [
        sched_planet, sched_umbra, sched_capella, sched_loft,
        sched_ubotica, sched_mc, sched_aero, sched_iceye,
    ]
    for c in all_constellations:
        world.add_constellation(c)

    return world, all_constellations


# ===========================================================================
# SIMULATION RUNNER
# ===========================================================================

def run_simulation_forward(world, max_ticks: int = 40000, profile_ticks: int = 0):
    """Tick the event loop to completion (retcode=0) or safety cut-off.

    Args:
        profile_ticks: if > 0, print per-tick action vs deepcopy timing for the
                       first N ticks so you can identify the real bottleneck.
    """
    import time as _time
    ticks = 0
    last_report = 0
    t_action_total = 0.0
    t_copy_total = 0.0
    profiling = profile_ticks > 0
    while True:
        profiling_this_tick = profiling and ticks < profile_ticks
        retcode, t_action, t_copy = world.tick(
            print_forbidden_prefixes=[
                "Downlink", "End of downlink", "Unlock uplink",
                "Unlock satellite after obs",
            ],
            record_history=False,
            profile=profiling_this_tick,
        )
        if profiling:
            t_action_total += t_action
            t_copy_total += t_copy
        ticks += 1
        if ticks - last_report >= 5000:
            print(f"    [Sim] {ticks} ticks, time: {world.time}")
            last_report = ticks
        if retcode == 0:
            break
        if ticks >= max_ticks:
            print(f"    [Sim] Safety cut-off at {max_ticks} ticks.")
            break
    print(f"    [Sim] Done. {ticks} ticks, final time: {world.time}")
    if profiling:
        print(f"    [Sim profile] action={t_action_total:.2f}s  deepcopy={t_copy_total:.2f}s  ({ticks} ticks profiled)")


# ===========================================================================
# MEMORY DIAGNOSTICS
# ===========================================================================

def rss_gb() -> float:
    """Peak resident set size in GB (diagnostic only)."""
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / 1e9 if rss > 1e7 else rss / 1e6
    except Exception:
        return float('nan')

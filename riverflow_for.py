"""Field-of-regard access filter for the river high-flow case study.

Volcano, earthquake, and MSA keep using ``find_observation_opportunities``
as written: one ``min_elevation_deg`` on the request (20°). This module is
not imported by those drivers.

Access in FAME is elevation-only. Swath and pointing never enter the pass
search. For TIR that is wrong: a 17.5 km SkyBee and a 400 km FOREST are not
the same camera. This file converts swath / off-nadir FOR into a per-satellite
elevation floor and drops passes that clear the request gate but miss the
instrument.

SAR and RGB have no entry here, so they keep the request's 20° rule.

Call ``attach_for_limits(satellites)`` then ``install()`` from the river
driver before the first schedule. ``uninstall()`` restores the originals.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import NamedTuple, Optional

import pyorbital.orbital

# Mean Earth radius used by the elevation conversions (km).
R_EARTH_KM = 6371.0


class SensorAccess(NamedTuple):
    """How a named family can see the ground.

    ``for_off_nadir_deg`` is None when the payload is a fixed nadir / lateral
    scan: the swath edge is the access edge. When set, pointing wins and the
    elevation floor is the more permissive of the two (SkyBee ±30°).
    """
    swath_km: float
    for_off_nadir_deg: Optional[float]
    note: str = ""


# Longest / most specific name fragments first.
_ACCESS_BY_SUBSTRING: tuple[tuple[str, SensorAccess], ...] = (
    ("HOTSAT", SensorAccess(25.0, 30.0, "SatVu; 25 km swath unverified")),
    ("SKYBEE", SensorAccess(17.5, 30.0, "constellr HiVE, ±30° FOR")),
    ("LANDSAT", SensorAccess(185.0, 15.0, "TIRS; ±15° emergency roll")),
    ("SENTINEL 3", SensorAccess(1420.0, None, "SLSTR dual-view TIR")),
    ("FOREST-2", SensorAccess(410.0, None, "OroraTech FOREST-2 / SAFIRE")),
    ("FOREST-4P", SensorAccess(410.0, None, "OTC-P1")),
    ("FOREST-5P", SensorAccess(410.0, None, "OTC-P1")),
    ("FOREST-6P", SensorAccess(410.0, None, "OTC-P1")),
    ("FOREST-7P", SensorAccess(410.0, None, "OTC-P1")),
    ("FOREST-8P", SensorAccess(410.0, None, "OTC-P1")),
    ("FOREST-9P", SensorAccess(410.0, None, "OTC-P1")),
    ("FOREST-10P", SensorAccess(410.0, None, "OTC-P1")),
    ("FOREST-11P", SensorAccess(410.0, None, "OTC-P1")),
    ("FOREST", SensorAccess(400.0, None, "OroraTech FOREST-3 / 16–19")),
)


# Modules that did ``from fame_geometry import find_observation_opportunities``
# (or ``import *``) at load time. fame_workflow_stochastic re-imports inside
# the function, so patching fame_geometry is enough for it.
_PATCH_MODULE_NAMES = (
    "fame_geometry",
    "fame_agents_base",
    "fame_broker",
    "fame_constellation_scheduler",
    "fame_workflow",
    "riverflow_coobservation",
    "riverflow_utils",
)

_orig_find = None
_installed = False


def min_elevation_from_swath_deg(swath_km: float, altitude_km: float) -> float:
    """Elevation at the edge of a nadir-centred swath of width ``swath_km``."""
    if swath_km <= 0.0 or altitude_km <= 0.0:
        return 90.0
    lam = (swath_km / 2.0) / R_EARTH_KM
    num = math.cos(lam) - R_EARTH_KM / (R_EARTH_KM + altitude_km)
    return math.degrees(math.atan2(num, math.sin(lam)))


def min_elevation_from_for_deg(for_off_nadir_deg: float, altitude_km: float) -> float:
    """Elevation at the maximum off-nadir pointing angle."""
    if altitude_km <= 0.0:
        return 90.0
    s = math.sin(math.radians(for_off_nadir_deg)) * (R_EARTH_KM + altitude_km) / R_EARTH_KM
    if s >= 1.0:
        return 0.0
    return math.degrees(math.acos(s))


def effective_min_elevation_deg(
    swath_km: float,
    altitude_km: float,
    for_off_nadir_deg: Optional[float] = None,
) -> float:
    """Access floor: FOR if the sensor can point, otherwise the swath edge."""
    if for_off_nadir_deg is not None:
        return min_elevation_from_for_deg(for_off_nadir_deg, altitude_km)
    return min_elevation_from_swath_deg(swath_km, altitude_km)


def access_spec_for_name(name: str) -> Optional[SensorAccess]:
    u = name.upper()
    for key, spec in _ACCESS_BY_SUBSTRING:
        if key in u:
            return spec
    return None


def satellite_altitude_km(satellite) -> float:
    semi_major = satellite.orbit.orbit_elements.semi_major_axis * pyorbital.orbital.A
    return float(semi_major - pyorbital.orbital.A)


def for_min_elevation_deg(satellite) -> Optional[float]:
    """Per-satellite floor, or None to leave the request gate unchanged."""
    cached = getattr(satellite, "for_min_elevation_deg", None)
    if cached is not None:
        return float(cached)
    spec = access_spec_for_name(getattr(satellite, "name", ""))
    if spec is None:
        return None
    try:
        h = satellite_altitude_km(satellite)
    except Exception:
        return None
    return effective_min_elevation_deg(spec.swath_km, h, spec.for_off_nadir_deg)


def attach_for_limits(satellites) -> None:
    """Stamp ``for_*`` attributes on river-fleet satellites. No-op for others."""
    for sat in satellites:
        spec = access_spec_for_name(sat.name)
        if spec is None:
            continue
        try:
            h = satellite_altitude_km(sat)
        except Exception:
            continue
        sat.for_swath_km = spec.swath_km
        sat.for_off_nadir_deg = spec.for_off_nadir_deg
        sat.for_min_elevation_deg = effective_min_elevation_deg(
            spec.swath_km, h, spec.for_off_nadir_deg
        )
        sat.for_note = spec.note


def pass_inside_for(obs_pass, satellite) -> bool:
    """True if the peak of ``obs_pass`` clears this satellite's FOR floor."""
    floor = for_min_elevation_deg(satellite)
    if floor is None:
        return True
    highest = obs_pass.highest if hasattr(obs_pass, "highest") else obs_pass
    return float(highest.look_angle_dec_deg) >= floor


FUSION_SAT_NAME = "FUSION-GROUND"
_VIS_BLOCKED = ("FOREST", "SKYBEE", "HOTSAT", FUSION_SAT_NAME)


def _is_fusion_request(request) -> bool:
    if getattr(request, "is_fusion", False):
        return True
    # fame_workflow builds a fresh ObservationRequest ("..._fusion_trimmed")
    # and drops custom attributes. Name is the stable signal.
    name = getattr(request, "name", "") or ""
    return "_fusion" in name


def _is_vis_request(request) -> bool:
    inst = getattr(request, "instrument", None)
    name = getattr(inst, "name", str(inst))
    return (name == "RGB") and not _is_fusion_request(request)


def exclude_secondary_from_vis(opportunities: dict) -> dict:
    """SkyBee VNIR and FOREST-3 RGB must not satisfy the VIS leg."""
    out = {}
    for request, by_sat in opportunities.items():
        if not _is_vis_request(request):
            out[request] = by_sat
            continue
        kept = {}
        for sat, passes in by_sat.items():
            u = sat.name.upper()
            if any(key in u for key in _VIS_BLOCKED):
                continue
            kept[sat] = passes
        out[request] = kept
    return out


def make_fusion_pass(request, satellite):
    """One free, nadir, always-on opportunity after the imaging window.

    Times are staggered by request name so five fusions on one ground node
    do not all collide at the same instant.
    """
    from fame_geometry import ObservationOpportunity, ObservationPass
    name = getattr(request, "name", "") or ""
    slot = sum(ord(c) for c in name) % 8
    t = request.min_time + dt.timedelta(seconds=120 + 90 * slot)
    if request.max_time is not None and t > request.max_time:
        t = request.max_time
    opp_kw = dict(
        time=t,
        lon_deg=request.lon_deg,
        lat_deg=request.lat_deg,
        alt_km=request.alt_km if request.alt_km is not None else 0.0,
        look_angle_az_deg=0.0,
        look_angle_dec_deg=90.0,
        sun_zenith_angle_deg=0.0,
        range_km=1.0,
        name=getattr(request, "name", "fusion"),
        satellite=satellite,
        instrument=request.instrument,
    )
    opp = ObservationOpportunity(**opp_kw)
    return ObservationPass(rise=opp, fall=opp, highest=opp)


def filter_opportunities(opportunities: dict) -> dict:
    """Drop passes that beat the request elevation but miss the instrument FOR.

    Input/output shape matches ``find_observation_opportunities``:
    ``{request: {satellite: [ObservationPass, ...]}}``.
    """
    filtered = {}
    for request, by_sat in opportunities.items():
        kept = {}
        for satellite, passes in by_sat.items():
            ok = [p for p in passes if pass_inside_for(p, satellite)]
            if ok:
                kept[satellite] = ok
        filtered[request] = kept
    return filtered


def _wrapped_find(observation_requests, satellites, passes_error_s=60):
    fusion_reqs = [r for r in observation_requests if _is_fusion_request(r)]
    normal_reqs = [r for r in observation_requests if not _is_fusion_request(r)]
    imaging_sats = [s for s in satellites if s.name != FUSION_SAT_NAME]
    out = {}
    if normal_reqs:
        raw = _orig_find(normal_reqs, imaging_sats, passes_error_s)
        out.update(exclude_secondary_from_vis(filter_opportunities(raw)))
    fusion_sat = next((s for s in satellites if s.name == FUSION_SAT_NAME), None)
    for req in fusion_reqs:
        if fusion_sat is None:
            out[req] = {}
        else:
            out[req] = {fusion_sat: [make_fusion_pass(req, fusion_sat)]}
    return out


def install():
    """Route pass-finding through the FOR filter. River driver only."""
    global _orig_find, _installed
    if _installed:
        return
    import fame_geometry
    _orig_find = fame_geometry.find_observation_opportunities
    fame_geometry.find_observation_opportunities = _wrapped_find
    import sys
    for name in _PATCH_MODULE_NAMES + ("__main__",):
        if name == "fame_geometry":
            continue
        mod = sys.modules.get(name)
        if mod is not None and getattr(mod, "find_observation_opportunities", None) is _orig_find:
            setattr(mod, "find_observation_opportunities", _wrapped_find)
    _installed = True


def uninstall():
    """Put the original finder back. Safe to call if ``install`` was never used."""
    global _orig_find, _installed
    if not _installed:
        return
    import fame_geometry
    fame_geometry.find_observation_opportunities = _orig_find
    import sys
    for name in _PATCH_MODULE_NAMES + ("__main__",):
        if name == "fame_geometry":
            continue
        mod = sys.modules.get(name)
        if mod is not None and getattr(mod, "find_observation_opportunities", None) is _wrapped_find:
            setattr(mod, "find_observation_opportunities", _orig_find)
    _orig_find = None
    _installed = False


def _self_check() -> None:
    """Geometry numbers used in the §6 census. No TLE / network."""
    cases = [
        ("FOREST-16 class", 400.0, 586.0, None, 69.0, 70.0),
        ("OTC-P1", 410.0, 519.0, None, 66.5, 67.0),
        ("SkyBee nadir swath", 17.5, 500.0, None, 88.8, 89.1),
        ("SkyBee ±30° FOR", 17.5, 500.0, 30.0, 57.2, 57.6),
        ("HotSat ±30° FOR", 25.0, 500.0, 30.0, 57.2, 57.6),
    ]
    for label, swath, h, for_deg, lo, hi in cases:
        e = effective_min_elevation_deg(swath, h, for_deg)
        assert lo <= e <= hi, f"{label}: {e:.2f} not in [{lo}, {hi}]"
        print(f"  OK  {label:22}  elev={e:6.2f}°")
    assert access_spec_for_name("OT-FOREST-16 SIWITASGOHT").swath_km == 400.0
    assert access_spec_for_name("OT-FOREST-5P").swath_km == 410.0
    assert access_spec_for_name("SKYBEE-A01").for_off_nadir_deg == 30.0
    assert access_spec_for_name("LANDSAT 8").swath_km == 185.0
    assert access_spec_for_name("SENTINEL 3B").swath_km == 1420.0
    assert access_spec_for_name("ICEYE-X4") is None
    assert access_spec_for_name("SKYSAT-C1") is None
    print("  OK  name table (TIR on, SAR/RGB off)")


if __name__ == "__main__":
    print("[riverflow_for] self-check")
    _self_check()
    print("[riverflow_for] pass")

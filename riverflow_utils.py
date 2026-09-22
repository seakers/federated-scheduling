"""River high-flow co-observation: fan-in workflow, physics, quality.

Three sibling legs (SAR, VIS, TIR) under one AND. Mission value sits on a
fusion task — LogicNode cannot carry reward — so Q_f * W_abs_f prices the
product S_SAR · S_VIS · S_TIR. Legs get partial credit α V_e so greedy is
not degenerate.

Gorr et al. do not model illumination or cloud. Both are first-class here:
VIS needs sun elevation > 10°; VIS/TIR use p_exec = clip(p_geom · (1 − TCC));
SAR is cloud-immune with a high geometry floor. TCC comes from ERA5 when
cached, else a calibrated storm-tied field in the volcano/EQ probability band.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math

from fame_geometry import Location, InstrumentType, ObservationRequest
from fame_agents_base import Phenomenon
from fame_workflow import (
    ConstrainedObservationRequest, Constraint, ConstraintClass,
    TemporalConstraintType, SuccessConstraintType, Workflow,
)
from fame_workflow_stochastic import Lit, And

from riverflow_data_prep import (
    load_cached_events, load_cloud_meta, load_cloud_series, hydrograph_q_over_q75,
)
from riverflow_for import (
    FUSION_SAT_NAME, attach_for_limits,
)

# =============================================================================
# SCENARIO
# =============================================================================

CAMPAIGN_HORIZON_H = 28.0         # 4 h dispatch lead + full UTC day
MIN_ELEVATION_DEG = 20.0          # SAR / VIS request gate (TIR uses FOR)
SUN_ELEVATION_MIN_DEG = 10.0      # VIS only
# Execution band aligned with volcano/earthquake (~0.65-0.95), with modality
# dependence: SAR ignores cloud; VIS/TIR multiply geometry by (1 - TCC).
P_EXEC_GEOM_MIN = 0.65
P_EXEC_GEOM_MAX = 0.95
P_EXEC_SAR_FLOOR = 0.90
P_EXEC_CLIP = (0.55, 0.95)
PARTIAL_CREDIT_ALPHA = 0.12       # 0 ⇒ greedy never books a first leg
VALUE_SET = 80.0                  # scale of a completed set at Q = Q75, w = 1
MIN_EVENT_DURATION_H = 1.0        # keep Gorr short mode (1-2 h)
MIN_RIVER_WIDTH_M = 250.0

# Back-compat alias used by the driver metrics printout.
P_EXEC_SAR = P_EXEC_SAR_FLOOR

# GSD (m) by name fragment. SkyBee vs FOREST is the 7× gap the quality
# function must preserve — do not flatten it.
_GSD_M = (
    ("HOTSAT", 3.5),
    ("SKYBEE", 28.9),
    ("LANDSAT", 100.0),
    ("FOREST", 200.0),
    ("SENTINEL 3", 1000.0),
    ("SKYSAT", 0.8),
    ("PELICAN", 1.2),
    ("FLOCK", 3.7),
    ("TANAGER", 30.0),
    ("ICEYE", 3.0),
    ("UMBRA", 1.0),
    ("CAPELLA", 0.5),
    ("ACADIA", 0.5),
    ("YAM", 10.0),
    ("HAMMER", 10.0),
    ("ACCENTURE", 10.0),
    ("AEROCUBE", 5.0),
    ("PERSISTENCE", 5.0),
)


class RiverFeature(Phenomenon):
    """Typed product for one gauge / one modality."""

    def __init__(self, lon_deg, lat_deg, start_time, end_time, name,
                 gauge_id, kind, river_width_m):
        super().__init__(lon_deg=lon_deg, lat_deg=lat_deg, alt_km=0.0,
                         start_time=start_time, end_time=end_time, name=name)
        self.gauge_id = gauge_id
        self.kind = kind
        self.river_width_m = river_width_m


# =============================================================================
# EVENTS
# =============================================================================

def _parse_iso(s):
    return dt.datetime.fromisoformat(s)


def event_q_at(event, t):
    """Interpolate Q(t) / Q75 from the cached hydrograph."""
    q75 = float(event["Q75"])
    series = event.get("Q") or []
    if not series or q75 <= 0:
        peak = float(event.get("Q_peak_over_Q75") or 1.5)
        t0, t1 = _parse_iso(event["t_start"]), _parse_iso(event["t_end"])
        span = (t1 - t0).total_seconds()
        frac = 0.0 if span <= 0 else (t - t0).total_seconds() / span
        return hydrograph_q_over_q75(frac, peak)
    times = [_parse_iso(row["t"]) for row in series]
    qs = [float(row["Q_cfs"]) / q75 for row in series]
    if t <= times[0]:
        return qs[0]
    if t >= times[-1]:
        return qs[-1]
    for i in range(1, len(times)):
        if t <= times[i]:
            a = (t - times[i - 1]).total_seconds()
            b = (times[i] - times[i - 1]).total_seconds()
            w = 0.0 if b <= 0 else a / b
            return qs[i - 1] + w * (qs[i] - qs[i - 1])
    return qs[-1]


def event_weight(event):
    return math.log(1.0 + max(0.0, float(event["drainage_area_km2"])))


def event_value(event, t, *, clamp=False):
    """V_e(t) = w_e · (Q(t) / Q75). Zero outside the window unless clamped.

    Fusion is scheduled a few minutes after t_end so SUCCESS precedence can
    fire; clamp then uses V_e(t_end) instead of dropping the set to zero.
    """
    t0, t1 = _parse_iso(event["t_start"]), _parse_iso(event["t_end"])
    if clamp:
        if t < t0:
            t = t0
        elif t > t1:
            t = t1
    elif t < t0 or t > t1:
        return 0.0
    return event_weight(event) * event_q_at(event, t)


def load_events(min_duration_h=MIN_EVENT_DURATION_H, min_width_m=MIN_RIVER_WIDTH_M):
    raw = load_cached_events()
    events = []
    for e in raw:
        t0, t1 = _parse_iso(e["t_start"]), _parse_iso(e["t_end"])
        dur_h = (t1 - t0).total_seconds() / 3600.0
        width = float(e.get("river_width_m") or 0.0)
        if dur_h + 1e-9 < min_duration_h:
            continue
        if width + 1e-9 < min_width_m:
            continue
        events.append(e)
    if not events:
        raise ValueError("No river events survived the duration/width filter")
    return events


def events_as_locations(events):
    locs = []
    for e in events:
        loc = Location(e["lon"], e["lat"], 0.0, e["name"])
        loc.gauge_id = e["gauge_id"]
        loc.river_width_m = float(e["river_width_m"])
        loc.drainage_area_km2 = float(e["drainage_area_km2"])
        loc.t_start = _parse_iso(e["t_start"])
        loc.t_end = _parse_iso(e["t_end"])
        loc.value_weight = event_weight(e)
        loc.event = e
        locs.append(loc)
    return locs


# =============================================================================
# CLOUD / ILLUMINATION
# =============================================================================

def _unit_hash(*parts) -> float:
    h = hashlib.md5("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def tcc_at(lat, lon, t, events, meta=None, series=None):
    """Total cloud cover in [0, 1].

    Prefer ERA5 hourly series at the nearest gauge when cached; otherwise a
    storm-tied synthetic field (higher inside an event window).
    """
    meta = meta or load_cloud_meta()
    series = series if series is not None else load_cloud_series()
    if series:
        # Nearest gauge by lat/lon among those with a series.
        best_gid, best_d = None, 1e9
        for e in events:
            gid = e["gauge_id"]
            if gid not in series:
                continue
            d = (e["lat"] - lat) ** 2 + (e["lon"] - lon) ** 2
            if d < best_d:
                best_d, best_gid = d, gid
        if best_gid is not None and series.get(best_gid):
            rows = series[best_gid]
            times = [_parse_iso(r["t"]) for r in rows]
            # Nearest hour.
            best = min(range(len(times)),
                       key=lambda i: abs((times[i] - t).total_seconds()))
            return float(rows[best]["tcc"])

    hour = t.replace(minute=0, second=0, microsecond=0)
    u = _unit_hash(round(lat, 2), round(lon, 2), hour.isoformat())
    in_event = False
    for e in events:
        if abs(e["lat"] - lat) > 0.6 or abs(e["lon"] - lon) > 0.6:
            continue
        if _parse_iso(e["t_start"]) <= t <= _parse_iso(e["t_end"]):
            in_event = True
            break
    if in_event:
        mean, span = float(meta["tcc_event_mean"]), float(meta["tcc_event_span"])
    else:
        mean, span = float(meta["tcc_clear_mean"]), float(meta["tcc_clear_span"])
    return max(0.0, min(1.0, mean + (u - 0.5) * span))


def _look_angle_geom_p(opp) -> float:
    """Nadir → P_EXEC_GEOM_MAX, grazing → P_EXEC_GEOM_MIN (same idea as EQ/volcano)."""
    el = getattr(opp, "look_angle_dec_deg", None)
    if el is None:
        return 0.5 * (P_EXEC_GEOM_MIN + P_EXEC_GEOM_MAX)
    # Elevation 90° = nadir look from sat? In FAME, look_angle_dec_deg is
    # observer elevation of the satellite (0 horizon, 90 zenith). High elev → good.
    t = max(0.0, min(1.0, 1.0 - float(el) / 90.0))
    return P_EXEC_GEOM_MAX - t * (P_EXEC_GEOM_MAX - P_EXEC_GEOM_MIN)


def sun_elevation_deg(t, lon, lat):
    import pyorbital.astronomy
    zenith = pyorbital.astronomy.sun_zenith_angle(t, lon, lat)
    return 90.0 - float(zenith)


def gsd_m_for_satellite(sat_name: str) -> float:
    u = sat_name.upper()
    for key, gsd in _GSD_M:
        if key in u:
            return gsd
    return 20.0


def gsd_quality(sat_name: str, river_width_m: float) -> float:
    """1 at infinitely fine GSD, 0 when the pixel is wider than the river."""
    width = max(1.0, float(river_width_m))
    return max(0.05, 1.0 - gsd_m_for_satellite(sat_name) / width)


# =============================================================================
# FLEET HOOKS (river-only; do not touch the shared loader)
# =============================================================================

def enable_secondary_payloads(satellites):
    """Register SkyBee VNIR / FOREST-3 RGB on the bus, not for the VIS leg."""
    for sat in satellites:
        u = sat.name.upper()
        if not (("FOREST-3" in u) or ("SKYBEE" in u)):
            continue
        if InstrumentType.RGB in sat.instruments:
            continue
        sat.instruments = list(sat.instruments) + [InstrumentType.RGB]
        if sat.instrument_fov_rad:
            tir_fov = next(iter(sat.instrument_fov_rad.values()))
            sat.instrument_fov_rad[InstrumentType.RGB] = tir_fov


def make_fusion_satellite(template_orbit):
    from fame_geometry import Satellite
    sat = Satellite(
        FUSION_SAT_NAME, template_orbit,
        instruments=[InstrumentType.RGB],
        has_continuous_isl_to_ground=True,
    )
    sat.instrument_fov_rad = {InstrumentType.RGB: math.pi}
    sat.for_min_elevation_deg = 0.0
    return sat


# Extra TIR that the shared loader does not carry. Landsat is the quality
# increment (100 m); Sentinel-3 is the access increment (1420 km swath).
# Loaded only by the river driver.
_EXTRA_TIR = (
    ("LANDSAT 8", 185.0),
    ("LANDSAT 9", 185.0),
    ("SENTINEL 3A", 1420.0),
    ("SENTINEL 3B", 1420.0),
)


def load_extra_tir_satellites(sim_start, horizon_h, tle_file=None):
    """Append Landsat 8/9 + Sentinel-3A/B. No-op if a TLE is missing."""
    import glob
    import numpy as np
    import pyorbital.orbital
    from pyorbital.orbital import Orbital
    from fame_geometry import Satellite

    if tle_file is None:
        tle_files = sorted(glob.glob("tles/all_tles_*.txt"))
        if not tle_files:
            return []
        tle_file = tle_files[-1]
    min_time = sim_start
    max_time = sim_start + dt.timedelta(hours=horizon_h)
    out = []
    for name, swath_km in _EXTRA_TIR:
        try:
            orbit = Orbital(name, tle_file=tle_file)
            _ = orbit.get_lonlatalt(min_time)
            _ = orbit.get_lonlatalt(max_time)
            sat = Satellite(name, orbit, instruments=[InstrumentType.TIR],
                            has_continuous_isl_to_ground=True)
            semi_major = sat.orbit.orbit_elements.semi_major_axis * pyorbital.orbital.A
            altitude = semi_major - pyorbital.orbital.A
            fov = 2 * np.atan2(swath_km / 2, max(altitude, 1.0))
            sat.instrument_fov_rad = {InstrumentType.TIR: fov}
            out.append(sat)
        except Exception:
            continue
    print(f"[Fleet] extra TIR {len(out)}/{len(_EXTRA_TIR)} "
          f"({', '.join(s.name for s in out) or 'none'})")
    return out


def prepare_river_fleet(satellites):
    attach_for_limits(satellites)
    enable_secondary_payloads(satellites)
    return satellites


# =============================================================================
# PHENOMENA
# =============================================================================

def register_river_phenomena(world, events, min_time, max_time):
    for e in events:
        t0, t1 = _parse_iso(e["t_start"]), _parse_iso(e["t_end"])
        for kind in ("sar", "vis", "tir", "fusion"):
            feat = RiverFeature(
                lon_deg=e["lon"], lat_deg=e["lat"],
                start_time=t0, end_time=t1,
                name=f"{e['gauge_id']}_{kind}",
                gauge_id=e["gauge_id"], kind=kind,
                river_width_m=float(e["river_width_m"]),
            )
            world.phenomena.append(feat)


def _flatten_product(data_product):
    if data_product is None:
        return []
    if not isinstance(data_product, (list, tuple, set)):
        return [data_product]
    out = []
    for item in data_product:
        if isinstance(item, (list, tuple, set)):
            out.extend(_flatten_product(item))
        elif item is not None:
            out.append(item)
    return out


def _product_processor(gauge_id, kind):
    def _processor(_observation, _spacecraft, phenomena):
        kept = [
            item for item in _flatten_product(phenomena)
            if isinstance(item, RiverFeature)
            and item.gauge_id == gauge_id
            and item.kind == kind
        ]
        return kept
    return _processor


def _success_for(gauge_id, kind):
    def _success(data_product):
        return any(
            isinstance(item, RiverFeature)
            and item.gauge_id == gauge_id
            and item.kind == kind
            for item in _flatten_product(data_product)
        )
    return _success


# =============================================================================
# REWARD / FEASIBILITY
# =============================================================================

def _rewarder(event, kind):
    """Leg: α V_e(t) q_gsd. Fusion: V_e(t) (set-completion value)."""
    width = float(event["river_width_m"])
    scale = PARTIAL_CREDIT_ALPHA if kind != "fusion" else 1.0

    def _r(op, preferred_zenith_angle_deg=None):
        sat_name = getattr(getattr(op, "satellite", None), "name", "") or ""
        if kind == "fusion":
            q = 1.0
        else:
            q = gsd_quality(sat_name, width)
        return VALUE_SET * scale * event_value(event, op.time, clamp=(kind == "fusion")) * q
    return _r


def _vis_feasible(satellite, obs_pass):
    u = satellite.name.upper()
    if any(k in u for k in ("FOREST", "SKYBEE", "HOTSAT", "FUSION")):
        return False
    highest = obs_pass.highest if hasattr(obs_pass, "highest") else obs_pass
    return sun_elevation_deg(highest.time, highest.lon_deg, highest.lat_deg) >= SUN_ELEVATION_MIN_DEG


def _req(event, name, instrument, min_elevation=MIN_ELEVATION_DEG):
    return ObservationRequest(
        lon_deg=event["lon"], lat_deg=event["lat"], alt_km=0.0,
        min_time=_parse_iso(event["t_start"]),
        max_time=_parse_iso(event["t_end"]),
        instrument=instrument,
        request_name=name,
        min_elevation_deg=min_elevation,
    )


# =============================================================================
# WORKFLOW
# =============================================================================

def create_riverflow_workflow(events, min_time, max_time, max_num_instances=3):
    """One AND fan-in per event: SAR, VIS, TIR siblings + fusion child.

    Legs have no parents. Fusion has And(Lit(sar), Lit(vis), Lit(tir)) and
    SUCCESS edges on all three. Do not chain the legs — that would assert a
    false precedence and destroy the fan-in.
    """
    horizon_h = (max_time - min_time).total_seconds() / 3600.0
    if horizon_h + 1e-9 < CAMPAIGN_HORIZON_H:
        raise ValueError(
            f"River scenario needs ≥ {CAMPAIGN_HORIZON_H:g} h; got {horizon_h:g} h"
        )

    policy_schedule = {c: True for c in ConstraintClass}
    policy_dispatch = {
        ConstraintClass.TEMPORAL: True,
        ConstraintClass.SUCCESS: False,
        ConstraintClass.GEOMETRY: True,
    }

    tasks = []

    def _add(task, event, kind):
        task.rf_meta = (event["gauge_id"], kind)
        task.enforce_success_precedence = (kind == "fusion")
        tasks.append(task)

    for e in events:
        gid = e["gauge_id"]
        t0, t1 = _parse_iso(e["t_start"]), _parse_iso(e["t_end"])

        sar = ConstrainedObservationRequest(
            name=f"{gid}_sar",
            observation_request=_req(e, f"{gid}_sar", InstrumentType.SAR),
            is_mandatory=True,
            schedule_policy_if_constraint_unsatisfied=policy_schedule,
            dispatch_policy_if_constraint_unsatisfied=policy_dispatch,
            rewarder=_rewarder(e, "sar"),
            success_declarer=_success_for(gid, "sar"),
            phenomenon_processor=_product_processor(gid, "sar"),
            request_group=f"{gid}_sar",
            max_num_instances=max_num_instances,
        )
        sar.gate = None
        _add(sar, e, "sar")

        vis = ConstrainedObservationRequest(
            name=f"{gid}_vis",
            observation_request=_req(e, f"{gid}_vis", InstrumentType.RGB),
            is_mandatory=True,
            schedule_policy_if_constraint_unsatisfied=policy_schedule,
            dispatch_policy_if_constraint_unsatisfied=policy_dispatch,
            rewarder=_rewarder(e, "vis"),
            success_declarer=_success_for(gid, "vis"),
            phenomenon_processor=_product_processor(gid, "vis"),
            request_group=f"{gid}_vis",
            max_num_instances=max_num_instances,
        )
        vis.gate = None
        vis.opportunity_feasibility = _vis_feasible
        _add(vis, e, "vis")

        tir = ConstrainedObservationRequest(
            name=f"{gid}_tir",
            observation_request=_req(e, f"{gid}_tir", InstrumentType.TIR),
            is_mandatory=True,
            schedule_policy_if_constraint_unsatisfied=policy_schedule,
            dispatch_policy_if_constraint_unsatisfied=policy_dispatch,
            rewarder=_rewarder(e, "tir"),
            success_declarer=_success_for(gid, "tir"),
            phenomenon_processor=_product_processor(gid, "tir"),
            request_group=f"{gid}_tir",
            max_num_instances=max_num_instances,
        )
        tir.gate = None
        _add(tir, e, "tir")

        fusion_req = ObservationRequest(
            lon_deg=e["lon"], lat_deg=e["lat"], alt_km=0.0,
            min_time=t1,
            max_time=t1 + dt.timedelta(minutes=30),
            instrument=InstrumentType.RGB,
            request_name=f"{gid}_fusion",
            min_elevation_deg=0.0,
        )
        fusion_req.is_fusion = True
        fusion = ConstrainedObservationRequest(
            name=f"{gid}_fusion",
            observation_request=fusion_req,
            is_mandatory=True,
            task_constraints=[
                Constraint(ConstraintClass.SUCCESS,
                           SuccessConstraintType.START_IF_SUCCESSFUL, sar),
                Constraint(ConstraintClass.SUCCESS,
                           SuccessConstraintType.START_IF_SUCCESSFUL, vis),
                Constraint(ConstraintClass.SUCCESS,
                           SuccessConstraintType.START_IF_SUCCESSFUL, tir),
            ],
            schedule_policy_if_constraint_unsatisfied=policy_schedule,
            dispatch_policy_if_constraint_unsatisfied=policy_dispatch,
            rewarder=_rewarder(e, "fusion"),
            success_declarer=_success_for(gid, "fusion"),
            phenomenon_processor=_product_processor(gid, "fusion"),
            request_group=f"{gid}_fusion",
            max_num_instances=1,
        )
        fusion.gate = And(Lit(sar), Lit(vis), Lit(tir))
        fusion.success_constraint_mode = "all"
        _add(fusion, e, "fusion")

    return Workflow(
        constrained_observation_requests=tasks,
        timelines=[],
        timeline_updater=lambda t, r, tl: tl,
        request_updater=lambda t, r, tl: r,
    )


def compute_river_reachability(tasks, completed_tasks, no_pass_tasks=None):
    """Reachable set: every leaf, plus fusion only if all three legs completed
    or still have a chance (not in no_pass_tasks).
    """
    no_pass = set(no_pass_tasks or [])
    by_gauge = {}
    for task in tasks:
        gid, kind = getattr(task, "rf_meta", (None, None))
        if gid is None:
            continue
        by_gauge.setdefault(gid, {})[kind] = task

    completed_kinds = {}
    for task in completed_tasks:
        gid, kind = getattr(task, "rf_meta", (None, None))
        if gid is None:
            continue
        completed_kinds.setdefault(gid, set()).add(kind)

    reachable = set()
    for gid, kinds in by_gauge.items():
        for kind, task in kinds.items():
            req_name = task.observation_request.name
            if req_name in no_pass and kind != "fusion":
                continue
            if kind == "fusion":
                legs = {k for k in ("sar", "vis", "tir") if k in kinds}
                have = completed_kinds.get(gid, set())
                missing = legs - have
                # Fusion is unreachable if a missing leg has no passes.
                blocked = False
                for mk in missing:
                    mt = kinds.get(mk)
                    if mt is not None and mt.observation_request.name in no_pass:
                        blocked = True
                if blocked:
                    continue
            reachable.add(task)
    return reachable


# =============================================================================
# PHYSICS CALLBACKS
# =============================================================================

def make_execution_prob_function(events):
    meta = load_cloud_meta()
    series = load_cloud_series()

    def execution_prob_function(constrained_request, satellite, obs_opp):
        # Constellation execution calls this as f(None, sat, opportunity).
        sat_name = getattr(satellite, "name", "") or ""
        if sat_name.startswith(FUSION_SAT_NAME):
            return 1.0
        kind = None
        if constrained_request is not None:
            kind = getattr(constrained_request, "rf_meta", (None, None))[1]
        if kind == "fusion":
            return 1.0
        opp = obs_opp.highest if hasattr(obs_opp, "highest") else obs_opp
        inst = None
        if constrained_request is not None:
            inst = getattr(getattr(constrained_request, "observation_request", None),
                           "instrument", None)
        if inst is None:
            inst = getattr(opp, "instrument", None)

        p_geom = _look_angle_geom_p(opp)
        # SAR: cloud-immune; slight geometry dependence, floored high.
        if inst == InstrumentType.SAR or not getattr(inst, "cloud_sensitive", True):
            return max(P_EXEC_SAR_FLOOR, min(P_EXEC_GEOM_MAX, p_geom))

        tcc = tcc_at(opp.lat_deg, opp.lon_deg, opp.time, events, meta, series)
        p_clear = 1.0 - tcc
        lo, hi = P_EXEC_CLIP
        return max(lo, min(hi, p_geom * p_clear))

    return execution_prob_function


def is_fusion_task(task) -> bool:
    return getattr(task, "rf_meta", (None, None))[1] == "fusion"


# =============================================================================
# CENSUS
# =============================================================================

def opportunity_census(events, satellites, find_fn):
    """Per-event pass counts after FOR, sun, and secondary-payload filters."""
    rows = []
    for e in events:
        counts = {"sar": 0, "vis": 0, "tir": 0}
        distinct = {"sar": 0, "vis": 0, "tir": 0}
        for kind, inst in (("sar", InstrumentType.SAR),
                           ("vis", InstrumentType.RGB),
                           ("tir", InstrumentType.TIR)):
            req = _req(e, f"census_{e['gauge_id']}_{kind}", inst)
            if kind == "vis":
                # sun filter applied below
                pass
            opps = find_fn([req], satellites)
            times = []
            for sat, passes in opps.get(req, {}).items():
                for p in passes:
                    if kind == "vis" and not _vis_feasible(sat, p):
                        continue
                    counts[kind] += 1
                    times.append(p.highest.time)
            times.sort()
            n_dist, last = 0, None
            for t in times:
                if last is None or (t - last).total_seconds() > 3600:
                    n_dist += 1
                    last = t
            distinct[kind] = n_dist
        p_clear = []
        t0, t1 = _parse_iso(e["t_start"]), _parse_iso(e["t_end"])
        t = t0
        while t <= t1:
            p_clear.append(1.0 - tcc_at(e["lat"], e["lon"], t, events))
            t += dt.timedelta(hours=1)
        rows.append({
            "gauge": e["name"],
            "width_m": e["river_width_m"],
            "n_sar": counts["sar"], "n_vis": counts["vis"], "n_tir": counts["tir"],
            "distinct_tir": distinct["tir"],
            "p_clear_mean": sum(p_clear) / len(p_clear) if p_clear else 0.0,
        })
    return rows


def print_census_gate(rows):
    print("\n" + "=" * 72)
    print("RIVERFLOW CENSUS GATE")
    print("=" * 72)
    print(f"  {'gauge':32} {'SAR':>4} {'VIS':>4} {'TIR':>4} {'TIR>1h':>7} {'p_clear':>8}")
    for r in rows:
        print(f"  {r['gauge'][:32]:32} {r['n_sar']:4d} {r['n_vis']:4d} {r['n_tir']:4d} "
              f"{r['distinct_tir']:7d} {r['p_clear_mean']:8.2f}")
    n = len(rows) or 1
    mean_tir = sum(r["distinct_tir"] for r in rows) / n
    mean_tir_raw = sum(r["n_tir"] for r in rows) / n
    mean_vis = sum(r["n_vis"] for r in rows) / n
    mean_pc = sum(r["p_clear_mean"] for r in rows) / n
    min_tir = min(r["distinct_tir"] for r in rows)
    print(f"  mean distinct TIR/event = {mean_tir:.1f} (raw {mean_tir_raw:.1f}; need ~2-5; min {min_tir})")
    print(f"  mean VIS/event = {mean_vis:.1f}  (must stay above TIR scarcity)")
    print(f"  mean p_clear during events = {mean_pc:.2f}  (must not be ~0)")
    fails = []
    if mean_tir < 1.5:
        fails.append("distinct TIR opportunities < 1.5 - no scheduling decision on the binding leg")
    if mean_pc < 0.08:
        fails.append("p_clear near zero - every scheduler ties at zero completions")
    if mean_vis <= mean_tir * 0.8:
        fails.append("VIS supply collapsed to TIR levels - equal scarcity kills the result")
    if fails:
        print("  GATE FAILED:")
        for f in fails:
            print(f"    - {f}")
        return False
    print("  GATE PASSED")
    return True

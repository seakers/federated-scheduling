"""
Post-Earthquake Damage Assessment -- workflow utilities.

WHAT THIS CASE STUDY IS FOR
===========================
To show that the general AND/OR/NOT formulation buys something a pure-AND
reduction and a greedy baseline cannot get. That needs exactly four ingredients,
and this workflow is built to supply them and nothing else:

    1. an AND cascade   -- a mandatory root whose worth is mostly downstream
    2. an OR            -- two substitutable parents
    4. scarcity         -- more settlements than the fleet can fully serve, so
                           allocation is a real decision

THE SCENARIO, IN TWO SENTENCES
==============================
One large earthquake damages many towns. For each town you want, in order: a
regional survey to find where the damage concentrated, then imagery of the
affected area (optical OR radar -- either works), then a close-in damage grading
of the worst district, and finally either a very-high-resolution follow-up (only
if the optical characterisation failed) or a road-network check.

Later products need earlier ones because until the earlier product lands you do
not know where to point.

PHASES  (offsets relative to the REALISED parent acquisition)
============================================================
    PHASE 0   EXTENT     SAR, MANDATORY, [0, 8h]         aim: REGION   (+-30 km)
              Wide-area coherence-change over the region around one settlement.
              Identifies which built-up area actually took the damage.

    PHASE 1   URBAN_opt  optical, [ext+1h,  ext+12h]     aim: URBAN AREA (+-8 km)
              URBAN_sar  SAR,     [ext+1h,  ext+12h]     aim: URBAN AREA
              Characterise the affected area. Two modalities, either suffices:
              SAR sees through cloud and at night, optical resolves structures.

    PHASE 2   TRIAGE     optical, [ext+6h,  ext+24h]     aim: DISTRICT (+-2 km)
              Building-level damage grading; drives relief allocation.

    PHASE 3   HIRES      optical, [ext+12h, ext+48h]     aim: DISTRICT
              Very-high-resolution TASKING -- expensive, and only worth paying
              for when the cheap optical characterisation failed.
              ACCESS     SAR,     [ext+12h, ext+48h]     aim: DISTRICT
              Route and infrastructure assessment for ground logistics.

Six tasks per settlement.

The three aim points are the reason the phases must run in order: each is
genuinely unknown until its parent's product arrives, so the dispatcher blocks on
SUCCESS as a PHYSICAL constraint rather than a modelling convention.

Everything closes inside 72 h. Search-and-rescue survival falls sharply past that
point, the International Charter targets first products within 24 h, and EMS
Rapid Mapping delivers delineation inside roughly a day, so mission value DECAYS
with acquisition age (see _timeliness). That decay also gives the horizontal
track something to work with: on geometry alone, quality varies by about a factor
of two across a task's candidate passes, and the descending-quality ordering
Q_{r,1} >= Q_{r,2} >= ... has very little to exploit.

WHY THE GENERAL FORMULATION IS REQUIRED
=======================================
OR at TRIAGE      gate = Or(Lit(URBAN_opt), Lit(URBAN_sar))
    The modalities are genuine SUBSTITUTES. An AND reduction must either demand
    BOTH -- paying for two sensors when one suffices, and losing triage whenever
    either fails -- or DROP the prerequisite, crediting triage with no imagery at
    all. Neither is the real constraint.

NOT at HIRES      gate = And(Lit(TRIAGE), Not(Lit(URBAN_opt)))
    VHR tasking is bought commercially and is expensive. Disaster response works
    this way in practice: free and cheap sources first, paid VHR only for the
    gaps. So URBAN_opt and HIRES are ANTI-complementary -- every extra pass on
    URBAN_opt REDUCES the expected value of HIRES. A planner that cannot
    represent negation either over-books HIRES or ignores the recovery path.
    (HIRES therefore carries a higher execution cost; set that in the driver.)

CASCADE at EXTENT
    EXTENT is mandatory and gates five downstream tasks, so its marginal value is
    dS_EXTENT * sum(downstream value) -- an order of magnitude above its own
    reward. The objective is supermodular in EXTENT bookings; greedy sees only
    the local term and under-invests. This effect exists under AND too; the two
    gates above are what the AND reduction cannot reach.

DATA
====
USGS ComCat (earthquake.usgs.gov FDSN event service): free, no key, one GET
returns lat/lon/time/magnitude/depth as GeoJSON. Settlements come from a
worldcities-style CSV. ONE event is chosen -- the one with the highest exposed
population inside its own damage radius -- and the K most-exposed settlements
within that radius become the targets. That is the shape a real activation takes:
the 2023 Kahramanmaras M7.8 damaged eleven provinces and Copernicus EMS delivered
separate products for dozens of towns from a single activation.

The intensity prediction equation below is a STANDARD FORM with ILLUSTRATIVE
coefficients, not fitted values. It exists to answer one question -- which towns
did this event affect, and how badly -- reproducibly. Before publication,
substitute published coefficients (Allen, Wald & Worden 2012 for active crustal
regions) and say which you used.
"""

import csv
import datetime as dt
import json
import math
import os
import random
import urllib.request

from fame_geometry import Location, InstrumentType, ObservationRequest
from fame_agents_base import Phenomenon
from fame_workflow import (
    ConstrainedObservationRequest, Constraint, ConstraintClass,
    TemporalConstraintType, SuccessConstraintType, Workflow,
)
from fame_workflow_stochastic import Lit, And, Or, Not, LogicNode


class EarthquakeFeature(Phenomenon):
    """Persistent, typed evidence returned by an earthquake observation.

    ``phase`` identifies the spatial product represented by the feature.  A
    feature may also carry the next aim point revealed by that product.  The
    coordinates remain hidden inside the simulated world until a successful
    parent observation returns the feature in its data product.
    """

    def __init__(self, *, lon_deg, lat_deg, start_time, end_time, name,
                 settlement, phase, evidence_strength=1.0,
                 reveals_phase=None, reveals_lat_deg=None,
                 reveals_lon_deg=None):
        super().__init__(
            lon_deg=lon_deg,
            lat_deg=lat_deg,
            alt_km=0.0,
            start_time=start_time,
            end_time=end_time,
            name=name,
        )
        self.settlement = str(settlement)
        self.phase = str(phase)
        self.evidence_strength = float(evidence_strength)
        self.reveals_phase = reveals_phase
        self.reveals_lat_deg = (
            None if reveals_lat_deg is None else float(reveals_lat_deg)
        )
        self.reveals_lon_deg = (
            None if reveals_lon_deg is None else float(reveals_lon_deg)
        )


# =============================================================================
# TIMING  -- 24-hour response campaign
# =============================================================================
CAMPAIGN_HORIZON_H = 24.0
EXTENT_WINDOW_H = 6.0

# A phase may open as soon as its immediate SUCCESS predecessor completes.
# The close offsets remain EXTENT-relative campaign deadlines.
URBAN_OPEN_H, URBAN_CLOSE_H = 0.0, 4.5
TRIAGE_OPEN_H, TRIAGE_CLOSE_H = 0.0, 8.25
FINAL_OPEN_H, FINAL_CLOSE_H = 0.0, 12.0

# Usable-GSD floor. At 15 deg the planner books 1500 km slant ranges, which is
# not a product anyone would grade buildings from; at 25 deg a mandatory EXTENT
# can starve. 20 is the compromise.
#
# This and EXTENT_WINDOW_H control pass availability. A mandatory EXTENT with no
# feasible pass makes every downstream task unreachable.
MIN_ELEVATION_DEG = 20.0

# Value decay with acquisition age, floored: a 48 h product is worth much less
# than a 6 h one but is not worthless, and a term that reached zero would make
# PHASE 3 unbookable regardless of geometry.
DECAY_TAU_H      = 24.0
TIMELINESS_FLOOR = 0.25

# =============================================================================
# HAZARD  (illustrative coefficients -- see module docstring)
# =============================================================================
# I = C0 + C1*M - C2*ln(sqrt(R^2 + h^2)). Calibrated by eye so M7.8 at 10 km
# gives ~IX and at 100 km ~VII; M6.0 at 10 km gives ~VII.
IPE_C0, IPE_C1, IPE_C2 = 1.0, 1.5, 1.2
IPE_H_KM               = 10.0
# MMI V ("felt by nearly all, slight damage") is the INCLUSION threshold, not the
# onset of collapse. It is deliberately generous: an EMS activation AOI extends
# well past the towns that were flattened, and the marginal towns are exactly the
# ones the planner should be deciding whether to drop under scarcity. At MMI VI
# the radius is 26 km for M6.0 and 51 km for M6.5 -- smaller than the spacing
# between provincial capitals, which forces K = 1 and removes the allocation
# decision entirely.
DAMAGE_INTENSITY_MMI   = 5.0

# Beyond this, an event is not plausibly the reason you are tasking a town.
TARGET_SEARCH_CAP_KM   = 400.0

# Site effects and building stock: two towns at the same distance do not fare the
# same. Seeded, so every scheduler in a paired comparison faces one world.
SITE_SIGMA = 0.30

# Positional uncertainty per phase -- the reason the phases must run in order.
REGION_UNCERTAINTY_KM   = 30.0
URBAN_UNCERTAINTY_KM    = 8.0
DISTRICT_UNCERTAINTY_KM = 2.0

# Base mission values. TRIAGE drives relief allocation and is the most valuable
# single product; EXTENT is cheap but gates everything, so its worth is mostly
# INDIRECT -- which is precisely what greedy cannot see.
VALUE_EXTENT    = 60.0
VALUE_URBAN_OPT = 90.0
VALUE_URBAN_SAR = 70.0        # coarser: no building-level detail
VALUE_TRIAGE    = 130.0
VALUE_HIRES     = 100.0
VALUE_ACCESS    = 80.0

# Rewarder geometry weights (each group sums to 1.0)
W_LOOK, W_ILLUM, W_RANGE = 0.35, 0.25, 0.40
W_LOOK_SAR, W_RANGE_SAR  = 0.45, 0.55
RANGE_REF_KM = 400.0
PREFERRED_SUN_ZENITH_DEG = 45.0

USGS_QUERY_URL = (
    "https://earthquake.usgs.gov/fdsnws/event/1/query"
    "?format=geojson&minmagnitude={minmag}&starttime={start}&endtime={end}"
    "&orderby=magnitude"
)


# =============================================================================
# HELPERS
# =============================================================================

def _km_between(lat1, lon1, lat2, lon2):
    dlat = (lat2 - lat1) * 111.0
    dlon = (lon2 - lon1) * 111.0 * math.cos(math.radians(0.5 * (lat1 + lat2)))
    return math.hypot(dlat, dlon)


def _offset_latlon(lat0, lon0, north_km, east_km):
    lat = lat0 + north_km / 111.0
    lon = lon0 + east_km / (111.0 * max(0.2, math.cos(math.radians(lat0))))
    return lat, lon


def _scatter_point(rng, lat0, lon0, radius_km):
    """A point at a seeded bearing and radius from (lat0, lon0)."""
    b = rng.uniform(0.0, 2.0 * math.pi)
    return _offset_latlon(lat0, lon0, radius_km * math.cos(b), radius_km * math.sin(b))


def intensity_mmi(magnitude, distance_km, depth_km=IPE_H_KM):
    """Macroseismic intensity at a site. Illustrative IPE -- see docstring."""
    h = max(1.0, float(depth_km))
    return IPE_C0 + IPE_C1 * magnitude - IPE_C2 * math.log(
        math.sqrt(distance_km ** 2 + h ** 2))


def damage_radius_km(magnitude, depth_km=IPE_H_KM, threshold_mmi=DAMAGE_INTENSITY_MMI):
    """Distance at which intensity falls to the damage threshold.

    Replaces a fixed search radius: a M6.0 damages a ~25 km radius, a M7.8 a
    ~265 km radius. One number for both is what forced earlier drafts to pair
    unrelated events with distant cities.
    """
    r = math.exp((IPE_C0 + IPE_C1 * magnitude - threshold_mmi) / IPE_C2)
    h = max(1.0, float(depth_km))
    return math.sqrt(max(0.0, r ** 2 - h ** 2))


def _severity_from_intensity(mmi):
    """Damage severity in [0, 1]: 0 below the damage threshold, saturating near
    MMI X. Monotone and smooth, so the allocation ordering is stable."""
    return max(0.0, min(1.0, (mmi - DAMAGE_INTENSITY_MMI) / (10.0 - DAMAGE_INTENSITY_MMI)))


# =============================================================================
# DATA LOADING
# =============================================================================

def fetch_usgs_events(start, end, min_magnitude=6.0, cache_path=None):
    """Earthquake events from USGS ComCat, cached to disk.

    Cached because an experiment whose targets change between runs is not a
    controlled comparison.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    url = USGS_QUERY_URL.format(minmag=min_magnitude,
                                start=start.strftime("%Y-%m-%d"),
                                end=end.strftime("%Y-%m-%d"))
    print(f"[USGS] {url}")
    with urllib.request.urlopen(url, timeout=60) as resp:
        payload = json.load(resp)

    events = []
    for feat in payload.get("features", []):
        p = feat.get("properties", {})
        coords = (feat.get("geometry") or {}).get("coordinates") or []
        if len(coords) < 2 or p.get("time") is None:
            continue
        events.append({
            "id": feat.get("id"),
            "place": p.get("place", "unknown"),
            "magnitude": float(p.get("mag") or 0.0),
            "lon_deg": float(coords[0]),
            "lat_deg": float(coords[1]),
            "depth_km": float(coords[2]) if len(coords) > 2 else 10.0,
            "time": dt.datetime.utcfromtimestamp(p["time"] / 1000.0).isoformat(),
        })
    print(f"[USGS] {len(events)} event(s) M>={min_magnitude}")
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(events, f, indent=2)
    return events


def load_cities(cities_csv, min_population=50000):
    """Populated places from a worldcities-style CSV.

    NOTE the default: 50k, not 100k. At 100k most rupture zones yield only two or
    three qualifying towns, and with K small there is no allocation decision left
    for the planner to make.
    """
    cities = []
    with open(cities_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                pop = float(row.get("population") or 0)
                if pop < min_population:
                    continue
                cities.append({
                    "name": row.get("city_ascii") or row.get("city") or "city",
                    "lat_deg": float(row["lat"]),
                    "lon_deg": float(row["lng"]),
                    "population": pop,
                })
            except (KeyError, ValueError, TypeError):
                continue
    return cities


# =============================================================================
# TARGET SELECTION  --  ONE EVENT, K SETTLEMENTS
# =============================================================================

def _sanitise(name):
    return "".join(ch for ch in name if ch.isalnum()) or "site"


def select_targets(events, cities, max_targets=10, max_distance_km=None,
                   min_magnitude=6.0, seed=0):
    """The K most-exposed settlements affected by the SINGLE worst event.

    Signature unchanged so the driver needs no edit, but the semantics changed:
    `max_targets` is the number of SETTLEMENTS, not of events, and
    `max_distance_km` is ignored unless given -- the damage radius comes from the
    magnitude.

    Each returned Location is a SETTLEMENT whose OWN position is the PHASE 0
    (regional) aim point, and which carries:
        ev_*                    shared event parameters
        distance_km, mmi        site hazard
        population, severity    exposure inputs
        exposure, exposure_share  what the planner allocates against
        urban_*                 PHASE 1 aim point, revealed by EXTENT
        district_*              PHASE 2/3 aim point, revealed by PHASE 1

    Everything random is drawn HERE from a scheduler-independent seed, so every
    scheduler in a paired comparison faces an identical world.
    """
    # Pick the event by EXPOSED POPULATION inside its own damage radius, not by
    # magnitude: a M7.5 in open desert is a less interesting tasking problem than
    # a M6.5 under a city.
    best, candidates = None, []
    for ev in events:
        if ev["magnitude"] < min_magnitude:
            continue
        radius = max_distance_km or min(TARGET_SEARCH_CAP_KM,
                                        damage_radius_km(ev["magnitude"], ev["depth_km"]))
        exposed, hit = 0.0, []
        for c in cities:
            d = _km_between(ev["lat_deg"], ev["lon_deg"], c["lat_deg"], c["lon_deg"])
            if d > radius:
                continue
            mmi = intensity_mmi(ev["magnitude"], d, ev["depth_km"])
            sev = _severity_from_intensity(mmi)
            if sev <= 0.0:
                continue
            exposed += c["population"] * sev
            hit.append((c, d, mmi, sev))
        if not hit:
            continue
        candidates.append((exposed, ev, radius, len(hit)))
        if best is None or exposed > best[0]:
            best = (exposed, ev, radius, hit)

    # Show the runners-up. K is set by the EVENT, not by max_targets, so when the
    # scenario comes out too small this is the line that says why: either every
    # candidate has a small radius (raise min_magnitude) or every candidate has
    # few towns inside it (lower min_population in load_cities).
    if candidates:
        candidates.sort(key=lambda c: -c[0])
        print("[Targets] Top candidate events (settlements inside their own radius):")
        for exposed, ev, radius, n in candidates[:3]:
            print(f"    M{ev['magnitude']:.1f} {str(ev.get('place',''))[:34]:34s} "
                  f"r={radius:5.0f} km  n={n:3d}  exposed={exposed:12,.0f}")

    if best is None:
        print("[Targets] No event with an exposed population found. "
              "Lower min_population, or min_magnitude, or widen the USGS date range.")
        return []

    _exposed, ev, radius, hit = best
    n_affected = len(hit)

    rng = random.Random(seed)
    scored = []
    for city, dist, mmi, sev_base in hit:
        # Site effects / building stock: two towns at the same distance do not
        # fare the same. Static -- the planner knows it; nothing is revealed
        # later. (An earlier draft made this a belief that observations updated.
        # It was a nice story and pure cost; the paper's claim is about gate
        # structure, not adaptive value.)
        sev = max(0.0, min(1.0, sev_base * math.exp(rng.gauss(0.0, SITE_SIGMA))))
        scored.append((city["population"] * sev, city, dist, mmi, sev))

    scored.sort(key=lambda s: -s[0])
    scored = scored[:max_targets]

    targets = []
    for exposure, city, dist, mmi, sev in scored:
        loc = Location(city["lon_deg"], city["lat_deg"], 0.0, _sanitise(city["name"]))
        loc.settlement = city["name"]
        loc.population = city["population"]
        loc.distance_km = dist
        loc.mmi = mmi
        loc.severity = sev
        loc.exposure = exposure

        loc.ev_id = ev.get("id")
        loc.ev_lat_deg = ev["lat_deg"]
        loc.ev_lon_deg = ev["lon_deg"]
        loc.ev_magnitude = ev["magnitude"]
        loc.ev_depth_km = ev["depth_km"]
        loc.ev_time = ev["time"]
        loc.ev_place = ev.get("place", "unknown")
        loc.ev_radius_km = radius

        # PHASE 1 aim point: the built-up area that actually took the damage,
        # somewhere inside the region EXTENT surveys. Unknown until EXTENT lands.
        loc.urban_lat_deg, loc.urban_lon_deg = _scatter_point(
            rng, city["lat_deg"], city["lon_deg"], 0.5 * REGION_UNCERTAINTY_KM)
        # PHASE 2/3 aim point: the worst-hit district inside that area. Unknown
        # until PHASE 1 lands.
        loc.district_lat_deg, loc.district_lon_deg = _scatter_point(
            rng, loc.urban_lat_deg, loc.urban_lon_deg, 0.5 * URBAN_UNCERTAINTY_KM)

        targets.append(loc)

    # Exposure share, normalised to average 1.0 across K, so mission value scales
    # with who is worst affected without total scenario value drifting with K.
    tot = sum(t.exposure for t in targets) or 1.0
    for t in targets:
        t.exposure_share = t.exposure / tot
        t.value_weight = 0.5 + 0.5 * len(targets) * t.exposure_share

    print(f"[Event] M{ev['magnitude']:.1f} {ev.get('place','')} at "
          f"({ev['lat_deg']:.3f},{ev['lon_deg']:.3f}) depth {ev['depth_km']:.0f} km, "
          f"{ev['time']}")
    print(f"[Event] damage radius {radius:.0f} km; {len(targets)} of {n_affected} "
          f"affected settlement(s) selected")
    if n_affected < max_targets:
        print(f"[Event] WARNING: only {n_affected} settlement(s) inside the damage "
              f"radius. With K small there is no allocation decision left -- lower "
              f"min_population in load_cities, or raise min_magnitude to find a "
              f"larger event.")
    print(f"    {'settlement':18s} {'dist':>6s} {'MMI':>5s} {'pop':>10s} "
          f"{'sev':>5s} {'weight':>7s}")
    for t in targets:
        print(f"    {t.settlement[:18]:18s} {t.distance_km:6.0f} {t.mmi:5.1f} "
              f"{t.population:10,.0f} {t.severity:5.2f} {t.value_weight:7.2f}")
    return targets


# =============================================================================
# PHENOMENA REGISTRATION
# =============================================================================

def register_earthquake_phenomena(world, targets, min_time, max_time,
                                  n_region=18, n_urban=20, n_district=10,
                                  seed=0):
    """Register the observable damage signature in the world.

    WITHOUT THIS EVERY OBSERVATION RETURNS AN EMPTY DATA PRODUCT.
    An overpass only yields a product if some Phenomenon falls inside the
    instrument footprint, so with nothing registered success_declarer returns
    False for every task -- including the MANDATORY root. The extent literal then
    resolves to 0, A_parents goes to 0 for the whole subtree, the next solve books
    nothing, no further events queue, and the run drains its event queue far short
    of the horizon.

    Phenomena sit at ALL THREE aim points, matching the retargeting:

        REGION    +-30 km   surface rupture, landslides, scattered collapse.
                            What the wide-area EXTENT sweep sees.
        URBAN     +-8 km    collapsed structures across the affected built-up
                            area. What PHASE 1 characterises.
        DISTRICT  +-2 km    the concentrated worst-damage cluster PHASE 2/3
                            image at high resolution.

    A task aimed at the WRONG scale therefore sees little: a triage pass still
    pointed at the region because EXTENT has not completed will mostly miss. That
    is what makes the retargeting -- and hence the phase ordering -- operationally
    real rather than decorative.

    Density scales with severity, so a town that was hit hard is genuinely easier
    to detect than one that got off lightly.

    Damage is persistent, so each phenomenon spans the whole window; no
    Lagrangian tracers are needed (unlike a drifting volcanic plume).
    """
    if not targets:
        return 0
    rng = random.Random(seed)
    n_total = 0

    def _scatter(lat0, lon0, spread_km, n, tag, label, phase,
                 reveals_phase=None, reveals_lat_deg=None,
                 reveals_lon_deg=None):
        """Scatter n phenomena over a disc, CONCENTRATED toward the centre.

        r = spread * u**2, not spread * sqrt(u). Uniform-over-area scattering
        pushes most points toward the rim, and a footprint on the centre can then
        see ZERO of them -- which makes a mandatory root return empty and kills
        the cascade. Damage intensity genuinely decays away from the source, so
        centre-weighting is both physically right and what makes a
        correctly-aimed pass productive.
        """
        nonlocal n_total
        for i in range(int(n)):
            r = spread_km * (rng.random() ** 2)
            b = rng.uniform(0.0, 2.0 * math.pi)
            lat, lon = _offset_latlon(lat0, lon0, r * math.cos(b), r * math.sin(b))
            # Central evidence is clearer than evidence near the edge of the
            # affected area.  This gives the updater a deterministic rule for
            # choosing among several features returned by one observation.
            evidence_strength = max(0.05, 1.0 - r / max(spread_km, 1e-9))
            world.add_phenomenon(EarthquakeFeature(
                lon_deg=lon, lat_deg=lat,
                start_time=min_time, end_time=max_time,
                name=f"{label}_{tag}_{i:03d}",
                settlement=label,
                phase=phase,
                evidence_strength=evidence_strength,
                reveals_phase=reveals_phase,
                reveals_lat_deg=reveals_lat_deg,
                reveals_lon_deg=reveals_lon_deg,
            ))
            n_total += 1

    for tgt in targets:
        sev = max(0.25, tgt.severity)   # floor: even light damage is visible
        _scatter(
            tgt.lat_deg, tgt.lon_deg, REGION_UNCERTAINTY_KM,
            round(n_region * sev), "rupture", tgt.name, "region",
            reveals_phase="urban",
            reveals_lat_deg=tgt.urban_lat_deg,
            reveals_lon_deg=tgt.urban_lon_deg,
        )
        _scatter(
            tgt.urban_lat_deg, tgt.urban_lon_deg, URBAN_UNCERTAINTY_KM,
            round(n_urban * sev), "urban", tgt.name, "urban",
            reveals_phase="district",
            reveals_lat_deg=tgt.district_lat_deg,
            reveals_lon_deg=tgt.district_lon_deg,
        )
        _scatter(
            tgt.district_lat_deg, tgt.district_lon_deg,
            DISTRICT_UNCERTAINTY_KM, round(n_district * sev),
            "district", tgt.name, "district",
        )

    print(f"[Earthquake] Registered {n_total} damage phenomena over "
          f"{len(targets)} settlement(s), at three scales each")
    return n_total


# =============================================================================
# REWARDERS
# =============================================================================

def _look_quality(op):
    """1.0 overhead, 0.0 at the horizon.

    NOTE the sign: abs(90 - elevation)/90 is LARGEST at the horizon and would pay
    more for worse geometry.
    """
    return max(0.0, min(1.0, op.look_angle_dec_deg / 90.0))


def _illum_quality(op, preferred_deg=PREFERRED_SUN_ZENITH_DEG):
    return max(0.0, 1.0 - abs(preferred_deg - op.sun_zenith_angle_deg) / 90.0)


def _range_quality(op, ref_km=RANGE_REF_KM):
    """Bounded GSD proxy: 1.0 at zero range, 0.5 at ref_km. Bounded so it cannot
    swamp the angular terms the way an unbounded 1/range does."""
    return ref_km / (ref_km + max(float(op.range_km), 0.0))


def _timeliness(op, t_event):
    """Value decay with acquisition age, floored.

    A grading product at hour 6 and one at hour 60 are not the same product.
    """
    age_h = max(0.0, (op.time - t_event).total_seconds() / 3600.0)
    return TIMELINESS_FLOOR + (1.0 - TIMELINESS_FLOOR) * math.exp(-age_h / DECAY_TAU_H)


def _rewarder(base_value, weight, t_event, optical):
    """Reward = base value x exposure weight x geometry x timeliness.

    `weight` is a plain number: mission value is STATIC here. Making it a live
    belief that observations update was tried and removed -- it made the objective
    move between solves for no gain to any claim in the paper.
    """
    def _r(op, preferred_zenith_angle_deg=PREFERRED_SUN_ZENITH_DEG):
        if optical:
            geom = (W_LOOK * _look_quality(op)
                    + W_ILLUM * _illum_quality(op, preferred_zenith_angle_deg)
                    + W_RANGE * _range_quality(op))
        else:
            # No illumination term: SAR is its own source, which is exactly why it
            # substitutes for optical under cloud and at night.
            geom = W_LOOK_SAR * _look_quality(op) + W_RANGE_SAR * _range_quality(op)
        return base_value * weight * geom * _timeliness(op, t_event)
    return _r


def _flatten_product(data_product):
    """Return a flat list for the simulator's scalar or nested products."""
    if data_product is None:
        return []
    if not isinstance(data_product, (list, tuple, set)):
        return [data_product]
    flattened = []
    for item in data_product:
        if isinstance(item, (list, tuple, set)):
            flattened.extend(_flatten_product(item))
        elif item is not None:
            flattened.append(item)
    return flattened


def _product_processor(expected_phase, settlement):
    """Build the phase-specific product used by one workflow task.

    A wide footprint may contain evidence from several spatial scales or nearby
    settlements.  The requested product is successful only when it contains the
    evidence type and settlement that the task was intended to collect.
    """
    def _processor(_observation, _spacecraft, phenomena):
        is_collection = isinstance(phenomena, (list, tuple, set))
        kept = [
            item for item in _flatten_product(phenomena)
            if isinstance(item, EarthquakeFeature)
            and item.phase == expected_phase
            and item.settlement == settlement
        ]
        return kept if is_collection else (kept[0] if kept else None)
    return _processor


def _success_for(expected_phase, settlement):
    """Require the correct product, rather than any object in the footprint."""
    def _success(data_product):
        return any(
            isinstance(item, EarthquakeFeature)
            and item.phase == expected_phase
            and item.settlement == settlement
            for item in _flatten_product(data_product)
        )
    return _success


def _best_reveal(data_products, settlement, destination_phase):
    """Select the strongest returned feature that reveals an aim point."""
    candidates = [
        item for item in _flatten_product(data_products)
        if isinstance(item, EarthquakeFeature)
        and item.settlement == settlement
        and item.reveals_phase == destination_phase
        and item.reveals_lat_deg is not None
        and item.reveals_lon_deg is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.evidence_strength)


# =============================================================================
# WORKFLOW
# =============================================================================

def _req(target, name, lo, hi, instrument):
    """Initial request. Every task starts aimed at the REGION (the settlement's
    own coordinates); the request updater narrows it as parents complete."""
    return ObservationRequest(
        lon_deg=target.lon_deg, lat_deg=target.lat_deg, alt_km=target.alt_km,
        min_time=lo, max_time=hi,
        instrument=instrument,
        request_name=name,
        min_elevation_deg=MIN_ELEVATION_DEG,
    )


def _phase_constraints(temporal_parent, open_h, close_h, success_parents):
    """TEMPORAL bounds plus the SUCCESS prerequisites for one phase.

    Both TEMPORAL bounds name the SAME parent, so build_workflow_graph creates ONE
    edge per (parent, child) pair. Splitting them across different parents produces
    two direct predecessors and defeats the single-parent CHAIN shortcut in the
    formulation, forcing an exp()/log join at every node.

    SUCCESS constraints make the phases sequential. The stochastic Gurobi
    scheduler treats them as zero-delay precedence relations while the dispatcher
    waits for the parent result. They are not redundant with the gates: the gates
    tell the MILP what a task is worth, while the SUCCESS edges identify the
    immediate predecessor whose product makes the child actionable.

    CAUTION: the dispatcher ANDs these BY DEFAULT. `success_parents` means ALL of
    them must be satisfied, never any of them. A task whose real prerequisite is a
    DISJUNCTION over parents must set `task.success_constraint_mode = 'any'`, or
    the OR in its `.gate` silently becomes an AND at dispatch time and the task can
    block permanently once any one branch becomes unachievable.
    """
    cons = [
        Constraint(ConstraintClass.TEMPORAL, TemporalConstraintType.START_AFTER_OFFSET,
                   temporal_parent, {'offset': dt.timedelta(hours=open_h)}),
        Constraint(ConstraintClass.TEMPORAL, TemporalConstraintType.START_BEFORE_OFFSET,
                   temporal_parent, {'offset': dt.timedelta(hours=close_h)}),
    ]
    for sp in success_parents:
        cons.append(Constraint(ConstraintClass.SUCCESS,
                               SuccessConstraintType.START_IF_SUCCESSFUL, sp))
    return cons


def _validate_24h_timing(min_time, max_time):
    """Reject a driver horizon or phase window inconsistent with this scenario."""
    horizon_h = (max_time - min_time).total_seconds() / 3600.0
    if not math.isclose(horizon_h, CAMPAIGN_HORIZON_H, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"Earthquake scenario requires a {CAMPAIGN_HORIZON_H:g} h horizon; "
            f"received {horizon_h:g} h."
        )

    phase_windows = {
        "URBAN": (URBAN_OPEN_H, URBAN_CLOSE_H),
        "TRIAGE": (TRIAGE_OPEN_H, TRIAGE_CLOSE_H),
        "FINAL": (FINAL_OPEN_H, FINAL_CLOSE_H),
    }
    for phase, (open_h, close_h) in phase_windows.items():
        if open_h < 0.0 or close_h <= open_h:
            raise ValueError(
                f"Invalid {phase} window [{open_h:g}, {close_h:g}] h: "
                "the close offset must be greater than the open offset."
            )

    latest_possible_close_h = EXTENT_WINDOW_H + max(
        close_h for _, close_h in phase_windows.values()
    )
    if latest_possible_close_h > CAMPAIGN_HORIZON_H + 1e-9:
        raise ValueError(
            "Configured EXTENT-relative windows can exceed the campaign horizon: "
            f"latest possible close is Event +{latest_possible_close_h:g} h, "
            f"but the horizon is {CAMPAIGN_HORIZON_H:g} h."
        )


def create_earthquake_workflow(targets, min_time, max_time, max_num_instances=3,
                               satellites=None):
    """AND/OR/NOT damage-assessment workflow: six tasks per settlement.

    Must be solved with stochastic_formulation="general_logical_dag": tasks carry
    `.gate` attributes that a pure-AND formulation cannot represent.

    Every task gets `.eq_meta = (settlement_name, kind)` so downstream code
    identifies tasks by REGISTRY rather than by parsing their names. Name parsing
    broke as soon as a settlement contained an underscore.

    Each settlement has its OWN mandatory root, deliberately: an earlier draft
    shared roots across settlements (a tiled rupture) and one failed acquisition
    then killed every settlement beneath it.
    """
    if not targets:
        raise ValueError("create_earthquake_workflow: no targets")

    _validate_24h_timing(min_time, max_time)

    # The OR gate's justification is that SAR and optical are GENUINE substitutes.
    # If the fleet carries only one modality the gate still holds structurally but
    # the operational story does not, and that should be visible rather than silent.
    if satellites:
        _kinds = set()
        for s in satellites:
            for ins in (getattr(s, 'instruments', None) or []):
                _kinds.add(getattr(ins, 'type', ins))
        if _kinds and not ({InstrumentType.SAR, InstrumentType.RGB} <= _kinds):
            print(f"[Workflow] WARNING: fleet instruments = {_kinds}. The OR gate "
                  f"assumes SAR and optical are genuine substitutes; with a "
                  f"single-modality fleet that justification does not hold.")

    t_event = min_time      # the sim epoch IS the event; timeliness measures from it

    # SCHEDULE policy: plan everything, including tasks whose prerequisites are not
    # yet met -- that is the point of planning under uncertainty.
    policy_schedule = {c: True for c in ConstraintClass}
    # DISPATCH policy: block on SUCCESS. A follow-up must not be SUBMITTED before
    # its parent completes, because until then it is still aimed at the wrong
    # scale. With the retargeting below this is PHYSICAL, not a tuning knob.
    policy_dispatch = {
        ConstraintClass.TEMPORAL: True,
        ConstraintClass.SUCCESS: False,
        ConstraintClass.GEOMETRY: True,
    }

    tasks, logic_nodes = [], []

    def _add(task, settlement_name, kind):
        task.eq_meta = (settlement_name, kind)
        task.enforce_success_precedence = (kind != 'extent')
        task.target_stage = "region"
        task.target_revealed = (kind == "extent")
        task.retarget_count = 0
        task.geometry_revision = 0
        task.retarget_history = []
        tasks.append(task)

    for tgt in targets:
        g = tgt.name
        w = tgt.value_weight

        # --- PHASE 0: EXTENT (mandatory root, SAR, aimed at the region) ------
        extent = ConstrainedObservationRequest(
            name=f"{g}_extent",
            observation_request=_req(tgt, f"{g}_extent", min_time,
                                     min_time + dt.timedelta(hours=EXTENT_WINDOW_H),
                                     InstrumentType.SAR),
            is_mandatory=True,
            schedule_policy_if_constraint_unsatisfied=policy_schedule,
            dispatch_policy_if_constraint_unsatisfied=policy_dispatch,
            timeline_constraints=[], timeline_impacts=[],
            rewarder=_rewarder(VALUE_EXTENT, w, t_event, optical=False),
            success_declarer=_success_for("region", g),
            phenomenon_processor=_product_processor("region", g),
            request_group=f"{g}_extent",
            max_num_instances=max_num_instances,
        )
        extent.gate = None
        _add(extent, g, 'extent')

        # --- PHASE 1: substitutable characterisations of the urban area ------
        urban_lo = min_time + dt.timedelta(hours=URBAN_OPEN_H)
        urban_hi = min(min_time + dt.timedelta(hours=EXTENT_WINDOW_H + URBAN_CLOSE_H),
                       max_time)

        urban_opt = ConstrainedObservationRequest(
            name=f"{g}_urban_opt",
            observation_request=_req(tgt, f"{g}_urban_opt", urban_lo, urban_hi,
                                     InstrumentType.RGB),
            is_mandatory=False,
            task_constraints=_phase_constraints(extent, URBAN_OPEN_H, URBAN_CLOSE_H,
                                                [extent]),
            schedule_policy_if_constraint_unsatisfied=policy_schedule,
            dispatch_policy_if_constraint_unsatisfied=policy_dispatch,
            timeline_constraints=[], timeline_impacts=[],
            rewarder=_rewarder(VALUE_URBAN_OPT, w, t_event, optical=True),
            success_declarer=_success_for("urban", g),
            phenomenon_processor=_product_processor("urban", g),
            request_group=f"{g}_urban",
            max_num_instances=max_num_instances,
        )
        urban_opt.gate = Lit(extent)
        _add(urban_opt, g, 'urban_opt')

        urban_sar = ConstrainedObservationRequest(
            name=f"{g}_urban_sar",
            observation_request=_req(tgt, f"{g}_urban_sar", urban_lo, urban_hi,
                                     InstrumentType.SAR),
            is_mandatory=False,
            task_constraints=_phase_constraints(extent, URBAN_OPEN_H, URBAN_CLOSE_H,
                                                [extent]),
            schedule_policy_if_constraint_unsatisfied=policy_schedule,
            dispatch_policy_if_constraint_unsatisfied=policy_dispatch,
            timeline_constraints=[], timeline_impacts=[],
            rewarder=_rewarder(VALUE_URBAN_SAR, w, t_event, optical=False),
            success_declarer=_success_for("urban", g),
            phenomenon_processor=_product_processor("urban", g),
            request_group=f"{g}_urban",
            max_num_instances=max_num_instances,
        )
        urban_sar.gate = Lit(extent)
        _add(urban_sar, g, 'urban_sar')

        # CHAR: "the urban area has been characterised by EITHER modality".
        # A named LogicNode rather than an inline Or, so the DSOP compilation is
        # built once and shared by every child that references it.
        char = LogicNode(name=f"{g}_characterised",
                         gate=Or(Lit(urban_opt), Lit(urban_sar)),
                         request_group="state")
        logic_nodes.append(char)

        # --- PHASE 2: TRIAGE (the OR gate) -----------------------------------
        triage_lo = min_time + dt.timedelta(hours=TRIAGE_OPEN_H)
        triage_hi = min(min_time + dt.timedelta(hours=EXTENT_WINDOW_H + TRIAGE_CLOSE_H),
                        max_time)
        triage = ConstrainedObservationRequest(
            name=f"{g}_triage",
            observation_request=_req(tgt, f"{g}_triage", triage_lo, triage_hi,
                                     InstrumentType.RGB),
            is_mandatory=True,
            task_constraints=_phase_constraints(extent, TRIAGE_OPEN_H, TRIAGE_CLOSE_H,
                                                [urban_opt, urban_sar]),
            schedule_policy_if_constraint_unsatisfied=policy_schedule,
            dispatch_policy_if_constraint_unsatisfied=policy_dispatch,
            timeline_constraints=[], timeline_impacts=[],
            rewarder=_rewarder(VALUE_TRIAGE, w, t_event, optical=True),
            success_declarer=_success_for("district", g),
            phenomenon_processor=_product_processor("district", g),
            request_group=f"{g}_triage",
            max_num_instances=max_num_instances,
        )
        triage.gate = Lit(char)
        # Makes the dispatcher agree with char = Or(urban_opt, urban_sar). Without
        # it the dispatcher ANDs both parents and triage blocks permanently as soon
        # as one modality runs out of passes.
        triage.success_constraint_mode = 'any'
        _add(triage, g, 'triage')

        # --- PHASE 3: HIRES (the NOT gate) and ACCESS ------------------------
        final_lo = min_time + dt.timedelta(hours=FINAL_OPEN_H)
        final_hi = min(min_time + dt.timedelta(hours=EXTENT_WINDOW_H + FINAL_CLOSE_H),
                       max_time)

        hires = ConstrainedObservationRequest(
            name=f"{g}_hires",
            observation_request=_req(tgt, f"{g}_hires", final_lo, final_hi,
                                     InstrumentType.RGB),
            is_mandatory=True,
            task_constraints=_phase_constraints(extent, FINAL_OPEN_H, FINAL_CLOSE_H,
                                                [triage]),
            schedule_policy_if_constraint_unsatisfied=policy_schedule,
            dispatch_policy_if_constraint_unsatisfied=policy_dispatch,
            timeline_constraints=[], timeline_impacts=[],
            rewarder=_rewarder(VALUE_HIRES, w, t_event, optical=True),
            success_declarer=_success_for("district", g),
            phenomenon_processor=_product_processor("district", g),
            request_group=f"{g}_hires",
            max_num_instances=max_num_instances,
        )
        # ANTI-COMPLEMENTARITY: booking urban_opt more deeply REDUCES the expected
        # value of this task. No baseline can represent that. Pair this with a
        # raised c_exec on the hires group in the driver -- the gate is
        # cost-avoidance, not a logic demonstration.
        hires.gate = And(Lit(triage), Not(Lit(urban_opt)))
        _add(hires, g, 'hires')

        access = ConstrainedObservationRequest(
            name=f"{g}_access",
            observation_request=_req(tgt, f"{g}_access", final_lo, final_hi,
                                     InstrumentType.SAR),
            is_mandatory=True,
            task_constraints=_phase_constraints(extent, FINAL_OPEN_H, FINAL_CLOSE_H,
                                                [triage]),
            schedule_policy_if_constraint_unsatisfied=policy_schedule,
            dispatch_policy_if_constraint_unsatisfied=policy_dispatch,
            timeline_constraints=[], timeline_impacts=[],
            rewarder=_rewarder(VALUE_ACCESS, w, t_event, optical=False),
            success_declarer=_success_for("district", g),
            phenomenon_processor=_product_processor("district", g),
            request_group=f"{g}_access",
            max_num_instances=max_num_instances,
        )
        # Adds no new gate TYPE -- it is a second Lit(triage) child. What it buys is
        # resource contention: it demands SAR in the same window HIRES demands
        # optical. Drop it for a leaner 5-task subtree if that pressure is not
        # wanted.
        access.gate = Lit(triage)
        _add(access, g, 'access')

    # ---------------------------------------------------------------------
    # PROGRESSIVE RETARGETING
    # ---------------------------------------------------------------------
    def request_updater_earthquake(current_time, requests, timelines):
        """Reveal and apply aim points from successful parent data products.

            region (+-30 km) -> urban area (+-8 km) -> district (+-2 km)

        PHASE 1 reads the urban aim point from the EXTENT product. PHASE 2/3 read
        the district aim point from whichever urban product succeeds. The hidden
        truth lives on EarthquakeFeature objects in the simulated world; this
        callback cannot access it until those objects appear in a returned data
        product.

        MUTATES observation_request IN PLACE. Rebinding would orphan every booking
        row already recorded against the task in broker._requests, since those hold
        the object as it was at DISPATCH time -- and the more a planner replans, the
        more of its own bookings it would lose from its own metrics.
        """
        del timelines
        by_group = {}
        for task in requests:
            group, kind = getattr(task, "eq_meta", (None, None))
            if group is not None:
                by_group.setdefault(group, {})[kind] = task

        changes = []

        def _returned_product(task):
            if (task is None
                    or not getattr(task, "completed", False)
                    or not getattr(task, "successful_execution", False)):
                return []
            return _flatten_product(getattr(task, "data_product", []))

        def _retarget(task, reveal, stage, source_name):
            if (task is None or reveal is None
                    or getattr(task, "completed", False)
                    or getattr(task, "dispatched", False)):
                return False

            old_lat = float(task.observation_request.lat_deg)
            old_lon = float(task.observation_request.lon_deg)
            new_lat = float(reveal.reveals_lat_deg)
            new_lon = float(reveal.reveals_lon_deg)
            old_stage = getattr(task, "target_stage", "region")

            # Do not report the same reveal at every event-driven replan.
            if (old_stage == stage
                    and math.isclose(old_lat, new_lat, abs_tol=1e-12)
                    and math.isclose(old_lon, new_lon, abs_tol=1e-12)):
                return False

            task.observation_request.lat_deg = new_lat
            task.observation_request.lon_deg = new_lon
            task.target_stage = stage
            task.target_revealed = True
            task.retarget_count = int(getattr(task, "retarget_count", 0)) + 1
            task.geometry_revision = int(
                getattr(task, "geometry_revision", 0)
            ) + 1
            task.retarget_source = source_name
            task.retarget_feature = reveal.name

            # This task has not been dispatched, so any opportunity selected for
            # its previous aim point is only a stale internal plan.  Clear the
            # task-level geometry cache and selection fields so the next solve
            # must rebuild them for the revealed point.
            task.observation_opportunities = {}
            task.observation_opportunity = None
            task.observation_opportunity_pass = None
            task.observation_opportunity_satellite = None
            task.pending_dispatch_passes = []
            task.scheduled = False

            moved_km = _km_between(old_lat, old_lon, new_lat, new_lon)
            history = getattr(task, "retarget_history", None)
            if history is None:
                history = []
                task.retarget_history = history
            history.append({
                "time": current_time.isoformat()
                if hasattr(current_time, "isoformat") else str(current_time),
                "from_stage": old_stage,
                "to_stage": stage,
                "from_lat_deg": old_lat,
                "from_lon_deg": old_lon,
                "to_lat_deg": new_lat,
                "to_lon_deg": new_lon,
                "distance_km": moved_km,
                "source_task": source_name,
                "source_feature": reveal.name,
            })
            changes.append((task.name, old_stage, stage, moved_km, source_name))
            return True

        for group, parts in by_group.items():
            extent = parts.get("extent")
            extent_product = _returned_product(extent)
            urban_reveal = _best_reveal(extent_product, group, "urban")

            if urban_reveal is not None:
                source = extent.name
                for kind in ("urban_opt", "urban_sar"):
                    _retarget(parts.get(kind), urban_reveal, "urban", source)

            urban_products = []
            successful_urban_names = []
            for kind in ("urban_opt", "urban_sar"):
                parent = parts.get(kind)
                product = _returned_product(parent)
                if product:
                    urban_products.extend(product)
                    successful_urban_names.append(parent.name)
            district_reveal = _best_reveal(urban_products, group, "district")

            if district_reveal is not None:
                source = "|".join(successful_urban_names)
                for kind in ("triage", "hires", "access"):
                    _retarget(parts.get(kind), district_reveal, "district", source)
            elif urban_reveal is not None:
                # These tasks are still blocked on urban SUCCESS. Moving their
                # nominal look-ahead target to the best currently known area
                # makes the intermediate plan less fictional without dispatching
                # anything early.
                source = extent.name
                for kind in ("triage", "hires", "access"):
                    _retarget(parts.get(kind), urban_reveal, "urban", source)

        if changes:
            counts = {
                stage: sum(1 for _, _, new_stage, _, _ in changes
                           if new_stage == stage)
                for stage in ("urban", "district")
            }
            print(
                f"    [Retarget] {counts['urban']} task(s) -> urban, "
                f"{counts['district']} task(s) -> district; "
                f"mean move={sum(c[3] for c in changes) / len(changes):.1f} km"
            )
            for task_name, old_stage, new_stage, moved_km, source in changes:
                print(
                    f"        {task_name}: {old_stage} -> {new_stage} "
                    f"({moved_km:.1f} km), revealed by {source}"
                )
        return requests

    n_not = sum(1 for t in tasks if _has_not(getattr(t, 'gate', None)))
    print(f"[Workflow] {len(tasks)} observation tasks + {len(logic_nodes)} logic node(s) "
          f"over {len(targets)} settlement(s), 6 tasks each")
    print(f"[Workflow] Gates: {len(logic_nodes)} OR, {n_not} with NOT, "
          f"{sum(1 for t in tasks if t.is_mandatory)} mandatory root(s)")

    return Workflow(
        constrained_observation_requests=tasks,
        timelines=[],
        timeline_updater=lambda t, r, tl: tl,
        request_updater=request_updater_earthquake,
    )


def _has_not(gate):
    if gate is None:
        return False
    if isinstance(gate, Not):
        return True
    if isinstance(gate, (And, Or)):
        return any(_has_not(m) for m in gate.operands)
    return False


# =============================================================================
# GATE-AWARE REACHABILITY  (metrics)
# =============================================================================

def compute_earthquake_reachability(tasks, completed_tasks, no_pass_tasks=None):
    """Denominator for the completion rate.

    A task is EXCLUDED only when the run could not have completed it for a reason
    THE SCHEDULER IS NOT RESPONSIBLE FOR. There are exactly two:

      1. Its prerequisite gate resolved AGAINST it. HIRES is unreachable by
         construction whenever URBAN_opt succeeded, so counting it as a miss
         penalises exactly the planner that made the right call.

      2. It has NO candidate pass anywhere in the horizon -- fleet geometry, not
         a scheduling decision. Pass `no_pass_tasks` as a set of task NAMES
         computed ONCE before any scheduling and shared across every scheduler,
         so the denominator is identical for all of them. Leave it None to skip
         this exclusion.

    A task whose PARENT FAILED stays IN the denominator. This is the one that
    matters, and the previous version got it wrong: it walked the cascade and
    dropped children whenever an ancestor failed. Under that rule a scheduler
    that loses EXTENT for a settlement drops FIVE tasks from its OWN denominator
    and scores 14/19 = 74%, while one that gets EXTENT and completes four of the
    five scores 14/24 = 58% -- the first did strictly less work and looked
    better. Any completion rate whose denominator shrinks on the scheduler's own
    failures systematically flatters whichever scheduler gives up earliest.

    So: a cascade failure is a MISS for every task it took down, which is what
    makes the AND-cascade argument measurable in the first place.

    Identifies tasks through `.eq_meta` rather than by parsing names.
    """
    def ok(t):
        return t is not None and t in completed_tasks and \
            getattr(t, 'successful_execution', True)

    groups = {}
    for t in tasks:
        g, kind = getattr(t, 'eq_meta', (None, None))
        if kind is not None:
            groups.setdefault(g, {})[kind] = t

    no_pass_tasks = no_pass_tasks or set()

    reachable = set()
    n_suppressed = n_nopass = 0
    for g, parts in groups.items():
        u_opt = parts.get('urban_opt')
        for kind, t in parts.items():
            # (1) gate resolved against it
            if kind == 'hires' and ok(u_opt):
                n_suppressed += 1
                continue
            # (2) no candidate pass in the whole horizon
            if t.observation_request.name in no_pass_tasks:
                n_nopass += 1
                continue
            reachable.add(t)

    if n_suppressed or n_nopass:
        print(f"   [Reachability] {len(reachable)} task(s) in denominator; excluded "
              f"{n_suppressed} NOT-suppressed HIRES, {n_nopass} with no feasible pass")
    return reachable

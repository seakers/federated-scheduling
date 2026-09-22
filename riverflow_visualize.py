"""Leaflet animation of river-flow bookings: satellite looks, survive vs fail.

Writes a self-contained HTML file (Leaflet from a CDN). Open it in a browser.

    python riverflow_visualize.py --results-dir results/riverflow_YYYY-mm-dd_HHMMSS --seed 42

Existing campaigns only store DATA_RECEIVED rows in executions_*.json, so
failures/rejects appear after a re-run (the driver now also dumps bookings_*.json).
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re

from riverflow_data_prep import EVENTS_JSON, TOY_GAUGES
from riverflow_for import FUSION_SAT_NAME

# TLE name → display name (same aliases as benchmarking_utils.load_satellites_once).
_TLE_TO_DISPLAY = {
    "SKYSAT 1": "SKYSAT-A", "SKYSAT 2": "SKYSAT-B",
    "SKYSAT C1": "SKYSAT-C1", "SKYSAT C2": "SKYSAT-C2", "SKYSAT C3": "SKYSAT-C3",
    "SKYSAT C4": "SKYSAT-C4", "SKYSAT C5": "SKYSAT-C5", "SKYSAT C6": "SKYSAT-C6",
    "SKYSAT C7": "SKYSAT-C7", "SKYSAT C8": "SKYSAT-C8", "SKYSAT C9": "SKYSAT-C9",
    "SKYSAT C10": "SKYSAT-C10", "SKYSAT C11": "SKYSAT-C11",
    "SKYSAT C12": "SKYSAT-C12", "SKYSAT C13": "SKYSAT-C13",
    "PELICAN-1 3001": "PELICAN-3001", "PELICAN-2 3009": "PELICAN-3009",
    "PELICAN-3 300A": "PELICAN-300A", "PELICAN-4 300B": "PELICAN-300B",
    "PELICAN-5 300C": "PELICAN-5", "PELICAN-6 300D": "PELICAN-6",
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
    "LEMUR 2 EMBRIONOVIS": "OT-FOREST-2",
    "LEMUR 2 KREMPEL-BRO1": "OT-FOREST-4P",
    "LEMUR 2 THERMORAPTOR": "OT-FOREST-5P",
    "LEMUR 2 KREMPEL-BRO2": "OT-FOREST-6P",
    "LEMUR 2 LANGER2SPACE": "OT-FOREST-7P",
    "LEMUR 2 NEUROSPICY": "OT-FOREST-8P",
    "LEMUR 2 ESPERANZA": "OT-FOREST-9P",
    "LEMUR 2 TILLINFINITY": "OT-FOREST-10P",
    "LEMUR 2 UNTITLED-SC": "OT-FOREST-11P",
}
_DISPLAY_TO_TLE = {v: k for k, v in _TLE_TO_DISPLAY.items()}

_STATUS_OUTCOME = {
    "DATA_RECEIVED": "survived",
    "EXECUTION_FAILED": "failed",
    "CONSTELLATION_REJECTED": "rejected",
    "CANCELLED": "cancelled",
    "TIMEOUT": "rejected",
    "BOOKING_TOO_LATE": "rejected",
}

_KIND_FROM_TASK = re.compile(r"^(?P<gid>.+)_(?P<kind>sar|vis|tir|fusion)$", re.I)

_ORBIT_CACHE: dict = {}
_TLE_NAME_CACHE: list | None = None

_OSM_TILES = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
_CARTO_TILES = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"


def _env_value(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw and raw.strip():
        return raw.strip().strip('"').strip("'")
    candidates = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, val = line.partition("=")
                if k.strip() == name:
                    v = val.strip().strip('"').strip("'")
                    return v or None
    return None


def _basemap_tile_url() -> str:
    key = _env_value("CARTO_API_KEY")
    if key:
        return f"{_CARTO_TILES}?key={key}"
    return _OSM_TILES


def _missing(v) -> bool:
    if v is None:
        return True
    try:
        if v != v:  # NaN
            return True
    except Exception:
        pass
    s = str(v)
    return s in ("nan", "None", "<NA>", "NaT", "NaN")


def _status_name(status) -> str:
    if _missing(status):
        return "UNKNOWN"
    if hasattr(status, "name"):
        return str(status.name)
    s = str(status)
    if "." in s:
        s = s.rsplit(".", 1)[-1]
    return s


def _parse_time(value):
    if _missing(value):
        return None
    if isinstance(value, dt.datetime):
        return value.replace(tzinfo=None)
    s = str(value).strip().replace("Z", "")
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(s)
    except Exception:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def _iso(t) -> str | None:
    t = _parse_time(t)
    return t.isoformat(sep="T") if t is not None else None


def _kind_of(task_name, rf_meta=None) -> tuple[str | None, str]:
    if rf_meta and rf_meta[0]:
        return str(rf_meta[0]), str(rf_meta[1] or "unknown")
    m = _KIND_FROM_TASK.match(str(task_name or ""))
    if m:
        return m.group("gid"), m.group("kind").lower()
    return None, "unknown"


def _is_fusion_sat(name) -> bool:
    return bool(name) and str(name).startswith(FUSION_SAT_NAME)


def _tle_file() -> str | None:
    files = sorted(glob.glob(os.path.join("tles", "all_tles_*.txt")))
    return files[-1] if files else None


def _tle_catalog(tle_file: str) -> list[str]:
    global _TLE_NAME_CACHE
    if _TLE_NAME_CACHE is not None:
        return _TLE_NAME_CACHE
    names = []
    with open(tle_file, encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("1 ") and not s.startswith("2 "):
                names.append(s)
    _TLE_NAME_CACHE = names
    return names


def _orbit_for(sat_name: str):
    if not sat_name or _is_fusion_sat(sat_name):
        return None
    if sat_name in _ORBIT_CACHE:
        return _ORBIT_CACHE[sat_name]
    tle_file = _tle_file()
    if not tle_file:
        _ORBIT_CACHE[sat_name] = None
        return None
    from pyorbital.orbital import Orbital

    catalog = _tle_catalog(tle_file)
    upper = {n.upper(): n for n in catalog}
    candidates = []
    for c in (sat_name, _DISPLAY_TO_TLE.get(sat_name), sat_name.split("(")[0].strip()):
        if c and c not in candidates:
            candidates.append(c)
    if sat_name.upper() in upper:
        candidates.insert(0, upper[sat_name.upper()])
    for n in catalog:
        if sat_name.upper() in n.upper() or n.upper() in sat_name.upper():
            if n not in candidates:
                candidates.append(n)
            if len(candidates) > 8:
                break
    for c in candidates:
        try:
            orb = Orbital(c, tle_file=tle_file)
            _ORBIT_CACHE[sat_name] = orb
            return orb
        except Exception:
            continue
    _ORBIT_CACHE[sat_name] = None
    return None


def _lonlatalt(orbit, t):
    try:
        lon, lat, alt = orbit.get_lonlatalt(t)
        return float(lat), float(lon), float(alt)
    except Exception:
        return None, None, None


def _sample_track(orbit, t_rise, t_fall, n=11) -> list[dict]:
    if orbit is None or t_rise is None or t_fall is None:
        return []
    if t_fall <= t_rise:
        t_fall = t_rise + dt.timedelta(minutes=4)
        t_rise = t_fall - dt.timedelta(minutes=8)
    pts = []
    span = (t_fall - t_rise).total_seconds()
    steps = max(2, n)
    for i in range(steps):
        t = t_rise + dt.timedelta(seconds=span * i / (steps - 1))
        lat, lon, alt = _lonlatalt(orbit, t)
        if lat is None:
            continue
        pts.append({"t": _iso(t), "lat": round(lat, 4), "lon": round(lon, 4),
                    "alt_km": round(alt, 1)})
    return pts


def collect_booking_snapshot(broker, tasks) -> list[dict]:
    """All broker rows with a pass, including fails / rejects / cancels."""
    req_to_task = {t.observation_request: t for t in tasks}
    name_to_task = {t.observation_request.name: t for t in tasks}
    rows = []
    reqs = getattr(broker, "_requests", None)
    if reqs is None or len(reqs) == 0:
        return rows
    for _, row in reqs.iterrows():
        req = row.get("request")
        task = req_to_task.get(req)
        if task is None:
            task = name_to_task.get(getattr(req, "name", None))
        if task is None:
            continue
        gid, kind = _kind_of(
            getattr(task, "name", getattr(req, "name", "")),
            getattr(task, "rf_meta", None),
        )
        status = _status_name(row.get("status"))
        rp = row.get("assigned_pass")
        if _missing(rp):
            rp = row.get("requested_pass")
        if _missing(rp):
            rp = None
        sat = row.get("satellite")
        if _missing(sat):
            sat = row.get("requested_satellite")
        sat_name = getattr(sat, "name", None) if sat is not None and not _missing(sat) else None
        if not sat_name and not _missing(sat):
            sat_name = str(sat)

        t_high = t_rise = t_fall = None
        look_az = look_el = range_km = None
        tgt_lat = getattr(req, "lat_deg", None)
        tgt_lon = getattr(req, "lon_deg", None)
        orbit = getattr(sat, "orbit", None) if sat is not None and not _missing(sat) else None
        if rp is not None:
            h = rp.highest
            t_high = h.time
            t_rise = rp.rise.time
            t_fall = rp.fall.time
            look_az = float(h.look_angle_az_deg)
            look_el = float(h.look_angle_dec_deg)
            range_km = float(h.range_km)
            tgt_lat = float(h.lat_deg)
            tgt_lon = float(h.lon_deg)
        if t_high is None and kind != "fusion":
            continue

        sat_lat = sat_lon = sat_alt = None
        track = []
        if orbit is not None and t_high is not None and not _is_fusion_sat(sat_name):
            sat_lat, sat_lon, sat_alt = _lonlatalt(orbit, t_high)
            rise = t_rise or (t_high - dt.timedelta(minutes=4))
            fall = t_fall or (t_high + dt.timedelta(minutes=4))
            track = _sample_track(orbit, rise, fall)

        quality = None
        if status == "DATA_RECEIVED" and rp is not None:
            try:
                quality = float(task.rewarder(rp.highest))
            except Exception:
                quality = None

        rows.append({
            "task": getattr(task, "name", str(req)),
            "gauge_id": gid,
            "kind": kind,
            "satellite": sat_name,
            "status": status,
            "outcome": _STATUS_OUTCOME.get(status, "other"),
            "pass_time": _iso(t_high),
            "rise_time": _iso(t_rise),
            "fall_time": _iso(t_fall),
            "quality": None if quality is None else round(quality, 4),
            "lat": None if tgt_lat is None else round(float(tgt_lat), 4),
            "lon": None if tgt_lon is None else round(float(tgt_lon), 4),
            "sat_lat": None if sat_lat is None else round(sat_lat, 4),
            "sat_lon": None if sat_lon is None else round(sat_lon, 4),
            "sat_alt_km": None if sat_alt is None else round(sat_alt, 1),
            "look_az_deg": None if look_az is None else round(look_az, 2),
            "look_el_deg": None if look_el is None else round(look_el, 2),
            "range_km": None if range_km is None else round(range_km, 1),
            "track": track,
        })
    rows.sort(key=lambda r: r.get("pass_time") or "")
    return rows


def dump_bookings(path: str, broker, tasks) -> str:
    rows = collect_booking_snapshot(broker, tasks)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"  [Saved] {path}  ({len(rows)} bookings)")
    return path


def load_gauges(events=None) -> list[dict]:
    if events is None:
        events = []
        if os.path.exists(EVENTS_JSON):
            with open(EVENTS_JSON, encoding="utf-8") as f:
                events = json.load(f).get("events", [])
        if not events:
            events = [
                {**g, "t_start": None, "t_end": None, "name": g["name"]}
                for g in TOY_GAUGES
            ]
    out = []
    for e in events:
        out.append({
            "id": e["gauge_id"],
            "name": e.get("name") or e["gauge_id"],
            "lat": float(e["lat"]),
            "lon": float(e["lon"]),
            "t_start": e.get("t_start"),
            "t_end": e.get("t_end"),
            "river_width_m": e.get("river_width_m"),
        })
    return out


def _obs_from_booking(row: dict) -> dict:
    kind = (row.get("kind") or "unknown").lower()
    status = row.get("status") or "UNKNOWN"
    sat = row.get("satellite")
    if kind == "fusion" or _is_fusion_sat(sat):
        kind = "fusion"
    t = row.get("pass_time") or row.get("t")
    return {
        "task": row.get("task"),
        "gauge_id": row.get("gauge_id"),
        "kind": kind,
        "satellite": sat,
        "status": status,
        "outcome": row.get("outcome") or _STATUS_OUTCOME.get(status, "other"),
        "t": t,
        "t_rise": row.get("rise_time"),
        "t_fall": row.get("fall_time"),
        "quality": row.get("quality"),
        "lat": row.get("lat"),
        "lon": row.get("lon"),
        "sat_lat": row.get("sat_lat"),
        "sat_lon": row.get("sat_lon"),
        "sat_alt_km": row.get("sat_alt_km"),
        "look_el_deg": row.get("look_el_deg"),
        "range_km": row.get("range_km"),
        "track": row.get("track") or [],
    }


def _obs_from_execution(row: dict, gauges_by_id: dict) -> dict | None:
    task = row.get("task") or ""
    gid, kind = _kind_of(task)
    sat = row.get("satellite")
    if kind == "fusion" or _is_fusion_sat(sat):
        kind = "fusion"
    g = gauges_by_id.get(gid) or {}
    t = row.get("pass_time")
    return {
        "task": task,
        "gauge_id": gid,
        "kind": kind,
        "satellite": sat,
        "status": "DATA_RECEIVED",
        "outcome": "survived",
        "t": _iso(t) if t else None,
        "t_rise": None,
        "t_fall": None,
        "quality": row.get("quality"),
        "lat": g.get("lat"),
        "lon": g.get("lon"),
        "sat_lat": None,
        "sat_lon": None,
        "sat_alt_km": None,
        "look_el_deg": None,
        "range_km": None,
        "track": [],
    }


def enrich_tracks(observations: list[dict], n=11) -> None:
    for o in observations:
        if o.get("track") or o.get("kind") == "fusion" or _is_fusion_sat(o.get("satellite")):
            continue
        t = _parse_time(o.get("t"))
        if t is None:
            continue
        t_rise = _parse_time(o.get("t_rise")) or (t - dt.timedelta(minutes=4))
        t_fall = _parse_time(o.get("t_fall")) or (t + dt.timedelta(minutes=4))
        orbit = _orbit_for(o.get("satellite") or "")
        if orbit is None:
            continue
        lat, lon, alt = _lonlatalt(orbit, t)
        o["sat_lat"] = None if lat is None else round(lat, 4)
        o["sat_lon"] = None if lon is None else round(lon, 4)
        o["sat_alt_km"] = None if alt is None else round(alt, 1)
        o["t_rise"] = o.get("t_rise") or _iso(t_rise)
        o["t_fall"] = o.get("t_fall") or _iso(t_fall)
        o["track"] = _sample_track(orbit, t_rise, t_fall, n=n)


def _fill_gauge_coords(observations: list[dict], gauges: list[dict]) -> None:
    by_id = {g["id"]: g for g in gauges}
    for o in observations:
        g = by_id.get(o.get("gauge_id"))
        if g is None:
            continue
        if o.get("lat") is None:
            o["lat"] = g["lat"]
        if o.get("lon") is None:
            o["lon"] = g["lon"]


def _list_run_files(results_dir: str, prefix: str) -> list[str]:
    return sorted(glob.glob(os.path.join(results_dir, f"{prefix}_seed*.json")))


def _parse_seed_scheduler(path: str, prefix: str):
    base = os.path.basename(path)
    m = re.match(rf"{prefix}_seed(\d+)_(.+)\.json$", base)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)


def load_campaign(results_dir: str, seed: int | None = None, scheduler: str | None = None):
    gauges = load_gauges()
    by_id = {g["id"]: g for g in gauges}
    booking_files = _list_run_files(results_dir, "bookings")
    exec_files = _list_run_files(results_dir, "executions")
    run_files = _list_run_files(results_dir, "run")

    keys = []
    for path in booking_files + exec_files:
        pref = "bookings" if os.path.basename(path).startswith("bookings") else "executions"
        s, sch = _parse_seed_scheduler(path, pref)
        if s is None:
            continue
        keys.append((s, sch))
    if not keys:
        raise FileNotFoundError(
            f"No bookings_*.json or executions_*.json in {results_dir}"
        )
    seeds = sorted({s for s, _ in keys})
    if seed is None:
        seed = 42 if 42 in seeds else seeds[0]
    schedulers = sorted({sch for s, sch in keys if s == seed})
    if scheduler is not None:
        schedulers = [s for s in schedulers if s == scheduler]
        if not schedulers:
            raise FileNotFoundError(f"No files for seed={seed} scheduler={scheduler}")

    metrics = {}
    for path in run_files:
        s, sch = _parse_seed_scheduler(path, "run")
        if s != seed:
            continue
        with open(path, encoding="utf-8") as f:
            metrics[sch] = json.load(f)

    runs = {}
    for sch in schedulers:
        bpath = os.path.join(results_dir, f"bookings_seed{seed:04d}_{sch}.json")
        epath = os.path.join(results_dir, f"executions_seed{seed:04d}_{sch}.json")
        source = None
        observations = []
        if os.path.exists(bpath):
            with open(bpath, encoding="utf-8") as f:
                observations = [_obs_from_booking(r) for r in json.load(f)]
            source = "bookings"
        elif os.path.exists(epath):
            with open(epath, encoding="utf-8") as f:
                observations = [
                    o for o in (_obs_from_execution(r, by_id) for r in json.load(f))
                    if o is not None
                ]
            source = "executions"
        else:
            continue
        _fill_gauge_coords(observations, gauges)
        observations = [o for o in observations if o.get("t") or o.get("kind") == "fusion"]
        observations.sort(key=lambda o: o.get("t") or "")
        m = metrics.get(sch) or {}
        runs[sch] = {
            "source": source,
            "observations": observations,
            "utility": m.get("utility"),
            "realized_quality": m.get("realized_quality"),
            "total_cost": m.get("total_cost"),
            "n_sets_complete": m.get("n_sets_complete"),
            "n_events": m.get("n_events"),
            "task_completion_rate": m.get("task_completion_rate"),
        }
    if not runs:
        raise FileNotFoundError(f"No observation files for seed {seed} in {results_dir}")
    return {
        "results_dir": os.path.abspath(results_dir),
        "seed": seed,
        "gauges": gauges,
        "runs": runs,
        "incomplete": any(r["source"] == "executions" for r in runs.values()),
    }


def write_animation_html(results_dir: str, seed: int | None = None,
                         scheduler: str | None = None, tracks: bool = True,
                         out_path: str | None = None) -> str:
    payload = load_campaign(results_dir, seed=seed, scheduler=scheduler)
    if tracks:
        n_sats = 0
        for run in payload["runs"].values():
            enrich_tracks(run["observations"])
            n_sats += sum(1 for o in run["observations"] if o.get("track"))
        print(f"  [Viz] TLE tracks attached on {n_sats} observations")
    seed = payload["seed"]
    if out_path is None:
        tag = scheduler or "all"
        out_path = os.path.join(results_dir, f"animation_seed{seed:04d}_{tag}.html")
    html = (_HTML
            .replace("__DATA_JSON__", json.dumps(payload, ensure_ascii=False))
            .replace("__TILE_URL_JSON__", json.dumps(_basemap_tile_url())))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    src = "CARTO" if _env_value("CARTO_API_KEY") else "OSM fallback"
    print(f"  [Viz] {out_path}  ({src} basemap)")
    return out_path


_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>River-flow observation animation</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  :root {
    --bg: #0f1419; --panel: #1a2330; --ink: #e7eef8; --muted: #8aa0b8;
    --line: #2a3a4d; --sar: #3b82f6; --vis: #22c55e; --tir: #f59e0b;
    --fusion: #c084fc; --fail: #f87171; --ok: #34d399; --rej: #94a3b8;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; background: var(--bg); color: var(--ink);
    font: 13px/1.4 "Segoe UI", system-ui, sans-serif; }
  #app { display: grid; grid-template-columns: 340px 1fr; height: 100%; }
  #side { background: var(--panel); border-right: 1px solid var(--line);
    padding: 14px 14px 18px; overflow: auto; }
  #map { height: 100%; }
  h1 { font-size: 15px; margin: 0 0 4px; font-weight: 650; }
  .sub { color: var(--muted); font-size: 11px; margin-bottom: 12px; }
  .warn { background: #3a2a12; color: #fbbf24; padding: 8px 10px; border-radius: 6px;
    font-size: 11px; margin-bottom: 12px; }
  label { display: block; color: var(--muted); font-size: 11px; margin: 8px 0 4px; }
  select, button, input[type=range] { width: 100%; }
  select, button { background: #243044; color: var(--ink); border: 1px solid var(--line);
    border-radius: 6px; padding: 7px 8px; }
  button { cursor: pointer; font-weight: 600; }
  button:hover { background: #2d3d54; }
  .row { display: flex; gap: 6px; margin: 8px 0; }
  .row button { flex: 1; }
  #clock { font-variant-numeric: tabular-nums; font-size: 16px; font-weight: 650;
    margin: 6px 0 2px; }
  .legend { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 10px; margin: 10px 0 12px;
    font-size: 11px; color: var(--muted); }
  .sw { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 5px;
    vertical-align: -1px; }
  .gauges { display: flex; flex-direction: column; gap: 8px; }
  .gcard { background: #121a24; border: 1px solid var(--line); border-radius: 8px; padding: 8px 9px; }
  .gcard.done { border-color: #34d39988; }
  .gname { font-weight: 650; font-size: 12px; }
  .dots { margin-top: 5px; display: flex; gap: 6px; }
  .pill { font-size: 10px; padding: 2px 6px; border-radius: 999px; border: 1px solid var(--line);
    color: var(--muted); }
  .pill.on { color: #0b1220; font-weight: 700; }
  .pill.sar.on { background: var(--sar); border-color: var(--sar); }
  .pill.vis.on { background: var(--vis); border-color: var(--vis); }
  .pill.tir.on { background: var(--tir); border-color: var(--tir); }
  .pill.fusion.on { background: var(--fusion); border-color: var(--fusion); }
  .stats { color: var(--muted); font-size: 11px; margin: 8px 0 0; }
  .filters { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 4px; color: var(--muted); }
  .filters label { display: flex; align-items: center; gap: 4px; margin: 0; width: auto; }
  .sat-label { background: #0b1220cc; color: #fff; padding: 1px 5px; border-radius: 4px;
    font-size: 10px; white-space: nowrap; }
  .leaflet-container { background: #0b1a12; }
</style>
</head>
<body>
<div id="app">
  <aside id="side">
    <h1>River high-flow observations</h1>
    <div class="sub" id="subtitle"></div>
    <div class="warn" id="warn" hidden>
      This folder only has successful executions. Re-run the driver to dump
      bookings (fails, rejects, cancels) onto the map.
    </div>
    <label>Scheduler</label>
    <select id="scheduler"></select>
    <div class="stats" id="metrics"></div>
    <div id="clock">—</div>
    <input id="slider" type="range" min="0" max="1000" value="0"/>
    <div class="row">
      <button id="play">Play</button>
      <button id="step">Next pass</button>
    </div>
    <label>Speed</label>
    <select id="speed">
      <option value="300">300× (~2 min / 11 h)</option>
      <option value="900">900×</option>
      <option value="1800" selected>1800× (~22 s / 11 h)</option>
      <option value="3600">3600×</option>
      <option value="7200">7200×</option>
    </select>
    <div class="filters">
      <label><input type="checkbox" data-kind="sar" checked/> SAR</label>
      <label><input type="checkbox" data-kind="vis" checked/> VIS</label>
      <label><input type="checkbox" data-kind="tir" checked/> TIR</label>
      <label><input type="checkbox" data-kind="fusion" checked/> Fusion</label>
    </div>
    <div class="filters">
      <label><input type="checkbox" data-out="survived" checked/> Survived</label>
      <label><input type="checkbox" data-out="failed" checked/> Failed</label>
      <label><input type="checkbox" data-out="rejected" checked/> Rejected</label>
      <label><input type="checkbox" data-out="cancelled" checked/> Cancelled</label>
    </div>
    <div class="legend">
      <div><span class="sw" style="background:var(--sar)"></span>SAR</div>
      <div><span class="sw" style="background:var(--vis)"></span>VIS</div>
      <div><span class="sw" style="background:var(--tir)"></span>TIR</div>
      <div><span class="sw" style="background:var(--fusion)"></span>Fusion</div>
      <div><span class="sw" style="background:var(--ok)"></span>Survived</div>
      <div><span class="sw" style="background:var(--fail)"></span>Failed</div>
    </div>
    <div class="gauges" id="gauges"></div>
  </aside>
  <div id="map"></div>
</div>
<script>
const DATA = __DATA_JSON__;
const KIND_COLOR = {sar:'#3b82f6', vis:'#22c55e', tir:'#f59e0b', fusion:'#c084fc', unknown:'#64748b'};
const OUT_COLOR = {survived:'#34d399', failed:'#f87171', rejected:'#94a3b8', cancelled:'#64748b', other:'#64748b'};

function parseT(s){ return s ? Date.parse(s) : NaN; }
function fmt(ms){
  if (!isFinite(ms)) return '—';
  const d = new Date(ms);
  const p = n => String(n).padStart(2,'0');
  return d.getUTCFullYear()+'-'+p(d.getUTCMonth()+1)+'-'+p(d.getUTCDate())
    +' '+p(d.getUTCHours())+':'+p(d.getUTCMinutes())+':'+p(d.getUTCSeconds())+' UTC';
}

const TILE_URL = __TILE_URL_JSON__;
const OSM_URL = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
const map = L.map('map', {zoomControl: true, worldCopyJump: false});
const tiles = L.tileLayer(TILE_URL, {
  attribution: '&copy; OSM &copy; CARTO',
  subdomains: 'abcd',
  maxZoom: 12
}).addTo(map);
tiles.on('tileerror', function(){
  if (tiles._url === OSM_URL) return;
  tiles.setUrl(OSM_URL);
});

const gaugeLayer = L.layerGroup().addTo(map);
const histLayer = L.layerGroup().addTo(map);
const liveLayer = L.layerGroup().addTo(map);

const gauges = DATA.gauges || [];
map.setView([34.6, -90.4], 6);

const gaugeMarkers = {};
gauges.forEach(g => {
  const m = L.circleMarker([g.lat, g.lon], {
    radius: 7, color: '#e7eef8', weight: 2, fillColor: '#1a2330', fillOpacity: 0.9
  }).bindTooltip(g.name, {permanent: true, direction: 'right', offset: [8,0],
    className: 'sat-label'});
  m.addTo(gaugeLayer);
  gaugeMarkers[g.id] = m;
});
if (gauges.length) {
  map.fitBounds(L.latLngBounds(gauges.map(g => [g.lat, g.lon])).pad(0.35));
}

let currentSched = DATA.runs.stochastic_logical ? 'stochastic_logical' : Object.keys(DATA.runs)[0];
let obs = [];
let tMin = 0, tMax = 1, tNow = 0;
let playing = false, lastWall = 0, raf = 0;

const sel = document.getElementById('scheduler');
Object.keys(DATA.runs).forEach(k => {
  const o = document.createElement('option');
  o.value = k; o.textContent = k.replace(/_/g,' ');
  sel.appendChild(o);
});
document.getElementById('subtitle').textContent =
  (DATA.results_dir || '') + '  ·  seed ' + DATA.seed;
document.getElementById('warn').hidden = !DATA.incomplete;

function kindsOn(){
  return new Set([...document.querySelectorAll('[data-kind]')].filter(x=>x.checked).map(x=>x.dataset.kind));
}
function outsOn(){
  return new Set([...document.querySelectorAll('[data-out]')].filter(x=>x.checked).map(x=>x.dataset.out));
}
function visible(o){
  return kindsOn().has(o.kind) && outsOn().has(o.outcome);
}

function loadRun(name){
  currentSched = name;
  const run = DATA.runs[name];
  obs = (run.observations || []).slice().sort((a,b) => parseT(a.t) - parseT(b.t));
  const ts = obs.map(o => parseT(o.t)).filter(Number.isFinite);
  tMin = ts.length ? Math.min(...ts) - 30*1000 : Date.now();
  tMax = ts.length ? Math.max(...ts) + 8*60*1000 : tMin + 1;
  tNow = ts.length ? Math.min(...ts) : tMin;
  const m = run;
  const sets = (m.n_sets_complete!=null && m.n_events!=null)
    ? (m.n_sets_complete+'/'+m.n_events+' sets') : '';
  const u = (m.utility!=null) ? ('U='+Number(m.utility).toFixed(0)) : '';
  const q = (m.realized_quality!=null) ? ('Q='+Number(m.realized_quality).toFixed(0)) : '';
  const c = (m.total_cost!=null) ? ('cost='+Number(m.total_cost).toFixed(0)) : '';
  document.getElementById('metrics').textContent =
    [u,q,c,sets, run.source].filter(Boolean).join('  ·  ');
  document.getElementById('scheduler').value = name;
  render();
}

function interpTrack(o, t){
  const tr = o.track || [];
  if (!tr.length) {
    if (o.sat_lat!=null && o.sat_lon!=null) return [o.sat_lat, o.sat_lon];
    return null;
  }
  const pts = tr.map(p => ({t: parseT(p.t), lat: p.lat, lon: p.lon})).filter(p => Number.isFinite(p.t));
  if (!pts.length) return null;
  if (t <= pts[0].t) return [pts[0].lat, pts[0].lon];
  if (t >= pts[pts.length-1].t) return [pts[pts.length-1].lat, pts[pts.length-1].lon];
  for (let i=1;i<pts.length;i++){
    if (t <= pts[i].t){
      const a = pts[i-1], b = pts[i];
      const u = (t-a.t)/Math.max(1, b.t-a.t);
      return [a.lat + u*(b.lat-a.lat), a.lon + u*(b.lon-a.lon)];
    }
  }
  return [pts[pts.length-1].lat, pts[pts.length-1].lon];
}

function activeWindow(o){
  const t = parseT(o.t);
  const rise = parseT(o.t_rise);
  const fall = parseT(o.t_fall);
  const a = Number.isFinite(rise) ? rise : t - 4*60*1000;
  const b = Number.isFinite(fall) ? fall : t + 4*60*1000;
  return [a, b, t];
}

function render(){
  histLayer.clearLayers();
  liveLayer.clearLayers();
  document.getElementById('clock').textContent = fmt(tNow);
  const slider = document.getElementById('slider');
  slider.value = Math.round(1000 * (tNow - tMin) / Math.max(1, tMax - tMin));

  const done = {};
  gauges.forEach(g => { done[g.id] = {sar:false, vis:false, tir:false, fusion:false}; });

  obs.forEach(o => {
    if (!visible(o)) return;
    const [a,b,t] = activeWindow(o);
    const happened = tNow >= t;
    if (happened && o.outcome === 'survived' && done[o.gauge_id]) {
      done[o.gauge_id][o.kind] = true;
    }
    if (!happened) return;
    const gLat = o.lat, gLon = o.lon;
    if (gLat==null || gLon==null) return;
    const col = o.outcome === 'survived' ? KIND_COLOR[o.kind] : OUT_COLOR[o.outcome];
    const isFail = o.outcome === 'failed';
    const isLive = tNow >= a && tNow <= b;
    if (o.kind === 'fusion') {
      if (o.outcome === 'survived') {
        L.circleMarker([gLat, gLon], {radius: isLive?16:11, color: KIND_COLOR.fusion,
          weight: 2, fillColor: KIND_COLOR.fusion, fillOpacity: isLive?0.35:0.15})
          .bindTooltip('Fusion AND complete', {className:'sat-label'}).addTo(histLayer);
      }
      return;
    }
    const satLL = interpTrack(o, Math.min(Math.max(tNow, a), b));
    if (satLL && isLive) {
      L.polyline([satLL, [gLat, gLon]], {color: col, weight: 2, dashArray: '4 4',
        opacity: 0.9}).addTo(liveLayer);
      L.circleMarker(satLL, {radius: 6, color: '#fff', weight: 1, fillColor: col,
        fillOpacity: 1}).bindTooltip(
          (o.satellite||'?')+' · '+o.kind.toUpperCase()+' · '+o.outcome,
          {className:'sat-label', permanent: true, direction:'top', offset:[0,-8]}
        ).addTo(liveLayer);
    }
    const OFF = {sar:[0.14,-0.14], vis:[0.14,0.14], tir:[-0.14,0], fusion:[0,0]};
    const off = OFF[o.kind] || [0,0];
    const r = isFail ? 5 : 4;
    const mk = L.circleMarker([gLat + off[0], gLon + off[1]], {
      radius: r,
      color: isFail ? OUT_COLOR.failed : col,
      weight: isFail ? 2 : 1,
      fillColor: o.outcome === 'survived' ? col : 'transparent',
      fillOpacity: o.outcome === 'survived' ? 0.85 : 0,
      opacity: isLive ? 1 : 0.75
    });
    const q = (o.quality!=null) ? (' Q='+Number(o.quality).toFixed(0)) : '';
    mk.bindTooltip((o.satellite||'?')+' · '+o.kind.toUpperCase()+' · '+o.outcome+q,
      {className:'sat-label'});
    mk.addTo(histLayer);
  });

  const host = document.getElementById('gauges');
  host.innerHTML = '';
  gauges.forEach(g => {
    const d = done[g.id] || {};
    const andDone = d.sar && d.vis && d.tir;
    const card = document.createElement('div');
    card.className = 'gcard' + (andDone || d.fusion ? ' done' : '');
    const pill = (k, lab) => '<span class="pill '+k+(d[k]?' on':'')+'">'+lab+'</span>';
    card.innerHTML = '<div class="gname">'+g.name+'</div><div class="dots">'
      + pill('sar','SAR') + pill('vis','VIS') + pill('tir','TIR')
      + pill('fusion', andDone || d.fusion ? 'AND' : 'AND') + '</div>';
    host.appendChild(card);
    const gm = gaugeMarkers[g.id];
    if (gm) gm.setStyle({
      color: (andDone || d.fusion) ? '#34d399' : '#e7eef8',
      fillColor: (andDone || d.fusion) ? '#14532d' : '#1a2330'
    });
  });
}

function tick(now){
  if (!playing) return;
  const dt = now - lastWall;
  lastWall = now;
  const speed = Number(document.getElementById('speed').value) || 1800;
  tNow = Math.min(tMax, tNow + dt * speed);
  render();
  if (tNow >= tMax) { playing = false; document.getElementById('play').textContent = 'Play'; return; }
  raf = requestAnimationFrame(tick);
}

document.getElementById('play').onclick = () => {
  playing = !playing;
  document.getElementById('play').textContent = playing ? 'Pause' : 'Play';
  if (playing) { lastWall = performance.now(); raf = requestAnimationFrame(tick); }
  else cancelAnimationFrame(raf);
};
document.getElementById('step').onclick = () => {
  playing = false; document.getElementById('play').textContent = 'Play';
  const next = obs.filter(visible).map(o => parseT(o.t)).filter(t => t > tNow + 500).sort((a,b)=>a-b)[0];
  if (next) tNow = next;
  render();
};
document.getElementById('slider').oninput = (e) => {
  playing = false; document.getElementById('play').textContent = 'Play';
  tNow = tMin + (Number(e.target.value)/1000) * (tMax - tMin);
  render();
};
document.getElementById('scheduler').onchange = (e) => loadRun(e.target.value);
document.querySelectorAll('.filters input').forEach(el => el.onchange = () => render());

loadRun(currentSched);
playing = true;
document.getElementById('play').textContent = 'Pause';
lastWall = performance.now();
raf = requestAnimationFrame(tick);
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Write a Leaflet animation of river-flow satellite observations")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scheduler", default=None)
    parser.add_argument("--no-tracks", action="store_true",
                        help="Skip TLE ground-track sampling (faster, no moving sats).")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    write_animation_html(
        args.results_dir, seed=args.seed, scheduler=args.scheduler,
        tracks=not args.no_tracks, out_path=args.out,
    )


if __name__ == "__main__":
    main()

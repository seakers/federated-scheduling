"""Leaflet map of the earthquake damage-assessment scenario.

Shows the simulated geography the planners face:
  - epicentre + damage radius
  - selected settlements (exposure / MMI)
  - urban aim points (affected built-up area, revealed by EXTENT)
  - worst-hit districts (PHASE 2/3 aim points)
  - access aim points (SAR ACCESS tasks — same district reveal as triage/hires)

Writes a self-contained HTML file (Leaflet from a CDN). Open it in a browser.

  python earthquake_visualize.py
  python earthquake_visualize.py --out results/eq_scene.html
  python earthquake_visualize.py --results-dir results/earthquake_YYYY-mm-dd_HHMMSS
  python earthquake_visualize.py --scenario-json path/to/scenario.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from typing import Any

from earthquake_utils import (
    DISTRICT_UNCERTAINTY_KM,
    REGION_UNCERTAINTY_KM,
    URBAN_UNCERTAINTY_KM,
    fetch_usgs_events,
    load_cities,
    select_targets,
)

_OSM_TILES = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
_CARTO_TILES = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"


def _env_value(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw and raw.strip():
        return raw.strip().strip('"').strip("'")
    for path in (
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
    ):
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
    """Carto dark tiles when ``CARTO_API_KEY`` is set in the environment / ``.env``.

    Query param matches ``riverflow_visualize`` (``?key=``). HTML still falls
    back to OSM on tile errors.
    """
    key = _env_value("CARTO_API_KEY")
    if key:
        return f"{_CARTO_TILES}?key={key}"
    return _OSM_TILES


def targets_to_scenario(targets: list, sim_start: dt.datetime | None = None) -> dict[str, Any]:
    """Serialize selected targets into a JSON-friendly scenario dict."""
    if not targets:
        return {"event": None, "settlements": [], "sim_start": None}

    t0 = targets[0]
    event = {
        "id": getattr(t0, "ev_id", None),
        "lat": float(t0.ev_lat_deg),
        "lon": float(t0.ev_lon_deg),
        "magnitude": float(t0.ev_magnitude),
        "depth_km": float(t0.ev_depth_km),
        "place": getattr(t0, "ev_place", "unknown"),
        "time": str(getattr(t0, "ev_time", "")),
        "radius_km": float(t0.ev_radius_km),
    }

    settlements = []
    for t in targets:
        settlements.append({
            "id": t.name,
            "name": getattr(t, "settlement", t.name),
            "lat": float(t.lat_deg),
            "lon": float(t.lon_deg),
            "population": float(getattr(t, "population", 0.0)),
            "distance_km": float(getattr(t, "distance_km", 0.0)),
            "mmi": float(getattr(t, "mmi", 0.0)),
            "severity": float(getattr(t, "severity", 0.0)),
            "exposure": float(getattr(t, "exposure", 0.0)),
            "exposure_share": float(getattr(t, "exposure_share", 0.0)),
            "value_weight": float(getattr(t, "value_weight", 1.0)),
            # Progressive retarget aim points (simulated ground truth).
            "urban": {
                "lat": float(t.urban_lat_deg),
                "lon": float(t.urban_lon_deg),
                "uncertainty_km": float(URBAN_UNCERTAINTY_KM),
            },
            "district": {
                "lat": float(t.district_lat_deg),
                "lon": float(t.district_lon_deg),
                "uncertainty_km": float(DISTRICT_UNCERTAINTY_KM),
            },
            # ACCESS tasks observe the same district reveal (SAR logistics).
            "access": {
                "lat": float(t.district_lat_deg),
                "lon": float(t.district_lon_deg),
                "uncertainty_km": float(DISTRICT_UNCERTAINTY_KM),
                "instrument": "SAR",
            },
            "region_uncertainty_km": float(REGION_UNCERTAINTY_KM),
        })

    return {
        "sim_start": sim_start.isoformat() if sim_start else None,
        "event": event,
        "settlements": settlements,
        "phase_uncertainty_km": {
            "region": float(REGION_UNCERTAINTY_KM),
            "urban": float(URBAN_UNCERTAINTY_KM),
            "district": float(DISTRICT_UNCERTAINTY_KM),
        },
    }


def dump_scenario_json(scenario: dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scenario, f, indent=2)
    print(f"[Earthquake viz] Wrote scenario {path}")
    return path


def build_scenario_from_driver(
    sim_start: dt.datetime | None = None,
    max_targets: int = 4,
    min_magnitude: float = 6.0,
    seed: int = 0,
    horizon_h: float = 56.0,
) -> dict[str, Any]:
    """Rebuild the scenario the earthquake driver would select."""
    if sim_start is None:
        sim_start = dt.datetime(2026, 8, 1, 0, 0, 0)

    start_time = dt.datetime(2020, 1, 1)
    end_time = sim_start + dt.timedelta(hours=horizon_h)
    cache_path = os.path.join("cache", "usgs_events_historical.json")
    cities_csv = os.path.join("simplemaps_worldcities_basicv1.901", "worldcities.csv")

    events = fetch_usgs_events(
        start=start_time, end=end_time, min_magnitude=min_magnitude, cache_path=cache_path,
    )
    if os.path.exists(cities_csv):
        cities = load_cities(cities_csv, min_population=50000)
    else:
        cities = [
            {"name": "Tokyo", "lat_deg": 35.6762, "lon_deg": 139.6503, "population": 14000000},
            {"name": "Los Angeles", "lat_deg": 34.0522, "lon_deg": -118.2437, "population": 4000000},
            {"name": "San Francisco", "lat_deg": 37.7749, "lon_deg": -122.4194, "population": 880000},
            {"name": "Jakarta", "lat_deg": -6.2088, "lon_deg": 106.8456, "population": 10500000},
            {"name": "Kahramanmaraş", "lat_deg": 37.5833, "lon_deg": 36.9333, "population": 1100000},
        ]

    targets = select_targets(
        events, cities,
        max_targets=max_targets,
        max_distance_km=None,
        min_magnitude=min_magnitude,
        seed=seed,
    )
    return targets_to_scenario(targets, sim_start=sim_start)


def load_scenario(
    results_dir: str | None = None,
    scenario_json: str | None = None,
    **build_kwargs,
) -> dict[str, Any]:
    if scenario_json and os.path.isfile(scenario_json):
        with open(scenario_json, encoding="utf-8") as f:
            return json.load(f)
    if results_dir:
        cand = os.path.join(results_dir, "scenario.json")
        if os.path.isfile(cand):
            with open(cand, encoding="utf-8") as f:
                return json.load(f)
    return build_scenario_from_driver(**build_kwargs)


def write_earthquake_map_html(
    scenario: dict[str, Any],
    out_path: str,
    title: str | None = None,
) -> str:
    """Write a self-contained Leaflet HTML map for the scenario."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    data_json = json.dumps(scenario, default=str)
    tile_json = json.dumps(_basemap_tile_url())
    page_title = title or "Earthquake damage-assessment scene"

    html = _HTML
    html = html.replace("__TITLE_JSON__", json.dumps(page_title))
    html = html.replace("__DATA_JSON__", data_json)
    html = html.replace("__TILE_URL_JSON__", tile_json)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[Earthquake viz] Wrote map {out_path}")
    return out_path


def write_map_for_results_dir(
    results_dir: str,
    scenario: dict[str, Any] | None = None,
    out_name: str = "earthquake_scene.html",
) -> str:
    """Dump scenario.json + map HTML into a results directory."""
    os.makedirs(results_dir, exist_ok=True)
    if scenario is None:
        scenario = load_scenario(results_dir=results_dir)
    dump_scenario_json(scenario, os.path.join(results_dir, "scenario.json"))
    return write_earthquake_map_html(
        scenario,
        os.path.join(results_dir, out_name),
        title=f"Earthquake scene · {os.path.basename(results_dir)}",
    )


_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE_JSON__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
  :root {
    --bg: #0f1419;
    --panel: #1a2330;
    --text: #e7eef8;
    --muted: #8b9bb4;
    --epicentre: #ff4d4d;
    --settlement: #f0c14b;
    --urban: #5dade2;
    --district: #e74c3c;
    --access: #2ecc71;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; background: var(--bg); color: var(--text);
    font: 14px/1.4 "Segoe UI", system-ui, sans-serif; }
  #wrap { display: grid; grid-template-columns: 320px 1fr; height: 100%; }
  #side { background: var(--panel); padding: 16px 18px; overflow: auto;
    border-right: 1px solid #2a3544; }
  #map { height: 100%; }
  h1 { font-size: 16px; margin: 0 0 6px; font-weight: 650; }
  .sub { color: var(--muted); font-size: 12px; margin-bottom: 14px; }
  .event { background: #121820; border: 1px solid #2a3544; border-radius: 8px;
    padding: 10px 12px; margin-bottom: 14px; }
  .event .mag { color: var(--epicentre); font-weight: 700; font-size: 18px; }
  .event .meta { color: var(--muted); font-size: 12px; margin-top: 4px; }
  .legend { display: grid; gap: 8px; margin: 12px 0 16px; }
  .leg { display: flex; align-items: center; gap: 10px; font-size: 13px; }
  .swatch { width: 14px; height: 14px; border-radius: 50%; border: 2px solid #fff3; flex: 0 0 auto; }
  .swatch.epicentre { background: var(--epicentre); border-radius: 2px; transform: rotate(45deg); }
  .swatch.settlement { background: var(--settlement); }
  .swatch.urban { background: var(--urban); }
  .swatch.district { background: var(--district); }
  .swatch.access { background: var(--access); border-radius: 3px; }
  .filters label { display: flex; align-items: center; gap: 8px; margin: 4px 0;
    cursor: pointer; color: var(--text); }
  .filters input { accent-color: #6ea8fe; }
  .town { background: #121820; border: 1px solid #2a3544; border-radius: 8px;
    padding: 8px 10px; margin: 6px 0; }
  .town .name { font-weight: 600; }
  .town .row { color: var(--muted); font-size: 12px; display: flex; justify-content: space-between; }
  .hint { color: var(--muted); font-size: 11px; margin-top: 16px; line-height: 1.45; }
  .leaflet-tooltip.eq-label {
    background: #0f1419cc; color: #e7eef8; border: 1px solid #3a4a5c;
    border-radius: 4px; box-shadow: none; font-size: 11px; padding: 2px 6px;
  }
  .leaflet-tooltip.eq-label::before { border-top-color: #3a4a5c; }
  .eq-access-icon {
    background: transparent !important;
    border: none !important;
  }
  #map .leaflet-overlay-pane, #map .leaflet-marker-pane { z-index: 650; }
  @media (max-width: 800px) {
    #wrap { grid-template-columns: 1fr; grid-template-rows: auto 1fr; }
    #side { max-height: 42vh; }
  }
</style>
</head>
<body>
<div id="wrap">
  <aside id="side">
    <h1 id="title">Earthquake scene</h1>
    <div class="sub" id="subtitle"></div>
    <div class="event" id="eventBox"></div>
    <div class="legend">
      <div class="leg"><span class="swatch epicentre"></span> Epicentre + damage radius</div>
      <div class="leg"><span class="swatch settlement"></span> Selected settlements</div>
      <div class="leg"><span class="swatch urban"></span> Affected urban aim points</div>
      <div class="leg"><span class="swatch district"></span> Worst-hit districts</div>
      <div class="leg"><span class="swatch access"></span> Access (SAR) aim points</div>
    </div>
    <div class="filters">
      <label><input type="checkbox" data-layer="radius" checked> Damage radius</label>
      <label><input type="checkbox" data-layer="settlements" checked> Settlements</label>
      <label><input type="checkbox" data-layer="urban" checked> Urban areas</label>
      <label><input type="checkbox" data-layer="district" checked> Worst-hit districts</label>
      <label><input type="checkbox" data-layer="access" checked> Access points</label>
      <label><input type="checkbox" data-layer="links" checked> Retarget links</label>
      <label><input type="checkbox" data-layer="uncert" checked> Uncertainty rings</label>
    </div>
    <div id="towns"></div>
    <p class="hint">
      Urban / district / access coordinates are the simulated ground-truth aim
      points used by progressive retargeting (region → urban → district).
      ACCESS tasks observe the district location with SAR after triage succeeds.
    </p>
  </aside>
  <div id="map"></div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const DATA = __DATA_JSON__;
const TILE_URL = __TILE_URL_JSON__;
const OSM_URL = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
const TITLE = __TITLE_JSON__;

document.getElementById('title').textContent = TITLE;

if (typeof L === 'undefined') {
  document.getElementById('map').innerHTML =
    '<div style="padding:28px;color:#ff8a8a;max-width:420px;line-height:1.5">' +
    '<b>Leaflet failed to load</b> (CDN blocked). Open this HTML in Chrome or Edge ' +
    'via <code>file://</code>, not the IDE simple browser.</div>';
  throw new Error('Leaflet CDN failed to load');
}

const map = L.map('map', { zoomControl: true, worldCopyJump: false });
const tiles = L.tileLayer(TILE_URL, {
  attribution: '&copy; OSM &copy; CARTO',
  subdomains: 'abcd',
  maxZoom: 18
}).addTo(map);
let switchedToOsm = false;
tiles.on('tileerror', function(){
  if (switchedToOsm) return;
  switchedToOsm = true;
  tiles.setUrl(OSM_URL);
});

const layers = {
  radius: L.layerGroup().addTo(map),
  settlements: L.layerGroup().addTo(map),
  urban: L.layerGroup().addTo(map),
  district: L.layerGroup().addTo(map),
  access: L.layerGroup().addTo(map),
  links: L.layerGroup().addTo(map),
  uncert: L.layerGroup().addTo(map),
  epicentre: L.layerGroup().addTo(map),
};

const ev = DATA.event;
const settlements = DATA.settlements || [];

if (ev) {
  document.getElementById('subtitle').textContent =
    (DATA.sim_start ? ('sim start ' + DATA.sim_start + '  ·  ') : '') +
    (settlements.length + ' settlement(s)');
  document.getElementById('eventBox').innerHTML =
    '<div class="mag">M' + Number(ev.magnitude).toFixed(1) + '</div>' +
    '<div>' + (ev.place || 'event') + '</div>' +
    '<div class="meta">' +
      Number(ev.lat).toFixed(3) + ', ' + Number(ev.lon).toFixed(3) +
      '  ·  depth ' + Number(ev.depth_km).toFixed(0) + ' km' +
      '  ·  r≈' + Number(ev.radius_km).toFixed(0) + ' km' +
      (ev.time ? ('  ·  ' + ev.time) : '') +
    '</div>';

  const epi = L.circleMarker([ev.lat, ev.lon], {
    radius: 12, color: '#fff', weight: 3, fillColor: '#ff4d4d', fillOpacity: 1,
    pane: 'markerPane'
  }).bindTooltip('Epicentre M' + Number(ev.magnitude).toFixed(1), {
    permanent: true, direction: 'right', offset: [12,0], className: 'eq-label'
  });
  epi.addTo(layers.epicentre);

  L.circle([ev.lat, ev.lon], {
    radius: Number(ev.radius_km) * 1000,
    color: '#ff4d4d', weight: 2, dashArray: '6 6',
    fillColor: '#ff4d4d', fillOpacity: 0.10
  }).bindTooltip('Damage radius ≈ ' + Number(ev.radius_km).toFixed(0) + ' km', {
    className: 'eq-label'
  }).addTo(layers.radius);
}

function mmiColor(mmi) {
  if (mmi >= 8) return '#8b0000';
  if (mmi >= 7) return '#e74c3c';
  if (mmi >= 6) return '#e67e22';
  if (mmi >= 5) return '#f0c14b';
  return '#95a5a6';
}

const townsHost = document.getElementById('towns');
const bounds = [];

settlements.forEach((s, idx) => {
  bounds.push([s.lat, s.lon]);
  if (s.urban) bounds.push([s.urban.lat, s.urban.lon]);
  if (s.district) bounds.push([s.district.lat, s.district.lon]);

  const r = 8 + Math.min(12, Math.sqrt((s.population || 0) / 80000));
  const sett = L.circleMarker([s.lat, s.lon], {
    radius: r, color: '#ffffff', weight: 2,
    fillColor: mmiColor(s.mmi), fillOpacity: 1,
    pane: 'markerPane'
  }).bindTooltip(
    s.name + '  ·  MMI ' + Number(s.mmi).toFixed(1) +
    '  ·  pop ' + Math.round(s.population).toLocaleString(),
    { permanent: true, direction: 'right', offset: [12,0], className: 'eq-label' }
  );
  sett.bindPopup(
    '<b>' + s.name + '</b><br/>' +
    'MMI ' + Number(s.mmi).toFixed(1) + ' · severity ' + Number(s.severity).toFixed(2) + '<br/>' +
    'pop ' + Math.round(s.population).toLocaleString() +
    ' · ' + Number(s.distance_km).toFixed(0) + ' km from epicentre<br/>' +
    'value weight ' + Number(s.value_weight).toFixed(2)
  );
  sett.addTo(layers.settlements);

  L.circle([s.lat, s.lon], {
    radius: (s.region_uncertainty_km || 30) * 1000,
    color: '#f0c14b', weight: 1, dashArray: '2 4',
    fillOpacity: 0.03
  }).addTo(layers.uncert);

  if (s.urban) {
    L.circleMarker([s.urban.lat, s.urban.lon], {
      radius: 8, color: '#dff6ff', weight: 2,
      fillColor: '#5dade2', fillOpacity: 1, pane: 'markerPane'
    }).bindTooltip(s.name + ' · urban aim', {
      permanent: false, className: 'eq-label'
    }).bindPopup(
      '<b>Urban aim — ' + s.name + '</b><br/>Affected built-up area<br/>' +
      'uncertainty ±' + (s.urban.uncertainty_km || 8) + ' km'
    ).addTo(layers.urban);

    L.circle([s.urban.lat, s.urban.lon], {
      radius: (s.urban.uncertainty_km || 8) * 1000,
      color: '#5dade2', weight: 1, fillColor: '#5dade2', fillOpacity: 0.06
    }).addTo(layers.uncert);

    L.polyline([[s.lat, s.lon], [s.urban.lat, s.urban.lon]], {
      color: '#5dade2', weight: 2, opacity: 0.85, dashArray: '4 4'
    }).addTo(layers.links);
  }

  if (s.district) {
    L.circleMarker([s.district.lat, s.district.lon], {
      radius: 7, color: '#fff', weight: 2,
      fillColor: '#e74c3c', fillOpacity: 1, pane: 'markerPane'
    }).bindTooltip(s.name + ' · worst-hit district', {
      className: 'eq-label'
    }).bindPopup(
      '<b>Worst-hit district — ' + s.name + '</b><br/>TRIAGE / HIRES aim point<br/>' +
      'uncertainty ±' + (s.district.uncertainty_km || 2) + ' km'
    ).addTo(layers.district);

    L.circle([s.district.lat, s.district.lon], {
      radius: (s.district.uncertainty_km || 2) * 1000,
      color: '#e74c3c', weight: 1, fillColor: '#e74c3c', fillOpacity: 0.08
    }).addTo(layers.uncert);

    if (s.urban) {
      L.polyline([[s.urban.lat, s.urban.lon], [s.district.lat, s.district.lon]], {
        color: '#e74c3c', weight: 2, opacity: 0.85, dashArray: '4 4'
      }).addTo(layers.links);
    }
  }

  if (s.access) {
    const dlat = 0.006 * Math.cos(idx);
    const dlon = 0.006 * Math.sin(idx);
    L.marker([s.access.lat + dlat, s.access.lon + dlon], {
      icon: L.divIcon({
        className: 'eq-access-icon',
        html: '<div style="width:14px;height:14px;background:#2ecc71;border:2px solid #e7eef8;border-radius:2px;box-shadow:0 0 0 1px #14532d"></div>',
        iconSize: [14, 14],
        iconAnchor: [7, 7]
      })
    }).bindTooltip(s.name + ' · access (SAR)', {
      className: 'eq-label'
    }).bindPopup(
      '<b>Access aim — ' + s.name + '</b><br/>SAR logistics / access task<br/>' +
      'Same district reveal as triage/hires (offset on map for visibility)'
    ).addTo(layers.access);
  }

  const card = document.createElement('div');
  card.className = 'town';
  card.innerHTML =
    '<div class="name">' + s.name + '</div>' +
    '<div class="row"><span>MMI ' + Number(s.mmi).toFixed(1) + '</span>' +
      '<span>' + Number(s.distance_km).toFixed(0) + ' km</span></div>' +
    '<div class="row"><span>pop ' + Math.round(s.population).toLocaleString() + '</span>' +
      '<span>w=' + Number(s.value_weight).toFixed(2) + '</span></div>';
  card.onclick = () => {
    map.setView([s.lat, s.lon], Math.max(map.getZoom(), 11));
  };
  townsHost.appendChild(card);
});

if (ev) bounds.push([ev.lat, ev.lon]);

function refit() {
  map.invalidateSize();
  if (bounds.length) {
    map.fitBounds(L.latLngBounds(bounds).pad(0.25), { maxZoom: 9 });
  } else {
    map.setView([20, 0], 2);
  }
}
refit();
setTimeout(refit, 50);
setTimeout(refit, 250);
window.addEventListener('load', refit);
window.addEventListener('resize', () => map.invalidateSize());

document.querySelectorAll('.filters input').forEach(el => {
  el.onchange = () => {
    const name = el.dataset.layer;
    if (!layers[name]) return;
    if (el.checked) map.addLayer(layers[name]);
    else map.removeLayer(layers[name]);
  };
});
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Write a Leaflet map of the earthquake damage-assessment scene",
    )
    parser.add_argument("--results-dir", default=None,
                        help="If set, write scenario.json + HTML into this directory.")
    parser.add_argument("--scenario-json", default=None,
                        help="Load an existing scenario.json instead of rebuilding.")
    parser.add_argument("--out", default=None, help="Output HTML path.")
    parser.add_argument("--start", type=str, default=None,
                        help="ISO sim start (default 2026-08-01T00:00:00).")
    parser.add_argument("--max-targets", type=int, default=4)
    parser.add_argument("--min-magnitude", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon-h", type=float, default=56.0)
    args = parser.parse_args()

    sim_start = (
        dt.datetime.fromisoformat(args.start)
        if args.start else dt.datetime(2026, 8, 1, 0, 0, 0)
    )
    scenario = load_scenario(
        results_dir=args.results_dir,
        scenario_json=args.scenario_json,
        sim_start=sim_start,
        max_targets=args.max_targets,
        min_magnitude=args.min_magnitude,
        seed=args.seed,
        horizon_h=args.horizon_h,
    )

    if args.results_dir:
        write_map_for_results_dir(args.results_dir, scenario=scenario)
        return

    out = args.out or os.path.join(
        "results",
        f"earthquake_scene_{dt.datetime.now().strftime('%Y-%m-%d_%H%M%S')}.html",
    )
    # Also dump a sibling scenario.json next to the HTML when writing a standalone map.
    base, _ = os.path.splitext(out)
    dump_scenario_json(scenario, base + "_scenario.json")
    write_earthquake_map_html(scenario, out)


if __name__ == "__main__":
    main()

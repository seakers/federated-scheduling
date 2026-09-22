"""Offline cache for the river high-flow case study (Gorr-style).

Build once; Monte Carlo only reads ``cache/riverflow/``.

Event definition (Gorr et al.):
  - USGS Instantaneous Values discharge (00060)
  - Q75 from ~1 year of daily means
  - High-flow = contiguous intervals with Q >= Q75
  - Event ends at Q < Q75 or end of the simulation day

Default scenario day: 2026-06-17 (matches ``tles/all_tles_20260617.txt``;
many eastern CONUS gauges were above Q75 that day).

Cloud:
  - Prefer ERA5 total cloud cover at each gauge (CDS), cached hourly
  - Fallback: storm-tied synthetic TCC calibrated so VIS/TIR p_exec stays
    in the same band as volcano/earthquake (~0.65-0.95), with SAR immune

Usage:
  python riverflow_data_prep.py --build          # USGS + cloud (ERA5 if possible)
  python riverflow_data_prep.py --build --toy    # synthetic only (dev)
  python riverflow_data_prep.py --force          # rebuild even if cache exists
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import warnings

CACHE_DIR = os.path.join("cache", "riverflow")
EVENTS_JSON = os.path.join(CACHE_DIR, "events.json")
CLOUD_JSON = os.path.join(CACHE_DIR, "cloud_meta.json")
CLOUD_SERIES_JSON = os.path.join(CACHE_DIR, "cloud_series.json")
MANIFEST_JSON = os.path.join(CACHE_DIR, "manifest.json")

# Align with available TLEs. 2026-06-17 had ~16/20 wide gauges above Q75.
SCENARIO_DAY = dt.datetime(2026, 6, 17, 0, 0, 0)
SIM_DAY_END = SCENARIO_DAY + dt.timedelta(days=1)

# Kept for backward imports / toy fallback.
TOY_DAY = SCENARIO_DAY
TOY_DURATION_H = 24.0

# Curated eastern / central CONUS gauges with GRWL-class widths (m).
# Enough for a paper vignette; MILP-tractable under federated stochastic planning.
PAPER_GAUGES = [
    {"gauge_id": "07289000", "name": "Mississippi @ Vicksburg MS",
     "lat": 32.315, "lon": -90.905, "drainage_area_km2": 2_964_000.0, "river_width_m": 1400.0},
    {"gauge_id": "07374000", "name": "Mississippi @ Baton Rouge LA",
     "lat": 30.445, "lon": -91.192, "drainage_area_km2": 2_914_000.0, "river_width_m": 1100.0},
    {"gauge_id": "07032000", "name": "Mississippi @ Memphis TN",
     "lat": 35.124, "lon": -90.068, "drainage_area_km2": 2_416_000.0, "river_width_m": 900.0},
    {"gauge_id": "07010000", "name": "Mississippi @ St. Louis MO",
     "lat": 38.629, "lon": -90.180, "drainage_area_km2": 1_805_000.0, "river_width_m": 800.0},
    {"gauge_id": "05587450", "name": "Mississippi @ Grafton IL",
     "lat": 38.968, "lon": -90.429, "drainage_area_km2": 1_714_000.0, "river_width_m": 750.0},
    {"gauge_id": "05420500", "name": "Mississippi @ Clinton IA",
     "lat": 41.781, "lon": -90.252, "drainage_area_km2": 221_700.0, "river_width_m": 600.0},
    {"gauge_id": "06934500", "name": "Missouri @ Hermann MO",
     "lat": 38.710, "lon": -91.438, "drainage_area_km2": 1_353_000.0, "river_width_m": 550.0},
    {"gauge_id": "06893000", "name": "Missouri @ Kansas City MO",
     "lat": 39.112, "lon": -94.587, "drainage_area_km2": 1_256_000.0, "river_width_m": 500.0},
    {"gauge_id": "06486000", "name": "Missouri @ Omaha NE",
     "lat": 41.259, "lon": -95.922, "drainage_area_km2": 836_000.0, "river_width_m": 450.0},
    {"gauge_id": "03612600", "name": "Ohio @ Olmsted IL",
     "lat": 37.180, "lon": -89.058, "drainage_area_km2": 525_000.0, "river_width_m": 800.0},
    {"gauge_id": "03378500", "name": "Ohio @ Metropolis IL (alt)",
     "lat": 37.147, "lon": -88.741, "drainage_area_km2": 525_000.0, "river_width_m": 800.0},
    {"gauge_id": "03216600", "name": "Ohio @ Greenup KY",
     "lat": 38.577, "lon": -82.838, "drainage_area_km2": 160_500.0, "river_width_m": 450.0},
    {"gauge_id": "03303280", "name": "Ohio @ Cannelton IN",
     "lat": 37.900, "lon": -86.706, "drainage_area_km2": 251_000.0, "river_width_m": 550.0},
    {"gauge_id": "07022000", "name": "Arkansas @ Pendleton AR",
     "lat": 34.000, "lon": -91.365, "drainage_area_km2": 409_000.0, "river_width_m": 400.0},
    {"gauge_id": "02489500", "name": "Pearl @ Bogalusa LA",
     "lat": 30.793, "lon": -89.821, "drainage_area_km2": 17_000.0, "river_width_m": 280.0},
    {"gauge_id": "02492000", "name": "Pearl @ Walkiah Bluff MS",
     "lat": 30.655, "lon": -89.845, "drainage_area_km2": 20_000.0, "river_width_m": 300.0},
    {"gauge_id": "02169500", "name": "Congaree @ Columbia SC",
     "lat": 33.993, "lon": -81.025, "drainage_area_km2": 22_000.0, "river_width_m": 280.0},
    {"gauge_id": "02175000", "name": "Edisto @ Givhans SC",
     "lat": 33.032, "lon": -80.390, "drainage_area_km2": 7_070.0, "river_width_m": 250.0},
    {"gauge_id": "01646500", "name": "Potomac @ Little Falls MD",
     "lat": 38.949, "lon": -77.128, "drainage_area_km2": 29_940.0, "river_width_m": 350.0},
    {"gauge_id": "05543500", "name": "Illinois @ Marseilles IL",
     "lat": 41.328, "lon": -88.719, "drainage_area_km2": 69_270.0, "river_width_m": 320.0},
]

# Alias used by older call sites.
TOY_GAUGES = PAPER_GAUGES

# Paper subsample caps (federated MILP). Full USGS day can be larger.
MAX_EVENTS_PAPER = 20
MIN_WIDTH_M_CACHE = 250.0
MIN_DURATION_H_CACHE = 1.0  # keep Gorr short mode; planner may filter further


def _ensure_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def hydrograph_q_over_q75(frac: float, peak: float) -> float:
    """Raised-cosine toy hydrograph (dev fallback only)."""
    frac = max(0.0, min(1.0, frac))
    return 1.0 + (peak - 1.0) * 0.5 * (1.0 + math.cos(2.0 * math.pi * (frac - 0.5)))


def _naive(ts):
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.replace(tzinfo=None)
    return ts


def _discharge_col(df):
    cols = [c for c in df.columns if "00060" in str(c) and "cd" not in str(c).lower()]
    return cols[0] if cols else None


def _fetch_usgs_site(gauge_id: str):
    import urllib.parse
    import urllib.request

    url = (
        "https://waterservices.usgs.gov/nwis/site/"
        f"?format=rdb&sites={urllib.parse.quote(gauge_id)}"
        "&siteOutput=expanded&siteStatus=all"
    )
    with urllib.request.urlopen(url, timeout=30) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    lat = lon = drain = None
    header = None
    for line in text.splitlines():
        if not line or line[0] in "#":
            continue
        if line.startswith("agency_cd"):
            header = line.split("\t")
            continue
        if line.startswith("5s") or header is None:
            continue
        row = dict(zip(header, line.split("\t")))
        try:
            lat = float(row.get("dec_lat_va") or "")
            lon = float(row.get("dec_long_va") or "")
        except ValueError:
            pass
        try:
            drain = float(row.get("drain_area_va") or "") * 2.58999
        except ValueError:
            pass
        break
    return lat, lon, drain


def _contiguous_above(series, threshold):
    """Yield (start_ts, end_ts, subseries) for contiguous True runs."""
    above = series >= threshold
    if not above.any():
        return
    vals = above.astype(int).values
    idx = series.index
    i = 0
    n = len(vals)
    while i < n:
        if vals[i] != 1:
            i += 1
            continue
        j = i
        while j + 1 < n and vals[j + 1] == 1:
            j += 1
        sub = series.iloc[i:j + 1]
        yield _naive(idx[i]), _naive(idx[j]), sub
        i = j + 1


def build_usgs_events(day=SCENARIO_DAY, gauges=None, max_events=MAX_EVENTS_PAPER):
    """Pull NWIS and emit Gorr-style high-flow events for ``day`` (UTC calendar day)."""
    try:
        import dataretrieval.nwis as nwis
    except ImportError as exc:
        raise RuntimeError(
            "dataretrieval is required for USGS events. "
            "pip install dataretrieval"
        ) from exc

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    gauges = gauges or PAPER_GAUGES
    by_id = {g["gauge_id"]: g for g in gauges}
    day0 = day.date() if isinstance(day, dt.datetime) else day
    day1 = day0 + dt.timedelta(days=1)
    dv_start = (day0 - dt.timedelta(days=365)).isoformat()
    dv_end = day1.isoformat()

    events = []
    for meta in gauges:
        gid = meta["gauge_id"]
        try:
            dv = nwis.get_dv(sites=gid, parameterCd="00060",
                            start=dv_start, end=dv_end)[0]
            iv = nwis.get_iv(sites=gid, parameterCd="00060",
                            start=str(day0), end=str(day1))[0]
        except Exception as exc:
            print(f"[USGS] skip {gid}: {exc}")
            continue
        dv_col = _discharge_col(dv)
        iv_col = _discharge_col(iv)
        if dv_col is None or iv_col is None:
            print(f"[USGS] skip {gid}: missing discharge column")
            continue
        q75 = float(dv[dv_col].dropna().quantile(0.75))
        if not math.isfinite(q75) or q75 <= 0:
            print(f"[USGS] skip {gid}: bad Q75")
            continue
        series = iv[iv_col].dropna()
        if series.empty:
            continue
        # Drop timezone so clipping against naive day bounds is safe.
        try:
            series.index = series.index.tz_convert("UTC").tz_localize(None)
        except (TypeError, AttributeError):
            try:
                series.index = series.index.tz_localize(None)
            except Exception:
                pass

        lat = meta.get("lat")
        lon = meta.get("lon")
        drain = meta.get("drainage_area_km2")
        try:
            slat, slon, sdrain = _fetch_usgs_site(gid)
            lat = slat or lat
            lon = slon or lon
            drain = sdrain or drain
        except Exception:
            pass

        day_start = dt.datetime.combine(day0, dt.time(0, 0, 0))
        day_end = day_start + dt.timedelta(days=1)

        for t0, t1, sub in _contiguous_above(series, q75):
            # Gorr: event lives inside the simulation day.
            t0 = max(_naive(t0), day_start)
            t1 = min(_naive(t1), day_end - dt.timedelta(minutes=15))
            if t1 <= t0:
                continue
            sub = sub[(sub.index >= t0) & (sub.index <= t1)]
            if sub.empty:
                continue
            dur_h = max(0.25, (t1 - t0).total_seconds() / 3600.0)
            width = float(meta.get("river_width_m") or 0.0)
            if dur_h + 1e-9 < MIN_DURATION_H_CACHE:
                continue
            if width + 1e-9 < MIN_WIDTH_M_CACHE:
                continue
            q_series = [
                {"t": _naive(t).isoformat(), "Q_cfs": float(sub.loc[t])}
                for t in sub.index
            ]
            peak = float(sub.max() / q75) if q75 else 1.0
            events.append({
                "gauge_id": gid,
                "name": meta.get("name", gid),
                "lat": float(lat),
                "lon": float(lon),
                "drainage_area_km2": float(drain or 1.0),
                "river_width_m": width,
                "t_start": t0.isoformat(),
                "t_end": t1.isoformat(),
                "Q75": q75,
                "Q_peak_over_Q75": peak,
                "Q": q_series,
                "duration_h": round(dur_h, 3),
                "source": "usgs",
            })

    # Prefer longer / larger basins when capping for the MILP.
    events.sort(
        key=lambda e: (e["duration_h"], e["drainage_area_km2"]),
        reverse=True,
    )
    if max_events is not None and len(events) > max_events:
        # Stratify: keep some short (<3 h) and some long (>=8 h).
        short = [e for e in events if e["duration_h"] < 3.0]
        long = [e for e in events if e["duration_h"] >= 8.0]
        mid = [e for e in events if 3.0 <= e["duration_h"] < 8.0]
        n_short = min(len(short), max(2, max_events // 5))
        n_long = min(len(long), max(6, (2 * max_events) // 5))
        n_mid = max_events - n_short - n_long
        picked = long[:n_long] + mid[:max(0, n_mid)] + short[:n_short]
        # Fill remainder from leftovers.
        seen = {id(e) for e in picked}
        for e in events:
            if len(picked) >= max_events:
                break
            if id(e) not in seen:
                picked.append(e)
        events = picked[:max_events]
        events.sort(key=lambda e: e["t_start"])

    print(f"[USGS] {len(events)} high-flow events on {day0} "
          f"(from {len(gauges)} gauges, cap={max_events})")
    return events


def build_toy_events(day=SCENARIO_DAY, duration_h=11.0):
    """Synthetic fallback (dev only)."""
    t0 = day.replace(hour=12, minute=0, second=0, microsecond=0)
    t1 = t0 + dt.timedelta(hours=duration_h)
    events = []
    for g in PAPER_GAUGES[:5]:
        q_series = []
        steps = int(duration_h * 4) + 1
        peak = 1.7
        q75 = 100000.0
        for i in range(steps):
            frac = i / (steps - 1)
            t = t0 + dt.timedelta(hours=frac * duration_h)
            q = q75 * hydrograph_q_over_q75(frac, peak)
            q_series.append({"t": t.isoformat(), "Q_cfs": q})
        events.append({
            "gauge_id": g["gauge_id"],
            "name": g["name"],
            "lat": g["lat"],
            "lon": g["lon"],
            "drainage_area_km2": g["drainage_area_km2"],
            "river_width_m": g["river_width_m"],
            "t_start": t0.isoformat(),
            "t_end": t1.isoformat(),
            "Q75": q75,
            "Q_peak_over_Q75": peak,
            "Q": q_series,
            "duration_h": duration_h,
            "source": "toy",
        })
    return events


def _load_dotenv_key(name: str):
    raw = os.environ.get(name)
    if raw and raw.strip():
        return raw.strip().strip('"').strip("'")
    for path in (".env", os.path.join(os.path.dirname(__file__), ".env")):
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k.strip() == name:
                    return v.strip().strip('"').strip("'") or None
    return None


def try_era5_tcc(events, day=SCENARIO_DAY):
    """Fetch ERA5 total cloud cover at event gauges. Returns series or None."""
    try:
        import cdsapi
        import pandas as pd
        import xarray as xr
    except ImportError as exc:
        print(f"[ERA5] missing dependency ({exc}); using synthetic cloud")
        return None

    key = _load_dotenv_key("CDS_API_KEY")
    if not key:
        print("[ERA5] CDS_API_KEY not set; using synthetic cloud")
        return None

    cds_path = os.path.join(os.path.expanduser("~"), ".cdsapirc")
    with open(cds_path, "w", encoding="utf-8") as f:
        f.write(f"url: https://cds.climate.copernicus.eu/api\nkey: {key}\n")

    day0 = day.date() if isinstance(day, dt.datetime) else day
    lats = [e["lat"] for e in events]
    lons = [e["lon"] for e in events]
    area = [max(lats) + 0.75, min(lons) - 0.75, min(lats) - 0.75, max(lons) + 0.75]
    hours = [f"{h:02d}:00" for h in range(24)]
    out_nc = os.path.join(CACHE_DIR, f"era5_tcc_{day0.isoformat()}.nc")

    if not os.path.isfile(out_nc):
        print(f"[ERA5] requesting TCC for {day0} ...")
        try:
            c = cdsapi.Client()
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": "total_cloud_cover",
                    "year": str(day0.year),
                    "month": f"{day0.month:02d}",
                    "day": f"{day0.day:02d}",
                    "time": hours,
                    "area": area,
                    "format": "netcdf",
                },
                out_nc,
            )
        except Exception as exc:
            print(f"[ERA5] retrieve failed: {exc}")
            return None

    try:
        ds = xr.open_dataset(out_nc)
        var = next((c for c in ("tcc", "TCC", "total_cloud_cover") if c in ds),
                   list(ds.data_vars)[0])
        da = ds[var]
        lat_name = "latitude" if "latitude" in da.coords else "lat"
        lon_name = "longitude" if "longitude" in da.coords else "lon"
        time_name = next(
            (c for c in ("valid_time", "time", "forecast_period") if c in da.coords),
            None,
        )
        if time_name is None:
            raise KeyError(f"no time coordinate in {list(da.coords)}")
        series = {}
        for e in events:
            pt = da.sel({lat_name: e["lat"], lon_name: e["lon"]}, method="nearest")
            rows = []
            for t, val in zip(pt[time_name].values, pt.values.ravel()):
                tcc = float(val)
                if tcc > 1.5:
                    tcc /= 100.0
                tcc = max(0.0, min(1.0, tcc))
                ts = _naive(pd.Timestamp(t).to_pydatetime())
                rows.append({"t": ts.isoformat(), "tcc": round(tcc, 4)})
            series[e["gauge_id"]] = rows
        ds.close()
        print(f"[ERA5] TCC for {len(series)} gauges")
        return series
    except Exception as exc:
        print(f"[ERA5] parse failed: {exc}")
        return None


def write_cloud_cache(events, series=None):
    """Write cloud meta + optional per-gauge ERA5 series.

    Synthetic fallback is milder than the old toy (mean event TCC 0.30 →
    p_clear ~0.70) so VIS/TIR sit in the volcano/EQ execution band while SAR
    stays cloud-immune in the execution callback.
    """
    _ensure_dir()
    if series:
        meta = {
            "mode": "era5",
            "note": "ERA5 total cloud cover at gauge locations (hourly).",
            "tcc_event_mean": 0.30,
            "tcc_event_span": 0.10,
            "tcc_clear_mean": 0.20,
            "tcc_clear_span": 0.08,
        }
        with open(CLOUD_SERIES_JSON, "w", encoding="utf-8") as f:
            json.dump({"series": series}, f, indent=2)
    else:
        meta = {
            "mode": "synthetic_storm_calibrated",
            "note": (
                "Storm-tied TCC for high-flow windows. Calibrated so mean "
                "p_clear ≈ 0.70 (VIS/TIR) matches volcano/earthquake p_exec "
                "order of magnitude; SAR ignores cloud in the exec callback."
            ),
            "tcc_event_mean": 0.30,
            "tcc_event_span": 0.12,
            "tcc_clear_mean": 0.18,
            "tcc_clear_span": 0.08,
        }
        if os.path.isfile(CLOUD_SERIES_JSON):
            try:
                os.remove(CLOUD_SERIES_JSON)
            except OSError:
                pass
    with open(CLOUD_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def write_events_cache(events, day=SCENARIO_DAY):
    _ensure_dir()
    day_s = day.isoformat() if isinstance(day, dt.datetime) else str(day)
    with open(EVENTS_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "events": events,
            "day": day_s,
            "definition": "Q >= Q75 contiguous (Gorr et al.)",
            "n_events": len(events),
        }, f, indent=2)
    print(f"[riverflow_data_prep] wrote {len(events)} events -> {EVENTS_JSON}")


def write_manifest(day, events, cloud_meta):
    with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "scenario_day": day.isoformat() if isinstance(day, dt.datetime) else str(day),
            "n_events": len(events),
            "sources": sorted({e.get("source", "?") for e in events}),
            "cloud_mode": cloud_meta.get("mode"),
            "duration_h": {
                "min": min((e.get("duration_h") or 0) for e in events) if events else 0,
                "max": max((e.get("duration_h") or 0) for e in events) if events else 0,
                "mean": (sum(e.get("duration_h") or 0 for e in events) / len(events))
                        if events else 0,
            },
            "built_at": dt.datetime.utcnow().isoformat() + "Z",
        }, f, indent=2)


def write_toy_cache():
    """Dev fallback only — do not call from the Monte Carlo driver."""
    events = build_toy_events()
    write_events_cache(events, SCENARIO_DAY)
    meta = write_cloud_cache(events, series=None)
    write_manifest(SCENARIO_DAY, events, meta)
    return EVENTS_JSON


def build_real_cache(day=SCENARIO_DAY, max_events=MAX_EVENTS_PAPER, try_era5=True):
    """USGS events + ERA5 (or calibrated synthetic) cloud. Paper path."""
    _ensure_dir()
    events = build_usgs_events(day=day, max_events=max_events)
    if not events:
        raise RuntimeError(
            f"No USGS high-flow events on {day}. "
            "Pick another day or widen PAPER_GAUGES."
        )
    series = try_era5_tcc(events, day=day) if try_era5 else None
    meta = write_cloud_cache(events, series=series)
    write_events_cache(events, day)
    write_manifest(day, events, meta)
    return EVENTS_JSON


def cache_is_real():
    if not os.path.isfile(EVENTS_JSON):
        return False
    try:
        with open(EVENTS_JSON, encoding="utf-8") as f:
            payload = json.load(f)
        ev = payload.get("events") or []
        if not ev:
            return False
        return all(e.get("source") == "usgs" for e in ev)
    except Exception:
        return False


def ensure_cache(force=False, allow_toy=False):
    """Idempotent cache ensure. Prefer USGS; never overwrite real with toy."""
    if force:
        return build_real_cache()
    if cache_is_real():
        print(f"[riverflow_data_prep] using existing USGS cache ({EVENTS_JSON})")
        return EVENTS_JSON
    if os.path.isfile(EVENTS_JSON) and not allow_toy:
        # Stale toy cache — replace.
        print("[riverflow_data_prep] replacing non-USGS cache with USGS build")
        return build_real_cache()
    try:
        return build_real_cache()
    except Exception as exc:
        if not allow_toy:
            raise
        print(f"[riverflow_data_prep] USGS build failed ({exc}); writing toy")
        return write_toy_cache()


def load_cached_events(path=EVENTS_JSON):
    if not os.path.exists(path):
        ensure_cache(allow_toy=True)
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload["events"]


def load_cloud_meta(path=CLOUD_JSON):
    if not os.path.exists(path):
        ensure_cache(allow_toy=True)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_cloud_series(path=CLOUD_SERIES_JSON):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("series")


def load_scenario_day():
    if os.path.isfile(EVENTS_JSON):
        with open(EVENTS_JSON, encoding="utf-8") as f:
            day = json.load(f).get("day")
        if day:
            return dt.datetime.fromisoformat(day.replace("Z", ""))
    return SCENARIO_DAY


# Back-compat name used by older scripts.
def try_live_usgs(gauge_ids, day, duration_h=24.0):
    gauges = [g for g in PAPER_GAUGES if g["gauge_id"] in set(gauge_ids)]
    if not gauges:
        gauges = [{"gauge_id": gid, "name": gid, "lat": 35.0, "lon": -90.0,
                   "drainage_area_km2": 1.0, "river_width_m": 300.0}
                  for gid in gauge_ids]
    day_dt = day if isinstance(day, dt.datetime) else dt.datetime.combine(day, dt.time())
    return build_usgs_events(day=day_dt, gauges=gauges, max_events=None)


def main():
    parser = argparse.ArgumentParser(description="Build riverflow USGS/ERA5 cache")
    parser.add_argument("--build", action="store_true",
                        help="Build USGS cache for SCENARIO_DAY")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if a USGS cache exists")
    parser.add_argument("--toy", action="store_true",
                        help="Write synthetic toy cache (dev only)")
    parser.add_argument("--no-era5", action="store_true",
                        help="Skip ERA5; use calibrated synthetic TCC")
    parser.add_argument("--day", type=str, default=None,
                        help="YYYY-MM-DD (default 2026-06-17)")
    parser.add_argument("--max-events", type=int, default=MAX_EVENTS_PAPER)
    args = parser.parse_args()

    day = SCENARIO_DAY
    if args.day:
        day = dt.datetime.fromisoformat(args.day)

    if args.toy:
        write_toy_cache()
        return
    if args.build or args.force or not cache_is_real():
        build_real_cache(day=day, max_events=args.max_events,
                         try_era5=not args.no_era5)
    else:
        print(f"[riverflow_data_prep] cache already USGS at {EVENTS_JSON}")


if __name__ == "__main__":
    main()

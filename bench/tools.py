"""
Fixture-backed copies of the app's tools, for the routing benchmark.

Each tool here takes its name, description and argument schema from the real
tool in astro_tools.py, so the model sees exactly the interface the app
exposes. Only the body is swapped for deterministic fixture data.

The thing being measured is which tool the model chooses and with what
arguments, not whether Open-Meteo is up. Fixtures make runs reproducible,
free, fast, and remove third-party flakiness as a confound.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import astro_tools as real  # noqa: E402

# Bump when fixture outputs change. The harness records it on every run and the
# scorer warns when one result mixes runs made against different fixtures.
FIXTURE_VERSION = 2

# Deterministic fixture data. Weather is fixed; target positions and the
# darkness window are computed (see _altitude and _sun_altitude) so that a
# model checking them against what it knows of the sky finds them plausible.
CITIES = {
    "austin": {"lat": 30.267, "lon": -97.743, "tz": "America/Chicago"},
    "princeton": {"lat": 40.357, "lon": -74.667, "tz": "America/New_York"},
    "flagstaff": {"lat": 35.198, "lon": -111.651, "tz": "America/Phoenix"},
    "seattle": {"lat": 47.606, "lon": -122.332, "tz": "America/Los_Angeles"},
    "denver": {"lat": 39.739, "lon": -104.990, "tz": "America/Denver"},
}

# Common names plus the catalog designations a model is likely to use instead.
_OBJECTS = {
    ("orion nebula", "m42", "ngc 1976"): {"ra": 5.588, "dec": -5.39, "mag": 4.0},
    ("andromeda galaxy", "andromeda", "m31", "ngc 224"): {"ra": 0.712, "dec": 41.27, "mag": 3.4},
    ("ring nebula", "m57", "ngc 6720"): {"ra": 18.893, "dec": 33.03, "mag": 8.8},
    ("whirlpool galaxy", "whirlpool", "m51", "ngc 5194"): {"ra": 13.498, "dec": 47.20, "mag": 8.4},
    ("pleiades", "m45", "seven sisters"): {"ra": 3.790, "dec": 24.12, "mag": 1.6},
}

_CALL_LOG: list[dict[str, Any]] = []


def reset_log() -> None:
    _CALL_LOG.clear()


def drain_log() -> list[dict[str, Any]]:
    out = list(_CALL_LOG)
    _CALL_LOG.clear()
    return out


def record(name: str, args: dict[str, Any]) -> None:
    _CALL_LOG.append({"tool": name, "args": args})


def city(location: str) -> dict[str, Any] | None:
    key = (location or "").strip().lower()
    for name, data in CITIES.items():
        if name in key:
            return {"name": name, **data}
    return None


def _object(name: str) -> dict[str, Any] | None:
    key = (name or "").strip().lower().replace("messier ", "m")
    for aliases, data in _OBJECTS.items():
        if any(a == key or a in key for a in aliases):
            return data
    return None


# --- low-precision astronomy (good to about a degree / a few minutes) -------
# The same sampling as the real tools in astro_tools.py, with astropy swapped
# for textbook formulae so fixtures stay offline and deterministic.


def _night_start(date: str | None, lon: float) -> datetime:
    """About 18:00 local solar time on `date`, in UTC (as astro_tools does)."""
    day = datetime.strptime(date or "2026-01-01", "%Y-%m-%d")
    return day + timedelta(hours=18 - lon / 15.0)


def _days_since_j2000(t: datetime) -> float:
    return (t - datetime(2000, 1, 1, 12)).total_seconds() / 86400.0


def _altitude(ra_h: float, dec: float, lat: float, lon: float, t: datetime) -> float:
    gmst = (18.697374558 + 24.06570982441908 * _days_since_j2000(t)) % 24
    hour_angle = math.radians(((gmst + lon / 15.0 - ra_h) % 24) * 15)
    phi, delta = math.radians(lat), math.radians(dec)
    return math.degrees(math.asin(math.sin(phi) * math.sin(delta)
                                  + math.cos(phi) * math.cos(delta) * math.cos(hour_angle)))


def _sun_ra_dec(t: datetime) -> tuple[float, float]:
    d = _days_since_j2000(t)
    g = math.radians((357.529 + 0.98560028 * d) % 360)
    q = (280.459 + 0.98564736 * d) % 360
    ecl_lon = math.radians(q + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g))
    eps = math.radians(23.439 - 0.00000036 * d)
    ra = math.degrees(math.atan2(math.cos(eps) * math.sin(ecl_lon), math.cos(ecl_lon))) % 360
    dec = math.degrees(math.asin(math.sin(eps) * math.sin(ecl_lon)))
    return ra / 15.0, dec


def _sun_altitude(lat: float, lon: float, t: datetime) -> float:
    ra, dec = _sun_ra_dec(t)
    return _altitude(ra, dec, lat, lon, t)


def geocode_location(location: str) -> str:
    record("geocode_location", {"location": location})
    hit = city(location)
    if not hit:
        return f"No match for {location!r}. Try a city name."
    return (f"{hit['name'].title()}: lat={hit['lat']}, lon={hit['lon']}, "
            f"timezone={hit['tz']}")


def get_weather_conditions(lat: float, lon: float, date: str | None = None) -> str:
    record("get_weather_conditions", {"lat": lat, "lon": lon, "date": date})
    cloud = 15 if lat < 40 else 68
    return (f"Hourly cloud cover for Lat: {lat}, Lon: {lon}: mean {cloud}% "
            f"from 18:00 to 06:00 local.")


def get_target_transit(target_name: str, lat: float, lon: float, date: str | None = None) -> str:
    record("get_target_transit", {"target_name": target_name, "lat": lat,
                                  "lon": lon, "date": date})
    obj = _object(target_name)
    if not obj:
        return f"Error resolving target '{target_name}'. Ensure it is a valid catalog name."
    # As astro_tools does: hourly samples from ~18:00 local, peak = highest sample.
    start = _night_start(date, lon)
    times = [start + timedelta(hours=h) for h in range(12)]
    alts = [_altitude(obj["ra"], obj["dec"], lat, lon, t) for t in times]
    peak = max(range(12), key=alts.__getitem__)
    return (f"Target: {target_name}\n"
            f"Observatory Location: Lat {lat}, Lon {lon}\n"
            f"Peak Altitude: {alts[peak]:.2f} degrees\n"
            f"Transit Time (UTC): {times[peak]:%Y-%m-%d %H:%M:%S}\n"
            f"Target is above 30 degrees for: {sum(a > 30 for a in alts)} of the next 12 hours.")


def get_moon_phase(date: str) -> str:
    record("get_moon_phase", {"date": date})
    return f"{date}: moon 12% illuminated, waning."


def get_darkness_window(lat: float, lon: float, date: str) -> str:
    record("get_darkness_window", {"lat": lat, "lon": lon, "date": date})
    # As astro_tools does: 5-minute samples over 24 h from local solar noon.
    noon = _night_start(date, lon) - timedelta(hours=6)
    times = [noon + timedelta(minutes=5 * i) for i in range(24 * 12)]
    dark = [t for t in times if _sun_altitude(lat, lon, t) < -18]
    if not dark:
        return f"No astronomical darkness at Lat {lat}, Lon {lon} on the night of {date}."
    offset = timedelta(hours=lon / 15.0)
    fmt = lambda t: f"{t:%Y-%m-%d %H:%M} UTC (~{t + offset:%H:%M} local solar)"
    return (f"Astronomical darkness {fmt(dark[0])} to {fmt(dark[-1])}, "
            f"{len(dark) * 5 / 60:.1f} hours.")


def get_field_of_view(focal_length_mm: float, sensor: str) -> str:
    record("get_field_of_view", {"focal_length_mm": focal_length_mm, "sensor": sensor})
    return real.get_field_of_view.func(focal_length_mm, sensor)  # pure maths, no network


def save_session_plan(target_name: str, start_time: str, end_time: str, notes: str = "") -> str:
    record("save_session_plan", {"target_name": target_name, "start_time": start_time,
                                 "end_time": end_time, "notes": notes})
    return f"Session plan saved: {target_name} from {start_time} to {end_time}."


_FIXTURES = {fn.__name__: fn for fn in (
    geocode_location, get_weather_conditions, get_target_transit, get_moon_phase,
    get_darkness_window, get_field_of_view, save_session_plan,
)}

missing = {t.name for t in real.ALL_TOOLS} ^ set(_FIXTURES)
if missing:
    raise RuntimeError(f"bench fixtures out of sync with astro_tools: {sorted(missing)}")

ALL_TOOLS = [
    StructuredTool.from_function(
        func=_FIXTURES[t.name], name=t.name, description=t.description,
        args_schema=t.args_schema,
    )
    for t in real.ALL_TOOLS
]

TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}

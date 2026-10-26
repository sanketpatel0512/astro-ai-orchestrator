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

import sys
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import astro_tools as real  # noqa: E402

# Deterministic fixture data. Values are plausible but fixed.
CITIES = {
    "austin": {"lat": 30.267, "lon": -97.743, "tz": "America/Chicago"},
    "princeton": {"lat": 40.357, "lon": -74.667, "tz": "America/New_York"},
    "flagstaff": {"lat": 35.198, "lon": -111.651, "tz": "America/Phoenix"},
    "seattle": {"lat": 47.606, "lon": -122.332, "tz": "America/Los_Angeles"},
    "denver": {"lat": 39.739, "lon": -104.990, "tz": "America/Denver"},
}

# Common names plus the catalog designations a model is likely to use instead.
_OBJECTS = {
    ("orion nebula", "m42", "ngc 1976"): {"dec": -5.39, "mag": 4.0},
    ("andromeda galaxy", "andromeda", "m31", "ngc 224"): {"dec": 41.27, "mag": 3.4},
    ("ring nebula", "m57", "ngc 6720"): {"dec": 33.03, "mag": 8.8},
    ("whirlpool galaxy", "whirlpool", "m51", "ngc 5194"): {"dec": 47.20, "mag": 8.4},
    ("pleiades", "m45", "seven sisters"): {"dec": 24.12, "mag": 1.6},
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
    peak = round(90 - abs(lat - obj["dec"]), 1)
    return (f"Target: {target_name}\nPeak Altitude: {peak} degrees\n"
            f"Transit Time (UTC): 07:40\nTarget is above 30 degrees for: "
            f"{6 if peak > 45 else 3} of the next 12 hours.")


def get_moon_phase(date: str) -> str:
    record("get_moon_phase", {"date": date})
    return f"{date}: moon 12% illuminated, waning."


def get_darkness_window(lat: float, lon: float, date: str) -> str:
    record("get_darkness_window", {"lat": lat, "lon": lon, "date": date})
    hours = 10.4 if lat < 40 else 8.9
    return f"Astronomical darkness 01:05 UTC to 11:25 UTC on the night of {date}, {hours} hours."


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

"""
Agent tools shared by the Streamlit app and the routing benchmark.

The app binds these directly. The benchmark (bench/tools.py) reuses each tool's
name, description and argument schema but swaps in deterministic fixture
bodies, so what the benchmark measures is routing over exactly the interface
the app exposes.

Heavy imports (astropy, requests) are deferred to call time so the benchmark
can import the schemas without installing them.
"""

import math
import re
from datetime import datetime, timedelta, timezone

from langchain_core.tools import tool

TWILIGHT_DEG = -18.0  # astronomical twilight

# Sensor width/height in mm, plus pixel size in microns where known.
SENSORS = {
    "asi2600": (23.5, 15.7, 3.76),
    "asi6200": (36.0, 24.0, 3.76),
    "asi533": (11.31, 11.31, 3.76),
    "asi294": (19.1, 13.0, 4.63),
    "asi183": (13.2, 8.8, 2.4),
    "asi585": (11.2, 6.3, 2.9),
    "fullframe": (36.0, 24.0, None),
    "apsc": (23.5, 15.6, None),
    "microfourthirds": (17.3, 13.0, None),
    "m43": (17.3, 13.0, None),
}


def _approx_utc_offset_h(lon: float) -> float:
    """Mean solar offset from UTC. Good enough to centre a night window."""
    return lon / 15.0


def _evening_start_utc(date: str, lon: float) -> datetime:
    """Roughly 18:00 local on `date`, expressed in UTC."""
    day = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return day + timedelta(hours=18 - _approx_utc_offset_h(lon))


def field_of_view(focal_length_mm: float, sensor: str) -> dict | None:
    """Pure calculation, shared with the benchmark fixtures."""
    key = re.sub(r"[^a-z0-9]", "", (sensor or "").lower())
    for name, (w, h, pixel_um) in SENSORS.items():
        if name in key:
            fov_w = 2 * math.degrees(math.atan(w / (2 * focal_length_mm))) * 60
            fov_h = 2 * math.degrees(math.atan(h / (2 * focal_length_mm))) * 60
            scale = 206.265 * pixel_um / focal_length_mm if pixel_um else None
            return {"sensor": name, "width_arcmin": round(fov_w, 1),
                    "height_arcmin": round(fov_h, 1),
                    "pixel_scale": round(scale, 2) if scale else None}
    return None


@tool
def geocode_location(location: str) -> str:
    """Resolve a place name to latitude, longitude and timezone.

    Call this first whenever the user names a place rather than giving
    coordinates; the weather, transit and darkness tools all need coordinates.
    """
    import requests

    candidates = [location, location.split(",")[0]]
    for name in dict.fromkeys(c.strip() for c in candidates if c.strip()):
        resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": name, "count": 1}, timeout=15,
        )
        if resp.status_code == 200 and resp.json().get("results"):
            hit = resp.json()["results"][0]
            region = ", ".join(p for p in (hit.get("admin1"), hit.get("country")) if p)
            return (f"{hit['name']} ({region}): lat={hit['latitude']:.4f}, "
                    f"lon={hit['longitude']:.4f}, timezone={hit.get('timezone')}")
    return f"No match for {location!r}. Try a city name."


@tool
def get_weather_conditions(lat: float, lon: float, date: str | None = None) -> str:
    """Fetches hour-by-hour cloud cover for a latitude and longitude.

    Without `date` (YYYY-MM-DD) it covers the next 12 hours; with it, that
    night from 18:00 local. Forecasts only reach about 16 days ahead.
    """
    import requests

    params = {"latitude": lat, "longitude": lon, "hourly": "cloudcover",
              "timezone": "auto"}
    if date:
        next_day = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        params.update(start_date=date, end_date=next_day)
    else:
        params["forecast_days"] = 2

    response = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=15)
    if response.status_code != 200:
        return f"Error: could not fetch weather data ({response.status_code}). Forecasts reach ~16 days ahead."
    data = response.json()
    times = data["hourly"]["time"]
    cloud_cover = data["hourly"]["cloudcover"]

    # Index from the location's local clock, not this machine's.
    if date:
        start_key = f"{date}T18:00"
    else:
        local_now = datetime.now(timezone.utc) + timedelta(seconds=data.get("utc_offset_seconds", 0))
        start_key = local_now.strftime("%Y-%m-%dT%H:00")
    start = times.index(start_key) if start_key in times else 0

    report = f"Hourly cloud cover (%) for Lat: {lat}, Lon: {lon} (local time):\n"
    for i in range(start, min(start + 12, len(times))):
        report += f"{times[i]}: {cloud_cover[i]}% clouds\n"
    return report


@tool
def get_target_transit(target_name: str, lat: float, lon: float, date: str | None = None) -> str:
    """Calculates the transit time, peak altitude and visibility of a celestial target for a latitude and longitude.

    Without `date` (YYYY-MM-DD) it covers the next 12 hours; with it, that
    night from about 18:00 local.
    """
    try:
        from astropy import units as u
        from astropy.coordinates import AltAz, EarthLocation, SkyCoord
        from astropy.time import Time

        observatory = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=200 * u.m)
        start = Time(_evening_start_utc(date, lon)) if date else Time.now()
        target = SkyCoord.from_name(target_name)

        times = start + range(0, 12) * u.hour
        target_altaz = target.transform_to(AltAz(obstime=times, location=observatory))

        max_alt_idx = target_altaz.alt.argmax()
        max_alt = target_altaz.alt[max_alt_idx].degree
        transit_time = times[max_alt_idx].iso

        return (f"Target: {target_name}\n"
                f"Observatory Location: Lat {lat}, Lon {lon}\n"
                f"Peak Altitude: {max_alt:.2f} degrees\n"
                f"Transit Time (UTC): {transit_time}\n"
                f"Target is above 30 degrees for: {sum(target_altaz.alt.degree > 30)} of the next 12 hours.")
    except Exception as e:
        return f"Error resolving target '{target_name}'. Ensure it is a valid catalog name. Details: {e}"


@tool
def get_moon_phase(date: str) -> str:
    """Moon illumination and phase (waxing/waning) for a date (YYYY-MM-DD)."""
    from astropy.coordinates import get_body
    from astropy.time import Time

    def illumination(t):
        sun, moon = get_body("sun", t), get_body("moon", t)
        elong = sun.separation(moon).rad
        phase_angle = math.atan2(sun.distance.value * math.sin(elong),
                                 moon.distance.to(sun.distance.unit).value
                                 - sun.distance.value * math.cos(elong))
        return (1 + math.cos(phase_angle)) / 2

    t = Time(f"{date}T12:00:00")
    now, later = illumination(t), illumination(t + 1)
    trend = "waxing" if later > now else "waning"
    return f"{date}: moon {now:.0%} illuminated, {trend}."


@tool
def get_darkness_window(lat: float, lon: float, date: str) -> str:
    """Start and end of astronomical darkness (sun below -18 degrees) for the night of a date (YYYY-MM-DD) at a latitude and longitude."""
    from astropy import units as u
    from astropy.coordinates import AltAz, EarthLocation, get_body
    from astropy.time import Time

    site = EarthLocation(lat=lat * u.deg, lon=lon * u.deg)
    noon = _evening_start_utc(date, lon) - timedelta(hours=6)
    times = Time(noon) + (range(0, 24 * 12) * u.min * 5)
    sun_alt = get_body("sun", times).transform_to(AltAz(obstime=times, location=site)).alt.degree

    dark = [t for t, alt in zip(times, sun_alt) if alt < TWILIGHT_DEG]
    if not dark:
        return f"No astronomical darkness at Lat {lat}, Lon {lon} on the night of {date}."
    offset = _approx_utc_offset_h(lon)
    fmt = lambda t: f"{t.datetime:%Y-%m-%d %H:%M} UTC (~{(t.datetime + timedelta(hours=offset)):%H:%M} local solar)"
    hours = len(dark) * 5 / 60
    return f"Astronomical darkness {fmt(dark[0])} to {fmt(dark[-1])}, {hours:.1f} hours."


@tool
def get_field_of_view(focal_length_mm: float, sensor: str) -> str:
    """Field of view (and pixel scale where known) for a telescope focal length and camera sensor, e.g. 'APS-C', 'full frame', 'ASI2600MC Pro'."""
    fov = field_of_view(focal_length_mm, sensor)
    if not fov:
        return f"Unknown sensor {sensor!r}. Known: {', '.join(SENSORS)}."
    scale = f", pixel scale {fov['pixel_scale']} arcsec/px" if fov["pixel_scale"] else ""
    return (f"{focal_length_mm:g}mm on {sensor}: field of view "
            f"{fov['width_arcmin']} x {fov['height_arcmin']} arcmin{scale}.")


@tool
def save_session_plan(target_name: str, start_time: str, end_time: str, notes: str = "") -> str:
    """Record the final imaging session plan. Call this last, once, and only when the user asks for a plan."""
    return f"Session plan saved: {target_name} from {start_time} to {end_time}. {notes}".strip()


ALL_TOOLS = [
    geocode_location,
    get_weather_conditions,
    get_target_transit,
    get_moon_phase,
    get_darkness_window,
    get_field_of_view,
    save_session_plan,
]

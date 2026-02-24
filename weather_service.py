"""
weather_service.py
==================
Fetches current weather from Open-Meteo (free, no API key) and
the Open-Meteo Marine API for sea conditions.
Results are cached per (lat, lng, hour) bucket so a burst of
concurrent requests doesn't hammer the upstream API.
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# In-process cache:  key → (data, expiry_unix)
_weather_cache: Dict[str, Tuple[Any, float]] = {}
CACHE_TTL_SECONDS = 1800  # 30 min — weather doesn't change faster


def _cache_key(lat: float, lng: float) -> str:
    # Round to ~1 km resolution (~0.01°)
    return f"{round(lat, 2)}_{round(lng, 2)}"


def _get_cached(lat: float, lng: float) -> Optional[Dict]:
    key = _cache_key(lat, lng)
    if key in _weather_cache:
        data, exp = _weather_cache[key]
        if time.time() < exp:
            return data
        del _weather_cache[key]
    return None


def _set_cached(lat: float, lng: float, data: Dict) -> None:
    _weather_cache[_cache_key(lat, lng)] = (data, time.time() + CACHE_TTL_SECONDS)


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────
def get_weather(lat: float, lng: float) -> Dict[str, Any]:
    """
    Returns a normalised weather dict:
        {
            temperature_c: float,
            wind_speed_kmh: float,
            wind_direction_deg: float,
            wave_height_m: float,       # from marine API
            sea_surface_temp_c: float,  # from marine API
            weather_code: int,
            is_day: bool,
            source: str,
        }
    Returns empty dict on failure (graceful degradation).
    """
    cached = _get_cached(lat, lng)
    if cached:
        return cached

    data = {}

    # ── Atmospheric weather ───────────────────────────────────────────────────
    try:
        atm_url = "https://api.open-meteo.com/v1/forecast"
        atm_params = {
            "latitude": lat,
            "longitude": lng,
            "current_weather": True,
            "current": [
                "temperature_2m",
                "wind_speed_10m",
                "wind_direction_10m",
                "weather_code",
                "is_day",
            ],
            "wind_speed_unit": "kmh",
            "timezone": "auto",
        }
        r = requests.get(atm_url, params=atm_params, timeout=5)
        r.raise_for_status()
        atm = r.json()
        current = atm.get("current", atm.get("current_weather", {}))
        data["temperature_c"] = current.get(
            "temperature_2m", current.get("temperature")
        )
        data["wind_speed_kmh"] = current.get(
            "wind_speed_10m", current.get("windspeed")
        )
        data["wind_direction_deg"] = current.get("wind_direction_10m")
        data["weather_code"] = current.get("weather_code", current.get("weathercode"))
        data["is_day"] = bool(current.get("is_day", 1))
    except Exception as exc:
        logger.warning("Atmospheric weather fetch failed: %s", exc)

    # ── Marine weather ────────────────────────────────────────────────────────
    try:
        marine_url = "https://marine-api.open-meteo.com/v1/marine"
        marine_params = {
            "latitude": lat,
            "longitude": lng,
            "current": [
                "wave_height",
                "sea_surface_temperature",
                "wind_wave_height",
            ],
        }
        r2 = requests.get(marine_url, params=marine_params, timeout=5)
        r2.raise_for_status()
        marine = r2.json().get("current", {})
        data["wave_height_m"] = marine.get("wave_height")
        data["sea_surface_temp_c"] = marine.get("sea_surface_temperature")
        data["wind_wave_height_m"] = marine.get("wind_wave_height")
    except Exception as exc:
        logger.warning("Marine weather fetch failed: %s", exc)

    data["source"] = "open-meteo.com + marine-api.open-meteo.com"

    # Use SST if air temp unavailable, or vice versa as fallback
    if data.get("temperature_c") is None and data.get("sea_surface_temp_c") is not None:
        data["temperature_c"] = data["sea_surface_temp_c"]

    if data:
        _set_cached(lat, lng, data)

    return data


def safety_assessment(weather: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a safety summary for small fishing boats (< 10 m).
    Based on Bahrain Maritime Safety Authority guidelines.
    """
    wave_h = weather.get("wave_height_m") or 0.0
    wind   = weather.get("wind_speed_kmh") or 0.0

    if wave_h <= 0.5 and wind <= 20:
        level, level_ar = "safe", "آمن"
        message_ar = "الطقس مناسب للصيد بالقوارب الصغيرة"
        message_en = "Safe for small boat fishing"
    elif wave_h <= 1.2 and wind <= 35:
        level, level_ar = "caution", "تحذير"
        message_ar = "توخَّ الحذر — أمواج معتدلة. قوارب أكبر من 6 م فقط"
        message_en = "Caution — moderate waves. Boats >6m only"
    elif wave_h <= 2.0 and wind <= 50:
        level, level_ar = "warning", "خطر"
        message_ar = "لا يُنصح بالصيد — بحر متقلب"
        message_en = "Not recommended — rough seas"
    else:
        level, level_ar = "danger", "خطر شديد"
        message_ar = "خطر شديد — أوقفوا الصيد فوراً"
        message_en = "DANGER — stop fishing immediately"

    return {
        "level": level,
        "level_ar": level_ar,
        "message_ar": message_ar,
        "message_en": message_en,
        "wave_height_m": wave_h,
        "wind_speed_kmh": wind,
        "shamal_warning": wind >= 30,  # NW Shamal wind common in Bahrain winters
    }

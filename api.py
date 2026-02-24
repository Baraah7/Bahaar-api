"""
api.py
======
Bahaar Fishing AI — Flask REST API

Endpoints:
    POST /predict              → fish presence probability
    POST /report               → receive user report + trigger learning
    GET  /spots/nearby         → nearby fishing spots
    GET  /zone/info            → zone characteristics
    GET  /seasonal/advice      → seasonal fishing advice
    GET  /health               → health check
    GET  /species              → list all known species
    GET  /weather              → current weather + safety assessment

Run locally:
    export FLASK_ENV=development
    python api.py

Deploy on Render:
    gunicorn api:app --workers 2 --timeout 60
"""

import logging
import os
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Dict, Tuple

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Mobile app access

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "100 per minute"],
    storage_uri="memory://",
)


# ── Lazy imports (avoid import errors at startup if Firebase not configured) ──
def _firebase():
    import firebase_service
    return firebase_service


def _learning():
    from learning_engine import LearningEngine, PredictionEngine, classify_zone, get_mpa_restrictions
    return LearningEngine, PredictionEngine, classify_zone, get_mpa_restrictions


def _weather():
    import weather_service
    return weather_service


# ── Helpers ────────────────────────────────────────────────────────────────────
def _bad(msg: str, code: int = 400) -> Tuple[Any, int]:
    return jsonify({"error": msg, "error_ar": _translate_error(msg)}), code


def _translate_error(msg: str) -> str:
    translations = {
        "lat required": "خط العرض مطلوب",
        "lng required": "خط الطول مطلوب",
        "species_id required": "معرّف النوع مطلوب",
        "Invalid species_id": "معرّف النوع غير صحيح",
        "success_rating must be 1-5": "التقييم يجب أن يكون بين 1 و5",
        "Internal server error": "خطأ في الخادم الداخلي",
    }
    return translations.get(msg, msg)


def _require_floats(*fields) -> Tuple[Dict, str]:
    """Extract and validate float fields from request JSON."""
    data = request.get_json(silent=True) or {}
    out = {}
    for f in fields:
        v = data.get(f)
        if v is None:
            return {}, f"{f} required"
        try:
            out[f] = float(v)
        except (TypeError, ValueError):
            return {}, f"{f} must be a number"
    return out, ""


def handle_errors(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as exc:
            logger.exception("Unhandled error in %s", f.__name__)
            return _bad("Internal server error", 500)
    return wrapper


# ──────────────────────────────────────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "Bahaar Fishing AI API",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


# ──────────────────────────────────────────────────────────────────────────────
# GET /species
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/species", methods=["GET"])
@handle_errors
def list_species():
    from data.static_data import SPECIES
    result = []
    for sid, sp in SPECIES.items():
        result.append({
            "id": sid,
            "name_ar": sp["name_ar"],
            "name_en": sp["name_en"],
            "scientific": sp["scientific"],
            "peak_months": sp.get("peak_months", []),
            "best_methods": sp.get("best_methods", []),
            "price_bd_per_kg": sp.get("price_bd_per_kg"),
        })
    return jsonify({"species": result, "count": len(result)})


# ──────────────────────────────────────────────────────────────────────────────
# POST /predict
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
@limiter.limit("60 per minute")
@handle_errors
def predict():
    """
    Predict fish presence probability for a location + species.

    Body (JSON):
        lat         float   required
        lng         float   required
        species_id  str     required
        temp        float   optional  (°C, overrides API fetch)
        wind        float   optional  (km/h)
        fetch_weather bool  optional  default true

    Returns:
        probability     float   0–100
        factors         object  {static, seasonal, weather, human}
        zone            str
        nearby_spots    array
        restrictions    array   (MPA info)
        weather         object
        safety          object
    """
    data = request.get_json(silent=True) or {}

    # ── Validate ──────────────────────────────────────────────────────────────
    for req in ("lat", "lng", "species_id"):
        if req not in data:
            return _bad(f"{req} required")

    try:
        lat = float(data["lat"])
        lng = float(data["lng"])
    except (TypeError, ValueError):
        return _bad("lat/lng must be numbers")

    species_id = str(data["species_id"]).lower().strip()
    from data.static_data import SPECIES
    if species_id not in SPECIES:
        return _bad(f"Invalid species_id. Valid: {', '.join(SPECIES.keys())}")

    month = int(data.get("month") or datetime.now().month)
    if not 1 <= month <= 12:
        return _bad("month must be 1-12")

    # ── Weather ───────────────────────────────────────────────────────────────
    temp = data.get("temp")
    wind = data.get("wind")
    weather_data = {}

    if data.get("fetch_weather", True) and (temp is None or wind is None):
        try:
            ws = _weather()
            weather_data = ws.get_weather(lat, lng)
            temp = temp if temp is not None else weather_data.get("temperature_c")
            wind = wind if wind is not None else weather_data.get("wind_speed_kmh")
        except Exception as e:
            logger.warning("Weather fetch skipped: %s", e)

    safety = {}
    if weather_data:
        try:
            safety = _weather().safety_assessment(weather_data)
        except Exception:
            pass

    # ── Load data ─────────────────────────────────────────────────────────────
    fb = _firebase()
    static_weights   = fb.get_static_weights()
    seasonal_weights = fb.get_seasonal_weights()
    spots            = fb.get_fishing_spots()
    recent_reports   = fb.get_recent_reports(lat, lng, species_id, days=7, radius_km=3.0)
    nearby_spots     = fb.get_nearby_spots(lat, lng, radius_km=10.0, species=species_id)

    # ── Predict ───────────────────────────────────────────────────────────────
    LearningEngine, PredictionEngine, classify_zone, get_mpa_restrictions = _learning()
    engine = PredictionEngine()

    result = engine.predict(
        lat=lat, lng=lng,
        species_id=species_id,
        month=month,
        static_weights=static_weights,
        seasonal_weights=seasonal_weights,
        spots=spots,
        recent_reports=recent_reports,
        temp=temp,
        wind=wind,
    )

    # ── Enrich response ───────────────────────────────────────────────────────
    sp = SPECIES[species_id]
    result["species"] = {
        "id": species_id,
        "name_ar": sp["name_ar"],
        "name_en": sp["name_en"],
        "peak_months": sp["peak_months"],
        "peak_advice_ar": sp.get("peak_advice_ar", ""),
        "peak_advice_en": sp.get("peak_advice_en", ""),
    }
    result["nearby_spots"] = nearby_spots[:5]  # top 5 closest
    result["weather"] = weather_data
    result["safety"] = safety
    result["input"] = {
        "lat": lat, "lng": lng,
        "species_id": species_id,
        "month": month,
        "temp_used": temp,
        "wind_used": wind,
    }

    # Human-readable probability tier
    p = result["probability"]
    if p >= 75:
        result["tier"] = "excellent"
        result["tier_ar"] = "ممتاز"
    elif p >= 50:
        result["tier"] = "good"
        result["tier_ar"] = "جيد"
    elif p >= 25:
        result["tier"] = "fair"
        result["tier_ar"] = "متوسط"
    else:
        result["tier"] = "low"
        result["tier_ar"] = "ضعيف"

    return jsonify(result)


# ──────────────────────────────────────────────────────────────────────────────
# POST /report
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/report", methods=["POST"])
@limiter.limit("30 per minute")
@handle_errors
def receive_report():
    """
    Receive a user fishing report and trigger the learning engine.

    Body (JSON):
        user_id                 str     required
        lat                     float   required
        lng                     float   required
        species_id              str     required
        success_rating          int     required  1-5
        quantity                float   optional  kg caught
        method                  str     optional  fishing method
        predicted_probability   float   optional  what the app showed (0-100)
        notes                   str     optional

    Returns:
        learned             bool
        error               float   prediction error
        weights_updated     bool
        new_spot_candidate  bool
        candidate           obj|null
        report_id           str
    """
    data = request.get_json(silent=True) or {}

    # ── Validate ──────────────────────────────────────────────────────────────
    for req in ("user_id", "lat", "lng", "species_id", "success_rating"):
        if req not in data:
            return _bad(f"{req} required")

    try:
        lat = float(data["lat"])
        lng = float(data["lng"])
    except (TypeError, ValueError):
        return _bad("lat/lng must be numbers")

    try:
        rating = int(data["success_rating"])
        if not 1 <= rating <= 5:
            raise ValueError
    except (TypeError, ValueError):
        return _bad("success_rating must be 1-5")

    species_id = str(data["species_id"]).lower().strip()
    from data.static_data import SPECIES
    if species_id not in SPECIES:
        return _bad(f"Invalid species_id")

    # ── Build report doc ──────────────────────────────────────────────────────
    report = {
        "user_id":              str(data["user_id"]),
        "lat":                  lat,
        "lng":                  lng,
        "species_id":           species_id,
        "success_rating":       rating,
        "quantity":             float(data.get("quantity") or 0),
        "method":               str(data.get("method") or "unknown"),
        "predicted_probability": float(data.get("predicted_probability") or 50),
        "notes":                str(data.get("notes") or ""),
        "month":                datetime.now().month,
        "timestamp":            datetime.now(timezone.utc),
    }

    # ── Save raw report ───────────────────────────────────────────────────────
    fb = _firebase()
    report_id = fb.save_report(report)
    report["id"] = report_id

    # ── Run learning engine ────────────────────────────────────────────────────
    LearningEngine, PredictionEngine, classify_zone, _ = _learning()
    engine = LearningEngine(
        learning_rate=float(os.getenv("LEARNING_RATE", 0.05)),
    )

    static_weights   = fb.get_static_weights()
    seasonal_weights = fb.get_seasonal_weights()
    spots            = fb.get_fishing_spots()
    nearby_reports   = fb.get_nearby_reports_for_candidate(
        lat, lng, species_id, days=30, radius_km=0.5
    )

    result = engine.process_report(
        report=report,
        current_weights=static_weights,
        current_seasonal=seasonal_weights,
        nearby_recent_reports=nearby_reports,
        known_spots=spots,
    )

    # ── Persist weight updates ────────────────────────────────────────────────
    fb.save_weights(
        static_updates=result["new_static_weights"],
        seasonal_updates=result["new_seasonal_weights"],
    )

    # ── Save learning log ─────────────────────────────────────────────────────
    fb.save_learning_log(result["log_entry"])

    # ── Save spot candidate ────────────────────────────────────────────────────
    candidate_id = None
    if result["spot_candidate"]:
        candidate_id = fb.save_spot_candidate(result["spot_candidate"])
        logger.info("New spot candidate saved: %s", candidate_id)

    return jsonify({
        "learned": True,
        "report_id": report_id,
        "error": result["error"],
        "confidence_used": result["confidence"],
        "weights_updated": True,
        "static_update": result["static_update"],
        "seasonal_update": result["seasonal_update"],
        "new_spot_candidate": result["spot_candidate"] is not None,
        "candidate_id": candidate_id,
        "candidate": result["spot_candidate"],
    })


# ──────────────────────────────────────────────────────────────────────────────
# GET /spots/nearby
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/spots/nearby", methods=["GET"])
@handle_errors
def spots_nearby():
    """
    GET /spots/nearby?lat=26.45&lng=50.52&radius=10&species=hamour

    radius defaults to 10 km, max 50 km.
    """
    try:
        lat = float(request.args["lat"])
        lng = float(request.args["lng"])
    except (KeyError, TypeError, ValueError):
        return _bad("lat and lng query params required")

    radius = min(float(request.args.get("radius", 10)), 50)
    species = request.args.get("species")

    fb = _firebase()
    spots = fb.get_nearby_spots(lat, lng, radius_km=radius, species=species)

    # Clean up output
    result = []
    for spot in spots:
        result.append({
            "id": spot.get("id"),
            "name_ar": spot.get("name_ar"),
            "name_en": spot.get("name_en"),
            "lat": spot.get("lat"),
            "lng": spot.get("lng"),
            "distance_km": spot.get("distance_km"),
            "zone": spot.get("zone"),
            "depth_min": spot.get("depth_min"),
            "depth_max": spot.get("depth_max"),
            "bottom_type": spot.get("bottom_type"),
            "species": spot.get("species"),
            "gear": spot.get("gear"),
            "confidence": spot.get("confidence"),
            "mpa": spot.get("mpa", False),
            "base_score": spot.get("base_score"),
            "source": spot.get("source"),
        })

    return jsonify({
        "spots": result,
        "count": len(result),
        "query": {"lat": lat, "lng": lng, "radius_km": radius, "species": species},
    })


# ──────────────────────────────────────────────────────────────────────────────
# GET /zone/info
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/zone/info", methods=["GET"])
@handle_errors
def zone_info():
    """GET /zone/info?lat=26.45&lng=50.52"""
    try:
        lat = float(request.args["lat"])
        lng = float(request.args["lng"])
    except (KeyError, TypeError, ValueError):
        return _bad("lat and lng query params required")

    LearningEngine, PredictionEngine, classify_zone, get_mpa_restrictions = _learning()
    zone = classify_zone(lat, lng)

    from data.static_data import ZONES
    zd = ZONES[zone]

    restrictions = get_mpa_restrictions(lat, lng)

    return jsonify({
        "zone": zone,
        "name_ar": zd["name_ar"],
        "name_en": zd["name_en"],
        "description": zd["description"],
        "productivity_pct": zd["productivity_pct"],
        "bottom_types": zd["bottom_types"],
        "common_species": zd["common_species"],
        "dominant_gear": zd["dominant_gear"],
        "avg_depth_m": zd["avg_depth_m"],
        "nearby_mpas": restrictions,
        "is_restricted": any(r["restriction"] == "full" for r in restrictions),
    })


# ──────────────────────────────────────────────────────────────────────────────
# GET /seasonal/advice
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/seasonal/advice", methods=["GET"])
@handle_errors
def seasonal_advice():
    """GET /seasonal/advice?species_id=hamour&month=10"""
    species_id = request.args.get("species_id", "").lower().strip()
    month_str  = request.args.get("month", str(datetime.now().month))

    from data.static_data import SPECIES
    if not species_id or species_id not in SPECIES:
        return _bad(f"Valid species_id required: {', '.join(SPECIES.keys())}")

    try:
        month = int(month_str)
        if not 1 <= month <= 12:
            raise ValueError
    except (TypeError, ValueError):
        return _bad("month must be 1-12")

    sp = SPECIES[species_id]
    calendar = sp.get("seasonal", [1.0] * 12)
    factor = calendar[month - 1]

    is_peak = month in sp.get("peak_months", [])
    month_names = [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ]
    month_names_ar = [
        "يناير","فبراير","مارس","أبريل","مايو","يونيو",
        "يوليو","أغسطس","سبتمبر","أكتوبر","نوفمبر","ديسمبر"
    ]

    # Dynamic advice based on factor
    if factor >= 1.1:
        advice_en = f"Excellent month for {sp['name_en']}. {sp.get('peak_advice_en','')}"
        advice_ar = f"شهر ممتاز لصيد {sp['name_ar']}. {sp.get('peak_advice_ar','')}"
    elif factor >= 0.7:
        advice_en = f"Moderate activity expected for {sp['name_en']} this month."
        advice_ar = f"نشاط متوسط متوقع لـ{sp['name_ar']} هذا الشهر."
    else:
        advice_en = f"Low season for {sp['name_en']} — consider alternative species."
        advice_ar = f"موسم منخفض لـ{sp['name_ar']} — يُنصح بصيد أنواع أخرى."

    return jsonify({
        "species_id": species_id,
        "name_ar": sp["name_ar"],
        "name_en": sp["name_en"],
        "month": month,
        "month_name_en": month_names[month - 1],
        "month_name_ar": month_names_ar[month - 1],
        "seasonal_factor": round(factor, 3),
        "is_peak_season": is_peak,
        "peak_months": sp.get("peak_months", []),
        "advice_ar": advice_ar.strip(),
        "advice_en": advice_en.strip(),
        "best_methods": sp.get("best_methods", []),
        "typical_depth_m": f"{sp['depth_min']}–{sp['depth_max']}",
        "preferred_temp_c": f"{sp['temp_min']}–{sp['temp_max']}",
        "price_bd_per_kg": sp.get("price_bd_per_kg"),
        "full_calendar": [round(f, 3) for f in calendar],
    })


# ──────────────────────────────────────────────────────────────────────────────
# GET /weather
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/weather", methods=["GET"])
@handle_errors
def weather():
    """GET /weather?lat=26.2&lng=50.58"""
    try:
        lat = float(request.args["lat"])
        lng = float(request.args["lng"])
    except (KeyError, TypeError, ValueError):
        return _bad("lat and lng required")

    ws = _weather()
    data = ws.get_weather(lat, lng)
    safety = ws.safety_assessment(data) if data else {}

    return jsonify({
        "weather": data,
        "safety": safety,
        "location": {"lat": lat, "lng": lng},
    })


# ──────────────────────────────────────────────────────────────────────────────
# Error handlers
# ──────────────────────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found", "error_ar": "نقطة النهاية غير موجودة"}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(429)
def rate_limit_exceeded(e):
    return jsonify({
        "error": "Rate limit exceeded — max 100 requests/minute",
        "error_ar": "تجاوزت الحد المسموح — 100 طلب في الدقيقة كحد أقصى",
    }), 429


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)

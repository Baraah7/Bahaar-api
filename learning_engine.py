"""
learning_engine.py
==================
Continuous learning system for the Bahaar fishing prediction AI.
Implements the core equation:

    Probability = (Static Weight × Seasonal Factor × Weather Factor × Human Factor) / 100

Every user report refines the weights via gradient-descent-style updates.
New fishing spots are automatically detected when ≥3 successful reports
cluster within 500 m.
"""

import math
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# DEFAULT WEIGHT TABLES
# These are the Firestore-backed tables; we load them at startup and update them
# in real time.  Keys are "zone_species" strings.
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_STATIC_WEIGHTS: Dict[str, float] = {
    "northern_hamour":   1.35,
    "northern_safi":     1.25,
    "northern_shrimp":   1.05,
    "northern_crab":     0.95,
    "northern_kanad":    0.85,
    "northern_emperor":  1.15,
    "northern_bream":    1.05,
    "northern_barracuda":1.05,
    "northern_queenfish":1.00,
    "northern_squid":    1.15,
    "eastern_hamour":    0.85,
    "eastern_safi":      1.15,
    "eastern_shrimp":    1.45,
    "eastern_crab":      1.25,
    "eastern_kanad":     0.75,
    "eastern_emperor":   1.05,
    "eastern_bream":     0.95,
    "eastern_barracuda": 0.75,
    "eastern_squid":     0.95,
    "western_hamour":    1.05,
    "western_safi":      0.85,
    "western_shrimp":    0.55,
    "western_crab":      0.65,
    "western_kanad":     1.15,
    "western_emperor":   0.85,
    "western_bream":     0.85,
    "western_barracuda": 0.95,
    "southern_hamour":   0.40,
    "southern_safi":     0.60,
    "southern_shrimp":   0.75,
}

# ──────────────────────────────────────────────────────────────────────────────
# HELPER — Haversine distance (km)
# ──────────────────────────────────────────────────────────────────────────────
def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lng2 - lng1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ──────────────────────────────────────────────────────────────────────────────
# ZONE CLASSIFIER
# ──────────────────────────────────────────────────────────────────────────────
def classify_zone(lat: float, lng: float) -> str:
    from data.static_data import ZONES
    for zone_id, z in ZONES.items():
        if (z["lat_min"] <= lat <= z["lat_max"] and
                z["lng_min"] <= lng <= z["lng_max"]):
            return zone_id
    # Default: find nearest zone centroid
    centroids = {
        "northern": (26.45, 50.52),
        "eastern":  (26.05, 50.63),
        "western":  (26.15, 50.36),
        "southern": (25.67, 50.77),
    }
    return min(centroids, key=lambda z: haversine_km(lat, lng, *centroids[z]))


# ──────────────────────────────────────────────────────────────────────────────
# MPA CHECKER
# ──────────────────────────────────────────────────────────────────────────────
def get_mpa_restrictions(lat: float, lng: float) -> List[Dict]:
    from data.static_data import MPA_ZONES
    restrictions = []
    for mpa in MPA_ZONES:
        dist = haversine_km(lat, lng, mpa["lat"], mpa["lng"])
        if dist <= mpa["radius_km"]:
            restrictions.append({
                "id": mpa["id"],
                "name_ar": mpa["name_ar"],
                "name_en": mpa["name_en"],
                "restriction": mpa["restriction"],
                "restriction_ar": mpa["restriction_ar"],
                "restriction_en": mpa["restriction_en"],
                "distance_km": round(dist, 2),
            })
    return sorted(restrictions, key=lambda r: r["distance_km"])


# ──────────────────────────────────────────────────────────────────────────────
# PREDICTION ENGINE
# ──────────────────────────────────────────────────────────────────────────────
class PredictionEngine:
    """Stateless calculation engine.  All data is passed in; no DB calls here."""

    def calculate_static_factor(
        self,
        zone: str,
        species_id: str,
        lat: float,
        lng: float,
        weights: Dict[str, float],
        spots: List[Dict],
    ) -> Tuple[float, Optional[Dict]]:
        """
        Returns (static_factor, nearest_spot_or_None).
        static_factor accounts for zone productivity × proximity to known spots.
        """
        from data.static_data import ZONES, SPECIES

        # Base zone weight
        zone_data = ZONES.get(zone, ZONES["northern"])
        base = zone_data["base_weight"]

        # Species-zone preference modifier
        sp = SPECIES.get(species_id, {})
        zone_pref = sp.get("zone_weights", {}).get(zone, 1.0)

        # Learned weight (dynamic, from Firestore)
        key = f"{zone}_{species_id}"
        learned = weights.get(key, 1.0)

        # Proximity bonus: how close to a known spot for this species?
        nearest_spot = None
        proximity_bonus = 1.0
        for spot in spots:
            if spot.get("mpa", False):
                continue
            if species_id not in spot.get("species", []):
                continue
            dist = haversine_km(lat, lng, spot["lat"], spot["lng"])
            if dist < 5.0:  # within 5 km
                bonus = max(0.5, 1.0 - (dist / 10.0))  # 1.0 at 0 km → 0.5 at 10 km
                if bonus > proximity_bonus or nearest_spot is None:
                    proximity_bonus = bonus
                    nearest_spot = {**spot, "distance_km": round(dist, 2)}

        static = base * zone_pref * learned * proximity_bonus
        # Cap at 2.0
        return min(static, 2.0), nearest_spot

    def calculate_seasonal_factor(
        self,
        species_id: str,
        month: int,  # 1-12
        seasonal_weights: Dict[str, float],
    ) -> float:
        """Returns seasonal factor from species calendar, adjusted by learned weights."""
        from data.static_data import SPECIES
        sp = SPECIES.get(species_id, {})
        base_calendar = sp.get("seasonal", [1.0] * 12)
        base_factor = base_calendar[month - 1]  # 0-indexed

        # Learned seasonal adjustment
        key = f"{species_id}_month_{month}"
        learned_adj = seasonal_weights.get(key, 1.0)

        return round(base_factor * learned_adj, 4)

    def calculate_weather_factor(
        self,
        species_id: str,
        temp: Optional[float],
        wind: Optional[float],
    ) -> float:
        """
        Returns weather suitability factor (0.2 – 1.5).
        If weather data is unavailable, returns 1.0 (neutral).
        """
        if temp is None and wind is None:
            return 1.0

        from data.static_data import SPECIES
        sp = SPECIES.get(species_id, {
            "temp_min": 20, "temp_max": 35, "wind_max": 30
        })

        t_min, t_max = sp["temp_min"], sp["temp_max"]
        w_max = sp["wind_max"]

        factor = 1.0

        # Temperature component
        if temp is not None:
            if t_min <= temp <= t_max:
                # Ideal band
                t_center = (t_min + t_max) / 2
                deviation = abs(temp - t_center) / ((t_max - t_min) / 2)
                factor *= 1.0 - (0.2 * deviation)  # slight penalty near edges
            elif abs(temp - t_min) < 3 or abs(temp - t_max) < 3:
                factor *= 0.65  # marginal temp
            else:
                factor *= 0.35  # bad temp

        # Wind component
        if wind is not None:
            if wind <= w_max:
                wind_ratio = wind / w_max
                factor *= 1.0 - (0.15 * wind_ratio)  # gentle penalty for high wind
            elif wind <= w_max * 1.5:
                factor *= 0.55
            else:
                factor *= 0.25  # dangerous wind

        return round(max(0.2, min(1.5, factor)), 4)

    def calculate_human_factor(
        self,
        recent_reports: List[Dict],
    ) -> float:
        """
        Returns human factor (0.7 – 1.5) from recent successful reports
        within the location radius.  Newer + more-successful reports weighted higher.
        """
        if not recent_reports:
            return 1.0

        now = datetime.now(timezone.utc)
        total_weight = 0.0

        for report in recent_reports:
            ts = report.get("timestamp")
            if isinstance(ts, datetime):
                days_old = max(0, (now - ts).total_seconds() / 86400)
            else:
                days_old = 3  # fallback

            recency_w = max(0.4, 1.0 - (days_old / 14.0))
            success_w = min(report.get("success_rating", 3), 5) / 5.0
            total_weight += recency_w * success_w

        avg = total_weight / len(recent_reports)
        # Map 0–1 → 0.7–1.5
        return round(0.7 + (avg * 0.8), 4)

    def predict(
        self,
        lat: float,
        lng: float,
        species_id: str,
        month: int,
        static_weights: Dict[str, float],
        seasonal_weights: Dict[str, float],
        spots: List[Dict],
        recent_reports: List[Dict],
        temp: Optional[float] = None,
        wind: Optional[float] = None,
    ) -> Dict[str, Any]:
        zone = classify_zone(lat, lng)
        restrictions = get_mpa_restrictions(lat, lng)

        # MPA hard cap
        is_restricted = any(r["restriction"] == "full" for r in restrictions)
        if is_restricted:
            return {
                "probability": 5.0,
                "zone": zone,
                "factors": {"static": 0.0, "seasonal": 0.0, "weather": 0.0, "human": 0.0},
                "restrictions": restrictions,
                "mpa_blocked": True,
                "message_ar": "هذه المنطقة محمية — الصيد محظور",
                "message_en": "This area is a Marine Protected Area — fishing restricted",
            }

        static_f, nearest_spot = self.calculate_static_factor(
            zone, species_id, lat, lng, static_weights, spots
        )
        seasonal_f = self.calculate_seasonal_factor(species_id, month, seasonal_weights)
        weather_f = self.calculate_weather_factor(species_id, temp, wind)
        human_f = self.calculate_human_factor(recent_reports)

        # Core equation
        raw = static_f * seasonal_f * weather_f * human_f * 100.0
        # Scale to 0–100
        probability = round(min(max(raw, 0.0), 100.0), 1)

        return {
            "probability": probability,
            "zone": zone,
            "zone_name_ar": _zone_name_ar(zone),
            "zone_name_en": _zone_name_en(zone),
            "factors": {
                "static": round(static_f, 4),
                "seasonal": round(seasonal_f, 4),
                "weather": round(weather_f, 4),
                "human": round(human_f, 4),
            },
            "nearest_spot": nearest_spot,
            "restrictions": restrictions,
            "mpa_blocked": False,
            "report_count_used": len(recent_reports),
        }


def _zone_name_ar(zone: str) -> str:
    m = {"northern": "المنطقة الشمالية", "eastern": "المنطقة الشرقية",
         "western": "المنطقة الغربية", "southern": "المنطقة الجنوبية"}
    return m.get(zone, zone)

def _zone_name_en(zone: str) -> str:
    m = {"northern": "Northern Zone", "eastern": "Eastern Zone",
         "western": "Western Zone", "southern": "Southern Zone"}
    return m.get(zone, zone)


# ──────────────────────────────────────────────────────────────────────────────
# LEARNING ENGINE
# ──────────────────────────────────────────────────────────────────────────────
class LearningEngine:
    """
    Processes user reports and updates model weights.

    Weight update rule (gradient-descent style):
        error      = actual – predicted   (both in [0, 1])
        adjustment = 1 + (error × learning_rate)
        new_weight = old_weight × adjustment
        new_weight = clamp(new_weight, 0.3, 2.0)

    Seasonal factors update at half the base learning rate.
    New spot candidates are flagged when ≥3 successful reports (rating ≥ 4)
    cluster within 500 m over the past 30 days.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        min_reports_for_update: int = 1,
        new_spot_threshold: int = 3,
        new_spot_radius_km: float = 0.5,
    ):
        self.learning_rate = learning_rate
        self.min_reports_for_update = min_reports_for_update
        self.new_spot_threshold = new_spot_threshold
        self.new_spot_radius_km = new_spot_radius_km

    # ── core ─────────────────────────────────────────────────────────────────

    def calculate_error(self, predicted_pct: float, success_rating: int) -> float:
        """
        predicted_pct: 0–100 (model output)
        success_rating: 1–5 (user rating)
        Returns error in [-1, 1].  Positive = we underestimated.
        """
        predicted = predicted_pct / 100.0
        actual = success_rating / 5.0
        return round(actual - predicted, 6)

    def update_static_weight(
        self,
        current_weight: float,
        error: float,
        confidence: float = 1.0,
    ) -> float:
        """
        confidence 0–1 scales how aggressively we apply this update
        (higher confidence reports → bigger update).
        """
        adjustment = 1.0 + (error * self.learning_rate * confidence)
        new_w = current_weight * adjustment
        return round(max(0.3, min(2.0, new_w)), 6)

    def update_seasonal_factor(
        self,
        current_factor: float,
        error: float,
    ) -> float:
        """Seasonal factors update at half the base rate."""
        adjustment = 1.0 + (error * self.learning_rate * 0.5)
        new_f = current_factor * adjustment
        return round(max(0.1, min(2.0, new_f)), 6)

    def process_report(
        self,
        report: Dict[str, Any],
        current_weights: Dict[str, float],
        current_seasonal: Dict[str, float],
        nearby_recent_reports: List[Dict],
        known_spots: List[Dict],
    ) -> Dict[str, Any]:
        """
        Main entry point.  Returns a dict with all weight updates and
        optionally a new spot candidate.

        report keys required:
            user_id, lat, lng, species_id, success_rating (1-5),
            predicted_probability (0-100), timestamp
        """
        lat = report["lat"]
        lng = report["lng"]
        species = report["species_id"]
        rating = int(report["success_rating"])
        predicted = float(report.get("predicted_probability", 50.0))
        month = report.get("month") or datetime.now().month

        zone = classify_zone(lat, lng)
        error = self.calculate_error(predicted, rating)

        # Confidence: weight updates more for extreme ratings (1 or 5)
        confidence = abs(rating - 3) / 2.0 + 0.5  # 0.5 (rating=3) … 1.5 (rating=1/5)

        # ── static weight update ──────────────────────────────────────────────
        static_key = f"{zone}_{species}"
        old_static = current_weights.get(static_key, 1.0)
        new_static = self.update_static_weight(old_static, error, confidence)

        static_update = {
            "key": static_key,
            "old": old_static,
            "new": new_static,
            "error": error,
            "confidence": round(confidence, 3),
        }

        # ── seasonal factor update ────────────────────────────────────────────
        seasonal_key = f"{species}_month_{month}"
        old_seasonal = current_seasonal.get(seasonal_key, 1.0)
        new_seasonal = self.update_seasonal_factor(old_seasonal, error)

        seasonal_update = {
            "key": seasonal_key,
            "old": old_seasonal,
            "new": new_seasonal,
        }

        # ── spot candidate check ──────────────────────────────────────────────
        spot_candidate = None
        if rating >= 4:
            candidate = self._check_spot_candidate(
                lat, lng, species, nearby_recent_reports, known_spots
            )
            if candidate:
                spot_candidate = candidate

        # ── learning log entry ────────────────────────────────────────────────
        log_entry = {
            "report_id": report.get("id", "unknown"),
            "user_id": report.get("user_id"),
            "zone": zone,
            "species": species,
            "month": month,
            "predicted": predicted,
            "actual_rating": rating,
            "error": error,
            "static_key": static_key,
            "static_old": old_static,
            "static_new": new_static,
            "seasonal_key": seasonal_key,
            "seasonal_old": old_seasonal,
            "seasonal_new": new_seasonal,
            "spot_candidate_created": spot_candidate is not None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return {
            "error": error,
            "confidence": round(confidence, 3),
            "weights_updated": True,
            "static_update": static_update,
            "seasonal_update": seasonal_update,
            "spot_candidate": spot_candidate,
            "log_entry": log_entry,
            "new_static_weights": {static_key: new_static},
            "new_seasonal_weights": {seasonal_key: new_seasonal},
        }

    # ── spot candidate detection ──────────────────────────────────────────────

    def _check_spot_candidate(
        self,
        lat: float,
        lng: float,
        species: str,
        recent_reports: List[Dict],
        known_spots: List[Dict],
    ) -> Optional[Dict]:
        """Return a candidate dict if conditions met, else None."""
        # Filter: same species, high success, within radius
        qualifying = [
            r for r in recent_reports
            if r.get("species_id") == species
            and int(r.get("success_rating", 0)) >= 4
            and haversine_km(lat, lng, r["lat"], r["lng"]) <= self.new_spot_radius_km
        ]

        if len(qualifying) < self.new_spot_threshold:
            return None

        # Check if this is already within 1 km of a known spot
        for spot in known_spots:
            if haversine_km(lat, lng, spot["lat"], spot["lng"]) < 1.0:
                logger.debug(
                    "Candidate near known spot %s — skipping", spot.get("id")
                )
                return None

        return self._create_candidate(qualifying, species)

    def _create_candidate(
        self, reports: List[Dict], species: str
    ) -> Dict[str, Any]:
        n = len(reports)
        avg_lat = sum(r["lat"] for r in reports) / n
        avg_lng = sum(r["lng"] for r in reports) / n
        avg_qty = sum(r.get("quantity", 0.0) for r in reports) / n
        avg_rating = sum(r["success_rating"] for r in reports) / n

        methods: Dict[str, int] = {}
        for r in reports:
            m = r.get("method", "unknown")
            methods[m] = methods.get(m, 0) + 1

        zone = classify_zone(avg_lat, avg_lng)

        return {
            "name_ar": f"موقع صيد مقترح ({n} تقرير)",
            "name_en": f"Candidate Fishing Spot ({n} reports)",
            "lat": round(avg_lat, 6),
            "lng": round(avg_lng, 6),
            "zone": zone,
            "primary_species": species,
            "report_count": n,
            "avg_rating": round(avg_rating, 2),
            "avg_quantity_kg": round(avg_qty, 2),
            "top_method": max(methods, key=methods.get),
            "status": "candidate",
            "confidence": "low",
            "report_ids": [r.get("id", "") for r in reports],
        }

"""
firebase_service.py
===================
Thin wrapper around the Firebase Admin SDK.
All Firestore reads/writes pass through this module so the rest of
the codebase stays clean.  Implements a simple in-process TTL cache
so we don't hammer Firestore on every prediction request.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Lazy Firebase initialisation
# ──────────────────────────────────────────────────────────────────────────────
_firebase_app = None
_db = None  # Firestore client


def _init_firebase() -> None:
    global _firebase_app, _db
    if _firebase_app:
        return

    import firebase_admin
    from firebase_admin import credentials, firestore

    project_id    = os.getenv("FIREBASE_PROJECT_ID")
    private_key   = os.getenv("FIREBASE_PRIVATE_KEY", "").replace("\\n", "\n")
    client_email  = os.getenv("FIREBASE_CLIENT_EMAIL")

    if project_id and private_key and client_email:
        cred_dict = {
            "type": "service_account",
            "project_id": project_id,
            "private_key": private_key,
            "client_email": client_email,
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        cred = credentials.Certificate(cred_dict)
        _firebase_app = firebase_admin.initialize_app(cred)
    else:
        # Fallback: try GOOGLE_APPLICATION_CREDENTIALS env var / ADC
        logger.warning("Firebase env vars missing — trying Application Default Credentials")
        _firebase_app = firebase_admin.initialize_app()

    _db = firestore.client()
    logger.info("Firebase initialised (project: %s)", project_id or "ADC")


def db():
    _init_firebase()
    return _db


# ──────────────────────────────────────────────────────────────────────────────
# Simple TTL cache (avoids repeated Firestore reads for hot data)
# ──────────────────────────────────────────────────────────────────────────────
class _TTLCache:
    def __init__(self):
        self._store: Dict[str, tuple] = {}   # key → (value, expiry_ts)

    def get(self, key: str) -> Optional[Any]:
        if key in self._store:
            val, exp = self._store[key]
            if time.time() < exp:
                return val
            del self._store[key]
        return None

    def set(self, key: str, value: Any, ttl_seconds: int = 120) -> None:
        self._store[key] = (value, time.time() + ttl_seconds)

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)


_cache = _TTLCache()


# ──────────────────────────────────────────────────────────────────────────────
# WEIGHTS
# ──────────────────────────────────────────────────────────────────────────────
def get_static_weights() -> Dict[str, float]:
    """Return learned static weights.  Cached 2 min."""
    cached = _cache.get("static_weights")
    if cached:
        return cached

    from data.static_data import DEFAULT_STATIC_WEIGHTS
    try:
        doc = db().collection("ai_weights").document("static").get()
        if doc.exists:
            data = doc.to_dict() or {}
            weights = {**DEFAULT_STATIC_WEIGHTS, **data}
        else:
            weights = dict(DEFAULT_STATIC_WEIGHTS)
    except Exception as exc:
        logger.error("Failed to load static weights: %s", exc)
        weights = dict(DEFAULT_STATIC_WEIGHTS)

    _cache.set("static_weights", weights, ttl_seconds=120)
    return weights


def get_seasonal_weights() -> Dict[str, float]:
    """Return learned seasonal adjustments.  Cached 5 min."""
    cached = _cache.get("seasonal_weights")
    if cached:
        return cached

    try:
        doc = db().collection("ai_weights").document("seasonal").get()
        weights = doc.to_dict() or {} if doc.exists else {}
    except Exception as exc:
        logger.error("Failed to load seasonal weights: %s", exc)
        weights = {}

    _cache.set("seasonal_weights", weights, ttl_seconds=300)
    return weights


def save_weights(
    static_updates: Dict[str, float],
    seasonal_updates: Dict[str, float],
) -> None:
    """Merge new weights into Firestore.  Invalidates cache."""
    from firebase_admin import firestore as fs

    if static_updates:
        db().collection("ai_weights").document("static").set(
            static_updates, merge=True
        )
        _cache.invalidate("static_weights")

    if seasonal_updates:
        db().collection("ai_weights").document("seasonal").set(
            seasonal_updates, merge=True
        )
        _cache.invalidate("seasonal_weights")


# ──────────────────────────────────────────────────────────────────────────────
# FISHING SPOTS
# ──────────────────────────────────────────────────────────────────────────────
def get_fishing_spots(zone: Optional[str] = None) -> List[Dict]:
    """Return spots from Firestore, falling back to static_data.  Cached 10 min."""
    cache_key = f"spots_{zone or 'all'}"
    cached = _cache.get(cache_key)
    if cached:
        return cached

    from data.static_data import FISHING_SPOTS

    try:
        ref = db().collection("fishing_spots")
        if zone:
            ref = ref.where("zone", "==", zone)
        docs = ref.stream()
        fs_spots = [{"id": d.id, **d.to_dict()} for d in docs]
        # Merge: Firestore spots override static ones; add new ones
        static_by_id = {s["id"]: s for s in FISHING_SPOTS}
        fs_by_id = {s["id"]: s for s in fs_spots}
        merged = {**static_by_id, **fs_by_id}
        spots = list(merged.values())
    except Exception as exc:
        logger.error("Firestore spots failed (%s) — using static data", exc)
        spots = [s for s in FISHING_SPOTS if not zone or s["zone"] == zone]

    _cache.set(cache_key, spots, ttl_seconds=600)
    return spots


def get_nearby_spots(
    lat: float,
    lng: float,
    radius_km: float,
    species: Optional[str] = None,
) -> List[Dict]:
    from learning_engine import haversine_km

    all_spots = get_fishing_spots()
    result = []
    for spot in all_spots:
        dist = haversine_km(lat, lng, spot["lat"], spot["lng"])
        if dist > radius_km:
            continue
        if species and species not in spot.get("species", []):
            continue
        result.append({**spot, "distance_km": round(dist, 3)})

    return sorted(result, key=lambda s: s["distance_km"])


# ──────────────────────────────────────────────────────────────────────────────
# USER REPORTS
# ──────────────────────────────────────────────────────────────────────────────
def save_report(report_data: Dict[str, Any]) -> str:
    """Write report to Firestore, return document ID."""
    from firebase_admin import firestore as fs

    report_data["server_timestamp"] = fs.SERVER_TIMESTAMP
    ref = db().collection("user_reports").add(report_data)
    doc_id = ref[1].id
    logger.info("Saved report %s", doc_id)
    return doc_id


def get_recent_reports(
    lat: float,
    lng: float,
    species: str,
    days: int = 7,
    radius_km: float = 3.0,
) -> List[Dict]:
    """
    Fetch reports near (lat, lng) for a species in the last N days.
    Firestore doesn't support true geo queries; we filter by a bounding
    box then refine with haversine in Python.
    """
    from datetime import timedelta
    from learning_engine import haversine_km

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Bounding-box approximation: 1° lat ≈ 111 km
    lat_delta = radius_km / 111.0
    lng_delta = radius_km / (111.0 * abs(math.cos(math.radians(lat))) + 0.001)

    try:
        query = (
            db()
            .collection("user_reports")
            .where("species_id", "==", species)
            .where("timestamp", ">=", cutoff)
            .limit(500)
        )
        docs = query.stream()
        reports = []
        for doc in docs:
            r = {"id": doc.id, **doc.to_dict()}
            rlat, rlng = r.get("lat", 0), r.get("lng", 0)
            if (abs(rlat - lat) <= lat_delta and abs(rlng - lng) <= lng_delta):
                dist = haversine_km(lat, lng, rlat, rlng)
                if dist <= radius_km:
                    r["distance_km"] = round(dist, 3)
                    reports.append(r)
        return reports
    except Exception as exc:
        logger.error("Failed to get recent reports: %s", exc)
        return []


def get_nearby_reports_for_candidate(
    lat: float,
    lng: float,
    species: str,
    days: int = 30,
    radius_km: float = 0.5,
) -> List[Dict]:
    return get_recent_reports(lat, lng, species, days=days, radius_km=radius_km)


# ──────────────────────────────────────────────────────────────────────────────
# LEARNING LOG
# ──────────────────────────────────────────────────────────────────────────────
def save_learning_log(entry: Dict[str, Any]) -> None:
    from firebase_admin import firestore as fs

    entry["server_ts"] = fs.SERVER_TIMESTAMP
    try:
        db().collection("learning_log").add(entry)
    except Exception as exc:
        logger.error("Learning log write failed: %s", exc)


# ──────────────────────────────────────────────────────────────────────────────
# SPOT CANDIDATES
# ──────────────────────────────────────────────────────────────────────────────
def save_spot_candidate(candidate: Dict[str, Any]) -> str:
    from firebase_admin import firestore as fs

    candidate["created_at"] = fs.SERVER_TIMESTAMP
    candidate["status"] = "candidate"
    ref = db().collection("spot_candidates").add(candidate)
    return ref[1].id


import math  # needed by haversine_km indirectly; place at module level

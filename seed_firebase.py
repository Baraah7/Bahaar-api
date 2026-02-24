"""
seed_firebase.py
================
One-time script to populate Firestore with initial data:
  • fishing_spots (15 confirmed spots from MARGIS II / FAO)
  • ai_weights/static   (default zone-species weights)
  • ai_weights/seasonal (neutral seasonal adjustments = 1.0)

Run ONCE before first deploy:
    python seed_firebase.py

Requires FIREBASE_* env vars or GOOGLE_APPLICATION_CREDENTIALS set.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from firebase_service import db, _init_firebase
from data.static_data import FISHING_SPOTS, SPECIES, DEFAULT_STATIC_WEIGHTS


def seed_spots():
    print("Seeding fishing spots …")
    batch = db().batch()
    col = db().collection("fishing_spots")
    for spot in FISHING_SPOTS:
        ref = col.document(spot["id"])
        batch.set(ref, spot, merge=True)
    batch.commit()
    print(f"  ✓ {len(FISHING_SPOTS)} spots written")


def seed_weights():
    print("Seeding AI weights …")
    # Static weights
    db().collection("ai_weights").document("static").set(
        DEFAULT_STATIC_WEIGHTS, merge=True
    )
    print(f"  ✓ static weights ({len(DEFAULT_STATIC_WEIGHTS)} keys)")

    # Seasonal adjustments — start neutral (1.0 for every species×month)
    seasonal = {}
    for sp_id in SPECIES:
        for month in range(1, 13):
            seasonal[f"{sp_id}_month_{month}"] = 1.0
    db().collection("ai_weights").document("seasonal").set(seasonal, merge=True)
    print(f"  ✓ seasonal weights ({len(seasonal)} keys)")


if __name__ == "__main__":
    _init_firebase()
    seed_spots()
    seed_weights()
    print("\n✅ Seed complete.")

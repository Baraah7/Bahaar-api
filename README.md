# بحار — Bahaar Fishing AI API

A production-ready Flask API that powers AI-driven fish presence prediction for the Bahaar mobile app. It learns from every user report via a continuous gradient-descent weight update system.

---

## Architecture Overview

```
POST /predict      →  PredictionEngine  →  probability (0–100)
POST /report       →  LearningEngine    →  weight update + spot detection
GET  /spots/nearby →  Firebase          →  sorted fishing spots
GET  /zone/info    →  static_data       →  zone characteristics
GET  /seasonal/advice → species data   →  monthly fishing advice
GET  /weather      →  Open-Meteo        →  conditions + safety
GET  /health       →  –                 →  uptime check
GET  /species      →  –                 →  all 20 species
```

### Core Prediction Equation

```
Probability = Static × Seasonal × Weather × Human × 100
```

| Factor | Source | Range |
|--------|--------|-------|
| **Static** | Zone base weight × species preference × learned weight × proximity bonus | 0.3 – 2.0 |
| **Seasonal** | Species calendar (12-month array) × learned seasonal adjustment | 0.1 – 1.5 |
| **Weather** | Temperature + wind vs species preferences | 0.2 – 1.5 |
| **Human** | Recency-weighted recent catch reports within 3 km | 0.7 – 1.5 |

---

## File Structure

```
bahaar-api/
├── api.py                          # Flask app, all 8 endpoints
├── learning_engine.py              # PredictionEngine + LearningEngine
├── firebase_service.py             # Firestore read/write + TTL cache
├── weather_service.py              # Open-Meteo atmospheric + marine
├── data/
│   └── static_data.py              # All zones, MPAs, species, 25 spots
├── tests/
│   └── test_learning_engine.py     # pytest suite
├── flutter_integration/
│   └── fishing_ai_service.dart     # Drop-in Flutter service class
├── docs/
│   └── Bahaar_AI_API.postman_collection.json
├── seed_firebase.py                # One-time Firestore seed script
├── firestore.rules                 # Security rules
├── firestore.indexes.json          # Composite indexes
├── render.yaml                     # Render.com deployment config
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Firebase

Get a service account key from Firebase Console → Project Settings → Service Accounts.

```bash
export FIREBASE_PROJECT_ID="your-project-id"
export FIREBASE_CLIENT_EMAIL="firebase-adminsdk-xxx@your-project.iam.gserviceaccount.com"
export FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
```

Or use a `.env` file (python-dotenv is included).

### 3. Seed Firestore (once)

```bash
python seed_firebase.py
```

This writes 25 fishing spots, static weights, and seasonal adjustments to Firestore.

### 4. Run locally

```bash
export FLASK_ENV=development
python api.py
# → Running on http://0.0.0.0:5000
```

### 5. Test

```bash
pytest tests/ -v
```

---

## API Reference

### POST /predict

```json
// Request
{
  "lat": 26.45,
  "lng": 50.52,
  "species_id": "hamour",
  "month": 10,
  "fetch_weather": true
}

// Response
{
  "probability": 73.4,
  "tier": "good",
  "tier_ar": "جيد",
  "zone": "northern",
  "zone_name_ar": "المنطقة الشمالية",
  "factors": {
    "static": 1.28,
    "seasonal": 1.15,
    "weather": 0.94,
    "human": 1.10
  },
  "species": {
    "name_ar": "هامور",
    "name_en": "Grouper",
    "peak_months": [10, 11, 12, 1, 2, 3]
  },
  "nearby_spots": [ ... ],
  "restrictions": [],
  "safety": {
    "level": "safe",
    "message_ar": "الطقس مناسب للصيد بالقوارب الصغيرة"
  }
}
```

### POST /report

```json
// Request
{
  "user_id": "uid_abc123",
  "lat": 26.45,
  "lng": 50.52,
  "species_id": "hamour",
  "success_rating": 5,
  "quantity": 4.5,
  "method": "gargoor",
  "predicted_probability": 73.4
}

// Response
{
  "learned": true,
  "report_id": "abc123",
  "error": 0.27,
  "weights_updated": true,
  "static_update": {
    "key": "northern_hamour",
    "old": 1.35,
    "new": 1.369
  },
  "new_spot_candidate": false
}
```

### GET /spots/nearby

```
GET /spots/nearby?lat=26.45&lng=50.52&radius=10&species=hamour
```

### GET /zone/info

```
GET /zone/info?lat=26.45&lng=50.52
```

### GET /seasonal/advice

```
GET /seasonal/advice?species_id=rubyan&month=4
```

### GET /weather

```
GET /weather?lat=26.2&lng=50.58
```

---

## Learning Algorithm

Every report triggers:

1. **Error calculation**:  `error = (success_rating / 5) − (predicted / 100)`
2. **Confidence scaling**: extreme ratings (1 or 5) carry more weight than neutral (3)
3. **Static weight update**:
   ```
   adjustment = 1 + (error × learning_rate × confidence)
   new_weight = clamp(old_weight × adjustment, 0.3, 2.0)
   ```
4. **Seasonal update** (½ learning rate):
   ```
   adjustment = 1 + (error × learning_rate × 0.5)
   ```
5. **New spot detection**: if ≥3 reports with rating ≥4 cluster within 500 m (not near a known spot) → saved to `spot_candidates` collection

---

## Firestore Collections

| Collection | Purpose | Read | Write |
|-----------|---------|------|-------|
| `fishing_spots` | Master spot list | Public | Admin |
| `user_reports` | User catch logs | Public | Auth (own uid) |
| `ai_weights` | Live prediction weights | Public | Backend SDK only |
| `learning_log` | Audit trail | Admin | Backend SDK only |
| `spot_candidates` | Unverified new spots | Auth | Backend SDK only |
| `listings` | Marketplace | Public | Owner |
| `orders` | Transactions | Owner/Admin | Owner/Admin |

---

## Deploy to Render

1. Push to GitHub
2. Create a new **Web Service** on render.com, connect your repo
3. Render auto-detects `render.yaml`
4. Add the 3 Firebase env vars in the Render dashboard (Settings → Environment)
5. Run `seed_firebase.py` once from your local machine with the same env vars

**Free tier note**: Render free services spin down after 15 min idle.  
Upgrade to Starter ($7/mo) for always-on, or add a cron ping to keep it warm.

---

## Flutter Integration

Copy `flutter_integration/fishing_ai_service.dart` into your project.  
Set the base URL:

```dart
final _api = FishingAIService(baseUrl: 'https://bahaar-fishing-ai.onrender.com');

// Predict
final result = await _api.predict(lat: 26.45, lng: 50.52, speciesId: 'hamour');
print(result.probability);  // 73.4
print(result.tierAr);       // جيد

// Submit report
await _api.submitReport(
  userId: currentUser.uid,
  lat: 26.45, lng: 50.52,
  speciesId: 'hamour',
  successRating: 5,
  quantityKg: 4.5,
  method: 'gargoor',
  predictedProbability: 73.4,
);
```

---

## Species IDs

| ID | Arabic | English |
|----|--------|---------|
| `hamour` | هامور | Grouper |
| `safi` | صافي | Rabbitfish |
| `shrimp` | ربيان | Shrimp |
| `crab` | غبغب | Blue Crab |
| `kanad` | كنعد | Spanish Mackerel |
| `emperor` | شعري | Spangled Emperor |
| `bream` | شعوم | Sea Bream |
| `barracuda` | بيكودا | Barracuda |
| `queenfish` | شيم | Queenfish |
| `squid` | حبار | Squid |

---

## Data Sources

- **MARGIS II GIS Survey** — Bahrain Centre for Studies & Research, 2006 (zone boundaries, spot coordinates)
- **Ali & Abahussain (2013)** — "Status of Commercial Fisheries in the Kingdom of Bahrain" (CPUE, species distribution)
- **FAO Fisheries Country Profile — Bahrain** (fleet, gear, landings statistics)
- **Open-Meteo + Marine API** — real-time weather (free, no key)

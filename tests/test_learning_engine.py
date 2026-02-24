"""
tests/test_learning_engine.py
=============================
Unit tests for the core AI / learning logic.
Run with:  python -m pytest tests/ -v
"""

import sys
import os
import math
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from learning_engine import (
    LearningEngine,
    PredictionEngine,
    classify_zone,
    get_mpa_restrictions,
    haversine_km,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def engine():
    return LearningEngine(learning_rate=0.05, new_spot_threshold=3)


@pytest.fixture
def predictor():
    return PredictionEngine()


@pytest.fixture
def base_weights():
    return {
        "northern_hamour": 1.35,
        "eastern_shrimp": 1.45,
        "western_kanad": 1.15,
    }


@pytest.fixture
def base_seasonal():
    return {
        "hamour_month_10": 1.2,
        "shrimp_month_7": 1.3,
    }


@pytest.fixture
def sample_spots():
    return [
        {
            "id": "fasht_al_jarim",
            "name_ar": "فشت الجاريم",
            "name_en": "Fasht Al-Jarim",
            "lat": 26.45, "lng": 50.52,
            "zone": "northern",
            "species": ["hamour", "safi", "shrimp"],
            "mpa": False,
            "base_score": 95,
        },
        {
            "id": "hawar_mpa",
            "name_ar": "جزر حوار",
            "name_en": "Hawar Islands",
            "lat": 25.67, "lng": 50.77,
            "zone": "southern",
            "species": ["hamour"],
            "mpa": True,
            "base_score": 20,
        },
    ]


def make_report(**kwargs):
    defaults = {
        "id": "test_report_1",
        "user_id": "user_001",
        "lat": 26.45,
        "lng": 50.52,
        "species_id": "hamour",
        "success_rating": 4,
        "quantity": 2.5,
        "method": "gargoor",
        "predicted_probability": 60.0,
        "month": 10,
        "timestamp": datetime.now(timezone.utc),
    }
    defaults.update(kwargs)
    return defaults


# ──────────────────────────────────────────────────────────────────────────────
# haversine_km
# ──────────────────────────────────────────────────────────────────────────────
class TestHaversine:
    def test_zero_distance(self):
        assert haversine_km(26.0, 50.5, 26.0, 50.5) == pytest.approx(0.0, abs=0.001)

    def test_known_distance(self):
        # Manama to Fasht Al-Jarim approx
        d = haversine_km(26.22, 50.57, 26.45, 50.52)
        assert 20 < d < 30  # roughly 25 km

    def test_symmetry(self):
        d1 = haversine_km(26.0, 50.5, 25.5, 50.8)
        d2 = haversine_km(25.5, 50.8, 26.0, 50.5)
        assert d1 == pytest.approx(d2, rel=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# Zone classification
# ──────────────────────────────────────────────────────────────────────────────
class TestClassifyZone:
    def test_northern_fasht(self):
        assert classify_zone(26.45, 50.52) == "northern"

    def test_eastern_fasht_azm(self):
        assert classify_zone(25.92, 50.65) == "eastern"

    def test_western_zallaq(self):
        assert classify_zone(26.02, 50.49) == "western"

    def test_southern_hawar(self):
        assert classify_zone(25.67, 50.77) == "southern"


# ──────────────────────────────────────────────────────────────────────────────
# MPA restrictions
# ──────────────────────────────────────────────────────────────────────────────
class TestMPARestrictions:
    def test_inside_hawar(self):
        # Hawar centre
        restr = get_mpa_restrictions(25.6667, 50.7667)
        ids = [r["id"] for r in restr]
        assert "hawar_islands" in ids

    def test_inside_tubli(self):
        restr = get_mpa_restrictions(26.1667, 50.5667)
        ids = [r["id"] for r in restr]
        assert "tubli_bay" in ids

    def test_outside_mpas(self):
        # Open water, northern zone
        restr = get_mpa_restrictions(26.45, 50.52)
        assert len(restr) == 0

    def test_restriction_type(self):
        restr = get_mpa_restrictions(25.6667, 50.7667)
        hawar = next(r for r in restr if r["id"] == "hawar_islands")
        assert hawar["restriction"] == "full"


# ──────────────────────────────────────────────────────────────────────────────
# Prediction engine
# ──────────────────────────────────────────────────────────────────────────────
class TestPredictionEngine:
    def test_mpa_blocked(self, predictor, base_weights, base_seasonal, sample_spots):
        # Inside Hawar Islands MPA
        result = predictor.predict(
            lat=25.6667, lng=50.7667,
            species_id="hamour",
            month=10,
            static_weights=base_weights,
            seasonal_weights=base_seasonal,
            spots=sample_spots,
            recent_reports=[],
        )
        assert result["mpa_blocked"] is True
        assert result["probability"] <= 10

    def test_northern_hamour_good_conditions(
        self, predictor, base_weights, base_seasonal, sample_spots
    ):
        result = predictor.predict(
            lat=26.45, lng=50.52,
            species_id="hamour",
            month=10,  # peak month
            static_weights=base_weights,
            seasonal_weights=base_seasonal,
            spots=sample_spots,
            recent_reports=[],
            temp=26.0,
            wind=10.0,
        )
        assert not result["mpa_blocked"]
        assert result["probability"] > 40
        assert result["zone"] == "northern"

    def test_off_season_reduces_probability(
        self, predictor, base_weights, base_seasonal, sample_spots
    ):
        peak = predictor.predict(
            lat=26.45, lng=50.52, species_id="hamour",
            month=10, static_weights=base_weights, seasonal_weights=base_seasonal,
            spots=sample_spots, recent_reports=[],
        )
        off_season = predictor.predict(
            lat=26.45, lng=50.52, species_id="hamour",
            month=6,  # low season for hamour
            static_weights=base_weights, seasonal_weights=base_seasonal,
            spots=sample_spots, recent_reports=[],
        )
        assert peak["probability"] > off_season["probability"]

    def test_probability_bounds(
        self, predictor, base_weights, base_seasonal, sample_spots
    ):
        result = predictor.predict(
            lat=26.45, lng=50.52, species_id="safi",
            month=10, static_weights=base_weights, seasonal_weights=base_seasonal,
            spots=sample_spots, recent_reports=[],
        )
        assert 0.0 <= result["probability"] <= 100.0

    def test_human_factor_boosts_probability(
        self, predictor, sample_spots
    ):
        # Use reduced weights so we're not near the 100% cap
        low_weights = {"northern_hamour": 0.6}
        recent = [
            {
                "success_rating": 5,
                "timestamp": datetime.now(timezone.utc) - timedelta(days=1),
            }
            for _ in range(3)
        ]
        no_reports = predictor.predict(
            lat=26.45, lng=50.52, species_id="hamour",
            month=10, static_weights=low_weights, seasonal_weights={},
            spots=sample_spots, recent_reports=[],
        )
        with_reports = predictor.predict(
            lat=26.45, lng=50.52, species_id="hamour",
            month=10, static_weights=low_weights, seasonal_weights={},
            spots=sample_spots, recent_reports=recent,
        )
        # Human factor from 3× rating-5 reports should boost probability
        assert with_reports["factors"]["human"] > no_reports["factors"]["human"]
        assert with_reports["probability"] >= no_reports["probability"]

    def test_bad_weather_reduces_factor(self, predictor):
        good = predictor.calculate_weather_factor("hamour", temp=26.0, wind=10.0)
        bad_temp = predictor.calculate_weather_factor("hamour", temp=38.0, wind=10.0)
        bad_wind = predictor.calculate_weather_factor("hamour", temp=26.0, wind=60.0)
        assert good > bad_temp
        assert good > bad_wind

    def test_no_weather_neutral(self, predictor):
        factor = predictor.calculate_weather_factor("hamour", None, None)
        assert factor == pytest.approx(1.0)

    def test_seasonal_factor_ranges(self, predictor):
        for month in range(1, 13):
            f = predictor.calculate_seasonal_factor("hamour", month, {})
            assert 0.0 <= f <= 2.0, f"Month {month} factor {f} out of range"

    def test_human_factor_no_reports(self, predictor):
        assert predictor.calculate_human_factor([]) == pytest.approx(1.0)

    def test_human_factor_fresh_success(self, predictor):
        reports = [
            {"success_rating": 5, "timestamp": datetime.now(timezone.utc) - timedelta(hours=2)}
        ]
        f = predictor.calculate_human_factor(reports)
        assert f > 1.0  # should boost

    def test_human_factor_old_failure(self, predictor):
        reports = [
            {"success_rating": 1, "timestamp": datetime.now(timezone.utc) - timedelta(days=12)}
        ]
        f = predictor.calculate_human_factor(reports)
        assert f < 1.0  # should reduce


# ──────────────────────────────────────────────────────────────────────────────
# Learning engine
# ──────────────────────────────────────────────────────────────────────────────
class TestLearningEngine:
    def test_error_positive_when_underestimated(self, engine):
        # We predicted 40%, user rated 5/5 (100%)
        error = engine.calculate_error(40.0, 5)
        assert error > 0  # we underestimated

    def test_error_negative_when_overestimated(self, engine):
        # We predicted 90%, user rated 2/5 (40%)
        error = engine.calculate_error(90.0, 2)
        assert error < 0  # we overestimated

    def test_error_zero_perfect_prediction(self, engine):
        # Predicted 80% (0.8), rated 4/5 (0.8)
        error = engine.calculate_error(80.0, 4)
        assert abs(error) < 0.001

    def test_weight_increases_on_underestimate(self, engine):
        new = engine.update_static_weight(1.0, error=+0.3)
        assert new > 1.0

    def test_weight_decreases_on_overestimate(self, engine):
        new = engine.update_static_weight(1.0, error=-0.3)
        assert new < 1.0

    def test_weight_clamped_upper(self, engine):
        new = engine.update_static_weight(1.9, error=+0.99, confidence=2.0)
        assert new <= 2.0

    def test_weight_clamped_lower(self, engine):
        new = engine.update_static_weight(0.35, error=-0.99, confidence=2.0)
        assert new >= 0.3

    def test_seasonal_update_half_rate(self, engine):
        static_new  = engine.update_static_weight(1.0, error=+0.5)
        seasonal_new = engine.update_seasonal_factor(1.0, error=+0.5)
        # Seasonal should change less than static
        static_change   = abs(static_new - 1.0)
        seasonal_change = abs(seasonal_new - 1.0)
        assert seasonal_change < static_change

    def test_process_report_returns_updates(self, engine, sample_spots):
        report = make_report()
        result = engine.process_report(
            report=report,
            current_weights={"northern_hamour": 1.35},
            current_seasonal={"hamour_month_10": 1.2},
            nearby_recent_reports=[],
            known_spots=sample_spots,
        )
        assert result["weights_updated"] is True
        assert "static_update" in result
        assert "seasonal_update" in result
        assert "log_entry" in result

    def test_no_candidate_single_report(self, engine, sample_spots):
        report = make_report(success_rating=5)
        result = engine.process_report(
            report=report,
            current_weights={},
            current_seasonal={},
            nearby_recent_reports=[report],  # only 1 report nearby
            known_spots=sample_spots,
        )
        # Only 1 report — not enough to trigger candidate
        assert result["spot_candidate"] is None

    def test_candidate_triggered_three_reports(self, engine):
        base = {"lat": 26.30, "lng": 50.60, "species_id": "hamour"}
        reports = [
            {**base, "id": f"r{i}", "success_rating": 5,
             "timestamp": datetime.now(timezone.utc) - timedelta(days=i)}
            for i in range(4)
        ]
        report = make_report(lat=26.30, lng=50.60, success_rating=5)
        result = engine.process_report(
            report=report,
            current_weights={},
            current_seasonal={},
            nearby_recent_reports=reports,  # 4 recent successful reports
            known_spots=[],  # no known spots nearby
        )
        assert result["spot_candidate"] is not None
        cand = result["spot_candidate"]
        assert cand["status"] == "candidate"
        assert cand["primary_species"] == "hamour"

    def test_no_duplicate_candidate_near_known_spot(self, engine, sample_spots):
        # Reports near Fasht Al-Jarim (already known)
        reports = [
            {
                "lat": 26.451, "lng": 50.519,
                "species_id": "hamour",
                "success_rating": 5,
                "id": f"r{i}",
                "timestamp": datetime.now(timezone.utc),
            }
            for i in range(5)
        ]
        report = make_report(lat=26.451, lng=50.519, success_rating=5)
        result = engine.process_report(
            report=report,
            current_weights={},
            current_seasonal={},
            nearby_recent_reports=reports,
            known_spots=sample_spots,  # contains fasht_al_jarim at 26.45, 50.52
        )
        # Within 1 km of known spot — no candidate
        assert result["spot_candidate"] is None


# ──────────────────────────────────────────────────────────────────────────────
# Integration — full predict cycle
# ──────────────────────────────────────────────────────────────────────────────
class TestFullCycle:
    def test_predict_then_report_updates_weight(self, engine, predictor, base_weights, sample_spots):
        # Step 1: predict
        prediction = predictor.predict(
            lat=26.45, lng=50.52, species_id="hamour",
            month=10, static_weights=base_weights, seasonal_weights={},
            spots=sample_spots, recent_reports=[],
        )
        prob = prediction["probability"]

        # Step 2: user reports higher success than predicted
        report = make_report(
            lat=26.45, lng=50.52,
            species_id="hamour",
            success_rating=5,  # great catch
            predicted_probability=prob,
            month=10,
        )
        result = engine.process_report(
            report=report,
            current_weights=base_weights,
            current_seasonal={},
            nearby_recent_reports=[],
            known_spots=sample_spots,
        )

        # Weight should have increased (we underestimated)
        new_w = result["new_static_weights"]["northern_hamour"]
        assert new_w >= base_weights.get("northern_hamour", 1.0)

    def test_consistent_factors(self, predictor, base_weights, sample_spots):
        """All factor components should contribute to final probability."""
        result = predictor.predict(
            lat=26.45, lng=50.52, species_id="hamour",
            month=10, static_weights=base_weights, seasonal_weights={},
            spots=sample_spots, recent_reports=[],
            temp=25.0, wind=12.0,
        )
        factors = result["factors"]
        assert all(0 < v <= 2.0 for v in factors.values()), f"Bad factors: {factors}"

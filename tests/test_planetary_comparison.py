"""
test_planetary_comparison.py - Unit tests for Vornis AI Planetary Comparison & Analysis Module.
Tests scoring functions, mission weights, selection validation, and planetary data layer.
"""

import unittest
import sys
from pathlib import Path

# Add api directory to sys.path
api_dir = Path(__file__).resolve().parent.parent / "api"
if str(api_dir) not in sys.path:
    sys.path.insert(0, str(api_dir))

from planetary_data import PLANETARY_DATA, PRESETS
from feasibility_engine import (
    MISSION_WEIGHTS,
    calculate_sub_scores,
    calculate_feasibility,
    compare_and_rank_bodies,
    MISSION_TYPES
)
from ai_analysis import _generate_rule_based_analysis


class TestPlanetaryDataLayer(unittest.TestCase):
    def test_all_expected_bodies_exist(self):
        expected_bodies = [
            "Mercury", "Venus", "Earth", "Moon", "Mars", "Phobos", "Ceres",
            "Jupiter", "Io", "Europa", "Ganymede", "Callisto", "Saturn",
            "Titan", "Enceladus", "Uranus", "Neptune", "Triton", "Pluto"
        ]
        for name in expected_bodies:
            self.assertIn(name, PLANETARY_DATA, f"Missing expected body: {name}")

    def test_body_schema_completeness(self):
        for name, data in PLANETARY_DATA.items():
            self.assertIn("physical", data, f"{name} missing physical data")
            self.assertIn("atmospheric", data, f"{name} missing atmospheric data")
            self.assertIn("environmental", data, f"{name} missing environmental data")

            phys = data["physical"]
            self.assertGreater(phys["radius_km"], 0, f"{name} invalid radius")
            self.assertGreater(phys["mass_1e24_kg"], 0, f"{name} invalid mass")
            self.assertGreater(phys["distance_from_sun_au"], 0, f"{name} invalid solar distance")

            atm = data["atmospheric"]
            self.assertIn("surface_pressure_bar", atm)
            self.assertIn("composition", atm)
            self.assertIsInstance(atm["composition"], list)

            env = data["environmental"]
            self.assertIn("mean_temp_c", env)
            self.assertIn("radiation_tier", env)
            self.assertIn("water_or_ice", env)

    def test_presets_integrity(self):
        self.assertGreaterEqual(len(PRESETS), 4)
        for preset in PRESETS:
            self.assertIn("id", preset)
            self.assertIn("name", preset)
            self.assertIn("bodies", preset)
            self.assertTrue(2 <= len(preset["bodies"]) <= 4)
            for b in preset["bodies"]:
                self.assertIn(b, PLANETARY_DATA, f"Preset references unknown body: {b}")


class TestFeasibilityEngine(unittest.TestCase):
    def test_weights_sum_to_one(self):
        for m_type, weights in MISSION_WEIGHTS.items():
            total = sum(weights.values())
            self.assertAlmostEqual(total, 1.0, places=3, msg=f"Weights for {m_type} must sum to 1.0 (got {total})")

    def test_sub_score_bounds(self):
        for name, data in PLANETARY_DATA.items():
            for m_type in MISSION_TYPES:
                sub_scores = calculate_sub_scores(data, m_type)
                self.assertEqual(len(sub_scores), 7, f"Expected 7 sub-scores for {name}")
                for score_key, val in sub_scores.items():
                    self.assertGreaterEqual(val, 0.0, f"{name} {score_key} < 0 for {m_type}")
                    self.assertLessEqual(val, 100.0, f"{name} {score_key} > 100 for {m_type}")

    def test_feasibility_ranking_and_tiers(self):
        earth = PLANETARY_DATA["Earth"]
        mars = PLANETARY_DATA["Mars"]
        jupiter = PLANETARY_DATA["Jupiter"]

        res_earth = calculate_feasibility(earth, "Crewed")
        res_mars = calculate_feasibility(mars, "Crewed")
        res_jup = calculate_feasibility(jupiter, "Crewed")

        # Earth should be high feasibility for crewed
        self.assertGreaterEqual(res_earth["overall_score"], 70.0)
        self.assertEqual(res_earth["tier"], "High Feasibility")

        # Jupiter surface is impossible/deadly for Crewed landing
        self.assertLess(res_jup["overall_score"], 45.0)
        self.assertEqual(res_jup["tier"], "Low Feasibility / High Risk")

        ranked = compare_and_rank_bodies([earth, mars, jupiter], "Crewed")
        self.assertEqual(ranked[0]["body_name"], "Earth")
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertEqual(ranked[1]["body_name"], "Mars")
        self.assertEqual(ranked[1]["rank"], 2)
        self.assertEqual(ranked[2]["body_name"], "Jupiter")
        self.assertEqual(ranked[2]["rank"], 3)

    def test_mission_type_weight_adjustment(self):
        # A flyby mission to Jupiter is far more feasible than a crewed surface mission
        jupiter = PLANETARY_DATA["Jupiter"]
        flyby_score = calculate_feasibility(jupiter, "Flyby")["overall_score"]
        crewed_score = calculate_feasibility(jupiter, "Crewed")["overall_score"]
        self.assertGreater(flyby_score, crewed_score, "Flyby of Jupiter should score higher than Crewed mission")


class TestSelectionValidation(unittest.TestCase):
    def validate_selection(self, body_names: list) -> tuple:
        """Helper replicating the UI/API selection validator rule."""
        if not body_names or len(body_names) < 2:
            return False, "Please select at least 2 celestial bodies to compare."
        if len(body_names) > 4:
            return False, "Maximum 4 celestial bodies can be compared at once."
        for name in body_names:
            if name not in PLANETARY_DATA:
                return False, f"Unknown body: {name}"
        return True, "Valid selection"

    def test_selection_limits(self):
        # Under minimum (< 2)
        valid, msg = self.validate_selection([])
        self.assertFalse(valid)
        self.assertIn("at least 2", msg)

        valid, msg = self.validate_selection(["Mars"])
        self.assertFalse(valid)
        self.assertIn("at least 2", msg)

        # Within range (2 to 4)
        valid, _ = self.validate_selection(["Earth", "Mars"])
        self.assertTrue(valid)

        valid, _ = self.validate_selection(["Earth", "Moon", "Mars"])
        self.assertTrue(valid)

        valid, _ = self.validate_selection(["Earth", "Mars", "Europa", "Titan"])
        self.assertTrue(valid)

        # Over maximum (> 4)
        valid, msg = self.validate_selection(["Earth", "Mars", "Europa", "Titan", "Enceladus"])
        self.assertFalse(valid)
        self.assertIn("Maximum 4", msg)


class TestAIAnalysisEngine(unittest.TestCase):
    def test_deterministic_analysis_structure(self):
        bodies = [PLANETARY_DATA["Earth"], PLANETARY_DATA["Mars"]]
        feasibility = compare_and_rank_bodies(bodies, "Crewed")
        analysis = _generate_rule_based_analysis(bodies, "Crewed", feasibility)

        self.assertIn("summary", analysis)
        self.assertIn("key_differences", analysis)
        self.assertIn("body_assessments", analysis)
        self.assertIn("mission_recommendation", analysis)
        self.assertIn("uncertainties_and_open_questions", analysis)

        self.assertIn("Earth", analysis["body_assessments"])
        self.assertIn("Mars", analysis["body_assessments"])
        self.assertIn("strengths", analysis["body_assessments"]["Earth"])
        self.assertIn("risks", analysis["body_assessments"]["Earth"])


if __name__ == "__main__":
    unittest.main()

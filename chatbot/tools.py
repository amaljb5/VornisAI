"""
tools.py - connects the chatbot to the VornisAI planet database and the
trained mission-feasibility model.

    get_planet_data              -> planet_data.PLANETS_DB
    compare_planets              -> planet_data.PLANETS_DB
    predict_mission_feasibility  -> model/feasibility_model.pkl (trained model)

Every function returns a plain dict. On failure it returns {"error": "..."}.
"""

import json
import math
from functools import lru_cache
from pathlib import Path

from planet_data import PLANETS_DB

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "api" / "model" / "feasibility_model.pkl"
META_PATH = MODEL_PATH.with_name("model_metadata.json")
MISSION_TYPES = ["Orbiter", "Rover", "Lander", "Flyby"]
# Used only if model_metadata.json is missing; the real list comes from the file.
_FALLBACK_TRAINED_BODIES = ["Europa", "Jupiter", "Mars", "Mercury", "Neptune", "Pluto", "Saturn", "Titan", "Uranus", "Venus"]
_G = 6.6743e-11

_LOOKUP = {name.lower(): name for name in PLANETS_DB}


def _resolve_body(name: str):
    return _LOOKUP.get(str(name).strip().lower())


def _normalize_mission(raw: str):
    """Same matching rules as the web app (orb / rov / lan / fly)."""
    r = str(raw or "Orbiter").strip().lower()
    for prefix, canon in (("orb", "Orbiter"), ("rov", "Rover"), ("lan", "Lander"), ("fly", "Flyby")):
        if r.startswith(prefix):
            return canon
    return None


def _escape_velocity_km_s(p: dict) -> float:
    return math.sqrt(2 * _G * (p["mass"] * 1e24) / (p["radius"] * 1000)) / 1000


def _unknown_body(name: str) -> dict:
    return {"error": f"No data for '{name}'. Known bodies: {', '.join(PLANETS_DB)}"}


@lru_cache(maxsize=1)
def _load_model():
    import joblib  # imported lazily so the chat UI still starts without it

    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _load_meta() -> dict:
    try:
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _reliability_notes(body: str, user_inputs: dict):
    """Compare the request with what the model was trained on (from the metadata file)."""
    meta = _load_meta()
    warnings = []

    trained = meta.get("cat_unique_values", {}).get("target", _FALLBACK_TRAINED_BODIES)
    if body not in trained:
        warnings.append(
            f"The model was never trained on {body}, so this score is an extrapolation and is unreliable."
        )

    stats = meta.get("num_feature_stats", {})
    for name, value in user_inputs.items():
        st = stats.get(name)
        if st and not (st["min"] <= value <= st["max"]):
            warnings.append(
                f"{name}={value:g} is outside the range the model was trained on "
                f"({st['min']:g} to {st['max']:g}), so treat the score with extra caution."
            )

    note = "Score is an estimate from a small trained model."
    if meta.get("dataset_rows") and meta.get("metrics", {}).get("MAE") is not None:
        note = (
            f"Estimate from a model trained on only {meta['dataset_rows']} examples; "
            f"typical error is about {meta['metrics']['MAE']} points on the 0-100 scale."
        )
    return warnings, note


# --------------------------------------------------------------------------
# Tool functions
# --------------------------------------------------------------------------

def get_planet_data(planet: str) -> dict:
    body = _resolve_body(planet)
    if not body:
        return _unknown_body(planet)
    p = PLANETS_DB[body]
    return {
        "planet": body,
        "distance_AU": p["dist"],
        "mass_1e24_kg": p["mass"],
        "radius_km": p["radius"],
        "gravity_m_s2": p["gravity"],
        "escape_velocity_km_s": round(_escape_velocity_km_s(p), 2),
        "temperature_C": p["temp"],
        "pressure_bar": p["press"],
        "has_global_magnetic_field": bool(p["mag"]),
        "gas_giant": bool(p["gas"]),
        "solid_surface": bool(p["solid"]),
        "description": p["desc"],
    }


def compare_planets(planet_a: str, planet_b: str) -> dict:
    a, b = get_planet_data(planet_a), get_planet_data(planet_b)
    if "error" in a:
        return a
    if "error" in b:
        return b
    return {"a": a, "b": b}


def predict_mission_feasibility(
    planet: str,
    mission_type: str = "Orbiter",
    spacecraft_mass_kg: float = 2200,
    science_payload_kg: float = 120,
    cruise_days: float = 250,
) -> dict:
    body = _resolve_body(planet)
    if not body:
        return _unknown_body(planet)

    mission = _normalize_mission(mission_type)
    if not mission:
        return {"error": f"Unknown mission type '{mission_type}'. Use one of: {', '.join(MISSION_TYPES)}"}

    try:
        mass, payload, days = float(spacecraft_mass_kg), float(science_payload_kg), float(cruise_days)
    except (TypeError, ValueError):
        return {"error": "Mass, payload and cruise days must be numbers."}
    if mass <= 0 or payload < 0 or days <= 0:
        return {"error": "Mass and cruise days must be positive, and payload cannot be negative."}
    if payload >= mass:
        return {"error": "Science payload must be smaller than total spacecraft mass."}

    p = PLANETS_DB[body]
    # Same feature set and order as api/index.py -> /api/predict
    features = {
        "target": body,
        "mission_type": mission,
        "cruise_days": days,
        "spacecraft_mass_kg": mass,
        "science_payload_kg": payload,
        "historical_success": 1,  # the web app always sends 1
        "distance_AU": p["dist"],
        "mass_1e24_kg": p["mass"],
        "radius_km": p["radius"],
        "gravity_m_s2": p["gravity"],
        "escape_velocity_km_s": _escape_velocity_km_s(p),
        "temperature_C": p["temp"],
        "pressure_bar": p["press"],
        "global_magnetic_field": p["mag"],
        "gas_giant": p["gas"],
        "solid_surface": p["solid"],
        "is_orbiter": int(mission == "Orbiter"),
        "is_rover": int(mission == "Rover"),
        "is_lander": int(mission == "Lander"),
        "is_flyby": int(mission == "Flyby"),
    }

    try:
        import pandas as pd

        raw = _load_model().predict(pd.DataFrame([features]))[0]
    except FileNotFoundError:
        return {"error": f"Model file not found at {MODEL_PATH}. Copy feasibility_model.pkl into the model/ folder."}
    except Exception as e:
        return {"error": f"Model could not run ({type(e).__name__}: {e})"}

    score = max(0.0, min(100.0, float(raw)))
    if score >= 70:
        rating = "HIGH FEASIBILITY"
    elif score >= 45:
        rating = "MODERATE FEASIBILITY"
    else:
        rating = "LOW FEASIBILITY / HIGH RISK"

    warnings, note = _reliability_notes(
        body,
        {"spacecraft_mass_kg": mass, "science_payload_kg": payload, "cruise_days": days},
    )
    result = {
        "planet": body,
        "mission_type": mission,
        "spacecraft_mass_kg": mass,
        "science_payload_kg": payload,
        "cruise_days": days,
        "feasibility_score": round(score, 1),
        "score_scale": "0-100 (70+ high, 45-69 moderate, below 45 low)",
        "rating": rating,
        "source": "VornisAI trained feasibility model",
        "model_note": note,
    }
    if warnings:
        result["warnings"] = warnings
    return result


# --------------------------------------------------------------------------
# Tool registry (the chatbot reads these two names)
# --------------------------------------------------------------------------

TOOL_FUNCTIONS = {
    "get_planet_data": get_planet_data,
    "compare_planets": compare_planets,
    "predict_mission_feasibility": predict_mission_feasibility,
}

_BODY = {"type": "string", "description": "Body name, e.g. Mars, Europa, Titan"}

TOOLS = [
    {
        "name": "get_planet_data",
        "description": "Get physical and environmental data for one planet or moon from the VornisAI database.",
        "input_schema": {"type": "object", "properties": {"planet": _BODY}, "required": ["planet"]},
    },
    {
        "name": "compare_planets",
        "description": "Compare two planets or moons side by side using the VornisAI database.",
        "input_schema": {
            "type": "object",
            "properties": {"planet_a": _BODY, "planet_b": _BODY},
            "required": ["planet_a", "planet_b"],
        },
    },
    {
        "name": "predict_mission_feasibility",
        "description": (
            "Run VornisAI's trained ML model to score how feasible a space mission is (0-100). "
            "If the user did not give mass, payload or cruise days, leave them out to use defaults. "
            "The result may include warnings about reliability; always pass them on to the user."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "planet": _BODY,
                "mission_type": {"type": "string", "enum": MISSION_TYPES},
                "spacecraft_mass_kg": {"type": "number", "description": "Total spacecraft mass in kg (default 2200)"},
                "science_payload_kg": {"type": "number", "description": "Science payload mass in kg (default 120)"},
                "cruise_days": {"type": "number", "description": "Cruise duration in days (default 250)"},
            },
            "required": ["planet"],
        },
    },
]
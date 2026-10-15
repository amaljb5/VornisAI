"""
feasibility_engine.py - Transparent, Physics-Grounded Mission Feasibility Scoring Engine
Part of Vornis AI - Planetary Comparison & Analysis Module.

Formula:
    Composite Score = sum(sub_score_i * weight_i for i in sub_scores)
    All sub-scores and the composite score are clamped to [0, 100].

Color-Coded Tiers:
    - High Feasibility: >= 70.0 (Theme color: #00e676)
    - Moderate Feasibility: 45.0 - 69.9 (Theme color: #ffea00)
    - Low Feasibility / High Risk: < 45.0 (Theme color: #ff3d00)
"""

import math
from typing import Dict, Any, List, Tuple

# Speed of light in km/s and AU in km
SPEED_OF_LIGHT_KM_S = 299792.458
AU_IN_KM = 149597870.7

MISSION_TYPES = ["Flyby", "Orbiter", "Lander", "Crewed", "Colony"]

# Transparent weights dictionary by mission architecture. Each configuration sums to 1.00 (100%).
MISSION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Flyby": {
        "travel_time_delta_v": 0.35,
        "communication_delay": 0.25,
        "radiation_risk": 0.25,
        "resource_availability": 0.15,
        "landing_difficulty": 0.00,
        "surface_survivability": 0.00,
        "long_term_habitability": 0.00
    },
    "Orbiter": {
        "travel_time_delta_v": 0.30,
        "radiation_risk": 0.25,
        "communication_delay": 0.20,
        "resource_availability": 0.15,
        "surface_survivability": 0.10,
        "landing_difficulty": 0.00,
        "long_term_habitability": 0.00
    },
    "Lander": {
        "landing_difficulty": 0.25,
        "surface_survivability": 0.25,
        "travel_time_delta_v": 0.20,
        "communication_delay": 0.15,
        "radiation_risk": 0.10,
        "resource_availability": 0.05,
        "long_term_habitability": 0.00
    },
    "Crewed": {
        "surface_survivability": 0.25,
        "radiation_risk": 0.20,
        "landing_difficulty": 0.15,
        "travel_time_delta_v": 0.15,
        "long_term_habitability": 0.10,
        "communication_delay": 0.10,
        "resource_availability": 0.05
    },
    "Colony": {
        "long_term_habitability": 0.25,
        "resource_availability": 0.25,
        "surface_survivability": 0.20,
        "radiation_risk": 0.15,
        "landing_difficulty": 0.10,
        "travel_time_delta_v": 0.05,
        "communication_delay": 0.00
    }
}

SUB_SCORE_METADATA: Dict[str, Dict[str, str]] = {
    "landing_difficulty": {
        "label": "Landing Difficulty",
        "description": "Evaluates descent dynamics, atmospheric aerobraking potential vs crush risk, surface gravity, and terrain.",
        "higher_is": "Easier to land (higher score = lower hazard)"
    },
    "surface_survivability": {
        "label": "Surface Survivability",
        "description": "Assesses thermal extremes, surface pressure, atmospheric corrosiveness, and active volcanic activity.",
        "higher_is": "More benign surface conditions"
    },
    "communication_delay": {
        "label": "Communication Delay",
        "description": "Calculates one-way light time latency for telemetry, autonomous reaction buffer, and real-time intervention.",
        "higher_is": "Shorter latency (closer to Earth)"
    },
    "resource_availability": {
        "label": "Resource Availability (ISRU)",
        "description": "Quantifies access to water/ice, extractable atmospheric volatiles (CO₂, N₂, CH₄), and solar irradiance.",
        "higher_is": "Richer in-situ resources"
    },
    "radiation_risk": {
        "label": "Radiation Environment",
        "description": "Measures magnetospheric dipole shielding and atmospheric column absorption versus trapped belts and cosmic rays.",
        "higher_is": "Lower radiation hazard (safer for equipment and crew)"
    },
    "travel_time_delta_v": {
        "label": "Travel Time & Delta-v",
        "description": "Computes orbital transfer energy, C3 injection, Hohmann transfer time, and cruise longevity requirements.",
        "higher_is": "Less delta-v required and faster transit"
    },
    "long_term_habitability": {
        "label": "Long-Term Habitability",
        "description": "Evaluates gravity adequacy for musculoskeletal health, biosphere support, and self-sufficient settlement feasibility.",
        "higher_is": "More sustainable for extended human presence"
    }
}


def calculate_sub_scores(body: Dict[str, Any], mission_type: str = "Orbiter") -> Dict[str, float]:
    """
    Computes 7 transparent, physics-grounded sub-scores (0-100) for a given celestial body.
    """
    phys = body.get("physical", {})
    atm = body.get("atmospheric", {})
    env = body.get("environmental", {})
    b_type = body.get("type", "")

    dist_au = float(phys.get("distance_from_sun_au", 1.0))
    radius_km = float(phys.get("radius_km", 1000.0))
    mass_1e24 = float(phys.get("mass_1e24_kg", 0.1))
    gravity = float(phys.get("surface_gravity_m_s2", 3.7))
    escape_vel = float(phys.get("escape_velocity_km_s", 5.0))
    pressure = float(atm.get("surface_pressure_bar", 0.0))
    mean_temp = float(env.get("mean_temp_c", 0.0))
    min_temp = float(env.get("min_temp_c", -100.0))
    max_temp = float(env.get("max_temp_c", 100.0))
    rad_tier = env.get("radiation_tier", "Moderate")

    is_gas_giant = "Gas Giant" in b_type or "Ice Giant" in b_type

    # 1. Landing Difficulty (0-100; higher = easier/safer)
    if is_gas_giant:
        landing_score = 0.0  # Impossible to land on gas/ice giant surface
    elif mission_type in ["Flyby", "Orbiter"]:
        landing_score = 100.0  # Not landing, no penalty
    else:
        # Penalties: extreme gravity (>12 m/s²), extreme vacuum needing 100% propulsion, or extreme pressure
        score = 80.0
        # Gravity factor: optimal landing gravity is 1.5 - 5.0 m/s²
        if gravity > 12.0:
            score -= (gravity - 12.0) * 4.0
        elif gravity < 0.1:
            score -= 25.0  # Microgravity bouncing / anchoring challenge
        elif gravity < 1.0:
            score -= 10.0

        # Atmosphere factor:
        if 0.005 <= pressure <= 2.0:
            score += 15.0  # Useful for aerobraking & chutes without crush risk
        elif pressure > 10.0:
            score -= min(55.0, (pressure - 10.0) * 0.6)  # Crushing pressure
        else:
            score -= 15.0  # Airless vacuum requires heavy propellant for 100% retro-burn

        # Escape velocity penalty
        if escape_vel > 10.0:
            score -= (escape_vel - 10.0) * 3.0

        landing_score = max(5.0, min(95.0, score))

    # 2. Surface Survivability (0-100; higher = more benign)
    if is_gas_giant:
        surface_surv = 0.0
    else:
        # Optimal temperature range: -20 to +40 C
        temp_penalty = 0.0
        if mean_temp < -20:
            temp_penalty = min(50.0, abs(mean_temp - (-20)) * 0.25)
        elif mean_temp > 40:
            temp_penalty = min(65.0, (mean_temp - 40) * 0.16)

        diurnal_spread = abs(max_temp - min_temp)
        thermal_cycle_penalty = min(20.0, diurnal_spread * 0.04)

        # Pressure penalty
        press_penalty = 0.0
        if pressure > 5.0:
            press_penalty = min(50.0, pressure * 0.5)
        elif pressure < 0.001:
            press_penalty = 15.0  # Full vacuum requires rigid pressure hulls

        # Volcanism / Acid penalty
        hazard_penalty = 0.0
        if "Critical" in rad_tier or "Lethal" in rad_tier or "Active volcanism" in env.get("geological_activity", ""):
            hazard_penalty = 25.0

        surface_surv = max(2.0, min(100.0, 100.0 - temp_penalty - thermal_cycle_penalty - press_penalty - hazard_penalty))

    # 3. Communication Delay (0-100; higher = shorter latency)
    # Estimate distance from Earth (AU): approx |dist_au - 1.0| for baseline, with min 0.0026 AU for Moon
    if body.get("name") == "Earth":
        comm_score = 100.0
    elif body.get("name") == "Moon":
        comm_score = 98.0
    else:
        delta_au = max(0.2, abs(dist_au - 1.0))
        # 1-way light seconds = delta_au * AU_IN_KM / SPEED_OF_LIGHT_KM_S
        light_sec = (delta_au * AU_IN_KM) / SPEED_OF_LIGHT_KM_S
        # Scale: 0 sec = 100, 1000 sec (~16 min) = 70, 5000 sec (~1.4 hr) = 40, 20000 sec = 15
        comm_score = max(5.0, min(95.0, 100.0 * math.exp(-light_sec / 7000.0)))

    # 4. Resource Availability (ISRU) (0-100; higher = richer resources)
    resource_score = 20.0
    water_info = env.get("water_or_ice", "").lower()
    if "ocean" in water_info or "abundant" in water_info:
        resource_score += 45.0
    elif "ice" in water_info or "permafrost" in water_info or "pockets" in water_info:
        resource_score += 30.0

    # Atmospheric gases: CO2, N2, CH4, O2
    comp = atm.get("composition", [])
    has_carbon = any("Carbon" in g.get("gas", "") or "Methane" in g.get("gas", "") for g in comp)
    has_nitrogen = any("Nitrogen" in g.get("gas", "") for g in comp)
    if has_carbon and has_nitrogen:
        resource_score += 25.0
    elif has_carbon or has_nitrogen:
        resource_score += 15.0

    # Solar flux factor (1 / dist_au^2)
    solar_flux = 1.0 / (dist_au ** 2)
    if solar_flux >= 0.4:
        resource_score += 10.0
    elif solar_flux < 0.05:
        resource_score -= 10.0  # Requires nuclear power only

    resource_score = max(5.0, min(100.0, resource_score))

    # 5. Radiation Risk (0-100; higher = safer / lower risk)
    if rad_tier == "Safe":
        rad_score = 96.0
    elif rad_tier == "Low (Surface)":
        rad_score = 80.0
    elif rad_tier == "Moderate":
        rad_score = 65.0
    elif rad_tier == "High":
        rad_score = 45.0
    elif rad_tier == "Severe":
        rad_score = 25.0
    elif "Critical" in rad_tier or "Lethal" in rad_tier:
        rad_score = 5.0
    else:
        rad_score = 50.0

    # 6. Travel Time & Delta-v (0-100; higher = easier / faster / lower dv)
    if body.get("name") == "Earth":
        travel_score = 100.0
    elif body.get("name") == "Moon":
        travel_score = 95.0
    else:
        # Hohmann transfer semi-major axis: a = (1 + dist_au) / 2
        a_tr = (1.0 + dist_au) / 2.0
        # Transfer time in years = 0.5 * a_tr^(1.5)
        transfer_years = 0.5 * (a_tr ** 1.5)
        # Delta-v proxy: |sqrt(2 - 1/a_tr) - 1| * 29.78 km/s
        v_tr = 29.78 * math.sqrt(abs(2.0 - (1.0 / a_tr)))
        delta_v_earth = abs(v_tr - 29.78)
        # Total travel score scaled
        penalty = (transfer_years * 14.0) + (delta_v_earth * 3.5)
        travel_score = max(5.0, min(95.0, 100.0 - penalty))

    # 7. Long-term Habitability (0-100; higher = more sustainable)
    if is_gas_giant:
        habitability_score = 0.0
    elif body.get("name") == "Earth":
        habitability_score = 100.0
    else:
        h_score = 10.0
        # Gravity: 0.38g (Mars) gives decent basis; 0.16g (Moon) partial; microgravity minimal
        if 8.0 <= gravity <= 11.0:
            h_score += 25.0
        elif 3.5 <= gravity < 8.0:
            h_score += 18.0
        elif 1.0 <= gravity < 3.5:
            h_score += 10.0

        # Atmospheric shielding / pressure
        if 0.5 <= pressure <= 1.5:
            h_score += 25.0
        elif 0.005 <= pressure:
            h_score += 10.0

        # Water presence
        if "ocean" in water_info:
            h_score += 20.0
        elif "ice" in water_info:
            h_score += 12.0

        # Radiation protection
        if rad_score >= 60.0:
            h_score += 15.0
        elif rad_score <= 20.0:
            h_score -= 15.0

        habitability_score = max(2.0, min(95.0, h_score))

    return {
        "landing_difficulty": round(landing_score, 1),
        "surface_survivability": round(surface_surv, 1),
        "communication_delay": round(comm_score, 1),
        "resource_availability": round(resource_score, 1),
        "radiation_risk": round(rad_score, 1),
        "travel_time_delta_v": round(travel_score, 1),
        "long_term_habitability": round(habitability_score, 1)
    }


def calculate_feasibility(body: Dict[str, Any], mission_type: str = "Orbiter") -> Dict[str, Any]:
    """
    Computes total feasibility score and tier rating for a body given the mission type.
    """
    if mission_type not in MISSION_WEIGHTS:
        mission_type = "Orbiter"

    weights = MISSION_WEIGHTS[mission_type]
    sub_scores = calculate_sub_scores(body, mission_type)

    composite = 0.0
    for key, weight in weights.items():
        composite += sub_scores.get(key, 50.0) * weight

    composite = round(max(0.0, min(100.0, composite)), 1)

    if composite >= 70.0:
        tier = "High Feasibility"
        color = "#00e676"  # Theme High tier green
        badge_class = "tier-high"
    elif composite >= 45.0:
        tier = "Moderate Feasibility"
        color = "#ffea00"  # Theme Moderate tier yellow
        badge_class = "tier-moderate"
    else:
        tier = "Low Feasibility / High Risk"
        color = "#ff3d00"  # Theme Low tier red
        badge_class = "tier-low"

    return {
        "body_name": body.get("name", "Unknown"),
        "mission_type": mission_type,
        "overall_score": composite,
        "tier": tier,
        "color": color,
        "badge_class": badge_class,
        "weights": weights,
        "sub_scores": sub_scores,
        "sub_score_meta": SUB_SCORE_METADATA
    }


def compare_and_rank_bodies(bodies: List[Dict[str, Any]], mission_type: str = "Orbiter") -> List[Dict[str, Any]]:
    """
    Takes a list of bodies, calculates feasibility scores, and ranks them in descending order.
    """
    results = [calculate_feasibility(b, mission_type) for b in bodies]
    results.sort(key=lambda r: r["overall_score"], reverse=True)
    for idx, r in enumerate(results):
        r["rank"] = idx + 1
    return results

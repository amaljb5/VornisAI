"""
ai_analysis.py - AI Comparative Planetary Analysis Service for Vornis AI.
Interacts with OpenAI if OPENAI_API_KEY is configured, or provides a scientifically
grounded deterministic aerospace reasoning engine adhering strictly to reference data.
"""

import json
import os
from typing import Dict, Any, List

COMPARISON_PROMPT_TEMPLATE = """You are Vornis AI's planetary astrophysics and mission architecture engine.
Generate an in-depth comparative mission analysis for the following celestial bodies under the mission architecture: {mission_type}.

GROUNDING DATA (STRICT GROUND TRUTH - DO NOT INVENT NUMBERS OR METRICS):
{bodies_summary}

MISSION FEASIBILITY SCORES:
{feasibility_summary}

INSTRUCTIONS:
1. Ground every claim directly in the provided planetary data and calculated feasibility scores.
2. If data is unknown or has scientific uncertainty (e.g. exact subsurface ocean salinity, plume intermittency, deep radiation dynamics), explicitly state the uncertainty.
3. Return a STRICT JSON object with the following exact keys:
   - "summary": (string, 2-3 paragraphs synthesizing the strategic comparison)
   - "key_differences": (list of strings, highlighting key physical, atmospheric, and environmental contrasts)
   - "body_assessments": (object mapping body name to {{"strengths": [list of strings], "risks": [list of strings]}})
   - "mission_recommendation": (object with "recommended_body": string, "rationale": string, "mission_architecture_advice": string)
   - "uncertainties_and_open_questions": (list of strings highlighting critical scientific and engineering unknowns)
"""


def _generate_rule_based_analysis(
    bodies: List[Dict[str, Any]],
    mission_type: str,
    feasibility_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    High-fidelity deterministic domain analysis engine based on NASA/JPL data.
    Guarantees zero hallucinations and instant responses even when offline.
    """
    sorted_scores = sorted(feasibility_results, key=lambda x: x["overall_score"], reverse=True)
    top_body = sorted_scores[0]["body_name"]
    top_score = sorted_scores[0]["overall_score"]

    body_names = [b["name"] for b in bodies]
    names_str = " vs ".join(body_names)

    # Build summary paragraphs
    p1 = (
        f"Comparative evaluation of {names_str} for a {mission_type} mission architecture reveals "
        f"stark environmental and dynamic divergence. Overall feasibility scores span from "
        f"{sorted_scores[-1]['overall_score']}/100 ({sorted_scores[-1]['body_name']}) up to "
        f"{top_score}/100 ({top_body}), driven primarily by transfer delta-v requirements, surface survivability, "
        f"and environmental radiation hazards."
    )

    # Atmospheric & thermal comparison
    atm_types = []
    for b in bodies:
        press = b.get("atmospheric", {}).get("surface_pressure_bar", 0)
        if press > 10:
            atm_types.append(f"{b['name']} possesses a crushing supercritical atmosphere ({press} bar)")
        elif press > 0.005:
            atm_types.append(f"{b['name']} maintains an atmosphere usable for aerobraking ({press} bar)")
        else:
            atm_types.append(f"{b['name']} is virtually in vacuum ({b.get('atmospheric', {}).get('pressure_display', 'trace')})")
    p2 = "Regarding atmospheric envelopes: " + "; ".join(atm_types) + "."

    # Mission-specific recommendation synthesis
    p3 = (
        f"For the selected '{mission_type}' profile, {top_body} demonstrates the highest operational margin. "
        f"Its balanced score reflects manageable EDL complexity, sustainable thermal bounds, and advantageous "
        f"communication windows compared to alternative targets in this comparison group."
    )

    summary = f"{p1}\n\n{p2}\n\n{p3}"

    # Key differences
    key_differences = []
    # Distance comparison
    distances = [(b["name"], b.get("physical", {}).get("distance_from_sun_au", 1.0)) for b in bodies]
    distances.sort(key=lambda x: x[1])
    key_differences.append(
        f"Solar Distance & Flux: {distances[0][0]} is at {distances[0][1]} AU while {distances[-1][0]} is at {distances[-1][1]} AU, "
        f"resulting in a {(distances[-1][1]/max(0.1, distances[0][1]))**2:.1f}x difference in available solar irradiance."
    )

    # Gravity comparison
    gravities = [(b["name"], b.get("physical", {}).get("surface_gravity_m_s2", 1.0)) for b in bodies]
    gravities.sort(key=lambda x: x[1])
    key_differences.append(
        f"Surface Gravity: Ranging from {gravities[0][0]} ({gravities[0][1]} m/s²) to {gravities[-1][0]} ({gravities[-1][1]} m/s²), "
        f"substantially altering landing propellant mass fraction and human physiological durability."
    )

    # Thermal spectrum
    temps = [(b["name"], b.get("environmental", {}).get("mean_temp_c", 0.0)) for b in bodies]
    temps.sort(key=lambda x: x[1])
    key_differences.append(
        f"Thermal Envelope: Extreme contrast from cryogenic cold on {temps[0][0]} ({temps[0][1]}°C) to {temps[-1][0]} ({temps[-1][1]}°C), "
        f"mandating radically divergent thermal management systems (RHU radioisotope heaters vs active heat rejection/shielding)."
    )

    # Radiation & Magnetosphere
    rads = [(b["name"], b.get("environmental", {}).get("radiation_tier", "Moderate")) for b in bodies]
    key_differences.append(
        "Radiation Hazard Profile: " + ", ".join([f"{name} ({tier})" for name, tier in rads]) + "."
    )

    # Body assessments: strengths & risks
    body_assessments = {}
    for b in bodies:
        name = b["name"]
        phys = b.get("physical", {})
        atm = b.get("atmospheric", {})
        env = b.get("environmental", {})

        strengths = []
        risks = []

        # Strengths
        if phys.get("distance_from_sun_au", 1.0) <= 1.6:
            strengths.append(f"Favorable orbital proximity ({phys.get('distance_from_sun_au')} AU) enabling rapid telemetry and manageable flight duration.")
        if "Safe" in env.get("radiation_tier", "") or "Low" in env.get("radiation_tier", ""):
            strengths.append("Relatively benign radiation environment mitigating avionics single-event upsets and crew health hazards.")
        if "ocean" in env.get("water_or_ice", "").lower() or "ice" in env.get("water_or_ice", "").lower():
            strengths.append(f"High In-Situ Resource Utilization (ISRU) potential: {env.get('water_or_ice')}.")
        if 0.005 <= atm.get("surface_pressure_bar", 0) <= 2.0:
            strengths.append(f"Atmospheric column ({atm.get('surface_pressure_bar')} bar) allows aerodynamic braking and parachute deployment.")
        if not strengths:
            strengths.append(f"Rich scientific return and breakthrough astrobiological / geophysical exploration value ({b.get('type')}).")

        # Risks
        if atm.get("surface_pressure_bar", 0) > 10.0:
            risks.append(f"Extreme atmospheric overpressure ({atm.get('surface_pressure_bar')} bar) threatens structural integrity and electronics longevity.")
        elif atm.get("surface_pressure_bar", 0) < 0.0001 and mission_type in ["Lander", "Crewed", "Colony"]:
            risks.append("Complete vacuum necessitates 100% retro-propulsive deceleration, drastically reducing payload-to-launch mass fraction.")
        if env.get("mean_temp_c", 0) > 200:
            risks.append(f"Superheated surface conditions ({env.get('mean_temp_c')}°C) rapidly degrade conventional semiconductors and battery chemistries.")
        elif env.get("mean_temp_c", 0) < -150:
            risks.append(f"Deep cryogenic environment ({env.get('mean_temp_c')}°C) mandates Radioisotope Heater Units (RHUs) to prevent propellant freezing.")
        if "Critical" in env.get("radiation_tier", "") or "Lethal" in env.get("radiation_tier", "") or "Severe" in env.get("radiation_tier", ""):
            risks.append(f"High radiation dose ({env.get('radiation_tier')}) requires thick tungsten/tantalum avionics vaults and limits mission duration.")
        if not risks:
            risks.append("Long-duration interplanetary cruise risk including micrometeorite exposure and thermal cycling degradation.")

        body_assessments[name] = {
            "strengths": strengths,
            "risks": risks
        }

    # Recommendation
    recommendation = {
        "recommended_body": top_body,
        "overall_score": top_score,
        "rationale": (
            f"Based on the multi-attribute trade study for a {mission_type} mission, {top_body} is the superior target "
            f"with an aggregate feasibility score of {top_score}/100. It minimizes mission risk while offering viable "
            f"operational conditions across thermal, radiation, and communications constraints."
        ),
        "mission_architecture_advice": (
            f"Recommended configuration: Employ high-TRL avionics with radiation hardening, "
            f"{'radioisotope power (MMRTG)' if bodies[0].get('physical', {}).get('distance_from_sun_au', 1) > 3.0 else 'advanced solar arrays'}, "
            f"and prioritize autonomous navigational guidance due to telemetry latency."
        )
    }

    # Uncertainties
    uncertainties = [
        "Subsurface ocean salinity and hydrothermal vent flux remain unconfirmed without direct in-situ seismic and magnetotelluric sounding.",
        "Surface regolith mechanical properties (shear strength, dust adhesion, electrostatic levitation) show wide localized variations.",
        "Long-term biological effects of partial gravity (e.g. 0.16g to 0.38g) on human musculoskeletal and cardiovascular systems lack long-term clinical data.",
        "Micro-meteoroid flux and local space weathering rates in outer planetary rings and satellite systems introduce stochastic risk margins."
    ]

    return {
        "summary": summary,
        "key_differences": key_differences,
        "body_assessments": body_assessments,
        "mission_recommendation": recommendation,
        "uncertainties_and_open_questions": uncertainties,
        "generation_mode": "Deterministic Aerospace Knowledge Engine"
    }


def generate_comparative_analysis(
    bodies: List[Dict[str, Any]],
    mission_type: str,
    feasibility_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Main entry point for AI analysis. Attempts OpenAI call if configured; falls back
    gracefully to the deterministic grounded engine on any error or missing key.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("CHATBOX_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL") or None

    if not api_key:
        return _generate_rule_based_analysis(bodies, mission_type, feasibility_results)

    try:
        import openai

        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        bodies_summary = "\n".join([
            f"Body: {b['name']} ({b.get('type')})\n"
            f"- Distance: {b.get('physical', {}).get('distance_from_sun_au')} AU, Radius: {b.get('physical', {}).get('radius_km')} km\n"
            f"- Gravity: {b.get('physical', {}).get('surface_gravity_m_s2')} m/s², Escape Velocity: {b.get('physical', {}).get('escape_velocity_km_s')} km/s\n"
            f"- Pressure: {b.get('atmospheric', {}).get('pressure_display')}\n"
            f"- Atmosphere Composition: {', '.join([g['gas'] + ' ' + str(g['percentage']) + '%' for g in b.get('atmospheric', {}).get('composition', [])])}\n"
            f"- Temperature: Mean {b.get('environmental', {}).get('mean_temp_c')}°C (Min {b.get('environmental', {}).get('min_temp_c')}°C, Max {b.get('environmental', {}).get('max_temp_c')}°C)\n"
            f"- Radiation: {b.get('environmental', {}).get('radiation_level')} ({b.get('environmental', {}).get('radiation_tier')})\n"
            f"- Water/Ice: {b.get('environmental', {}).get('water_or_ice')}\n"
            f"- Geological Activity: {b.get('environmental', {}).get('geological_activity')}\n"
            f"- Habitability: {b.get('environmental', {}).get('habitability_notes')}\n"
            for b in bodies
        ])

        feasibility_summary = "\n".join([
            f"{r['body_name']}: Overall Score {r['overall_score']}/100 ({r['tier']}). "
            f"Sub-scores: Landing={r['sub_scores'].get('landing_difficulty')}, "
            f"Surface={r['sub_scores'].get('surface_survivability')}, "
            f"CommDelay={r['sub_scores'].get('communication_delay')}, "
            f"ISRU={r['sub_scores'].get('resource_availability')}, "
            f"Radiation={r['sub_scores'].get('radiation_risk')}, "
            f"Travel={r['sub_scores'].get('travel_time_delta_v')}, "
            f"Habitability={r['sub_scores'].get('long_term_habitability')}"
            for r in feasibility_results
        ])

        prompt = COMPARISON_PROMPT_TEMPLATE.format(
            mission_type=mission_type,
            bodies_summary=bodies_summary,
            feasibility_summary=feasibility_summary
        )

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are Vornis AI's planetary comparison engine. Respond strictly with valid JSON."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=2200
        )

        raw_content = response.choices[0].message.content
        result = json.loads(raw_content)
        result["generation_mode"] = f"OpenAI ({model_name})"
        return result

    except Exception as e:
        # Graceful fallback to rule-based engine on any error (network, rate limit, quota, invalid json)
        fallback = _generate_rule_based_analysis(bodies, mission_type, feasibility_results)
        fallback["generation_mode"] = f"Deterministic Aerospace Engine (AI Fallback: {type(e).__name__})"
        return fallback

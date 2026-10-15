"""
VornisAI — AI-Powered Space Mission & Vehicle Design Assistant
===============================================================
Single-Page Unified Dashboard featuring:
1. 📊 Mission Feasibility AI Score
2. 🧠 AI Vehicle Design Recommendations
3. 🪐 Planetary Environment Intelligence
4. ⚡ Launch & Transfer Physics Dynamics
5. 🔬 Planetary Comparison & Mission Analysis
"""

import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Make the shared planetary-analysis modules available when launched by Streamlit.
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from api.ai_analysis import generate_comparative_analysis
from api.feasibility_engine import (
    MISSION_TYPES,
    SUB_SCORE_METADATA,
    compare_and_rank_bodies,
)
from api.planetary_data import PLANETARY_DATA, PRESETS

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "feasibility_model.pkl"
META_PATH = BASE_DIR / "model" / "model_metadata.json"

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VornisAI — Spacecraft Design Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS Design ─────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #070913 0%, #0d1527 50%, #151d36 100%);
    color: #e2e8f0;
}

/* Hero Header */
.hero-header {
    text-align: center;
    padding: 1.2rem 1rem 0.8rem;
    margin-bottom: 1.2rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.hero-header h1 {
    font-family: 'Outfit', sans-serif;
    background: linear-gradient(135deg, #00d2ff 0%, #7a5cff 50%, #ff6ec7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
    letter-spacing: -0.5px;
}
.hero-header p {
    color: #94a3b8;
    font-size: 1.1rem;
    font-weight: 300;
}

/* Section Division Headers */
.section-title {
    font-family: 'Outfit', sans-serif;
    color: #38bdf8;
    font-size: 1.4rem;
    font-weight: 600;
    margin-top: 1rem;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    border-bottom: 1px solid rgba(56, 189, 248, 0.2);
    padding-bottom: 0.4rem;
}

/* Glassmorphism Card */
.glass-card {
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.4rem;
    backdrop-filter: blur(16px);
    margin-bottom: 1rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
}

/* Score Display */
.score-card {
    text-align: center;
    padding: 1.6rem 1rem;
    border-radius: 16px;
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(122, 92, 255, 0.3);
    height: 100%;
}
.score-number {
    font-family: 'Outfit', sans-serif;
    font-size: 4.8rem;
    font-weight: 700;
    line-height: 1;
    margin: 0.4rem 0;
}
.score-label {
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #94a3b8;
    font-size: 0.85rem;
    font-weight: 600;
}

/* AI Recommendation Boxes */
.rec-box {
    background: rgba(122, 92, 255, 0.07);
    border-left: 4px solid #7a5cff;
    padding: 0.9rem 1.1rem;
    border-radius: 0 12px 12px 0;
    margin-bottom: 0.9rem;
}
.rec-box h4 {
    color: #38bdf8;
    margin: 0 0 0.3rem 0;
    font-size: 1.05rem;
    font-weight: 600;
}
.rec-box p {
    color: #cbd5e1;
    margin: 0;
    font-size: 0.92rem;
    line-height: 1.4;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: rgba(7, 9, 19, 0.95) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}

/* Metric Display */
div[data-testid="stMetricValue"] {
    font-family: 'Outfit', sans-serif;
    color: #38bdf8 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Load Model & Metadata ─────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    return model, meta

model, meta = load_resources()

def render_planetary_comparison():
    st.markdown(
        "<div class='section-title'>🔬 Planetary Comparison & Mission Analysis</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Compare mission feasibility across 19 planets, dwarf planets, and major moons."
    )

    preset_names = {preset["name"]: preset["bodies"] for preset in PRESETS}

    def apply_comparison_preset():
        selected_preset = st.session_state["comparison_preset"]
        if selected_preset in preset_names:
            st.session_state["comparison_bodies"] = preset_names[selected_preset]

    preset_name = st.selectbox(
        "Quick comparison",
        ["Custom"] + list(preset_names),
        key="comparison_preset",
        on_change=apply_comparison_preset,
    )
    default_bodies = preset_names.get(preset_name, ["Earth", "Mars"])
    selected_bodies = st.multiselect(
        "Celestial bodies (select 2–4)",
        options=list(PLANETARY_DATA),
        default=default_bodies,
        max_selections=4,
        key="comparison_bodies",
    )
    mission_type = st.selectbox("Mission architecture", MISSION_TYPES, index=3)

    if not 2 <= len(selected_bodies) <= 4:
        st.info("Select between 2 and 4 celestial bodies to compare.")
        return

    bodies = [PLANETARY_DATA[name] for name in selected_bodies]
    results = compare_and_rank_bodies(bodies, mission_type)
    scores = pd.DataFrame(
        {
            "Feasibility score": {
                result["body_name"]: result["overall_score"] for result in results
            }
        }
    )

    score_tab, analysis_tab, details_tab = st.tabs(
        ["Feasibility scores", "AI analysis", "Body details"]
    )
    with score_tab:
        st.bar_chart(scores, y="Feasibility score", color="#38bdf8")
        for result in results:
            st.markdown(
                f"**#{result['rank']} {result['body_name']}** — "
                f"{result['overall_score']:.1f}/100 · {result['tier']}"
            )
        with st.expander("View sub-scores"):
            score_table = pd.DataFrame(
                {
                    result["body_name"]: result["sub_scores"]
                    for result in results
                }
            ).rename(index={key: value["label"] for key, value in SUB_SCORE_METADATA.items()})
            st.dataframe(score_table, use_container_width=True)

    with analysis_tab:
        if st.button("Generate comparative analysis", type="primary"):
            with st.spinner("Analyzing mission trade-offs..."):
                st.session_state["planetary_analysis"] = generate_comparative_analysis(
                    bodies, mission_type, results
                )
                st.session_state["planetary_analysis_key"] = (
                    tuple(selected_bodies),
                    mission_type,
                )

        analysis_key = (tuple(selected_bodies), mission_type)
        analysis = st.session_state.get("planetary_analysis")
        if st.session_state.get("planetary_analysis_key") == analysis_key and analysis:
            st.caption(f"Analysis mode: {analysis['generation_mode']}")
            st.markdown(analysis["summary"])
            st.markdown("#### Key differences")
            for difference in analysis["key_differences"]:
                st.markdown(f"- {difference}")

            recommendation = analysis["mission_recommendation"]
            st.markdown("#### Mission recommendation")
            st.success(
                f"**{recommendation['recommended_body']}** — "
                f"{recommendation['rationale']}\n\n"
                f"{recommendation['mission_architecture_advice']}"
            )
            for name, assessment in analysis["body_assessments"].items():
                with st.expander(f"{name}: strengths and risks"):
                    st.markdown("**Strengths**")
                    for strength in assessment["strengths"]:
                        st.markdown(f"- {strength}")
                    st.markdown("**Risks**")
                    for risk in assessment["risks"]:
                        st.markdown(f"- {risk}")
            st.markdown("#### Uncertainties and open questions")
            for question in analysis["uncertainties_and_open_questions"]:
                st.markdown(f"- {question}")
        else:
            st.info("Generate an analysis to see a mission recommendation and trade-offs.")

    with details_tab:
        for body in bodies:
            physical = body["physical"]
            atmosphere = body["atmospheric"]
            environment = body["environmental"]
            with st.expander(body["name"]):
                st.write(body["type"])
                st.write(
                    f"Gravity: {physical['surface_gravity_m_s2']} m/s² · "
                    f"Temperature: {environment['mean_temp_c']} °C · "
                    f"Pressure: {atmosphere['pressure_display']}"
                )
                st.write(f"Radiation: {environment['radiation_tier']}")
                st.write(f"Water / ice: {environment['water_or_ice']}")

# ── Planetary Database Dictionary ──────────────────────────────────────────
PLANETS_DB = {
    "Mercury": {"dist": 0.387, "mass": 0.330, "radius": 2439, "gravity": 3.70, "temp": 167, "press": 1e-11, "mag": 0, "gas": 0, "solid": 1, "desc": "Extreme thermal fluctuations, high solar radiation, thin exosphere."},
    "Venus": {"dist": 0.723, "mass": 4.867, "radius": 6051, "gravity": 8.87, "temp": 464, "press": 92.0, "mag": 0, "gas": 0, "solid": 1, "desc": "Dense carbon dioxide atmosphere, supercritical pressures, corrosive sulfuric acid clouds."},
    "Earth": {"dist": 1.000, "mass": 5.972, "radius": 6371, "gravity": 9.81, "temp": 15, "press": 1.013, "mag": 1, "gas": 0, "solid": 1, "desc": "Habitable nitrogen-oxygen atmosphere, protective magnetosphere, liquid surface oceans."},
    "Moon": {"dist": 1.000, "mass": 0.073, "radius": 1737, "gravity": 1.62, "temp": -20, "press": 3e-15, "mag": 0, "gas": 0, "solid": 1, "desc": "Airless vacuum, high micrometeoroid risk, sharp abrasive lunar regolith."},
    "Mars": {"dist": 1.524, "mass": 0.642, "radius": 3389, "gravity": 3.71, "temp": -65, "press": 0.0069, "mag": 0, "gas": 0, "solid": 1, "desc": "Thin CO2 atmosphere, global dust storms, polar ice caps, legacy surface water channels."},
    "Jupiter": {"dist": 5.203, "mass": 1898.0, "radius": 69911, "gravity": 24.79, "temp": -110, "press": 100.0, "mag": 1, "gas": 1, "solid": 0, "desc": "Massive gas giant, extreme radiation belts, violent storm systems (Great Red Spot)."},
    "Europa": {"dist": 5.203, "mass": 0.048, "radius": 1560, "gravity": 1.31, "temp": -160, "press": 1e-12, "mag": 0, "gas": 0, "solid": 1, "desc": "Subsurface liquid ocean beneath an ice crust, intense Jovian magnetospheric radiation."},
    "Saturn": {"dist": 9.537, "mass": 568.3, "radius": 58232, "gravity": 10.44, "temp": -140, "press": 100.0, "mag": 1, "gas": 1, "solid": 0, "desc": "Ringed gas giant, low bulk density, extensive satellite system."},
    "Titan": {"dist": 9.580, "mass": 0.135, "radius": 2575, "gravity": 1.35, "temp": -179, "press": 1.46, "mag": 0, "gas": 0, "solid": 1, "desc": "Dense nitrogen-methane atmosphere, liquid hydrocarbon lakes, cryovolcanism."},
    "Uranus": {"dist": 19.19, "mass": 86.81, "radius": 25362, "gravity": 8.87, "temp": -195, "press": 100.0, "mag": 1, "gas": 1, "solid": 0, "desc": "Ice giant, extreme axial tilt (97.8°), freezing atmospheric temperatures."},
    "Neptune": {"dist": 30.07, "mass": 102.4, "radius": 24622, "gravity": 11.15, "temp": -200, "press": 100.0, "mag": 1, "gas": 1, "solid": 0, "desc": "Outer ice giant, supersonic atmospheric winds (up to 2,100 km/h)."},
    "Pluto": {"dist": 39.48, "mass": 0.013, "radius": 1188, "gravity": 0.62, "temp": -225, "press": 1e-5, "mag": 0, "gas": 0, "solid": 1, "desc": "Kuiper belt dwarf planet, nitrogen-methane ice surface, faint seasonal atmosphere."}
}

# ── App Header ────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero-header">
    <h1>🚀 VornisAI</h1>
    <p>AI-Powered Space Mission & Vehicle Design Assistant</p>
</div>
""",
    unsafe_allow_html=True,
)

# ── Sidebar Inputs ────────────────────────────────────────────────────────
with st.sidebar:
    app_section = st.radio(
        "VornisAI modules",
        ["Mission Design", "Planetary Comparison"],
        key="app_section",
    )

if app_section == "Planetary Comparison":
    render_planetary_comparison()
    st.stop()

# ── Sidebar Inputs ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🪐 Mission Configuration")
    
    target_planet = st.selectbox("Target Celestial Body", list(PLANETS_DB.keys()), index=4)
    planet_info = PLANETS_DB[target_planet]
    
    mission_type = st.selectbox("Mission Architecture", ["Orbiter", "Rover", "Lander", "Flyby"], index=0)
    
    st.markdown("---")
    st.markdown("### ⚙️ Spacecraft Specs")
    
    cruise_days = st.number_input("Cruise Duration (days)", min_value=1.0, max_value=10000.0, value=250.0, step=10.0)
    spacecraft_mass = st.number_input("Total Mass (kg)", min_value=10.0, max_value=50000.0, value=2200.0, step=100.0)
    science_payload = st.number_input("Science Payload (kg)", min_value=1.0, max_value=5000.0, value=120.0, step=10.0)
    hist_success = st.selectbox("Historical Precedent", [1, 0], format_func=lambda x: "Yes (Target Reached Before)" if x == 1 else "No (New Frontier)")
    
    st.markdown("---")
    st.markdown("### 🚀 Engine & Thruster Specs")
    isp = st.slider("Specific Impulse $I_{sp}$ (seconds)", min_value=200, max_value=4500, value=320, step=10, help="Chemical ~320s, Solar Electric ~3000s")

# Extract environmental parameters
dist_au = planet_info["dist"]
mass_1e24 = planet_info["mass"]
radius_km = planet_info["radius"]
gravity_m_s2 = planet_info["gravity"]
escape_velocity = math.sqrt(2 * (6.6743e-11) * (mass_1e24 * 1e24) / (radius_km * 1000)) / 1000
temp_c = planet_info["temp"]
press_bar = planet_info["press"]

# ==========================================================================
# DIVISION 1: MISSION FEASIBILITY AI SCORE
# ==========================================================================
st.markdown("<div class='section-title'>📊 1. Mission Feasibility AI Score</div>", unsafe_allow_html=True)

input_df = pd.DataFrame([{
    "target": target_planet,
    "mission_type": mission_type,
    "cruise_days": cruise_days,
    "spacecraft_mass_kg": spacecraft_mass,
    "science_payload_kg": science_payload,
    "historical_success": hist_success,
    "distance_AU": dist_au,
    "mass_1e24_kg": mass_1e24,
    "radius_km": radius_km,
    "gravity_m_s2": gravity_m_s2,
    "escape_velocity_km_s": escape_velocity,
    "temperature_C": temp_c,
    "pressure_bar": press_bar,
    "global_magnetic_field": planet_info["mag"],
    "gas_giant": planet_info["gas"],
    "solid_surface": planet_info["solid"],
    "is_orbiter": 1 if mission_type == "Orbiter" else 0,
    "is_rover": 1 if mission_type == "Rover" else 0,
    "is_lander": 1 if mission_type == "Lander" else 0,
    "is_flyby": 1 if mission_type == "Flyby" else 0,
}])

pred_val = model.predict(input_df)[0]
score = float(np.clip(pred_val, 0, 100))

if score >= 70:
    score_color = "#00e676"
    status_text = "🟢 HIGH FEASIBILITY"
    eval_desc = "Mission parameters align well with historical telemetry and technology readiness."
elif score >= 45:
    score_color = "#ffea00"
    status_text = "🟡 MODERATE FEASIBILITY"
    eval_desc = "Mission is viable, but requires robust risk mitigation for cruise duration, payload mass, or environmental conditions."
else:
    score_color = "#ff3d00"
    status_text = "🔴 LOW FEASIBILITY / HIGH RISK"
    eval_desc = "Extreme environmental conditions (surface pressure/heat/radiation) or extreme orbital distance require specialized engineering."

col_score_l, col_score_r = st.columns([2, 3])

with col_score_l:
    st.markdown(
        f"""
        <div class="score-card">
            <div class="score-label">Mission Feasibility Score</div>
            <div class="score-number" style="color: {score_color};">{score:.1f}</div>
            <div style="font-weight: 600; color: {score_color}; font-size: 1.1rem; margin-bottom: 0.3rem;">{status_text}</div>
            <div style="color: #94a3b8; font-size: 0.88rem;">{eval_desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_score_r:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### 🔍 AI Feasibility Score Breakdown & Driving Factors")
    
    reasons = []
    if press_bar > 10.0 and mission_type in ["Lander", "Rover"]:
        reasons.append(f"⚠️ **Atmospheric Overpressure:** Extreme surface pressure ({press_bar} bar) imposes structural crush hazard for surface landing.")
    elif press_bar > 0.001:
        reasons.append("💨 **Aero-braking Capability:** Atmosphere presence enables aerodynamic deceleration, reducing propellant needed for orbital entry.")

    if temp_c > 300.0:
        reasons.append(f"🔥 **Extreme Surface Thermal Load:** Surface temperature ({temp_c}°C) degrades standard avionics and solar cells without active cooling.")
    elif temp_c < -150.0:
        reasons.append(f"❄️ **Cryogenic Environment:** Deep space cold ({temp_c}°C) mandates Radioisotope Heater Units (RHUs) to prevent propellant freezing.")

    if dist_au > 4.0:
        reasons.append(f"🌌 **Deep Solar Distance:** At {dist_au:.2f} AU, solar flux drops by >93%, ruling out standard solar arrays and mandating MMRTG nuclear power.")
    else:
        reasons.append(f"☀️ **High Solar Flux:** Orbital radius ({dist_au:.2f} AU) allows lightweight GaAs solar arrays.")

    if target_planet in ["Mars", "Moon"]:
        reasons.append("✅ **Proven Flight Heritage:** Target has high historical success rates (TRL 8-9) with extensive telemetry.")

    if mission_type == "Flyby":
        reasons.append("🚀 **Simplified Trajectory:** Flyby avoids entry, descent, landing (EDL), and orbit insertion burn hazards.")

    for r in reasons:
        st.markdown(f"- {r}")
        
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================================
# DIVISION 2: AI VEHICLE DESIGN RECOMMENDATIONS
# ==========================================================================
st.markdown("<div class='section-title'>🧠 2. AI Vehicle Design Recommendations</div>", unsafe_allow_html=True)

# Structure material
if planet_info["gravity"] > 15:
    struct = "High-strength Titanium Alloy (Ti-6Al-4V) with reinforced internal ribs."
    struct_why = "Necessary to withstand massive gravitational stress during entry/close proximity."
elif planet_info["temp"] > 300:
    struct = "Titanium-Matrix Composite with nickel-chromium superalloy outer frame."
    struct_why = "Resists thermal expansion and structural softening under extreme temperatures."
else:
    struct = "Aluminum-Lithium Alloy with Carbon-Fiber Reinforced Polymer (CFRP) honeycomb panels."
    struct_why = "Maximizes structural rigidity while minimizing dry mass fraction."

# Propulsion System
if dist_au > 4.0:
    prop = "Dual-mode Radioisotope Electric Propulsion (REP) / Monopropellant hydrazine for RCS."
    prop_why = "Solar flux is insufficient beyond 4 AU; electric propulsion delivers high specific impulse ($I_{sp}$) for deep space cruise."
elif mission_type in ["Lander", "Rover"] and planet_info["press"] > 0.001:
    prop = "Hypergolic Bipropellant (MMH/NTO) + Supersonic Retro-propulsion / Aero-braking heat shield."
    prop_why = "Provides instant restart capability and high thrust-to-weight ratio during entry, descent, and landing (EDL)."
else:
    prop = "Solar Electric Propulsion (Hall-Effect Thrusters) paired with cold gas reaction control."
    prop_why = "Highly efficient delta-v maneuvers within inner solar system."

# Power Module
if dist_au > 3.5:
    power = "Next-Gen Multi-Mission Radioisotope Thermoelectric Generator (MMRTG)."
    power_why = "Solar energy density is too low at this orbital radius to power science instruments."
else:
    power = "Ultra-flex Gallium Arsenide (GaAs) Triple-Junction Solar Arrays with Lithium-ion battery storage."
    power_why = "High efficiency (>30%) conversion rate under sunlight."

# Thermal & Environmental Protection
if planet_info["temp"] > 400 or planet_info["press"] > 10:
    thermal = "Multi-layer ablative Phenolic-Impregnated Carbon Ablator (PICA-X) thermal shield + Active internal refrigeration."
    thermal_why = "Critical for atmospheric survival under extreme heating and pressure."
elif planet_info["temp"] < -100:
    thermal = "Multi-Layer Insulation (MLI) blankets with internal Radioisotope Heater Units (RHUs)."
    thermal_why = "Prevents propellant freezing and electronics failure in cryogenic vacuum."
else:
    thermal = "Passive Multi-Layer Insulation (MLI) + Louvers and heat-pipes."
    thermal_why = "Standard thermal control balance."

col_r1, col_r2 = st.columns(2)
with col_r1:
    st.markdown(
        f"""
        <div class="rec-box">
            <h4>🛠️ Structural Material & Chassis</h4>
            <p><strong>Recommendation:</strong> {struct}</p>
            <p style="margin-top:0.3rem; font-size:0.85rem; color:#94a3b8;"><em>Scientific Rationale:</em> {struct_why}</p>
        </div>
        <div class="rec-box">
            <h4>🚀 Propulsion & Reaction Control</h4>
            <p><strong>Recommendation:</strong> {prop}</p>
            <p style="margin-top:0.3rem; font-size:0.85rem; color:#94a3b8;"><em>Scientific Rationale:</em> {prop_why}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_r2:
    st.markdown(
        f"""
        <div class="rec-box">
            <h4>⚡ Power Generation System</h4>
            <p><strong>Recommendation:</strong> {power}</p>
            <p style="margin-top:0.3rem; font-size:0.85rem; color:#94a3b8;"><em>Scientific Rationale:</em> {power_why}</p>
        </div>
        <div class="rec-box">
            <h4>🔥 Thermal Management & Shielding</h4>
            <p><strong>Recommendation:</strong> {thermal}</p>
            <p style="margin-top:0.3rem; font-size:0.85rem; color:#94a3b8;"><em>Scientific Rationale:</em> {thermal_why}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==========================================================================
# DIVISION 3: PLANETARY INTELLIGENCE OVERVIEW
# ==========================================================================
st.markdown("<div class='section-title'>🪐 3. Planetary Intelligence — " + target_planet + "</div>", unsafe_allow_html=True)

st.info(f"**Target Overview:** {planet_info['desc']}")

p1, p2, p3, p4, p5 = st.columns(5)
p1.metric("Surface Gravity", f"{gravity_m_s2:.2f} m/s²")
p2.metric("Escape Velocity", f"{escape_velocity:.2f} km/s")
p3.metric("Mean Temperature", f"{temp_c} °C")
p4.metric("Surface Pressure", f"{press_bar} bar")
p5.metric("Magnetosphere", "Present" if planet_info["mag"] == 1 else "None/Weak")

# ==========================================================================
# DIVISION 4: LAUNCH & TRANSFER PHYSICS DYNAMICS
# ==========================================================================
st.markdown("<div class='section-title'>⚡ 4. Launch & Orbital Physics Dynamics</div>", unsafe_allow_html=True)

r1 = 1.0  # Earth AU
r2 = dist_au  # Target AU
a_transfer = (r1 + r2) / 2.0

# Velocity calculations
v_earth = 29.78  # km/s
v_transfer_earth = v_earth * math.sqrt(2 - (r1 / a_transfer))
delta_v_launch = abs(v_transfer_earth - v_earth)

# Hohmann transfer duration
time_years = 0.5 * (a_transfer ** 1.5)
transfer_days = time_years * 365.25

# Tsiolkovsky Rocket Equation
g0 = 9.80665
required_dv = (delta_v_launch + escape_velocity) * 1000  # m/s
mass_ratio = math.exp(required_dv / (isp * g0))
propellant_mass = spacecraft_mass * (1 - (1 / mass_ratio))

col_dyn1, col_dyn2 = st.columns(2)
with col_dyn1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### 📐 Hohmann Transfer Dynamics")
    st.metric(r"Estimated Transfer Delta-V ($\Delta v$)", f"{delta_v_launch:.2f} km/s")
    st.metric("Theoretical Orbital Transfer Time", f"{transfer_days:.0f} days (~{time_years:.2f} years)")
    st.markdown("</div>", unsafe_allow_html=True)

with col_dyn2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### 🚀 Tsiolkovsky Rocket Equation")
    st.metric("Required Mass Ratio ($m_0 / m_f$)", f"{mass_ratio:.2f}")
    st.metric("Estimated Propellant Mass Fraction", f"{(propellant_mass / spacecraft_mass) * 100:.1f} % ({propellant_mass:.0f} kg)")
    st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1rem 0;">
    VornisAI System Architecture v2.0 • Merging Aerospace Physics & Machine Learning • Developed by @amaljb5
</div>
""",
    unsafe_allow_html=True,
)

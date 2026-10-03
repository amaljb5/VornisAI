import json
import math
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request, jsonify

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "feasibility_model.pkl"
META_PATH = BASE_DIR / "model" / "model_metadata.json"

app = Flask(__name__)

# Global model cache
model = None
meta = None

def get_model():
    global model, meta
    if model is None:
        model = joblib.load(MODEL_PATH)
    if meta is None:
        with open(META_PATH, "r") as f:
            meta = json.load(f)
    return model, meta

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

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VornisAI — Spacecraft & Mission Feasibility Assistant</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #070913 0%, #0d1527 50%, #151d36 100%);
            color: #e2e8f0;
            min-height: 100vh;
            padding: 20px;
        }
        .header { text-align: center; padding: 20px 0; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 25px; }
        .header h1 {
            font-family: 'Outfit', sans-serif;
            background: linear-gradient(135deg, #00d2ff, #7a5cff, #ff6ec7);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-size: 2.8rem; font-weight: 700;
        }
        .header p { color: #94a3b8; font-size: 1.1rem; font-weight: 300; margin-top: 5px; }
        .layout { display: grid; grid-template-columns: 320px 1fr; gap: 25px; max-width: 1300px; margin: 0 auto; }
        @media(max-width: 900px) { .layout { grid-template-columns: 1fr; } }
        .panel {
            background: rgba(255, 255, 255, 0.035);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px; padding: 22px; backdrop-filter: blur(16px);
        }
        .form-group { margin-bottom: 16px; }
        .form-group label { display: block; font-size: 0.85rem; font-weight: 600; color: #94a3b8; margin-bottom: 6px; text-transform: uppercase; }
        select, input {
            width: 100%; padding: 10px 14px; background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(255,255,255,0.12); border-radius: 10px; color: #fff; font-size: 0.95rem; outline: none;
        }
        select:focus, input:focus { border-color: #38bdf8; }
        button {
            width: 100%; padding: 12px; background: linear-gradient(135deg, #7a5cff, #00d2ff);
            border: none; border-radius: 10px; color: #fff; font-weight: 600; font-size: 1rem; cursor: pointer; transition: 0.3s;
        }
        button:hover { opacity: 0.9; transform: translateY(-1px); }
        .section-title {
            font-family: 'Outfit', sans-serif; color: #38bdf8; font-size: 1.3rem; font-weight: 600;
            margin: 20px 0 12px 0; border-bottom: 1px solid rgba(56, 189, 248, 0.2); padding-bottom: 6px;
        }
        .score-card {
            text-align: center; padding: 25px; border-radius: 16px; background: rgba(15, 23, 42, 0.75);
            border: 1px solid rgba(122, 92, 255, 0.3); margin-bottom: 20px;
        }
        .score-num { font-family: 'Outfit', sans-serif; font-size: 4.5rem; font-weight: 700; line-height: 1; margin: 10px 0; }
        .rec-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        @media(max-width: 700px) { .rec-grid { grid-template-columns: 1fr; } }
        .rec-box {
            background: rgba(122, 92, 255, 0.07); border-left: 4px solid #7a5cff; padding: 14px 16px; border-radius: 0 12px 12px 0;
        }
        .rec-box h4 { color: #38bdf8; margin-bottom: 4px; font-size: 1rem; }
        .rec-box p { color: #cbd5e1; font-size: 0.9rem; margin: 0; }
        .metrics-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-top: 10px; }
        .metric-card { background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.06); }
        .metric-card span { display: block; font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; }
        .metric-card strong { font-family: 'Outfit', sans-serif; font-size: 1.15rem; color: #38bdf8; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 VornisAI</h1>
        <p>AI-Powered Space Mission & Spacecraft Design Assistant</p>
    </div>
    
    <div class="layout">
        <div class="panel">
            <h3 style="color: #38bdf8; font-family: 'Outfit'; margin-bottom: 15px;">🪐 Mission Config</h3>
            <form id="configForm">
                <div class="form-group">
                    <label>Target Body</label>
                    <select id="target">
                        {% for planet in planets %}
                        <option value="{{ planet }}" {% if planet == 'Mars' %}selected{% endif %}>{{ planet }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="form-group">
                    <label>Mission Architecture</label>
                    <select id="mission_type">
                        <option value="Orbiter" selected>Orbiter</option>
                        <option value="Rover">Rover</option>
                        <option value="Lander">Lander</option>
                        <option value="Flyby">Flyby</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Total Mass (kg)</label>
                    <input type="number" id="spacecraft_mass" value="2200" step="100">
                </div>
                <div class="form-group">
                    <label>Science Payload (kg)</label>
                    <input type="number" id="science_payload" value="120" step="10">
                </div>
                <div class="form-group">
                    <label>Cruise Duration (days)</label>
                    <input type="number" id="cruise_days" value="250" step="10">
                </div>
                <div class="form-group">
                    <label>Thruster Specific Impulse (s)</label>
                    <input type="number" id="isp" value="320" step="10">
                </div>
                <button type="button" onclick="runInference()">Recalculate Mission</button>
            </form>
        </div>

        <div>
            <div id="results">Loading dynamic recommendations...</div>
        </div>
    </div>

    <script>
        const planetsDB = {{ planets_json|safe }};

        function runInference() {
            const target = document.getElementById('target').value;
            const mission_type = document.getElementById('mission_type').value;
            const spacecraft_mass = parseFloat(document.getElementById('spacecraft_mass').value) || 2200;
            const science_payload = parseFloat(document.getElementById('science_payload').value) || 120;
            const cruise_days = parseFloat(document.getElementById('cruise_days').value) || 250;
            const isp = parseFloat(document.getElementById('isp').value) || 320;
            
            const pInfo = planetsDB[target];
            const payload_ratio = ((science_payload / spacecraft_mass) * 100).toFixed(1);

            fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    target, mission_type, spacecraft_mass_kg: spacecraft_mass,
                    science_payload_kg: science_payload, cruise_days,
                    historical_success: 1, distance_AU: pInfo.dist, mass_1e24_kg: pInfo.mass,
                    radius_km: pInfo.radius, gravity_m_s2: pInfo.gravity,
                    escape_velocity_km_s: Math.sqrt(2 * 6.6743e-11 * (pInfo.mass * 1e24) / (pInfo.radius * 1000)) / 1000,
                    temperature_C: pInfo.temp, pressure_bar: pInfo.press,
                    global_magnetic_field: pInfo.mag, gas_giant: pInfo.gas, solid_surface: pInfo.solid
                })
            })
            .then(res => res.json())
            .then(data => {
                renderDashboard(data.score, data.rating, data.color, target, mission_type, pInfo, spacecraft_mass, science_payload, payload_ratio, cruise_days, isp);
            });
        }

        function renderDashboard(score, rating, color, target, mission_type, pInfo, mass, payload, payload_ratio, days, isp) {
            const vesc = (Math.sqrt(2 * 6.6743e-11 * (pInfo.mass * 1e24) / (pInfo.radius * 1000)) / 1000).toFixed(2);
            
            // Hohmann calculations
            const r1 = 1.0, r2 = pInfo.dist;
            const a_tr = (r1 + r2) / 2.0;
            const v_earth = 29.78;
            const v_tr = v_earth * Math.sqrt(2 - (r1 / a_tr));
            const delta_v = Math.abs(v_tr - v_earth).toFixed(2);
            const time_yrs = (0.5 * Math.pow(a_tr, 1.5)).toFixed(2);
            const tr_days = (time_yrs * 365.25).toFixed(0);
            
            // Rocket equation
            const req_dv = (parseFloat(delta_v) + parseFloat(vesc)) * 1000;
            const mass_ratio = Math.exp(req_dv / (isp * 9.80665)).toFixed(2);
            const prop_mass = (mass * (1 - (1 / mass_ratio))).toFixed(0);

            document.getElementById('results').innerHTML = `
                <div class="section-title">📊 1. Mission Feasibility AI Score</div>
                <div class="score-card">
                    <div style="color:#94a3b8; font-size:0.85rem; text-transform:uppercase; letter-spacing:2px;">Mission Feasibility Score</div>
                    <div class="score-num" style="color:${color}">${score}</div>
                    <div style="font-weight:600; color:${color}">${rating}</div>
                </div>

                <div class="section-title">🧠 2. AI Vehicle Design Recommendations</div>
                <div class="rec-grid">
                    <div class="rec-box">
                        <h4>🛠️ Structural Material</h4>
                        <p><strong>Rec:</strong> ${pInfo.gravity > 15 ? 'Titanium Alloy (Ti-6Al-4V)' : 'Aluminum-Lithium Alloy with CFRP Honeycomb'}</p>
                    </div>
                    <div class="rec-box">
                        <h4>🚀 Propulsion System</h4>
                        <p><strong>Rec:</strong> ${pInfo.dist > 4.0 ? 'Radioisotope Electric Propulsion (REP)' : 'Solar Electric Propulsion (Hall Thrusters)'}</p>
                    </div>
                    <div class="rec-box">
                        <h4>⚡ Power System</h4>
                        <p><strong>Rec:</strong> ${pInfo.dist > 3.5 ? 'Multi-Mission RTG (MMRTG)' : 'GaAs Triple-Junction Solar Arrays'}</p>
                    </div>
                    <div class="rec-box">
                        <h4>🔥 Thermal Management</h4>
                        <p><strong>Rec:</strong> ${pInfo.temp > 300 ? 'Ablative PICA-X Thermal Shielding' : 'Multi-Layer Insulation (MLI) & RHUs'}</p>
                    </div>
                </div>

                <div class="section-title">🪐 3. Planetary Intelligence — ${target}</div>
                <div class="metrics-row">
                    <div class="metric-card"><span>Gravity</span><strong>${pInfo.gravity} m/s²</strong></div>
                    <div class="metric-card"><span>Escape Velocity</span><strong>${vesc} km/s</strong></div>
                    <div class="metric-card"><span>Temp</span><strong>${pInfo.temp} °C</strong></div>
                    <div class="metric-card"><span>Pressure</span><strong>${pInfo.press} bar</strong></div>
                </div>

                <div class="section-title">⚡ 4. Launch & Physics Dynamics</div>
                <div class="metrics-row">
                    <div class="metric-card"><span>Transfer Δv</span><strong>${delta_v} km/s</strong></div>
                    <div class="metric-card"><span>Transfer Time</span><strong>${tr_days} days</strong></div>
                    <div class="metric-card"><span>Mass Ratio</span><strong>${mass_ratio}</strong></div>
                    <div class="metric-card"><span>Propellant Required</span><strong>${prop_mass} kg</strong></div>
                </div>
            `;
        }

        window.onload = runInference;
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(
        HTML_TEMPLATE,
        planets=list(PLANETS_DB.keys()),
        planets_json=json.dumps(PLANETS_DB)
    )

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        model_obj, _ = get_model()
        
        input_dict = {
            "target": data.get("target"),
            "mission_type": data.get("mission_type"),
            "cruise_days": float(data.get("cruise_days", 0)),
            "spacecraft_mass_kg": float(data.get("spacecraft_mass_kg", 0)),
            "science_payload_kg": float(data.get("science_payload_kg", 0)),
            "historical_success": int(data.get("historical_success", 1)),
            "distance_AU": float(data.get("distance_AU", 1.0)),
            "mass_1e24_kg": float(data.get("mass_1e24_kg", 0.1)),
            "radius_km": float(data.get("radius_km", 1000)),
            "gravity_m_s2": float(data.get("gravity_m_s2", 3.7)),
            "escape_velocity_km_s": float(data.get("escape_velocity_km_s", 5.0)),
            "temperature_C": float(data.get("temperature_C", 20)),
            "pressure_bar": float(data.get("pressure_bar", 1.0)),
            "global_magnetic_field": int(data.get("global_magnetic_field", 0)),
            "gas_giant": int(data.get("gas_giant", 0)),
            "solid_surface": int(data.get("solid_surface", 1)),
            "is_orbiter": 1 if data.get("mission_type") == "Orbiter" else 0,
            "is_rover": 1 if data.get("mission_type") == "Rover" else 0,
            "is_lander": 1 if data.get("mission_type") == "Lander" else 0,
            "is_flyby": 1 if data.get("mission_type") == "Flyby" else 0,
        }
        
        input_df = pd.DataFrame([input_dict])
        raw_score = model_obj.predict(input_df)[0]
        score = float(np.clip(raw_score, 0, 100))
        
        if score >= 70:
            rating = "🟢 HIGH FEASIBILITY"
            color = "#00e676"
        elif score >= 45:
            rating = "🟡 MODERATE FEASIBILITY"
            color = "#ffea00"
        else:
            rating = "🔴 LOW FEASIBILITY / HIGH RISK"
            color = "#ff3d00"
            
        return jsonify({
            "status": "success",
            "score": round(score, 1),
            "rating": rating,
            "color": color
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Vercel entry point export
handler = app

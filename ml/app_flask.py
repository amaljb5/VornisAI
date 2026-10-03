"""
VornisAI — Mission Feasibility Web Application (Flask)
======================================================
Lightweight, instant-start web app for mission feasibility predictions.

Usage:
    python ml/app_flask.py
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "feasibility_model.pkl"
META_PATH = BASE_DIR / "model" / "model_metadata.json"

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

# Load model and metadata
model = joblib.load(MODEL_PATH)
with open(META_PATH, "r") as f:
    meta = json.load(f)

@app.route("/")
def index():
    return render_template("index.html", meta=meta)

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Prepare DataFrame with expected columns
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
        raw_score = model.predict(input_df)[0]
        score = float(np.clip(raw_score, 0, 100))
        
        # Rating text
        if score >= 80:
            rating = "🟢 Excellent — Mission Highly Feasible"
            color = "#00e676"
        elif score >= 65:
            rating = "🟡 Good — Feasible with Standard Risk"
            color = "#ffea00"
        elif score >= 50:
            rating = "🟠 Moderate — Requires Careful Planning"
            color = "#ff9100"
        elif score >= 35:
            rating = "🔴 Low — Significant Challenges Expected"
            color = "#ff3d00"
        else:
            rating = "⛔ Very Low — Mission Redesign Recommended"
            color = "#ff1744"
            
        return jsonify({
            "status": "success",
            "score": round(score, 1),
            "rating": rating,
            "color": color
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    print("🚀 Starting VornisAI Flask App on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=True)

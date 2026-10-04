import json
import math
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Global model cache
model = None
meta = None

def get_model():
    global model, meta
    if model is None or meta is None:
        base = Path(__file__).resolve().parent
        model_path = base / "model" / "feasibility_model.pkl"
        meta_path = base / "model" / "model_metadata.json"
        
        if not model_path.exists():
            model_path = Path.cwd() / "api" / "model" / "feasibility_model.pkl"
            meta_path = Path.cwd() / "api" / "model" / "model_metadata.json"
            
        if not model_path.exists():
            model_path = Path.cwd() / "ml" / "model" / "feasibility_model.pkl"
            meta_path = Path.cwd() / "ml" / "model" / "model_metadata.json"

        model = joblib.load(model_path)
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
    return model, meta

# Planetary database — includes all targets from the CSV training dataset + Uranus/Neptune
PLANETS_DB = {
    "Mercury": {"dist": 0.387, "mass": 0.330, "radius": 2439, "gravity": 3.70, "temp": 167, "press": 1e-11, "mag": 0, "gas": 0, "solid": 1, "desc": "Extreme thermal fluctuations, high solar radiation, thin exosphere."},
    "Venus": {"dist": 0.723, "mass": 4.867, "radius": 6051, "gravity": 8.87, "temp": 464, "press": 92.0, "mag": 0, "gas": 0, "solid": 1, "desc": "Dense carbon dioxide atmosphere, supercritical pressures, corrosive sulfuric acid clouds."},
    "Earth": {"dist": 1.000, "mass": 5.972, "radius": 6371, "gravity": 9.81, "temp": 15, "press": 1.013, "mag": 1, "gas": 0, "solid": 1, "desc": "Habitable nitrogen-oxygen atmosphere, protective magnetosphere, liquid surface oceans."},
    "Moon": {"dist": 1.000, "mass": 0.073, "radius": 1737, "gravity": 1.62, "temp": -20, "press": 3e-15, "mag": 0, "gas": 0, "solid": 1, "desc": "Airless vacuum, high micrometeoroid risk, sharp abrasive lunar regolith."},
    "Mars": {"dist": 1.524, "mass": 0.642, "radius": 3389, "gravity": 3.71, "temp": -65, "press": 0.0069, "mag": 0, "gas": 0, "solid": 1, "desc": "Thin CO2 atmosphere, global dust storms, polar ice caps, legacy surface water channels."},
    "Phobos": {"dist": 1.524, "mass": 1.07e-5, "radius": 11, "gravity": 0.0057, "temp": -40, "press": 0.0, "mag": 0, "gas": 0, "solid": 1, "desc": "Irregular Martian moon, ultra-low gravity, grooved surface, likely captured asteroid."},
    "Ceres": {"dist": 2.768, "mass": 0.000938, "radius": 470, "gravity": 0.27, "temp": -106, "press": 0.0, "mag": 0, "gas": 0, "solid": 1, "desc": "Largest asteroid belt object, dwarf planet, water-ice subsurface, bright salt deposits."},
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
        .reason-card {
            background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px; padding: 16px; margin-top: 15px; text-align: left;
        }
        .reason-card h4 { color: #38bdf8; font-size: 0.95rem; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px; }
        .reason-card ul { list-style: none; padding: 0; }
        .reason-card li { margin-bottom: 8px; font-size: 0.92rem; line-height: 1.45; color: #cbd5e1; }
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

        /* autocomplete */
        .ac-wrap { position: relative; }
        .ac-input {
            width: 100%; padding: 10px 14px 10px 38px;
            background: rgba(15,23,42,0.8);
            border: 1px solid rgba(255,255,255,0.12); border-radius: 10px;
            color: #fff; font-size: 0.95rem; outline: none;
            font-family: inherit; transition: border-color .2s, box-shadow .2s;
        }
        .ac-input::placeholder { color: #4a5568; }
        .ac-input:focus { border-color: #38bdf8; box-shadow: 0 0 0 2px rgba(56,189,248,.15); }
        .ac-icon {
            position: absolute; left: 12px; top: 50%; transform: translateY(-50%);
            font-size: 1rem; pointer-events: none; opacity: .55; line-height: 1;
        }
        .ac-dropdown {
            position: absolute; top: calc(100% + 6px); left: 0; right: 0; z-index: 999;
            background: rgba(8,16,34,.97); border: 1px solid rgba(56,189,248,.22);
            border-radius: 12px; overflow: hidden; backdrop-filter: blur(24px);
            box-shadow: 0 16px 48px rgba(0,0,0,.65); display: none;
        }
        .ac-dropdown.open { display: block; animation: acIn .14s ease; }
        @keyframes acIn { from { opacity:0; transform:translateY(-5px); } to { opacity:1; transform:translateY(0); } }
        .ac-item {
            display: flex; align-items: center; gap: 9px;
            padding: 9px 14px; cursor: pointer; font-size: .91rem; color: #94a3b8;
            transition: background .12s, color .12s;
            border-bottom: 1px solid rgba(255,255,255,.04);
        }
        .ac-item:last-child { border-bottom: none; }
        .ac-item:hover, .ac-item.highlighted { background: rgba(56,189,248,.10); color: #e2e8f0; }
        .ac-badge {
            margin-left: auto; flex-shrink: 0; font-size: .70rem; color: #7a5cff;
            background: rgba(122,92,255,.14); padding: 2px 8px; border-radius: 99px; white-space: nowrap;
        }
        .ac-no-match { padding: 10px 14px; color: #475569; font-size: .86rem; font-style: italic; }
        mark { background: rgba(56,189,248,.22); color: #fff; border-radius: 2px; padding: 0 1px; }
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

                    <div class="ac-wrap">

                        <span class="ac-icon">&#127760;</span>

                        <input class="ac-input" id="target" type="text" placeholder="e.g. Mars, Europa, Titan..." value="Mars" spellcheck="false">

                        <div class="ac-dropdown" id="planetDrop"></div>

                    </div>

                </div>
                <div class="form-group">

                    <label>Mission Architecture</label>

                    <div class="ac-wrap">

                        <span class="ac-icon">&#128752;</span>

                        <input class="ac-input" id="mission_type" type="text" placeholder="e.g. Orbiter, Rover, Lander..." value="Orbiter" spellcheck="false">

                        <div class="ac-dropdown" id="missionDrop"></div>

                    </div>

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
            <div id="results">Calculating physics & AI model parameters...</div>
        </div>
    </div>

    <script>
        const planetsDB = {{ planets_json|safe }};

        /* ── Autocomplete factory ── */
        function makeAutocomplete({ inputId, dropId, items, iconFn, badgeFn }) {
            const inp  = document.getElementById(inputId);
            const drop = document.getElementById(dropId);
            let hiIdx  = -1;

            function score(item, q) {
                const s = item.toLowerCase(), ql = q.toLowerCase();
                if (s === ql) return 3;
                if (s.startsWith(ql)) return 2;
                if (s.includes(ql)) return 1;
                return 0;
            }
            function hl(text, q) {
                if (!q) return text;
                const re = new RegExp('(' + q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'gi');
                return text.replace(re, '<mark>$1</mark>');
            }
            function selectVal(val) { inp.value = val; drop.classList.remove('open'); }
            function render(q) {
                const filtered = q
                    ? items.map(i => ({ i, s: score(i, q) })).filter(x => x.s > 0).sort((a,b) => b.s-a.s).map(x => x.i)
                    : items;
                if (!filtered.length) {
                    drop.innerHTML = '<div class="ac-no-match">No match &mdash; custom value will be used</div>';
                } else {
                    drop.innerHTML = filtered.map(item =>
                        '<div class="ac-item" data-val="' + item + '">' +
                        '<span>' + (iconFn ? iconFn(item) : '') + '</span>' +
                        '<span>' + hl(item, q) + '</span>' +
                        (badgeFn ? '<span class="ac-badge">' + badgeFn(item) + '</span>' : '') +
                        '</div>'
                    ).join('');
                    drop.querySelectorAll('.ac-item').forEach(el => {
                        el.addEventListener('mousedown', e => e.preventDefault());
                        el.addEventListener('click', () => selectVal(el.dataset.val));
                    });
                }
                hiIdx = -1; drop.classList.add('open');
            }
            inp.addEventListener('focus', () => render(inp.value));
            inp.addEventListener('input', () => render(inp.value));
            inp.addEventListener('blur',  () => setTimeout(() => drop.classList.remove('open'), 160));
            inp.addEventListener('keydown', e => {
                const rows = Array.from(drop.querySelectorAll('.ac-item'));
                if (!rows.length) return;
                if (e.key === 'ArrowDown') { e.preventDefault(); hiIdx = Math.min(hiIdx+1, rows.length-1); }
                else if (e.key === 'ArrowUp') { e.preventDefault(); hiIdx = Math.max(hiIdx-1, 0); }
                else if (e.key === 'Enter' && hiIdx >= 0) { e.preventDefault(); selectVal(rows[hiIdx].dataset.val); return; }
                else if (e.key === 'Escape') { drop.classList.remove('open'); return; }
                rows.forEach((r, i) => r.classList.toggle('highlighted', i === hiIdx));
            });
        }

        const planetNames = Object.keys(planetsDB);
        const PLANET_ICONS = { Mercury:'&#9791;', Venus:'&#9792;', Earth:'&#127757;', Moon:'&#127765;', Mars:'&#128308;', Phobos:'&#9673;', Ceres:'&#9672;', Jupiter:'&#129504;', Europa:'&#128309;', Saturn:'&#128344;', Titan:'&#128993;', Uranus:'&#128309;', Neptune:'&#128309;', Pluto:'&#9899;' };
        makeAutocomplete({ inputId:'target', dropId:'planetDrop', items:planetNames, iconFn: p => PLANET_ICONS[p] || '&#127760;', badgeFn: p => planetsDB[p].dist + ' AU' });

        const MISSION_TYPES = ['Orbiter','Rover','Lander','Flyby','Sample Return','Atmospheric Probe','Impactor','Lander + Penetrators','Lander/Rover','Orbiter/Flyby','Orbiter/Lander','Orbiter/Probe','Mixed','Space Telescope','Crewed'];
        const MISSION_ICONS = { Orbiter:'&#128752;', Rover:'&#128663;', Lander:'&#128748;', Flyby:'&#128168;', 'Sample Return':'&#129532;', 'Atmospheric Probe':'&#127744;', Impactor:'&#128165;', 'Lander + Penetrators':'&#128204;', 'Lander/Rover':'&#128663;', 'Orbiter/Flyby':'&#128752;', 'Orbiter/Lander':'&#128752;', 'Orbiter/Probe':'&#128752;', Mixed:'&#128256;', 'Space Telescope':'&#128301;', Crewed:'&#128104;&#8205;&#128640;' };
        const MISSION_DESC  = { Orbiter:'Remote sensing', Rover:'Surface mobility', Lander:'Fixed surface', Flyby:'Gravity assist', 'Sample Return':'Return material to Earth', 'Atmospheric Probe':'Atmosphere descent', Impactor:'Kinetic impact study', 'Lander + Penetrators':'Surface+subsurface', 'Lander/Rover':'Surface+mobility', 'Orbiter/Flyby':'Orbit+gravity assist', 'Orbiter/Lander':'Orbit+surface', 'Orbiter/Probe':'Orbit+descent probe', Mixed:'Multi-architecture', 'Space Telescope':'Deep space observation', Crewed:'Human spaceflight' };
        makeAutocomplete({ inputId:'mission_type', dropId:'missionDrop', items:MISSION_TYPES, iconFn: m => MISSION_ICONS[m] || '&#128752;', badgeFn: m => MISSION_DESC[m] || '' });

        function normalizeMissionType(raw) {
            if (!raw) return 'orbiter';
            const r = raw.trim().toLowerCase();
            const MAP = {
                'orbiter': 'orbiter', 'rover': 'rover', 'lander': 'lander', 'flyby': 'flyby',
                'sample return': 'sample_return', 'atmospheric probe': 'atmospheric_probe',
                'impactor': 'impactor', 'lander + penetrators': 'lander + penetrators',
                'lander/rover': 'lander_rover', 'orbiter/flyby': 'orbiter_flyby',
                'orbiter/lander': 'orbiter_lander', 'orbiter/probe': 'orbiter_probe',
                'mixed': 'mixed', 'space telescope': 'orbiter', 'crewed': 'lander'
            };
            if (MAP[r]) return MAP[r];
            if (r.startsWith('orb')) return 'orbiter';
            if (r.startsWith('rov')) return 'rover';
            if (r.startsWith('lan')) return 'lander';
            if (r.startsWith('fly') || r === 'fl') return 'flyby';
            if (r.startsWith('sam')) return 'sample_return';
            if (r.startsWith('atm')) return 'atmospheric_probe';
            if (r.startsWith('imp')) return 'impactor';
            if (r.startsWith('mix')) return 'mixed';
            return raw.trim().toLowerCase().replace(/\//g, '_').replace(/ /g, '_');
        }


        function generateReasons(target, mission_type, pInfo, mass, payload_ratio, days, score) {
            let reasons = [];
            
            if (pInfo.press > 10.0 && (mission_type === 'Lander' || mission_type === 'Rover')) {
                reasons.push("⚠️ <strong>Atmospheric Overpressure:</strong> Extreme surface pressure (" + pInfo.press + " bar) imposes structural crush hazard for surface landing.");
            } else if (pInfo.press > 0.001) {
                reasons.push("💨 <strong>Aero-braking Capability:</strong> Atmosphere presence enables aerodynamic deceleration, reducing propellant needed for orbital entry.");
            }

            if (pInfo.temp > 300.0) {
                reasons.push("🔥 <strong>Extreme Surface Thermal Load:</strong> Surface temperature (" + pInfo.temp + "°C) degrades standard avionics and solar cells without active cooling.");
            } else if (pInfo.temp < -150.0) {
                reasons.push("❄️ <strong>Cryogenic Environment:</strong> Deep space cold (" + pInfo.temp + "°C) mandates Radioisotope Heater Units (RHUs) to prevent propellant freezing.");
            }

            if (pInfo.dist > 4.0) {
                reasons.push("🌌 <strong>Deep Solar Distance:</strong> At " + pInfo.dist + " AU, solar flux drops by >93%, ruling out standard solar arrays and mandating MMRTG nuclear power.");
            } else {
                reasons.push("☀️ <strong>High Solar Flux:</strong> Orbital radius (" + pInfo.dist + " AU) allows lightweight GaAs solar arrays.");
            }

            if (target === 'Mars' || target === 'Moon') {
                reasons.push("✅ <strong>Proven Flight Heritage:</strong> Target has high historical success rates (TRL 8-9) with extensive telemetry.");
            }

            if (mission_type === 'Flyby') {
                reasons.push("🚀 <strong>Simplified Trajectory:</strong> Flyby avoids entry, descent, landing (EDL), and orbit insertion burn hazards.");
            }

            return reasons;
        }

        function calculateFallbackScore(target, mission_type, dist, temp, press, gravity) {
            let score = 55.0;
            if (target === 'Mars') score += 25.0;
            else if (target === 'Moon') score += 20.0;
            else if (target === 'Venus') score += (mission_type === 'Orbiter' ? 5.0 : -20.0);
            else if (dist > 5.0) score -= (dist * 0.8);
            
            score = Math.min(100.0, Math.max(0.0, score));
            let rating = "MODERATE FEASIBILITY", color = "#ffea00";
            if (score >= 70) { rating = "HIGH FEASIBILITY"; color = "#00e676"; }
            else if (score < 45) { rating = "LOW FEASIBILITY / HIGH RISK"; color = "#ff3d00"; }
            return { score: score.toFixed(1), rating, color };
        }

        function runInference() {
            const rawTarget   = document.getElementById('target').value.trim();
            const rawMission  = document.getElementById('mission_type').value.trim();
            const target       = planetNames.find(p => p.toLowerCase() === rawTarget.toLowerCase()) || rawTarget;
            const mission_type = normalizeMissionType(rawMission);
            const spacecraft_mass = parseFloat(document.getElementById('spacecraft_mass').value) || 2200;
            const science_payload = parseFloat(document.getElementById('science_payload').value) || 120;
            const cruise_days = parseFloat(document.getElementById('cruise_days').value) || 250;
            const isp = parseFloat(document.getElementById('isp').value) || 320;
            
            let pInfo = planetsDB[target];
            if (!pInfo) { pInfo = { dist:1.5, mass:0.5, radius:3000, gravity:4.0, temp:-60, press:0.01, mag:0, gas:0, solid:1, desc:'Custom body.' }; }
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
                let finalScore, finalRating, finalColor;
                if (data && data.status === 'success' && data.score !== undefined) {
                    finalScore = data.score;
                    finalRating = data.rating;
                    finalColor = data.color;
                } else {
                    const fallback = calculateFallbackScore(target, mission_type, pInfo.dist, pInfo.temp, pInfo.press, pInfo.gravity);
                    finalScore = fallback.score;
                    finalRating = fallback.rating;
                    finalColor = fallback.color;
                }
                renderDashboard(finalScore, finalRating, finalColor, target, mission_type, pInfo, spacecraft_mass, science_payload, payload_ratio, cruise_days, isp);
            })
            .catch(err => {
                const fallback = calculateFallbackScore(target, mission_type, pInfo.dist, pInfo.temp, pInfo.press, pInfo.gravity);
                renderDashboard(fallback.score, fallback.rating, fallback.color, target, mission_type, pInfo, spacecraft_mass, science_payload, payload_ratio, cruise_days, isp);
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

            const reasons = generateReasons(target, mission_type, pInfo, mass, payload_ratio, days, score);
            const reasonsHTML = reasons.map(r => `<li>${r}</li>`).join('');

            document.getElementById('results').innerHTML = `
                <div class="section-title">📊 1. Mission Feasibility AI Score</div>
                <div class="score-card">
                    <div style="color:#94a3b8; font-size:0.85rem; text-transform:uppercase; letter-spacing:2px;">Mission Feasibility Score</div>
                    <div class="score-num" style="color:${color}">${score}</div>
                    <div style="font-weight:600; color:${color}">${rating}</div>
                    
                    <div class="reason-card">
                        <h4>🔍 Score Breakdown & Driving Factors</h4>
                        <ul>${reasonsHTML}</ul>
                    </div>
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
            "target_canonical": data.get("target"),
            "mission_type_normalized": data.get("mission_type"),
            "cruise_duration_days": float(data.get("cruise_days", 0)),
            "spacecraft_mass_kg": float(data.get("spacecraft_mass_kg", 0)),
            "science_payload_mass_kg": float(data.get("science_payload_kg", 0)),
            "historical_success_label": int(data.get("historical_success", 1)),
            "target_heliocentric_AU": float(data.get("distance_AU", 1.0)),
            "target_mass_1e24_kg": float(data.get("mass_1e24_kg", 0.1)),
            "target_radius_km": float(data.get("radius_km", 1000)),
            "target_gravity_m_s2": float(data.get("gravity_m_s2", 3.7)),
            "target_escape_velocity_km_s": float(data.get("escape_velocity_km_s", 5.0)),
            "blackbody_equilibrium_temperature_K": float(data.get("temperature_C", 20)) + 273.15,
            "target_has_conventional_solid_surface": int(data.get("solid_surface", 1)),
            "is_orbiter": 1 if "orbiter" in str(data.get("mission_type", "")).lower() else 0,
            "is_rover": 1 if "rover" in str(data.get("mission_type", "")).lower() else 0,
            "is_lander": 1 if "lander" in str(data.get("mission_type", "")).lower() else 0,
            "is_flyby": 1 if "flyby" in str(data.get("mission_type", "")).lower() else 0,
        }
        
        input_df = pd.DataFrame([input_dict])
        raw_score = model_obj.predict(input_df)[0]
        score = float(np.clip(raw_score, 0, 100))
        
        if score >= 70:
            rating = "HIGH FEASIBILITY"
            color = "#00e676"
        elif score >= 45:
            rating = "MODERATE FEASIBILITY"
            color = "#ffea00"
        else:
            rating = "LOW FEASIBILITY / HIGH RISK"
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

if __name__ == "__main__":
    print("Starting VornisAI local server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=True)

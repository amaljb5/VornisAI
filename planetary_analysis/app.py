from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

BASE = Path(__file__).resolve().parent
ASSET_DIR = BASE / "assets" / "planets"
SOLAR_API = "https://api.le-systeme-solaire.net/rest/bodies"
HORIZONS_API = "https://ssd.jpl.nasa.gov/api/horizons.api"
AU_KM = 149_597_870.7

PLANETS = {
    "Mercury": {"id": "mercury", "horizons": "199"},
    "Venus": {"id": "venus", "horizons": "299"},
    "Earth": {"id": "earth", "horizons": "399"},
    "Mars": {"id": "mars", "horizons": "499"},
    "Jupiter": {"id": "jupiter", "horizons": "599"},
    "Saturn": {"id": "saturn", "horizons": "699"},
    "Uranus": {"id": "uranus", "horizons": "799"},
    "Neptune": {"id": "neptune", "horizons": "899"},
}

# Local scientific fallback. Live APIs are always attempted first.
FALLBACK = {
    "Mercury": dict(mass=0.33011, radius=2439.7, gravity=3.70, escape=4.25, temp=167, pressure=0.0,
                     atmosphere="Extremely thin exosphere; sodium, oxygen, hydrogen, helium and potassium",
                     magnetic="Weak global magnetic field", orbit_days=87.969, rotation_hours=1407.6,
                     moons=0, semi_major=0.3871,
                     description="Small rocky planet with a very thin exosphere and extreme day-night temperature contrast."),
    "Venus": dict(mass=4.8675, radius=6051.8, gravity=8.87, escape=10.36, temp=464, pressure=92.0,
                   atmosphere="~96.5% CO₂, ~3.5% N₂; sulfuric-acid cloud layers",
                   magnetic="No intrinsic global magnetic field", orbit_days=224.701, rotation_hours=-5832.5,
                   moons=0, semi_major=0.7233,
                   description="Hot terrestrial planet with a dense CO₂ atmosphere, intense greenhouse heating and extreme pressure."),
    "Earth": dict(mass=5.97237, radius=6371.0, gravity=9.81, escape=11.19, temp=15, pressure=1.013,
                   atmosphere="~78% N₂, ~21% O₂, ~0.93% Ar", magnetic="Strong global magnetic field",
                   orbit_days=365.256, rotation_hours=23.934, moons=1, semi_major=1.0,
                   description="Rocky planet with a nitrogen-oxygen atmosphere, liquid surface water and a protective magnetosphere."),
    "Mars": dict(mass=0.64169, radius=3389.5, gravity=3.71, escape=5.03, temp=-63, pressure=0.00636,
                  atmosphere="~95% CO₂, ~2.7% N₂, ~1.6% Ar", magnetic="No global magnetic field",
                  orbit_days=686.98, rotation_hours=24.623, moons=2, semi_major=1.5237,
                  description="Cold terrestrial planet with a thin CO₂ atmosphere, dust storms, polar ice and evidence of ancient water."),
    "Jupiter": dict(mass=1898.13, radius=69911, gravity=24.79, escape=59.5, temp=-110, pressure=None,
                     atmosphere="~90% H₂, ~10% He with trace compounds", magnetic="Very strong global magnetic field",
                     orbit_days=4332.59, rotation_hours=9.925, moons=95, semi_major=5.2028,
                     description="Gas giant with powerful radiation belts, rapid rotation and no solid surface."),
    "Saturn": dict(mass=568.32, radius=58232, gravity=10.44, escape=35.5, temp=-140, pressure=None,
                    atmosphere="~96% H₂, ~3% He with trace methane and ammonia", magnetic="Strong global magnetic field",
                    orbit_days=10759.22, rotation_hours=10.656, moons=274, semi_major=9.5388,
                    description="Gas giant surrounded by an extensive ring system and a large satellite family."),
    "Uranus": dict(mass=86.811, radius=25362, gravity=8.69, escape=21.3, temp=-195, pressure=None,
                    atmosphere="H₂, He and CH₄; methane gives the atmosphere its blue-green color",
                    magnetic="Strong, highly tilted and offset magnetic field", orbit_days=30687.15,
                    rotation_hours=-17.24, moons=28, semi_major=19.1914,
                    description="Ice giant with a 97.8° axial tilt, cold atmosphere and unusual magnetic geometry."),
    "Neptune": dict(mass=102.409, radius=24622, gravity=11.15, escape=23.5, temp=-200, pressure=None,
                     atmosphere="H₂, He and CH₄ with methane-rich upper atmosphere",
                     magnetic="Strong, tilted and offset magnetic field", orbit_days=60190.03,
                     rotation_hours=16.11, moons=16, semi_major=30.0611,
                     description="Distant ice giant with extremely fast winds and a dynamic methane-rich atmosphere."),
}

st.set_page_config(page_title="VornisAI | Planetary Analysis", page_icon="🪐", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@500;600;700&display=swap');
html,body,[class*="css"]{font-family:Inter,sans-serif}
.stApp{background:radial-gradient(circle at 70% 20%,rgba(57,91,170,.18),transparent 35%),linear-gradient(135deg,#060914,#0b1223 55%,#111a34);color:#e8eefc}
.block-container{max-width:1380px;padding-top:2rem}
.hero{padding:10px 0 22px;border-bottom:1px solid rgba(255,255,255,.08);margin-bottom:24px}
.hero h1{font-family:Outfit;font-size:3rem;margin:0;background:linear-gradient(90deg,#eaf5ff,#53c8ff,#9c7cff);-webkit-background-clip:text;color:transparent}
.hero p{color:#94a7c6;margin:.35rem 0 0;font-size:1.05rem}
.card{background:rgba(20,29,50,.78);border:1px solid rgba(117,151,210,.18);border-radius:18px;padding:20px;box-shadow:0 12px 35px rgba(0,0,0,.22);margin-bottom:18px}
.section{font-family:Outfit;font-size:1.35rem;color:#4ccaff;margin:26px 0 12px}
.kicker{color:#4ccaff;text-transform:uppercase;letter-spacing:2px;font-size:.75rem;font-weight:700}
.planet-name{font-family:Outfit;font-size:2.4rem;margin:4px 0}.muted{color:#9aa9c0}.value{font-size:1.45rem;font-weight:700;color:#f3f7ff}.label{font-size:.78rem;color:#8ea0bc;text-transform:uppercase;letter-spacing:1px}
.insight{border-left:3px solid #6d5dfc;background:rgba(109,93,252,.08);padding:13px 16px;border-radius:0 12px 12px 0;margin:9px 0}
.source{font-size:.75rem;color:#7487a6}
[data-testid="stSidebar"]{background:#070b16}
button[kind="primary"]{background:linear-gradient(90deg,#157cff,#755cf8);border:0}
/* Planet image: fixed visual area, centered and never cropped */
[data-testid="stImage"]{min-height:420px;display:flex;align-items:center;justify-content:center;background:transparent;border:none;padding:0;box-sizing:border-box;overflow:hidden}
[data-testid="stImage"] img{width:390px !important;height:390px !important;max-width:100% !important;object-fit:contain !important;object-position:center center !important;display:block;margin:auto;background:transparent}
@media (max-width: 900px){[data-testid="stImage"]{min-height:340px}[data-testid="stImage"] img{width:320px !important;height:320px !important}}
</style>
""", unsafe_allow_html=True)


def get_json(url: str, params: dict | None = None, timeout: int = 15):
    r = requests.get(url, params=params, timeout=timeout,
                     headers={"User-Agent": "VornisAI-Planetary-Analysis/1.0"})
    r.raise_for_status()
    return r.json()


def num(value):
    return value if isinstance(value, (int, float)) else None


def extract_mass(body):
    m = body.get("mass") or {}
    try:
        return float(m["massValue"]) * 10 ** int(m["massExponent"]) / 1e24
    except Exception:
        return None


def extract_pressure(body):
    # The OpenData API can expose pressure in Pa. Convert to bar when appropriate.
    for key in ("pressure", "surfacePressure"):
        v = body.get(key)
        if isinstance(v, (int, float)):
            v = float(v)
            return v / 1e5 if abs(v) > 1000 else v
    return None


def extract_atmosphere(body):
    atmosphere = body.get("atmosphere")
    if isinstance(atmosphere, str):
        return atmosphere
    if isinstance(atmosphere, dict):
        comp = atmosphere.get("composition") or atmosphere.get("compositionList")
        if isinstance(comp, list):
            parts = []
            for item in comp[:10]:
                if isinstance(item, dict):
                    name = item.get("element") or item.get("name") or item.get("molecule")
                    pct = item.get("percentage") or item.get("percent")
                    if name:
                        parts.append(f"{name} {pct}%" if pct is not None else str(name))
            if parts:
                return ", ".join(parts)
        return atmosphere.get("description") or atmosphere.get("composition") or None
    return None


def planetary_api(name):
    body = get_json(f"{SOLAR_API}/{PLANETS[name]['id']}")
    d = FALLBACK[name].copy()

    fields = {
        "radius": "meanRadius",
        "gravity": "gravity",
        "escape": "escape",
        "temp": "avgTemp",
        "orbit_days": "sideralOrbit",
        "rotation_hours": "sideralRotation",
    }
    for output, key in fields.items():
        value = num(body.get(key))
        if value is not None:
            d[output] = float(value)

    mass = extract_mass(body)
    if mass is not None:
        d["mass"] = mass

    pressure = extract_pressure(body)
    if pressure is not None:
        d["pressure"] = pressure

    atmosphere = extract_atmosphere(body)
    if atmosphere:
        d["atmosphere"] = atmosphere

    moons = body.get("moons")
    if isinstance(moons, list):
        d["moons"] = len(moons)

    # OpenData returns semimajorAxis in km. Convert to AU.
    semi_major_km = num(body.get("semimajorAxis"))
    if semi_major_km is not None and semi_major_km > 1000:
        d["semi_major"] = float(semi_major_km) / AU_KM

    d["planetary_api"] = "live"
    return d


def horizons_distance(name):
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = now.strftime("%Y-%m-%d %H:%M")
    stop = (now + timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M")

    params = {
        "format": "json",
        "COMMAND": f"'{PLANETS[name]['horizons']}'",
        "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "VECTORS",
        "CENTER": "'500@10'",
        "START_TIME": f"'{start}'",
        "STOP_TIME": f"'{stop}'",
        "STEP_SIZE": "1 m",
        "OUT_UNITS": "AU-D",
        "REF_PLANE": "ECLIPTIC",
        "VEC_TABLE": "1",
    }
    result = get_json(HORIZONS_API, params=params)
    text = result.get("result", "")
    chunk = text.split("$$SOE", 1)[-1].split("$$EOE", 1)[0]

    # Horizons normally labels the vector values as X, Y and Z.
    match = re.search(
        r"X\s*=\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+"
        r"Y\s*=\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+"
        r"Z\s*=\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)",
        chunk,
    )
    if match:
        x, y, z = map(float, match.groups())
        return math.sqrt(x * x + y * y + z * z), start

    # CSV-style fallback.
    for line in chunk.splitlines():
        if not line.strip() or line.startswith("*"):
            continue
        values = re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", line)
        if len(values) >= 4:
            try:
                x, y, z = map(float, values[-3:])
                return math.sqrt(x * x + y * y + z * z), start
            except Exception:
                pass

    raise RuntimeError("NASA/JPL Horizons returned no parseable position vector")


def load_planet(name):
    data = FALLBACK[name].copy()
    statuses = []

    try:
        data.update(planetary_api(name))
        statuses.append("Planetary API: LIVE")
    except Exception:
        data["planetary_api"] = "fallback"
        statuses.append("Planetary API: FALLBACK")

    try:
        distance, timestamp = horizons_distance(name)
        data["current_distance"] = distance
        data["distance_time"] = timestamp
        statuses.append("NASA/JPL Horizons: LIVE")
    except Exception:
        data["current_distance"] = data["semi_major"]
        data["distance_time"] = None
        statuses.append("NASA/JPL Horizons: FALLBACK")

    data["status"] = statuses
    return data


def environmental_analysis(name, d):
    pressure = d.get("pressure")
    temp = d["temp"]
    insights = []

    if name in {"Jupiter", "Saturn", "Uranus", "Neptune"}:
        insights.append("This is a gas/ice giant with no conventional solid surface, so surface-landing conditions are not directly applicable.")
    elif pressure is not None and pressure < 0.01:
        insights.append("The atmosphere is extremely thin, limiting aerodynamic braking and requiring systems designed for near-vacuum conditions.")
    elif pressure is not None and pressure > 10:
        insights.append(f"Surface pressure is extreme at about {pressure:g} bar, creating major structural and entry/descent constraints.")
    else:
        insights.append("The atmosphere provides a meaningful environmental envelope for entry, thermal control and surface operations.")

    if temp >= 300:
        insights.append(f"The high temperature of about {temp:.0f} °C creates severe thermal-control requirements for electronics, power systems and materials.")
    elif temp <= -150:
        insights.append(f"The very low temperature of about {temp:.0f} °C creates cryogenic thermal-control and energy-storage challenges.")
    else:
        insights.append(f"The reference temperature is about {temp:.0f} °C, but local and seasonal conditions can vary substantially.")

    if "strong" in d["magnetic"].lower() or "very strong" in d["magnetic"].lower():
        insights.append("A substantial magnetic field can alter the charged-particle environment, but it does not eliminate radiation risk.")
    else:
        insights.append("The absence of a strong intrinsic global magnetic field increases reliance on spacecraft shielding and radiation planning.")

    if d["current_distance"] > 4:
        insights.append(f"At about {d['current_distance']:.2f} AU from the Sun, sunlight is much weaker than at Earth, making power-system design an important constraint.")
    elif d["current_distance"] < 0.7:
        insights.append(f"The planet is close to the Sun at about {d['current_distance']:.2f} AU, increasing solar flux and thermal loading compared with Earth.")
    else:
        insights.append(f"The current Sun distance is about {d['current_distance']:.2f} AU, which affects solar flux, thermal balance and communications planning.")

    return insights


def research_summary(name, d, insights):
    pressure = "not applicable as a conventional surface value" if d["pressure"] is None else f"{d['pressure']:g} bar"
    return (
        f"{name} is a {('gas/ice giant' if name in {'Jupiter','Saturn','Uranus','Neptune'} else 'terrestrial rocky planet')} "
        f"with a mean radius of {d['radius']:,.1f} km and surface/reference gravity of {d['gravity']:.2f} m/s². "
        f"Its reference temperature is about {d['temp']:.0f} °C and its surface pressure is {pressure}. "
        f"The atmosphere is described as {d['atmosphere']}. The planet is currently about {d['current_distance']:.4f} AU from the Sun according to NASA/JPL Horizons when live retrieval succeeds. "
        f"These characteristics indicate that thermal control, atmospheric interaction, radiation protection and power availability should be considered when studying missions to this body."
    )


# ------------------------------ UI ---------------------------------
st.markdown(
    '<div class="hero"><div class="kicker">VornisAI • Planetary Intelligence</div>'
    '<h1>Planetary Analysis</h1>'
    '<p>Planetary data, environmental analysis and research information from live space-data APIs.</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🪐 Target Planet")
    selected = st.selectbox("Select a planet", list(PLANETS), index=0)
    refresh = st.button("↻ Refresh planetary data", use_container_width=True, type="primary")
    st.caption("Live retrieval uses NASA/JPL Horizons for the current Sun-distance/orbital state and a planetary data API for physical and environmental parameters. Local scientific values are used only if an API is temporarily unavailable.")

if refresh:
    st.cache_data.clear()

@st.cache_data(ttl=900, show_spinner=False)
def cached_planet(name):
    return load_planet(name)

with st.spinner("Retrieving planetary data…"):
    d = cached_planet(selected)

# Only the selected planet image is displayed.

image_path = ASSET_DIR / f"{PLANETS[selected]['id']}.png"


left, right = st.columns([1.0, 1.7])
with left:
    st.image(str(image_path), width=390)
with right:
    st.markdown(
        f'<div class="card"><div class="kicker">Selected Planet</div><div class="planet-name">{selected}</div>'
        f'<p class="muted">{d["description"]}</p><div class="source">{" • ".join(d["status"])}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Current orbital state")
    st.metric("Current distance from Sun", f"{d['current_distance']:.4f} AU")
    if d.get("distance_time"):
        st.caption(f"NASA/JPL Horizons epoch: {d['distance_time']} UTC")

st.markdown('<div class="section">◆ Physical Parameters</div>', unsafe_allow_html=True)
cols = st.columns(5)
items = [
    ("Mass", f"{d['mass']:.5g} × 10²⁴ kg"),
    ("Mean Radius", f"{d['radius']:,.1f} km"),
    ("Surface Gravity", f"{d['gravity']:.2f} m/s²"),
    ("Escape Velocity", f"{d['escape']:.2f} km/s"),
    ("Moons", str(d['moons'])),
]
for col, (label, value) in zip(cols, items):
    with col:
        st.markdown(f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section">◆ Atmospheric & Environmental Parameters</div>', unsafe_allow_html=True)
a, b, c = st.columns(3)
with a:
    pressure_text = "Not applicable for a solid surface" if d["pressure"] is None else f"{d['pressure']:g} bar"
    st.markdown(
        f'<div class="card"><div class="label">Reference temperature</div>'
        f'<div class="value">{d["temp"]:.0f} °C</div>'
        f'<div class="label" style="margin-top:18px">Surface pressure</div>'
        f'<div class="value" style="font-size:1.05rem">{pressure_text}</div></div>',
        unsafe_allow_html=True,
    )
with b:
    st.markdown(f'<div class="card"><div class="label">Atmosphere</div><p>{d["atmosphere"]}</p><div class="label">Magnetic environment</div><p>{d["magnetic"]}</p></div>', unsafe_allow_html=True)
with c:
    classification = "Gas/Ice giant" if selected in {"Jupiter", "Saturn", "Uranus", "Neptune"} else "Terrestrial / rocky planet"
    surface = "No conventional solid surface" if selected in {"Jupiter", "Saturn", "Uranus", "Neptune"} else "Solid planetary surface"
    st.markdown(f'<div class="card"><div class="label">Environmental classification</div><p>{classification}</p><div class="label">Surface</div><p>{surface}</p></div>', unsafe_allow_html=True)

st.markdown('<div class="section">◆ Environmental Analysis</div>', unsafe_allow_html=True)
insights = environmental_analysis(selected, d)
for insight in insights:
    st.markdown(f'<div class="insight">{insight}</div>', unsafe_allow_html=True)

st.markdown('<div class="section">◆ Orbital & Rotational Parameters</div>', unsafe_allow_html=True)
orb = pd.DataFrame([
    {"Parameter": "Semi-major axis", "Value": f"{d['semi_major']:.4f} AU"},
    {"Parameter": "Current Sun distance", "Value": f"{d['current_distance']:.4f} AU"},
    {"Parameter": "Orbital period", "Value": f"{d['orbit_days']:.2f} days"},
    {"Parameter": "Rotation period", "Value": f"{d['rotation_hours']:.2f} hours"},
    {"Parameter": "Moons", "Value": d['moons']},
])
st.dataframe(orb, use_container_width=True, hide_index=True)

st.markdown('<div class="section">◆ Planetary Research Summary</div>', unsafe_allow_html=True)
st.markdown(f'<div class="card"><p>{research_summary(selected, d, insights)}</p></div>', unsafe_allow_html=True)

st.markdown('<div class="source">Data sources: NASA/JPL Horizons • Solar System OpenData • Local scientific fallback values used only when live retrieval is unavailable.</div>', unsafe_allow_html=True)

# 🚀 VornisAI  
### _"AI-Powered Space Mission & Vehicle Design Assistant"_

**VornisAI** is an intelligent software system that helps design **space exploration vehicles** — satellites, rovers, and orbiters — using **planetary data, AI-driven recommendations**, and **mission simulation**.  

It’s your virtual aerospace partner: capable of analyzing planetary environments and suggesting how to design spacecraft optimized for those conditions.  

---

## 🌌 Overview

VornisAI combines **AI reasoning**, **planetary databases**, and **aerospace engineering principles** to generate:
- Design ideas for satellites, rovers, and orbiters  
- Feasibility reports for interplanetary missions  
- Entry and exit analysis for different planets  
- Planetary data visualization and AI explanations  

The goal is to **simplify mission planning and inspire innovation** in space technology through intelligent automation.

---

## 🪐 Core Features

### 🧠 1. AI Design Recommendation Engine
- Generates smart design suggestions based on planetary data.  
- Considers factors like:
  - Atmospheric composition  
  - Gravity and pressure  
  - Temperature ranges  
  - Terrain and surface type  
- Recommends:
  - Structural materials  
  - Propulsion systems  
  - Communication and power modules  
  - Landing and heat management systems  

> Example:  
> _For Venus, use titanium alloy for structure and reinforced thermal shielding due to high surface temperature._

---

### 🌍 2. Planetary Intelligence Database
- Detailed database for every major planet and moon in our solar system.  
- Includes:
  - Surface gravity  
  - Chemical composition  
  - Radiation levels  
  - Atmospheric pressure and density  
  - Orbital and rotational characteristics  
- Connected to **NASA’s Open APIs** for real-time planetary data.  

---

### 🚀 3. Mission Feasibility Calculator
- Calculates physics-based parameters like:
  - Escape velocity  
  - Delta-v requirements  
  - Launch mass and payload ratio  
  - Thrust-to-weight balance  
- Helps determine if a mission concept is **realistic or needs redesign**.  

---

### 🛰️ 4. Launch & Entry Dynamics
- Simulates planetary **entry, descent, and landing** (EDL) conditions.  
- Estimates optimal **launch windows** and **orbital paths**.  
- Suggests best propulsion and burn-stage configurations for efficient transfers.  

---

### 🧩 5. Modular Architecture
| Module | Purpose |
|---------|----------|
| 🪐 Planetary Data Module | Fetches and stores planetary environment data |
| 🧠 AI Design Module | Generates spacecraft/rover design suggestions |
| 🔬 Planetary Comparison & Analysis Module | Multi-body side-by-side trade studies, feasibility scoring & AI comparative reports |
| ⚙️ Simulation Module *(not yet planned)*| Performs mission feasibility and physics analysis |
| 🛰️ Visualization Module *(not yet planned)* | Displays vehicle schematics and orbital paths |

---

### 🔬 6. Planetary Comparison & Analysis Module

The **Planetary Comparison & Analysis Module** enables side-by-side multi-body aerospace trade studies across 19 major Solar System bodies (planets, dwarf planets, and major moons).

#### 📁 Module Architecture
```text
api/
├── planetary_data.py       # Typed NASA/JPL reference data (19 bodies + presets)
├── planetary_data.json     # Standardized JSON data layer for UI and external clients
├── feasibility_engine.py   # Multi-attribute scoring engine (7 sub-scores & 5 mission profiles)
├── ai_analysis.py          # AI service wrapper (OpenAI + deterministic aerospace reasoning fallback)
└── index.py                # Compatibility launcher for the Streamlit dashboard
ml/
└── app.py                  # Main Streamlit dashboard (mission design and planetary comparison)
tests/
├── run_tests.ps1           # Automated verification test suite (152 tests, 100% pass)
├── test_planetary_comparison.py # Python unittest test suite
```

#### 📊 How Scoring Works
The composite feasibility score is a transparent weighted sum of **7 physics & engineering sub-scores** evaluated on a `[0, 100]` scale:

$$\text{Composite Feasibility Score} = \sum_{i=1}^{7} \left( \text{Sub-Score}_i \times \text{Weight}_i \right)$$

1. **Landing Difficulty:** Evaluates gravity hazards, aerobraking potential vs. crush overpressure, and descent propellant mass.
2. **Surface Survivability:** Evaluates temperature bounds, diurnal thermal swings, vacuum/pressure extremes, and volcanic/acid hazards.
3. **Communication Delay:** Calculates one-way light-time telemetry latency based on distance from Earth ($d \approx \Delta\text{AU} \times 499\text{s}$).
4. **Resource Availability (ISRU):** Measures access to water/ice, extractable atmospheric gases (CO₂, N₂, CH₄), and solar irradiance.
5. **Radiation Risk:** Assesses magnetospheric dipole shielding vs. trapped Jovian belts and galactic cosmic rays.
6. **Travel Time & Delta-v:** Models Hohmann transfer energy, launch C3, and interplanetary cruise duration.
7. **Long-Term Habitability:** Evaluates musculoskeletal gravity adequacy, atmospheric shielding, and closed-loop life support viability.

##### Mission Architecture Weight Profiles
| Sub-Score Factor | Flyby | Orbiter | Lander | Crewed | Colony |
|:-----------------|:-----:|:-------:|:------:|:------:|:------:|
| Travel Time & Delta-v | **35%** | **30%** | **20%** | 15% | 5% |
| Radiation Risk | 25% | 25% | 10% | **20%** | 15% |
| Communication Delay | 25% | 20% | 15% | 10% | 0% |
| Resource Availability (ISRU) | 15% | 15% | 5% | 5% | **25%** |
| Landing Difficulty | 0% | 0% | **25%** | 15% | 10% |
| Surface Survivability | 0% | 10% | **25%** | **25%** | 20% |
| Long-Term Habitability | 0% | 0% | 0% | 10% | **25%** |

##### Color-Coded Tiers
- 🟢 **High Feasibility:** $\ge 70.0$ (`#00e676`)
- 🟡 **Moderate Feasibility:** $45.0 - 69.9$ (`#ffea00`)
- 🔴 **Low Feasibility / High Risk:** $< 45.0$ (`#ff3d00`)

#### 🪐 How to Add a New Celestial Body
To add a new body (e.g., an asteroid or newly characterized moon), append its entry to `api/planetary_data.py` (and optionally `api/planetary_data.json`) under `PLANETARY_DATA`:

```python
"Vesta": {
    "id": "vesta",
    "name": "Vesta",
    "type": "Asteroid / Protoplanet",
    "parent_body": "Sun",
    "icon": "☄️",
    "color": "#94a3b8",
    "physical": {
        "radius_km": 262.7,
        "mass_1e24_kg": 0.000259,
        "density_g_cm3": 3.456,
        "surface_gravity_m_s2": 0.25,
        "escape_velocity_km_s": 0.36,
        "rotation_period_hours": 5.34,
        "orbital_period_days": 1325.7,
        "axial_tilt_deg": 29.0,
        "distance_from_sun_au": 2.36,
        "number_of_moons": 0
    },
    "atmospheric": {
        "surface_pressure_bar": 0.0,
        "pressure_display": "None (Hard Vacuum)",
        "composition": [{"gas": "Vacuum", "percentage": 100.0}],
        "magnetosphere": "None",
        "greenhouse_effect": "None",
        "weather_notes": "Direct solar exposure and micrometeorite sputtering."
    },
    "environmental": {
        "mean_temp_c": -135,
        "min_temp_c": -188,
        "max_temp_c": -20,
        "radiation_level": "High (Unmitigated cosmic and solar radiation)",
        "radiation_tier": "Severe",
        "surface_type": "Basaltic eucrite crust; massive South Pole impact crater (Rheasilvia)",
        "water_or_ice": "Trace hydrated minerals; no free ice detected",
        "geological_activity": "Inactive",
        "habitability_notes": "Low gravity staging point for asteroid belt prospecting."
    },
    "sources": "NASA Dawn Mission Archives"
}
```
The UI, radar charts, comparative tables, feasibility engine, and AI analysis automatically adapt to any new body added without touching frontend components.

---

---

### 💡 6. Educational & Research Mode
- Every AI suggestion includes a brief scientific explanation.  
- Ideal for **students**, **researchers**, and **aerospace educators**.  
- Encourages learning about **planetary environments**, **mission constraints**, and **vehicle design trade-offs**.  

---

## ⚙️ Tech Stack (Planned)

| Layer | Tools / Frameworks |
|--------|---------------------|
| **Application** | Python / Streamlit |
| **AI Engine** | Rule-based AI + ML models |
| **Database** | MySQL / MongoDB |
| **Visualization** | Three.js / Plotly |
| **APIs** | NASA Open APIs, SpaceX API |

## 🚀 Run the Streamlit app

Install dependencies with `pip install -r requirements.txt`, then launch the unified
mission design and planetary comparison dashboard:

```bash
python run.py
```

Alternatively, run Streamlit directly with `streamlit run ml/app.py`.

---

## 🌠 Future Roadmap
- 🌍 Expand to **exoplanetary mission simulation**  
- 🤖 Integrate **deep learning models** for adaptive design patterns  
- 🧑‍🚀 Multi-user collaboration for joint mission planning  
- 🪐 Build full **3D visualization lab** for spacecraft design  

---

## 🧑‍💻 Author
**Developed by:** [@amaljb5](https://github.com/amaljb5)  
**Vision:**  
> To merge **AI and space exploration** into a single intelligent system that designs, learns, and inspires the next generation of explorers.

---

## 📜 License
This project is licensed under the **Apache License 2.0**.  
You may freely use, modify, and distribute it with proper attribution.  
See the [LICENSE](./LICENSE) file for details.  

---

> _"The stars are not beyond reach — only beyond imagination.  
VornisAI brings that imagination closer."_ ✨

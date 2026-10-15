"""
planet_data.py - Planetary database for the Vornis AI Chatbot.
Synced with api/planetary_data.py (19 celestial bodies).
"""

import sys
from pathlib import Path

# Add api to path if possible
api_dir = Path(__file__).resolve().parent.parent / "api"
if str(api_dir) not in sys.path:
    sys.path.insert(0, str(api_dir))

try:
    from planetary_data import PLANETARY_DATA

    PLANETS_DB = {
        name: {
            "dist": b["physical"]["distance_from_sun_au"],
            "mass": b["physical"]["mass_1e24_kg"],
            "radius": b["physical"]["radius_km"],
            "gravity": b["physical"]["surface_gravity_m_s2"],
            "temp": b["environmental"]["mean_temp_c"],
            "press": b["atmospheric"]["surface_pressure_bar"],
            "mag": 1 if "intrinsic" in b["atmospheric"]["magnetosphere"].lower() or "strong" in b["atmospheric"]["magnetosphere"].lower() else 0,
            "gas": 1 if "Gas Giant" in b["type"] or "Ice Giant" in b["type"] else 0,
            "solid": 0 if "Gas Giant" in b["type"] or "Ice Giant" in b["type"] else 1,
            "desc": b["environmental"]["habitability_notes"]
        }
        for name, b in PLANETARY_DATA.items()
    }
except ImportError:
    PLANETS_DB = {
        "Mercury": {"dist": 0.387, "mass": 0.330, "radius": 2439, "gravity": 3.70, "temp": 167, "press": 1e-11, "mag": 0, "gas": 0, "solid": 1, "desc": "Extreme thermal fluctuations, high solar radiation, thin exosphere."},
        "Venus": {"dist": 0.723, "mass": 4.867, "radius": 6051, "gravity": 8.87, "temp": 464, "press": 92.0, "mag": 0, "gas": 0, "solid": 1, "desc": "Dense carbon dioxide atmosphere, supercritical pressures, corrosive sulfuric acid clouds."},
        "Earth": {"dist": 1.000, "mass": 5.972, "radius": 6371, "gravity": 9.81, "temp": 15, "press": 1.013, "mag": 1, "gas": 0, "solid": 1, "desc": "Habitable nitrogen-oxygen atmosphere, protective magnetosphere, liquid surface oceans."},
        "Moon": {"dist": 1.000, "mass": 0.073, "radius": 1737, "gravity": 1.62, "temp": -20, "press": 3e-15, "mag": 0, "gas": 0, "solid": 1, "desc": "Airless vacuum, high micrometeoroid risk, sharp abrasive lunar regolith."},
        "Mars": {"dist": 1.524, "mass": 0.642, "radius": 3389, "gravity": 3.71, "temp": -65, "press": 0.0069, "mag": 0, "gas": 0, "solid": 1, "desc": "Thin CO2 atmosphere, global dust storms, polar ice caps, legacy surface water channels."},
        "Phobos": {"dist": 1.524, "mass": 0.00000001, "radius": 11, "gravity": 0.0057, "temp": -40, "press": 0.0, "mag": 0, "gas": 0, "solid": 1, "desc": "Irregular carbonaceous moon of Mars, low escape velocity."},
        "Ceres": {"dist": 2.767, "mass": 0.00094, "radius": 473, "gravity": 0.28, "temp": -106, "press": 1e-12, "mag": 0, "gas": 0, "solid": 1, "desc": "Water ice rich dwarf planet in asteroid belt with cryovolcanism."},
        "Jupiter": {"dist": 5.203, "mass": 1898.0, "radius": 69911, "gravity": 24.79, "temp": -110, "press": 100.0, "mag": 1, "gas": 1, "solid": 0, "desc": "Massive gas giant, extreme radiation belts, violent storm systems."},
        "Io": {"dist": 5.203, "mass": 0.089, "radius": 1821, "gravity": 1.80, "temp": -143, "press": 1e-9, "mag": 0, "gas": 0, "solid": 1, "desc": "Hyperactive volcanic Jovian moon with lethal radiation exposure."},
        "Europa": {"dist": 5.203, "mass": 0.048, "radius": 1560, "gravity": 1.31, "temp": -160, "press": 1e-12, "mag": 0, "gas": 0, "solid": 1, "desc": "Subsurface liquid ocean beneath ice crust, high radiation."},
        "Ganymede": {"dist": 5.203, "mass": 0.148, "radius": 2634, "gravity": 1.43, "temp": -163, "press": 1e-11, "mag": 1, "gas": 0, "solid": 1, "desc": "Largest moon in solar system with intrinsic magnetic dipole."},
        "Callisto": {"dist": 5.203, "mass": 0.108, "radius": 2410, "gravity": 1.24, "temp": -139, "press": 7.5e-12, "mag": 0, "gas": 0, "solid": 1, "desc": "Ancient cratered ice/rock world with low radiation hazard."},
        "Saturn": {"dist": 9.537, "mass": 568.3, "radius": 58232, "gravity": 10.44, "temp": -140, "press": 100.0, "mag": 1, "gas": 1, "solid": 0, "desc": "Ringed gas giant, low bulk density, extensive satellite system."},
        "Titan": {"dist": 9.580, "mass": 0.135, "radius": 2575, "gravity": 1.35, "temp": -179, "press": 1.46, "mag": 0, "gas": 0, "solid": 1, "desc": "Dense nitrogen atmosphere, liquid methane lakes, cryovolcanism."},
        "Enceladus": {"dist": 9.537, "mass": 0.0001, "radius": 252, "gravity": 0.113, "temp": -198, "press": 1e-13, "mag": 0, "gas": 0, "solid": 1, "desc": "Active cryovolcanic water vapor and ice plumes into Saturn orbit."},
        "Uranus": {"dist": 19.19, "mass": 86.81, "radius": 25362, "gravity": 8.87, "temp": -195, "press": 100.0, "mag": 1, "gas": 1, "solid": 0, "desc": "Ice giant, extreme axial tilt (97.8°), freezing atmospheric temperatures."},
        "Neptune": {"dist": 30.07, "mass": 102.4, "radius": 24622, "gravity": 11.15, "temp": -200, "press": 100.0, "mag": 1, "gas": 1, "solid": 0, "desc": "Outer ice giant, supersonic atmospheric winds (up to 2,100 km/h)."},
        "Triton": {"dist": 30.07, "mass": 0.021, "radius": 1353, "gravity": 0.78, "temp": -235, "press": 1.4e-5, "mag": 0, "gas": 0, "solid": 1, "desc": "Retrograde nitrogen-ice moon of Neptune with active geysers."},
        "Pluto": {"dist": 39.48, "mass": 0.013, "radius": 1188, "gravity": 0.62, "temp": -225, "press": 1e-5, "mag": 0, "gas": 0, "solid": 1, "desc": "Kuiper belt dwarf planet, nitrogen-methane ice surface, faint seasonal atmosphere."}
    }

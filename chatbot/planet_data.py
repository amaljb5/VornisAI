# Copied from api/index.py

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

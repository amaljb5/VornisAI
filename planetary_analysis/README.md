# VornisAI - Planetary Analysis

This package contains only the Planetary Analysis module.

## Features
- Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus and Neptune
- Local image for the selected planet only
- Live physical/environmental retrieval from the Solar System OpenData planetary API
- Current Sun-distance/orbital state from NASA/JPL Horizons
- Physical, atmospheric and orbital parameters
- Environmental analysis
- Research summary generated from the retrieved planetary data
- Local scientific fallback if an external API is temporarily unavailable

## Run
```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open http://localhost:8501

No API key is required for the planetary-data sources used by this module.

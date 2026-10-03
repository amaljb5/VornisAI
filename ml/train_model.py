"""
VornisAI — Mission Feasibility Score Prediction Model
======================================================
Trains a regression model to predict `feasibility_score` (0-100) from
mission parameters and planetary data.

Usage:
    python train_model.py

Outputs:
    ml/model/feasibility_model.pkl   – trained sklearn pipeline
    ml/model/model_metadata.json     – feature names, metrics, training info
"""

import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "vornisai_feasibility_dataset_v2.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# ── load data ────────────────────────────────────────────────────────────
print("📂 Loading dataset …")
df = pd.read_csv(CSV_PATH)
print(f"   Rows: {len(df)}  |  Columns: {len(df.columns)}")

# ── feature engineering ──────────────────────────────────────────────────
# Drop identifiers & the label column (keep only features + target)
DROP_COLS = ["mission", "outcome", "feasibility_label"]
TARGET = "feasibility_score"

# Categorical & numeric feature lists
CAT_FEATURES = ["target", "mission_type"]
NUM_FEATURES = [
    "cruise_days",
    "spacecraft_mass_kg",
    "science_payload_kg",
    "historical_success",
    "distance_AU",
    "mass_1e24_kg",
    "radius_km",
    "gravity_m_s2",
    "escape_velocity_km_s",
    "temperature_C",
    "pressure_bar",
    "global_magnetic_field",
    "gas_giant",
    "solid_surface",
    "is_orbiter",
    "is_rover",
    "is_lander",
    "is_flyby",
]

ALL_FEATURES = CAT_FEATURES + NUM_FEATURES

X = df[ALL_FEATURES].copy()
y = df[TARGET].copy()

print(f"\n🔧 Features ({len(ALL_FEATURES)}):")
for f in ALL_FEATURES:
    print(f"   • {f}")
print(f"\n🎯 Target: {TARGET}")

# ── preprocessing pipeline ──────────────────────────────────────────────
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUM_FEATURES),
        ("cat", categorical_transformer, CAT_FEATURES),
    ]
)

# ── model: ensemble of RandomForest + GradientBoosting ───────────────────
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
)
gb = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
)
ensemble = VotingRegressor(estimators=[("rf", rf), ("gb", gb)])

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", ensemble),
    ]
)

# ── evaluation (Leave-One-Out CV — best for small datasets) ─────────────
print("\n📊 Evaluating with Leave-One-Out Cross-Validation …")
loo = LeaveOneOut()
y_pred_cv = cross_val_predict(pipeline, X, y, cv=loo)

mae = mean_absolute_error(y, y_pred_cv)
rmse = np.sqrt(mean_squared_error(y, y_pred_cv))
r2 = r2_score(y, y_pred_cv)

print(f"   MAE  = {mae:.2f}")
print(f"   RMSE = {rmse:.2f}")
print(f"   R²   = {r2:.4f}")

# ── train final model on all data ────────────────────────────────────────
print("\n🏋️ Training final model on full dataset …")
pipeline.fit(X, y)

# ── save artefacts ───────────────────────────────────────────────────────
model_path = MODEL_DIR / "feasibility_model.pkl"
joblib.dump(pipeline, model_path)
print(f"   ✅ Model saved → {model_path}")

# Unique values for categorical features (for the UI dropdowns)
cat_unique = {col: sorted(X[col].dropna().unique().tolist()) for col in CAT_FEATURES}

metadata = {
    "target": TARGET,
    "cat_features": CAT_FEATURES,
    "num_features": NUM_FEATURES,
    "cat_unique_values": cat_unique,
    "num_feature_stats": {
        col: {
            "min": float(X[col].min()) if X[col].notna().any() else None,
            "max": float(X[col].max()) if X[col].notna().any() else None,
            "median": float(X[col].median()) if X[col].notna().any() else None,
        }
        for col in NUM_FEATURES
    },
    "metrics": {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)},
    "dataset_rows": len(df),
    "model_file": "feasibility_model.pkl",
}

meta_path = MODEL_DIR / "model_metadata.json"
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"   ✅ Metadata saved → {meta_path}")

print("\n🚀 Done!  Run the Streamlit app with:")
print("   streamlit run ml/app.py")

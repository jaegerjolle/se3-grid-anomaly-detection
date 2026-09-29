# Elnatse-3: Real-Time Grid Anomaly Detection for SE3

A machine learning system for detecting anomalies in the Swedish power grid (SE3) using XGBoost and synchronized real-time API streams from ENTSO-E (grid operations) and Open-Meteo (meteorological analytics).

## What It Does

Combines weather data and grid metrics to predict power transmission losses and identify maintenance needs via condition-based monitoring and geographic triangulation across 5 Swedish cities.

Input: Load + temperatures from Gävle, Stockholm, Västerås, Örebro, Jönköping
Output: Predicted vs. actual grid losses -> Anomaly alert if residual > 15 MWh

---

## Quick Start

1. Setup:
pip install -r requirements.txt

2. Get API key: 
In order to get the API key you need to have an account on:
https://transparency.entsoe.eu and send a mail to them requesting the API key.

3. Fetch Data:
python smhi_weather_harvest.py    # Weather from SMHI (2 years historical)
python fetch_svk_data.py          # Grid data from SVK (2 years historical)
python feature.py                 # Merge & engineer features

4. Train Model:
python train.py

5. Run Live Dashboard:
streamlit run live_detect.py

Visit http://localhost:8501

---

## File Guide

- smhi_weather_harvest.py: Fetch 2-year temperature history from 5 SMHI stations for ML training
- fetch_svk_data.py: Fetch 2-year grid load & loss data from SVK API for ML training
- feature.py: Merge datasets, pivot cities, engineer features, output final_training_features.csv
- train.py: Train XGBoost model, save to saved_models/xgboost_se3_losses.json
- live_detect.py: Streamlit dashboard: synchronized live API mode (ENTSO-E + Open-Meteo) + offline simulation
- requirements.txt: Python dependencies

---

## Key Configuration

In live_detect.py, adjust these constants:

ANOMALY_THRESHOLD = 15.0      # MWh (alert if residual exceeds this)
DATA_LOOKBACK_HOURS = 5       # Dynamic synchronization window for ENTSO-E edge processing lag
DOMAIN_SE3 = "10YSE-1--------K" # ENTSO-E grid domain EIC code

---

## How It Works

### Data Pipeline
1. Historical Weather (SMHI) -> CSV with columns: Datum, Tid (UTC), Lufttemperatur, Stad
2. Historical Grid (SVK) -> CSV with columns: timestamp, totalLoadMwh, gridLossesMwh
3. Feature Engineering -> Merges on timestamp, adds load_squared (I2R), temporal features
4. Training Data -> final_training_features.csv

### Model Training
- Algorithm: XGBoost Regressor
- Target: Predict gridLossesMwh
- Features: Load, 5 city temperatures, hour, month, load_squared, load_temp_interaction
- Output: saved_models/xgboost_se3_losses.json

### Live Detection
- Offline Mode: Manual sliders for testing
- Live Mode: 
  - Queries ENTSO-E for active load with a rolling 5-hour synchronization window
  - Queries Open-Meteo for real-time grid-interpolated temperatures using exact geospatial coordinates
  - CRITICAL: Both APIs synchronized to the exact same date + hour
  - Compares actual vs. predicted losses and flags active system time-stamps per node

---

## Data Synchronization (Important!)

ENTSO-E provides data with a rolling lag (optimized via a 5-hour lookup lock).

Open-Meteo provides immediate real-time tracking, which we lock to match the exact ENTSO-E timeframe.

Timeline Example:
Current Time: June 16, 20:00 UTC
live_detect.py calculates: target_time = now() - 5 hours = June 16, 15:00 UTC
ENTSO-E API: Query block: 15:00 -> Returns 15:00 MWh Load
Open-Meteo API: Look for: June 16, 15:00 UTC hour -> Returns 15:00 temperatures
Result: Perfect real-time data alignment without data gaps!

---

## Dashboard Modes

### Simulering (Offline Dummy)
- Use manual sliders for load, losses, and temperatures
- Test anomaly detection logic and fallback workflows
- No active API connections

### Skarpt Live-läge (API)
- Fetches real-time grid load averages from ENTSO-E (5h lag lock)
- Fetches real-time temperatures from Open-Meteo for all 5 cities (same timestamp)
- Displays individual verified time-stamps and active fallback logs per node
- Shows system residual (measured - predicted)

---

## Troubleshooting

### Model file not found
python train.py

### API Error: No data for exact timestamp
- Check internet connection
- ENTSO-E data might have an extended reporting delay; try increasing DATA_LOOKBACK_HOURS in config
- If Open-Meteo fails, the system automatically deploys localized safe-state thermal fallbacks

### CSVs missing
The workflow expects these to exist before running feature.py:
- smhi_se3_2ar_temperatur.csv (from smhi_weather_harvest.py)
- svk_se3_2ar_natdata.csv (from fetch_svk_data.py)

---

## Git & Storage

NOT stored in Git (see .gitignore):
- *.csv files (large datasets)
- saved_models/ (trained models)
- *.pkl, *.json (model artifacts)

IS stored in Git:
- Python source code
- requirements.txt
- .gitignore
- Documentation

---

## Maintenance Workflow

### Weekly
python smhi_weather_harvest.py  # Fetch new historical weather data
python fetch_svk_data.py         # Fetch new historical grid data
python feature.py                # Regenerate features
python train.py                  # Retrain model

### Then Run Live Dashboard
streamlit run live_detect.py

---

## Geographic Triangulation

Dashboard shows 5 city status cards with independent data tracking:

Gävle              Stockholm          Västerås           Örebro             Jönköping
15.0C             14.4C             15.0C             16.0C             18.5C
NORMAL            NORMAL            NORMAL            CRITICAL ALERT    NORMAL
2026-06-16 UTC    2026-06-16 UTC    2026-06-16 UTC    2026-06-16 UTC    2026-06-16 UTC

If residual > 15 MWh, the city with the highest temperature deviation from the regional mean is flagged as CRITICAL ALERT. This instantly narrows down which geographical transmission sector requires condition-based maintenance.

---

## Debug Info

Click "Avancerad Systemanalys (Debug)" expander in dashboard to see:
- Target timestamp (chronologically locked)
- Dynamic feature matrix injected into the model
- Geographic mean temperature and total thermal span gradient
- Residual calculation flags

---

## Production Geospatial Node Coordinates

- Gävle: Lat 60.6749, Lon 17.1412 (Training Baseline: SMHI Station 107420)
- Stockholm: Lat 59.3293, Lon 18.0686 (Training Baseline: SMHI Station 98210)
- Västerås: Lat 59.6099, Lon 16.5448 (Training Baseline: SMHI Station 96190)
- Örebro: Lat 59.2753, Lon 15.2134 (Training Baseline: SMHI Station 95160)
- Jönköping: Lat 57.7826, Lon 14.1618 (Training Baseline: SMHI Station 74460)

---

## Key Design Decisions

Why a hybrid meteorological pipeline? The model utilizes SMHI's pristine corrected-archive during the offline training stage to ensure data validation. For production live-tracking, it interfaces with Open-Meteo's grid-interpolated arrays to bypass SMHI's historical validation lag and secure 100% dashboard uptime.

Why update to timezone-aware datetime objects? Switched from datetime.utcnow() to datetime.now(timezone.utc) to prevent runtime deprecation warnings, protect against daylight saving shifts, and stabilize the API lookup intervals.

Why XGBoost? Grid losses have a tight, non-linear relationship with transmission load. Gradient boosted trees isolate these trends alongside thermal weather inputs exceptionally well.

Why the load_squared feature? Physics-informed machine learning design: transmission losses are directly proportional to the square of the current (I2), meaning a quadratic load transformation guides the model toward real-world physics.

---

## Next Steps

- Spatial Anomaly Map: Integrate an interactive geospatial rendering mesh directly into the dashboard (st.pydeck).
- Predictive Forecasting Module: Feed ENTSO-E Day-Ahead load forecasts and 7-day weather trends into the model to predict structural losses 24 hours in advance.
- XAI Integration: Add a SHAP analytics layer to break down the specific feature impacts during threshold anomalies.

---

Last Updated: June 16, 2026

# Elnatse-3: Real-Time Grid Anomaly Detection for SE3

A physics-informed machine learning architecture for detecting transmission anomalies in the Swedish power grid (SE3) using XGBoost and synchronized real-time API streams from ENTSO-E (grid operations) and Open-Meteo (meteorological analytics).

## 🎯 Project Overview

This system combines macro-grid load telemetry and regional thermal metrics to predict transmission losses in real time. By contrasting actual losses against an AI baseline, the system executes **condition-based monitoring (CBM) and spatial anomaly localization** across 5 strategic nodes in the SE3 bidding area.

* **Inputs:** Active Grid Load (MWh) + Synchronized ambient temperatures from Gävle, Stockholm, Västerås, Örebro, and Jönköping.
* **Outputs:** AI-Predicted Grid Losses vs. Actual Grid Losses → System Residual Alert triggered if $|Residual| > 15.0\text{ MWh}$.

---

## ⚡ Quick Start

### 1. Environment Setup
```bash
pip install -r requirements.txt
2. Historical Pipeline (Training)
Bash
python smhi_weather_harvest.py    # Historical baseline harvester (SMHI)
python fetch_svk_data.py          # Historical grid telemetry harvester (SVK)
python feature.py                 # Feature Engineering & dataset fusion
3. Model Training
Bash
python train.py                   # Outputs trained XGBoost model artifacts
4. Launch Live Control Room Dashboard
Bash
streamlit run live_detect.py
Open your browser at http://localhost:8501
📁 File & Architecture Guide
File	Type	Purpose
live_detect.py	Streamlit App	Production environment. Features dual-mode execution (Skarpt Live-läge via APIs / Offline Simulation) with end-to-end data integrity validation.
train.py	Core ML	Trains the XGBRegressor on historical grid states and saves structural matrices to saved_models/.
feature.py	Pipeline	Fuses regional weather time-series with grid telemetry, adding non-linear physics-informed features.
smhi_weather_harvest.py	Ingestion	Pipeline for historical training data ingestion (SMHI MetObs archive).
fetch_svk_data.py	Ingestion	Ingests historical structural baselines for grid operations.
🔑 System Configuration & Boundaries
Global thresholds locked within live_detect.py:
Python
ANOMALY_THRESHOLD = 15.0      # MWh (Tolerance window before triggering alerts)
DATA_LOOKBACK_HOURS = 5       # Dynamic synchronization window to account for ENTSO-E edge processing lag
DOMAIN_SE3 = "10YSE-1--------K" # ENTSO-E EIC code for Sweden Bidding Zone 3
⚙️ How It Works (Production Engine)
[ENTSO-E API]  ──(5h Sync Window)──> Active Grid Load (MWh)  ──┐
                                                               ├──> [XGBoost Core] ──> Predicted Loss
[Open-Meteo]   ──(Geo-Coordinates)─> Regional Temps (°C)     ──┘                          │
                                                                                          ▼
[Control Room] <── (Visual Alarm Flag) 🚀 ── [Calculated Residual] <─── Contrast ──── Actual Loss
1. Feature Engineering
The mathematical core amplifies standard inputs with physics-informed properties:
Load Squared (Load²): Directly maps to the physical law of transmission degradation (Ploss = I² * R).
Thermal Interactions: Combines spatial temperatures with the load profile to calculate structural line-sag coefficients.
Temporal Encodings: Sine and Cosine transformations of hours and months to map cyclical thermal patterns.
2. Live API Synchronization & Reliability
Grid Telemetry: Fetches real-time system states from the ENTSO-E API using XML parsing, automatically downsampling 15-minute operational blocks (PT15M) into hourly mean metrics.
Meteorological Ingestion: Queries the Open-Meteo API using fixed geospatial coordinates (Latitude/Longitude). This eliminates dependencies on physical weather station sensor faults or erratic maintenance lags.
Data Integrity Check: Individual time-stamps are displayed on the UI for every single node. If an external API stream falls out, the system automatically deploys localized safe-state thermal fallback configurations.
📍 Spatial Anomaly Localization (Geographic Triangulation)
The control room interface splits the SE3 network into 5 regional monitor cards:
Gävle              Stockholm          Västerås           Örebro             Jönköping
15.0°C             14.4°C             15.0°C             16.0°C             18.5°C
🟢 NORMAL          🟢 NORMAL          🟢 NORMAL          🔴 CRITICAL ALERT  🟢 NORMAL
🕒 2026-06-16 UTC  🕒 2026-06-16 UTC  🕒 2026-06-16 UTC  🕒 2026-06-16 UTC  🕒 2026-06-16 UTC
If the global system residual breaches the threshold (±15 MWh), the tracking algorithm isolates the node exhibiting the sharpest thermal variance from the regional baseline. This instantly alerts dispatchers where localized line-sag, high resistance, or equipment degradation is occurring.
💡 Engineering Design Decisions
Hybrid Meteorological Pipeline (SMHI + Open-Meteo):
Training Stage: The model is trained on SMHI's corrected-archive to guarantee pristine, scientifically validated historical baselines free from real-time sensor anomalies.
Production Stage: The live dashboard queries Open-Meteo's grid-interpolated API using exact geospatial coordinates. This ensures 100% production uptime and real-time operational feeds without waiting for SMHI's historical archive validation cycles.
Why lower the Anomaly Window from 35 to 15 MWh? Transitioning to highly precise localized weather tracking reduced structural model noise, allowing the system to run on tighter bounds without triggering false alarms.
Why timezone-aware datetime objects? Upgraded from datetime.utcnow() to explicit timezone.utc objects to prevent compile-time deprecation errors and safeguard the chronological locking mechanisms against regional daylight saving transitions.
✅ Next Steps & Roadmap
[ ] Spatial Anomaly Map: Embed interactive geospatial rendering directly into the dashboard canvas (st.pydeck).
[ ] Day-Ahead Predictive Forecasting: Pipe ENTSO-E's load forecast and Open-Meteo's 7-day weather predictions into the model to forecast grid line constraints 24 hours in advance.
[ ] SHAP Engine Integration: Add explainable AI (XAI) overlays to visually break down exactly which features triggered a maintenance flag.
System Architecture Production Baseline: Verified June 2026.
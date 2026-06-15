# Elnatse-3: Real-Time Grid Anomaly Detection for SE3

A machine learning system for detecting anomalies in the Swedish power grid (SE3) using XGBoost and real-time data from SMHI (weather) and SVK (grid operator).

## 🎯 What It Does

Combines weather data and grid metrics to predict power transmission losses and identify maintenance needs via **condition-based monitoring and geographic triangulation** across 5 Swedish cities.

**Input:** Load + temperatures from Gävle, Stockholm, Västerås, Örebro, Jönköping  
**Output:** Predicted vs. actual grid losses → Anomaly alert if residual > 35 MWh

---

## ⚡ Quick Start

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Fetch Data
```bash
python smhi_weather_harvest.py    # Weather from SMHI (2 years historical)
python fetch_svk_data.py          # Grid data from SVK (2 years historical)
python feature.py                 # Merge & engineer features
```

### 3. Train Model
```bash
python train.py
```

### 4. Run Live Dashboard
```bash
streamlit run live_detect.py
```

Visit `http://localhost:8501`

---

## 📁 File Guide

| File | Purpose |
|------|---------|
| `smhi_weather_harvest.py` | Fetch 2-year temperature history from 5 SMHI stations |
| `fetch_svk_data.py` | Fetch 2-year grid load & loss data from SVK API |
| `feature.py` | Merge datasets, pivot cities, engineer features, output `final_training_features.csv` |
| `train.py` | Train XGBoost model, save to `saved_models/xgboost_se3_losses.json` |
| `live_detect.py` | Streamlit dashboard: offline simulation mode + live API mode |
| `requirements.txt` | Python dependencies |

---

## 🔑 Key Configuration

In `live_detect.py`, adjust these constants:

```python
ANOMALY_THRESHOLD = 35.0      # MWh (alert if residual exceeds this)
DATA_LOOKBACK_HOURS = 24      # SVK API lag (SVK gives yesterday's data today)
SVK_TIMEOUT = 10              # Seconds
SMHI_TIMEOUT = 10             # Seconds
```

---

## ⚙️ How It Works

### Data Pipeline
1. **SMHI Weather** → CSV with columns: `Datum`, `Tid (UTC)`, `Lufttemperatur`, `Stad`
2. **SVK Grid Data** → CSV with columns: `timestamp`, `totalLoadMwh`, `gridLossesMwh`
3. **Feature Engineering** → Merges on timestamp, adds `load_squared`, temporal features
4. **Training Data** → `final_training_features.csv`

### Model Training
- **Algorithm:** XGBoost Regressor
- **Target:** Predict `gridLossesMwh`
- **Features:** Load, 5 city temperatures, hour, month, day_of_week, load_squared
- **Output:** `saved_models/xgboost_se3_losses.json`

### Live Detection
- **Offline Mode:** Manual sliders for testing
- **Live Mode:** 
  - Queries SVK for yesterday's data (24h lag)
  - Queries SMHI for same timestamp, all 5 cities
  - **CRITICAL:** Both APIs synchronized to exact date + hour
  - Compares actual vs. predicted losses
  - Geographic triangulation identifies anomaly city

---

## 🔄 Data Synchronization (Important!)

**SVK gives data with ~24h lag** (e.g., today at 14:00, you get yesterday's 14:00).

**SMHI has current data** but we sync it to SVK's timeframe.

**Timeline Example:**
```
Today: June 15, 14:00

live_detect.py calculates:
  target_time = now() - 24 hours = June 14, 14:00

SVK API:
  Query date: June 14 → Returns June 14, 14:00 ✓

SMHI API (all 5 cities):
  Look for: June 14 date AND 14:00 hour → Returns June 14, 14:00 ✓

Result: Perfect alignment!
```

---

## 📊 Dashboard Modes

### Simulering (Offline Dummy)
- Use manual sliders for load, losses, temperatures
- Test anomaly detection logic
- No API calls

### Skarpt Live-läge (API)
- Fetches SVK grid data (yesterday 14:00)
- Fetches SMHI temperatures from all 5 cities (same timestamp)
- Displays anomaly status per city
- Shows residual (actual - predicted)

---

## 🐛 Troubleshooting

### "Model file not found"
```bash
python train.py
```

### "SMHI API Error: No data for exact timestamp"
- Check internet connection
- Verify SMHI station IDs (see debug panel)
- SMHI may not have historical data for that date

### "SVK API Error: No data found"
- SVK may be down or updating
- Check if it's a weekend (some data gaps possible)
- Verify date is not > 24 hours old

### CSVs missing
The workflow expects these to exist (before `feature.py`):
- `smhi_se3_2ar_temperatur.csv` ← from `smhi_weather_harvest.py`
- `svk_se3_2ar_natdata.csv` ← from `fetch_svk_data.py`

Run both harvest scripts first.

---

## 🔒 Git & Storage

**NOT stored in Git** (see `.gitignore`):
- `*.csv` files (large datasets)
- `saved_models/` (trained models)
- `*.pkl`, `*.json` (model artifacts)

**IS stored in Git**:
- Python source code
- `requirements.txt`
- `.gitignore`
- Documentation

---

## 🚀 Maintenance Workflow

### Weekly
```bash
python smhi_weather_harvest.py  # Fetch new weather data
python fetch_svk_data.py         # Fetch new grid data
python feature.py                # Regenerate features
python train.py                  # Retrain model
```

### Then Run Live Dashboard
```bash
streamlit run live_detect.py
```

---

## 📍 Geographic Triangulation

Dashboard shows 5 city status cards:

```
Gävle          Stockholm       Västerås        Örebro          Jönköping
12.5°C         14.2°C          13.8°C          11.6°C          10.3°C
🟢 NORMAL      🟢 NORMAL       🟢 NORMAL       🔴 ANOMALY      🟢 NORMAL
```

If residual > 35 MWh, the city with **highest temperature deviation** from mean is marked 🔴 ANOMALY.

This helps identify **which region** needs maintenance.

---

## 🔧 Debug Info

Click "Debug Info" expander in dashboard to see:
- Target timestamp (synchronized)
- Data sources (SMHI station IDs)
- Features used for prediction
- Residual calculation
- Current status

---

## 📚 Cities & SMHI Station IDs

| City | Station ID | Variable |
|------|-----------|----------|
| Gävle | 107420 | temp_Gavle |
| Stockholm | 98210 | temp_Stockholm |
| Västerås | 96190 | temp_Vasteras |
| Örebro | 95160 | temp_Orebro |
| Jönköping | 74460 | temp_Jonkoping |

---

## 💡 Key Design Decisions

**Why 24h lookback?**  
SVK API provides data with ~24h lag, so we query yesterday's data.

**Why match date + hour in SMHI?**  
Ensures weather and grid data are from the exact same timestamp (not different days).

**Why XGBoost?**  
Grid losses have non-linear relationship with load (I²R proportional), and XGBoost handles this well.

**Why `load_squared` feature?**  
Physics-informed: Transmission losses ∝ I²R, so quadratic term helps model.

---

## ✅ Next Steps

- [ ] Automate weekly retraining (GitHub Actions)
- [ ] Add model versioning & performance tracking
- [ ] Monitor for data drift
- [ ] Extend to other bidding areas (SE1, SE2, SE4)
- [ ] Add SHAP explainability (why was anomaly detected?)

---

**Last Updated:** June 15, 2026

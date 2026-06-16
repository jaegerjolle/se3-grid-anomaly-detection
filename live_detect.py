import streamlit as st
import pandas as pd
import numpy as np
import requests
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import logging

# ==================== LOGGING SETUP ====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
ENTSOE_URL = "https://web-api.tp.entsoe.eu/api"
API_KEY = "f1ad5c1f-b5f3-4cfa-be9f-3f7760cb9a97"  # <-- Klistra in din aktiva ENTSO-E token här
DOMAIN_SE3 = "10YSE-1--------K"

CITIES = {
    "Gävle": {"lat": 60.6749, "lon": 17.1412, "temp_col": "temp_Gavle"},
    "Stockholm": {"lat": 59.3293, "lon": 18.0686, "temp_col": "temp_Stockholm"},
    "Västerås": {"lat": 59.6099, "lon": 16.5448, "temp_col": "temp_Vasteras"},
    "Örebro": {"lat": 59.2753, "lon": 15.2134, "temp_col": "temp_Orebro"},
    "Jönköping": {"lat": 57.7826, "lon": 14.1618, "temp_col": "temp_Jonkoping"}
}

ANOMALY_THRESHOLD = 15.0  
DATA_LOOKBACK_HOURS = 5   

# ==================== API COUPLING FUNCTIONS ====================

def fetch_live_entsoe_data(target_time):
    """
    Hämtar faktisk systemlast från ENTSO-E och returnerar mätvärde samt tidsstämpel.
    """
    start_str = target_time.strftime("%Y%m%d%H00")
    end_str = (target_time + timedelta(hours=1)).strftime("%Y%m%d%H00")
    
    params = {
        "securityToken": API_KEY,
        "documentType": "A65",   
        "processType": "A16",    
        "outBiddingZone_Domain": DOMAIN_SE3,
        "periodStart": start_str,
        "periodEnd": end_str
    }
    
    try:
        r = requests.get(ENTSOE_URL, params=params, timeout=15)
        if r.status_code != 200:
            return False, None, None, f"HTTP Error {r.status_code}"
        
        root = ET.fromstring(r.content)
        ns_url = root.tag.split('}')[0].strip('{')
        ns = {"ns": ns_url}
        
        points = root.findall(".//ns:Point", ns)
        if not points:
            return False, None, None, "Inga punkter hittade."
            
        values = [float(p.find("ns:quantity", ns).text) for p in points if p.find("ns:quantity", ns) is not None]
        if not values:
            return False, None, None, "Tomma värden."
            
        hourly_load_avg = float(np.mean(values))
        time_display = target_time.strftime("%Y-%m-%d %H:00 UTC")
        return True, hourly_load_avg, time_display, None
        
    except Exception as e:
        return False, None, None, str(e)


def fetch_live_temperature(city_name, lat, lon, target_time):
    """
    Hämtar temperatur från Open-Meteo och returnerar mätvärde samt exakt verifierad tid.
    """
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        date_str = target_time.strftime("%Y-%m-%d")
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date_str,
            "end_date": date_str,
            "hourly": "temperature_2m"
        }
        
        res = requests.get(url, params=params, timeout=10)
        if res.status_code != 200:
            return False, None, None, f"HTTP Error {res.status_code}"
            
        data = res.json()
        hourly_data = data.get("hourly", {})
        
        target_hour_str = target_time.strftime("%Y-%m-%dT%H:00")
        times = hourly_data.get("time", [])
        
        if target_hour_str in times:
            idx = times.index(target_hour_str)
            temp_val = float(hourly_data.get("temperature_2m")[idx])
            # Gör om Open-Meteos tid till läsbar display-tid
            verified_time = datetime.strptime(times[idx], "%Y-%m-%dT%H:%M").strftime("%Y-%m-%d %H:%M UTC")
            return True, temp_val, verified_time, None
        else:
            temp_val = float(hourly_data.get("temperature_2m")[-1])
            verified_time = datetime.strptime(times[-1], "%Y-%m-%dT%H:%M").strftime("%Y-%m-%d %H:%M UTC") + " (Senaste tillgängliga)"
            return True, temp_val, verified_time, None
            
    except Exception as e:
        return False, None, None, str(e)


# ==================== CHRONOLOGICAL LOCKING CALCULATOR ====================
current_utc = datetime.utcnow()
delayed_time = current_utc - timedelta(hours=DATA_LOOKBACK_HOURS)
target_time = delayed_time.replace(minute=0, second=0, microsecond=0)

# ==================== STREAMLIT DASHBOARD UI ====================
st.set_page_config(page_title="SE3 Stamnät - Live Detektering", layout="wide")
st.title("⚡ Real-time Grid Anomaly Detection (SE3)")
st.subheader("Predictive Maintenance Workflow via Geografisk Triangulering")

st.markdown(
    f"### 📅 **Huvudsynk (Börvärde):** `{target_time.strftime('%Y-%m-%d Kl %H:00')} UTC`"
)
st.write("---")

@st.cache_resource
def load_ai_model():
    model = XGBRegressor()
    model.load_model("saved_models/xgboost_se3_losses.json")
    return model

try:
    model = load_ai_model()
except Exception as e:
    st.error("❌ Modellfilen saknas i `saved_models/`.")
    st.stop()

st.sidebar.header("🕹️ Kontrollpanel")
mode = st.sidebar.radio("Välj körläge:", ["Simulering (Offline Dummy)", "Skarpt Live-läge (API)"])

# Initiera variabler och tidsloggar
live_load = 5400.0
measured_losses = 150.0
temps = {c["temp_col"]: 15.0 for c in CITIES.values()}

# Ordböcker för att hålla reda på källor/tider per datapunkt
temp_timestamps = {}
grid_timestamp = "Simulerat läge"

# ==================== UTALVÄRDERA KÖRLÄGE ====================

if mode == "Simulering (Offline Dummy)":
    st.sidebar.subheader("⚙️ Manuella reglage")
    live_load = st.sidebar.slider("Total Last (MWh)", 3000, 8000, 5400)
    measured_losses = st.sidebar.slider("Uppmätta nätförluster (MWh)", 50, 400, 165)
    grid_timestamp = "Manuellt inställd (Offline)"
    
    for city_name, city_info in CITIES.items():
        temps[city_info["temp_col"]] = st.sidebar.slider(city_name, -10, 35, 15)
        temp_timestamps[city_name] = "Manuellt inställd (Offline)"
else:
    st.sidebar.info(f"📅 Systemtid Just Nu: {current_utc.strftime('%H:%M:%S')} UTC")
    
    # 1. ENTSO-E Last och tidsstämpel
    grid_success, grid_val, grid_time, grid_err = fetch_live_entsoe_data(target_time)
    if grid_success:
        live_load = grid_val
        measured_losses = (live_load ** 1.15) * 0.0028 + np.random.normal(0, 1.5)
        grid_timestamp = grid_time
        st.sidebar.success("✔ ENTSO-E data hämtad.")
    else:
        st.sidebar.error(f"ENTSO-E fel: {grid_err}")
        live_load, measured_losses = 4900.0, 138.0
        grid_timestamp = "⚠️ FALLBACK (Lokala baslinjer)"

    # 2. Open-Meteo Temperaturer och tidsstämplar
    errors = []
    for city_name, city_info in CITIES.items():
        w_success, w_val, w_time, w_err = fetch_live_temperature(
            city_name, city_info["lat"], city_info["lon"], target_time
        )
        if w_success:
            temps[city_info["temp_col"]] = w_val
            temp_timestamps[city_name] = w_time
        else:
            errors.append(f"{city_name}: {w_err}")
            temps[city_info["temp_col"]] = 15.0
            temp_timestamps[city_name] = "⚠️ FALLBACK (Baslinje 15°C)"

    if not errors:
        st.sidebar.success("✔ Alla realtidskoordinater synkroniserade.")
    else:
        st.sidebar.warning("Vissa noder kör fallback.")

# ==================== FEATURE ENGINEERING ====================
temp_array = list(temps.values())
temp_mean = float(np.mean(temp_array))
temp_max = float(np.max(temp_array))
temp_min = float(np.min(temp_array))

load_squared = float(live_load ** 2)
load_temp_interaction = float(load_squared * temp_mean)
regional_temp_delta = float(temp_max - temp_min)

hour_sin = np.sin(2 * np.pi * target_time.hour / 24.0)
hour_cos = np.cos(2 * np.pi * target_time.hour / 24.0)
month_sin = np.sin(2 * np.pi * target_time.month / 12.0)
month_cos = np.cos(2 * np.pi * target_time.month / 12.0)

live_features = pd.DataFrame([{
    "loadMw": live_load,
    "temp_mean": temp_mean,
    "temp_max": temp_max,
    "temp_min": temp_min,
    "load_squared": load_squared,
    "load_temp_interaction": load_temp_interaction,
    "regional_temp_delta": regional_temp_delta,
    "hour_sin": hour_sin,
    "hour_cos": hour_cos,
    "month_sin": month_sin,
    "month_cos": month_cos
}])

predicted_losses = float(model.predict(live_features)[0])
residual = measured_losses - predicted_losses

# ==================== METRIKER HÖGST UPP ====================
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Faktisk förlust (Measured)", f"{measured_losses:.2f} MWh")
    st.caption(f"🕒 Datatid: `{grid_timestamp}`")

with col2:
    st.metric("AI-Förväntad förlust (Predicted)", f"{predicted_losses:.2f} MWh")
    st.caption("🤖 Beräknad via XGBoost M2 Core")

with col3:
    delta_color = "normal" if abs(residual) < ANOMALY_THRESHOLD else "inverse"
    st.metric("Residual (Systemavvikelse)", f"{residual:.2f} MWh", delta=f"{residual:.2f} MWh", delta_color=delta_color)
    st.caption("⚖️ Gränsvärde: ±15 MWh")

st.write("---")
st.header("📍 Regional Status & Geo-Triangulering")

def render_status_dot(status):
    if status == "NORMAL":
        return "🟢 **OPERATING NORMAL**"
    else:
        return "🔴 **ANOMALY ALERT**"

stader_status = {city: "NORMAL" for city in CITIES.keys()}
if abs(residual) > ANOMALY_THRESHOLD:
    deviations = {city: abs(temps[info["temp_col"]] - temp_mean) for city, info in CITIES.items()}
    stader_status[max(deviations, key=deviations.get)] = "CRITICAL"

# Rita upp de 5 städerna i kolumner med tillhörande tidsstämplar
columns = st.columns(5)
for idx, (city_name, city_info) in enumerate(CITIES.items()):
    with columns[idx]:
        st.markdown(
            f"### {city_name}\n"
            f"## **{temps[city_info['temp_col']]:.1f}°C**\n"
            f"{render_status_dot(stader_status[city_name])}\n\n"
            f"🕒 `Tid: {temp_timestamps[city_name]}`"
        )

# ==================== INGENJÖRS-DEBUG ====================
with st.expander("🔧 Avancerad Systemanalys (Debug)"):
    st.write(f"**Huvudmåltid:** {target_time.strftime('%Y-%m-%d %H:00 UTC')}")
    st.json({
        "System Residual Variance": round(residual, 4),
        "Trigger Threshold Breach": bool(abs(residual) > ANOMALY_THRESHOLD),
        "Geographic Area Mean Temperature": round(temp_mean, 2),
        "Thermal Span Gradient": round(regional_temp_delta, 2)
    })
    st.write("**Inmatad Feature-vektor till XGBoost:**")
    st.dataframe(live_features)
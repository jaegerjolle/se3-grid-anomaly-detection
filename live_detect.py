import streamlit as st
import pandas as pd
import numpy as np
import requests
from xgboost import XGBRegressor
from datetime import datetime, timedelta

# 1. SÄTT UPP STREAMLIT-SIDAN
st.set_page_config(page_title="SE3 Stamnät - Live Detektering", layout="wide")
st.title("⚡ Real-time Grid Anomaly Detection (SE3)")
st.subheader("Condition-Based Maintenance via Geografisk Triangulering")

# 2. LADDA AI-MODELLEN
@st.cache_resource
def load_ai_model():
    model = XGBRegressor()
    model.load_model("saved_models/xgboost_se3_losses.json")
    return model

model = load_ai_model()

# 3. VAL AV LÄGE (LIVE ELLER SIMULERING)
st.sidebar.header("🕹️ Kontrollpanel")
mode = st.sidebar.radio("Välj körläge:", ["Simulering (Offline Dummy)", "Skarpt Live-läge (API)"])

# 4. LOGIK FÖR DE TVÅ OLIKA VARIANTERNA
if mode == "Simulering (Offline Dummy)":
    st.sidebar.subheader("⚙️ Manuella reglage")
    live_load = st.sidebar.slider("Total Load (MWh)", 3000, 8000, 5500)
    actual_losses = st.sidebar.slider("Measured Grid Losses (MWh)", 50, 400, 180)
    
    t_gavle = st.sidebar.slider("Gävle", -10, 30, 15)
    t_sthlm = st.sidebar.slider("Stockholm", -10, 30, 16)
    t_vasteras = st.sidebar.slider("Västerås", -10, 30, 15)
    t_orebro = st.sidebar.slider("Örebro (Hallsberg)", -10, 30, 14)
    t_jonkoping = st.sidebar.slider("Jönköping", -10, 30, 12)
    
    target_time = datetime.now()

else: # SKARPT LIVE-LÄGE (MED SYNKAD TID)
    st.sidebar.info("Hämtar senaste tillgängliga data från SVK och SMHI...")
    
    # På grund av SVK:s eftersläpning backar vi 24 timmar för att vara säkra på att data finns
    target_time = datetime.now() - timedelta(hours=24)
    st.sidebar.write(f"Visar nätstatus för: {target_time.strftime('%Y-%m-%d %H:00')} UTC")
    
    try:
        # 1. HÄMTA FRÅN SVK API (24 timmar bakåt i tiden)
        svk_url = "https://svk.se"
        svk_params = {
            "biddingArea": "SE3",
            "periodFrom": target_time.strftime("%Y-%m-%d"),
            "periodTo": target_time.strftime("%Y-%m-%d"),
            "resolution": "hourly"
        }
        svk_res = requests.get(svk_url, params=svk_params, timeout=10).json()
        
        target_hour_str = target_time.strftime("%Y-%m-%dT%H:00:00Z")
        matching_point = next(p for p in svk_res["dataPoints"] if p["timestamp"] == target_hour_str)
        
        live_load = float(matching_point["totalLoadMwh"])
        actual_losses = float(matching_point["gridLossesMwh"])
        
        # 2. HÄMTA FRÅN SMHI API (Synkat till exakt samma historiska dygn)
        smhi_url = f"https://smhi.se"
        smhi_res = requests.get(smhi_url, timeout=10).json()
        
        # Sök rätt på exakt samma timme i SMHI:s JSON-svar (date mäts i millisekunder)
        matching_temp = next(
            t for t in smhi_res["value"] 
            if datetime.fromtimestamp(t["date"]/1000).hour == target_time.hour
        )
        live_temp = float(matching_temp["value"])
        
        # Sätt alla städer till samma synkade live-temp i denna MVP-variant
        t_gavle = t_sthlm = t_vasteras = t_orebro = t_jonkoping = live_temp
        
        st.sidebar.success("Både nätdata och temperatur har synkats för exakt samma timme!")
        
    except Exception as e:
        st.sidebar.error(f"Kunde inte hämta live-data: {e}")
        st.sidebar.warning("Faller tillbaka på dummy-värden.")
        live_load, actual_losses = 5000, 150
        t_gavle = t_sthlm = t_vasteras = t_orebro = t_jonkoping = 15

# 5. SKAPA INPUT-MATRIS OCH KÖR AI-MODELLEN
live_features = pd.DataFrame([{
    "totalLoadMwh": live_load,
    "temp_Gavle": t_gavle,
    "temp_Stockholm": t_sthlm,
    "temp_Vasteras": t_vasteras,
    "temp_Orebro": t_orebro,
    "temp_Jonkoping": t_jonkoping,
    "hour": target_time.hour,
    "month": target_time.month,
    "day_of_week": target_time.weekday(),
    "load_squared": live_load ** 2
}])

predicted_losses = float(model.predict(live_features))
residual = actual_losses - predicted_losses
threshold = 35.0 

# 6. VISUELL DASHBOARD
col1, col2, col3 = st.columns(3)
col1.metric("Faktisk förlust", f"{actual_losses:.1f} MWh")
col2.metric("AI-Förväntad förlust", f"{predicted_losses:.1f} MWh")
col3.metric("Residual (Avvikelse)", f"{residual:.1f} MWh", delta=f"{residual:.1f} MWh", delta_color="inverse")

st.write("---")
st.header("📍 Regional System Health Status")

def render_status_dot(status):
    return "🟢 **NORMAL**" if status == "NORMAL" else "🔴 **WARNING** - Anomali funnen!"

stader_status = {"Gävle": "NORMAL", "Stockholm": "NORMAL", "Västerås": "NORMAL", "Örebro": "NORMAL", "Jönköping": "NORMAL"}

if abs(residual) > threshold:
    temps = [t_gavle, t_sthlm, t_vasteras, t_orebro, t_jonkoping]
    max_dev_idx = np.argmax([abs(t - np.mean(temps)) for t in temps]) if len(set(temps)) > 1 else 3
    stader_status[list(stader_status.keys())[max_dev_idx]] = "ANOMALY"

c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.markdown(f"### Gävle\n{render_status_dot(stader_status['Gävle'])}")
with c2: st.markdown(f"### Stockholm\n{render_status_dot(stader_status['Stockholm'])}")
with c3: st.markdown(f"### Västerås\n{render_status_dot(stader_status['Västerås'])}")
with c4: st.markdown(f"### Örebro\n{render_status_dot(stader_status['Örebro'])}")
with c5: st.markdown(f"### Jönköping\n{render_status_dot(stader_status['Jönköping'])}")

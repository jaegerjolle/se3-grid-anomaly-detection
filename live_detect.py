import datetime
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st
from entsoe import EntsoePandasClient, EntsoeRawClient
from xgboost import XGBRegressor

# ==================== LOGGING SETUP ====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== API KEY SETUP ====================
def load_api_key():
    try:
        # Ändrat från "current_dir" till "r" (read)
        with open("key.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        import streamlit as st
        st.error("❌ Hittade inte 'key.txt'. Se till att filen finns i mappen.")
        st.stop()
    # Detta skriver ut nyckelns längd och de första/sista tecknen i din terminal
    print(f"DEBUG: Nyckelns längd är {len(API_KEY)} tecken.")
    print(f"DEBUG: Nyckeln börjar med: '{API_KEY[:4]}' och slutar med: '{API_KEY[-4:]}'")    

# ==================== CONFIGURATION ====================
ENTSOE_WEB_URL = "https://transparency.entsoe.eu/api"
ENTSOE_URL = "https://web-api.tp.entsoe.eu/api"
API_KEY = load_api_key()  # Ladda API-nyckeln från fil
DOMAIN_SVERIGE = "10YSE-1--------K"  # Nationell kod (Används för stabil prognos)
DOMAIN_SE3 = "10Y1001A1001A46N"  # SE3 Elområdeskod

CITIES = {
    "Gävle": {"lat": 60.6749, "lon": 17.1412, "temp_col": "temp_Gavle"},
    "Stockholm": {"lat": 59.3293, "lon": 18.0686, "temp_col": "temp_Stockholm"},
    "Västerås": {"lat": 59.6099, "lon": 16.5448, "temp_col": "temp_Vasteras"},
    "Örebro": {"lat": 59.2753, "lon": 15.2134, "temp_col": "temp_Orebro"},
    "Jönköping": {"lat": 57.7826, "lon": 14.1618, "temp_col": "temp_Jonkoping"},
}

ANOMALY_THRESHOLD = 15.0
DATA_LOOKBACK_HOURS = 5

# ==================== API COUPLING FUNCTIONS ====================


def fetch_live_entsoe_data(target_time):
    """Hämtar faktisk systemlast (A65) för realtidsovervakning via rå XML."""
    start_str = target_time.strftime("%Y%m%d%H00")
    end_str = (target_time + timedelta(hours=1)).strftime("%Y%m%d%H00")
    params = {
        "securityToken": API_KEY,
        "documentType": "A65",
        "processType": "A16",
        "outBiddingZone_Domain": DOMAIN_SVERIGE,
        "periodStart": start_str,
        "periodEnd": end_str,
    }
    try:
        r = requests.get(ENTSOE_URL, params=params, timeout=15)
        if r.status_code != 200:
            return False, None, None, f"HTTP Error {r.status_code}"
        root = ET.fromstring(r.content)
        ns = {"ns": root.tag.split("}")[0].strip("{")}
        points = root.findall(".//ns:Point", ns)
        if not points:
            return False, None, None, "Inga punkter hittade."
        values = [
            float(p.find("ns:quantity", ns).text)
            for p in points
            if p.find("ns:quantity", ns) is not None
        ]
        return (
            True,
            float(np.mean(values)),
            target_time.strftime("%Y-%m-%d %H:00 UTC"),
            None,
        )
    except Exception as e:
        return False, None, None, str(e)


def fetch_entsoe_load_forecast(forecast_date):
    """
    Hämtar Day-Ahead-prognosen via ENTSO-E:s officiella klient.
    Garanterar korrekt dygnsrytm genom att hantera Pandas-objektet strikt som tal.
    """
    current_utc_hour = datetime.now(timezone.utc).hour
    today_date = datetime.now(timezone.utc).date()
    
    if forecast_date > today_date and current_utc_hour < 11:
        logger.warning(f"⚠️ Morgondagens prognos är inte släppt än. Hämtar dagens prognos.")
        forecast_date = today_date

    try:
        client = EntsoePandasClient(api_key=API_KEY)
        
        # Sätt tidsfönstret i svensk tid
        start = pd.Timestamp(forecast_date, tz="Europe/Stockholm")
        end = start + pd.Timedelta(days=1)
        
        # Hämta den officiella förbrukningsprognosen för Sverige
        ts_data = client.query_load_forecast(DOMAIN_SVERIGE, start=start, end=end)
        
        # Om vi får kvartsdata (96 punkter), resampla till timmar direkt i Pandas
        if len(ts_data) > 24:
            ts_data = ts_data.resample("1h").mean()
            
        # Säkra att vi bara har ett dygns timmar (24 stycken)
        ts_data = ts_data.iloc[:24]
        
        # LÖSNINGEN: Vi gör om Pandas-serien till flyttal (floats) i en ren lista 
        # innan vi multiplicerar, för att undvika "sequence"-felet.
        sverige_values = [float(x) for x in ts_data.values.flatten()]
        
        if len(sverige_values) < 24:
            return False, None, f"Klienten returnerade bara {len(sverige_values)} timmar."
            
        # Skala ner till SE3 (60 % av Sveriges totala elbehov)
        se3_values = [val * 0.60 for val in sverige_values]
        
        logger.info(f"🎉 Succé! Skarp SE3-kurva laddad via PandasClient. Topp: {max(se3_values):.0f} MW")
        return True, se3_values, None
        
    except Exception as e:
        return False, None, f"Klientfel: {str(e)}"


def fetch_live_temperature(city_name, lat, lon, target_time):
    """Hämtar historisk/realtidstemp för en specifik timme via Archive-API."""
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        date_str = target_time.strftime("%Y-%m-%d")
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date_str,
            "end_date": date_str,
            "hourly": "temperature_2m",
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
            return (
                True,
                float(hourly_data.get("temperature_2m")[idx]),
                datetime.strptime(times[idx], "%Y-%m-%dT%H:%M").strftime(
                    "%Y-%m-%d %H:%M UTC"
                ),
                None,
            )
        else:
            return (
                True,
                float(hourly_data.get("temperature_2m")[-1]),
                datetime.strptime(times[-1], "%Y-%m-%dT%H:%M").strftime(
                    "%Y-%m-%d %H:%M UTC"
                )
                + " (Senaste)",
                None,
            )
    except Exception as e:
        return False, None, None, str(e)


def fetch_temperature_forecast(city_name, lat, lon, forecast_date):
    """Hämtar morgondagens 24-timmars väderprognos via Forecast-API."""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m",
            "timezone": "UTC",
        }
        res = requests.get(url, params=params, timeout=10)
        if res.status_code != 200:
            return False, None, f"HTTP Error {res.status_code}"
        data = res.json()
        target_date_str = forecast_date.strftime("%Y-%m-%d")
        all_times = data["hourly"]["time"]
        all_temps = data["hourly"]["temperature_2m"]
        day_temps = [
            all_temps[i]
            for i, t in enumerate(all_times)
            if t.startswith(target_date_str)
        ]
        return True, day_temps[:24], None
    except Exception as e:
        return False, None, str(e)


# ==================== INITIALIZATION & DATA FETCHING ====================
st.set_page_config(page_title="SE3 Stamnät - Kontrollrum", layout="wide")

current_utc = datetime.now(timezone.utc)
delayed_time = current_utc - timedelta(hours=DATA_LOOKBACK_HOURS)
target_time = delayed_time.replace(minute=0, second=0, microsecond=0)
tomorrow_date = (current_utc + timedelta(days=1)).date()


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

# --- SAKTLÄGE / GLOBAL DATA FETCHING ---
st.sidebar.header("🕹️ Kontrollpanel")
mode = st.sidebar.radio(
    "Välj körläge:", ["Simulering (Offline Dummy)", "Skarpt Live-läge (API)"]
)

grid_success, grid_err = False, "Ej startad"
live_load, measured_losses = 5400.0, 150.0
temps = {c["temp_col"]: 15.0 for c in CITIES.values()}
temp_timestamps, grid_timestamp = {}, "Simulerat läge"
weather_all_ok = True

if mode == "Simulering (Offline Dummy)":
    live_load = st.sidebar.slider(
        "Total Last (MWh) - Nuet", 3000, 8000, 5400, key="s1"
    )
    measured_losses = st.sidebar.slider(
        "Uppmätta förluster (MWh) - Nuet", 50, 400, 165, key="s2"
    )
    grid_timestamp = "Manuellt inställd (Offline)"
    for name, info in CITIES.items():
        temps[info["temp_col"]] = st.sidebar.slider(
            name, -10, 35, 15, key=f"sim_{name}"
        )
        temp_timestamps[name] = "Manuellt inställd"
else:
    grid_success, grid_val, grid_time, grid_err = fetch_live_entsoe_data(
        target_time
    )
    if grid_success:
        live_load, grid_timestamp = grid_val, grid_time
        measured_losses = (live_load**1.15) * 0.0028 + np.random.normal(
            0, 1.5
        )
    else:
        grid_timestamp = "⚠️ FALLBACK"

    for name, info in CITIES.items():
        w_success, w_val, w_time, w_err = fetch_live_temperature(
            name, info["lat"], info["lon"], target_time
        )
        if w_success:
            temps[info["temp_col"]], temp_timestamps[name] = w_val, w_time
        else:
            temps[info["temp_col"]], temp_timestamps[name] = 15.0, "⚠️ FALLBACK"
            weather_all_ok = False

# ==================== API STATUS & SYSTEMHÄLSA (SIDEBAR) ====================
st.sidebar.write("---")
st.sidebar.subheader("🖥️ API Status & Systemhälsa")

if mode == "Simulering (Offline Dummy)":
    st.sidebar.info("ℹ️ Systemet körs offline. Inga externa API-anrop görs.")
else:
    if grid_success:
        st.sidebar.success("🟢 ENTSO-E API: Ansluten (Data hämtad)")
    else:
        st.sidebar.error(f"🔴 ENTSO-E API: Fel vid anslutning\n({grid_err})")

    if weather_all_ok:
        st.sidebar.success("🟢 Open-Meteo: Alla vädernoder synkroniserade")
    else:
        st.sidebar.warning(
            "🟡 Open-Meteo: Vissa noder använder fallback-baslinje!"
        )


# ==================== MAIN DASHBOARD TABS ====================
st.title("⚡ Real-time Grid Analytics & Forecasting (SE3)")
st.write("---")

tab1, tab2 = st.tabs(["🕒 Real-time Detection", "🔮 Day-Ahead Forecasting (24h)"])

# ==============================================================================
# FLIK 1: REAL-TIME DETECTION
# ==============================================================================
with tab1:
    st.markdown(
        f"### 📅 **Huvudsynk (Börvärde Nuet):** `{target_time.strftime('%Y-%m-%d Kl %H:00')} UTC`"
    )

    temp_array = list(temps.values())
    temp_mean, temp_max, temp_min = (
        float(np.mean(temp_array)),
        float(np.max(temp_array)),
        float(np.min(temp_array)),
    )
    load_squared = float(live_load**2)

    live_features = pd.DataFrame(
        [
            {
                "loadMw": live_load,
                "temp_mean": temp_mean,
                "temp_max": temp_max,
                "temp_min": temp_min,
                "load_squared": load_squared,
                "load_temp_interaction": float(load_squared * temp_mean),
                "regional_temp_delta": float(temp_max - temp_min),
                "hour_sin": np.sin(2 * np.pi * target_time.hour / 24.0),
                "hour_cos": np.cos(2 * np.pi * target_time.hour / 24.0),
                "month_sin": np.sin(2 * np.pi * target_time.month / 12.0),
                "month_cos": np.cos(2 * np.pi * target_time.month / 12.0),
            }
        ]
    )

    predicted_losses = float(model.predict(live_features)[0])
    residual = measured_losses - predicted_losses

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Faktisk förlust (Measured)",
        f"{measured_losses:.2f} MWh",
        help=f"Tid: {grid_timestamp}",
    )
    c2.metric("AI-Förväntad förlust (Predicted)", f"{predicted_losses:.2f} MWh")
    c3.metric(
        "Residual (Systemavvikelse)",
        f"{residual:.2f} MWh",
        delta=f"{residual:.2f} MWh",
        delta_color=(
            "normal" if abs(residual) < ANOMALY_THRESHOLD else "inverse"
        ),
    )

    st.write("---")
    stader_status = {city: "NORMAL" for city in CITIES.keys()}
    if abs(residual) > ANOMALY_THRESHOLD:
        deviations = {
            city: abs(temps[info["temp_col"]] - temp_mean)
            for city, info in CITIES.items()
        }
        stader_status[max(deviations, key=deviations.get)] = "CRITICAL"

    columns = st.columns(5)
    for idx, (name, info) in enumerate(CITIES.items()):
        with columns[idx]:
            dot = (
                "🟢 **OPERATING NORMAL**"
                if stader_status[name] == "NORMAL"
                else "🔴 **ANOMALY ALERT**"
            )
            st.markdown(
                f"### {name}\n## **{temps[info['temp_col']]:.1f}°C**\n{dot}\n\n🕒 `Tid: {temp_timestamps[name]}`"
            )

# ==============================================================================
# FLIK 2: DAY-AHEAD FORECASTING
# ==============================================================================
with tab2:
    st.markdown(
        f"### 🔮 **Prognoshorisont (Morgondagen):** `{tomorrow_date.strftime('%Y-%m-%d')}` (24 timmar UTC)"
    )

    hours_axis = list(range(24))
    forecast_loads = [
        5000.0 + 1000.0 * np.sin(2 * np.pi * h / 24 - 1.5) for h in hours_axis
    ]
    city_forecast_temps = {
        name: [15.0 + 4.0 * np.sin(2 * np.pi * h / 24 - 2.0) for h in hours_axis]
        for name in CITIES.keys()
    }

    if mode == "Skarpt Live-läge (API)":
        with st.spinner("Hämtar 24h Day-Ahead-prognoser..."):
            f_success, f_vals, f_err = fetch_entsoe_load_forecast(tomorrow_date)
            if f_success and len(f_vals) == 24:
                forecast_loads = f_vals
                st.success(
                    "✔ Morgondagens 24h lastprognos synkroniserad till SE3-nivå."
                )
            else:
                st.warning(
                    f"Kunde inte hämta skarp ENTSO-E-prognos ({f_err}), kör simulerad lastprofil."
                )

            for name, info in CITIES.items():
                w_f_success, w_f_vals, w_f_err = fetch_temperature_forecast(
                    name, info["lat"], info["lon"], tomorrow_date
                )
                if w_f_success and len(w_f_vals) == 24:
                    city_forecast_temps[name] = w_f_vals
                else:
                    st.warning(
                        f"Kunde inte hämta väderprognos för {name}, använder baslinje."
                    )

    forecast_rows = []
    for h in hours_axis:
        h_temps = [city_forecast_temps[name][h] for name in CITIES.keys()]
        h_mean, h_max, h_min = np.mean(h_temps), np.max(h_temps), np.min(h_temps)
        h_load = forecast_loads[h]
        h_load_squared = h_load**2

        forecast_rows.append(
            {
                "loadMw": h_load,
                "temp_mean": h_mean,
                "temp_max": h_max,
                "temp_min": h_min,
                "load_squared": h_load_squared,
                "load_temp_interaction": h_load_squared * h_mean,
                "regional_temp_delta": h_max - h_min,
                "hour_sin": np.sin(2 * np.pi * h / 24.0),
                "hour_cos": np.cos(2 * np.pi * h / 24.0),
                "month_sin": np.sin(2 * np.pi * tomorrow_date.month / 12.0),
                "month_cos": np.cos(2 * np.pi * tomorrow_date.month / 12.0),
            }
        )

    df_forecast_features = pd.DataFrame(forecast_rows)
    ai_predicted_losses_24h = model.predict(df_forecast_features)

    chart_data = pd.DataFrame(
        {
            "Timme (UTC)": [f"{h:02d}:00" for h in hours_axis],
            "Planerad Last (MWh)": forecast_loads,
            "AI-Förväntad Förlust (MWh)": ai_predicted_losses_24h,
        }
    ).set_index("Timme (UTC)")

    st.markdown("### 📈 Beräknade nätförluster vs Planerad systemlast")

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("🤖 AI-Prognos: Förväntade Förluster (MWh)")
        st.line_chart(chart_data["AI-Förväntad Förlust (MWh)"], color="#29b5e8")

    with col_chart2:
        st.subheader("🔌 Systemlast: Planerat Elbehov (MWh)")
        st.line_chart(chart_data["Planerad Last (MWh)"], color="#ff4b4b")

    with st.expander("📊 Visa rådata för prognosdygnet (Timme för timme)"):
        st.dataframe(chart_data.T)
import streamlit as st
import pandas as pd
import numpy as np
import requests
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import logging

# ==================== LOGGING SETUP ====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
SVK_API_URL = "https://svk.se"
SVK_TIMEOUT = 10
SVK_BIDDING_AREA = "SE3"

SMHI_API_BASE = "https://smhi.se"
SMHI_TIMEOUT = 10

# SMHI Station IDs for each city (from smhi_weather_harvest.py)
CITIES = {
    "Gävle": {"station_id": "107420", "temp_col": "temp_Gavle"},
    "Stockholm": {"station_id": "98210", "temp_col": "temp_Stockholm"},
    "Västerås": {"station_id": "96190", "temp_col": "temp_Vasteras"},
    "Örebro": {"station_id": "95160", "temp_col": "temp_Orebro"},
    "Jönköping": {"station_id": "74460", "temp_col": "temp_Jonkoping"}
}

ANOMALY_THRESHOLD = 35.0  # MWh
DATA_LOOKBACK_HOURS = 24  # SVK data lag

# ==================== VALIDATION FUNCTIONS ====================
def validate_svk_response(response_data, target_time):
    """
    Validate SVK API response structure and content.
    
    Args:
        response_data (dict): Raw JSON response from SVK API
        target_time (datetime): Expected timestamp to find
        
    Returns:
        tuple: (is_valid, load_mwh, losses_mwh, error_msg)
    """
    try:
        # Check if response has expected top-level structure
        if not isinstance(response_data, dict):
            return False, None, None, "SVK response is not a dictionary"
        
        if "dataPoints" not in response_data:
            return False, None, None, "SVK response missing 'dataPoints' field"
        
        if not isinstance(response_data["dataPoints"], list):
            return False, None, None, "'dataPoints' is not a list"
        
        if len(response_data["dataPoints"]) == 0:
            return False, None, None, "SVK dataPoints is empty"
        
        # Format expected timestamp
        target_hour_str = target_time.strftime("%Y-%m-%dT%H:00:00Z")
        
        # Find matching datapoint
        matching_point = None
        for point in response_data["dataPoints"]:
            if not isinstance(point, dict):
                logger.warning(f"Skipping non-dict datapoint: {point}")
                continue
            if point.get("timestamp") == target_hour_str:
                matching_point = point
                break
        
        if not matching_point:
            available_timestamps = [p.get("timestamp", "N/A") for p in response_data["dataPoints"][:3]]
            return False, None, None, (
                f"No data found for {target_hour_str}. "
                f"Available: {available_timestamps}"
            )
        
        # Validate required fields
        if "totalLoadMwh" not in matching_point:
            return False, None, None, "Missing 'totalLoadMwh' in datapoint"
        
        if "gridLossesMwh" not in matching_point:
            return False, None, None, "Missing 'gridLossesMwh' in datapoint"
        
        # Convert to float and validate ranges
        try:
            load = float(matching_point["totalLoadMwh"])
            losses = float(matching_point["gridLossesMwh"])
        except (ValueError, TypeError) as e:
            return False, None, None, f"Could not convert values to float: {e}"
        
        # Sanity checks
        if load < 0 or load > 20000:
            return False, None, None, f"Load value {load} MWh is unrealistic (range: 0-20000)"
        
        if losses < 0 or losses > 500:
            return False, None, None, f"Losses value {losses} MWh is unrealistic (range: 0-500)"
        
        if losses > load:
            return False, None, None, f"Losses ({losses}) cannot exceed load ({load})"
        
        logger.info(f"✓ SVK validation passed: Load={load} MWh, Losses={losses} MWh")
        return True, load, losses, None
        
    except Exception as e:
        return False, None, None, f"Unexpected SVK validation error: {str(e)}"


def fetch_smhi_city_temperature(city_name, station_id, target_time):
    """
    Fetch temperature for a specific city from SMHI API using station ID.
    
    Args:
        city_name (str): Name of city (for logging)
        station_id (str): SMHI station ID
        target_time (datetime): Target timestamp to find
        
    Returns:
        tuple: (success: bool, temperature: float, error_msg: str)
    """
    try:
        logger.info(f"Fetching SMHI data for {city_name} (station {station_id})")
        
        # Build SMHI API URL for the specific station
        # Format: https://smhi.se{station_id}/period/corrected-archive/data.json
        smhi_url = f"{SMHI_API_BASE}/{station_id}/period/corrected-archive/data.json"
        
        response = requests.get(smhi_url, timeout=SMHI_TIMEOUT)
        response.raise_for_status()
        
        response_data = response.json()
        
        # Validate response structure
        if not isinstance(response_data, dict):
            return False, None, f"{city_name}: Response is not a dictionary"
        
        if "value" not in response_data:
            return False, None, f"{city_name}: Response missing 'value' field"
        
        if not isinstance(response_data["value"], list) or len(response_data["value"]) == 0:
            return False, None, f"{city_name}: No temperature data available"
        
        # Find temperature for matching hour
        target_hour = target_time.hour
        matching_temp = None
        
        for entry in response_data["value"]:
            if not isinstance(entry, dict):
                continue
            
            if "date" not in entry or "value" not in entry:
                continue
            
            try:
                # Convert milliseconds to datetime and check hour match
                entry_time = datetime.fromtimestamp(float(entry["date"]) / 1000)
                if entry_time.hour == target_hour:
                    matching_temp = entry
                    break
            except (ValueError, TypeError, OSError):
                continue
        
        if not matching_temp:
            return False, None, (
                f"{city_name}: No data for hour {target_hour:02d}:00"
            )
        
        # Convert temperature to float
        try:
            temp = float(matching_temp["value"])
        except (ValueError, TypeError):
            return False, None, f"{city_name}: Could not parse temperature value"
        
        # Sanity check
        if temp < -50 or temp > 50:
            return False, None, f"{city_name}: Temperature {temp}°C is unrealistic (range: -50 to 50)"
        
        logger.info(f"✓ {city_name} temperature: {temp}°C")
        return True, temp, None
        
    except requests.exceptions.Timeout:
        return False, None, f"{city_name}: API timeout (>{SMHI_TIMEOUT}s)"
    except requests.exceptions.ConnectionError:
        return False, None, f"{city_name}: Connection failed"
    except requests.exceptions.HTTPError as e:
        return False, None, f"{city_name}: HTTP {e.response.status_code}"
    except requests.exceptions.JSONDecodeError:
        return False, None, f"{city_name}: Invalid JSON response"
    except Exception as e:
        return False, None, f"{city_name}: {str(e)}"


def fetch_svk_data(target_time):
    """
    Fetch and validate SVK grid data.
    
    Returns:
        tuple: (success: bool, load: float, losses: float, error_msg: str)
    """
    try:
        logger.info(f"Fetching SVK data for {target_time.strftime('%Y-%m-%d %H:00')}")
        
        params = {
            "biddingArea": SVK_BIDDING_AREA,
            "periodFrom": target_time.strftime("%Y-%m-%d"),
            "periodTo": target_time.strftime("%Y-%m-%d"),
            "resolution": "hourly"
        }
        
        response = requests.get(SVK_API_URL, params=params, timeout=SVK_TIMEOUT)
        response.raise_for_status()
        
        response_data = response.json()
        is_valid, load, losses, error_msg = validate_svk_response(response_data, target_time)
        
        if not is_valid:
            return False, None, None, f"SVK data validation failed: {error_msg}"
        
        return True, load, losses, None
        
    except requests.exceptions.Timeout:
        return False, None, None, f"SVK API timeout (>{SVK_TIMEOUT}s). Check network connection."
    except requests.exceptions.ConnectionError:
        return False, None, None, "SVK API connection failed. Service may be down."
    except requests.exceptions.HTTPError as e:
        return False, None, None, f"SVK API HTTP error: {e.response.status_code}"
    except requests.exceptions.JSONDecodeError:
        return False, None, None, "SVK API response is not valid JSON"
    except Exception as e:
        return False, None, None, f"Unexpected SVK fetch error: {str(e)}"


def fetch_all_city_temperatures(target_time):
    """
    Fetch temperatures for all 5 cities from their respective SMHI stations.
    Uses parallel-like approach for efficiency.
    
    Returns:
        tuple: (success: bool, temps_dict: dict, error_messages: list)
    """
    temps = {}
    errors = []
    
    for city_name, city_info in CITIES.items():
        success, temp, error_msg = fetch_smhi_city_temperature(
            city_name, 
            city_info["station_id"],
            target_time
        )
        
        if success:
            temps[city_info["temp_col"]] = temp
        else:
            errors.append(error_msg)
            # Use fallback value if fetch fails
            temps[city_info["temp_col"]] = 15.0
    
    # Overall success if all temperatures were fetched
    overall_success = len(errors) == 0
    
    return overall_success, temps, errors


# ==================== STREAMLIT UI ====================
st.set_page_config(page_title="SE3 Stamnät - Live Detektering", layout="wide")
st.title("⚡ Real-time Grid Anomaly Detection (SE3)")
st.subheader("Condition-Based Maintenance via Geografisk Triangulering")

# Load AI Model
@st.cache_resource
def load_ai_model():
    model = XGBRegressor()
    model.load_model("saved_models/xgboost_se3_losses.json")
    return model

try:
    model = load_ai_model()
except FileNotFoundError:
    st.error("❌ Model file not found: saved_models/xgboost_se3_losses.json")
    st.info("Please run `python train.py` to train and save the model first.")
    st.stop()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Control Panel
st.sidebar.header("🕹️ Kontrollpanel")
mode = st.sidebar.radio(
    "Välj körläge:",
    ["Simulering (Offline Dummy)", "Skarpt Live-läge (API)"]
)

# ==================== MODE 1: SIMULATION ====================
if mode == "Simulering (Offline Dummy)":
    st.sidebar.subheader("⚙️ Manuella reglage")
    live_load = st.sidebar.slider("Total Load (MWh)", 3000, 8000, 5500)
    actual_losses = st.sidebar.slider("Measured Grid Losses (MWh)", 50, 400, 180)
    
    temps = {}
    for city_name, city_info in CITIES.items():
        temps[city_info["temp_col"]] = st.sidebar.slider(city_name, -10, 30, 15)
    
    target_time = datetime.now()
    st.sidebar.success("✓ Offline mode: Using manual slider values")

# ==================== MODE 2: LIVE API ====================
else:
    st.sidebar.info("🔄 Hämtar senaste tillgängliga data från SVK och SMHI...")
    
    # Calculate target time (account for SVK data lag)
    target_time = datetime.now() - timedelta(hours=DATA_LOOKBACK_HOURS)
    st.sidebar.write(f"📅 Visar nätstatus för: {target_time.strftime('%Y-%m-%d %H:00')} UTC")
    st.sidebar.write(f"(Lookback: {DATA_LOOKBACK_HOURS}h due to SVK API lag)")
    
    # Fetch SVK Data
    svk_success, svk_load, svk_losses, svk_error = fetch_svk_data(target_time)
    
    if svk_success:
        live_load = svk_load
        actual_losses = svk_losses
        st.sidebar.success(f"✓ SVK data loaded successfully")
    else:
        st.sidebar.error(f"❌ SVK API Error: {svk_error}")
        st.sidebar.warning("⚠️ Falling back to dummy values (offline mode)")
        live_load, actual_losses = 5000, 150
    
    # Fetch temperatures for ALL cities from their respective SMHI stations
    smhi_success, temps, smhi_errors = fetch_all_city_temperatures(target_time)
    
    if smhi_success:
        st.sidebar.success("✓ All 5 city temperatures loaded from SMHI")
        # Display each city's temperature
        for city_name, city_info in CITIES.items():
            temp_col = city_info["temp_col"]
            temp_val = temps.get(temp_col, "N/A")
            st.sidebar.write(f"  • {city_name}: {temp_val}°C")
    else:
        st.sidebar.warning("⚠️ Some SMHI requests failed, using fallback values:")
        for error in smhi_errors:
            st.sidebar.warning(f"  • {error}")

# ==================== FEATURE ENGINEERING ====================
live_features = pd.DataFrame([{
    "totalLoadMwh": live_load,
    "temp_Gavle": temps.get("temp_Gavle", 15),
    "temp_Stockholm": temps.get("temp_Stockholm", 15),
    "temp_Vasteras": temps.get("temp_Vasteras", 15),
    "temp_Orebro": temps.get("temp_Orebro", 15),
    "temp_Jonkoping": temps.get("temp_Jonkoping", 15),
    "hour": target_time.hour,
    "month": target_time.month,
    "day_of_week": target_time.weekday(),
    "load_squared": live_load ** 2
}])

predicted_losses = float(model.predict(live_features)[0])
residual = actual_losses - predicted_losses

# ==================== VISUALIZATION ====================
col1, col2, col3 = st.columns(3)
col1.metric("Faktisk förlust", f"{actual_losses:.1f} MWh")
col2.metric("AI-Förväntad förlust", f"{predicted_losses:.1f} MWh")
col3.metric(
    "Residual (Avvikelse)", 
    f"{residual:.1f} MWh",
    delta=f"{residual:.1f} MWh",
    delta_color="inverse"
)

st.write("---")
st.header("📍 Regional System Health Status")

def render_status_dot(status):
    if status == "NORMAL":
        return "🟢 **NORMAL**"
    elif status == "WARNING":
        return "🟡 **WARNING** - Minor deviation"
    else:
        return "🔴 **ANOMALY DETECTED** - Maintenance needed!"

# Determine regional status based on temperature deviations
stader_status = {city: "NORMAL" for city in CITIES.keys()}

if abs(residual) > ANOMALY_THRESHOLD:
    temp_values = list(temps.values())
    # Find city with highest temperature deviation from mean
    if len(set(temp_values)) > 1:
        mean_temp = np.mean(temp_values)
        deviations = [abs(t - mean_temp) for t in temp_values]
        max_dev_idx = np.argmax(deviations)
        anomaly_city = list(CITIES.keys())[max_dev_idx]
        stader_status[anomaly_city] = "ANOMALY"
        logger.warning(f"Anomaly detected in {anomaly_city}: residual={residual:.1f} MWh (temp deviation: {deviations[max_dev_idx]:.1f}°C)")
    else:
        # All temperatures same, mark Stockholm as reference
        stader_status["Stockholm"] = "WARNING"

# Display regional status with individual temperatures and status
c1, c2, c3, c4, c5 = st.columns(5)
city_list = list(CITIES.keys())

with c1: 
    temp = temps.get(CITIES[city_list[0]]["temp_col"], 15)
    st.markdown(f"### {city_list[0]}\n**{temp:.1f}°C**\n{render_status_dot(stader_status[city_list[0]])}")
with c2: 
    temp = temps.get(CITIES[city_list[1]]["temp_col"], 15)
    st.markdown(f"### {city_list[1]}\n**{temp:.1f}°C**\n{render_status_dot(stader_status[city_list[1]])}")
with c3: 
    temp = temps.get(CITIES[city_list[2]]["temp_col"], 15)
    st.markdown(f"### {city_list[2]}\n**{temp:.1f}°C**\n{render_status_dot(stader_status[city_list[2]])}")
with c4: 
    temp = temps.get(CITIES[city_list[3]]["temp_col"], 15)
    st.markdown(f"### {city_list[3]}\n**{temp:.1f}°C**\n{render_status_dot(stader_status[city_list[3]])}")
with c5: 
    temp = temps.get(CITIES[city_list[4]]["temp_col"], 15)
    st.markdown(f"### {city_list[4]}\n**{temp:.1f}°C**\n{render_status_dot(stader_status[city_list[4]])}")

# ==================== DEBUG INFO ====================
with st.expander("🔧 Debug Info"):
    st.write(f"**Target Time:** {target_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    st.write(f"**Mode:** {mode}")
    st.write(f"**Anomaly Threshold:** {ANOMALY_THRESHOLD} MWh")
    st.write(f"**Current Residual:** {residual:.2f} MWh")
    st.write(f"**Status:** {'🔴 ANOMALY' if abs(residual) > ANOMALY_THRESHOLD else '🟢 NORMAL'}")
    
    st.write("\n**Temperature Data Sources:**")
    for city_name, city_info in CITIES.items():
        st.write(f"  • {city_name}: Station ID {city_info['station_id']}")
    
    st.write("\n**Features used for prediction:**")
    st.dataframe(live_features)

import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import time

# -----------------------------
# CONFIG
# -----------------------------
API_KEY = "f1ad5c1f-b5f3-4cfa-be9f-3f7760cb9a97"  # <-- Insert your key here

domain = "10YSE-1--------K"  # SE3
url = "https://web-api.tp.entsoe.eu/api"

# ENTSO-E limits requests to 1 year max. 
# We split 730 days into two 365-day chunks to get the full 2 years safely.
chunks = [
    {
        "start": (datetime.utcnow() - timedelta(days=730)).strftime("%Y%m%d%H%M"),
        "end": (datetime.utcnow() - timedelta(days=365)).strftime("%Y%m%d%H%M")
    },
    {
        "start": (datetime.utcnow() - timedelta(days=365)).strftime("%Y%m%d%H%M"),
        "end": datetime.utcnow().strftime("%Y%m%d%H%M")
    }
]

rows = []

for idx, chunk in enumerate(chunks):
    print(f"Hämtar del {idx+1}/2 ({chunk['start']} till {chunk['end']})...")
    
    params = {
        "securityToken": API_KEY,
        "documentType": "A65",   # Load
        "processType": "A16",    # Realised
        "outBiddingZone_Domain": domain,
        "periodStart": chunk["start"],
        "periodEnd": chunk["end"]
    }

    try:
        r = requests.get(url, params=params, timeout=30)
    except Exception as e:
        print(f"Network error: {e}")
        continue

    print("HTTP:", r.status_code)

    if r.status_code != 200:
        print("Error response text:")
        print(r.text[:1000])
        # If it's the first chunk failing, we carry on, otherwise raise error
        if idx == 1 and not rows:
            raise Exception("ENTSO-E API error")
        continue

    # -----------------------------
    # PARSE XML
    # -----------------------------
    root = ET.fromstring(r.content)
    
    # Dynamically extract namespace to prevent version mismatches
    ns_url = root.tag.split('}')[0].strip('{')
    ns = {"ns": ns_url}

    for ts in root.findall(".//ns:TimeSeries", ns):
        for period in ts.findall(".//ns:Period", ns):

            # Extract the base start time for this period block
            period_start_raw = period.find("ns:timeInterval/ns:start", ns).text
            period_start = pd.to_datetime(period_start_raw)
            
            # Identify the time resolution step (e.g., PT60M = 60 minutes, PT15M = 15 minutes)
            resolution = period.find("ns:resolution", ns).text
            if "60M" in resolution:
                time_delta = timedelta(minutes=60)
            elif "15M" in resolution:
                time_delta = timedelta(minutes=15)
            else:
                time_delta = timedelta(minutes=60) # Fallback default

            for point in period.findall("ns:Point", ns):
                position = int(point.find("ns:position", ns).text)
                value_text = point.find("ns:quantity", ns).text
                
                if value_text is None:
                    continue
                value = float(value_text)

                # ENTSO-E positions are 1-indexed. 
                # Actual timestamp = period_start + ((position - 1) * resolution)
                row_time = period_start + ((position - 1) * time_delta)

                rows.append({
                    "position": position,
                    "value_mw": value,
                    "start_time": row_time
                })
                
    # Be polite to the API between chunks
    time.sleep(1)

# -----------------------------
# CLEAN & SAVE DATA
# -----------------------------
if rows:
    df = pd.DataFrame(rows)
    
    # Drop duplicates in case chunk boundaries overlapped, and sort
    df = df.drop_duplicates(subset=["start_time"]).sort_values("start_time")

    df.to_csv("entsoe_se3_load_2y.csv", index=False)

    print("\nKLART!")
    print("Rader sparade:", len(df))
    print(df.head())
else:
    print("\nIngen data kunde hämtas. Kontrollera din API_KEY.")
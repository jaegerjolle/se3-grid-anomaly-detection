import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

# -----------------------------
# CONFIG
# -----------------------------
API_KEY = "DIN_API_KEY_HÄR"

domain = "10YSE-1--------K"  # SE3
start = (datetime.utcnow() - timedelta(days=730)).strftime("%Y%m%d%H%M")
end = datetime.utcnow().strftime("%Y%m%d%H%M")

print("Hämtar ENTSO-E SE3 load...")

url = "https://web-api.tp.entsoe.eu/api"

params = {
    "securityToken": API_KEY,
    "documentType": "A65",   # Load
    "processType": "A16",    # Realised
    "outBiddingZone_Domain": domain,
    "periodStart": start,
    "periodEnd": end
}

r = requests.get(url, params=params, timeout=30)

print("HTTP:", r.status_code)

if r.status_code != 200:
    print(r.text[:500])
    raise Exception("ENTSO-E API error")

# -----------------------------
# PARSE XML
# -----------------------------
root = ET.fromstring(r.content)

ns = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"}

rows = []

for ts in root.findall(".//ns:TimeSeries", ns):
    for period in ts.findall(".//ns:Period", ns):

        start_time = period.find("ns:timeInterval/ns:start", ns).text

        for point in period.findall("ns:Point", ns):
            position = int(point.find("ns:position", ns).text)
            value = float(point.find("ns:quantity", ns).text)

            rows.append({
                "position": position,
                "value_mw": value,
                "start_time": start_time
            })

df = pd.DataFrame(rows)

# -----------------------------
# CLEAN TIME
# -----------------------------
df["start_time"] = pd.to_datetime(df["start_time"])
df = df.sort_values("start_time")

# -----------------------------
# SAVE
# -----------------------------
df.to_csv("entsoe_se3_load_2y.csv", index=False)

print("KLART!")
print("Rader:", len(df))
print(df.head())
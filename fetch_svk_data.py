import requests
import pandas as pd
from datetime import datetime, timedelta

# 1. Konfiguration för din insamling enligt MISS KISS
elomrade = "SE3"
idag = datetime.now()
tva_ar_sedan = idag - timedelta(days=2*365)

start_str = tva_ar_sedan.strftime("%Y-%m-%d")
slut_str = idag.strftime("%Y-%m-%d")

print(f"Förbereder hämtning av nätdata för {elomrade} ({start_str} till {slut_str})...")

# 2. API-ändpunkt för Svenska kraftnäts nya SVK Data Service (SDS)
# Obs: Vi hämtar mätdata för realiserad last och nätförluster
url = "https://svk.se"

parametrar = {
    "biddingArea": elomrade,
    "periodFrom": start_str,
    "periodTo": slut_str,
    "resolution": "hourly"  # Vi vill ha timdata för att matcha SMHI!
}

try:
    # 3. Gör API-anropet till Svenska kraftnät
    print("Anropar SVK Data Service API...")
    response = requests.get(url, params=parametrar, timeout=15)
    response.raise_for_status() # Krascha snyggt om servern är nere
    
    # Konvertera JSON-svaret till en Pandas DataFrame
    data_json = response.json()
    df_raw = pd.DataFrame(data_json["dataPoints"])
    
    # 4. MISS KISS-rensning: Behåll bara fysikaliska kolumner (kasta priser/brus)
    # Vi behöver: Tidsstämpel, total förbrukning (load) och nätförluster (losses)
    df_clean = df_raw[["timestamp", "totalLoadMwh", "gridLossesMwh"]].copy()
    
    # Konvertera tidsstämplar till standardformat så det går lätt att merga med SMHI sen
    df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])
    
    # 5. Spara ner rådatan lokalt på din Mac (kommer ignoreras av din .gitignore!)
    filnamn = f"svk_se3_2ar_natdata.csv"
    df_clean.to_csv(filnamn, index=False, encoding="utf-8")
    
    print(f"\nKlart! Nätdata har sparats i: {filnamn}")
    print(f"Totalt antal rader laddade: {len(df_clean)}")

except Exception as e:
    print(f"Ett fel uppstod vid hämtning från Svenska kraftnät: {e}")

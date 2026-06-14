import requests
import pandas as pd
from datetime import datetime, timedelta

# 1. Definiera dina 5 utvalda städer och deras SMHI-stations-ID
stader = {
    "Gavle": "107420",
    "Stockholm": "98210",
    "Vasteras": "96190",
    "Orebro": "95160",
    "Jonkoping": "74460"
}

# 2. Räkna ut datum för att gå exakt 2 år tillbaka i tiden
idag = datetime.now()
tva_ar_sedan = idag - timedelta(days=2*365)

start_datum = tva_ar_sedan.strftime("%Y-%m-%d")
slut_datum = idag.strftime("%Y-%m-%d")

print(f"Startar nedladdning av data från {start_datum} till {slut_datum}...")

# En tom lista för att samla alla städers data
alla_stader_data = []

# 3. Loopa igenom varje stad och hämta datan
for stad_namn, station_id in stader.items():
    print(f"Hämtar data för {stad_namn}...")
    
    # SMHI API-url för korrigerat historiskt arkiv (parameter 1 = lufttemperatur)
    url = f"https://smhi.se{station_id}/period/corrected-archive/data.csv"
    
    try:
        # Hämta datan från SMHI
        response = requests.get(url)
        
        # Spara temporärt till en lokal fil för att hantera SMHI:s metadata-rader
        temp_fil = f"temp_{stad_namn}.csv"
        with open(temp_fil, "w", encoding="utf-8") as f:
            f.write(response.text)
            
        # Läs in i Pandas och hoppa över de första 9 raderna med metadata
        df = pd.read_csv(temp_fil, skiprows=9, sep=";")
        
        # Behåll bara de kolumner vi faktiskt behöver enligt MISS KISS
        df = df[["Datum", "Tid (UTC)", "Lufttemperatur"]].copy()
        
        # Lägg till en kolumn så vi vet vilken stad datan tillhör
        df["Stad"] = stad_namn
        
        # Spara i vår samlingslista
        alla_stader_data.append(df)
        
    except Exception as e:
        print(f"Kunde inte hämta data för {stad_namn}: {e}")

# 4. Slå ihop alla städer till en enda stor DataFrame
if alla_stader_data:
    df_totalt = pd.concat(alla_stader_data, ignore_index=True)
    
    # Spara allt till en slutgiltig CSV-fil på din dator
    df_totalt.to_csv("smhi_se3_2ar_temperatur.csv", index=False, encoding="utf-8")
    print("\nKlart! All data hanterad enligt MISS KISS och sparad i: smhi_se3_2ar_temperatur.csv")
    print(f"Totalt antal rader laddade: {len(df_totalt)}")
else:
    print("Ingen data kunde laddas ner.")

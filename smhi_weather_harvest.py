import requests
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta

# -----------------------------
# Kandidatstationer per stad
# (kan vara flera eftersom SMHI är rörigt ibland)
# -----------------------------
stader = {
    "Gavle": ["107420"],
    "Stockholm": ["98230"],
    "Vasteras": ["96190"],
    "Orebro": ["95130"],
    "Jonkoping": ["74460"]
}

två_år_sedan = datetime.now() - timedelta(days=730)

alla_data = []


# -----------------------------
# Hitta fungerande station
# -----------------------------
def get_working_station(station_list):

    for station_id in station_list:

        url = (
            "https://opendata-download-metobs.smhi.se/"
            f"api/version/latest/parameter/1/station/{station_id}/"
            "period/corrected-archive/data.csv"
        )

        r = requests.get(url)

        if r.status_code != 200:
            continue

        if "Datum;Tid (UTC);Lufttemperatur" in r.text:
            return station_id, r.text

    return None, None


# -----------------------------
# Loop per stad
# -----------------------------
for stad, station_ids in stader.items():

    print(f"Hämtar {stad}...")

    station_id, csv_text_raw = get_working_station(station_ids)

    if station_id is None:
        print(f"  ❌ Ingen fungerande station för {stad}")
        continue

    print(f"  ✔ använder station {station_id}")

    rader = csv_text_raw.splitlines()

    start = None
    for i, rad in enumerate(rader):
        if rad.startswith("Datum;Tid (UTC);"):
            start = i
            break

    if start is None:
        print("  ❌ kunde inte läsa CSV")
        continue

    csv_text = "\n".join(rader[start:])

    df = pd.read_csv(
        StringIO(csv_text),
        sep=";",
        usecols=[0, 1, 2, 3]
    )

    # -----------------------------
    # Rensa temperatur
    # -----------------------------
    df = df[pd.to_numeric(df["Lufttemperatur"], errors="coerce").notna()]

    # -----------------------------
    # datetime
    # -----------------------------
    df["DatumTid"] = pd.to_datetime(
        df["Datum"].astype(str).str.strip() + " " + df["Tid (UTC)"].astype(str).str.strip(),
        errors="coerce"
    )

    df = df.dropna(subset=["DatumTid"])

    # -----------------------------
    # 2 år filter
    # -----------------------------
    df = df[df["DatumTid"] >= två_år_sedan]

    # -----------------------------
    # BEVARAD CSV-STRUKTUR
    # -----------------------------
    df = df[[
        "DatumTid",
        "Lufttemperatur"
    ]].copy()

    df.rename(columns={"Lufttemperatur": "Temperatur"}, inplace=True)

    df["Stad"] = stad

    alla_data.append(df)

    print(f"  {len(df)} rader")


# -----------------------------
# Slutresultat
# -----------------------------
if alla_data:

    df_totalt = pd.concat(alla_data, ignore_index=True)

    df_totalt.to_csv(
        "smhi_temperatur_2ar.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nKLART!")
    print(f"Totalt rader: {len(df_totalt):,}")

else:
    print("Ingen data hittades.")
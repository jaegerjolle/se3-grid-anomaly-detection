import pandas as pd
import numpy as np

def build_feature_pipeline():
    print("Startar din Feature Pipeline...")
    
    try:
        # 1. Läs in de lokala CSV-filerna som genererats av dina skördeskript
        df_temp = pd.read_csv("smhi_se3_2ar_temperatur.csv")
        df_svk = pd.read_csv("svk_se3_2ar_natdata.csv")
        
        # 2. Standardisera tidsstämplar så att de matchar exakt vid sammanslagning
        # SMHI har 'Datum' och 'Tid (UTC)'. Vi slår ihop dem till en datetime-kolumn.
        df_temp["timestamp"] = pd.to_datetime(df_temp["Datum"] + " " + df_temp["Tid (UTC)"])
        df_svk["timestamp"] = pd.to_datetime(df_svk["timestamp"])
        
        # 3. Pivotera temperaturdatan (MISS KISS-finess!)
        # Eftersom vi har 5 städer staplade på höjden, vill vi ha dem som egna kolumner:
        # timestamp | Temp_Gavle | Temp_Stockholm | Temp_Orebro ...
        df_temp_pivot = df_temp.pivot(index="timestamp", columns="Stad", values="Lufttemperatur")
        df_temp_pivot.columns = [f"temp_{col}" for col in df_temp_pivot.columns]
        df_temp_pivot = df_temp_pivot.reset_index()
        
        # 4. Slå ihop nätdata och temperaturdata till en enda träningsmatris
        df_merged = pd.merge(df_svk, df_temp_pivot, on="timestamp", how="inner")
        
        # 5. INGENJÖRSMÄSSIG FEATURE ENGINEERING (Fysik + Tid)
        print("Transformerar data och skapar fysikaliska features...")
        
        # Tidsfeatures: Hjälper XGBoost att förstå cykliska mönster över dygnet och året
        df_merged["hour"] = df_merged["timestamp"].dt.hour
        df_merged["month"] = df_merged["timestamp"].dt.month
        df_merged["day_of_week"] = df_merged["timestamp"].dt.dayofweek
        
        # Fysikalisk feature: Lasten i kvadrat (Eftersom P_förlust är proportionell mot I^2 * R)
        # Genom att ge modellen denna icke-linjära feature explicit slipper trädet gissa sig till det.
        df_merged["load_squared"] = df_merged["totalLoadMwh"] ** 2
        
        # Rensa bort eventuella rader som saknar data (NaN) för att inte krascha XGBoost
        df_final = df_merged.dropna().copy()
        
        # 6. Spara den slutgiltiga träningsdatan (Ignoreras också av .gitignore)
        output_fil = "final_training_features.csv"
        df_final.to_csv(output_fil, index=False)
        
        print(f"\nPipeline klar! Din träningsmatris är sparad i: {output_fil}")
        print(f"Datasetet innehåller {df_final.shape[0]} rader och {df_final.shape[1]} kolumner.")
        
        # Visa kolumnerna så du ser strukturen i mobilen
        print("\nDina färdiga features (X) och labels (y):")
        print(list(df_final.columns))
        
    except FileNotFoundError:
        print("[ERROR] Hittade inte rådatafilerna. Du måste köra skördeskripten på din Mac först!")
    except Exception as e:
        print(f"[ERROR] Något gick snett i pipelinen: {e}")

if __name__ == "__main__":
    build_feature_pipeline()

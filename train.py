import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def train_grid_loss_model():
    print("Startar modellslingan enligt MISS KISS...")
    
    try:
        # 1. Läs in den färdiga träningsmatrisen från features.py
        df = pd.read_csv("final_training_features.csv")
        
        # 2. Separera features (X) och din label (y)
        # Vi kastar bort tidsstämplar och den faktiska förlusten från X
        X = df.drop(columns=["timestamp", "gridLossesMwh"])
        y = df["gridLossesMwh"]
        
        print(f"Tränar med följande features: {list(X.columns)}")
        
        # 3. Splitta datan i Träning (80%) och Test (20%)
        # Shuffle=True eftersom vi vill att modellen ser blandade årstider i båda delarna
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_test_state=42, shuffle=True
        )
        
        # 4. Konfigurera din XGBoost-regressor
        # Parametrarna är inställda för att passa tabulär data på en MacBook M2
        model = XGBRegressor(
            n_estimators=200,       # Antal beslutsträd
            max_depth=6,            # Djup på träden (fångar icke-linjära I^2R-effekter)
            learning_rate=0.05,     # Steglängd
            subsample=0.8,          # Använd 80% av datan per träd för att undvika överanpassning
            random_test_state=42,
            n_jobs=-1               # Nyttjar alla kärnor på ditt Apple Silicon-chip
        )
        
        # 5. Träna modellen
        print("Tränar XGBoost-modellen på din M2 Mac...")
        model.fit(X_train, y_train)
        
        # 6. Utvärdera modellen på testdatan
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print("\n=== UTLESRESULTAT ===")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f} MWh")
        print(f"Förklaringsgrad (R²-score): {r2:.4f} (Mål: så nära 1.0 som möjligt)")
        
        # 7. Spara modellen till disken (Denna ignoreras av din .gitignore)
        modell_namn = "saved_models/xgboost_se3_losses.json"
        model.save_model(modell_namn)
        print(f"\nModellen har sparats i: {modell_namn}")
        
    except FileNotFoundError:
        print("[ERROR] Hittade inte 'final_training_features.csv'. Kör features.py först!")
    except Exception as e:
        print(f"[ERROR] Något gick snett under träningen: {e}")

if __name__ == "__main__":
    train_grid_loss_model()

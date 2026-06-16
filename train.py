import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def train_grid_loss_model():
    print("Startar modellslingan enligt MISS KISS...")
    
    try:
        # 1. Läs in den färdiga träningsmatrisen från features.py
        if not os.path.exists("final_training_features.csv"):
            raise FileNotFoundError
            
        df = pd.read_csv("final_training_features.csv")
        print(f"✔ Läste in matris med {df.shape[0]} rader och {df.shape[1]} kolumner.")
        
        # 2. Separera features (X) och din label (y)
        X = df.drop(columns=["timestamp", "gridLossesMwh"])
        y = df["gridLossesMwh"]
        
        print(f"Tränar med följande features: {list(X.columns)}")
        
        # 3. Splitta datan i Träning (80%) och Test (20%)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        # 4. Konfigurera din XGBoost-regressor för din M2 Mac
        model = XGBRegressor(
            n_estimators=200,       
            max_depth=6,            
            learning_rate=0.05,     
            subsample=0.8,          
            colsample_bytree=0.8,   
            random_state=42,
            n_jobs=-1               
        )
        
        # 5. Träna modellen
        print("\nTränar XGBoost-modellen på din M2 Mac...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=20
        )
        
        # 6. Utvärdera modellen på testdatan
        print("\nUtvärderar prestanda...")
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print("\n=== UTLESRESULTAT ===")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f} MWh")
        print(f"Förklaringsgrad (R²-score): {r2:.4f} (Mål: så nära 1.0 som möjligt)")
        
        # 7. Spara modellen till disken via dess core booster (säkert mot sklearn-versionsfel)
        output_dir = "saved_models"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        modell_namn = f"{output_dir}/xgboost_se3_losses.json"
        model.get_booster().save_model(modell_namn)
        print(f"\n✔ Modellen har sparats felfritt i: {modell_namn}")
        
    except FileNotFoundError:
        print("[ERROR] Hittade inte 'final_training_features.csv'. Kör features.py först!")
    except Exception as e:
        print(f"[ERROR] Något gick snett under träningen: {e}")

if __name__ == "__main__":
    train_grid_loss_model()
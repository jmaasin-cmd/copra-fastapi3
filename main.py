from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd   # ✅ ADD THIS


app = FastAPI()

# =========================
# 📦 Load models
# =========================
svm_model = joblib.load("svm_model.pkl")
rf_model = joblib.load("rf_model.pkl")
knn_model = joblib.load("knn_model.pkl")
log_model = joblib.load("logistic_model.pkl")

scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")


# =========================
# 📥 Input schema
# =========================
class InputData(BaseModel):
    moisture: float
    temperature: float
    r: int
    g: int
    b: int


# =========================
# 🏠 Home route
# =========================
@app.get("/")
def home():
    return {"message": "Copra Quality ML API running"}


# =========================
# 🔮 Prediction route
# =========================
@app.post("/predict")
def predict(data: InputData):

    try:
        # ✅ FIXED input (no extra brackets)
        input_data = pd.DataFrame([{
            "Moisture": data.moisture,
            "Temperature": data.temperature,
            "R_value": data.r,
            "G_value": data.g,
            "B_value": data.b
        }])[["Moisture", "Temperature", "R_value", "G_value", "B_value"]]

        
        # Apply scaler
        input_scaled = scaler.transform(input_data)

        # Predict
        results = {
            "SVM": label_encoder.inverse_transform(svm_model.predict(input_scaled))[0],
            "Random Forest": label_encoder.inverse_transform(rf_model.predict(input_scaled))[0],
            "KNN": label_encoder.inverse_transform(knn_model.predict(input_scaled))[0],
            "Logistic Regression": label_encoder.inverse_transform(log_model.predict(input_scaled))[0]
        }
        
        print(f"ML Result: {results}")
        
        return {
            "input": {
                "Moisture": data.moisture,
                "Temperature": data.temperature,
                "R": data.r,
                "G": data.g,
                "B": data.b
            },
            "predictions": results
        }

    except Exception as e:
        return {"error": str(e)}

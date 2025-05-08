from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import traceback

# Load the trained model
try:
    model = joblib.load("model.pkl")
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Failed to load model:", e)

# Define input schema
class Patient(BaseModel):
    ArtRegimen: str
    DurationOnArt: float
    CD4Count: float
    ViralLoad: float
    Gender: str
    BMI: float
    AgeAtArtStart: float
    FamilyHistoryDiabetes: bool
    Lifestyle: str

# Initialize FastAPI
app = FastAPI()

@app.post("/predict")
def predict(patient: Patient):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([patient.dict()])

        # Predict using the model
        prediction = model.predict(input_df)[0]
        return {"PredictedDiabetesOnsetAge": round(prediction, 2)}

    except Exception as e:
        print("🔥 Prediction error:")
        traceback.print_exc()
        return {"error": "Prediction failed", "details": str(e)}

# HIV-DiabetesPredictor

![ChatGPT Image May 8, 2025, 10_49_53 PM](https://github.com/user-attachments/assets/b5140203-b94b-40a2-962c-edf21bf80feb)

from fastapi import FastAPI - framework to build the web api
from pydantic import BaseModel - pydantic defines structure of json input (data validation schemas)
import joblib - joblib is used to load the trained ML model from model.pkl. This model was saved earlier after training.
import pandas as pd - pandas is used to convert incoming JSON input into a DataFrame, which the model expects for prediction.
app = FastAPI() -  Creates a new FastAPI app — this is the server that runs your API.
model = joblib.load("model.pkl") -  Loads the trained model from disk. Now it's ready to use for real-time predictions.
input_df = pd.DataFrame([patient.dict()]) - Converts the input to a DataFrame.
The model expects tabular input (not plain dict or JSON).
prediction = model.predict(input_df)[0] - Uses the loaded model to predict the diabetes onset age.
.predict() returns an array — [0] gets the first value.
Step	Purpose
✅ Load Model	Load trained .pkl file for prediction
✅ Define Input	Ensure incoming JSON matches expected structure
✅ Predict	Use model to predict diabetes onset age
✅ Serve API	Available in real-time via /predict route

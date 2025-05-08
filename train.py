import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_csv("data.csv")

X = df.drop("DiabetesOnsetAge", axis=1)
y = df["DiabetesOnsetAge"]

categorical = ["ArtRegimen", "Gender", "Lifestyle"]
numerical = ["DurationOnArt", "CD4Count", "ViralLoad", "BMI", "AgeAtArtStart", "FamilyHistoryDiabetes"]


preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(), categorical)
], remainder="passthrough")

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor())
])

model.fit(X, y)
joblib.dump(model, "model.pkl")
print("✅ Model trained and saved.")
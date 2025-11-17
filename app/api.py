# app/api.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import pandas as pd

app = FastAPI(title="Automation Risk Predictor")

# Carregar modelo
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
model = joblib.load(MODEL_PATH)


# Definir esquema de entrada (mesmas features usadas no treino)
class JobFeatures(BaseModel):
    Industry: str
    Job_Status: str
    AI_Impact_Level: str
    Required_Education: str
    Median_Salary_USD: float
    Experience_Required_Years: float
    Job_Openings_2024: float
    Projected_Openings_2030: float
    Remote_Work_Ratio: float
    Gender_Diversity: float


@app.get("/")
def root():
    return {"message": "API de previsão de Automation Risk (%) está no ar."}


@app.post("/predict")
def predict(features: JobFeatures):
    # Converter entrada em DataFrame com os MESMOS nomes do treino
    data = {
        "Industry": [features.Industry],
        "Job Status": [features.Job_Status],
        "AI Impact Level": [features.AI_Impact_Level],
        "Required Education": [features.Required_Education],
        "Median Salary (USD)": [features.Median_Salary_USD],
        "Experience Required (Years)": [features.Experience_Required_Years],
        "Job Openings (2024)": [features.Job_Openings_2024],
        "Projected Openings (2030)": [features.Projected_Openings_2030],
        "Remote Work Ratio (%)": [features.Remote_Work_Ratio],
        "Gender Diversity (%)": [features.Gender_Diversity],
    }

    df = pd.DataFrame(data)

    prediction = model.predict(df)[0]

    return {
        "automation_risk_predicted": float(prediction)
    }

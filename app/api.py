# app/api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Automation Risk Predictor")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
model = joblib.load(MODEL_PATH)

origins = [
    "http://localhost:8000",         # para testes locais (ajuste se usar outra porta)
    "https://ai-impact-on-job-market-2024-2030-front.onrender.com" # troque pela URL do front-end no Render
]

app.add_middleware(
    CORSMiddleware,
    allow_origins= ['*'], #origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mapas PT -> EN
industry_map_pt_en = {
    "TI": "IT",
    "Manufatura": "Manufacturing",
    "Finanças": "Finance",
    "Saúde": "Healthcare",
    "Educação": "Education",
    "Entretenimento": "Entertainment",
    "Varejo": "Retail",
    "Transporte": "Transportation",
}

job_status_map_pt_en = {
    "Em crescimento": "Increasing",
    "Em declínio": "Decreasing",
}

ai_impact_map_pt_en = {
    "Baixo": "Low",
    "Moderado": "Moderate",
    "Alto": "High",
}

education_map_pt_en = {
    "Ensino médio": "High School",
    "Tecnólogo": "Associate Degree",
    "Graduação": "Bachelor’s Degree",
    "Mestrado": "Master’s Degree",
    "Doutorado": "PhD",
}


def normalize_category(value: str, allowed_en: list[str], pt_to_en: dict[str, str], field_name: str) -> str:
    """
    Aceita:
      - valor já em inglês (se estiver na lista allowed_en)
      - valor em português (se estiver nas chaves de pt_to_en)
    Retorna SEMPRE o valor em inglês (que é o que o modelo conhece).
    """
    if value in allowed_en:
        return value
    if value in pt_to_en:
        return pt_to_en[value]
    raise HTTPException(
        status_code=400,
        detail=f"Valor inválido para {field_name}: '{value}'. "
               f"Use um dos valores em inglês {allowed_en} ou em português {list(pt_to_en.keys())}."
    )


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
    # Normalizar categorias (aceitando PT e EN)
    industry_en = normalize_category(
        features.Industry,
        ["IT", "Manufacturing", "Finance", "Healthcare", "Education",
         "Entertainment", "Retail", "Transportation"],
        industry_map_pt_en,
        "Industry"
    )

    job_status_en = normalize_category(
        features.Job_Status,
        ["Increasing", "Decreasing"],
        job_status_map_pt_en,
        "Job_Status"
    )

    ai_impact_en = normalize_category(
        features.AI_Impact_Level,
        ["Low", "Moderate", "High"],
        ai_impact_map_pt_en,
        "AI_Impact_Level"
    )

    education_en = normalize_category(
        features.Required_Education,
        ["High School", "Associate Degree", "Bachelor’s Degree",
         "Master’s Degree", "PhD"],
        education_map_pt_en,
        "Required_Education"
    )

    # Montar DataFrame com os NOMES de coluna iguais aos do treino
    data = {
        "Industry": [industry_en],
        "Job Status": [job_status_en],
        "AI Impact Level": [ai_impact_en],
        "Required Education": [education_en],
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

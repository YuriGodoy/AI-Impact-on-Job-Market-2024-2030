# app/train.py

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Agora usamos o CSV MODIFICADO
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "jobs_ai.csv"
)

def main():
    # 1. Carregar dados
    df = pd.read_csv(DATA_PATH)

    # 👉 Target NOVO criado pelo script generate_new_target.py
    target_col = "Automation_Risk_New"

    numeric_features = [
        "Median Salary (USD)",
        "Experience Required (Years)",
        "Job Openings (2024)",
        "Projected Openings (2030)",
        "Remote Work Ratio (%)",
        "Gender Diversity (%)",
    ]

    categorical_features = [
        "Industry",
        "Job Status",
        "AI Impact Level",
        "Required Education",
    ]

    # 2. Criar bins do target para estratificação (regressão estratificada)
    df["target_bin"] = pd.qcut(df[target_col], q=10, duplicates="drop")

    X = df[numeric_features + categorical_features]
    y = df[target_col]

    # 3. Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=df["target_bin"]
    )

    # Não precisamos mais da coluna auxiliar
    df.drop(columns=["target_bin"], inplace=True)

    # 4. Pré-processamento
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # 5. Pipeline final
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    # 6. Treinar
    model.fit(X_train, y_train)

    # 7. Avaliar
    y_pred = model.predict(X_test)

    print("=== Métricas do modelo ===")
    print(f"R² : {r2_score(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print()
    print("Distribuição das previsões:")
    print(f"min = {y_pred.min():.2f} | max = {y_pred.max():.2f} | média = {y_pred.mean():.2f}")

    # 8. Salvar modelo
    model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
    joblib.dump(model, model_path)
    print(f"\nModelo salvo em: {model_path}")


if __name__ == "__main__":
    main()

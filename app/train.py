# app/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

# Caminho do CSV
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "jobs_ai.csv")

def main():
    # 1. Carregar dados
    df = pd.read_csv(DATA_PATH)

    # 2. Definir colunas
    target_col = "Automation Risk (%)"

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

    # 3. Selecionar X e y
    X = df[numeric_features + categorical_features]
    y = df[target_col]

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

    # 5. Montar pipeline completo
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    # 6. Treino / teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    # 7. Avaliação simples
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"R²: {r2:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"MSE: {mse:.3f}")

    # 8. Salvar modelo treinado
    model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
    joblib.dump(model, model_path)
    print(f"Modelo salvo em: {model_path}")


if __name__ == "__main__":
    main()

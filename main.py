from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import pandas as pd
import json

# ============================================
# API FASTAPI – MODELO DE SCORING (RF + CUT-OFF)
# Autor: Andrés Roldán
# Fecha: Nov 2025
# ============================================

# -----------------------------
# Cargar modelo y configuración
# -----------------------------
# 1) Cargar configuración DESDE JSON
with open("modelo_scoring_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

X_cols = config["X_cols"]
cutoff_final = float(config["cutoff_final"])
modelo_path = config.get("modelo_path", "modelo_scoring_rf_compressed.joblib")

# 2) Cargar el modelo desde el archivo .joblib
pipeline = load(modelo_path)

# -----------------------------
# Inicializar API
# -----------------------------
app = FastAPI(title="API de Scoring Financiero – Microfinanzas")

# -----------------------------
# Modelo de entrada (JSON)
# -----------------------------
class Cliente(BaseModel):
    Edad: int
    Ciudad: str
    Grado_Escolaridad: str
    Tipo_Empleo: str
    Nivel_Endeudamiento: float
    Score_Credito: float
    Antiguedad_Empleo: float
    Rango_Edad: str
    Endeudamiento_Alto: int
    Ingreso_Log: float

# -----------------------------
# Función de inferencia
# -----------------------------
@app.post("/predict")
def predecir(cliente: Cliente):

    df = pd.DataFrame([cliente.dict()])

    # Asegurar columnas necesarias
    for col in X_cols:
        if col not in df.columns:
            df[col] = 0

    # Reordenar columnas
    df = df[X_cols]

    # Predicción de probabilidad de default
    pd_val = pipeline.predict_proba(df)[:, 1][0]

    # Score
    score = (1 - pd_val) * 1000

    # Decisión final
    decision = 1 if pd_val >= cutoff_final else 0
    decision_txt = "Rechazar" if decision == 1 else "Aprobar"

    # Segmento riesgo
    if score >= 800:
        segmento = "Muy bajo"
    elif score >= 650:
        segmento = "Bajo"
    elif score >= 550:
        segmento = "Medio"
    elif score >= 450:
        segmento = "Medio-alto"
    elif score >= 300:
        segmento = "Alto"
    else:
        segmento = "Muy alto"

    # Respuesta final
    return {
        "PD": round(pd_val, 4),
        "Score": round(score, 2),
        "Segmento_Riesgo": segmento,
        "Cutoff": cutoff_final,
        "Decision": decision,
        "Decision_texto": decision_txt
    }

# -----------------------------
# Endpoint de prueba
# -----------------------------
@app.get("/")
def root():
    return {"mensaje": "API de Scoring funcionando correctamente"}

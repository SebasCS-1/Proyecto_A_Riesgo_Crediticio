"""
API REST — Modelos de Riesgo Crediticio
----------------------------------------
Expone dos endpoints de predicción:
  POST /predecir/binario     → Riesgo Bueno / Malo
  POST /predecir/multiclase  → Nivel Bajo / Medio / Alto / Crítico


Iniciar:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# api/main.py -> api/ -> raíz del proyecto
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

app = FastAPI(
    title="API de Riesgo Crediticio",
    description=(
        "Predice el riesgo crediticio de un cliente usando dos modelos ANN: "
        "uno binario (Bueno/Malo) y uno multiclase (Bajo/Medio/Alto/Crítico)."
    ),
    version="1.0.0",
)


def _cargar_artefactos():
    """Carga los modelos y artefactos de preprocesamiento al iniciar la API."""

    # Scaler ajustado únicamente con los datos de entrenamiento
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")

    # Orden exacto de las 61 columnas que espera el modelo
    columnas_modelo: list[str] = joblib.load(MODELS_DIR / "columnas_modelo.pkl")

    # Umbrales y categorías de riesgo para el puntaje de exposición
    regla_multiclase: dict = joblib.load(MODELS_DIR / "regla_multiclase.pkl")

    # Modelo ANN binario (bueno/malo)
    modelo_binario = tf.keras.models.load_model(MODELS_DIR / "modelo_pred_binomial.keras")

    # Modelo ANN multiclase (Bajo/Medio/Alto/Crítico)
    modelo_multiclase = tf.keras.models.load_model(MODELS_DIR / "modelo_riesgo_multiclase.keras")

    return scaler, columnas_modelo, regla_multiclase, modelo_binario, modelo_multiclase


try:
    (
        SCALER,
        COLUMNAS_MODELO,
        REGLA_MULTICLASE,
        MODELO_BINARIO,
        MODELO_MULTICLASE,
    ) = _cargar_artefactos()
    print("✓ Modelos y artefactos cargados correctamente.")
except Exception as exc:
    # Si los modelos no existen todavía, la API arranca pero devuelve 503
    # en los endpoints de predicción hasta que estén disponibles.
    SCALER = COLUMNAS_MODELO = REGLA_MULTICLASE = None
    MODELO_BINARIO = MODELO_MULTICLASE = None
    print(f"⚠ No se pudieron cargar los modelos: {exc}")


# Valores válidos para cada variable categórica del dataset German Credit
_ESTADO_CUENTA       = Literal["A11", "A12", "A13", "A14"]
_HISTORIAL           = Literal["A30", "A31", "A32", "A33", "A34"]
_PROPOSITO           = Literal["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A48", "A49", "A410"]
_CUENTA_AHORROS      = Literal["A61", "A62", "A63", "A64", "A65"]
_ANTIGUEDAD_EMPLEO   = Literal["A71", "A72", "A73", "A74", "A75"]
_ESTADO_PERSONAL     = Literal["A91", "A92", "A93", "A94"]
_OTROS_DEUDORES      = Literal["A101", "A102", "A103"]
_PROPIEDAD           = Literal["A121", "A122", "A123", "A124"]
_OTROS_PLANES_PAGO   = Literal["A141", "A142", "A143"]
_VIVIENDA            = Literal["A151", "A152", "A153"]
_TRABAJO             = Literal["A171", "A172", "A173", "A174"]
_TELEFONO            = Literal["A191", "A192"]
_TRABAJADOR_EXT      = Literal["A201", "A202"]


class ClienteInput(BaseModel):
    """Características de un cliente del dataset German Credit."""

    # Variables numéricas
    duracion_meses:       int   = Field(..., ge=1,    le=72,    description="Duración del crédito en meses")
    monto_credito:        int   = Field(..., ge=250,  le=20000, description="Monto del crédito en DM")
    tasa_cuotas:          int   = Field(..., ge=1,    le=4,     description="Tasa de cuotas como % del ingreso disponible (1-4)")
    tiempo_residencia:    int   = Field(..., ge=1,    le=4,     description="Años en residencia actual (1-4)")
    edad:                 int   = Field(..., ge=18,   le=90,    description="Edad del cliente en años")
    creditos_existentes:  int   = Field(..., ge=1,    le=4,     description="Número de créditos existentes en este banco")
    personas_dependientes:int   = Field(..., ge=1,    le=2,     description="Número de personas a cargo (1 o 2)")

    # Variables categóricas (códigos del dataset German Credit)
    estado_cuenta:        _ESTADO_CUENTA     = Field(..., description="Estado de la cuenta corriente: A11=< 0 DM | A12=0-200 DM | A13=>200 DM | A14=sin cuenta")
    historial_crediticio: _HISTORIAL         = Field(..., description="Historial crediticio: A30=sin créditos | A31=todos pagados otros bancos | A32=pagados puntualmente | A33=mora pasada | A34=cuenta crítica")
    proposito:            _PROPOSITO         = Field(..., description="Propósito del crédito: A40=auto nuevo | A41=auto usado | A42=muebles | A43=tv/radio | A44=electrodomésticos | A45=reparaciones | A46=educación | A48=reentrenamiento | A49=negocio | A410=otros")
    cuenta_ahorros:       _CUENTA_AHORROS    = Field(..., description="Cuenta de ahorros: A61=<100 DM | A62=100-500 DM | A63=500-1000 DM | A64=>1000 DM | A65=sin cuenta")
    antiguedad_empleo:    _ANTIGUEDAD_EMPLEO = Field(..., description="Antigüedad en empleo actual: A71=desempleado | A72=<1 año | A73=1-4 años | A74=4-7 años | A75=>7 años")
    estado_personal:      _ESTADO_PERSONAL   = Field(..., description="Estado civil y sexo: A91=hombre divorciado | A92=mujer divorciada/casada | A93=hombre soltero | A94=hombre casado")
    otros_deudores:       _OTROS_DEUDORES    = Field(..., description="Otros deudores/garantes: A101=ninguno | A102=co-solicitante | A103=garante")
    propiedad:            _PROPIEDAD         = Field(..., description="Propiedad: A121=bienes raíces | A122=seguro de vida | A123=auto | A124=sin propiedad")
    otros_planes_pago:    _OTROS_PLANES_PAGO = Field(..., description="Otros planes de pago: A141=banco | A142=tiendas | A143=ninguno")
    vivienda:             _VIVIENDA          = Field(..., description="Vivienda: A151=alquiler | A152=propia | A153=gratuita")
    trabajo:              _TRABAJO           = Field(..., description="Trabajo: A171=desempleado no residente | A172=no calificado residente | A173=empleado calificado | A174=directivo/universitario")
    telefono:             _TELEFONO          = Field(..., description="Teléfono: A191=sin teléfono | A192=registrado a nombre del cliente")
    trabajador_extranjero:_TRABAJADOR_EXT    = Field(..., description="Trabajador extranjero: A201=sí | A202=no")

    model_config = {"json_schema_extra": {"example": {
        "duracion_meses": 24,
        "monto_credito": 4500,
        "tasa_cuotas": 2,
        "tiempo_residencia": 3,
        "edad": 35,
        "creditos_existentes": 1,
        "personas_dependientes": 1,
        "estado_cuenta": "A12",
        "historial_crediticio": "A32",
        "proposito": "A43",
        "cuenta_ahorros": "A62",
        "antiguedad_empleo": "A73",
        "estado_personal": "A93",
        "otros_deudores": "A101",
        "propiedad": "A121",
        "otros_planes_pago": "A143",
        "vivienda": "A152",
        "trabajo": "A173",
        "telefono": "A192",
        "trabajador_extranjero": "A202",
    }}}


_CAT_COLS = [
    "estado_cuenta", "historial_crediticio", "proposito", "cuenta_ahorros",
    "antiguedad_empleo", "estado_personal", "otros_deudores", "propiedad",
    "otros_planes_pago", "vivienda", "trabajo", "telefono", "trabajador_extranjero",
]


def _preprocesar(cliente: ClienteInput) -> pd.DataFrame:
    """
    Convierte el JSON de entrada en el vector de 61 columnas que esperan los modelos:
      1. Construye un DataFrame con los 20 campos.
      2. Aplica get_dummies sobre las variables categóricas.
      3. Alinea las columnas con el orden exacto guardado en columnas_modelo.pkl
         (columnas dummies ausentes se rellenan con 0).
      4. Escala con el MinMaxScaler ajustado en el entrenamiento.
    """
    fila = cliente.model_dump()
    df = pd.DataFrame([fila])

    # get_dummies replica el mismo preprocesamiento que PreparadorDatos
    df_dummies = pd.get_dummies(df, columns=_CAT_COLS, dtype=int)

    # Alinear columnas: agrega las que falten con 0 y elimina las que sobren
    df_alineado = df_dummies.reindex(columns=COLUMNAS_MODELO, fill_value=0)

    # Escalar con el scaler del entrenamiento
    X_scaled = SCALER.transform(df_alineado)

    return pd.DataFrame(X_scaled, columns=COLUMNAS_MODELO)


def _calcular_puntaje_exposicion(cliente: ClienteInput, regla: dict) -> int:
    """
    Calcula el puntaje de exposición (0-5) con las mismas reglas que PreparadorDatos,
    usando las medianas del conjunto de entrenamiento guardadas en regla_multiclase.pkl.
    """
    puntaje = 0

    if cliente.duracion_meses > float(regla["mediana_duracion"]):
        puntaje += 1

    if cliente.monto_credito > float(regla["mediana_monto"]):
        puntaje += 1

    if cliente.estado_cuenta in regla["estado_cuenta_riesgo"]:
        puntaje += 1

    if cliente.cuenta_ahorros in regla["cuenta_ahorros_riesgo"]:
        puntaje += 1

    if cliente.historial_crediticio in regla["historial_riesgo"]:
        puntaje += 1

    return puntaje


def _verificar_modelos():
    """Lanza HTTP 503 si los modelos no están disponibles."""
    if MODELO_BINARIO is None or MODELO_MULTICLASE is None:
        raise HTTPException(
            status_code=503,
            detail="Los modelos no están disponibles. Entrena y guarda los modelos primero.",
        )



@app.get("/", tags=["Estado"])
def raiz():
    """Verifica que la API esté funcionando."""
    binario_ok    = MODELO_BINARIO is not None
    multiclase_ok = MODELO_MULTICLASE is not None
    return {
        "estado": "activo",
        "modelo_binario_cargado":    binario_ok,
        "modelo_multiclase_cargado": multiclase_ok,
    }


@app.get("/columnas", tags=["Información"])
def columnas():
    # Devuelve las 61 columnas (en orden) que esperan los modelos tras el preprocesamiento.
    _verificar_modelos()
    return {"columnas": COLUMNAS_MODELO, "total": len(COLUMNAS_MODELO)}


@app.post("/predecir/binario", tags=["Predicción"])
def predecir_binario(cliente: ClienteInput):
    
    # Predice si el cliente representa un **riesgo Bueno (0) o Malo (1)**.
    
    _verificar_modelos()

    try:
        X = _preprocesar(cliente)
        prob_malo = float(MODELO_BINARIO.predict(X, verbose=0)[0][0])
        prediccion = int(prob_malo > 0.5)
        etiqueta   = "Malo" if prediccion == 1 else "Bueno"
        confianza  = prob_malo if prediccion == 1 else 1 - prob_malo

        return {
            "prediccion":       prediccion,
            "etiqueta":         etiqueta,
            "probabilidad_malo": round(prob_malo, 4),
            "confianza":         round(confianza, 4),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predecir/multiclase", tags=["Predicción"])
def predecir_multiclase(cliente: ClienteInput):
    """
    Predice el **nivel de riesgo crediticio**: Bajo (0), Medio (1), Alto (2) o Crítico (3).

    El modelo multiclase recibe las 61 columnas preprocesadas más:
    - `prob_binaria`: salida del modelo binario (P de crédito malo)
    - `puntaje_exposicion`: puntaje 0-5 calculado con las mismas reglas del entrenamiento

    """
    _verificar_modelos()

    NIVELES = {0: "Bajo", 1: "Medio", 2: "Alto", 3: "Crítico"}

    try:
        # 1) Preprocesamiento compartido (61 columnas escaladas)
        X_base = _preprocesar(cliente)

        # 2) Probabilidad del modelo binario
        prob_malo = float(MODELO_BINARIO.predict(X_base, verbose=0)[0][0])

        # 3) Puntaje de exposición con las medianas del entrenamiento
        puntaje = _calcular_puntaje_exposicion(cliente, REGLA_MULTICLASE)

        # 4) Agregar las dos columnas extra que espera el modelo multiclase
        X_multi = X_base.copy()
        X_multi["prob_binaria"]        = prob_malo
        X_multi["puntaje_exposicion"]  = puntaje / 5.0   # normalizado 0-1

        # 5) Predicción multiclase
        probabilidades = MODELO_MULTICLASE.predict(X_multi, verbose=0)[0]
        prediccion     = int(np.argmax(probabilidades))
        etiqueta       = NIVELES[prediccion]
        confianza      = float(probabilidades[prediccion])

        return {
            "prediccion":  prediccion,
            "etiqueta":    etiqueta,
            "confianza":   round(confianza, 4),
            "probabilidades": {
                NIVELES[i]: round(float(p), 4)
                for i, p in enumerate(probabilidades)
            },
            "puntaje_exposicion": puntaje,
            "prob_binaria":       round(prob_malo, 4),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

<<<<<<< HEAD
# Proyecto A: Sistema de Predicción de Riesgo Crediticio

**Curso:** BD-151 Inteligencia Artificial Aplicada – Colegio Universitario de Cartago
**Profesor:** Osvaldo González Chaves
**Año:** 2026

## Integrantes

| Nombre | Carné | Correo |
|---|---|---|
| | | |
| | | |
| | | |

## Descripción del problema

Sistema inteligente que predice el riesgo crediticio de clientes bancarios para apoyar decisiones de aprobación de préstamos. Clasifica a los clientes en categorías de riesgo y proporciona predicciones binarias de aprobación/rechazo.

## Dataset

- **Nombre:** German Credit Data (UCI Machine Learning Repository)
- **URL:** https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
- **Registros:** 1,000 clientes
- **Variables:** 20 (numéricas y categóricas)

Colocar los archivos originales en `data/raw/` sin modificarlos.

## Modelos

- **Modelo 1 – Clasificación Binaria:** Predecir aprobación de crédito (Bueno/Malo)
- **Modelo 2 – Clasificación Multiclase:** Clasificar nivel de riesgo (Bajo/Medio/Alto/Crítico). El dataset solo incluye la etiqueta Bueno/Malo: el grupo debe definir y justificar el criterio para construir los cuatro niveles

Lineamientos de entrenamiento:

- Normalización con `MinMaxScaler` ajustado solo sobre el conjunto de entrenamiento.
- Variables categóricas con `pd.get_dummies`; guardar la lista de columnas resultante en `models/columnas.pkl`.
- Entrenamiento con `validation_split` y `EarlyStopping`; el conjunto de prueba se usa solo para la evaluación final.
- Comparar al menos dos configuraciones por modelo (por ejemplo, con y sin `Dropout`).

## API REST

| Método | Endpoint | Respuesta |
|---|---|---|
| POST | `/predict/binary` | Aprobación de crédito (Bueno/Malo) |
| POST | `/predict/risk_level` | Nivel de riesgo (Bajo/Medio/Alto/Crítico) |

La documentación automática queda disponible en `http://localhost:8000/docs`.

## Estructura del proyecto

```
Proyecto_A_Riesgo_Crediticio/
│
├── README.md                      ← Guía completa de instalación y uso del proyecto
├── requirements.txt               ← Dependencias Python
├── .gitignore                     ← Archivos excluidos del control de versiones
│
├── data/
│   ├── raw/                       ← Datos originales sin procesar
│   └── processed/                 ← Datos limpios y preprocesados
│       ├── train.csv              ← Conjunto de entrenamiento
│       └── test.csv               ← Conjunto de prueba
│
├── notebooks/
│   ├── 01_EDA.ipynb               ← Análisis Exploratorio de Datos
│   ├── 02_Preprocesamiento.ipynb  ← Limpieza, variables dummy, normalización
│   ├── 03_ANN_Modelo1.ipynb       ← Entrenamiento del Modelo 1
│   ├── 04_ANN_Modelo2.ipynb       ← Entrenamiento del Modelo 2
│   └── 05_Comparacion_Modelos.ipynb ← Evaluación y selección del mejor modelo
│
├── src/
│   ├── __init__.py
│   ├── config.py                  ← Configuraciones globales (rutas, parámetros)
│   ├── data_prep.py               ← Funciones de preprocesamiento
│   └── train/
│       ├── __init__.py
│       ├── model1.py              ← Entrenamiento del Modelo 1
│       ├── model2.py              ← Entrenamiento del Modelo 2
│       └── utils.py               ← Utilidades compartidas (métricas, gráficas)
│
├── models/
│   ├── model1.keras               ← Modelo 1 guardado (formato Keras)
│   ├── model2.keras               ← Modelo 2 guardado (formato Keras)
│   ├── scaler.pkl                 ← MinMaxScaler entrenado
│   └── columnas.pkl               ← Columnas finales tras get_dummies
│
├── api/
│   ├── main.py                    ← Aplicación FastAPI con endpoints
│   ├── schemas.py                 ← Modelos Pydantic para validación
│   └── predict.py                 ← Lógica de predicción e inferencia
│
└── app/
    ├── Home.py                    ← Página principal del dashboard
    └── pages/
        ├── 1_Prediccion.py        ← Predicciones individuales
        ├── 2_Analisis.py          ← Análisis de lotes
        └── 3_Metricas.py          ← Métricas y rendimiento
```

## Instalación

Requisitos: Python 3.11 y Git.

```bash
git clone <url-del-repositorio>
cd Proyecto_A_Riesgo_Crediticio
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
pip install -r requirements.txt
```

## Ejecución

1. Ejecutar los notebooks en orden (`01` a `05`) desde `notebooks/`:
   ```bash
   jupyter notebook
   ```
2. Levantar la API (desde la raíz del proyecto):
   ```bash
   uvicorn api.main:app --reload
   ```
3. Levantar el frontend en otra terminal:
   ```bash
   streamlit run app/Home.py
   ```

## Entregables

- [ ] Notebook de EDA con análisis de distribuciones y correlaciones
- [ ] Notebook de preprocesamiento con variables dummy (get_dummies) y normalización (MinMaxScaler)
- [ ] Dos modelos ANN entrenados y guardados (.keras)
- [ ] API REST con endpoints /predict/binary y /predict/risk_level
- [ ] Frontend Streamlit con formulario de entrada y visualización de resultados
- [ ] README.md con documentación completa

## Resultados

| Modelo | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Modelo 1 | | | | |
| Modelo 2 | | | | |

**Modelo seleccionado y justificación:**

_Completar._

## Conclusiones y recomendaciones

_Completar._
=======
# Proyecto_A_Riesgo_Crediticio
>>>>>>> d23558035bbbb419b3d1b4a0f5483c60ec8a3b33

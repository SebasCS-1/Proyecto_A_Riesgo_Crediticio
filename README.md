# Proyecto A: Sistema de Predicción de Riesgo Crediticio

Proyecto desarrollado para el curso de **Inteligencia Artificial Aplicada** del Colegio Universitario de Cartago.

El sistema utiliza el conjunto de datos **German Credit** para preparar y entrenar dos modelos de redes neuronales artificiales:

1. Un modelo de clasificación binaria:
   - Crédito Bueno.
   - Crédito Malo.

2. Un modelo de clasificación multiclase:
   - Riesgo Bajo.
   - Riesgo Medio.
   - Riesgo Alto.
   - Riesgo Crítico.

El proyecto incluye análisis exploratorio de datos, preprocesamiento, programación orientada a objetos, redes neuronales artificiales, comparación de modelos, una API REST desarrollada con FastAPI y una interfaz interactiva desarrollada con Streamlit.

---

## Integrantes

- Sebastián Calvo Solano
- Carlos Cambronero Zuñiga
- Josué Francisco Redondo Gómez
- Steven Zamora Reyes

---

## Objetivo general

Desarrollar un sistema de inteligencia artificial capaz de analizar la información de una solicitud de crédito y generar una predicción de riesgo mediante redes neuronales artificiales.

---

## Objetivos específicos

- Analizar la estructura y calidad del conjunto de datos German Credit.
- Identificar relaciones entre las características crediticias y la clasificación del crédito.
- Preparar las variables numéricas y categóricas para su utilización en redes neuronales.
- Construir una variable objetivo binaria.
- Construir y justificar cuatro niveles académicos de riesgo.
- Implementar el preprocesamiento mediante programación orientada a objetos.
- Entrenar y comparar diferentes configuraciones de redes neuronales.
- Evaluar los modelos mediante métricas y matrices de confusión.
- Exponer las predicciones mediante una API REST.
- Desarrollar una interfaz para ingresar datos y visualizar resultados.
- Documentar el proceso completo de forma reproducible.

---

## Dataset

El proyecto utiliza el conjunto de datos **German Credit Data**, procedente del UCI Machine Learning Repository.

El archivo principal es:

```text
data/raw/german.data
```

El conjunto contiene:

- 1.000 registros.
- 20 variables de entrada.
- Una variable objetivo.
- Siete variables numéricas.
- Trece variables categóricas.

La clasificación original utiliza:

```text
1 = Crédito bueno
2 = Crédito malo
```

También se conservan los archivos originales de documentación y referencia:

```text
data/raw/german.data
data/raw/german.data-numeric
data/raw/german.doc
data/raw/Index
```

El archivo `german.data` se utiliza como fuente principal porque conserva las variables categóricas originales y permite demostrar su transformación mediante `get_dummies`.

El archivo `german.doc` se utiliza para interpretar el significado de las variables y de los códigos categóricos, como `A11`, `A43` y `A61`.

---

## Estructura del proyecto

```text
Proyecto_Riesgo_Crediticio/
│
├── .streamlit/
│   └── config.toml
│
├── api/
│   └── main.py
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   ├── german.data
│   │   ├── german.data-numeric
│   │   ├── german.doc
│   │   └── Index
│   │
│   └── processed/
│       ├── X_train_binario.csv
│       ├── X_test_binario.csv
│       ├── y_train_binario.csv
│       ├── y_test_binario.csv
│       ├── X_train_multiclase.csv
│       ├── X_test_multiclase.csv
│       ├── y_train_multiclase.csv
│       └── y_test_multiclase.csv
│
├── models/
│   ├── modelo_pred_binomial.keras
│   ├── modelo_riesgo_multiclase.keras
│   ├── scaler.pkl
│   ├── columnas_modelo.pkl
│   ├── informacion_preprocesamiento.pkl
│   └── regla_multiclase.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocesamiento.ipynb
│   ├── 03_ANN_Modelo1.ipynb
│   ├── 04_ANN_Modelo_2.ipynb
│   └── 05_Comparacion_Modelos.ipynb
│
├── src/
│   ├── data_prep.py
│   ├── probar_data_prep.py
│   └── train/
│
├── .gitignore
├── README.md
├── requirements.txt
└── USO_DE_IA.md
```

---

## Análisis exploratorio de datos

El análisis exploratorio se encuentra en:

```text
notebooks/01_EDA.ipynb
```

El notebook incluye:

- Carga y descripción del dataset.
- Asignación de nombres a las columnas.
- Verificación de dimensiones.
- Revisión de valores nulos.
- Revisión de registros duplicados.
- Estadísticas descriptivas.
- Distribución de la variable objetivo.
- Análisis de duración, monto y edad.
- Comparación de variables numéricas según la clase.
- Análisis de variables categóricas seleccionadas.
- Tablas de cantidades y porcentajes.
- Matriz de correlación.
- Conclusiones del análisis.

### Resultados iniciales

El conjunto contiene:

```text
1.000 registros
20 variables de entrada
Una variable objetivo
Cero valores nulos
Cero registros duplicados
```

La distribución de la variable objetivo es:

```text
Créditos buenos: 700 registros, 70 %
Créditos malos:  300 registros, 30 %
```

Debido a esta distribución, la evaluación de los modelos no debe depender exclusivamente de la exactitud.

También deben considerarse métricas como:

- Precision.
- Recall.
- F1-score.
- Matriz de confusión.
- ROC-AUC cuando corresponda.

### Hallazgos principales

Entre los hallazgos documentados durante el análisis se encuentran:

- Los créditos clasificados como malos presentan una duración promedio mayor.
- Los créditos malos presentan un monto promedio mayor.
- La edad promedio muestra diferencias menores entre ambas clases.
- La duración y el monto presentan una relación positiva.
- El estado de la cuenta, el historial crediticio y los ahorros presentan diferencias entre las clases.
- Las relaciones observadas representan asociaciones dentro del conjunto de datos y no demuestran causalidad.

---

## Decisión sobre las variables

Se conservaron las 20 variables originales para construir los modelos base.

Antes de tomar esta decisión se comprobó que:

- No existen identificadores de clientes.
- No hay columnas duplicadas.
- No hay variables constantes.
- No existen valores nulos.
- Ninguna variable de entrada contiene directamente la respuesta.
- Las variables describen diferentes aspectos personales, laborales y crediticios.

Después de aplicar `get_dummies`, las 20 variables originales quedan representadas mediante 61 entradas numéricas.

Estas 61 entradas no representan 61 variables originales independientes. Corresponden a:

- Las siete variables numéricas.
- La representación disyuntiva de las trece variables categóricas.

---

## Preprocesamiento

El procedimiento detallado se encuentra en:

```text
notebooks/02_Preprocesamiento.ipynb
```

Las etapas realizadas son:

1. Carga de `german.data`.
2. Asignación de nombres a las 21 columnas.
3. Verificación de calidad.
4. Separación de variables numéricas y categóricas.
5. Transformación de la variable objetivo binaria.
6. Separación de entradas y salida.
7. Conversión de categorías mediante `pd.get_dummies`.
8. División en entrenamiento y prueba.
9. Normalización mediante `MinMaxScaler`.
10. Creación de los cuatro niveles de riesgo.
11. Guardado de los datos y objetos de transformación.
12. Verificación de dimensiones y archivos.

---

## Clasificación binaria

La clase original se transforma de la siguiente manera:

```text
Clase original 1 = Crédito bueno → 0
Clase original 2 = Crédito malo  → 1
```

La salida binaria utiliza:

```text
0 = Bueno
1 = Malo
```

El valor `1` representa el resultado Malo. Por ello, la salida Sigmoid del modelo se interpreta como la probabilidad estimada de que un crédito sea clasificado como Malo.

---

## Separación de entradas y salida

Las variables de entrada se almacenan en `X`.

La respuesta esperada se almacena en `y`.

Las columnas relacionadas con la respuesta se excluyen de las entradas para evitar fuga de información.

Antes de la codificación:

```text
X = 1.000 registros y 20 variables originales
```

Después de `get_dummies`:

```text
X = 1.000 registros y 61 entradas numéricas
```

---

## Conversión de variables categóricas

Las trece variables categóricas contienen códigos como:

```text
A11
A12
A43
A61
A101
```

Las redes neuronales necesitan entradas numéricas. Por ello, se utiliza:

```python
pd.get_dummies()
```

Cada categoría se transforma en una columna binaria de cero o uno.

Ejemplo:

```text
Propósito original: A43

proposito_A42 = 0
proposito_A43 = 1
proposito_A45 = 0
```

Esta transformación no crea información nueva. Representa numéricamente las categorías existentes.

---

## División de los datos

Se utiliza la siguiente distribución:

```text
80 % para entrenamiento
20 % para prueba
```

La división se realiza con:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=101,
    stratify=y
)
```

El uso de `stratify` conserva la proporción original de las clases.

Los resultados son:

```text
X_train: 800 registros y 61 entradas
X_test:  200 registros y 61 entradas
```

La proporción en ambos conjuntos permanece aproximadamente en:

```text
70 % créditos buenos
30 % créditos malos
```

El conjunto de prueba se reserva para la evaluación final. Durante el entrenamiento, la validación se obtiene mediante `validation_split` sobre el conjunto de entrenamiento.

---

## Normalización

Se utiliza `MinMaxScaler` para transformar los valores a una escala adecuada para las redes neuronales.

El escalador se ajusta únicamente con los datos de entrenamiento:

```python
X_train_escalado = scaler.fit_transform(X_train)
```

El conjunto de prueba se transforma utilizando el mismo escalador:

```python
X_test_escalado = scaler.transform(X_test)
```

Esto evita que la información del conjunto de prueba influya en la preparación del entrenamiento.

El escalador se guarda como:

```text
models/scaler.pkl
```

También se guarda el nombre y el orden de las entradas finales en:

```text
models/columnas_modelo.pkl
```

Esto permite reproducir en la API el mismo orden utilizado durante el entrenamiento.

---

## Clasificación multiclase

El dataset original contiene únicamente las clases Bueno y Malo.

Para cumplir con el segundo modelo se construyeron cuatro niveles académicos:

```text
0 = Bajo
1 = Medio
2 = Alto
3 = Crítico
```

Los registros originalmente clasificados como buenos se dividen entre:

```text
Bajo
Medio
```

Los registros originalmente clasificados como malos se dividen entre:

```text
Alto
Crítico
```

### Puntaje de exposición

La división utiliza cinco condiciones relacionadas con la exposición crediticia:

1. Duración del crédito superior a la mediana del conjunto de entrenamiento.
2. Monto del crédito superior a la mediana del conjunto de entrenamiento.
3. Estado de cuenta corriente correspondiente a `A11` o `A12`.
4. Cuenta de ahorros correspondiente a `A61` o `A62`.
5. Historial crediticio correspondiente a `A30` o `A31`.

Cada condición cumplida suma un punto.

El puntaje de exposición puede tomar valores entre cero y cinco:

```text
Puntaje mínimo = 0
Puntaje máximo = 5
```

La clasificación multiclase conserva la relación con la clase binaria:

- Los créditos buenos se clasifican como Bajo o Medio.
- Los créditos malos se clasifican como Alto o Crítico.

El puntaje se normaliza dividiéndolo entre cinco antes de incorporarlo como entrada del modelo.

```python
puntaje_exposicion_normalizado = puntaje_exposicion / 5
```

Este criterio fue diseñado con fines académicos para construir los cuatro niveles solicitados.

El puntaje de exposición:

- No es una probabilidad.
- No representa directamente la predicción final.
- No corresponde a una escala oficial de una institución financiera.
- Se utiliza como una entrada adicional del modelo multiclase.

Las reglas utilizadas se guardan en:

```text
models/regla_multiclase.pkl
```

---

## Dimensiones finales

### Modelo binario

```text
X_train_binario = 800 registros y 61 entradas
X_test_binario  = 200 registros y 61 entradas
```

### Datos multiclase procesados

Los archivos multiclase incorporan el puntaje de exposición normalizado:

```text
X_train_multiclase = 800 registros y 62 entradas
X_test_multiclase  = 200 registros y 62 entradas
```

### Modelo multiclase híbrido

El notebook multiclase incorpora la probabilidad producida por el modelo binario:

```text
61 entradas preprocesadas
+ 1 puntaje de exposición normalizado
+ 1 probabilidad binaria
= 63 entradas finales
```

Por tanto:

```text
Entrada del modelo multiclase = 63 valores
```

---

## Programación orientada a objetos

El preprocesamiento también se implementa mediante la clase:

```python
PreparadorDatos
```

La clase se encuentra en:

```text
src/data_prep.py
```

La implementación orientada a objetos automatiza el procedimiento desarrollado inicialmente en el notebook.

La clase organiza el proceso mediante métodos con responsabilidades específicas:

- Cargar los datos.
- Verificar valores nulos y duplicados.
- Crear la variable objetivo binaria.
- Separar entradas y salida.
- Convertir las variables categóricas.
- Dividir los datos.
- Normalizar los datos.
- Crear el objetivo multiclase.
- Verificar la coherencia entre las clases.
- Guardar los archivos procesados.
- Verificar los archivos generados.
- Comprobar las dimensiones finales.

El método:

```python
ejecutar_preprocesamiento()
```

coordina la ejecución completa en el orden correcto.

La implementación mediante programación orientada a objetos permite:

- Evitar la repetición del código.
- Centralizar las rutas.
- Conservar las transformaciones utilizadas.
- Ejecutar los pasos en un orden controlado.
- Reutilizar el procedimiento.
- Verificar automáticamente los resultados.
- Reproducir los archivos procesados y los artefactos.

---

## Modelo ANN binario

El entrenamiento del modelo binario se encuentra en:

```text
notebooks/03_ANN_Modelo1.ipynb
```

El modelo recibe 61 entradas numéricas y genera una probabilidad asociada con la clase Malo.

En el notebook se evaluaron dos configuraciones.

### Configuración 1: EarlyStopping sin Dropout

Arquitectura:

```text
Entrada: 61 valores
Capa oculta 1: 61 neuronas, ReLU
Capa oculta 2: 20 neuronas, ReLU
Salida: una neurona, Sigmoid
```

Configuración:

- Función de pérdida `binary_crossentropy`.
- Optimizador Adam.
- Máximo de 600 épocas.
- `validation_split=0.20`.
- EarlyStopping con paciencia de cinco épocas.
- Sin capas Dropout.

El entrenamiento:

- Finalizó en la época 16.
- Registró su menor pérdida de validación en la época 11.
- La menor pérdida de validación fue 0.5026.

### Configuración 2: EarlyStopping con Dropout

Arquitectura:

```text
Entrada: 61 valores
Capa oculta 1: 61 neuronas, ReLU
Dropout: 0.5
Capa oculta 2: aproximadamente 31 neuronas, ReLU
Dropout: 0.5
Salida: una neurona, Sigmoid
```

Configuración:

- Función de pérdida `binary_crossentropy`.
- Optimizador Adam.
- Máximo de 600 épocas.
- `validation_split=0.20`.
- EarlyStopping con paciencia de cinco épocas.
- Dos capas Dropout de 0.5.

El entrenamiento:

- Finalizó en la época 28.
- Registró su menor pérdida de validación en la época 23.
- La menor pérdida de validación fue 0.4872.

La configuración con Dropout alcanzó una pérdida de validación inferior:

```text
Sin Dropout: 0.5026
Con Dropout: 0.4872
```

La evaluación final registrada en el notebook corresponde a la última configuración entrenada, es decir, al modelo con Dropout.

### Resultados del modelo binario

Resultados globales:

```text
Accuracy:        0.77
F1 macro:        0.68
F1 weighted:     0.74
```

Resultados por clase:

| Clase | Precision | Recall | F1-score | Soporte |
|---|---:|---:|---:|---:|
| Bueno | 0.78 | 0.92 | 0.85 | 140 |
| Malo | 0.69 | 0.40 | 0.51 | 60 |

Matriz de confusión:

```text
[[129, 11],
 [ 36, 24]]
```

Interpretación:

- El modelo identificó correctamente 129 de los 140 créditos buenos.
- El recall de la clase Bueno fue 0.92.
- El modelo identificó correctamente 24 de los 60 créditos malos.
- El recall de la clase Malo fue 0.40.
- El modelo clasificó 36 créditos malos como buenos.

La principal limitación es la identificación de la clase Malo.

El modelo se guarda como:

```text
models/modelo_pred_binomial.keras
```

---

## Modelo ANN multiclase

El entrenamiento del modelo multiclase se encuentra en:

```text
notebooks/04_ANN_Modelo_2.ipynb
```

Se utiliza un modelo híbrido porque incorpora la probabilidad generada por el modelo binario.

La entrada final se construye mediante:

```text
61 entradas preprocesadas
+ 1 puntaje de exposición normalizado
+ 1 probabilidad generada por el modelo binario
= 63 entradas
```

### Arquitectura

```text
Entrada: 63 valores

Capa oculta 1:
126 neuronas
Activación ReLU
Regularización L2

Dropout:
0.3

Capa oculta 2:
63 neuronas
Activación ReLU
Regularización L2

Dropout:
0.3

Capa oculta 3:
31 neuronas
Activación ReLU
Regularización L2

Dropout:
0.3

Salida:
Cuatro neuronas
Activación Softmax
```

### Configuración de entrenamiento

- Función de pérdida `sparse_categorical_crossentropy`.
- Optimizador Adam.
- Learning rate inicial de 0.0005.
- Máximo de 200 épocas.
- Batch size de 60.
- Validación del 20 %.
- Pesos de clase balanceados.
- EarlyStopping con paciencia de 15 épocas.
- Recuperación de los mejores pesos.
- ReduceLROnPlateau con paciencia de 10 épocas.

El entrenamiento:

- Finalizó en la época 71.
- Restauró los pesos correspondientes a la época 56.
- La época 56 presentó la mejor pérdida de validación.

### Resultados del modelo multiclase

Resultados globales:

```text
Loss:             0.7403
Accuracy:         0.7200
F1 macro:         0.65
F1 weighted:      0.71
ROC-AUC macro:    0.8756
```

Resultados por nivel:

| Nivel | Precision | Recall | F1-score | Soporte |
|---|---:|---:|---:|---:|
| Bajo | 0.85 | 0.88 | 0.86 | 101 |
| Medio | 0.56 | 0.46 | 0.51 | 39 |
| Alto | 0.57 | 0.50 | 0.53 | 34 |
| Crítico | 0.61 | 0.77 | 0.68 | 26 |

Interpretación:

- El nivel Bajo presentó el mejor F1-score, con 0.86.
- El nivel Crítico alcanzó un recall de 0.77.
- Los niveles Medio y Alto fueron las categorías más difíciles de diferenciar.
- El nivel Medio obtuvo un F1-score de 0.51.
- El nivel Alto obtuvo un F1-score de 0.53.
- Los pesos de clase ayudan a dar mayor importancia a las clases menos frecuentes.

El modelo se guarda como:

```text
models/modelo_riesgo_multiclase.keras
```

---

## Comparación de modelos

La comparación se encuentra en:

```text
notebooks/05_Comparacion_Modelos.ipynb
```

Este notebook utiliza los resultados registrados durante los entrenamientos originales. No vuelve a entrenar los modelos.

Incluye:

- Comparación de las configuraciones binarias con y sin Dropout.
- Comparación de la pérdida de validación.
- Comparación de la cantidad de épocas.
- Métricas por clase del modelo binario.
- Matriz de confusión binaria.
- Métricas por categoría del modelo multiclase.
- Comparación funcional de los modelos finales.
- Conclusiones y limitaciones.

### Comparación binaria

| Configuración | Dropout | Épocas ejecutadas | Mejor época | Mejor val_loss |
|---|---|---:|---:|---:|
| EarlyStopping sin Dropout | No | 16 | 11 | 0.5026 |
| EarlyStopping con Dropout | 0.5 | 28 | 23 | 0.4872 |

La configuración con Dropout obtuvo la menor pérdida de validación.

Una limitación es que el notebook original no conservó una evaluación independiente sobre prueba para la configuración sin Dropout. Por ello, la comparación directa entre ambas configuraciones se basa en:

- Las curvas de pérdida.
- La mejor pérdida de validación.
- La cantidad de épocas ejecutadas.
- El comportamiento observado durante el entrenamiento.

### Diferencia entre los modelos finales

El modelo binario y el multiclase no se comparan para decidir cuál es mejor porque resuelven objetivos diferentes.

El modelo binario responde:

> ¿El crédito se clasifica como Bueno o Malo?

El modelo multiclase responde:

> ¿El nivel de riesgo se clasifica como Bajo, Medio, Alto o Crítico?

Ambos modelos son complementarios porque la probabilidad del modelo binario se utiliza como entrada adicional del modelo multiclase.

---

## API REST

La API se implementa con FastAPI y se encuentra en:

```text
api/main.py
```

La API carga:

- El modelo binario.
- El modelo multiclase.
- El escalador.
- Los nombres y el orden de las 61 entradas.
- Las reglas utilizadas para construir el puntaje de exposición.

La API no vuelve a entrenar los modelos.

Su función es:

1. Recibir las 20 variables originales.
2. Validar los datos recibidos.
3. Transformar las variables categóricas.
4. Alinear las 61 entradas.
5. Aplicar el escalador.
6. Ejecutar el modelo binario.
7. Calcular el puntaje de exposición.
8. Construir las 63 entradas multiclase.
9. Ejecutar el modelo multiclase.
10. Devolver una respuesta en formato JSON.

### Endpoints

```text
GET  /
GET  /columnas
POST /predecir/binario
POST /predecir/multiclase
```

### Estado de la API

```text
GET /
```

Permite comprobar si la API está activa y si los modelos fueron cargados.

### Columnas esperadas

```text
GET /columnas
```

Permite consultar información sobre las entradas utilizadas por los modelos.

### Clasificación binaria

```text
POST /predecir/binario
```

Devuelve:

- Predicción numérica.
- Etiqueta Bueno o Malo.
- Probabilidad estimada de crédito malo.
- Confianza de la clasificación.

Ejemplo de estructura:

```json
{
  "prediccion": 0,
  "etiqueta": "Bueno",
  "probabilidad_malo": 0.072,
  "confianza": 0.928
}
```

### Clasificación multiclase

```text
POST /predecir/multiclase
```

Devuelve:

- Predicción numérica.
- Etiqueta Bajo, Medio, Alto o Crítico.
- Confianza.
- Probabilidades de cada nivel.
- Puntaje de exposición.
- Probabilidad generada por el modelo binario.

Ejemplo de estructura:

```json
{
  "prediccion": 0,
  "etiqueta": "Bajo",
  "confianza": 0.4452,
  "probabilidades": {
    "Bajo": 0.4452,
    "Medio": 0.2451,
    "Alto": 0.2956,
    "Crítico": 0.0141
  },
  "puntaje_exposicion": 4,
  "prob_binaria": 0.072
}
```

La documentación interactiva se encuentra en:

```text
http://127.0.0.1:8000/docs
```

---

## Interfaz Streamlit

La interfaz se encuentra en:

```text
app/streamlit_app.py
```

La aplicación incluye:

- Presentación general del proyecto.
- Descripción del dataset.
- Resultados del análisis exploratorio.
- Explicación del preprocesamiento.
- Programación orientada a objetos.
- Arquitecturas y métricas de los modelos.
- Explicación del funcionamiento de la API.
- Formulario con las 20 variables originales.
- Predicción binaria.
- Predicción multiclase.
- Confianza de ambos modelos.
- Puntaje de exposición.
- Gráfico de probabilidades por nivel.
- Conclusiones y limitaciones.

El diseño visual se configura desde:

```text
.streamlit/config.toml
```

Streamlit no realiza directamente la predicción.

El flujo es:

1. El usuario completa las 20 variables.
2. Streamlit construye una solicitud JSON.
3. Streamlit envía la solicitud a FastAPI.
4. FastAPI realiza el preprocesamiento.
5. FastAPI ejecuta los dos modelos.
6. Streamlit presenta los resultados.

---

## Flujo completo del sistema

```text
Usuario
→ formulario de Streamlit
→ JSON con 20 variables originales
→ FastAPI
→ validación mediante Pydantic
→ get_dummies
→ alineación de 61 entradas
→ MinMaxScaler
→ modelo binario
→ probabilidad de crédito malo
→ puntaje de exposición normalizado
→ 61 + 1 + 1 = 63 entradas
→ modelo multiclase
→ respuesta JSON
→ visualización en Streamlit
```

---

## Requisitos

Se recomienda utilizar Python 3.11.

Las dependencias se encuentran en:

```text
requirements.txt
```

Para instalarlas:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Instalación

### 1. Clonar o descargar el proyecto

Ubicarse en la carpeta donde se almacenará el proyecto.

### 2. Crear un entorno virtual

En Windows PowerShell:

```powershell
py -3.11 -m venv .venv311
```

### 3. Activar el entorno virtual

```powershell
.\.venv311\Scripts\Activate.ps1
```

La terminal debe mostrar:

```text
(.venv311)
```

### 4. Instalar las dependencias

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Ejecución

Los comandos deben ejecutarse desde la raíz del proyecto:

```text
Proyecto_Riesgo_Crediticio
```

### 1. Ejecutar FastAPI

En una primera terminal:

```powershell
uvicorn api.main:app --reload --port 8000
```

La terminal debe mostrar:

```text
Modelos y artefactos cargados correctamente.
Application startup complete.
```

La documentación interactiva estará disponible en:

```text
http://127.0.0.1:8000/docs
```

La terminal de FastAPI debe permanecer abierta.

### 2. Ejecutar Streamlit

En una segunda terminal:

```powershell
streamlit run app/streamlit_app.py
```

Streamlit estará disponible normalmente en:

```text
http://localhost:8501
```

La aplicación debe mostrar:

```text
API y modelos disponibles
```

antes de realizar las predicciones.

---

## Ejecución del preprocesamiento

La preparación automatizada se encuentra en:

```text
src/data_prep.py
```

El archivo de prueba se encuentra en:

```text
src/probar_data_prep.py
``
# Proyecto A: Sistema de Predicción de Riesgo Crediticio

Proyecto desarrollado para el curso de **Inteligencia Artificial Aplicada**.

El sistema utiliza el conjunto de datos **German Credit** para preparar y
entrenar dos modelos de redes neuronales artificiales:

1. Un modelo de clasificación binaria:
   - Crédito bueno.
   - Crédito malo.

2. Un modelo de clasificación multiclase:
   - Riesgo Bajo.
   - Riesgo Medio.
   - Riesgo Alto.
   - Riesgo Crítico.

El proyecto incluye análisis exploratorio, preprocesamiento, programación
orientada a objetos, modelos ANN, una API REST y una interfaz para realizar
predicciones.

---

## Integrantes

- Sebastián Calvo Solano
- Carlos Cambronero Zuñiga
- Josué Francisco Redondo Gómez
- Steven Zamora Reyes

---

## Objetivo general

Desarrollar un sistema de inteligencia artificial capaz de analizar la
información de una solicitud de crédito y generar una predicción de riesgo
mediante redes neuronales artificiales.

---

## Objetivos específicos

- Analizar la estructura y calidad del conjunto German Credit.
- Identificar relaciones entre las características crediticias y la
  clasificación del crédito.
- Preparar las variables numéricas y categóricas para los modelos.
- Construir una variable objetivo binaria.
- Construir y justificar cuatro niveles académicos de riesgo.
- Implementar el preprocesamiento mediante programación orientada a objetos.
- Entrenar y comparar diferentes configuraciones de redes neuronales.
- Exponer las predicciones mediante una API REST.
- Desarrollar una interfaz sencilla para ingresar datos y consultar resultados.

---

## Dataset

El proyecto utiliza el conjunto de datos **German Credit**.

El archivo principal es:

```text
data/raw/german.data
```

El conjunto contiene:

- 1.000 registros.
- 20 variables de entrada.
- 1 variable objetivo.
- 7 variables numéricas.
- 13 variables categóricas.

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

El archivo `german.data` se utiliza como fuente principal porque conserva las
variables categóricas originales y permite aplicar `get_dummies` durante el
preprocesamiento.

---

## Estructura del proyecto

```text
Proyecto_Riesgo_Crediticio/
│
├── api/
│   └── Archivos de la API REST
│
├── app/
│   └── Archivos de la interfaz
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
│   ├── scaler.pkl
│   ├── columnas_modelo.pkl
│   ├── informacion_preprocesamiento.pkl
│   ├── regla_multiclase.pkl
│   └── Modelos entrenados en formato .keras
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocesamiento.ipynb
│   └── Notebooks de entrenamiento
│
├── src/
│   ├── data_prep.py
│   ├── probar_data_prep.py
│   └── train/
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Análisis exploratorio de datos

El análisis exploratorio se encuentra en:

```text
notebooks/01_EDA.ipynb
```

El notebook incluye:

- Carga y descripción del dataset.
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
1 variable objetivo
0 valores nulos
0 registros duplicados
```

La distribución de la variable objetivo es:

```text
Créditos buenos: 700 registros, 70 %
Créditos malos:  300 registros, 30 %
```

Debido a esta distribución, la evaluación de los modelos no debe depender
exclusivamente de la exactitud.

También deben considerarse métricas como:

- Precisión.
- Recall.
- F1-score.
- Matriz de confusión.

---

## Decisión sobre las variables

Se conservaron las 20 variables originales para construir los modelos base.

Antes de tomar esta decisión se comprobó que:

- No existen identificadores de clientes.
- No hay columnas duplicadas.
- No hay variables constantes.
- No existen valores nulos.
- Ninguna entrada contiene directamente la respuesta.
- Las variables describen diferentes aspectos personales, laborales y
  crediticios.

Después de aplicar `get_dummies`, las 20 variables originales quedan
representadas mediante 61 columnas numéricas.

Estas 61 columnas no representan 61 variables originales independientes.
Corresponden a las siete variables numéricas y a la representación de las
13 variables categóricas.

---

## Preprocesamiento

El procedimiento detallado se encuentra en:

```text
notebooks/02_Preprocesamiento.ipynb
```

Las etapas realizadas son:

1. Carga de `german.data`.
2. Verificación de calidad.
3. Separación de variables numéricas y categóricas.
4. Transformación de la variable objetivo binaria.
5. Separación de entradas y salida.
6. Conversión de categorías mediante `pd.get_dummies`.
7. División en entrenamiento y prueba.
8. Normalización mediante `MinMaxScaler`.
9. Creación de los cuatro niveles de riesgo.
10. Guardado de los datos y objetos de transformación.
11. Verificación de dimensiones y archivos.

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

El valor `1` representa el resultado de riesgo malo, por lo que la probabilidad
del modelo puede interpretarse como la probabilidad estimada de un crédito
malo.

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

La proporción en ambos conjuntos permanece en:

```text
70 % créditos buenos
30 % créditos malos
```

El conjunto de prueba debe reservarse para la evaluación final. Durante el
entrenamiento, la validación debe obtenerse mediante `validation_split` sobre
el conjunto de entrenamiento.

---

## Normalización

Se utiliza `MinMaxScaler` para transformar los valores a una escala adecuada
para las redes neuronales.

El escalador se ajusta únicamente con entrenamiento:

```python
scaler.fit_transform(X_train)
```

El conjunto de prueba se transforma utilizando el mismo escalador:

```python
scaler.transform(X_test)
```

Esto evita que la información del conjunto de prueba influya en la preparación
del entrenamiento.

---

## Clasificación multiclase

El dataset original contiene únicamente las clases Bueno y Malo.

Para cumplir con el modelo multiclase se construyeron cuatro niveles académicos:

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

La división utiliza 
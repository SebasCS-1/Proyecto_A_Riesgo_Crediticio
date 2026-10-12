from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Sistema de Riesgo Crediticio",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --azul-oscuro: #17324D;
        --azul: #2E6F95;
        --azul-claro: #DCEAF4;
        --fondo: #E8EFF5;
        --tarjeta: #F7FAFC;
        --texto: #172533;
        --borde: #B8CBD9;
    }

    .stApp {
        background: linear-gradient(180deg, #E8EFF5 0%, #DDE8F0 100%);
        color: var(--texto);
    }

    [data-testid="stHeader"] {
        background: #E8EFF5;
    }

    .block-container {
        max-width: 1380px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: var(--azul-oscuro);
        border-right: 1px solid var(--azul);
    }

    [data-testid="stSidebar"] * {
        color: #F7FAFC !important;
    }

    h1, h2, h3, h4, p, li, label {
        color: var(--texto);
    }

    .hero {
        background: linear-gradient(125deg, #17324D, #2E6F95);
        color: white;
        padding: 2.3rem;
        border-radius: 18px;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 25px rgba(23, 50, 77, 0.20);
    }

    .hero h1, .hero p {
        color: white !important;
    }

    .hero h1 {
        margin: 0 0 0.5rem 0;
        font-size: 2.35rem;
    }

    .hero p {
        margin: 0;
        font-size: 1.04rem;
        opacity: 0.96;
    }

    .card, .step {
        background: var(--tarjeta);
        color: var(--texto) !important;
        border: 1px solid var(--borde);
        border-radius: 14px;
        padding: 1.1rem;
        box-shadow: 0 4px 14px rgba(23, 50, 77, 0.08);
    }

    .card h3, .card p, .step {
        color: var(--texto) !important;
    }

    .step {
        border-left: 5px solid var(--azul);
        margin: 0.5rem 0;
    }

    div[data-testid="stMetric"] {
        background: var(--tarjeta);
        border: 1px solid var(--borde);
        padding: 0.85rem;
        border-radius: 13px;
        box-shadow: 0 3px 12px rgba(23, 50, 77, 0.07);
    }

    div[data-testid="stMetric"] * {
        color: var(--texto) !important;
    }

    button[data-baseweb="tab"] {
        color: var(--azul-oscuro) !important;
        font-weight: 700;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: var(--azul-claro) !important;
        border-radius: 9px 9px 0 0;
    }

    [data-testid="stDataFrame"], [data-testid="stTable"] {
        background: var(--tarjeta) !important;
        border: 1px solid var(--borde);
        border-radius: 10px;
        overflow: hidden;
    }

    [data-testid="stForm"] {
        background: var(--tarjeta);
        border: 1px solid var(--borde);
        padding: 1.25rem;
        border-radius: 15px;
    }

    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    input {
        background: white !important;
        color: var(--texto) !important;
    }

    .resultado-bajo {
        padding: 1rem;
        border-radius: 12px;
        background: #DDEFE4;
        color: #173B2A !important;
        border: 1px solid #7ABB91;
    }

    .resultado-medio {
        padding: 1rem;
        border-radius: 12px;
        background: #F6EDCE;
        color: #4A3512 !important;
        border: 1px solid #C9A34B;
    }

    .resultado-alto {
        padding: 1rem;
        border-radius: 12px;
        background: #F2E1D0;
        color: #533019 !important;
        border: 1px solid #C98954;
    }

    .resultado-critico {
        padding: 1rem;
        border-radius: 12px;
        background: #F1DCD9;
        color: #4A1E18 !important;
        border: 1px solid #C56A5F;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CATÁLOGOS DE LAS VARIABLES CATEGÓRICAS
# ============================================================

CODIGOS = {
    "estado_cuenta": {
        "A11": "Menos de 0 DM",
        "A12": "Entre 0 y 200 DM",
        "A13": "Más de 200 DM",
        "A14": "Sin cuenta corriente",
    },
    "historial_crediticio": {
        "A30": "Sin créditos o créditos pagados",
        "A31": "Créditos pagados en otros bancos",
        "A32": "Créditos pagados puntualmente",
        "A33": "Mora en el pasado",
        "A34": "Cuenta crítica u otros créditos",
    },
    "proposito": {
        "A40": "Automóvil nuevo",
        "A41": "Automóvil usado",
        "A42": "Muebles o equipo",
        "A43": "Radio o televisión",
        "A44": "Electrodomésticos",
        "A45": "Reparaciones",
        "A46": "Educación",
        "A48": "Reentrenamiento",
        "A49": "Negocio",
        "A410": "Otros",
    },
    "cuenta_ahorros": {
        "A61": "Menos de 100 DM",
        "A62": "Entre 100 y 500 DM",
        "A63": "Entre 500 y 1.000 DM",
        "A64": "Más de 1.000 DM",
        "A65": "Sin cuenta de ahorros",
    },
    "antiguedad_empleo": {
        "A71": "Desempleado",
        "A72": "Menos de un año",
        "A73": "Entre uno y cuatro años",
        "A74": "Entre cuatro y siete años",
        "A75": "Más de siete años",
    },
    "estado_personal": {
        "A91": "Divorciado o separado",
        "A92": "Divorciada, separada o casada",
        "A93": "Soltero",
        "A94": "Casado o viudo",
    },
    "otros_deudores": {
        "A101": "Ninguno",
        "A102": "Co-solicitante",
        "A103": "Garante",
    },
    "propiedad": {
        "A121": "Bienes raíces",
        "A122": "Seguro de vida",
        "A123": "Automóvil u otros",
        "A124": "Sin propiedad",
    },
    "otros_planes_pago": {
        "A141": "Banco",
        "A142": "Tiendas",
        "A143": "Ninguno",
    },
    "vivienda": {
        "A151": "Alquiler",
        "A152": "Propia",
        "A153": "Gratuita",
    },
    "trabajo": {
        "A171": "Desempleado o no residente",
        "A172": "No calificado residente",
        "A173": "Empleado calificado",
        "A174": "Directivo o universitario",
    },
    "telefono": {
        "A191": "Sin teléfono",
        "A192": "Registrado a nombre del cliente",
    },
    "trabajador_extranjero": {
        "A201": "Sí",
        "A202": "No",
    },
}

VALORES_INICIALES = {
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
}


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================


def selector(campo: str, etiqueta: str) -> str:
    opciones = list(CODIGOS[campo].keys())
    valor_inicial = VALORES_INICIALES[campo]

    return st.selectbox(
        etiqueta,
        options=opciones,
        index=opciones.index(valor_inicial),
        format_func=lambda codigo: f"{CODIGOS[campo][codigo]} ({codigo})",
    )


@st.cache_data(ttl=5)
def consultar_estado_api() -> dict[str, Any] | None:
    try:
        respuesta = requests.get(f"{API_URL}/", timeout=2)
        respuesta.raise_for_status()
        return respuesta.json()
    except requests.RequestException:
        return None


def llamar_api(ruta: str, datos: dict[str, Any]) -> dict[str, Any]:
    respuesta = requests.post(
        f"{API_URL}{ruta}",
        json=datos,
        timeout=30,
    )

    if respuesta.status_code != 200:
        try:
            detalle = respuesta.json().get("detail", respuesta.text)
        except ValueError:
            detalle = respuesta.text

        raise RuntimeError(
            f"La API respondió con el código {respuesta.status_code}: {detalle}"
        )

    return respuesta.json()


# ============================================================
# BARRA LATERAL
# ============================================================

with st.sidebar:
    st.header("Contenido del proyecto")

    st.markdown(
        """
        1. Inicio y objetivo
        2. Dataset
        3. Análisis exploratorio
        4. Preprocesamiento
        5. Programación orientada a objetos
        6. Modelos ANN
        7. API y arquitectura
        8. Predicción interactiva
        9. Conclusiones
        """
    )

    st.divider()

    estado_api = consultar_estado_api()

    if estado_api is not None:
        st.success("API disponible")
    else:
        st.error("API no disponible")
        st.caption(
            "Ejecuta: uvicorn api.main:app --reload --port 8000"
        )


# ============================================================
# ENCABEZADO Y NAVEGACIÓN
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>Sistema de predicción de riesgo crediticio</h1>
        <p>
            Análisis exploratorio, preprocesamiento, programación orientada
            a objetos, redes neuronales, FastAPI y Streamlit.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

(
    tab_inicio,
    tab_dataset,
    tab_eda,
    tab_preprocesamiento,
    tab_poo,
    tab_modelos,
    tab_api,
    tab_prediccion,
    tab_conclusiones,
) = st.tabs(
    [
        "Inicio",
        "Dataset",
        "EDA",
        "Preprocesamiento",
        "POO",
        "Modelos",
        "API",
        "Predicción",
        "Conclusiones",
    ]
)


# ============================================================
# 1. INICIO
# ============================================================

with tab_inicio:
    st.header("Objetivo y alcance")

    st.write(
        "El sistema analiza información crediticia y produce dos resultados: "
        "una clasificación binaria, Bueno o Malo, y un nivel de riesgo, "
        "Bajo, Medio, Alto o Crítico."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", "1.000")
    c2.metric("Variables originales", "20")
    c3.metric("Entradas binarias", "61")
    c4.metric("Modelos ANN", "2")

    st.subheader("Flujo general")

    pasos = [
        "German Credit: 1.000 registros, 20 variables de entrada y una salida.",
        "EDA: calidad, distribuciones, balance de clases y relaciones.",
        "Preprocesamiento: get_dummies, división estratificada y MinMaxScaler.",
        "POO: automatización mediante la clase PreparadorDatos.",
        "ANN binaria: predicción de crédito Bueno o Malo.",
        "ANN multiclase: predicción de riesgo Bajo, Medio, Alto o Crítico.",
        "FastAPI: validación, transformación y ejecución de los modelos.",
        "Streamlit: presentación del proyecto y predicción interactiva.",
    ]

    for paso in pasos:
        st.markdown(
            f'<div class="step">{paso}</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# 2. DATASET
# ============================================================

with tab_dataset:
    st.header("Comprensión del dataset")

    st.write(
        "Cada fila representa un registro crediticio. Las columnas representan "
        "atributos del cliente o del crédito. El archivo original contiene "
        "20 variables de entrada y una variable objetivo, para un total de "
        "21 columnas."
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="card">
                <h3>Variables numéricas</h3>
                <p>
                    Duración, monto, tasa de cuotas, tiempo de residencia,
                    edad, créditos existentes y personas dependientes.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="card">
                <h3>Variables categóricas</h3>
                <p>
                    Estado de cuenta, historial, propósito, ahorros, empleo,
                    estado personal, propiedad, vivienda, trabajo y otras
                    condiciones representadas mediante códigos.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Archivos utilizados")

    archivos_dataset = pd.DataFrame(
        {
            "Archivo": [
                "german.data",
                "german.data-numeric",
                "german.doc",
                "Index",
            ],
            "Función": [
                "Fuente principal con las 20 variables originales",
                "Versión numérica previamente transformada",
                "Documentación de los atributos y códigos",
                "Información general de los archivos",
            ],
        }
    )

    st.dataframe(
        archivos_dataset,
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Decisión sobre la fuente principal")

    st.write(
        "Se utilizó german.data porque conserva las categorías originales y "
        "permite demostrar su transformación mediante get_dummies. La última "
        "columna ya contenía la respuesta. Se le asignó el nombre clase sin "
        "modificar sus valores."
    )

    st.info(
        "german.data contiene los datos y german.doc explica el significado "
        "de los códigos como A11, A43 o A61."
    )


# ============================================================
# 3. EDA
# ============================================================

with tab_eda:
    st.header("Análisis exploratorio de datos")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Valores nulos", "0")
    m2.metric("Duplicados", "0")
    m3.metric("Créditos buenos", "700")
    m4.metric("Créditos malos", "300")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Distribución de la clase")

        distribucion_clase = pd.DataFrame(
            {
                "Clase": ["Bueno", "Malo"],
                "Registros": [700, 300],
            }
        )

        st.bar_chart(
            distribucion_clase,
            x="Clase",
            y="Registros",
            color="#2E6F95",
        )

        st.caption(
            "La distribución 70/30 demuestra que la exactitud no debe "
            "interpretarse de forma aislada."
        )

    with c2:
        st.subheader("Promedios por clase")

        promedios = pd.DataFrame(
            {
                "Grupo": ["Bueno", "Malo"],
                "Duración promedio": [19.21, 24.86],
                "Edad promedio": [36.22, 33.96],
            }
        ).set_index("Grupo")

        st.bar_chart(promedios)

    st.subheader("Monto promedio del crédito")

    montos = pd.DataFrame(
        {
            "Grupo": ["Bueno", "Malo"],
            "Monto promedio": [2985.46, 3938.13],
        }
    )

    st.bar_chart(
        montos,
        x="Grupo",
        y="Monto promedio",
        color="#5A879F",
    )

    st.subheader("Hallazgos principales")

    st.markdown(
        """
        - El conjunto contiene 1.000 registros completos.
        - No se encontraron valores nulos ni registros duplicados.
        - La clase objetivo presenta una distribución de 70 % Bueno y 30 % Malo.
        - Los créditos malos presentan mayor duración promedio.
        - Los créditos malos presentan mayor monto promedio.
        - La correlación numérica más destacada fue entre duración y monto,
          con un valor aproximado de 0,62.
        - Estado de cuenta, ahorros e historial mostraron diferencias entre
          créditos buenos y malos.
        - Las relaciones observadas representan asociaciones y no causalidad.
        """
    )


# ============================================================
# 4. PREPROCESAMIENTO
# ============================================================

with tab_preprocesamiento:
    st.header("Preprocesamiento")

    etapas = pd.DataFrame(
        {
            "Etapa": [
                "Objetivo binario",
                "Separación",
                "Codificación",
                "División",
                "Escalado",
                "Salida multiclase",
                "Persistencia",
            ],
            "Procedimiento": [
                "1/2 original se representa como 0=Bueno y 1=Malo",
                "X contiene entradas; y contiene la respuesta",
                "13 variables categóricas se convierten con get_dummies",
                "80 % entrenamiento y 20 % prueba",
                "MinMaxScaler se ajusta solo con entrenamiento",
                "Se construyen Bajo, Medio, Alto y Crítico",
                "Se guardan CSV, scaler, columnas y reglas",
            ],
            "Razón": [
                "Compatibilidad con la salida Sigmoid",
                "Evitar fuga de información",
                "La red necesita entradas numéricas",
                "Evaluar generalización con datos separados",
                "Evitar que prueba influya en la preparación",
                "Crear el objetivo solicitado para el segundo modelo",
                "Reproducir exactamente el procedimiento",
            ],
        }
    )

    st.dataframe(
        etapas,
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("De 20 variables a 61 entradas")

    st.write(
        "get_dummies aplica codificación disyuntiva. Cada categoría se "
        "convierte en una columna de cero o uno. Las 61 entradas siguen "
        "representando las 20 variables originales."
    )

    ejemplo_dummies = pd.DataFrame(
        {
            "Registro": ["Cliente A", "Cliente B"],
            "Propósito original": ["A43", "A45"],
            "proposito_A43": [1, 0],
            "proposito_A45": [0, 1],
        }
    )

    st.dataframe(
        ejemplo_dummies,
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("División y normalización")

    d1, d2, d3 = st.columns(3)
    d1.metric("Entrenamiento", "800 registros")
    d2.metric("Prueba", "200 registros")
    d3.metric("Entradas binarias", "61")

    st.markdown(
        """
        - `random_state=101` permite reproducir la misma división.
        - `stratify=y` conserva la proporción 70/30.
        - `fit_transform` se utiliza únicamente con entrenamiento.
        - `transform` aplica la misma escala al conjunto de prueba.
        """
    )

    st.subheader("Construcción multiclase")

    st.write(
        "El dataset original solo contenía Bueno y Malo. Para construir los "
        "cuatro niveles se conservó la clase original como base y se añadió "
        "un puntaje académico de exposición entre cero y cinco."
    )

    condiciones = pd.DataFrame(
        {
            "Condición que suma un punto": [
                "Duración superior a la mediana de entrenamiento",
                "Monto superior a la mediana de entrenamiento",
                "Estado de cuenta A11 o A12",
                "Cuenta de ahorros A61 o A62",
                "Historial crediticio A30 o A31",
            ]
        }
    )

    st.dataframe(
        condiciones,
        hide_index=True,
        use_container_width=True,
    )

    st.code(
        """
Bueno + puntaje menor o igual al umbral  -> Bajo
Bueno + puntaje superior al umbral      -> Medio
Malo  + puntaje menor o igual al umbral -> Alto
Malo  + puntaje superior al umbral      -> Crítico
        """,
        language="text",
    )

    st.write(
        "La POO genera 61 entradas normalizadas más el puntaje de exposición "
        "normalizado, para un total de 62 columnas multiclase. El notebook y "
        "la API agregan después la probabilidad del modelo binario y forman "
        "las 63 entradas finales."
    )

    st.warning(
        "Los cuatro niveles son una construcción académica. No representan "
        "una escala bancaria oficial."
    )


# ============================================================
# 5. PROGRAMACIÓN ORIENTADA A OBJETOS
# ============================================================

with tab_poo:
    st.header("Programación orientada a objetos")

    st.write(
        "El notebook muestra el procedimiento paso a paso. La clase "
        "PreparadorDatos automatiza ese mismo proceso mediante atributos y "
        "métodos reutilizables."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="card">
                <h3>Clase</h3>
                <p>
                    PreparadorDatos representa el componente encargado de
                    transformar German Credit.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="card">
                <h3>Atributos</h3>
                <p>
                    Guardan rutas, columnas, DataFrames, escalador, medianas,
                    umbrales y resultados intermedios.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="card">
                <h3>Métodos</h3>
                <p>
                    Cada método realiza una responsabilidad concreta del
                    preprocesamiento.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Proceso coordinado")

    flujo_poo = pd.DataFrame(
        {
            "Orden": list(range(1, 13)),
            "Método": [
                "cargar_datos",
                "verificar_datos",
                "crear_objetivo_binario",
                "separar_entradas_salida",
                "convertir_categorias",
                "dividir_datos",
                "normalizar_datos",
                "crear_objetivo_multiclase",
                "verificar_coherencia_multiclase",
                "guardar_resultados",
                "verificar_archivos",
                "verificar_dimensiones",
            ],
            "Resultado": [
                "Lee german.data y asigna 21 nombres",
                "Comprueba nulos y duplicados",
                "Transforma 1/2 a 0/1",
                "Separa X e y y evita fuga de información",
                "Aplica get_dummies y obtiene 61 entradas",
                "Divide 80/20 con stratify",
                "Ajusta MinMaxScaler solo con entrenamiento",
                "Crea los cuatro niveles y agrega el puntaje",
                "Valida la relación binaria y multiclase",
                "Guarda CSV y artefactos .pkl",
                "Comprueba los archivos esperados",
                "Comprueba 61 entradas binarias y 62 multiclase",
            ],
        }
    )

    st.dataframe(
        flujo_poo,
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Dimensiones verificadas por la clase")

    dimensiones_poo = pd.DataFrame(
        {
            "Conjunto": [
                "X_train_binario",
                "X_test_binario",
                "X_train_multiclase",
                "X_test_multiclase",
            ],
            "Dimensiones": [
                "800 x 61",
                "200 x 61",
                "800 x 62",
                "200 x 62",
            ],
        }
    )

    st.dataframe(
        dimensiones_poo,
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Ventajas de la POO")

    st.markdown(
        """
        - Evita repetir el preprocesamiento.
        - Centraliza rutas, columnas y transformaciones.
        - Obliga a ejecutar las operaciones en el orden correcto.
        - Incluye controles de calidad y mensajes de error.
        - Permite regenerar los mismos archivos procesados.
        """
    )


# ============================================================
# 6. MODELOS
# ============================================================

with tab_modelos:
    st.header("Modelos de redes neuronales")

    modelo_binario_tab, modelo_multiclase_tab = st.tabs(
        ["ANN binaria", "ANN multiclase híbrida"]
    )

    with modelo_binario_tab:
        st.subheader("Clasificación binaria: Bueno o Malo")

        st.write(
            "El notebook binario evaluó dos configuraciones. La primera "
            "utilizó EarlyStopping sin Dropout. La segunda utilizó "
            "EarlyStopping y dos capas Dropout de 0.5. La evaluación final "
            "corresponde a la segunda configuración."
        )

        comparacion_binaria = pd.DataFrame(
            {
                "Configuración": [
                    "EarlyStopping sin Dropout",
                    "EarlyStopping con Dropout",
                ],
                "Arquitectura": [
                    "61 - 20 - 1",
                    "61 - 31 - 1",
                ],
                "Dropout": [
                    "No utiliza",
                    "0.5 después de cada capa oculta",
                ],
                "Épocas ejecutadas": [16, 28],
                "Mejor época": [11, 23],
                "Mejor val_loss": [0.5026, 0.4872],
            }
        )

        st.dataframe(
            comparacion_binaria,
            hide_index=True,
            use_container_width=True,
        )

        grafico_val_loss = comparacion_binaria[
            ["Configuración", "Mejor val_loss"]
        ].set_index("Configuración")

        st.subheader("Mejor pérdida de validación")
        st.bar_chart(grafico_val_loss, color="#2E6F95")

        st.info(
            "La configuración con Dropout obtuvo la menor pérdida de "
            "validación y corresponde al modelo evaluado y guardado."
        )

        st.subheader("Resultados del modelo binario final")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", "0,77")
        m2.metric("Recall Bueno", "0,92")
        m3.metric("Recall Malo", "0,40")
        m4.metric("F1 macro", "0,68")

        metricas_binarias = pd.DataFrame(
            {
                "Clase": ["Bueno", "Malo"],
                "Precision": [0.78, 0.69],
                "Recall": [0.92, 0.40],
                "F1-score": [0.85, 0.51],
                "Soporte": [140, 60],
            }
        )

        st.dataframe(
            metricas_binarias,
            hide_index=True,
            use_container_width=True,
        )

        st.bar_chart(
            metricas_binarias.set_index("Clase")[
                ["Precision", "Recall", "F1-score"]
            ]
        )

        st.subheader("Matriz de confusión")

        matriz_binaria = pd.DataFrame(
            [[129, 11], [36, 24]],
            index=["Real Bueno", "Real Malo"],
            columns=["Predicho Bueno", "Predicho Malo"],
        )

        st.dataframe(
            matriz_binaria,
            use_container_width=True,
        )

        st.warning(
            "La principal limitación es el recall de la clase Malo. El "
            "modelo identificó correctamente 24 de los 60 créditos malos."
        )

    with modelo_multiclase_tab:
        st.subheader("Clasificación multiclase híbrida")

        st.write(
            "El modelo multiclase utiliza como entradas las 61 variables "
            "preprocesadas, el puntaje de exposición normalizado y la "
            "probabilidad producida por el modelo binario."
        )

        st.code(
            "61 entradas + puntaje_exposicion + prob_binaria = 63 entradas",
            language="text",
        )

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Entradas", "63")
        a2.metric("Accuracy", "0,72")
        a3.metric("F1 macro", "0,65")
        a4.metric("ROC-AUC macro", "0,8756")

        arquitectura_multiclase = pd.DataFrame(
            {
                "Componente": [
                    "Capa oculta 1",
                    "Dropout 1",
                    "Capa oculta 2",
                    "Dropout 2",
                    "Capa oculta 3",
                    "Dropout 3",
                    "Salida",
                ],
                "Neuronas o valor": [
                    "126",
                    "0.30",
                    "63",
                    "0.30",
                    "31",
                    "0.30",
                    "4",
                ],
                "Configuración": [
                    "ReLU y L2",
                    "Regularización",
                    "ReLU y L2",
                    "Regularización",
                    "ReLU y L2",
                    "Regularización",
                    "Softmax",
                ],
            }
        )

        st.dataframe(
            arquitectura_multiclase,
            hide_index=True,
            use_container_width=True,
        )

        st.markdown(
            """
            - Pérdida: `sparse_categorical_crossentropy`.
            - Optimizador: Adam con learning rate de 0.0005.
            - Máximo de 200 épocas.
            - Batch size de 60.
            - Validación del 20 %.
            - Pesos de clase balanceados.
            - EarlyStopping con paciencia de 15 épocas.
            - ReduceLROnPlateau con paciencia de 10 épocas.
            - El entrenamiento finalizó en la época 71.
            - Se restauraron los pesos correspondientes a la época 56.
            """
        )

        metricas_multiclase = pd.DataFrame(
            {
                "Nivel": ["Bajo", "Medio", "Alto", "Crítico"],
                "Precision": [0.85, 0.56, 0.57, 0.61],
                "Recall": [0.88, 0.46, 0.50, 0.77],
                "F1-score": [0.86, 0.51, 0.53, 0.68],
                "Soporte": [101, 39, 34, 26],
            }
        )

        st.subheader("Métricas por nivel")

        st.dataframe(
            metricas_multiclase,
            hide_index=True,
            use_container_width=True,
        )

        st.bar_chart(
            metricas_multiclase.set_index("Nivel")[[
                "Precision",
                "Recall",
                "F1-score",
            ]]
        )

        st.info(
            "Bajo fue la categoría con mejor F1-score. Medio y Alto fueron "
            "los niveles más difíciles de diferenciar."
        )


# ============================================================
# 7. API Y ARQUITECTURA
# ============================================================

with tab_api:
    st.header("API y arquitectura de la solución")

    st.code(
        """
Usuario
  ↓ completa 20 variables originales
Streamlit
  ↓ envía un JSON
FastAPI
  ↓ valida los datos
get_dummies + alineación de 61 columnas
  ↓
MinMaxScaler
  ↓
Modelo binario
  ↓ produce probabilidad de Malo
Puntaje de exposición normalizado
  ↓
61 + 1 + 1 = 63 entradas
  ↓
Modelo multiclase
  ↓
Respuesta JSON
  ↓
Streamlit presenta los resultados
        """,
        language="text",
    )

    st.subheader("Artefactos utilizados")

    artefactos = pd.DataFrame(
        {
            "Artefacto": [
                "modelo_pred_binomial.keras",
                "modelo_riesgo_multiclase.keras",
                "scaler.pkl",
                "columnas_modelo.pkl",
                "informacion_preprocesamiento.pkl",
                "regla_multiclase.pkl",
            ],
            "Función": [
                "ANN binaria entrenada",
                "ANN multiclase entrenada",
                "Escala nuevos registros con la escala de entrenamiento",
                "Conserva el nombre y orden de las 61 entradas",
                "Describe variables originales y transformaciones",
                "Conserva medianas, umbrales y categorías del puntaje",
            ],
        }
    )

    st.dataframe(
        artefactos,
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Endpoints")

    endpoints = pd.DataFrame(
        {
            "Método": ["GET", "GET", "POST", "POST"],
            "Ruta": [
                "/",
                "/columnas",
                "/predecir/binario",
                "/predecir/multiclase",
            ],
            "Función": [
                "Comprobar el estado de la API",
                "Consultar información de las columnas",
                "Obtener la clasificación Bueno o Malo",
                "Obtener el nivel Bajo, Medio, Alto o Crítico",
            ],
        }
    )

    st.dataframe(
        endpoints,
        hide_index=True,
        use_container_width=True,
    )

    st.info(
        "La API carga los modelos y artefactos. No vuelve a entrenar las "
        "redes neuronales."
    )


# ============================================================
# 8. PREDICCIÓN INTERACTIVA
# ============================================================

with tab_prediccion:
    st.header("Predicción interactiva")

    st.write(
        "El formulario recibe las 20 variables originales. Streamlit envía "
        "los datos a FastAPI y presenta las respuestas de los dos modelos."
    )

    with st.form("formulario_credito"):
        st.subheader("Variables numéricas")

        n1, n2, n3 = st.columns(3)

        with n1:
            duracion_meses = st.number_input(
                "Duración del crédito en meses",
                min_value=1,
                max_value=72,
                value=24,
            )

            monto_credito = st.number_input(
                "Monto del crédito en DM",
                min_value=250,
                max_value=20000,
                value=4500,
            )

            tasa_cuotas = st.number_input(
                "Tasa de cuotas",
                min_value=1,
                max_value=4,
                value=2,
            )

        with n2:
            tiempo_residencia = st.number_input(
                "Tiempo de residencia",
                min_value=1,
                max_value=4,
                value=3,
            )

            edad = st.number_input(
                "Edad",
                min_value=18,
                max_value=90,
                value=35,
            )

        with n3:
            creditos_existentes = st.number_input(
                "Créditos existentes",
                min_value=1,
                max_value=4,
                value=1,
            )

            personas_dependientes = st.number_input(
                "Personas dependientes",
                min_value=1,
                max_value=2,
                value=1,
            )

        st.subheader("Variables categóricas")

        c1, c2, c3 = st.columns(3)

        with c1:
            estado_cuenta = selector(
                "estado_cuenta",
                "Estado de la cuenta corriente",
            )
            historial_crediticio = selector(
                "historial_crediticio",
                "Historial crediticio",
            )
            proposito = selector(
                "proposito",
                "Propósito del crédito",
            )
            cuenta_ahorros = selector(
                "cuenta_ahorros",
                "Cuenta de ahorros",
            )
            antiguedad_empleo = selector(
                "antiguedad_empleo",
                "Antigüedad en el empleo",
            )

        with c2:
            estado_personal = selector(
                "estado_personal",
                "Estado personal",
            )
            otros_deudores = selector(
                "otros_deudores",
                "Otros deudores o garantes",
            )
            propiedad = selector(
                "propiedad",
                "Propiedad",
            )
            otros_planes_pago = selector(
                "otros_planes_pago",
                "Otros planes de pago",
            )

        with c3:
            vivienda = selector(
                "vivienda",
                "Vivienda",
            )
            trabajo = selector(
                "trabajo",
                "Trabajo",
            )
            telefono = selector(
                "telefono",
                "Teléfono",
            )
            trabajador_extranjero = selector(
                "trabajador_extranjero",
                "Trabajador extranjero",
            )

        enviar = st.form_submit_button(
            "Analizar riesgo crediticio",
            type="primary",
            use_container_width=True,
        )

    if enviar:
        datos_cliente = {
            "duracion_meses": int(duracion_meses),
            "monto_credito": int(monto_credito),
            "tasa_cuotas": int(tasa_cuotas),
            "tiempo_residencia": int(tiempo_residencia),
            "edad": int(edad),
            "creditos_existentes": int(creditos_existentes),
            "personas_dependientes": int(personas_dependientes),
            "estado_cuenta": estado_cuenta,
            "historial_crediticio": historial_crediticio,
            "proposito": proposito,
            "cuenta_ahorros": cuenta_ahorros,
            "antiguedad_empleo": antiguedad_empleo,
            "estado_personal": estado_personal,
            "otros_deudores": otros_deudores,
            "propiedad": propiedad,
            "otros_planes_pago": otros_planes_pago,
            "vivienda": vivienda,
            "trabajo": trabajo,
            "telefono": telefono,
            "trabajador_extranjero": trabajador_extranjero,
        }

        try:
            with st.spinner("Procesando la solicitud..."):
                resultado_binario = llamar_api(
                    "/predecir/binario",
                    datos_cliente,
                )

                resultado_multiclase = llamar_api(
                    "/predecir/multiclase",
                    datos_cliente,
                )

            st.subheader("Resultados")

            r1, r2, r3, r4 = st.columns(4)
            r1.metric(
                "Clasificación binaria",
                resultado_binario["etiqueta"],
            )
            r2.metric(
                "Confianza binaria",
                f'{resultado_binario["confianza"] * 100:.2f} %',
            )
            r3.metric(
                "Nivel de riesgo",
                resultado_multiclase["etiqueta"],
            )
            r4.metric(
                "Confianza multiclase",
                f'{resultado_multiclase["confianza"] * 100:.2f} %',
            )

            nivel = resultado_multiclase["etiqueta"]

            clases_css = {
                "Bajo": "resultado-bajo",
                "Medio": "resultado-medio",
                "Alto": "resultado-alto",
                "Crítico": "resultado-critico",
            }

            clase_css = clases_css.get(nivel, "resultado-medio")

            st.markdown(
                f"""
                <div class="{clase_css}">
                    <strong>Clasificación binaria:</strong>
                    {resultado_binario['etiqueta']}<br>
                    <strong>Nivel de riesgo:</strong>
                    {resultado_multiclase['etiqueta']}<br>
                    <strong>Puntaje de exposición:</strong>
                    {resultado_multiclase['puntaje_exposicion']} de 5
                </div>
                """,
                unsafe_allow_html=True,
            )

            probabilidades = pd.DataFrame(
                {
                    "Nivel": list(
                        resultado_multiclase["probabilidades"].keys()
                    ),
                    "Probabilidad": [
                        valor * 100
                        for valor in resultado_multiclase[
                            "probabilidades"
                        ].values()
                    ],
                }
            )

            st.subheader("Probabilidad por nivel")

            st.bar_chart(
                probabilidades,
                x="Nivel",
                y="Probabilidad",
                color="#2E6F95",
            )

            st.caption(
                "La clase multiclase corresponde al nivel con la mayor "
                "probabilidad. El puntaje de exposición es una entrada "
                "adicional y no representa por sí solo la decisión final."
            )

            with st.expander("Ver respuesta técnica de la API"):
                st.json(
                    {
                        "binario": resultado_binario,
                        "multiclase": resultado_multiclase,
                    }
                )

        except requests.RequestException as error:
            st.error(
                f"No fue posible conectar con la API: {error}"
            )

        except RuntimeError as error:
            st.error(str(error))


# ============================================================
# 9. CONCLUSIONES
# ============================================================

with tab_conclusiones:
    st.header("Conclusiones y limitaciones")

    st.subheader("Resultados alcanzados")

    st.markdown(
        """
        - Se interpretó correctamente el conjunto German Credit.
        - Se conservaron las 20 variables originales.
        - Las categorías se transformaron en 61 entradas numéricas.
        - Se automatizó el preprocesamiento mediante POO.
        - Se compararon dos configuraciones binarias.
        - Se entrenó una ANN binaria con salida Sigmoid.
        - Se construyó una salida académica de cuatro niveles.
        - Se entrenó una ANN multiclase híbrida con 63 entradas.
        - Los modelos se integraron mediante FastAPI.
        - Streamlit permite presentar el proyecto y realizar predicciones.
        """
    )

    st.subheader("Limitaciones")

    st.markdown(
        """
        - El conjunto contiene únicamente 1.000 registros.
        - La clase binaria presenta una distribución de 70 % Bueno y 30 % Malo.
        - El modelo binario presenta un recall de 0.40 para la clase Malo.
        - No se conserva una evaluación separada de prueba para la configuración
          binaria sin Dropout.
        - No se entrenó una segunda configuración multiclase independiente.
        - Los niveles Bajo, Medio, Alto y Crítico son etiquetas académicas.
        - El puntaje de exposición no es una probabilidad.
        - El sistema no ha sido validado para decisiones financieras reales.
        """
    )

    st.warning(
        "Este prototipo tiene fines académicos. Una implementación real "
        "requiere datos actuales, validación experta, análisis de equidad, "
        "seguridad, monitoreo y cumplimiento regulatorio."
    )

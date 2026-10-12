# ============================================================
# PREPROCESAMIENTO DEL DATASET GERMAN CREDIT CON POO
# ============================================================

# Path permite construir rutas que funcionan sin depender
# de la ubicación personal de la computadora.
from pathlib import Path

# Pandas permite cargar y transformar los datos.
import pandas as pd

# Joblib permite guardar objetos de Python, como el escalador.
import joblib

# Herramientas para dividir y normalizar los datos.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class PreparadorDatos:
    """
    Clase encargada de preparar el dataset German Credit.

    La clase realiza las siguientes tareas:

    1. Carga el dataset original.
    2. Verifica la calidad de los datos.
    3. Crea la salida binaria.
    4. Convierte las variables categóricas.
    5. Divide entrenamiento y prueba.
    6. Normaliza las entradas.
    7. Crea los cuatro niveles de riesgo.
    8. Guarda los archivos procesados.
    """

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(self, ruta_proyecto=None):
        """
        El constructor se ejecuta cuando se crea un objeto
        de la clase PreparadorDatos.

        Aquí se definen las rutas, columnas y objetos que
        utilizará posteriormente el preprocesamiento.
        """

        # ----------------------------------------------------
        # 1. Definir la raíz del proyecto
        # ----------------------------------------------------

        if ruta_proyecto is None:
            # __file__ corresponde a src/data_prep.py.
            # parent corresponde a la carpeta src.
            # parent.parent corresponde a la raíz del proyecto.
            self.ruta_proyecto = (
                Path(__file__).resolve().parent.parent
            )
        else:
            self.ruta_proyecto = Path(ruta_proyecto)

        # Ruta donde se encuentra german.data.
        self.ruta_dataset = (
            self.ruta_proyecto
            / "data"
            / "raw"
            / "german.data"
        )

        # Carpeta donde se guardarán los archivos CSV.
        self.carpeta_procesados = (
            self.ruta_proyecto
            / "data"
            / "processed"
        )

        # Carpeta donde se guardarán el escalador y las reglas.
        self.carpeta_modelos = (
            self.ruta_proyecto
            / "models"
        )

        # ----------------------------------------------------
        # 2. Columnas originales del dataset
        # ----------------------------------------------------

        self.columnas = [
            "estado_cuenta",
            "duracion_meses",
            "historial_crediticio",
            "proposito",
            "monto_credito",
            "cuenta_ahorros",
            "antiguedad_empleo",
            "tasa_cuotas",
            "estado_personal",
            "otros_deudores",
            "tiempo_residencia",
            "propiedad",
            "edad",
            "otros_planes_pago",
            "vivienda",
            "creditos_existentes",
            "trabajo",
            "personas_dependientes",
            "telefono",
            "trabajador_extranjero",
            "clase"
        ]

        # ----------------------------------------------------
        # 3. Variables numéricas
        # ----------------------------------------------------

        self.columnas_numericas = [
            "duracion_meses",
            "monto_credito",
            "tasa_cuotas",
            "tiempo_residencia",
            "edad",
            "creditos_existentes",
            "personas_dependientes"
        ]

        # ----------------------------------------------------
        # 4. Variables categóricas
        # ----------------------------------------------------

        self.columnas_categoricas = [
            "estado_cuenta",
            "historial_crediticio",
            "proposito",
            "cuenta_ahorros",
            "antiguedad_empleo",
            "estado_personal",
            "otros_deudores",
            "propiedad",
            "otros_planes_pago",
            "vivienda",
            "trabajo",
            "telefono",
            "trabajador_extranjero"
        ]

        # ----------------------------------------------------
        # 5. Objetos que se llenarán durante el proceso
        # ----------------------------------------------------

        # Dataset completo.
        self.df = None

        # Variables de entrada antes de get_dummies.
        self.X = None

        # Variables de entrada después de get_dummies.
        self.X_transformado = None

        # Variable objetivo binaria completa.
        self.y_binario = None

        # Conjuntos separados.
        self.X_train = None
        self.X_test = None
        self.y_train_binario = None
        self.y_test_binario = None

        # Conjuntos normalizados.
        self.X_train_scaled = None
        self.X_test_scaled = None

        # Variables objetivo multiclase.
        self.y_train_multiclase = None
        self.y_test_multiclase = None

        # Entradas del modelo multiclase
        self.X_train_multiclase = None
        self.X_test_multiclase = None

        # Lista de las 61 columnas finales.
        self.columnas_modelo = None

        # Objeto que realizará la normalización.
        self.scaler = MinMaxScaler()

        # Valores utilizados para crear los cuatro niveles.
        self.mediana_duracion = None
        self.mediana_monto = None
        self.umbral_buenos = None
        self.umbral_malos = None

        # Significado de las clases del modelo multiclase.
        self.nombres_niveles = {
            0: "Bajo",
            1: "Medio",
            2: "Alto",
            3: "Crítico"
        }

    # ========================================================
    # MÉTODO 1: CARGAR LOS DATOS
    # ========================================================

    def cargar_datos(self):
        """
        Carga german.data y asigna los nombres correctos
        a las 21 columnas.
        """

        # Se verifica que el archivo realmente exista.
        if not self.ruta_dataset.exists():
            raise FileNotFoundError(
                f"No se encontró el dataset en: "
                f"{self.ruta_dataset}"
            )

        # german.data está separado por espacios y no contiene
        # nombres de columnas.
        self.df = pd.read_csv(
            self.ruta_dataset,
            sep=r"\s+",
            header=None,
            names=self.columnas
        )

        print("\n1. Datos cargados correctamente.")
        print("Dimensiones:", self.df.shape)

        return self.df

    # ========================================================
    # MÉTODO 2: VERIFICAR LOS DATOS
    # ========================================================

    def verificar_datos(self):
        """
        Comprueba las dimensiones, valores nulos
        y registros duplicados.
        """

        if self.df is None:
            raise ValueError(
                "Primero debe ejecutar cargar_datos()."
            )

        cantidad_nulos = self.df.isnull().sum().sum()
        cantidad_duplicados = self.df.duplicated().sum()

        print("\n2. Verificación inicial.")
        print("Cantidad de registros:", self.df.shape[0])
        print("Cantidad de columnas:", self.df.shape[1])
        print("Valores nulos:", cantidad_nulos)
        print("Registros duplicados:", cantidad_duplicados)

        # El dataset utilizado en este proyecto no debería
        # contener valores nulos.
        if cantidad_nulos > 0:
            raise ValueError(
                "El dataset contiene valores nulos."
            )

        # Tampoco debería contener filas completamente repetidas.
        if cantidad_duplicados > 0:
            raise ValueError(
                "El dataset contiene registros duplicados."
            )

    # ========================================================
    # MÉTODO 3: CREAR OBJETIVO BINARIO
    # ========================================================

    def crear_objetivo_binario(self):
        """
        Transforma la clase original:

        1 = crédito bueno -> 0
        2 = crédito malo  -> 1
        """

        if self.df is None:
            raise ValueError(
                "Primero debe ejecutar cargar_datos()."
            )

        # Se conserva la columna clase y se crea una nueva.
        self.df["riesgo_binario"] = self.df["clase"].map({
            1: 0,
            2: 1
        })

        # Si existiera un valor distinto de 1 o 2,
        # map produciría un valor nulo.
        if self.df["riesgo_binario"].isnull().sum() > 0:
            raise ValueError(
                "La variable clase contiene valores no reconocidos."
            )

        self.y_binario = self.df["riesgo_binario"]

        print("\n3. Objetivo binario creado.")
        print(
            self.y_binario
            .value_counts()
            .sort_index()
        )

        return self.y_binario

    # ========================================================
    # MÉTODO 4: SEPARAR ENTRADAS Y SALIDA
    # ========================================================

    def separar_entradas_salida(self):
        """
        Separa:

        X = las 20 variables utilizadas para predecir.
        y = el resultado binario que se desea predecir.
        """

        if self.y_binario is None:
            raise ValueError(
                "Primero debe crear el objetivo binario."
            )

        # Se eliminan ambas respuestas de las entradas.
        # Dejar clase dentro de X produciría fuga de información.
        self.X = self.df.drop(
            columns=["clase", "riesgo_binario"]
        )

        print("\n4. Entradas y salida separadas.")
        print("Dimensiones de X:", self.X.shape)
        print("Dimensiones de y:", self.y_binario.shape)

        return self.X, self.y_binario

    # ========================================================
    # MÉTODO 5: CONVERTIR VARIABLES CATEGÓRICAS
    # ========================================================

    def convertir_categorias(self):
        """
        Convierte las 13 variables categóricas mediante
        pd.get_dummies.

        Las 20 variables originales quedan representadas
        mediante 61 columnas numéricas.
        """

        if self.X is None:
            raise ValueError(
                "Primero debe separar las entradas y la salida."
            )

        # Se crea una columna de 0 y 1 por cada categoría.
        self.X_transformado = pd.get_dummies(
            self.X,
            columns=self.columnas_categoricas,
            dtype=int
        )

        # Se guarda el nombre y el orden exacto de las columnas.
        self.columnas_modelo = (
            self.X_transformado.columns.tolist()
        )

        # Se comprueba que no queden variables de texto.
        columnas_no_numericas = (
            self.X_transformado
            .select_dtypes(exclude="number")
            .columns
        )

        if len(columnas_no_numericas) > 0:
            raise ValueError(
                "Todavía existen columnas no numéricas."
            )

        # Se comprueba que get_dummies no produzca valores nulos.
        if self.X_transformado.isnull().sum().sum() > 0:
            raise ValueError(
                "La transformación generó valores nulos."
            )

        print("\n5. Variables categóricas transformadas.")
        print("Dimensiones originales:", self.X.shape)
        print(
            "Dimensiones transformadas:",
            self.X_transformado.shape
        )

        return self.X_transformado

    # ========================================================
    # MÉTODO 6: DIVIDIR ENTRENAMIENTO Y PRUEBA
    # ========================================================

    def dividir_datos(self):
        """
        Divide los datos:

        80 % para entrenamiento.
        20 % para prueba.

        stratify conserva la distribución 70/30.
        """

        if self.X_transformado is None:
            raise ValueError(
                "Primero debe convertir las categorías."
            )

        (
            self.X_train,
            self.X_test,
            self.y_train_binario,
            self.y_test_binario
        ) = train_test_split(
            self.X_transformado,
            self.y_binario,
            test_size=0.20,
            random_state=101,
            stratify=self.y_binario
        )

        print("\n6. Datos divididos.")
        print("X_train:", self.X_train.shape)
        print("X_test:", self.X_test.shape)
        print("y_train:", self.y_train_binario.shape)
        print("y_test:", self.y_test_binario.shape)

        return (
            self.X_train,
            self.X_test,
            self.y_train_binario,
            self.y_test_binario
        )

    # ========================================================
    # MÉTODO 7: NORMALIZAR LOS DATOS
    # ========================================================

    def normalizar_datos(self):
        """
        Ajusta MinMaxScaler solamente con entrenamiento.

        Después utiliza el mismo escalador para transformar
        el conjunto de prueba.
        """

        if self.X_train is None or self.X_test is None:
            raise ValueError(
                "Primero debe dividir los datos."
            )

        # fit_transform aprende los mínimos y máximos
        # del conjunto de entrenamiento.
        datos_train_scaled = self.scaler.fit_transform(
            self.X_train
        )

        # transform utiliza exactamente lo aprendido
        # para transformar el conjunto de prueba.
        datos_test_scaled = self.scaler.transform(
            self.X_test
        )

        # MinMaxScaler devuelve arreglos sin nombres.
        # Por eso se convierten nuevamente en DataFrame.
        self.X_train_scaled = pd.DataFrame(
            datos_train_scaled,
            columns=self.columnas_modelo,
            index=self.X_train.index
        )

        self.X_test_scaled = pd.DataFrame(
            datos_test_scaled,
            columns=self.columnas_modelo,
            index=self.X_test.index
        )

        print("\n7. Datos normalizados.")
        print(
            "X_train_scaled:",
            self.X_train_scaled.shape
        )
        print(
            "X_test_scaled:",
            self.X_test_scaled.shape
        )
        print(
            "Rango de entrenamiento:",
            self.X_train_scaled.min().min(),
            "a",
            self.X_train_scaled.max().max()
        )

        return self.X_train_scaled, self.X_test_scaled

    # ========================================================
    # MÉTODO 8: CALCULAR PUNTAJE DE EXPOSICIÓN
    # ========================================================

    def calcular_puntaje_exposicion(
        self,
        fila,
        limite_duracion,
        limite_monto
    ):
        """
        Calcula un puntaje académico entre 0 y 5.

        Cada condición cumplida suma un punto.
        """

        puntaje = 0

        # Punto por duración superior a la mediana.
        if fila["duracion_meses"] > limite_duracion:
            puntaje += 1

        # Punto por monto superior a la mediana.
        if fila["monto_credito"] > limite_monto:
            puntaje += 1

        # Punto por estado de cuenta asociado con una
        # mayor proporción observada de créditos malos.
        if fila["estado_cuenta"] in ("A11", "A12"):
            puntaje += 1

        # Punto por categorías de ahorro asociadas con una
        # mayor proporción observada de créditos malos.
        if fila["cuenta_ahorros"] in ("A61", "A62"):
            puntaje += 1

        # Punto por historiales asociados con una mayor
        # proporción observada de créditos malos.
        if fila["historial_crediticio"] in ("A30", "A31"):
            puntaje += 1

        return puntaje

    # ========================================================
    # MÉTODO 9: ASIGNAR UN NIVEL DE RIESGO
    # ========================================================

    def asignar_nivel_riesgo(
        self,
        riesgo_binario,
        puntaje
    ):
        """
        Asigna uno de los cuatro niveles:

        0 = Bajo
        1 = Medio
        2 = Alto
        3 = Crítico

        Los créditos originalmente buenos se dividen entre
        Bajo y Medio.

        Los créditos originalmente malos se dividen entre
        Alto y Crítico.
        """

        # Crédito originalmente bueno.
        if riesgo_binario == 0:

            # Puntaje menor o igual al umbral de buenos.
            if puntaje <= self.umbral_buenos:
                return 0

            # Puntaje superior al umbral de buenos.
            return 1

        # Crédito originalmente malo.
        if riesgo_binario == 1:

            # Puntaje menor o igual al umbral de malos.
            if puntaje <= self.umbral_malos:
                return 2

            # Puntaje superior al umbral de malos.
            return 3

        # Esta parte solo se ejecutaría si llega un valor
        # binario diferente de 0 o 1.
        raise ValueError(
            f"Valor de riesgo binario no reconocido: "
            f"{riesgo_binario}"
        )

    # ========================================================
    # MÉTODO 10: CREAR OBJETIVO MULTICLASE
    # ========================================================

    def crear_objetivo_multiclase(self):
        """
        Crea los niveles Bajo, Medio, Alto y Crítico.

        Todas las medianas y umbrales se calculan únicamente
        utilizando el conjunto de entrenamiento.
        """

        if self.X_train is None or self.X_test is None:
            raise ValueError(
                "Primero debe dividir los datos."
            )

        # Se recuperan los registros originales usando los
        # mismos índices de entrenamiento y prueba.
        df_train_original = self.df.loc[
            self.X_train.index
        ].copy()

        df_test_original = self.df.loc[
            self.X_test.index
        ].copy()

        # Estos límites se calculan solamente con entrenamiento.
        self.mediana_duracion = (
            df_train_original["duracion_meses"].median()
        )

        self.mediana_monto = (
            df_train_original["monto_credito"].median()
        )

        # Calcular el puntaje de entrenamiento.
        df_train_original["puntaje_exposicion"] = (
            df_train_original.apply(
                self.calcular_puntaje_exposicion,
                axis=1,
                limite_duracion=self.mediana_duracion,
                limite_monto=self.mediana_monto
            )
        )

        # Calcular el puntaje de prueba usando los mismos
        # límites obtenidos del entrenamiento.
        df_test_original["puntaje_exposicion"] = (
            df_test_original.apply(
                self.calcular_puntaje_exposicion,
                axis=1,
                limite_duracion=self.mediana_duracion,
                limite_monto=self.mediana_monto
            )
        )

        # Mediana del puntaje entre créditos buenos.
        # Divide los créditos buenos entre Bajo y Medio.
        self.umbral_buenos = df_train_original.loc[
            df_train_original["riesgo_binario"] == 0,
            "puntaje_exposicion"
        ].median()

        # Mediana del puntaje entre créditos malos.
        # Divide los créditos malos entre Alto y Crítico.
        self.umbral_malos = df_train_original.loc[
            df_train_original["riesgo_binario"] == 1,
            "puntaje_exposicion"
        ].median()

        # Crear los niveles de entrenamiento.
        self.y_train_multiclase = (
            df_train_original.apply(
                lambda fila: self.asignar_nivel_riesgo(
                    fila["riesgo_binario"],
                    fila["puntaje_exposicion"]
                ),
                axis=1
            )
        )

        self.y_train_multiclase.name = "nivel_riesgo"

        # Crear los niveles de prueba.
        self.y_test_multiclase = (
            df_test_original.apply(
                lambda fila: self.asignar_nivel_riesgo(
                    fila["riesgo_binario"],
                    fila["puntaje_exposicion"]
                ),
                axis=1
            )
        )

        self.y_test_multiclase.name = "nivel_riesgo"

        # Comprobar que entrenamiento contenga las cuatro clases.
        niveles_encontrados = sorted(
            self.y_train_multiclase.unique()
        )

        if niveles_encontrados != [0, 1, 2, 3]:
            raise ValueError(
                "No se generaron correctamente "
                "los cuatro niveles."
            )

        # Comprobar que no existan valores nulos.
        if self.y_train_multiclase.isnull().sum() > 0:
            raise ValueError(
                "La salida multiclase de entrenamiento "
                "contiene valores nulos."
            )

        if self.y_test_multiclase.isnull().sum() > 0:
            raise ValueError(
                "La salida multiclase de prueba "
                "contiene valores nulos."
            )

        print("\n8. Objetivo multiclase creado.")
        print("Mediana de duración:", self.mediana_duracion)
        print("Mediana de monto:", self.mediana_monto)
        print("Umbral de créditos buenos:", self.umbral_buenos)
        print("Umbral de créditos malos:", self.umbral_malos)

        print("\nDistribución de entrenamiento:")
        print(
            self.y_train_multiclase
            .value_counts()
            .sort_index()
            .rename(index=self.nombres_niveles)
        )

        # Se agrega el puntaje como entrada del modelo multiclase.
        # El máximo es 5, así que dividir entre 5 lo deja en rango 0 a 1.
        self.X_train_multiclase = self.X_train_scaled.copy()
        self.X_test_multiclase = self.X_test_scaled.copy()

        self.X_train_multiclase["puntaje_exposicion"] = (
                df_train_original["puntaje_exposicion"] / 5
        )
        self.X_test_multiclase["puntaje_exposicion"] = (
                df_test_original["puntaje_exposicion"] / 5
        )

        return (
            self.y_train_multiclase,
            self.y_test_multiclase
        )

    # ========================================================
    # MÉTODO 11: VERIFICAR COHERENCIA MULTICLASE
    # ========================================================

    def verificar_coherencia_multiclase(self):
        """
        Comprueba que:

        - Los créditos buenos solo sean Bajo o Medio.
        - Los créditos malos solo sean Alto o Crítico.
        """

        if self.y_train_multiclase is None:
            raise ValueError(
                "Primero debe crear el objetivo multiclase."
            )

        tabla = pd.crosstab(
            self.y_train_binario,
            self.y_train_multiclase
        )

        # Se obliga a que aparezcan las dos filas y las
        # cuatro columnas, aunque alguna tuviera cero casos.
        tabla = tabla.reindex(
            index=[0, 1],
            columns=[0, 1, 2, 3],
            fill_value=0
        )

        # Se colocan etiquetas comprensibles.
        tabla.index = ["Bueno", "Malo"]

        tabla.columns = [
            "Bajo",
            "Medio",
            "Alto",
            "Crítico"
        ]

        # Estas combinaciones deberían ser siempre cero.
        if tabla.loc["Bueno", "Alto"] != 0:
            raise ValueError(
                "Existen créditos buenos clasificados como Alto."
            )

        if tabla.loc["Bueno", "Crítico"] != 0:
            raise ValueError(
                "Existen créditos buenos clasificados como Crítico."
            )

        if tabla.loc["Malo", "Bajo"] != 0:
            raise ValueError(
                "Existen créditos malos clasificados como Bajo."
            )

        if tabla.loc["Malo", "Medio"] != 0:
            raise ValueError(
                "Existen créditos malos clasificados como Medio."
            )

        print("\n9. Coherencia multiclase comprobada.")
        print(tabla)

        return tabla

    # ========================================================
    # MÉTODO 12: GUARDAR RESULTADOS
    # ========================================================

    def guardar_resultados(self):
        """
        Guarda los archivos utilizados por los modelos:

        - Entradas binarias.
        - Salidas binarias.
        - Entradas multiclase.
        - Salidas multiclase.
        - Escalador.
        - Orden de columnas.
        - Información de preprocesamiento.
        - Regla multiclase.
        """

        if self.X_train_scaled is None:
            raise ValueError(
                "Primero debe normalizar los datos."
            )

        if self.y_train_multiclase is None:
            raise ValueError(
                "Primero debe crear el objetivo multiclase."
            )

        # Crear las carpetas si todavía no existen.
        self.carpeta_procesados.mkdir(
            parents=True,
            exist_ok=True
        )

        self.carpeta_modelos.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Archivos del modelo binario
        # ----------------------------------------------------

        self.X_train_scaled.to_csv(
            self.carpeta_procesados
            / "X_train_binario.csv",
            index=False
        )

        self.X_test_scaled.to_csv(
            self.carpeta_procesados
            / "X_test_binario.csv",
            index=False
        )

        self.y_train_binario.to_frame(
            name="riesgo_binario"
        ).to_csv(
            self.carpeta_procesados
            / "y_train_binario.csv",
            index=False
        )

        self.y_test_binario.to_frame(
            name="riesgo_binario"
        ).to_csv(
            self.carpeta_procesados
            / "y_test_binario.csv",
            index=False
        )

        # ----------------------------------------------------
        # Archivos del modelo multiclase
        # ----------------------------------------------------

        # Las entradas son las mismas para ambos modelos.
        self.X_train_multiclase.to_csv(
            self.carpeta_procesados / "X_train_multiclase.csv",
            index=False
        )

        self.X_test_multiclase.to_csv(
            self.carpeta_procesados / "X_test_multiclase.csv",
            index=False
        )

        self.y_train_multiclase.to_frame().to_csv(
            self.carpeta_procesados
            / "y_train_multiclase.csv",
            index=False
        )

        self.y_test_multiclase.to_frame().to_csv(
            self.carpeta_procesados
            / "y_test_multiclase.csv",
            index=False
        )

        # ----------------------------------------------------
        # Guardar escalador y columnas
        # ----------------------------------------------------

        joblib.dump(
            self.scaler,
            self.carpeta_modelos / "scaler.pkl"
        )

        joblib.dump(
            self.columnas_modelo,
            self.carpeta_modelos
            / "columnas_modelo.pkl"
        )

        # ----------------------------------------------------
        # Información general del preprocesamiento
        # ----------------------------------------------------

        informacion_preprocesamiento = {
            "columnas_originales": self.columnas[:-1],
            "columnas_numericas": self.columnas_numericas,
            "columnas_categoricas": self.columnas_categoricas,
            "columnas_modelo": self.columnas_modelo,
            "clase_original": {
                1: "Bueno",
                2: "Malo"
            },
            "clase_binaria": {
                0: "Bueno",
                1: "Malo"
            }
        }

        joblib.dump(
            informacion_preprocesamiento,
            self.carpeta_modelos
            / "informacion_preprocesamiento.pkl"
        )

        # ----------------------------------------------------
        # Información de los cuatro niveles
        # ----------------------------------------------------

        regla_multiclase = {
            "mediana_duracion": self.mediana_duracion,
            "mediana_monto": self.mediana_monto,
            "umbral_buenos": self.umbral_buenos,
            "umbral_malos": self.umbral_malos,
            "niveles": self.nombres_niveles,
            "estado_cuenta_riesgo": ["A11", "A12"],
            "cuenta_ahorros_riesgo": ["A61", "A62"],
            "historial_riesgo": ["A30", "A31"]
        }

        joblib.dump(
            regla_multiclase,
            self.carpeta_modelos
            / "regla_multiclase.pkl"
        )

        print("\n10. Archivos guardados correctamente.")

    # ========================================================
    # MÉTODO 13: VERIFICAR ARCHIVOS
    # ========================================================

    def verificar_archivos(self):
        """
        Comprueba que los 12 archivos esperados
        fueron creados.
        """

        archivos = [
            self.carpeta_procesados
            / "X_train_binario.csv",

            self.carpeta_procesados
            / "X_test_binario.csv",

            self.carpeta_procesados
            / "y_train_binario.csv",

            self.carpeta_procesados
            / "y_test_binario.csv",

            self.carpeta_procesados
            / "X_train_multiclase.csv",

            self.carpeta_procesados
            / "X_test_multiclase.csv",

            self.carpeta_procesados
            / "y_train_multiclase.csv",

            self.carpeta_procesados
            / "y_test_multiclase.csv",

            self.carpeta_modelos
            / "scaler.pkl",

            self.carpeta_modelos
            / "columnas_modelo.pkl",

            self.carpeta_modelos
            / "informacion_preprocesamiento.pkl",

            self.carpeta_modelos
            / "regla_multiclase.pkl"
        ]

        print("\n11. Verificación de archivos.")

        archivos_faltantes = []

        for archivo in archivos:
            if archivo.exists():
                print(archivo.name, "-> Creado")
            else:
                print(archivo.name, "-> No creado")
                archivos_faltantes.append(archivo.name)

        if len(archivos_faltantes) > 0:
            raise FileNotFoundError(
                f"Faltan archivos: {archivos_faltantes}"
            )

    # ========================================================
    # MÉTODO 14: VERIFICAR DIMENSIONES
    # ========================================================

    def verificar_dimensiones(self):
        """
        Comprueba que los conjuntos finales tengan
        las dimensiones esperadas.
        """

        # Dimensiones que esperamos obtener.
        dimensiones_esperadas = {
            "X_train": (800, 61),
            "X_test": (200, 61),
            "X_train_multiclase": (800, 62),
            "X_test_multiclase": (200, 62),
            "y_train_binario": (800,),
            "y_test_binario": (200,),
            "y_train_multiclase": (800,),
            "y_test_multiclase": (200,)
        }

        # Dimensiones realmente obtenidas.
        dimensiones_obtenidas = {
            "X_train": self.X_train_scaled.shape,
            "X_test": self.X_test_scaled.shape,
            "y_train_binario": self.y_train_binario.shape,
            "y_test_binario": self.y_test_binario.shape,
            "X_train_multiclase": self.X_train_multiclase.shape,
            "X_test_multiclase": self.X_test_multiclase.shape,
            "y_train_multiclase": self.y_train_multiclase.shape,
            "y_test_multiclase": self.y_test_multiclase.shape
        }

        print("\n12. Verificación de dimensiones.")

        # Comparar cada dimensión obtenida con la esperada.
        for nombre, dimension in dimensiones_obtenidas.items():

            print(nombre, "->", dimension)

            if dimension != dimensiones_esperadas[nombre]:
                raise ValueError(
                    f"Dimensión inesperada en {nombre}: "
                    f"{dimension}. Se esperaba "
                    f"{dimensiones_esperadas[nombre]}."
                )

    # ========================================================
    # MÉTODO PRINCIPAL
    # ========================================================

    def ejecutar_preprocesamiento(self):
        """
        Ejecuta todo el preprocesamiento en el orden correcto.

        Este método coordina los demás métodos de la clase.
        """

        self.cargar_datos()
        self.verificar_datos()
        self.crear_objetivo_binario()
        self.separar_entradas_salida()
        self.convertir_categorias()
        self.dividir_datos()
        self.normalizar_datos()
        self.crear_objetivo_multiclase()
        self.verificar_coherencia_multiclase()
        self.guardar_resultados()
        self.verificar_archivos()
        self.verificar_dimensiones()

        print("\n==========================================")
        print("PREPROCESAMIENTO COMPLETO FINALIZADO")
        print("==========================================")
# Librerías a utilizar

import os # Permite una interacción con el sistema operativo
import numpy as np # Permite control de arreglos y operaciones matemáticas
import pandas as pd # Permite el manejo de datos
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential # Permite creación de la red por capas
from tensorflow.keras.layers import Dense, Activation, Dropout # Permite a la neurona aprender
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # Permite la finalización temprana en caso de necesitarlo
from tensorflow.keras.regularizers import l2 # Permite que el modelo le de "mas importancia" a pesos grandes
from tensorflow.keras.models import load_model # Permite cargar el modelo
from sklearn.metrics import classification_report, confusion_matrix # Permite la visualización del desempeño del modelo
from sklearn.metrics import roc_auc_score # Permite llamar al modelo binario
from sklearn.utils.class_weight import compute_class_weight # Permite calcular pesos para evitar desbalance

tf.keras.utils.set_random_seed(42)


class ClasificadorRiesgoCrediticio:
    def __init__(self, base_dir, nombres_clases=None):
        self.base_dir = base_dir
        # Permite la obtención de la ruta del script

        self.nombres_clases = nombres_clases or ['Bajo', 'Medio', 'Alto', 'Crítico']
        # Permite reemplazar los números por las categorías que se inserten en el arreglo

        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.X_train_bi = None
        self.X_test_bi = None
        self.model = None
        self.modelo_binario = None
        self.num_neuronas = None
        self.num_neuronas_salida = None
        self.clases = None
        self.probabilidades = None
        self.y_pred = None
        self.y_real = None
        self.class_weights_dict = None

    def cargar_datos(self):
        self.X_train = pd.read_csv(os.path.join(self.base_dir, '../../data/processed/X_train_multiclase.csv'))
        self.Y_train = pd.read_csv(os.path.join(self.base_dir, '../../data/processed/y_train_multiclase.csv'))
        self.X_test  = pd.read_csv(os.path.join(self.base_dir, '../../data/processed/X_test_multiclase.csv'))
        self.Y_test  = pd.read_csv(os.path.join(self.base_dir, '../../data/processed/y_test_multiclase.csv'))
        self.X_train_bi = pd.read_csv(os.path.join(self.base_dir, '../../data/processed/X_train_binario.csv'))
        self.X_test_bi  = pd.read_csv(os.path.join(self.base_dir, '../../data/processed/X_test_binario.csv'))


    def cargar_modelo_binario(self):
        current_dir = os.getcwd()

        ruta_modelo_binario = os.path.abspath(
            os.path.join(current_dir, '../../src/train/modelo_pred_binomial.keras')
        )
        # Define la ruta donde se guarda el modelo binario

        self.modelo_binario = load_model(ruta_modelo_binario)
        # Carga el modelo binario en memoria como un objeto

        pred_binaria_train = self.modelo_binario.predict(self.X_train_bi)
        pred_binaria_test  = self.modelo_binario.predict(self.X_test_bi)
        # Genera la probabilidad binaria para los datos

        self.X_train['prob_binaria'] = pred_binaria_train
        self.X_test['prob_binaria']  = pred_binaria_test
        # Iserta la probabilidad como una columna adicional en ambos conjuntos

    def construir_modelo(self):
        self.model = Sequential()
        # Definición del modelo

        self.num_neuronas = self.X_train.shape[1]
        # Cantidad de neuronas de entrada

        clases_totales = pd.concat([self.Y_train.squeeze(), self.Y_test.squeeze()]).unique()
        self.num_neuronas_salida = len(clases_totales)
        print(self.num_neuronas_salida)
        # Cantidad de neuronas de salida

        y_train_array = self.Y_train.squeeze().to_numpy()
        clases_unicas = np.unique(y_train_array)
        pesos = compute_class_weight(
            class_weight='balanced', classes=clases_unicas, y=y_train_array
        )
        self.class_weights_dict = dict(zip(clases_unicas, pesos))
        # Calcula pesos para compensar el desbalance

        self.model.add(Dense(units=self.num_neuronas * 2, kernel_regularizer=l2(0.0001), activation='relu'))
        self.model.add(Dropout(0.3)) # Permite apagar un porcentaje de las neuronas para que encuentre distintas maneras

        self.model.add(Dense(units=self.num_neuronas, kernel_regularizer=l2(0.0001), activation='relu'))
        self.model.add(Dropout(0.3))

        self.model.add(Dense(units=int(self.num_neuronas / 2), kernel_regularizer=l2(0.0001), activation='relu'))
        self.model.add(Dropout(0.3))
        # Capas ocultas

        self.model.add(Dense(self.num_neuronas_salida, activation='softmax'))
        # Capa de salida múltiple

        self.model.compile(
            loss='sparse_categorical_crossentropy', # "Sparse" se utiliza para cuando son dígitos enteros
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            metrics=['accuracy']
        )

    def entrenar(self):
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15, restore_best_weights=True)
        reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001, verbose=1)
        # Permite detenerse antes cuando detecta que el modelo no aprende de manera significativa después de 15 repeticiones

        self.model.fit(
            x=self.X_train,
            y=self.Y_train,
            epochs=200,
            batch_size=60,
            validation_split=0.20,
            verbose=1,
            callbacks=[early_stop, reduce_lr],
            class_weight=self.class_weights_dict
        )


    def _carpeta_graficos(self):
        ruta = os.path.abspath(os.path.join(self.base_dir, '../../models'))
        os.makedirs(ruta, exist_ok=True)
        return ruta

    def graficar_historial(self):
        model_loss = pd.DataFrame(self.model.history.history)
        # Convierte el historial a DataFrame

        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        # Permite mostrar los dos gráficos juntos

        ax[0].plot(model_loss['loss'], label='Entrenamiento (Loss)')
        if 'val_loss' in model_loss.columns:
            ax[0].plot(model_loss['val_loss'], label='Validación (Val Loss)')
        ax[0].set_title('Pérdida del Modelo')
        ax[0].set_xlabel('Épocas')
        ax[0].set_ylabel('Pérdida')
        ax[0].legend()
        ax[0].grid(True)
        # Gráfico de Pérdida (Loss)

        if 'accuracy' in model_loss.columns:
            ax[1].plot(model_loss['accuracy'], label='Entrenamiento (Accuracy)')
            if 'val_accuracy' in model_loss.columns:
                ax[1].plot(model_loss['val_accuracy'], label='Validación (Val Accuracy)')
            ax[1].set_title('Precisión del Modelo')
            ax[1].set_xlabel('Épocas')
            ax[1].set_ylabel('Precisión')
            ax[1].legend()
            ax[1].grid(True)
        # Gráfico de Precisión (Accuracy)

        plt.tight_layout()
        ruta = os.path.join(self._carpeta_graficos(), 'historial_entrenamiento.png')
        plt.savefig(ruta, dpi=150, bbox_inches='tight')
        print(f'Gráfico guardado en: {ruta}')
        plt.show()

    def graficar_matriz_confusion(self):
        cm = confusion_matrix(self.y_real, self.y_pred, labels=self.clases)
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.nombres_clases, yticklabels=self.nombres_clases, ax=ax[0])
        ax[0].set_title('Matriz de confusión')
        ax[0].set_xlabel('Predicho')
        ax[0].set_ylabel('Real')
        # Genera una matriz de confusión con valores absolutos

        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=self.nombres_clases, yticklabels=self.nombres_clases, ax=ax[1])
        ax[1].set_title('Matriz de confusión normalizada (por clase real)')
        ax[1].set_xlabel('Predicho')
        ax[1].set_ylabel('Real')
        plt.tight_layout()
        # Genera una matriz de confusión con valores estandarizados

        ruta = os.path.join(self._carpeta_graficos(), 'matriz_confusion.png')
        plt.savefig(ruta, dpi=150, bbox_inches='tight')
        print(f'Gráfico guardado en: {ruta}')
        plt.show()

    def evaluar(self):
        self.clases = list(range(self.num_neuronas_salida))

        self.probabilidades = self.model.predict(self.X_test)  # probabilidad por clase (n_muestras x 4)
        self.y_pred = np.argmax(self.probabilidades, axis=1)  # clase con mayor probabilidad
        self.y_real = self.Y_test.squeeze().to_numpy() # (n,1) DataFrame -> vector

        loss, acc = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
        print(f"Loss: {loss:.4f} | Accuracy: {acc:.4f}\n")
        # Nos da métricas globales

        print(classification_report(self.y_real, self.y_pred, labels=self.clases,
                                    target_names=self.nombres_clases, zero_division=0))
        # Genera un reporte por clase (f1, precision, recall)

        print(f"ROC-AUC (One-vs-Rest, Macro): {roc_auc_score(self.y_real, self.probabilidades, multi_class='ovr', average='macro'):.4f}")

    def predecir_muestra(self, n=5, random_state=42):
        X_nuevos = self.X_test.sample(n, random_state=random_state)   # 5 clientes al azar

        prob_nuevos = self.model.predict(X_nuevos)
        resultado   = np.argmax(prob_nuevos, axis=1)

        df_resultado = pd.DataFrame({
            'riesgo_predicho': [self.nombres_clases[i] for i in resultado],
            'confianza':       prob_nuevos.max(axis=1).round(3),
            'riesgo_real':     [self.nombres_clases[i] for i in self.Y_test.loc[X_nuevos.index].squeeze()]
        }, index=X_nuevos.index)
        print(df_resultado)
        return df_resultado

    def guardar_modelo(self, nombre_archivo='modelo_riesgo_multiclase.keras'):
        carpeta = os.path.abspath(os.path.join(self.base_dir, '../../models'))
        os.makedirs(carpeta, exist_ok=True)
        ruta = os.path.join(carpeta, nombre_archivo)
        self.model.save(ruta)
        print(f'Modelo guardado en: {ruta}')

    def ejecutar(self):
        self.cargar_datos()
        self.cargar_modelo_binario()
        self.construir_modelo()
        self.entrenar()
        self.graficar_historial()
        self.evaluar()
        self.graficar_matriz_confusion()
        self.predecir_muestra()
        self.guardar_modelo()


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Permite la obtención de la ruta del script

    clasificador = ClasificadorRiesgoCrediticio(base_dir)
    clasificador.ejecutar()
# Uso de Inteligencia Artificial

## Herramienta utilizada

Durante el desarrollo del proyecto, el equipo utilizó Microsoft 365 Copilot
como herramienta de consulta, orientación técnica y apoyo al aprendizaje.

La herramienta se utilizó principalmente para:

- Resolver consultas relacionadas con Python.
- Aclarar conceptos sobre programación orientada a objetos.
- Comprender el análisis exploratorio y el preprocesamiento de datos.
- Revisar la transformación de variables categóricas mediante `get_dummies`.
- Comprender la división de los datos y la prevención de fuga de información.
- Explicar el funcionamiento de `MinMaxScaler`.
- Revisar conceptos relacionados con redes neuronales artificiales.
- Comprender las funciones de activación ReLU, Sigmoid y Softmax.
- Interpretar métricas como accuracy, precision, recall, F1-score y ROC-AUC.
- Comparar configuraciones con y sin capas Dropout.
- Interpretar curvas de entrenamiento y matrices de confusión.
- Orientar la integración de los modelos entrenados con FastAPI.
- Apoyar el desarrollo y la organización de la interfaz en Streamlit.
- Aclarar errores encontrados durante la ejecución del proyecto.
- Resolver dudas sobre PyCharm, Sourcetree, Git y GitHub.
- Mejorar comentarios, explicaciones y documentación.
- Revisar la correspondencia entre el proyecto y la rúbrica de evaluación.
- Preparar el material utilizado durante la exposición.

Las respuestas generadas por la herramienta se utilizaron como orientación.
El código y la documentación fueron incorporados gradualmente, ejecutados,
revisados y validados por los integrantes antes de aceptarse como parte del
proyecto.

La herramienta no ejecutó ni entregó automáticamente el proyecto final. Cada
integrante participó en la implementación, revisión y validación de los
componentes correspondientes.

---

## Alcance del apoyo recibido

El apoyo de inteligencia artificial se utilizó en diferentes etapas del
proyecto.

### Análisis exploratorio

La herramienta se utilizó para:

- Explicar la estructura del conjunto German Credit.
- Comprender las diferencias entre `german.data` y
  `german.data-numeric`.
- Interpretar los códigos incluidos en `german.doc`.
- Revisar estadísticas descriptivas.
- Interpretar distribuciones, tablas y gráficos.
- Diferenciar asociación de causalidad.
- Preparar conclusiones basadas en los resultados observados.

### Preprocesamiento

La herramienta se utilizó para:

- Comprender la separación entre variables de entrada y salida.
- Revisar la transformación de la respuesta original.
- Explicar el funcionamiento de `get_dummies`.
- Comprender el cambio de 20 variables originales a 61 entradas numéricas.
- Revisar la división estratificada de los datos.
- Explicar por qué el escalador debe ajustarse solamente con entrenamiento.
- Identificar posibles situaciones de fuga de información.
- Documentar la construcción académica de los cuatro niveles de riesgo.

### Programación orientada a objetos

La herramienta se utilizó para:

- Organizar el preprocesamiento dentro de la clase `PreparadorDatos`.
- Separar responsabilidades mediante métodos.
- Revisar la función del constructor y los atributos.
- Comprender errores relacionados con métodos inexistentes.
- Revisar posibles problemas de indentación.
- Comprobar el orden de ejecución del método
  `ejecutar_preprocesamiento()`.
- Mejorar los comentarios y las explicaciones de la clase.

### Redes neuronales artificiales

La herramienta se utilizó para:

- Comprender la arquitectura de los modelos.
- Revisar el funcionamiento de las capas densas.
- Explicar las activaciones ReLU, Sigmoid y Softmax.
- Comprender el uso de `binary_crossentropy`.
- Comprender el uso de `sparse_categorical_crossentropy`.
- Explicar el propósito de EarlyStopping.
- Explicar el propósito de las capas Dropout.
- Comprender la regularización L2.
- Comprender el uso de pesos de clase.
- Interpretar las curvas de pérdida y validación.
- Interpretar los reportes de clasificación.
- Interpretar las matrices de confusión.
- Comparar las configuraciones binarias con y sin Dropout.
- Documentar las limitaciones de los modelos.

### API REST

La herramienta se utilizó para:

- Comprender la función de FastAPI.
- Organizar la carga de modelos y artefactos.
- Revisar la validación de entradas.
- Explicar el formato JSON.
- Revisar el procesamiento de las 20 variables originales.
- Comprender la alineación de las 61 entradas.
- Integrar la probabilidad binaria con el modelo multiclase.
- Interpretar códigos de respuesta HTTP.
- Revisar la documentación automática de Swagger.
- Orientar la resolución de errores de conexión.

### Interfaz Streamlit

La herramienta se utilizó para:

- Organizar la navegación de la aplicación.
- Presentar la información del proyecto de forma ordenada.
- Diseñar el formulario con las 20 variables originales.
- Consumir los endpoints de FastAPI.
- Mostrar las respuestas binaria y multiclase.
- Presentar las probabilidades mediante gráficos.
- Mejorar el contraste, los colores y la legibilidad.
- Configurar el tema mediante `.streamlit/config.toml`.
- Revisar mensajes relacionados con la disponibilidad de la API.

### Documentación y control de versiones

La herramienta se utilizó para:

- Organizar el contenido del `README.md`.
- Preparar el notebook de comparación.
- Revisar el archivo `.gitignore`.
- Documentar el uso responsable de inteligencia artificial.
- Revisar la estructura del repositorio.
- Resolver dudas relacionadas con Sourcetree.
- Orientar la preparación de commits y Push.
- Revisar los entregables contra la rúbrica.
- Preparar una secuencia lógica para la exposición.

---

## Ejemplos de consultas realizadas

### Consulta 1: adaptación al material del curso

> ¿Cómo podemos mantener el código lo más parecido posible a los notebooks
> proporcionados por el profesor, pero adaptándolo a los requisitos del
> proyecto de riesgo crediticio?

Esta consulta ayudó a conservar el orden y la sencillez de los ejemplos vistos
en clase, sin eliminar los requisitos específicos del proyecto.

### Consulta 2: comprensión del dataset

> ¿Cuál es la diferencia entre `german.data` y `german.data-numeric`, y cuál
> conviene utilizar si el proyecto solicita aplicar `get_dummies`?

La respuesta se comparó con la documentación incluida en el dataset antes de
seleccionar el archivo principal.

El equipo eligió `german.data` porque conserva las categorías originales y
permite demostrar su transformación durante el preprocesamiento.

### Consulta 3: prevención de fuga de información

> ¿Por qué se debe dividir el dataset antes de ajustar `MinMaxScaler`, y por
> qué el escalador debe aprender únicamente de los datos de entrenamiento?

Esta consulta ayudó al equipo a comprender por qué el conjunto de prueba no
debe influir en las transformaciones aprendidas durante el entrenamiento.

### Consulta 4: programación orientada a objetos

> ¿Cómo se puede organizar el preprocesamiento en una clase de POO sin perder
> la explicación paso a paso presentada en los notebooks?

La respuesta se utilizó como guía para organizar la clase
`PreparadorDatos`, sus atributos y sus métodos.

### Consulta 5: corrección de errores en POO

> ¿Qué significa el error `AttributeError: 'PreparadorDatos' object has no
> attribute 'ejecutar_preprocesamiento'` y cómo se puede comprobar si el
> método quedó fuera de la clase?

La orientación permitió revisar:

- La ubicación de los archivos.
- La versión importada de la clase.
- La indentación.
- La definición de los métodos.
- La necesidad de reiniciar el intérprete o el kernel después de modificar
  un archivo.

### Consulta 6: redes neuronales

> ¿Cuál es la diferencia entre Sigmoid y Softmax, y por qué se utiliza
> Sigmoid en el modelo binario y Softmax en el modelo multiclase?

Esta consulta permitió comprender que:

- Sigmoid genera una salida entre cero y uno para la clasificación binaria.
- Softmax distribuye la probabilidad entre las cuatro categorías del modelo
  multiclase.

### Consulta 7: Dropout y EarlyStopping

> ¿Qué función cumplen Dropout y EarlyStopping, y cómo ayudan a reducir el
> sobreajuste?

La respuesta se utilizó para interpretar las configuraciones entrenadas y sus
curvas de validación.

### Consulta 8: interpretación de métricas

> ¿Por qué la accuracy no es suficiente si la clase Bueno representa el
> 70 % de los registros y la clase Malo representa el 30 %?

Esta consulta ayudó a justificar la utilización de:

- Precision.
- Recall.
- F1-score.
- Matriz de confusión.

También permitió identificar el recall de la clase Malo como una limitación
importante del modelo binario.

### Consulta 9: comparación de modelos

> ¿Cómo podemos comparar las configuraciones disponibles sin inventar métricas
> ni volver a entrenar los modelos?

La orientación permitió construir el notebook
`05_Comparacion_Modelos.ipynb` utilizando:

- Las pérdidas de validación registradas.
- La cantidad de épocas.
- Las métricas finales disponibles.
- Las matrices de confusión.
- Las limitaciones documentadas.

### Consulta 10: FastAPI

> ¿Cómo se puede exponer el modelo binario y el modelo multiclase mediante
> endpoints de FastAPI?

La respuesta se utilizó como orientación para organizar:

- La carga de los modelos.
- La validación de entradas.
- El preprocesamiento previo a la predicción.
- Las respuestas JSON.
- La documentación automática de los endpoints.

### Consulta 11: Streamlit

> ¿Cómo se puede construir una interfaz en Streamlit que envíe las 20
> variables originales a FastAPI y presente las respuestas de ambos modelos?

La orientación permitió organizar:

- El formulario.
- La solicitud a la API.
- Los resultados binarios.
- Los resultados multiclase.
- Las probabilidades.
- Los mensajes de conexión.
- El diseño visual de la aplicación.

### Consulta 12: control de versiones

> ¿Qué archivos deben incluirse en `.gitignore` y cuáles deben conservarse
> como parte de la entrega?

La consulta permitió diferenciar entre:

- Entornos virtuales y archivos temporales, que no deben subirse.
- Código, notebooks, modelos, datos procesados y documentación, que forman
  parte de la entrega.

---

## Reflexión sobre el aprendizaje

La inteligencia artificial fue útil como apoyo para formular preguntas,
recibir explicaciones y revisar alternativas cuando aparecían conceptos,
errores o decisiones que requerían mayor comprensión.

Las explicaciones paso a paso ayudaron al equipo especialmente en:

- La interpretación del dataset.
- La transformación de variables categóricas.
- La separación entre entrenamiento y prueba.
- La normalización.
- La prevención de fuga de información.
- La organización del código mediante programación orientada a objetos.
- La interpretación de las redes neuronales.
- La lectura de curvas y métricas.
- La integración entre modelos, API e interfaz.
- La preparación de la documentación.

La herramienta no sustituyó la ejecución ni la revisión del trabajo. Cada
bloque fue probado en PyCharm o en los notebooks, y los resultados se
comprobaron antes de continuar.

Cuando aparecieron errores, el equipo tuvo que:

- Revisar el código.
- Corregir caracteres.
- Ajustar nombres de archivos y rutas.
- Modificar la indentación.
- Comprobar dimensiones.
- Reiniciar kernels o intérpretes.
- Ejecutar nuevamente las pruebas.
- Comparar las salidas con los resultados esperados.

Este proceso permitió utilizar la inteligencia artificial como una guía de
aprendizaje, mientras que la validación y las decisiones finales quedaron a
cargo del equipo.

---

## Modificaciones y validaciones realizadas

Las sugerencias obtenidas mediante inteligencia artificial no se incorporaron
automáticamente.

El equipo realizó las siguientes revisiones y modificaciones:

- Comparó las sugerencias con el enunciado del proyecto.
- Comparó las sugerencias con la rúbrica.
- Revisó los notebooks proporcionados por el profesor.
- Conservó el orden general utilizado en los ejemplos del curso.
- Seleccionó `german.data` después de revisar la documentación del dataset.
- Revisó los nombres y el orden de las columnas.
- Ejecutó los notebooks para comprobar su funcionamiento.
- Verificó valores nulos.
- Verificó registros duplicados.
- Verificó dimensiones y tipos de datos.
- Conservó las 20 variables originales después de revisar su función.
- Comprobó que no existieran identificadores de clientes.
- Revisó posibles situaciones de fuga de información.
- Comprobó la transformación de las variables categóricas en 61 entradas.
- Verificó la división de 800 registros de entrenamiento y 200 registros de
  prueba.
- Comprobó que `MinMaxScaler` se ajustara únicamente con entrenamiento.
- Verificó la utilización de `stratify`.
- Documentó que los cuatro niveles de riesgo son una propuesta académica.
- Documentó que el puntaje de exposición no es una probabilidad.
- Probó los métodos de la clase `PreparadorDatos`.
- Verificó los archivos generados por el preprocesamiento.
- Comparó las configuraciones binarias existentes.
- Revisó las curvas de entrenamiento.
- Interpretó las matrices de confusión.
- Verificó las métricas finales de los modelos.
- Comprobó la carga de los archivos `.keras`.
- Validó los endpoints de FastAPI.
- Comprobó las respuestas con estado HTTP 200.
- Revisó las respuestas JSON.
- Comprobó la conexión entre Streamlit y FastAPI.
- Ejecutó predicciones desde la interfaz.
- Revisó los colores, el contraste y la navegación de Streamlit.
- Corrigió errores de sintaxis, indentación, rutas y estructura.
- Revisó la documentación para que coincidiera con los resultados realmente
  obtenidos.
- Evitó incluir métricas que no estuvieran respaldadas por los notebooks.
- Documentó las limitaciones existentes.

---

## Criterios aplicados para validar las sugerencias

Antes de aceptar una sugerencia, el equipo comprobó al menos uno de los
siguientes elementos:

1. Que coincidiera con el enunciado.
2. Que coincidiera con la rúbrica.
3. Que fuera compatible con los notebooks del curso.
4. Que funcionara con la estructura real del proyecto.
5. Que no modificara incorrectamente las dimensiones esperadas.
6. Que pudiera ejecutarse sin errores.
7. Que produjera resultados coherentes.
8. Que no generara fuga de información.
9. Que no sobrescribiera innecesariamente los modelos.
10. Que la documentación coincidiera con el código implementado.

Cuando una sugerencia no cumplía estos criterios, fue modificada o descartada.

---

## Limitaciones del uso de inteligencia artificial

El equipo reconoce las siguientes limitaciones:

- Las respuestas podían requerir adaptación a la estructura real del
  repositorio.
- Algunas rutas dependían de la ubicación desde la cual se ejecutaba el
  código.
- Las recomendaciones generales debían compararse con el material del curso.
- Las métricas no podían asumirse ni inventarse.
- Las sugerencias de código debían probarse antes de incorporarse.
- Las decisiones metodológicas requerían revisión por parte del equipo.
- La inteligencia artificial no sustituye el conocimiento experto.
- La documentación generada debía comprobarse contra los resultados reales.
- La herramienta podía proponer funcionalidades adicionales que no eran
  necesarias para cumplir con el proyecto.
- Los integrantes debían comprender el contenido antes de presentarlo.

---

## Transparencia en la documentación

El equipo procuró diferenciar claramente entre:

- Los resultados originales del dataset.
- Las transformaciones aplicadas.
- Las reglas construidas con fines académicos.
- Las predicciones de los modelos.
- Las sugerencias obtenidas con apoyo de inteligencia artificial.
- Las decisiones finales tomadas por el equipo.

No se presentaron como resultados reales métricas que no estuvieran
respaldadas por la ejecución de los notebooks.

También se documentaron las siguientes limitaciones:

- El recall reducido de la clase Malo.
- La ausencia de una evaluación independiente de prueba para la
  configuración binaria sin Dropout.
- La ausencia de una segunda configuración multiclase.
- La naturaleza académica de los niveles Bajo, Medio, Alto y Crítico.
- La imposibilidad de utilizar el prototipo para decisiones financieras
  reales.

---

## Responsabilidad del equipo

Todos los integrantes son responsables de revisar, comprender y explicar el
código y la documentación entregados.

La inteligencia artificial se utilizó como herramienta de:

- Consulta.
- Explicación.
- Orientación técnica.
- Revisión.
- Organización.
- Apoyo para resolver errores.

Las decisiones sobre:

- Los datos.
- El preprocesamiento.
- Las reglas académicas.
- Los modelos.
- La interpretación de los resultados.
- La API.
- La interfaz.
- La documentación.
- La entrega final.

fueron tomadas, ejecutadas y verificadas por el equipo.

---

## Declaración final

IA se utilizó como una herramienta complementaria de apoyo
durante el desarrollo del proyecto.

El equipo mantuvo la responsabilidad sobre:

- La comprensión del problema.
- La selección de los procedimientos.
- La integración del código.
- La ejecución de los notebooks.
- La validación de los modelos.
- La comprobación de la API.
- La prueba de la interfaz.
- La interpretación de las métricas.
- La elaboración de las conclusiones.
- La presentación del proyecto.

El uso de inteligencia artificial se documenta por razones de transparencia,
trazabilidad y responsabilidad académica.
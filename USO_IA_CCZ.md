# Uso de Inteligencia Artificial

## Herramienta utilizada

Durante el proyecto se utilize Microsoft 365 Copilot 
como herramienta de consulta y apoyo al aprendizaje.

La herramienta se utilizó principalmente para:

- Resolver preguntas sobre Python y programación orientada a objetos.
- Explicar instrucciones y errores con lenguaje sencillo.
- Aclarar conceptos relacionados con el análisis y preprocesamiento de datos.
- Comparar algunas decisiones con los ejemplos proporcionados por el profesor.
- Sugerir mejoras en comentarios, explicaciones y documentación.
- Orientar la solución de errores encontrados durante la ejecución.
- Resolver dudas básicas sobre PyCharm, Sourcetree y GitHub.

Las respuestas de la IA se utilizaron como orientación. El código fue
incorporado gradualmente, ejecutado y revisado por los integrantes antes de
considerarse parte del proyecto.

---

## Ejemplos de consultas realizadas

### Consulta 1: adaptación al material del curso

> ¿Cómo podemos mantener el código lo más parecido posible a los notebooks
> proporcionados por el profesor, pero adaptándolo a los requisitos del
> proyecto de riesgo crediticio?

Esta consulta ayudó a conservar el orden y la sencillez de los ejemplos vistos
en clase.

### Consulta 2: comprensión del dataset

> ¿Cuál es la diferencia entre `german.data` y `german.data-numeric`, y cuál
> conviene utilizar si el proyecto solicita aplicar `get_dummies`?

La respuesta se comparó con la documentación incluida con el dataset antes de
seleccionar el archivo principal.

### Consulta 3: preprocesamiento

> ¿Por qué se debe dividir el dataset antes de ajustar `MinMaxScaler`, y por
> qué el escalador debe aprender únicamente de los datos de entrenamiento?

Esta consulta ayudó a comprender cómo evitar que la información del conjunto
de prueba influyera en la preparación de los datos.

### Consulta 4: programación orientada a objetos

> ¿Cómo se puede organizar el preprocesamiento en una clase de POO sin perder
> la explicación paso a paso presentada en los notebooks?

La respuesta se utilizó como guía para organizar la clase `PreparadorDatos`.

### Consulta 5: corrección de errores

> ¿Qué significa el error `AttributeError: 'PreparadorDatos' object has no
> attribute 'ejecutar_preprocesamiento'` y cómo se puede comprobar si el
> método quedó fuera de la clase?

La orientación permitió revisar la ubicación de los archivos, la indentación
y los métodos de la clase.

---

## Reflexión sobre el aprendizaje

La IA fue útil como apoyo para formular preguntas y recibir explicaciones
inmediatas cuando aparecían conceptos o errores que no comprendía. Las
explicaciones paso a paso ayudaron especialmente a entender la transformación
de variables categóricas, la división de los datos, la normalización y la
organización del código mediante programación orientada a objetos.

La herramienta no sustituyó la ejecución ni la revisión del trabajo. Cada
bloque fue probado en PyCharm o en los notebooks, y los resultados se
comprobaron antes de continuar. Cuando aparecieron errores, fue necesario
revisar el código, corregir caracteres, modificar la indentación y ejecutar
nuevamente las pruebas. Esto permitió utilizar la IA como una guía de
aprendizaje, mientras la validación y las decisiones finales quedaron a cargo
de los estudiantes.

---

## Modificaciones y validaciones realizadas

Las sugerencias obtenidas mediante IA no se incorporaron automáticamente. Se
realizaron las siguientes revisiones y modificaciones:

- Se compararon las sugerencias con el enunciado y los notebooks del profesor.
- Se mantuvo el orden general utilizado en los ejemplos de clase.
- Se seleccionó `german.data` después de revisar la documentación del dataset.
- Se corrigieron los nombres y el orden de las columnas.
- Se ejecutaron los notebooks desde un kernel reiniciado para comprobar su
  funcionamiento completo.
- Se verificaron valores nulos, duplicados, dimensiones y tipos de datos.
- Se conservaron las 20 variables originales después de analizar si existían
  identificadores, fuga de información o variables sin utilidad evidente.
- Se comprobó la transformación de las variables categóricas en 61 entradas.
- Se verificó la división de 800 registros de entrenamiento y 200 de prueba.
- Se comprobó que `MinMaxScaler` se ajustara únicamente con entrenamiento.
- Se documentó que los cuatro niveles de riesgo son una propuesta académica y
  no una clasificación bancaria oficial.
- Se probaron los métodos de la clase `PreparadorDatos`.
- Se corrigieron errores de sintaxis, indentación y estructura.
- Se confirmó la ejecución final mediante código de salida `0`.
- Se revisó la documentación para que coincidiera con los resultados realmente
  obtenidos.

---

## Responsabilidad del grupo

Los integrantes son responsables de revisar, comprender y explicar el código
entregado.

La IA se utilizó como herramienta de consulta, explicación y apoyo técnico.
Las decisiones sobre los datos, la ejecución del código, la interpretación de
los resultados y la integración final del proyecto fueron realizadas y
verificadas por los estudiantes.
``
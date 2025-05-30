Este documento describe el flujo metodológico implementado para el realce, análisis y segmentación de imágenes de heridas crónicas de pie diabético, las cuales se clasifican en tres tipos de tejido: granulatorio, esfacelo y necrótico. El proceso se organiza en etapas iterativas que buscan optimizar la segmentación de estos tejidos.
Imagen de flujo metodologico



1. Preparación y Análisis Inicial de Imágenes
El proceso comienza con la Preparación de imágenes, donde se organizan y alistan las imágenes de las heridas para su procesamiento. Seguidamente, se realiza un Análisis cualitativo para una inspección inicial de las características de las imágenes.
2. Aplicación de Procesamiento Inicial (Realce.py)
La fase de Aplicación de procesamiento es la primera etapa de las "Etapas de una iteración". Aquí, el script Realce.py juega un rol central al aplicar técnicas de realce a las imágenes. Este script está diseñado para:
* Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization): Mejora el contraste local de las imágenes, siendo fundamental para resaltar detalles en zonas con poca variación de intensidad. La función aplicar_clahe permite configurar parámetros como clip_limit y tile_grid_size.
* Aplicar Realce Morfológico / de Colores: Aunque se refiere a operaciones morfológicas, la función aplicar_realce_colores en Realce.py se enfoca en realzar la saturación y normalizar el valor (V) en el espacio de color HSV, utilizando operaciones morfológicas (apertura, cierre, dilatación, erosión) sobre el canal V.
* Aplicar Transformación Gamma: Ajusta la luminosidad y el contraste de la imagen mediante la función aplicar_transformacion_gamma, controlando la relación tonal con un valor gamma.
Realce.py procesa las imágenes con configuraciones iniciales generales para cada técnica y las organiza en subcarpetas según el tipo de tejido (granulatorio, esfacelo, necrótico) y el conjunto de parámetros (SP1, SP2, etc.) utilizado.
3. Cálculo Estadístico de Parámetros HSV (Stats.py)
Una vez que las imágenes han sido realzadas, se procede al Cálculo estadístico de parámetros HSV. Este paso es manejado por el script Stats.py, el cual:
* Identifica la Región de la Herida: Utiliza la máscara de referencia para obtener las coordenadas de la herida (obtener_coordenadas_herida).
* Calcula Media y Desviación Estándar: La función calcular_media_desviacion determina la media y desviación estándar de los valores de Hue (H), Saturación (S) y Valor (V) dentro de la región de la herida.
* Establece Rangos Mínimos y Máximos Empíricos: A partir de la media y desviación estándar, se definen rangos HSV iniciales que representan los valores más comunes para cada tejido.
* Genera Visualizaciones y Reportes: Stats.py produce gráficas de media y desviación, boxplots de la distribución HSV y comparativas entre tejidos. También guarda los resultados numéricos de estos cálculos en archivos de texto, incluyendo un resumen global por cada combinación de tejido y técnica/parámetro. Estos resultados se conocen como Rangos HSV calculados empíricamente.
4. Optimización de Rangos HSV con Algoritmos Genéticos (gens.py)
Con los rangos HSV empíricos obtenidos, el siguiente paso es la Optimización de rangos HSV con algoritmos genéticos, implementada en el script gens.py. Este script:
* Carga Imágenes y Máscaras: Carga las imágenes procesadas y sus máscaras de referencia correspondientes.
* Ejecuta el Algoritmo Genético (GA): La función run_ga_with_ranges utiliza la librería DEAP para ejecutar un GA. Los individuos del GA representan combinaciones de dos rangos HSV (inferior y superior para H, S, V).
* Evaluación por Índice de Jaccard: La función evaluate_individual es la función de fitness del GA. Para cada individuo (conjunto de rangos HSV), se realiza una segmentación por umbral (threshold_segmentation) y se calcula el Índice de Jaccard comparando la segmentación resultante con la máscara de referencia (calculate_jaccard_index). El GA busca maximizar este índice.
* Resultados de Optimización: El GA retorna el "mejor individuo" (los rangos HSV optimizados) y el índice de Jaccard obtenido con la población inicial y el final. Estos son los Resultados iteración actual.
5. Ajuste de Parámetros y Postprocesamiento (postn.py)
Los rangos HSV optimizados obtenidos del algoritmo genético informan el Ajuste de parámetros, lo que implica refinar las configuraciones para futuras iteraciones o para la etapa final de segmentación.
Finalmente, se aplican técnicas de postprocesamiento para la segmentación utilizando el script postn.py. Este script se encarga de:
* Operaciones Morfológicas: Aplica operaciones morfológicas (cierre y apertura) a las máscaras binarizadas para refinar la segmentación y eliminar ruido (operaciones_morfologicas).
* Selección de Región Central: Identifica y selecciona la región de interés dentro de los 2/5 centrales de la imagen, lo que ayuda a enfocar la segmentación en el área relevante de la herida (seleccionar_region_central).
* Evaluación Cuantitativa: Crucialmente, postn.py también es responsable de calcular las métricas de evaluación del rendimiento de la segmentación. Utiliza la función calcular_metricas para determinar el Índice de Jaccard, F1-score, Precisión, Recall, Verdaderos Positivos (TP), Falsos Positivos (FP), Verdaderos Negativos (TN) y Falsos Negativos (FN), proporcionando los Resultados finales de la calidad de la segmentación.
Este procedimiento iterativo y multi-etapa permite un enfoque sistemático para el realce y la segmentación precisa de los diferentes tipos de tejido en heridas crónicas de pie diabético.
Conceptos Clave
Espacio de Color HSV
El espacio de color HSV (Hue, Saturation, Value) es un modelo de color alternativo al RGB (Red, Green, Blue) que se asemeja más a la forma en que el ojo humano percibe el color. Es particularmente útil en el procesamiento de imágenes para la segmentación, ya que sus componentes están más desacoplados de la iluminación que en el modelo RGB.
* Hue (Tono): Representa el color puro, como el rojo, verde o azul. Se mide en grados de 0 a 360 (aunque en OpenCV y otras librerías puede estar escalado a 0-179 para ajustarse a un byte). Es la cualidad que distingue un color de otro.
* Saturation (Saturación): Indica la pureza o intensidad del color. Un valor de saturación alto significa un color más puro y vibrante, mientras que un valor bajo indica un color más descolorido o grisáceo. Se mide como un porcentaje de 0% a 100% (o de 0 a 255).
* Value (Valor/Brillo): Describe la luminosidad o brillantez del color. Un valor alto significa un color más claro (más cercano al blanco), y un valor bajo significa un color más oscuro (más cercano al negro). También se mide como un porcentaje de 0% a 100% (o de 0 a 255).
La ventaja de HSV en aplicaciones de segmentación, como la que se describe en este flujo, radica en que los componentes de tono y saturación son menos sensibles a las variaciones de iluminación, lo que facilita la identificación de colores específicos de los tejidos independientemente de las condiciones de luz.
Índice de Jaccard (Coeficiente de Similitud de Jaccard)
El Índice de Jaccard, también conocido como Coeficiente de Similitud de Jaccard o Intersection Over Union (IoU), es una métrica utilizada para evaluar la similitud y diversidad de conjuntos de muestras. En el contexto de la segmentación de imágenes, se usa para medir la similitud entre la máscara de segmentación predicha y la máscara de referencia (verdad fundamental).
La fórmula del Índice de Jaccard es la siguiente:
J(A,B)=∣A∪B∣∣A∩B∣​=Aˊrea de UnioˊnAˊrea de Interseccioˊn​
Donde:
* A es el conjunto de píxeles en la máscara de segmentación predicha.
* B es el conjunto de píxeles en la máscara de referencia (verdad fundamental).
* ∣A∩B∣ representa el número de píxeles que son comunes tanto en la máscara predicha como en la máscara de referencia (intersección).
* ∣A∪B∣ representa el número de píxeles que están presentes en la máscara predicha o en la máscara de referencia, o en ambas (unión).
El valor del Índice de Jaccard oscila entre 0 y 1:
* Un valor de 0 indica que no hay superposición entre los dos conjuntos (no hay píxeles comunes).
* Un valor de 1 indica una superposición perfecta, es decir, la máscara predicha es idéntica a la máscara de referencia.
En el flujo de trabajo descrito, el Índice de Jaccard es la métrica principal utilizada para evaluar la efectividad de las técnicas de realce y segmentación, permitiendo cuantificar qué tan bien la segmentación automática se alinea con las etiquetas manuales de las heridas.

import os
import numpy as np
from skimage.io import imread, imsave
from skimage import img_as_ubyte
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
import cv2

# Función para operaciones morfológicas
def operaciones_morfologicas(mascara):
    kernel = np.ones((5, 5), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    return mascara

# Función para seleccionar la región dentro de los 2/5 centrales de la imagen
def seleccionar_region_central(mascara):
    height, width = mascara.shape
    # Definir límites para la región central (2/5 del área total)
    x_start = int(width * 1/5)
    x_end = int(width * 4/5)
    y_start = int(height * 1/5)
    y_end = int(height * 4/5)

    # Crear un ROI (Región de Interés) de los 2/5 centrales
    mascara_central = np.zeros_like(mascara)
    mascara_central[y_start:y_end, x_start:x_end] = mascara[y_start:y_end, x_start:x_end]

    # Encontrar contornos en la región central
    contornos, _ = cv2.findContours(mascara_central, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contornos:
        return np.zeros_like(mascara)

    # Seleccionar el contorno más grande en la región central
    contorno_mas_grande = max(contornos, key=cv2.contourArea)

    # Crear una nueva máscara con el contorno seleccionado
    nueva_mascara = np.zeros_like(mascara)
    cv2.drawContours(nueva_mascara, [contorno_mas_grande], -1, 255, thickness=cv2.FILLED)

    return nueva_mascara

# Función para calcular métricas de evaluación
def calcular_metricas(mascara_pred, referencia):
    referencia = (referencia > 0).astype(np.uint8)
    mascara_pred = (mascara_pred > 0).astype(np.uint8)

    TP = np.sum((mascara_pred == 1) & (referencia == 1))
    FP = np.sum((mascara_pred == 1) & (referencia == 0))
    TN = np.sum((mascara_pred == 0) & (referencia == 0))
    FN = np.sum((mascara_pred == 0) & (referencia == 1))

    jaccard = jaccard_score(referencia.flatten(), mascara_pred.flatten(), average='binary')
    f1 = f1_score(referencia.flatten(), mascara_pred.flatten(), average='binary')
    precision = precision_score(referencia.flatten(), mascara_pred.flatten(), average='binary')
    recall = recall_score(referencia.flatten(), mascara_pred.flatten(), average='binary')

    return TP, FP, TN, FN, jaccard, f1, precision, recall

# Procesar imágenes y calcular métricas
def procesar_imagenes(dir_entrada, dir_salida, dir_referencia, tecnica, tejido, parametro_id):
    if not os.path.exists(dir_salida):
        os.makedirs(dir_salida)

    imagenes = [f for f in os.listdir(dir_entrada) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    resultados_metricas = []

    for imagen_nombre in imagenes:
        # Extraer el ID de la imagen (suponemos que está en la segunda posición del nombre)
        id_imagen = imagen_nombre.split('_')[1]

        # Construir el nombre de la referencia
        referencia_path = os.path.join(dir_referencia, f"{id_imagen}.png")

        if not os.path.exists(referencia_path):
            print(f"Referencia no encontrada para {imagen_nombre}")
            continue

        # Leer imágenes
        imagen = imread(os.path.join(dir_entrada, imagen_nombre))
        referencia = imread(referencia_path, as_gray=True)

        # Convertir a escala de grises si es necesario
        if len(imagen.shape) == 3:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        # Postprocesamiento: Operaciones morfológicas
        mascara = operaciones_morfologicas(imagen)

        # Seleccionar la región central (2/5)
        mascara_final = seleccionar_region_central(mascara)

        # Guardar resultado
        output_path = os.path.join(dir_salida, f"procesada_{imagen_nombre}")
        imsave(output_path, img_as_ubyte(mascara_final))

        # Calcular métricas
        TP, FP, TN, FN, jaccard, f1, precision, recall = calcular_metricas(mascara_final, referencia)
        resultados_metricas.append((imagen_nombre, TP, FP, TN, FN, jaccard, f1, precision, recall))

        print(f"{imagen_nombre}: Jaccard={jaccard}, F1={f1}, Precision={precision}, Recall={recall}")

    # Guardar métricas en archivo
    with open(os.path.join(dir_salida, "metricas.txt"), 'w') as f:
        f.write("Imagen, TP, FP, TN, FN, Jaccard, F1, Precision, Recall\n")
        for res in resultados_metricas:
            f.write(",".join(map(str, res)) + "\n")

    # Calcular métricas promedio
    if resultados_metricas:
        metricas = np.array(resultados_metricas)[:, 1:].astype(float)
        promedios = np.mean(metricas, axis=0)

        with open(os.path.join(dir_salida, "metricas_promedio.txt"), 'w') as f:
            f.write(f"Técnica: {tecnica}, Tejido: {tejido}, ID Parámetro: {parametro_id}\n")
            f.write(f"Promedios: TP={promedios[0]}, FP={promedios[1]}, TN={promedios[2]}, FN={promedios[3]}, "
                    f"Jaccard={promedios[4]}, F1={promedios[5]}, Precision={promedios[6]}, Recall={promedios[7]}\n")

if __name__ == "__main__":
    dir_entrada = r"C:\Users\Kevin\Desktop\Data\Aplicacion CLAHE\Heridas_procesadas\binarizado\esfacelo\CLAHE_SP1_bin"
    dir_salida = r"C:\Users\Kevin\Desktop\Data\Aplicacion CLAHE\Heridas_procesadas\binarizado\esfacelo\CLAHE_SP1_result"
    dir_referencia = r"C:\Users\Kevin\Desktop\Data\Aplicacion CLAHE\Tejido Esfacelo\Etiquetas esfacelo recortado"

    tecnica = "CLAHE"
    tejido = "esfacelo"
    parametro_id = "SP1"

    procesar_imagenes(dir_entrada, dir_salida, dir_referencia, tecnica, tejido, parametro_id)

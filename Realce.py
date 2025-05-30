import cv2
import numpy as np
import os
from skimage.io import imread, imsave
from skimage import img_as_ubyte

# --- Funciones de Procesamiento ---
def aplicar_clahe(imagen, clip_limit, tile_grid_size):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    if len(imagen.shape) == 3 and imagen.shape[2] == 3:  # Imagen en color
        imagen_lab = cv2.cvtColor(imagen, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(imagen_lab)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    else:  # Imagen en escala de grises
        return clahe.apply(imagen)

def aplicar_transformacion_gamma(imagen, gamma):
    inv_gamma = 1.0 / gamma
    tabla = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(imagen, tabla)

def aplicar_realce_colores(imagen, kernel_size, operacion, incremento_saturacion):
    hsv = cv2.cvtColor(imagen, cv2.COLOR_RGB2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if operacion == 'apertura':
        hsv[:, :, 2] = cv2.morphologyEx(hsv[:, :, 2], cv2.MORPH_OPEN, kernel)
    elif operacion == 'cierre':
        hsv[:, :, 2] = cv2.morphologyEx(hsv[:, :, 2], cv2.MORPH_CLOSE, kernel)
    elif operacion == 'dilatacion':
        hsv[:, :, 2] = cv2.dilate(hsv[:, :, 2], kernel)
    elif operacion == 'erosion':
        hsv[:, :, 2] = cv2.erode(hsv[:, :, 2], kernel)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * incremento_saturacion, 0, 255)
    hsv[:, :, 2] = cv2.normalize(hsv[:, :, 2], None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

# --- Proceso Principal ---
def procesar_imagenes(dir_entrada, dir_salida, parametros):
    if not os.path.exists(dir_salida):
        os.makedirs(dir_salida)

    imagenes = [f for f in os.listdir(dir_entrada) if os.path.isfile(os.path.join(dir_entrada, f))]
    for nombre_imagen in imagenes:
        ruta_imagen = os.path.join(dir_entrada, nombre_imagen)
        imagen = imread(ruta_imagen)
        
        # Extraer tipo de tejido del nombre de la imagen
        try:
            tipo_tejido = nombre_imagen.split("_")[1].split(".")[0]
        except IndexError:
            print(f"Error procesando el nombre de la imagen: {nombre_imagen}")
            continue

        subcarpeta_tejido = os.path.join(dir_salida, tipo_tejido)
        if not os.path.exists(subcarpeta_tejido):
            os.makedirs(subcarpeta_tejido)

        id_imagen = os.path.splitext(nombre_imagen)[0]

        for tecnica, param_list in parametros.items():
            for idx, params in enumerate(param_list):
                if tecnica == 'clahe':
                    imagen_procesada = aplicar_clahe(imagen, params['clip_limit'], params['tile_grid_size'])
                elif tecnica == 'gamma':
                    imagen_procesada = aplicar_transformacion_gamma(imagen, params['gamma'])
                elif tecnica == 'realce':
                    imagen_procesada = aplicar_realce_colores(imagen, params['kernel_size'], params['operacion'], params['incremento_saturacion'])
                else:
                    raise ValueError("Técnica desconocida")

                # Crear subcarpeta para técnica y set de parámetros dentro del tipo de tejido
                subcarpeta_tecnica = os.path.join(subcarpeta_tejido, f"{tecnica.upper()}_SP{idx + 1}")
                os.makedirs(subcarpeta_tecnica, exist_ok=True)

                # Guardar imagen procesada
                nombre_procesada = f"{id_imagen}_{tecnica}_sp{idx + 1}.png"
                ruta_guardado = os.path.join(subcarpeta_tecnica, nombre_procesada)
                imsave(ruta_guardado, img_as_ubyte(imagen_procesada))

                print(f"Guardado: {ruta_guardado}")


# --- Parámetros de Procesamiento ---
#Segunda vuelta con parámetros distintos para cada tejido(necrotico)
# parametros = {
#     'clahe': [{'clip_limit': 1.0, 'tile_grid_size': 4},
#               {'clip_limit': 1.0, 'tile_grid_size': 8},
#               {'clip_limit': 1.0, 'tile_grid_size': 16},
#               {'clip_limit': 3.5, 'tile_grid_size': 24},
#               {'clip_limit': 3.5, 'tile_grid_size': 32},
#               {'clip_limit': 3.5, 'tile_grid_size': 40}],
#     'gamma': [{'gamma': 1.1}, {'gamma': 1.2}, {'gamma': 1.3}, {'gamma': 1.4}, {'gamma': 1.5}, {'gamma': 1.6}],
#     'realce': [{'kernel_size': 11, 'operacion': 'erosion', 'incremento_saturacion': 1.4},
#                {'kernel_size': 13, 'operacion': 'apertura', 'incremento_saturacion': 1.4},
#                {'kernel_size': 11, 'operacion': 'erosion', 'incremento_saturacion': 1.6},
#                {'kernel_size': 13, 'operacion': 'apertura', 'incremento_saturacion': 1.6},
#                {'kernel_size': 11, 'operacion': 'erosion', 'incremento_saturacion': 1.8},
#                {'kernel_size': 13, 'operacion': 'apertura', 'incremento_saturacion': 1.8}]
# }
#Segunda vuelta con parámetros distintos para cada tejido(esfacelo)
#parametros = {
#    'clahe': [{'clip_limit': 1.0, 'tile_grid_size': 8},
#              {'clip_limit': 2.0, 'tile_grid_size': 4},
#              {'clip_limit': 1.5, 'tile_grid_size': 8},
#              {'clip_limit': 1.5, 'tile_grid_size': 4},
#              {'clip_limit': 2.5, 'tile_grid_size': 8},
#              {'clip_limit': 2.5, 'tile_grid_size': 4}],
#    'gamma': [{'gamma': 0.7}, {'gamma': 0.8}, {'gamma': 0.9}, {'gamma': 1.1}, {'gamma': 1.2}, {'gamma': 1.3}],
#    'realce': [{'kernel_size': 5, 'operacion': 'apertura', 'incremento_saturacion': 1.2},
#               {'kernel_size': 7, 'operacion': 'cierre', 'incremento_saturacion': 1.4},
#               {'kernel_size': 5, 'operacion': 'apertura', 'incremento_saturacion': 1.6},
#               {'kernel_size': 7, 'operacion': 'cierre', 'incremento_saturacion': 1.2},
#               {'kernel_size': 5, 'operacion': 'apertura', 'incremento_saturacion': 1.4},
#               {'kernel_size': 7, 'operacion': 'cierre', 'incremento_saturacion': 1.6}]
#}
#Segunda vuelta con parámetros distintos para cada tejido(granulatorio)
#parametros = {
#    'clahe': [{'clip_limit': 1.0, 'tile_grid_size': 8},
#              {'clip_limit': 2.0, 'tile_grid_size': 4},
#              {'clip_limit': 1.5, 'tile_grid_size': 8},
#              {'clip_limit': 1.5, 'tile_grid_size': 4},
#              {'clip_limit': 2.5, 'tile_grid_size': 8},
#              {'clip_limit': 2.5, 'tile_grid_size': 4}],
#    'gamma': [{'gamma': 0.7}, {'gamma': 0.8}, {'gamma': 0.9}, {'gamma': 1.1}, {'gamma': 1.2}, {'gamma': 1.3}],
#    'realce': [{'kernel_size': 5, 'operacion': 'apertura', 'incremento_saturacion': 1.2},
#               {'kernel_size': 7, 'operacion': 'cierre', 'incremento_saturacion': 1.4},
#               {'kernel_size': 5, 'operacion': 'apertura', 'incremento_saturacion': 1.6},
#               {'kernel_size': 7, 'operacion': 'cierre', 'incremento_saturacion': 1.2},
#               {'kernel_size': 5, 'operacion': 'apertura', 'incremento_saturacion': 1.4},
#               {'kernel_size': 7, 'operacion': 'cierre', 'incremento_saturacion': 1.6}]
#}
#Primera vuelta con parámetros iguales para cada tejido
parametros = {
    'clahe': [{'clip_limit': 1.0, 'tile_grid_size': 4},
              {'clip_limit': 2.0, 'tile_grid_size': 8},
              {'clip_limit': 2.5, 'tile_grid_size': 16},
              {'clip_limit': 3.0, 'tile_grid_size': 24},
              {'clip_limit': 3.5, 'tile_grid_size': 32},
              {'clip_limit': 4.0, 'tile_grid_size': 40}],
    'gamma': [{'gamma': 0.4}, {'gamma': 0.6}, {'gamma': 0.8}, {'gamma': 1.0}, {'gamma': 1.2}, {'gamma': 1.4}],
    'realce': [{'kernel_size': 5, 'operacion': 'apertura', 'incremento_saturacion': 1.2},
               {'kernel_size': 7, 'operacion': 'cierre', 'incremento_saturacion': 1.4},
               {'kernel_size': 9, 'operacion': 'dilatacion', 'incremento_saturacion': 1.6},
               {'kernel_size': 11, 'operacion': 'erosion', 'incremento_saturacion': 1.8},
               {'kernel_size': 13, 'operacion': 'apertura', 'incremento_saturacion': 1.6},
               {'kernel_size': 13, 'operacion': 'cierre', 'incremento_saturacion': 1.8}]
}

# --- Ejecución Principal ---
if __name__ == "__main__":
    dir_entrada = r"C:\Users\Kevin\Desktop\Data\Aplicacion CLAHE\Heridas_todas"
    dir_salida = r"C:\Users\Kevin\Desktop\Data\Aplicacion CLAHE\Heridas_procesadas"
    #dir_salida = r"C:\Users\Kevin\Desktop\Data\Aplicacion CLAHE\Resultados Procesados"
    procesar_imagenes(dir_entrada, dir_salida, parametros)

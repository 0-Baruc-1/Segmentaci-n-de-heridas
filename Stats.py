import cv2
import numpy as np
import os
from skimage.io import imread
import matplotlib.pyplot as plt

# Definimos los valores mínimos y máximos teóricos para HSV
valores_minimos_posibles_hsv = [0, 0, 0]   # H, S, V
valores_maximos_posibles_hsv = [179, 255, 255]  # H, S, V

def obtener_coordenadas_herida(mascara_referencia):
    contornos, _ = cv2.findContours(mascara_referencia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contornos:
        raise ValueError("No se encontraron contornos en la máscara de referencia.")
    
    contorno_mas_grande = max(contornos, key=cv2.contourArea)
    x, y, ancho, alto = cv2.boundingRect(contorno_mas_grande)
    return x, y, ancho, alto

def graficar_media_desviacion(medias, desviaciones, nombre_imagen, dir_salida):
    componentes = ['Hue', 'Saturation', 'Value']
    x = np.arange(len(componentes))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 0.2, medias, 0.4, label='Media', color='blue')
    ax.bar(x + 0.2, desviaciones, 0.4, label='Desviación Estándar', color='orange')
    
    ax.set_xlabel('Componentes HSV')
    ax.set_ylabel('Valor')
    ax.set_title(f'Media y Desviación Estándar - {nombre_imagen}')
    ax.set_xticks(x)
    ax.set_xticklabels(componentes)
    ax.legend()
    
    plt.savefig(os.path.join(dir_salida, f"media_desviacion_{nombre_imagen}.png"))
    plt.close()

def graficar_boxplot(herida_recortada, herida_mascara, nombre_imagen, dir_salida):
    herida_hsv = cv2.cvtColor(herida_recortada, cv2.COLOR_RGB2HSV)
    
    mascara_valida = herida_mascara > 0
    herida_hsv_valida = herida_hsv[mascara_valida]

    if herida_hsv_valida.size == 0:
        print(f"No hay píxeles válidos para graficar el boxplot en {nombre_imagen}.")
        return

    meanprops = dict(marker='o', markerfacecolor='red', markeredgecolor='black', markersize=8)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot([herida_hsv_valida[:, 0], 
                herida_hsv_valida[:, 1], 
                herida_hsv_valida[:, 2]],
               labels=['Hue', 'Saturation', 'Value'],
               showmeans=True,  
               meanprops=meanprops)  
    
    ax.set_title(f'Distribución de HSV - {nombre_imagen}')
    ax.set_ylabel('Valor')

    plt.savefig(os.path.join(dir_salida, f"boxplot_{nombre_imagen}.png"))
    plt.close()

def generar_graficas_comparativas(vectores_min_max, dir_salida):
    # Colores para cada tipo de tejido
    colores = {
        'granulatorio': 'red',
        'esfacelo': 'yellow',
        'necrotico': 'black'
    }

    # Listas de valores para cada canal y su respectivo mínimo y máximo
    parametros = ['h_min', 'h_max', 's_min', 's_max', 'v_min', 'v_max']
    nombres_graficas = {
        'h_min': 'Comparación de H_min por Tejido',
        'h_max': 'Comparación de H_max por Tejido',
        's_min': 'Comparación de S_min por Tejido',
        's_max': 'Comparación de S_max por Tejido',
        'v_min': 'Comparación de V_min por Tejido',
        'v_max': 'Comparación de V_max por Tejido'
    }

    # Crear una gráfica para cada parámetro
    for parametro in parametros:
        plt.figure(figsize=(10, 6))
        for tejido, color in colores.items():
            if parametro in vectores_min_max[tejido]:
                plt.plot(vectores_min_max[tejido][parametro],
                         label=tejido.capitalize(), color=color)

        plt.xlabel('Índice de Imagen')
        plt.ylabel('Valor HSV')
        plt.title(nombres_graficas[parametro])
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(dir_salida, f"{parametro}_comparativa.png"))
        plt.close()

def calcular_z_scores(herida_hsv_valida, medias, desviaciones):
    z_scores_hue = (herida_hsv_valida[:, 0] - medias[0]) / desviaciones[0]
    z_scores_saturation = (herida_hsv_valida[:, 1] - medias[1]) / desviaciones[1]
    z_scores_value = (herida_hsv_valida[:, 2] - medias[2]) / desviaciones[2]
    return z_scores_hue, z_scores_saturation, z_scores_value

def calcular_media_desviacion(herida_recortada, herida_mascara):
    herida_hsv = cv2.cvtColor(herida_recortada, cv2.COLOR_RGB2HSV)
    
    mascara_valida = herida_mascara > 0
    herida_hsv_valida = herida_hsv[mascara_valida]

    # Reemplazamos ceros por NaN para no sesgar la estadística
    herida_hsv_valida = np.where(herida_hsv_valida == 0, np.nan, herida_hsv_valida)

    medias = np.nanmean(herida_hsv_valida, axis=0)
    desviaciones = np.nanstd(herida_hsv_valida, axis=0)

    rangos_min = medias - desviaciones
    rangos_max = medias + desviaciones

    condicion = (
        (herida_hsv_valida >= rangos_min) &
        (herida_hsv_valida <= rangos_max) &
        (herida_hsv_valida >= [0, 0, 0]) &
        (herida_hsv_valida <= [179, 255, 255])
    )
    condicion = np.all(condicion, axis=1)
    pixels_within_ranges = herida_hsv_valida[condicion]

    if pixels_within_ranges.size == 0:
        print("No hay píxeles válidos que cumplan con las condiciones para esta imagen.")
        min_hsv = [np.nan, np.nan, np.nan]
        max_hsv = [np.nan, np.nan, np.nan]
    else:
        min_hsv = np.nanmin(pixels_within_ranges, axis=0)
        max_hsv = np.nanmax(pixels_within_ranges, axis=0)

    return medias, desviaciones, rangos_min, rangos_max, min_hsv, max_hsv


def procesar_subcarpeta(
    dir_subcarpeta,      # p. ej., ...\esfacelo\CLAHE_SP1
    tissue_expected,     # 'esfacelo', 'necrotico', 'granulatorio'
    dir_referencia,
    dir_salida,
    vectores_min_max
):
    """
    Procesa todas las imágenes dentro de una subcarpeta (con técnicas específicas),
    asumiendo que pertenecen al tejido `tissue_expected`, y guarda los resultados 
    en `dir_salida`. Actualiza el diccionario `vectores_min_max` con los valores 
    min/max de cada tejido.
    """

    # Crear en el directorio de salida una subcarpeta equivalente
    nombre_subcarpeta = os.path.basename(dir_subcarpeta)  # p. ej.: CLAHE_SP1
    dir_subcarpeta_salida = os.path.join(dir_salida, tissue_expected, nombre_subcarpeta)
    os.makedirs(dir_subcarpeta_salida, exist_ok=True)

    # Diccionario para estadísticas de la SUBCARPETA
    # Aquí almacenaremos los min y max de cada imagen procesada,
    # luego haremos un resumen global al final de la subcarpeta.
    subcarpeta_stats = {
        'h_min': [],
        'h_max': [],
        's_min': [],
        's_max': [],
        'v_min': [],
        'v_max': []
    }

    # Listar archivos en la subcarpeta de entrada
    for filename in os.listdir(dir_subcarpeta):
        if not (filename.lower().endswith(".png") or filename.lower().endswith(".jpg")):
            continue

        base_name, ext = os.path.splitext(filename)
        # p. ej.: "0031_granulatorio_clahe_sp1" => parts = ['0031','granulatorio','clahe','sp1']
        parts = base_name.split('_')
        if len(parts) < 2:
            print(f"Formato de nombre inesperado en: {filename}, se ignora.")
            continue

        # La segunda parte (parts[1]) debería ser el tejido reportado
        tissue_in_filename = parts[1].lower()
        if tissue_in_filename != tissue_expected:
            print(f"Tejido en el nombre ({tissue_in_filename}) no coincide con la carpeta ({tissue_expected}). Ignorando {filename}.")
            continue

        # La primera parte es el ID de la imagen, p. ej. "0031"
        image_id = parts[0]
        # Construimos el nombre de referencia (p. ej. "0031_granulatorio.png")
        ref_filename = f"{image_id}_{tissue_in_filename}{ext}"
        ruta_referencia = os.path.join(dir_referencia, ref_filename)

        if not os.path.exists(ruta_referencia):
            print(f"No se encontró etiqueta para la imagen {filename}. Se ignora.")
            continue

        ruta_imagen = os.path.join(dir_subcarpeta, filename)
        imagen = imread(ruta_imagen)
        mascara_referencia = imread(ruta_referencia, as_gray=True)
        mascara_referencia = (mascara_referencia > 0).astype(np.uint8) * 255

        try:
            x, y, ancho, alto = obtener_coordenadas_herida(mascara_referencia)
        except ValueError as e:
            print(f"Error en {filename}: {e}")
            continue

        herida = imagen[y:y+alto, x:x+ancho]
        herida_mascara = mascara_referencia[y:y+alto, x:x+ancho]
        herida_recortada = cv2.bitwise_and(herida, herida, mask=herida_mascara)

        # Guardamos la región de la herida en un PNG
        plt.imshow(herida_recortada)
        plt.title(f"Región de la Herida - {filename}")
        plt.savefig(os.path.join(dir_subcarpeta_salida, f"region_herida_{filename}.png"))
        plt.close()

        # Cálculos estadísticos
        medias, desviaciones, rangos_min, rangos_max, min_hsv, max_hsv = calcular_media_desviacion(
            herida_recortada, herida_mascara)

        # Graficar y guardar media/desviación
        graficar_media_desviacion(medias, desviaciones, filename, dir_subcarpeta_salida)
        # Graficar y guardar boxplot
        graficar_boxplot(herida_recortada, herida_mascara, filename, dir_subcarpeta_salida)

        # Actualizar el diccionario min/max global (por tejido)
        if not np.isnan(min_hsv).any() and not np.isnan(max_hsv).any():
            vectores_min_max[tissue_in_filename]['h_min'].append(min_hsv[0])
            vectores_min_max[tissue_in_filename]['h_max'].append(max_hsv[0])
            vectores_min_max[tissue_in_filename]['s_min'].append(min_hsv[1])
            vectores_min_max[tissue_in_filename]['s_max'].append(max_hsv[1])
            vectores_min_max[tissue_in_filename]['v_min'].append(min_hsv[2])
            vectores_min_max[tissue_in_filename]['v_max'].append(max_hsv[2])

            # También guardamos la información a nivel de SUBCARPETA
            subcarpeta_stats['h_min'].append(min_hsv[0])
            subcarpeta_stats['h_max'].append(max_hsv[0])
            subcarpeta_stats['s_min'].append(min_hsv[1])
            subcarpeta_stats['s_max'].append(max_hsv[1])
            subcarpeta_stats['v_min'].append(min_hsv[2])
            subcarpeta_stats['v_max'].append(max_hsv[2])

        # Guardar resultados numéricos (por imagen) en un archivo de texto
        with open(os.path.join(dir_subcarpeta_salida, f"resultados_{filename}.txt"), 'w') as f:
            f.write(f"Imagen: {filename}\n")
            f.write(f"Tejido: {tissue_in_filename.capitalize()}\n")
            f.write(f"Medias (H, S, V): {medias}\n")
            f.write(f"Desviaciones (H, S, V): {desviaciones}\n")
            f.write(f"Rangos Mínimos (H, S, V): {rangos_min}\n")
            f.write(f"Rangos Máximos (H, S, V): {rangos_max}\n")
            f.write(f"Valores Mínimos (H, S, V): {min_hsv}\n")
            f.write(f"Valores Máximos (H, S, V): {max_hsv}\n")

    # === RESUMEN GLOBAL DE LA SUBCARPETA (técnica y set de parámetros) ===
    # Calculamos el min, max y promedio para cada parámetro (h_min, h_max, etc.)
    # y lo guardamos en un archivo de texto con el formato deseado, PERO en el
    # directorio grande (dir_salida), NO dentro de la subcarpeta.

    # Nombre que refleje tejido y subcarpeta, p. ej.: 
    # "estadisticas_globales_esfacelo_CLAHE_SP1.txt"
    nombre_archivo_global = f"estadisticas_globales_{tissue_expected}_{nombre_subcarpeta}.txt"
    ruta_archivo_global = os.path.join(dir_salida, nombre_archivo_global)

    parametros_orden = [
        ('H_min', subcarpeta_stats['h_min']),
        ('H_max', subcarpeta_stats['h_max']),
        ('S_min', subcarpeta_stats['s_min']),
        ('S_max', subcarpeta_stats['s_max']),
        ('V_min', subcarpeta_stats['v_min']),
        ('V_max', subcarpeta_stats['v_max']),
    ]

    with open(ruta_archivo_global, 'w') as f:
        # Encabezado
        f.write("Vector   Minimum   Maximum   Average\n")
        # Recorremos cada parámetro y calculamos sus estadísticos
        for idx, (nombre_param, valores) in enumerate(parametros_orden):
            if len(valores) == 0:
                # Si no hay valores (no se procesaron imágenes válidas)
                continue
            param_min = np.min(valores)
            param_max = np.max(valores)
            param_avg = np.mean(valores)

            # Formato de línea: índice, nombre de vector, min, max, promedio
            f.write(f"{idx:<5d} {nombre_param:<7s} {param_min:<9.2f} {param_max:<9.2f} {param_avg:<.6f}\n")

    print(f"Resumen global guardado en: {ruta_archivo_global}")

def procesar_imagenes_por_tejido(
    dir_entrada,     # carpeta que contiene \esfacelo, \necrotico, \granulatorio
    dir_referencia,  # carpeta con imágenes de referencia (ej: "0012_esfacelo.png")
    dir_salida       # carpeta de salida principal
):
    """
    Recorre cada subcarpeta de cada tejido (esfacelo, necrotico, granulatorio),
    procesa las imágenes encontradas y guarda los resultados en 'dir_salida'.
    """

    vectores_min_max = {
        'granulatorio': {'h_min': [], 'h_max': [], 's_min': [], 's_max': [], 'v_min': [], 'v_max': []},
        'esfacelo':     {'h_min': [], 'h_max': [], 's_min': [], 's_max': [], 'v_min': [], 'v_max': []},
        'necrotico':    {'h_min': [], 'h_max': [], 's_min': [], 's_max': [], 'v_min': [], 'v_max': []}
    }

    tejidos = ['esfacelo', 'necrotico', 'granulatorio']
    os.makedirs(dir_salida, exist_ok=True)

    for tejido in tejidos:
        dir_tejido = os.path.join(dir_entrada, tejido)
        if not os.path.exists(dir_tejido):
            print(f"No se encontró la carpeta para el tejido: {tejido}")
            continue

        # Para cada subcarpeta (CLAHE_SP1, CLAHE_SP2, ...)
        for subcarpeta in os.listdir(dir_tejido):
            dir_subcarpeta = os.path.join(dir_tejido, subcarpeta)
            if not os.path.isdir(dir_subcarpeta):
                continue

            procesar_subcarpeta(
                dir_subcarpeta=dir_subcarpeta,
                tissue_expected=tejido,
                dir_referencia=dir_referencia,
                dir_salida=dir_salida,
                vectores_min_max=vectores_min_max
            )

    # Guardar vectores min_max en un txt global (consolidado para TODO el tejido)
    txt_global_path = os.path.join(dir_salida, "vectores_min_max_por_tejido.txt")
    with open(txt_global_path, 'w') as f:
        for tissue, data in vectores_min_max.items():
            f.write(f"Tejido: {tissue}\n")
            f.write(f"H_min: {data['h_min']}\n")
            f.write(f"H_max: {data['h_max']}\n")
            f.write(f"S_min: {data['s_min']}\n")
            f.write(f"S_max: {data['s_max']}\n")
            f.write(f"V_min: {data['v_min']}\n")
            f.write(f"V_max: {data['v_max']}\n\n")

    print("Procesamiento completado. Resultados y gráficas guardados en:", dir_salida)

    # Graficar comparativas finales entre tejidos (en un solo lugar)
    generar_graficas_comparativas(vectores_min_max, dir_salida)

if __name__ == "__main__":
    # Ajusta estas rutas según tu estructura local
    dir_entrada = r"C:\Users\Kevin\Desktop\Data\Aplicacion CLAHE\Heridas_procesadas"  
    dir_referencia = r"C:\Users\Kevin\Desktop\Data\Aplicacion CLAHE\Etiquetas_todas"    
    dir_salida = r"C:\Users\Kevin\Desktop\Data\Datos_3"

    procesar_imagenes_por_tejido(dir_entrada, dir_referencia, dir_salida)

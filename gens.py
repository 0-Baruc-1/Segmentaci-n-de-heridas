import os
import cv2
import numpy as np
import random
from deap import base, creator, tools, algorithms

##############################################################################
# 1) Funciones auxiliares
##############################################################################

def load_images_from_folder(folder):
    """
    Carga todas las imágenes .png/.jpg de 'folder' y retorna una lista de (filename, image).
    """
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(".png") or filename.lower().endswith(".jpg"):
            path = os.path.join(folder, filename)
            img = cv2.imread(path)
            if img is not None:
                images.append((filename, img))
    return images

def load_mask_by_name(mask_folder, image_filename):
    """
    Empareja una imagen procesada como '0051_esfacelo_clahe_sp1.png'
    con una máscara '0051_esfacelo.png' o '.jpg' en 'mask_folder'.
    """
    base_name, _ = os.path.splitext(image_filename)  # "0051_esfacelo_clahe_sp1"
    parts = base_name.split('_')
    if len(parts) < 2:
        return None
    
    # '0051_esfacelo'
    id_tejido = parts[0] + "_" + parts[1]

    mask_path_png = os.path.join(mask_folder, id_tejido + ".png")
    if os.path.exists(mask_path_png):
        return cv2.imread(mask_path_png)

    mask_path_jpg = os.path.join(mask_folder, id_tejido + ".jpg")
    if os.path.exists(mask_path_jpg):
        return cv2.imread(mask_path_jpg)

    return None

def read_ranges_from_txt(txt_path):
    """
    Lee un archivo .txt con 6 líneas (H_min, H_max, S_min, S_max, V_min, V_max),
    generando (r1_lower, r1_upper, r2_lower, r2_upper).
    """
    data = {}
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for i in range(1, 7):
        line = lines[i].strip()
        parts = line.split()
        param_name = parts[1]  # "H_min", ...
        min_value = float(parts[2])
        max_value = float(parts[3])
        data[param_name] = (min_value, max_value)

    # Construir Rango1, Rango2
    r1_lower = [
        int(data["H_min"][0]),
        int(data["S_min"][0]),
        int(data["V_min"][0])
    ]
    r1_upper = [
        int(data["H_max"][0]),
        int(data["S_max"][1]),
        int(data["V_max"][1])
    ]
    r2_lower = [
        int(data["H_min"][1]),
        int(data["S_min"][0]),
        int(data["V_min"][0])
    ]
    r2_upper = [
        int(data["H_max"][1]),
        int(data["S_max"][1]),
        int(data["V_max"][1])
    ]

    return r1_lower, r1_upper, r2_lower, r2_upper

##############################################################################
# 2) Función GA para optimizar 2 rangos HSV
##############################################################################

def run_ga_with_ranges(
    r1_lower, r1_upper,
    r2_lower, r2_upper,
    images, masks,
    pop_size=20, ngen=30,
    cxpb=0.5, mutpb=0.2
):
    """
    Corre un GA para optimizar dos rangos HSV, retornando:
        (best_individual, best_fitness_initial, best_fitness_final)
    
    * best_fitness_initial = mejor índice de Jaccard en la población inicial.
    * best_fitness_final   = mejor índice de Jaccard tras todas las generaciones.
    """

    # Definiciones DEAP (evitar redefinirlas múltiples veces)
    try:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    except:
        pass
    try:
        creator.create("Individual", list, fitness=creator.FitnessMax)
    except:
        pass

    toolbox = base.Toolbox()

    # Inicialización de atributos dentro de los rangos
    def init_h1_low():
        return random.randint(r1_lower[0], r1_upper[0])
    def init_s1_low():
        return random.randint(r1_lower[1], r1_upper[1])
    def init_v1_low():
        return random.randint(r1_lower[2], r1_upper[2])
    
    def init_h1_up():
        return random.randint(r1_lower[0], r1_upper[0])
    def init_s1_up():
        return random.randint(r1_lower[1], r1_upper[1])
    def init_v1_up():
        return random.randint(r1_lower[2], r1_upper[2])

    def init_h2_low():
        return random.randint(r2_lower[0], r2_upper[0])
    def init_s2_low():
        return random.randint(r2_lower[1], r2_upper[1])
    def init_v2_low():
        return random.randint(r2_lower[2], r2_upper[2])

    def init_h2_up():
        return random.randint(r2_lower[0], r2_upper[0])
    def init_s2_up():
        return random.randint(r2_lower[1], r2_upper[1])
    def init_v2_up():
        return random.randint(r2_lower[2], r2_upper[2])

    toolbox.register("attr_h1_low", init_h1_low)
    toolbox.register("attr_s1_low", init_s1_low)
    toolbox.register("attr_v1_low", init_v1_low)
    toolbox.register("attr_h1_up", init_h1_up)
    toolbox.register("attr_s1_up", init_s1_up)
    toolbox.register("attr_v1_up", init_v1_up)

    toolbox.register("attr_h2_low", init_h2_low)
    toolbox.register("attr_s2_low", init_s2_low)
    toolbox.register("attr_v2_low", init_v2_low)
    toolbox.register("attr_h2_up", init_h2_up)
    toolbox.register("attr_s2_up", init_s2_up)
    toolbox.register("attr_v2_up", init_v2_up)

    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (
            toolbox.attr_h1_low, toolbox.attr_s1_low, toolbox.attr_v1_low,
            toolbox.attr_h1_up,  toolbox.attr_s1_up,  toolbox.attr_v1_up,
            toolbox.attr_h2_low, toolbox.attr_s2_low, toolbox.attr_v2_low,
            toolbox.attr_h2_up,  toolbox.attr_s2_up,  toolbox.attr_v2_up
        ),
        n=1
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def threshold_segmentation(image, lb1, ub1, lb2, ub2):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array(lb1, dtype=np.uint8), np.array(ub1, dtype=np.uint8))
        mask2 = cv2.inRange(hsv, np.array(lb2, dtype=np.uint8), np.array(ub2, dtype=np.uint8))
        combined = cv2.bitwise_or(mask1, mask2)
        return combined

    def calculate_jaccard_index(seg_mask, ref_mask):
        if len(ref_mask.shape) == 3:
            ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2GRAY)
        seg_bin = (seg_mask > 0).astype(np.uint8)
        ref_bin = (ref_mask > 0).astype(np.uint8)
        intersection = np.logical_and(seg_bin, ref_bin)
        union = np.logical_or(seg_bin, ref_bin)
        return np.sum(intersection) / np.sum(union)

    def evaluate_individual(ind):
        lb1 = [ind[0],  ind[1],  ind[2]]
        ub1 = [ind[3],  ind[4],  ind[5]]
        lb2 = [ind[6],  ind[7],  ind[8]]
        ub2 = [ind[9],  ind[10], ind[11]]
        total_jaccard = 0
        for img, msk in zip(images, masks):
            seg = threshold_segmentation(img, lb1, ub1, lb2, ub2)
            ji = calculate_jaccard_index(seg, msk)
            total_jaccard += ji
        return (total_jaccard / len(images),)

    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=255, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Crear población
    population = toolbox.population(n=pop_size)

    # ------------------------------------------------------------------------
    # Evaluar la población inicial para obtener el Jaccard inicial
    # ------------------------------------------------------------------------
    # 1) Calcular fitness de todos en la población inicial
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # 2) Obtener el mejor fitness (Jaccard) en la población inicial
    best_in_pop = tools.selBest(population, k=1)
    best_fitness_initial = best_in_pop[0].fitness.values[0]  # Se asume un solo valor (Jaccard)

    # ------------------------------------------------------------------------
    # Hall of Fame para capturar el mejor individuo final
    # ------------------------------------------------------------------------
    hof = tools.HallOfFame(1, similar=np.array_equal)

    # Estadísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # ------------------------------------------------------------------------
    # Correr el GA con la población ya evaluada
    # (eaSimple no reevalúa la población inicial si passfit=False en la 2.0,
    #  en caso de versiones anteriores, se evaluará de nuevo, no afecta el resultado)
    # ------------------------------------------------------------------------
    algorithms.eaSimple(
        population, toolbox,
        cxpb=cxpb, mutpb=mutpb,
        ngen=ngen, stats=stats,
        halloffame=hof, verbose=False
    )

    # best_individual final
    best_individual = hof[0]

    # Para conocer el fitness del best_individual final
    best_fitness_final = best_individual.fitness.values[0]

    return best_individual, best_fitness_initial, best_fitness_final

def main():
    # Directorios principales (ajusta a tu estructura real)
    dir_resultados = r"C:\Users\Kevin\Desktop\Data\Aplicacion CLAHE\Heridas_procesadas"
    dir_estadisticas_globales = r"C:\Users\Kevin\Desktop\Data\Datos_2\estadisticas_globales"
    dir_mascaras = r"C:\Users\Kevin\Desktop\Data\Aplicacion CLAHE\Etiquetas_todas"
    dir_salida = r"C:\Users\Kevin\Desktop\Data\Resultados_GA2"
    os.makedirs(dir_salida, exist_ok=True)

    # Archivo CSV para guardar resultados del GA
    csv_path = os.path.join(dir_salida, "ResultadosGA.csv")
    with open(csv_path, 'w') as f_out:
        # Encabezado con Jaccard inicial y final
        f_out.write("Tejido,Tecnica,JaccardInicial,JaccardFinal,BestIndividual\n")

        # Lista de tejidos
        tejidos = ["granulatorio", "necrotico", "esfacelo"]

        # Lista de tecnicas/subcarpetas
        tecnicas = [
            "CLAHE_SP1", "CLAHE_SP2", "CLAHE_SP3", "CLAHE_SP4", "CLAHE_SP5", "CLAHE_SP6",
            "GAMMA_SP1", "GAMMA_SP2", "GAMMA_SP3", "GAMMA_SP4", "GAMMA_SP5", "GAMMA_SP6",
            "REALCE_SP1", "REALCE_SP2", "REALCE_SP3", "REALCE_SP4", "REALCE_SP5", "REALCE_SP6"
        ]

        for tejido in tejidos:
            for tecnica in tecnicas:
                # Directorio de las imágenes para este tejido+técnica
                dir_subcarpeta = os.path.join(dir_resultados, tejido, tecnica)
                if not os.path.isdir(dir_subcarpeta):
                    print(f"No existe {dir_subcarpeta}, se omite.")
                    continue

                # .txt con rangos empíricos: "estadisticas_globales_{tejido}_{tecnica}.txt"
                txt_name = f"estadisticas_globales_{tejido}_{tecnica}.txt"
                txt_path = os.path.join(dir_estadisticas_globales, txt_name)
                if not os.path.exists(txt_path):
                    print(f"No existe {txt_path}, se omite.")
                    continue

                # Leer rangos
                (r1_lower, r1_upper, r2_lower, r2_upper) = read_ranges_from_txt(txt_path)
                print(f"\nProcesando {tejido}/{tecnica}")
                print("Rango1:", r1_lower, "-", r1_upper)
                print("Rango2:", r2_lower, "-", r2_upper)

                # Cargar imágenes de la subcarpeta
                imgs_with_names = load_images_from_folder(dir_subcarpeta)

                # Emparejar con máscaras
                images = []
                masks = []
                for (img_filename, img) in imgs_with_names:
                    mask = load_mask_by_name(dir_mascaras, img_filename)
                    if mask is not None:
                        images.append(img)
                        masks.append(mask)
                    else:
                        print(f"No se encontró máscara para {img_filename}; se omite.")

                if len(images) == 0:
                    print("No hay imágenes con máscaras en esta subcarpeta; se omite GA.")
                    continue

                # Ejecutar el GA (obtenemos el mejor individuo y los Jaccard inicial/final)
                best_ind, jaccard_init, jaccard_final = run_ga_with_ranges(
                    r1_lower, r1_upper, r2_lower, r2_upper,
                    images, masks,
                    pop_size=200, ngen=50, cxpb=0.5, mutpb=0.2
                )

                print(f"Jaccard Inicial: {jaccard_init:.4f}  |  Jaccard Final: {jaccard_final:.4f}")
                print(f"BestIndividual: {best_ind}")

                # Guardar en CSV
                f_out.write(f"{tejido},{tecnica},{jaccard_init:.6f},{jaccard_final:.6f},{best_ind}\n")

    print("\nFinalizado. Revisa el CSV en:", csv_path)


if __name__ == "__main__":
    main()

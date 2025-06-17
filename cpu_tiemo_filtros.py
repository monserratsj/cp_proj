import os, time, cv2
import numpy as np
from glob import glob
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from utils import download_dataset, calcular_textura


tiempos_acumulados = {
    "bicubico": 0.0,
    "sharpen": 0.0,
    "ecualizar": 0.0,
    "count": 0
}

# Ecualización de la imagen 
def ecualizar_rgb(img_rgb):
    canales = cv2.split(img_rgb)
    canales_eq = [cv2.equalizeHist(c) for c in canales]
    return cv2.merge(canales_eq)

# bicubico
def bicubico_ups(img_np, scale=2.0):  
    zoom_factors = (scale, scale, 1)
    img_zoom = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return np.clip(img_zoom, 0, 255).astype(np.uint8)

# sharpening
def definir_img_cpu(img_np):
    kernel_sharpen = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])  # kernel
    img_sharpened = np.zeros_like(img_np)
    
    for c in range(3):
        img_sharpened[:, :, c] = cv2.filter2D(img_np[:, :, c], -1, kernel_sharpen)
    
    img_sharpened = np.clip(img_sharpened, 0, 255)
    
    return img_sharpened

# preproceso con medición de tiempos y retorno de tiempos individuales
def procesar_cpu(path):
    tiempos_imagen = {}
    
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None  # o manejar error

    if len(img.shape) < 3 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    start = time.time()
    img_calidad_up = bicubico_ups(img)
    tiempos_imagen["bicubico"] = time.time() - start

    start = time.time()
    img_sharp_np = definir_img_cpu(img_calidad_up)
    tiempos_imagen["sharpen"] = time.time() - start

    start = time.time()
    img_eq = ecualizar_rgb(img_sharp_np)
    tiempos_imagen["ecualizar"] = time.time() - start

    nombre_carpeta = os.path.basename(os.path.dirname(path))
    output_dir = os.path.join("cpu_output", nombre_carpeta)
    os.makedirs(output_dir, exist_ok=True)

    nombre = os.path.basename(path)
    cv2.imwrite(os.path.join(output_dir, nombre), img_eq)

    img_gray = cv2.cvtColor(img_eq, cv2.COLOR_BGR2GRAY)
    textura = calcular_textura(img_gray)

    # Devolvemos también la textura y los tiempos de cada filtro
    return textura, tiempos_imagen

def version_secuencial(paths):
    texturas = []
    tiempos_acum = {
        "bicubico": 0.0,
        "sharpen": 0.0,
        "ecualizar": 0.0,
        "count": 0
    }
    for p in paths:
        resultado = procesar_cpu(p)
        if resultado is None:
            continue
        textura, tiempos = resultado
        texturas.append(textura)
        tiempos_acum["bicubico"] += tiempos["bicubico"]
        tiempos_acum["sharpen"] += tiempos["sharpen"]
        tiempos_acum["ecualizar"] += tiempos["ecualizar"]
        tiempos_acum["count"] += 1
    return texturas, tiempos_acum

def version_paralela(paths, num_procesos):
    texturas = []
    tiempos_acum = {
        "bicubico": 0.0,
        "sharpen": 0.0,
        "ecualizar": 0.0,
        "count": 0
    }
    with ProcessPoolExecutor(max_workers=num_procesos) as executor:
        resultados = list(executor.map(procesar_cpu, paths))

    for res in resultados:
        if res is None:
            continue
        textura, tiempos = res
        texturas.append(textura)
        tiempos_acum["bicubico"] += tiempos["bicubico"]
        tiempos_acum["sharpen"] += tiempos["sharpen"]
        tiempos_acum["ecualizar"] += tiempos["ecualizar"]
        tiempos_acum["count"] += 1

    return texturas, tiempos_acum

if _name_ == "_main_":
    dataset_path = "triplicado"
    os.makedirs("cpu_output", exist_ok=True)
    paths = glob(os.path.join(dataset_path, "", "*.jpg"), recursive=True)

    # Secuencial
    start = time.time()
    texturas_secuencial, tiempos_secuencial = version_secuencial(paths)
    end = time.time()
    tiempo_secuencial_total = end - start

    print(f"Tiempo Secuencial Total: {tiempo_secuencial_total:.2f} s")
    if tiempos_secuencial["count"] > 0:
        print("Tiempos medios por filtro secuencial:")
        for filtro in ["bicubico", "sharpen", "ecualizar"]:
            print(f"{filtro}: {tiempos_secuencial[filtro] / tiempos_secuencial['count']:.4f} s")
        tiempo_medio_imagen = tiempo_secuencial_total / tiempos_secuencial["count"]
        print(f"Tiempo medio de procesado por imagen (secuencial): {tiempo_medio_imagen:.4f} s")

    nucleos_lista = [2, 4, 6, 8]
    tiempos_paralelos = []
    speedups = []
    eficiencias = []

    for num_procesos in nucleos_lista:
        start = time.time()
        texturas_paralela, tiempos_paralela = version_paralela(paths, num_procesos)
        end = time.time()
        tiempo_paralelo_total = end - start
        tiempos_paralelos.append(tiempo_paralelo_total)

        # Speed-up y eficiencia
        speedup = tiempo_secuencial_total / tiempo_paralelo_total
        eficiencia = speedup / num_procesos

        speedups.append(speedup)
        eficiencias.append(eficiencia)

        print(f"\nParalela ({num_procesos} núcleos): {tiempo_paralelo_total:.2f} s")
        if tiempos_paralela["count"] > 0:
            print("Tiempos medios por filtro paralelos:")
            for filtro in ["bicubico", "sharpen", "ecualizar"]:
                print(f"{filtro}: {tiempos_paralela[filtro] / tiempos_paralela['count']:.4f} s")
            tiempo_medio_imagen_paralelo = tiempo_paralelo_total / tiempos_paralela["count"]
            print(f"Tiempo medio de procesado por imagen (paralelo): {tiempo_medio_imagen_paralelo:.4f} s")

        print(f"Speed-up: {speedup:.2f}")
        print(f"Eficiencia: {eficiencia:.2f}")

    # Gráficas
    plt.subplot(1, 2, 1)
    plt.plot(nucleos_lista, speedups, marker='o', color='b', label="Speed-up")
    plt.xlabel('Número de núcleos')
    plt.ylabel('Speed-up')
    plt.title('Speed-up - Número de núcleos')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(nucleos_lista, eficiencias, marker='o', color='r', label="Eficiencia")
    plt.xlabel('Número de núcleos')
    plt.ylabel('Eficiencia')
    plt.title('Eficiencia - Número de núcleos')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
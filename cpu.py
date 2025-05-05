import os, time, cv2
import numpy as np
from glob import glob
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from utils import download_dataset, calcular_textura

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
        # convolucion en canales
        img_sharpened[:, :, c] = cv2.filter2D(img_np[:, :, c], -1, kernel_sharpen)
    
    # overfloating
    img_sharpened = np.clip(img_sharpened, 0, 255)
    
    return img_sharpened

# preproceso
def procesar_cpu(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    # grises a color
    if len(img.shape) < 3 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 1. resolucion
    img_calidad_up = bicubico_ups(img)

    # 2. sharp
    img_sharp_np = definir_img_cpu(img_calidad_up)

    # 3. ecualizacion
    img_eq = ecualizar_rgb(img_sharp_np)

    # carpeta destino
    nombre_carpeta = os.path.basename(os.path.dirname(path))
    
    # carpeta nombre
    output_dir = os.path.join("cpu_output", nombre_carpeta)
    os.makedirs(output_dir, exist_ok=True)

    # output
    nombre = os.path.basename(path)
    cv2.imwrite(os.path.join(output_dir, nombre), img_eq)

    # textura
    img_gray = cv2.cvtColor(img_eq, cv2.COLOR_BGR2GRAY)
    return calcular_textura(img_gray)

def version_secuencial(paths):
    texturas = [procesar_cpu(p) for p in paths]
    return texturas

def version_paralela(paths, num_procesos):
    with ProcessPoolExecutor(max_workers=num_procesos) as executor:
        texturas_paralelas = list(executor.map(procesar_cpu, paths))
    return texturas_paralelas

if __name__ == "__main__":
    dataset_path = "triplicado"  # dataset triplicado
    os.makedirs("cpu_output", exist_ok=True)
    paths = glob(os.path.join(dataset_path, "**", "*.jpg"), recursive=True)

#secuencial
    start = time.time()
    texturas_secuencial = version_secuencial(paths)
    end = time.time()
    tiempo_secuencial = end - start

    print(f"Tiempo Secuencial: {tiempo_secuencial:.2f}")


    nucleos_lista = [2, 4, 6, 8]  
    tiempos_paralelos = []
    speedups = []
    eficiencias = []
    print(f"Tiempo Secuencial: {tiempo_secuencial:.2f}")

    for num_procesos in nucleos_lista:
        start = time.time()
        texturas_paralela = version_paralela(paths, num_procesos)
        end = time.time()
        tiempo_paralelo = end - start
        tiempos_paralelos.append(tiempo_paralelo)

        # speed-up y eficiencia
        speedup = tiempo_secuencial / tiempo_paralelo
        eficiencia = speedup / num_procesos

        speedups.append(speedup)
        eficiencias.append(eficiencia)

        # resultados
        print(f"Paralela ({num_procesos} núcleos): {tiempo_paralelo:.2f} s")
        print(f"Speed-up: {speedup:.2f}")
        print(f"Eficiencia: {eficiencia:.2f}")

    # Gráfica de Speed-up
    plt.subplot(1, 2, 1)
    plt.plot(nucleos_lista, speedups, marker='o', color='b', label="Speed-up")
    plt.xlabel('Número de núcleos')
    plt.ylabel('Speed-up')
    plt.title('Speed-up - Número de núcleos')
    plt.grid(True)
    plt.legend()

    # Gráfica de Eficiencia
    plt.subplot(1, 2, 2)
    plt.plot(nucleos_lista, eficiencias, marker='o', color='r', label="Eficiencia")
    plt.xlabel('Número de núcleos')
    plt.ylabel('Eficiencia')
    plt.title('Eficiencia - Número de núcleos')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

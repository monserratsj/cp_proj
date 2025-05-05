from multiprocessing import Pool, cpu_count
import os, time, cv2
import cupy as cp
from glob import glob
from cupyx.scipy.ndimage import zoom, convolve
from utils import  calcular_textura

def ecualizar_rgb(img_rgb):
    canales = cv2.split(img_rgb)
    canales_eq = [cv2.equalizeHist(c) for c in canales]
    return cv2.merge(canales_eq)

def upscale_gpu(img_cp, scale=2.0):
    img_cp = cp.asarray(img_cp)
    zoom_factors = (scale, scale, 1)
    img_zoom = zoom(img_cp, zoom_factors, order=3)
    return cp.clip(img_zoom, 0, 255).astype(cp.uint8)

def sharpen_image_gpu(img_cp):
    kernel_sharpen = cp.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
    img_sharpened = cp.zeros_like(img_cp)
    for c in range(3):
        img_sharpened[:, :, c] = convolve(img_cp[:, :, c], kernel_sharpen, mode='reflect')
    return cp.clip(img_sharpened, 0, 255)

def procesar_gpu(path):
    try:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if len(img.shape) < 3 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img_upscaled_cp = upscale_gpu(img)
        img_sharp_cp = sharpen_image_gpu(img_upscaled_cp)
        img_sharpened = cp.asnumpy(img_sharp_cp)
        img_eq = ecualizar_rgb(img_sharpened)

        nombre = os.path.basename(path)
        output_path = os.path.join("gpu_output", nombre)
        cv2.imwrite(output_path, img_eq)

        img_gray = cv2.cvtColor(img_eq, cv2.COLOR_BGR2GRAY)
        return calcular_textura(img_gray)

    except Exception as e:
        print(f"Error procesando {path}: {e}")
        return None

if __name__ == "__main__":
    dataset_path = "triplicado"
    os.makedirs("gpu_output", exist_ok=True)
    paths = glob(os.path.join(dataset_path, "**", "*.jpg"), recursive=True)

    num_procesos = min(4, cpu_count())  # 4 procesos

    start = time.time()
    with Pool(processes=num_procesos) as pool:
        texturas = pool.map(procesar_gpu, paths)
    end = time.time()

    print(f"⚡ GPU (paralelizado): {end - start:.2f} s para {len(paths)} imágenes")

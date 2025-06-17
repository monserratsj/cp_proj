import cv2
import numpy as np
import os
from glob import glob
import time

def preprocess_image_cpu(img, zoom_factor=1.5):
    if zoom_factor <= 1.0:
        return img

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    new_size = (int(l.shape[1] * zoom_factor), int(l.shape[0] * zoom_factor))


    l_resized = cv2.resize(l, new_size, interpolation=cv2.INTER_CUBIC)
    a_resized = cv2.resize(a, new_size, interpolation=cv2.INTER_CUBIC)
    b_resized = cv2.resize(b, new_size, interpolation=cv2.INTER_CUBIC)

    lab_resized = cv2.merge((l_resized, a_resized, b_resized))
    return cv2.cvtColor(lab_resized, cv2.COLOR_LAB2BGR)

def restore_image_cpu(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    timings = {}

    start = time.perf_counter()

    l_bilateral = cv2.bilateralFilter(l, d=9, sigmaColor=75, sigmaSpace=75)
    timings['bilateral_filter'] = time.perf_counter() - start

    start = time.perf_counter()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l_bilateral)
    timings['clahe'] = time.perf_counter() - start

    start = time.perf_counter()
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    l_sharp = cv2.filter2D(l_clahe, -1, kernel)
    timings['sharpen'] = time.perf_counter() - start

    start = time.perf_counter()
 
    a_blur = cv2.GaussianBlur(a, (3,3), 0)
    b_blur = cv2.GaussianBlur(b, (3,3), 0)
    timings['color_blur'] = time.perf_counter() - start

    start = time.perf_counter()

    lab_restored = cv2.merge((l_sharp, a_blur, b_blur))
    img_restored = cv2.cvtColor(lab_restored, cv2.COLOR_LAB2BGR)
    timings['reconstruction'] = time.perf_counter() - start

    return img_restored, timings

def process_folder_cpu(input_dir, output_dir, zoom_factor=1.5):
    os.makedirs(output_dir, exist_ok=True)
    image_files = glob(os.path.join(input_dir, "*.[pj][pn]g"))
    
    total_timings = {
        'bilateral_filter': 0,
        'clahe': 0,
        'sharpen': 0,
        'color_blur': 0,
        'reconstruction': 0,
        'count': 0
    }

    for img_path in image_files:
        try:
            print(f"Procesando {os.path.basename(img_path)}...")
            img = cv2.imread(img_path)
            if img is None:
                print(f"  No se pudo leer {img_path}")
                continue
            

            img_zoomed = preprocess_image_cpu(img, zoom_factor)

    
            restored, timings = restore_image_cpu(img_zoomed)

            for key in timings:
                total_timings[key] += timings[key]
            total_timings['count'] += 1

            cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), restored)
 
            print(f"  Tiempos (s): {', '.join([f'{k}: {v:.4f}' for k,v in timings.items()])}")

        except Exception as e:
            print(f"Error procesando {img_path}: {e}")

    if total_timings['count'] > 0:
        print("\nTiempos promedio por filtro:")
        for key in total_timings:
            if key != 'count':
                avg_time = total_timings[key] / total_timings['count']
                print(f"  {key}: {avg_time:.4f} s")

if __name__ == "__main__":
    INPUT_DIR = "triplicado"       
    OUTPUT_DIR = "restored_cpu"    
    ZOOM_FACTOR = 1.5              

    start_total = time.perf_counter()
    process_folder_cpu(INPUT_DIR, OUTPUT_DIR, ZOOM_FACTOR)
    print(f"\nProcesamiento total finalizó en {time.perf_counter() - start_total:.2f} segundos")

import cv2
import numpy as np
from numba import cuda
import os
from glob import glob
import time
import math

# ------------------------- GPU FILTROS -------------------------

@cuda.jit(device=True)
def cuda_exp(x):
    return math.exp(x)

@cuda.jit
def smooth_segmentation_kernel(input, output, sigma_space, sigma_color):
    x, y = cuda.grid(2)
    if x < 2 or y < 2 or x >= input.shape[0]-2 or y >= input.shape[1]-2:
        return
    
    center = input[x,y]
    total_weight = 0.0
    sum_val = 0.0
    
    for i in range(-2, 3):
        for j in range(-2, 3):
            neighbor = input[x+i,y+j]
            space_dist = float(i**2 + j**2)
            color_dist = abs(center - neighbor)
            
            weight = cuda_exp(-space_dist/(2*sigma_space**2) - color_dist/(2*sigma_color**2))
            sum_val += neighbor * weight
            total_weight += weight
    
    output[x,y] = sum_val / total_weight if total_weight > 0 else center

@cuda.jit
def soft_equalization_kernel(input, output):
    x, y = cuda.grid(2)
    if x >= input.shape[0] or y >= input.shape[1]:
        return
    
    val = input[x,y]
    if val < 0.3:
        output[x,y] = val * 0.8
    elif val > 0.7:
        output[x,y] = 0.7 + (val-0.7)*0.6
    else:
        output[x,y] = 0.3 + (val-0.3)*1.1

@cuda.jit
def cuda_sharpen_kernel(input, output, strength):
    x, y = cuda.grid(2)

    height, width = input.shape

    if x < 2 or y < 2 or x >= height - 2 or y >= width - 2:
        return

    # Laplacian extendido 
    laplacian = (
        -0.5 * input[x-2, y] +
        -1.0 * input[x-1, y] +
        -1.0 * input[x+1, y] +
        -0.5 * input[x+2, y] +
        -0.5 * input[x, y-2] +
        -1.0 * input[x, y-1] +
        -1.0 * input[x, y+1] +
        -0.5 * input[x, y+2] +
         6.0 * input[x, y]
    ) / 6.0

    sharpened = input[x, y] + strength * laplacian
    output[x, y] = min(max(sharpened, 0.0), 1.0)
@cuda.jit
def cuda_zoom_kernel(input, output, zoom_factor):
    x, y = cuda.grid(2)
    if x >= output.shape[0] or y >= output.shape[1]:
        return
    
    src_x = x / zoom_factor
    src_y = y / zoom_factor
    
    x1 = int(math.floor(src_x))
    y1 = int(math.floor(src_y))
    
    if x1 < 1 or y1 < 1 or x1 >= input.shape[0]-2 or y1 >= input.shape[1]-2:
        output[x,y] = 0
        return
    
    dx = src_x - x1
    dy = src_y - y1
    
    pixels = cuda.local.array((4,4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            pixels[i,j] = input[x1-1+i, y1-1+j]
    
    output[x,y] = (
        pixels[0,0]*(1-dx)**3*(1-dy)**3 +
        pixels[0,1]*3*(1-dx)**3*dy*(1-dy)**2 +
        pixels[0,2]*3*(1-dx)**3*dy**2*(1-dy) +
        pixels[0,3]*(1-dx)**3*dy**3 +
        pixels[1,0]*3*dx*(1-dx)**2*(1-dy)**3 +
        pixels[1,1]*9*dx*(1-dx)**2*dy*(1-dy)**2 +
        pixels[1,2]*9*dx*(1-dx)**2*dy**2*(1-dy) +
        pixels[1,3]*3*dx*(1-dx)**2*dy**3 +
        pixels[2,0]*3*dx**2*(1-dx)*(1-dy)**3 +
        pixels[2,1]*9*dx**2*(1-dx)*dy*(1-dy)**2 +
        pixels[2,2]*9*dx**2*(1-dx)*dy**2*(1-dy) +
        pixels[2,3]*3*dx**2*(1-dx)*dy**3 +
        pixels[3,0]*dx**3*(1-dy)**3 +
        pixels[3,1]*3*dx**3*dy*(1-dy)**2 +
        pixels[3,2]*3*dx**3*dy**2*(1-dy) +
        pixels[3,3]*dx**3*dy**3
    )

# ------------------------- FLUJO -------------------------

def preprocess_image(img, zoom_factor=1.5):
    if zoom_factor <= 1.0:
        return img
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    l_float = l.astype(np.float32)/255.0
    new_height = int(l_float.shape[0] * zoom_factor)
    new_width = int(l_float.shape[1] * zoom_factor)
    
    d_input = cuda.to_device(l_float)
    d_zoomed = cuda.device_array((new_height, new_width), dtype=np.float32)
    
    threads = (16, 16)
    grid = (
        (new_height + threads[0] - 1) // threads[0],
        (new_width + threads[1] - 1) // threads[1]
    )
    
    cuda_zoom_kernel[grid, threads](d_input, d_zoomed, zoom_factor)
    cuda.synchronize()
    
    zoomed_l = (d_zoomed.copy_to_host() * 255).clip(0, 255).astype(np.uint8)
    a = cv2.resize(a, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    b = cv2.resize(b, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    zoomed_lab = cv2.merge((zoomed_l, a, b))
    return cv2.cvtColor(zoomed_lab, cv2.COLOR_LAB2BGR)

def restore_image(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # L
    l_float = l.astype(np.float32)/255.0
    
    # memoria GPU
    d_input = cuda.to_device(l_float)
    d_temp1 = cuda.device_array_like(l_float)
    d_temp2 = cuda.device_array_like(l_float)
    d_temp3 = cuda.device_array_like(l_float)
    d_processed = cuda.device_array_like(l_float)
    
    threads = (16, 16)
    grid = (
        (l_float.shape[0] + threads[0] - 1) // threads[0],
        (l_float.shape[1] + threads[1] - 1) // threads[1]
    )
    
    # segmentation soft
    smooth_segmentation_kernel[grid, threads](d_input, d_temp1, 3.0, 0.1)
    cuda.synchronize()
    
    # equalizacion 
    soft_equalization_kernel[grid, threads](d_temp1, d_temp2)
    cuda.synchronize()
    
    # sharpening
    cuda_sharpen_kernel[grid, threads](d_temp2, d_temp3, 1.5)
    cuda.synchronize()
    
    # sharpening final
    cuda_sharpen_kernel[grid, threads](d_temp3, d_processed, 1.0)
    cuda.synchronize()
    
    processed_l = (d_processed.copy_to_host() * 255).clip(0, 255).astype(np.uint8)
    
    # ab color blur
    a = cv2.GaussianBlur(a, (3, 3), 0)
    b = cv2.GaussianBlur(b, (3, 3), 0)
    
    # clahe
    processed_lab = cv2.merge((processed_l, a, b))
    color_processed = cv2.cvtColor(processed_lab, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(color_processed, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    return gray

def restore_and_zoom(img, zoom_factor=1.5):
    preprocessed = preprocess_image(img, zoom_factor)
    return restore_image(preprocessed)

# ------------------------- procesamiento -------------------------

def process_folder(input_dir, output_dir, zoom_factor=1.0):
    os.makedirs(output_dir, exist_ok=True)
    image_files = glob(os.path.join(input_dir, "*.[pj][np]g"))
    
    for img_path in image_files:
        try:
            print(f"Processing {os.path.basename(img_path)}...")
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            restored = restore_and_zoom(img, zoom_factor)
            cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), restored)
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

if __name__ == "__main__":
    INPUT_DIR = "triplicado"
    OUTPUT_DIR = "restored_output"
    
    start_time = time.time()
    process_folder(INPUT_DIR, OUTPUT_DIR)
    print(f"Finished in {time.time()-start_time:.2f} seconds")
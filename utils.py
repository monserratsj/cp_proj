import kagglehub
import mahotas
import os
import shutil
from pathlib import Path
from kagglehub import dataset_download


def download_dataset(dest_folder="triplicado"):

    original_path = dataset_download("marcinrutecki/old-photos")
    print(f"📂 Dataset descargado en: {original_path}")


    dest_path = os.path.abspath(dest_folder)
    os.makedirs(dest_path, exist_ok=True)


    for item in os.listdir(original_path):
        s = os.path.join(original_path, item)
        d = os.path.join(dest_path, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

    print(f" Dataset movido a: {dest_path}")
    return dest_path

def triplicar_imagenes(dataset_path):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
    
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                original_path = os.path.join(root, file)
                name, ext = os.path.splitext(file)
                
                # Crear 100 copias adicionales
                for i in range(100):
                    new_filename = f"{name}_copy{i+1}{ext}"
                    new_path = os.path.join(root, new_filename)
                    shutil.copy2(original_path, new_path)
                    print(f"📸 Copiada: {new_path}")

if __name__ == "__main__":
    dataset_dir = download_dataset("triplicado")
    triplicar_imagenes(dataset_dir)

def calcular_textura(img):
    # características de Haralick
    return mahotas.features.haralick(img).mean(axis=0)


if __name__ == "__main__":
    dataset_dir = download_dataset()
    triplicar_imagenes(dataset_dir)
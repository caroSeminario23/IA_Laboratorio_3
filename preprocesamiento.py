import cv2
import os

def preprocess_images(input_dir, output_dir, img_size=(224, 224)):
    # Verifica que el directorio de salida exista, si no, créalo
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for subdir in os.listdir(input_dir):
        subpath = os.path.join(input_dir, subdir)
        output_subpath = os.path.join(output_dir, subdir)
        if not os.path.exists(output_subpath):
            os.makedirs(output_subpath)
        
        if os.path.isdir(subpath):
            for filename in os.listdir(subpath):
                img_path = os.path.join(subpath, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    # Redimensiona la imagen
                    img_resized = cv2.resize(img, img_size)
                    # Guarda la imagen procesada
                    output_path = os.path.join(output_subpath, filename)
                    cv2.imwrite(output_path, img_resized)
                    print(f"Processed {output_path}")

input_dir = 'bts_images'  # Directorio de imágenes originales
output_dir = 'bts_images_procesadas'  # Directorio de imágenes procesadas

preprocess_images(input_dir, output_dir)

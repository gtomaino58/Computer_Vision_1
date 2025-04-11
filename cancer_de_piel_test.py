# Vamos a preparar el entorno para el proyecto

# Visualizamos varios resultados en cada celta
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Importamos las librerias necesarias
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Vamos a cargar el archivo de datos de las imagenes 
#C:\Users\gtoma\Downloads\VisArt\skin_cancer_dataset\train-20240324T151905Z-001\train\benign
#C:\Users\gtoma\Downloads\VisArt\skin_cancer_dataset\train-20240324T151905Z-001\train\malignant
# Vamos a cargar el archivo de datos de las imagenes
# Cargamos las librerias necesarias

# Definimos la ruta de los datos
data_dir = r'C:\Users\gtoma\Downloads\VisArt\skin_cancer_dataset\train-20240324T151905Z-001\train'

# Definimos la ruta de los datos de las imagenes benignas y malignas
benign_dir = os.path.join(data_dir, 'benign')
malignant_dir = os.path.join(data_dir, 'malignant')

# Definimos la ruta de los datos de las imagenes benignas y malignas
benign_images = os.listdir(benign_dir)
malignant_images = os.listdir(malignant_dir)

# Vamos a cargar las imagenes benignas y malignas
benign_images = [os.path.join(benign_dir, img) for img in benign_images]
malignant_images = [os.path.join(malignant_dir, img) for img in malignant_images]
benign_images
malignant_images

# Vamos a convertir las imagenes a un formato que podamos usar
# Vamos a cargar las imagenes benignas y malignas
from PIL import Image

# Definimos la funcion para cargar las imagenes
def load_images(image_paths):
    images = []
    for path in image_paths:
        img = Image.open(path)
        img = img.resize((224, 224))  # Redimensionamos la imagen a 224x224
        images.append(np.array(img))
    return np.array(images)

# Cargamos las imagenes benignas y malignas

benign_images = load_images(benign_images)
malignant_images = load_images(malignant_images)
# Vamos a ver las imagenes benignas y malignas
benign_images.shape, malignant_images.shape

# Vamos a ver las imagenes de train
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title('Benign Image')
plt.imshow(benign_images[0])
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title('Malignant Image')
plt.imshow(malignant_images[0])
plt.axis('off')
plt.show();
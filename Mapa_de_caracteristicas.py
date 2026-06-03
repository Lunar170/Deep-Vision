# -*- coding: utf-8 -*-
"""
Proyecto Machine Learning 
Visualización de Mapas de Características (Feature Maps)
Abril Guadalupe
Rafael Lunar 
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Cargar el modelo completo preentrenado desde tu ruta local
ruta_modelo = r"D:\Deep Vision\SteelSense_Model_6Clasesprueba.keras"
print("Cargando el cerebro de Deep Vision AI...")
model = tf.keras.models.load_model(ruta_modelo)
print("Modelo cargado exitosamente.")

# Extraer el modelo base (MobileNetV2) que está dentro de la  arquitectura
# Buscamos la capa por su nombre o tipo para extraer sus subcapas
base_model = None
for layer in model.layers:
    if 'mobilenetv2' in layer.name.lower():
        base_model = layer
        break
#opción por si no encuentra 
if base_model is None:
    raise ValueError("No se encontró el bloque base de MobileNetV2 en el modelo.")

#  Seleccionar las capas convolucionales que queremos inspeccionar
# Vamos a elegir una capa inicial (identifica bordes), una intermedia (texturas) y una profunda (patrones abstractos)
nombres_capas = [
    'block_1_expand',   # Capa inicial (baja abstracción)
    'block_7_expand',   # Capa intermedia (media abstracción)
    'block_14_expand'   # Capa profunda (alta abstracción)
]

# Extraemos las salidas de esas capas específicas
salidas_capas = [base_model.get_layer(name).output for name in nombres_capas]

# Crear el modelo de activación intermedio
# Saca las características de las 3 capas elegidas
activation_model = tf.keras.Model(inputs=base_model.input, outputs=salidas_capas)

def visualizar_mapas_caracteristicas(ruta_imagen):
    """
    Procesa una imagen y despliega los mapas de características 
    de los filtros convolucionales en diferentes niveles de la red.
    """
    try:
        # Cargar y adaptar la imagen al formato requerido por la red (224x224)
        img = image.load_img(ruta_imagen, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión de lote (1, 224, 224, 3)
        
        # El modelo de reescalado del modelo principal escala de [0,255] a [-1,1]
        # Hacemos el reescalado matemático manual idéntico para la activación
        img_array = (img_array / 127.5) - 1.0
        
        # Obtener las activaciones (mapas de características) pasando la imagen por el modelo
        activaciones = activation_model.predict(img_array, verbose=0)
        
        # Graficar los mapas de características capa por capa
        for i, mapa in enumerate(activaciones):
            nombre_capa = nombres_capas[i]
            
            # El mapa tiene la forma: (1, alto, ancho, número_de_filtros)
            # Extraemos el número total de filtros disponibles en esa capa
            num_filtros = mapa.shape[-1]
            
            # Vamos a visualizar una cuadrícula de los primeros 8 filtros de cada capa
            filtros_a_mostrar = min(8, num_filtros)
            
            fig, ejes = plt.subplots(1, filtros_a_mostrar, figsize=(15, 3))
            fig.suptitle(f"Mapas de Características - Capa: {nombre_capa} ({num_filtros} filtros totales)", 
                         fontsize=12, fontweight='bold', color='#2c3e50')
            
            for j in range(filtros_a_mostrar):
                # Extraer la matriz del filtro 'j'
                camara_filtro = mapa[0, :, :, j]
                
                # Desplegar el mapa en escala de grises o formato 'viridis' para ver la intensidad de activación
                ejes[j].imshow(camara_filtro, cmap='viridis')
                ejes[j].axis('off')
                ejes[j].set_title(f"Filtro {j+1}", fontsize=9)
                
            plt.show()
            
    except Exception as e:
        print(f" Error al procesar los mapas de características: {e}")

# Prueba con cualquiera de tus imágenes del disco D
visualizar_mapas_caracteristicas(r"D:\Deep Vision\crazing_254.jpg")


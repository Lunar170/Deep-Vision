# -*- coding: utf-8 -*-
"""
Proyecto Machine Learning 
Deep vision 
Abril Guadalupe
Rafael Lunar 
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# 1. Definir las etiquetas en el orden alfabético estricto
class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

# 2. Cargar el modelo guardado con extensión .keras)
ruta_modelo = r"D:\Deep Vision\SteelSense_Model_6Clasesprueba.keras"
print("Cargando el cerebro de SteelSense AI...")
model = tf.keras.models.load_model(ruta_modelo)
print(" Modelo cargado exitosamente en la computadora.")

def probar_imagen_local(nombre_imagen):
    """
    Carga una imagen de la carpeta local de forma dinámica, 
    realiza la inferencia y despliega la alerta visual.
    """
    try:
    
        img_original = image.load_img(nombre_imagen)
        img_rescaled = image.load_img(nombre_imagen, target_size=(224, 224))
        
        # Convertir a matriz estándar de píxeles [0, 255]
        img_array = image.img_to_array(img_rescaled)
        img_array = np.expand_dims(img_array, axis=0)  # Crear dimensión de Batch
        
        # Inferencia directa con el modelo local
        predicciones = model.predict(img_array, verbose=0)
        indice_predicho = np.argmax(predicciones[0])
        confianza = predicciones[0][indice_predicho] * 100
        clase_resultado = class_names[indice_predicho]
        
        # Interfaz gráfica de la alerta industrial
        plt.figure(figsize=(6, 6))
        plt.imshow(img_original)
        
        color_alerta = '#e67e22'  # Naranja industrial
        mensaje = f" DEFECTO: {clase_resultado.upper()}\nConfianza: {confianza:.2f}%"
            
        plt.title(mensaje, color=color_alerta, fontsize=13, fontweight='bold', pad=15)
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f" Error al procesar la imagen: {e}")
        print("Verifica que el nombre del archivo sea exacto, incluya su extensión (.jpg) y esté en la misma carpeta.")

#"D:\Deep Vision\crazing_254.jpg"
#"D:\Deep Vision\inclusion_250.jpg"
#"D:\Deep Vision\rolled-in_scale_245.jpg"
#"D:\Deep Vision\scratches_264.jpg"
#
probar_imagen_local(r"D:\Deep Vision\crazing_254.jpg")
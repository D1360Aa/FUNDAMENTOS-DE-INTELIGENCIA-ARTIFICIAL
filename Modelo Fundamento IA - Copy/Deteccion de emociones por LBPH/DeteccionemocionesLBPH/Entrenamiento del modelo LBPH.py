import cv2
import os
import numpy as np

rutaDataset = 'C:\proyectos\Modelo Fundamento IA\Deteccion de emociones por LBPH\DeteccionemocionesLBPH\Dataset' # Se define la ruta donde se encuentra la carpeta para nuestro dataset
listaEmociones = os.listdir(rutaDataset) # Se realiza una lista con las carpetas dentro de nuestro dataset, estas carpetas corresponden a las emociones
print('Lista de emociones: ', listaEmociones) # Se imprime la lista de las carpetas guardadas en la variable emotionsList
labels = [] # Se define un arreglo de etiquetas para identificar cada una de las emociones con valores de 0, 1 y 2
facesData = [] # Se define un arreglo en el que se almacenaran cada una de las imagenes pertenecientes a las emociones
label = 0 # Variable tipo contador para asignar un numero a cada una de las emociones 
for emociones in listaEmociones: # Se recorre cada carpeta de las emociones
    rutaEmociones = rutaDataset + '/' + emociones # Se define la ruta para la instancia actual en el for
    for rostros in os.listdir(rutaEmociones): # Se recorre cada una de las imagenes de rostros en la carpeta actual del for
        labels.append(label) # Se le asigna al array labels el valor actual de contador, esto asigna a las imagenes que se recorren dicha etiqueta
        facesData.append(cv2.imread(rutaEmociones+'/'+rostros,0)) # Se guardan cada una de las imagenes en el array de facesData en escala de grises (0)
    label = label + 1 # Se aumenta el contador

emotion_recognizer = cv2.face.LBPHFaceRecognizer_create() # Se crea el reconocedor de emociones usando un modelo LBPH
print("Entrenando LBPH...") # Se imprime el mensaje de que se entrenara el modelo
emotion_recognizer.train(facesData, np.array(labels)) # Se entrena el modelo con las imagenes guardades en el array facesData
# El array labels le indica al modelo que etiqueta corresponde a cada imagen
rutadeguardado = 'C:\proyectos\Modelo Fundamento IA\Deteccion de emociones por LBPH\DeteccionemocionesLBPH\Modelo.xml' # Se define la ruta en donde se guarda el modelo
emotion_recognizer.write(rutadeguardado) # Se guarda el modelo en la ruta definida
print("Modelo guardado") # Se imprime un mensaje para asegurar que todo funciono correctamente
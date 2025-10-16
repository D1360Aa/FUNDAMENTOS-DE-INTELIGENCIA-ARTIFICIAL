# 🤖 Reconocimiento de Emociones con OpenCV y DeepFace
**Autores:** Diego Cárdenas, Jefrey Correa, Juan Dominguez y Luis de la Cruz
**Año:** 2025  
**Desarrollo**  

---

## 📘 Introducción

El reconocimiento automático de emociones humanas es un campo de estudio que combina técnicas de **visión por computadora**, **aprendizaje automático** y **procesamiento de imágenes**.  
En este proyecto se desarrollan y comparan dos enfoques diferentes para el reconocimiento de emociones en tiempo real utilizando **OpenCV** y **DeepFace**.

### 🔹 ¿Qué es OpenCV?  
**OpenCV (Open Source Computer Vision Library)** es una biblioteca de código abierto enfocada en el procesamiento de imágenes y visión artificial.  
Permite realizar tareas como detección de rostros, reconocimiento facial, análisis de movimiento y manipulación de video en tiempo real.  
Su flexibilidad y compatibilidad con múltiples lenguajes (Python, C++, Java) la convierten en una herramienta esencial para la visión computacional.

### 🔹 ¿Qué es DeepFace?  
**DeepFace** es una biblioteca de alto nivel basada en modelos de **redes neuronales convolucionales (CNN)** preentrenadas.  
Proporciona una interfaz simple para el reconocimiento facial, detección de emociones, estimación de edad y género.  
A diferencia de OpenCV, DeepFace no requiere entrenamiento manual de modelos, ya que integra arquitecturas como **VGG-Face, FaceNet y OpenFace**, entrenadas sobre millones de rostros.

---

## ⚙️ Desarrollo del proyecto

El proyecto se estructura en cuatro fases principales:

1. **Recolección de imágenes para el dataset (OpenCV).**  
2. **Entrenamiento del modelo LBPH (OpenCV).**  
3. **Reconocimiento de emociones en tiempo real (LBPH).**  
4. **Reconocimiento de emociones con DeepFace (modelo preentrenado).**  

Cada sección incluye el código completo y una explicación detallada línea por línea.

---

## 📸 1. Recolección de imágenes para el dataset

```python
import cv2
import os

#emocion = 'Neutral'
#emocion = 'Felicidad'
emocion = 'Tristeza'

rutaDataset = 'Deteccion de emociones por LBPH\DeteccionemocionesLBPH\Dataset' #Cambia a la ruta donde hayas almacenado el dataset
rutaEmociones = rutaDataset + '/' + emocion

if not os.path.exists(rutaEmociones):
    print('Carpeta creada: ',rutaEmociones)
    os.makedirs(rutaEmociones)

# Detector de rostros de OpenCV
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Iniciar webcam
cap = cv2.VideoCapture(0) # Se prueba primero con la camara principal (0)
if not cap.isOpened():
    cap=cv2.VideoCapture(1) # Si la camara principal no enciende se prueba con la camara secundaria (1)
if not cap.isOpened():
    print("No se puede abrir la cámara") # Si ninguna enciende se envia un mensaje de error

contador = 400

while True:

    ret, frame = cap.read() # Se incluye cada fotograma del video en frame, ret es una variable booliana que indica la lectura fue exitosa
    if ret == False: # Si ret es falso, se rompe el ciclo while
        break
    # Detectar rostros con OpenCV
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Se convierte cada frame en una escala de grises, ya que el clasificador Haar solo trabaja en un canal.
    faces = faceClassif.detectMultiScale(gray, 1.3, 5) #detectMultiscale detecta las coordenadas del rostro
    # 1.3 es el factor de escala y 5 es el numero minimo de vecinos para considerar un objeto un rostro

    for (x,y,w,h) in faces: # Se dibuja un rectangulo de color azul donde se encuentre el rostro
        cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)
        rostro = gray[y:y+h, x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
        # Se redimensiona el tamaño de cada rostro capturado a un tamaño de 150x150 píxeles antes de guardarlo para facilitar el entrenamiento.
        cv2.imwrite(rutaEmociones + '/rotro_{}.jpg'.format(contador),rostro) # Se guardan las imagenes en la ruta rutaEmociones
        contador = contador + 1 # Se aumenta el contador en 1

    # Mostrar el video
    cv2.imshow('Recolección para Dataset',frame)
    # Salir con tecla 'ESC'
    k =  cv2.waitKey(1)
    if k == 27 or contador >= 600:
        break

cap.release() # Se apaga la camara
cv2.destroyAllWindows() # Se cierra la ventana de video
```

### 🔍 Explicación detallada
- **Líneas 1-2:** Se importan las bibliotecas necesarias: `cv2` para visión por computadora y `os` para manejo de archivos.  
- **Líneas 4-6:** Se define la emoción actual a capturar. Cada carpeta representa una emoción.  
- **Líneas 8-9:** Se definen las rutas del dataset y la carpeta específica de la emoción.  
- **Líneas 11-13:** Se crea la carpeta si no existe.  
- **Línea 15:** Se carga el clasificador Haar para detección de rostros.  
- **Líneas 17-22:** Se abre la cámara (0 = principal, 1 = secundaria). Si falla, imprime mensaje de error.  
- **Línea 24:** Se inicia un contador para los nombres de archivos.  
- **Bucle principal:** Captura frames, los convierte a escala de grises, detecta rostros, los recorta y guarda cada rostro como imagen 150×150 px.  
- **Línea 41:** Si se presiona `ESC` o se alcanzan 200 imágenes, se detiene la captura.  
- **Líneas 44-45:** Se liberan recursos y se cierran ventanas.

---

## 🧠 2. Entrenamiento del modelo LBPH

```python
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
```

### 🔍 Explicación
- **Líneas 1-3:** Se importan las librerías requeridas (`cv2`, `os`, `numpy`).  
- **Líneas 5-7:** Se lista el contenido del dataset, cada carpeta representa una emoción.  
- **Líneas 8-12:** Se inicializan arreglos para etiquetas (`labels`) e imágenes (`facesData`).  
- **Bucle principal:**  
  - Recorre cada carpeta de emociones.  
  - Carga las imágenes en escala de grises con `cv2.imread(...,0)`.  
  - Asigna una etiqueta numérica por emoción.  
- **Línea 19:** Se crea el reconocedor LBPH.  
- **Líneas 21-22:** Se entrena con los datos recolectados.  
- **Líneas 23-25:** Se guarda el modelo entrenado en formato XML.

---

## 🎯 3. Reconocimiento de emociones en tiempo real (LBPH)

```python
import cv2
import os
import numpy as np

emotion_recognizer = cv2.face.LBPHFaceRecognizer_create() # Se crea el reconocedor de emociones usando un modelo LBPH
emotion_recognizer.read('C:\proyectos\Modelo Fundamento IA\Deteccion de emociones por LBPH\DeteccionemocionesLBPH\Modelo.xml') # Se sube el archivo del modelo entrenado anteriormente
# --------------------------------------------------------------------------------
rutaDataset = 'Deteccion de emociones por LBPH\DeteccionemocionesLBPH\Dataset' # Se define la ruta donde se encuentra la carpeta para nuestro dataset
listaEmociones = os.listdir(rutaDataset) # Se realiza una lista con las carpetas dentro de nuestro dataset, estas carpetas corresponden a las emociones
print('Lista de emociones: ',listaEmociones) # Se imprime la lista de emociones

# Detector de rostros de OpenCV
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Iniciar webcam
cap = cv2.VideoCapture(0) # Se prueba primero con la camara principal (0)
if not cap.isOpened():
    cap=cv2.VideoCapture(1) # Si la camara principal no enciende se prueba con la camara secundaria (1)
if not cap.isOpened():
    print("No se puede abrir la cámara") # Si ninguna enciende se envia un mensaje de error

while True:
    ret, frame = cap.read() # Se incluye cada fotograma del video en frame, ret es una variable booliana que indica la lectura fue exitosa
    if ret == False: # Si ret es falso, se rompe el ciclo while
        break
    # Detectar rostros con OpenCV
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Se convierte cada frame en una escala de grises, ya que el clasificador Haar solo trabaja en un canal.
    faces = faceClassif.detectMultiScale(gray, 1.3, 5) #detectMultiscale detecta las coordenadas del rostro
    # 1.3 es el factor de escala y 5 es el numero minimo de vecinos para considerar un objeto un rostro
    for (x,y,w,h) in faces:
        rostro = gray[y:y+h, x:x+w] # Utilizando las coordenadas detectadas se extrae la parte de la imagen con el rostro
        rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC) # Se redimensiona esta imagen para que sea de 150x150 pixeles tal y como los rostros del entrenamiento
        result = emotion_recognizer.predict(rostro) # Se predice la emocion del rostro
        if result[1] < 60: # Si la confianza es menor de 60
            cv2.putText(frame,'{}'.format(listaEmociones[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA) # Se imprime sobre el video el nombre de la emoción reconocida
            cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2) # Se dibuja un rectangulo de color azul
        else:
            cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA) # Se imprime No identificado si la confianza es mayor de 60
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2) # Se dibuja el rectangulo de color rojo
    # Mostrar el video
    cv2.imshow('Reconocimiento de emociones', frame)    
    # Salir con tecla 'ESC'    
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release() # Se apaga la camara
cv2.destroyAllWindows() # Se cierra la ventana de video
```

### 🔍 Explicación
- Carga el modelo entrenado (`Modelo.xml`).  
- Usa `CascadeClassifier` para detectar rostros.  
- Convierte el frame a escala de grises y recorta el rostro detectado.  
- Predice la emoción con `emotion_recognizer.predict()`.  
- Si el valor de confianza es menor a 60, se considera una detección válida.  
- Dibuja rectángulos y muestra el nombre de la emoción sobre el rostro.  
- Finaliza al presionar la tecla `ESC`.

---

## 🧠 4. Reconocimiento de emociones en tiempo real (DeepFace)

```python
import cv2 # Open CV para el procesamiento de imagenes y videos
from deepface import DeepFace #Deepface preentrenado para reconocer emociones, edad, raza y genero

# Se definen las emociones que se quieren mostrar ya que el modelo Deepface detecta muchas mas emociones
emociones_validas = ['neutral', 'happy', 'sad']

# Detector de rostros de OpenCV
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Iniciar webcam
cap = cv2.VideoCapture(0) # Se prueba primero con la camara principal (0)
if not cap.isOpened():
    cap=cv2.VideoCapture(1) # Si la camara principal no enciende se prueba con la camara secundaria (1)
if not cap.isOpened():
    print("No se puede abrir la cámara") # Si ninguna enciende se envia un mensaje de error

while True:
    ret, frame = cap.read() # Se incluye cada fotograma del video en frame, ret es una variable booliana que indica la lectura fue exitosa
    if ret == False: # Si ret es falso, se rompe el ciclo while
        break
    emocion = None  # Variable para guardar las emociones en caso de ser validas

    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False) #Se analiza el fotograma con DeepFace
    # actions=['emotion'] indica que solo se analizan las emociones, no las otras variables incluidas en DeepFace
    # enforce_detection=False evita fallos si no se detecta una cara

    emocion_detectada = result[0]['dominant_emotion'] # Se guarda la emocion dominante en en la variable emocion detectada
    if emocion_detectada in emociones_validas: # Se realiza una comparacion, solo se guarda la emoción si está en la lista permitida
        emocion = emocion_detectada
    
    # Detectar rostros con OpenCV
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    # Se convierte cada frame en una escala de grises, ya que el clasificador Haar solo trabaja en un canal.
    faces = faceClassif.detectMultiScale(gray, 1.3, 5) #detectMultiscale detecta las coordenadas del rostro
    # 1.3 es el factor de escala y 5 es el numero minimo de vecinos para considerar un objeto un rostro 
    
    for (x, y, w, h) in faces: # Se dibuja un rectangulo de color azul donde se encuentre el rostro
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Mostrar emoción solo si es válida
    if emocion:
        font = cv2.FONT_HERSHEY_SIMPLEX # Tipo de fuente para el texto
        cv2.putText(frame, emocion, (x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA) # Se imprime el texto en el video
    # Mostrar el video
    cv2.imshow('Detector de emociones', frame)
    # Salir con tecla 'ESC'
    k =  cv2.waitKey(1)
    if k == 27:
        break

cap.release() # Se apaga la camara
cv2.destroyAllWindows() # Se cierra la ventana de video
```

### 🔍 Explicación
- Se importa **DeepFace**, que utiliza modelos preentrenados para reconocer emociones.  
- `DeepFace.analyze()` analiza el fotograma completo y devuelve la emoción dominante.  
- `enforce_detection=False` evita errores cuando no se detectan rostros.  
- Se dibuja un rectángulo sobre cada rostro y se muestra la emoción detectada.  
- Finaliza al presionar la tecla `q`.

---

## 📊 Comparación entre modelos

| Característica | LBPH | DeepFace |
|----------------|------|-----------|
| Enfoque | Clásico basado en histogramas | Red neuronal convolucional (CNN) |
| Entrenamiento propio | Sí | No |
| Velocidad | Alta (CPU) | Moderada (requiere GPU para tiempo real) |
| Precisión | Media | Alta |
| Dependencias | OpenCV | DeepFace, TensorFlow/PyTorch |
| Escalabilidad | Limitada | Alta |

---

## 🧾 Conclusiones y recomendaciones

- **LBPH** es ideal para entornos educativos y demostrativos por su rapidez y simplicidad.  
- **DeepFace** ofrece mayor precisión y robustez, aprovechando arquitecturas modernas.  
- Se recomienda incorporar **data augmentation**, normalización de iluminación y guardar un **mapeo de etiquetas** (`label_map.json`).  
- Este trabajo demuestra la aplicabilidad de la visión por computadora para la interpretación emocional en tiempo real.

---


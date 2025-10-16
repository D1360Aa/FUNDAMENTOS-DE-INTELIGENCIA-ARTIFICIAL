# 🤖 Reconocimiento de Emociones con OpenCV y DeepFace
**Autor:** Diego Cárdenas  
**Año:** 2025  
**Informe técnico–académico**  

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

rutaDataset = 'Deteccion de emociones por LBPH\DeteccionemocionesLBPH\Dataset'
rutaEmociones = rutaDataset + '/' + emocion

if not os.path.exists(rutaEmociones):
    print('Carpeta creada: ',rutaEmociones)
    os.makedirs(rutaEmociones)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap=cv2.VideoCapture(1)
if not cap.isOpened():
    print("No se puede abrir la cámara")

contador = 400

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)
        rostro = gray[y:y+h, x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(rutaEmociones + '/rostro_{}.jpg'.format(contador),rostro)
        contador = contador + 1

    cv2.imshow('Recolección para Dataset',frame)
    k =  cv2.waitKey(1)
    if k == 27 or contador >= 600:
        break

cap.release()
cv2.destroyAllWindows()
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

rutaDataset = 'C:\proyectos\Modelo Fundamento IA\Deteccion de emociones por LBPH\DeteccionemocionesLBPH\Dataset'
listaEmociones = os.listdir(rutaDataset)
print('Lista de emociones: ', listaEmociones)
labels = []
facesData = []
label = 0
for emociones in listaEmociones:
    rutaEmociones = rutaDataset + '/' + emociones
    for rostros in os.listdir(rutaEmociones):
        labels.append(label)
        facesData.append(cv2.imread(rutaEmociones+'/'+rostros,0))
    label = label + 1

emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()
print("Entrenando LBPH...")
emotion_recognizer.train(facesData, np.array(labels))
rutadeguardado = 'C:\proyectos\Modelo Fundamento IA\Deteccion de emociones por LBPH\DeteccionemocionesLBPH\Modelo.xml'
emotion_recognizer.write(rutadeguardado)
print("Modelo guardado")
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

emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()
emotion_recognizer.read('C:\proyectos\Modelo Fundamento IA\Deteccion de emociones por LBPH\DeteccionemocionesLBPH\Modelo.xml')

rutaDataset = 'Deteccion de emociones por LBPH\DeteccionemocionesLBPH\Dataset'
listaEmociones = os.listdir(rutaDataset)
print('Lista de emociones: ',listaEmociones)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap=cv2.VideoCapture(1)
if not cap.isOpened():
    print("No se puede abrir la cámara")

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        rostro = gray[y:y+h, x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
        result = emotion_recognizer.predict(rostro)
        if result[1] < 60:
            cv2.putText(frame,'{}'.format(listaEmociones[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)
        else:
            cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow('Reconocimiento de emociones', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
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
import cv2
from deepface import DeepFace

emociones_validas = ['neutral', 'happy', 'sad']

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("No se puede abrir la cámara")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    emocion = None
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emocion_detectada = result[0]['dominant_emotion']
        if emocion_detectada in emociones_validas:
            emocion = emocion_detectada
    except:
        pass

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if emocion:
        cv2.putText(frame, emocion,(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)

    cv2.imshow('Detector de emociones', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
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

## 📚 Referencias

- OpenCV Documentation: [https://docs.opencv.org](https://docs.opencv.org)  
- DeepFace Documentation: [https://github.com/serengil/deepface](https://github.com/serengil/deepface)  
- LBPH Face Recognizer: [https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html](https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html)

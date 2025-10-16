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
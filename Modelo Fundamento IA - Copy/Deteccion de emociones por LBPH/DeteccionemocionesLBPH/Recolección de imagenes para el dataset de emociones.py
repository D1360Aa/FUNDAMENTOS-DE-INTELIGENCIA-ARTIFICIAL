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
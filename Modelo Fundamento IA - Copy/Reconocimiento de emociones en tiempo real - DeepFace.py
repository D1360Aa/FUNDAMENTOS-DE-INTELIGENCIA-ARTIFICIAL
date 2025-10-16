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
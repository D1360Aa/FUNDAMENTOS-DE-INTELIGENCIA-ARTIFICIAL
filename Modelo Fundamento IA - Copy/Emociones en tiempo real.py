import cv2
from deepface import DeepFace

# Emociones que se quiere mostrar
emociones_validas = ['neutral', 'happy', 'sad']

# Detector de rostros de OpenCV
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Iniciar webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("No se puede abrir la cámara")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    emocion = None  # Variable para guardar emoción si es válida

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emocion_detectada = result[0]['dominant_emotion']

        # Solo guardar la emoción si está en la lista permitida
        if emocion_detectada in emociones_validas:
            emocion = emocion_detectada

    except:
        pass  # Si no hay rostro o falla algo, no hacemos nada

    # Detectar rostros con OpenCV
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostrar emoción solo si es válida
    if emocion:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, emocion,(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)

    # Mostrar el video
    cv2.imshow('Detector de emociones', frame)

    # Salir con tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
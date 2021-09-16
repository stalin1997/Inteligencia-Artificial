# Importar librerias
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# Grabar las dimeciones que vamos a utilizar
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# Obtener las detecciones de rostros
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# inicializar nuestra lista de caras, sus ubicaciones correspondientes,
	# y la lista de predicciones de nuestra red de mascarillas
	faces = []
	locs = []
	preds = []

	# Recorrer la detecciones 
	for i in range(0, detections.shape[2]):
		# extraer la deteccion del rostro
		confidence = detections[0, 0, i, 2]

		# filtrar las detecciones
		if confidence > args["confidence"]:
			# calcular las coordenadas (x, y) del cuadro delimitador para el objeto

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# asegúrese de que los cuadros delimitadores estén dentro de las dimensiones de el marco

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# ordenar y procesar el tamaño
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# agregar los cuadro delimitadores a sus respectivas listas
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# hacer la prediccion si se detecta solo una
	if len(faces) > 0:
		#  el bucle que muestra las predicciones al mismo tiempo
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# devuelve la ubicacion de las caras 
	return (locs, preds)

# construir los argumentos para analizar
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# cargar nuestro analizador cuando se inicailze
print("[INFO] inicializando el modelo...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# cargar el modelo de detector
print("[INFO] inicializando la mascarrilla facial...")
maskNet = load_model(args["model"])

# inicializando la transmicion de video
print("[INFO] Inicializando transmicion por camara...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# recorrer los fotogramas de la secuencia de vídeo
while True:
	# tomar el fotograma de la secuencia de video enhebrada y cambiar su tamaño
	# para tener un ancho máximo de 400 píxeles	
	frame = vs.read()
	frame = imutils.resize(frame, width=800)

	# detectar rostros en el marco y determinar si están usando un
	# mascarilla facial o no
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# recorre las ubicaciones de las caras detectadas y sus correspondientes
	# ubicaciones
	for (box, pred) in zip(locs, preds):
		# desempaqueta el cuadro delimitador y las predicciones
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determinar la etiqueta de clase y el color que usaremos para dibujar
		# el cuadro delimitador y el texto
		label = "Con mascarrilla" if mask > withoutMask else "Sin mascarrilla"
		color = (0, 255, 0) if label == "Con mascarrilla" else (0, 0, 255)

		# incluir la probabilidad en la etiqueta
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# mostrar la etiqueta y el rectángulo del cuadro delimitador en la salida
		# marco
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# mostrar el marco de salida
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# si se presionó la tecla `q`, salga del bucle
	if key == ord("q"):
		break

# hacer un poco de limpieza
cv2.destroyAllWindows()
vs.stop()
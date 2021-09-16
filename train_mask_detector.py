# USAGE
# python train_mask_detector.py --dataset dataset

# import the necessary packages
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# construir argumento
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to output face mask detector model")
args = vars(ap.parse_args())

# inicializar las epocas de aprendizaje
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# grabar la lista de imagenes del dataset
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# extraer las clases 
for imagePath in imagePaths:
	# extraer la clases de imagenes
	label = imagePath.split(os.path.sep)[-2]

	# procesar las imagenes  (224x224) 
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	# subir el data set
	data.append(image)
	labels.append(label)

# convertir el data en un array
data = np.array(data, dtype="float32")
labels = np.array(labels)

# encontrar las etiquetas
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# particione los datos en divisiones de entrenamiento y prueba usando el 75% de
# los datos para entrenamiento y el 25% restante para pruebas
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# construir el generador de imágenes de entrenamiento para el aumento de datos
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# cargue la red MobileNetV2, asegurándose de que los conjuntos de capas FC principales estén
# Parado
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construya la cabeza del modelo que se colocará encima de la
# el modelo base
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# coloque el modelo FC de la cabeza encima del modelo base (esto se convertirá
# el modelo real que entrenaremos)
model = Model(inputs=baseModel.input, outputs=headModel)

# recorrer todas las capas del modelo base y congelarlas para que
# * no * actualizarse durante el primer proceso de formación
for layer in baseModel.layers:
	layer.trainable = False

# cmpilar nuestro modelo
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# entrenadonuestras redes neuronales
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# hacer las predicciones
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# por epocas entrenar cada imagen una por una
predIdxs = np.argmax(predIdxs, axis=1)

# clasificar las imagenes
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# serializar el modelo en el disco
print("[INFO] saving mask detector model...")
model.save(args["model"], save_format="h5")

# trazar la pérdida de entrenamiento y la precisión
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
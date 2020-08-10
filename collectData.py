import sqlite3
import tensorflow as tf
import glob
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# prepares data to run on facenet
def prepare_img(img):
    img = cv2.resize(img, (160, 160))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_pixels = img.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # face_pixels=np.expand_dims(face_pixels,axis=0)
    return face_pixels


def random_files(file_list, length):
    original_length = len(file_list)
    l_list = np.arange(length)
    np.random.shuffle(l_list)
    new_list = []
    for i in range(length):
        new_list += [file_list[l_list[i]]]
    return new_list

face_net = load_model("facenet_keras.h5")
face_net.compile(
    optimizer='adam',
    loss="binary_crossentropy",
    metrics=['accuracy']
)


destination = "G:/machinelearning/"
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
import cv2

file_list_c = glob.glob("E:\machinelearning/face project/celeb224/*.jpg")
file_list_c=random_files(file_list_c,10000)




embedding = []
for i in range(len(file_list_c)):
    try:
        if i % 100 == 0:
            print(str(i))
        color = cv2.resize(cv2.imread(file_list_c[i]), (224, 224))
        grey = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        grey_bgr = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
        temp = np.expand_dims(prepare_img(grey_bgr), axis=0)
        emm = face_net.predict(temp)[0]
        embedding += [[color, grey, emm]]

    except:
        print("error loading image")

embedding = np.array(embedding)
np.save("G:/machinelearning/imageneminiprepared/grey/faces.npy", embedding)

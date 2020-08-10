import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Conv1D, BatchNormalization, MaxPooling1D, Dense, Flatten
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from random import randint, uniform
import glob
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
import time


# prepares data to run on facenet
def prepare_img(img):
    img = cv2.resize(img, (160, 160))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_pixels = img.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # face_pixels=np.expand_dims(face_pixels,axis=0)
    return face_pixels

def extract_model_component(model, name, inp=None):
    if inp is None:
        collect_list = []
        for layer in model.layers:
            if name in layer._name:
                collect_list += [layer]
        length = len(collect_list) - 1
        z = collect_list[0](collect_list[0].input)
        for i in range(1, len(collect_list)):
            z = collect_list[i](z)
        new_model = Model(inputs=collect_list[0].input, outputs=z)
        for layer in new_model.layers:
            layer._name = layer._name + "extracted"
        return new_model
    else:
        collect_list = []
        for layer in model.layers:
            if name in layer._name:
                collect_list += [layer]
        length = len(collect_list) - 1
        z = collect_list[0](inp)
        for i in range(1, len(collect_list)):
            z = collect_list[i](z)
        new_model = Model(inputs=inp, outputs=z)
        for layer in new_model.layers:
            layer._name = layer._name + "extracted"
        return new_model


# [line_data.output, line_features_vertical.output, overall_features.output])
# model = Model(inputs=[vertical_input, overall_features_input], outputs=z)
def construct_model(model):
    over_all_input = Input(224,)
    vertical_input = Input(224,)
    v_features = extract_model_component(model, "_ver_", vertical_input)
    v_line = extract_model_component(model, "_raw_", vertical_input)
    combined = concatenate([v_line.output, v_features.output, over_all_input])
    collect_list = model.layers[52:60]
    z = collect_list[0](combined)
    for i in range(1, len(collect_list)):
        z = collect_list[i](z)
    new_model = Model(inputs=[vertical_input, over_all_input], outputs=z)
    return new_model


root = tk.Tk()
root.withdraw()
base_model = load_model("faces.h5")
f_model = extract_model_component(base_model, '_ov_')
final_model = construct_model(base_model)

model_fe = load_model("facenet_keras.h5")
while True:
    file_path = filedialog.askopenfilename()
    if file_path=="":
        break
    begin = time.time()
    grey_img = cv2.resize(cv2.imread(file_path), (224, 224))
    img = cv2.resize(cv2.imread(file_path), (224, 224))
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_grey_bgr = cv2.cvtColor(img_grey, cv2.COLOR_GRAY2BGR)

    np.expand_dims(preprocess_input(img_grey_bgr), axis=0)
    pre_features = model_fe.predict(np.expand_dims(prepare_img(img_grey_bgr), axis=0))[0]
    features = f_model.predict(np.expand_dims(pre_features, axis=0))[0]

    tiled_features = np.tile(features, (224, 1))
    full = final_model.predict_on_batch([img_grey / 255, tiled_features])

    mn = np.min(full)
    if mn < 0:
        full = full - mn
    # if any numbers larger than 1
    mx = np.max(full)
    if mx > 1:
        full = full * (255 / mx)
    else:
        full = full * 255

    full = np.uint8(full)
    end = time.time()
    print("process finished in", str(end - begin), "seconds")
    while True:
        cv2.imshow('window', full)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

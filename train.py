import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Conv1D, BatchNormalization, MaxPooling1D, Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model, load_model
from random import randint, uniform
import glob
import numpy as np
import cv2


def generator(batch_size=64):
    global data
    h = 224
    w = 224
    length = len(data)
    instructions = []
    for i in range(length):
        for j in range(224):
            instructions += [[i, j]]
    instructions = np.array(instructions)

    np.random.shuffle(instructions)
    count = 0
    while True:
        grey_line = []
        color_line = []
        embeddings = []
        for i in range(batch_size):
            inst = instructions[count]
            im_index = inst[0]
            a = inst[1]
            count += 1
            if count >= len(instructions) - 1:
                count = 0
            # rand = randint(0, len(data) - 1)
            D = data[im_index]
            color_img = D[0]
            grey_img = D[1]
            embedding = D[2]
            embeddings += [embedding]
            a = randint(0, 223)
            # chance to use a vertical or horizontal line
            line_c = color_img[a]
            line_g = grey_img[a]

            color_line += [line_c]
            grey_line += [line_g]
        grey_line = np.array(grey_line) / 255
        color_line = np.array(color_line) / 255
        embeddings = np.array(embeddings)

        yield [grey_line, embeddings], color_line


# model = Model(inputs=[pixel_input, vertical_input, horizontal_input, overall_features_input], outputs=z)

vertical_input = Input(shape=(224,))
overall_features_input = (Input(shape=(128,)))

overall_features_layers = [
    overall_features_input,
    Dense(224), BatchNormalization(), tf.keras.layers.LeakyReLU(alpha=0.3),
    Dense(224), BatchNormalization(), tf.keras.layers.LeakyReLU(alpha=0.3),
    Dense(224), BatchNormalization(), tf.keras.layers.LeakyReLU(alpha=0.3),
    Dense(224), BatchNormalization(), tf.keras.layers.LeakyReLU(alpha=0.3),
]

line_feature_layers_vertical = [vertical_input,
                                Reshape((224, 1), input_shape=(224,)),
                                Conv1D(filters=32, kernel_size=3, padding='same', input_shape=(224, 1)),
                                BatchNormalization(),
                                tf.keras.layers.LeakyReLU(alpha=0.3),
                                MaxPooling1D(pool_size=2),
                                Conv1D(filters=64, kernel_size=3, padding='same', input_shape=(112, 32)),
                                BatchNormalization(),
                                tf.keras.layers.LeakyReLU(alpha=0.3),
                                MaxPooling1D(pool_size=2),
                                Conv1D(filters=128, kernel_size=3, padding='same', input_shape=(56, 64)),
                                BatchNormalization(),
                                tf.keras.layers.LeakyReLU(alpha=0.3),
                                MaxPooling1D(pool_size=2),
                                Conv1D(filters=256, kernel_size=3, padding='same', input_shape=(28, 128)),
                                BatchNormalization(),
                                tf.keras.layers.LeakyReLU(alpha=0.3),
                                MaxPooling1D(pool_size=2),
                                Conv1D(filters=512, kernel_size=3, padding='same', input_shape=(14, 256)),
                                BatchNormalization(),
                                tf.keras.layers.LeakyReLU(alpha=0.3),
                                MaxPooling1D(pool_size=2),
                                tf.keras.layers.Flatten(),
                                Dense(224), BatchNormalization(), tf.keras.layers.LeakyReLU(alpha=0.3),
                                Dense(224), BatchNormalization(), tf.keras.layers.LeakyReLU(alpha=0.3),
                                Dense(224), BatchNormalization(), tf.keras.layers.LeakyReLU(alpha=0.3),
                                Dense(224), BatchNormalization(), tf.keras.layers.LeakyReLU(alpha=0.3), ]

line_data_layers = [
    vertical_input,
    Dense(224), BatchNormalization(), tf.keras.layers.LeakyReLU(alpha=0.3)
]

line_data = tf.keras.models.Sequential(line_data_layers)

line_features_vertical = tf.keras.models.Sequential(line_feature_layers_vertical)
overall_features = tf.keras.models.Sequential(overall_features_layers)

for layer in line_features_vertical.layers:
    layer._name = layer._name + "_ver_"

for layer in line_data.layers:
    layer._name = layer._name + "_raw_"

for layer in overall_features.layers:
    layer._name = layer._name + "_ov_"

combined = concatenate(
    [line_data.output, line_features_vertical.output, overall_features.output])

z = Dense(672)(combined)
z = tf.keras.layers.BatchNormalization()(z)
z = tf.keras.layers.LeakyReLU(alpha=0.3)(z)
z = Dense(672)(z)
z = tf.keras.layers.BatchNormalization()(z)
z = tf.keras.layers.LeakyReLU(alpha=0.3)(z)
z = Dense(672,activation='relu')(z)
z = tf.keras.layers.Reshape((224, 3), input_shape=(672,))(z)

model = Model(inputs=[vertical_input, overall_features_input], outputs=z)
print("loading data")
data = np.load("G:/machinelearning/imageneminiprepared/grey/faces.npy", allow_pickle=True)
print("done")

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.0001),
    loss="mse",
    metrics=['accuracy', 'mae']
)
batch_size = 256
data_size = len(data) * 224
print("data size at",str( len(data) * 224),"reduced to ",str(len(data) * 224 / 10))
steps = int(data_size / batch_size)

datagen = generator(batch_size=batch_size)
model.fit_generator(datagen, steps_per_epoch=steps, epochs=10, verbose=1)
model.save("E:/machinelearning/saved models/faces.h5")

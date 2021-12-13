import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, \
    BatchNormalization, Activation, GlobalAveragePooling2D, DepthwiseConv2D, Softmax, \
    Dropout, ReLU, Concatenate, Conv2DTranspose, Input, Add, AveragePooling2D, ZeroPadding2D
import tensorflow.keras.losses as losses
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot
import data_processing
import os


def build_conv_network(num_layers, filters, image_size=(360, 640), kernel=3, classes=2, activation='relu'):
    # initialize model:
    model = keras.Sequential()
    optimizer = optimizers.Adam(amsgrad=True)
    # Normalising layer:
    model.add(layers.BatchNormalization(input_shape=(image_size[0], image_size[1], 3)))
    # construct network:
    for i in range(num_layers):
        if i % int(2*kernel) == 0 or i == num_layers - 1:
            model.add(layers.Conv2D(filters, kernel, activation=activation))
            model.add(layers.Dropout(0.2))
            model.add(layers.MaxPooling2D(strides=(2, 2)))
        else:
            model.add(layers.Conv2D(filters, kernel, activation=activation))
            model.add(layers.Dropout(0.2))
            model.add(layers.MaxPooling2D(strides=(1, 1)))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(16, activation=activation))
    model.add(layers.Dense(classes))
    model.add(layers.Softmax())
    model.compile(
        optimizer=optimizer,
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])
    model.summary()

    return model
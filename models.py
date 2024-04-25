from keras.layers import (Input, Conv2D, MaxPooling2D,
                                     Flatten, Dense, Layer, BatchNormalization,
                                     Dropout, Reshape, TimeDistributed)
from keras.models import Model
from tensorflow import keras
import keras.backend as K
import tensorflow as tf
import numpy as np

def build_model(input_shape):

    model = keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), 
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(32, activation = 'relu'),
    Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
  
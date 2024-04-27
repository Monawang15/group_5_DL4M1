from keras.layers import (Input, Conv2D, MaxPooling2D,
                                     Flatten, Dense, Layer, BatchNormalization,
                                     Dropout, LSTM,Reshape, TimeDistributed)
from keras.models import Model
from tensorflow import keras
import keras.backend as K
import tensorflow as tf
import numpy as np

def build_model(input_shape):
    model = keras.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(13, 172, 1))) # Make sure use the correct shape, as your train_data_features shape
    model.add(MaxPooling2D((2, 1))) # try different pool size
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 1)))# try different pool size
    model.add(Conv2D(64, (2, 2), padding='same', activation='relu')) #Try (2, 2) kernel
    # Flatten the output of the convolutional layers
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation="sigmoid")) # dont forget activation here
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
   
   
'''
    model = keras.Sequential([
        Input(shape=input_shape),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        # Removed Flatten()
        Conv2D(128, kernel_size=(1, 1), activation='relu'),  # 1x1 convolution to act as dense layer
        Dropout(0.5),
        Conv2D(64, kernel_size=(1, 1), activation='relu'),   # Another 1x1 convolution
        Conv2D(10, kernel_size=(1, 1), activation='softmax') # Final layer with 10 channels for 10 classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
'''
  
def tcn_model(input_shape):

    model = keras.Sequential()
    
    # Adding the TCN layer
    model.add(TCN(input_shape=input_shape,
                  nb_filters=64,
                  kernel_size=5,
                  dilations=(1, 2, 4, 8, 16, 32)))

    # Adding the output layer
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
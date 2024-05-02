from keras.layers import (Input, Conv2D, MaxPooling2D,
                                     Flatten, Dense, Layer, BatchNormalization,
                                     Dropout, LSTM,Reshape, TimeDistributed)
from keras.models import Model
from tensorflow import keras
import keras.backend as K
import tensorflow as tf
import numpy as np
from tcn import TCN

def build_model(input_shape):
    """
    Create a Convolutional Neural Network (CNN) model for binary classification.

    The CNN architecture consists of three convolutional blocks followed by fully connected layers.
    Each convolutional block has the following layers:
    1. Conv2D layer with 32 filters, a kernel size of (3, 3), ReLU activation, and 'same' padding to maintain spatial dimensions
    2. MaxPooling2D layer with a pool size of (2, 1) to reduce the spatial size vertically
    3. Conv2D layer with 64 filters, a kernel size of (3, 3), ReLU activation, and 'same' padding
    4. MaxPooling2D layer with a pool size of (2, 1) to further reduce size vertically
    5. Conv2D layer with 64 filters, a kernel size of (2, 2), ReLU activation, and 'same' padding

    After the convolutional blocks, the following fully connected layers are added:
    1. Flatten layer to convert the 2D feature maps into a 1D feature vector
    2. Dense layer with 64 units and ReLU activation
    3. Dense layer with 1 unit and sigmoid activation (output layer)

    The model is compiled using the Adam optimizer, binary crossentropy loss,
    and accuracy metric.

    Parameters
    ----------
    input_shape : tuple
      The shape of the input data, which includes the height, width, and channels of the input feature map.
      For example, (13, 798, 1) indicates a feature map with 13 height, 798 width, and a single channel.

    Returns
    -------
    model : keras.Sequential
      A compiled Keras Sequential model with the CNN architecture suitable for binary classification tasks.
    """
     
    model = keras.Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(13, 798, 1))) # train_data_features shape
    model.add(MaxPooling2D((2, 1)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 1)))
    model.add(Conv2D(64, (2, 2), padding='same', activation='relu')) 
    # Flatten the output of the convolutional layers
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation="sigmoid")) 
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
   
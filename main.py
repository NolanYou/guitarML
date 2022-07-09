import numpy as np
import tensorflow
from keras import Sequential
from keras.layers import *
from keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import SGD

from matplotlib import pyplot as plt
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
if __name__ == '__main__':
    if tensorflow.test.gpu_device_name():
        print('Default GPU Device:{}'.format(tensorflow.test.gpu_device_name()))
    batch_size = 32
    epochs = 10

    (trainX, trainy), (testX, testy) = mnist.load_data()
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    trainY = to_categorical(trainy)
    testY = to_categorical(testy)
    trainX = trainX.astype('float32')
    testX = testX.astype('float32')

    trainX /= 255.0
    testX /= 255.0

    model = Sequential()
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())

    # model.add(Dense(units=64, activation='relu', input_dim=100))
    model.add(Dense(trainY.shape[1], activation=Activation(tensorflow.nn.softmax)))

    model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                  optimizer=SGD(learning_rate=0.01, momentum=0.9),
                  metrics=['accuracy'])
    history = model.fit(trainX, trainY, batch_size = batch_size, epochs = epochs, verbose=1)
    print(history.history.keys())

    # ----------------------------- #
    # end your neural network

    model.summary()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import numpy as np
from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense


class NeuralNetworkModel:
    def __init__(self):
        self.model = Sequential()

        self.model.add(Conv2D(filters=16, kernel_size=2, input_shape=(112, 112, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=2))
        
        self.model.add(Conv2D(filters=32, kernel_size=2, activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=2))
        
        self.model.add(Conv2D(filters=64, kernel_size=2, activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=2))
        
        self.model.add(Conv2D(filters=128, kernel_size=2, activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=2))
        
        self.model.add(Dropout(0.3))
        self.model.add(Flatten())
        self.model.add(Dense(100))
        self.model.add(Activation('relu'))
        self.model.add(Dense(10, activation='softmax'))
        self.model.summary()

    def compile(self):
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['categorical_accuracy']
        )

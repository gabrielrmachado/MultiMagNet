# Imported from https://blog.keras.io/building-autoencoders-in-keras.html

import tensorflow as tf
from keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import math
from team_techniques.techinique import Technique
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, BatchNormalization, Activation
from keras.models import Model
from keras import backend as K
from keras import regularizers as regs

class TF_CAE_MNIST(Technique):
    def __init__(self, data, name, structure, epochs = 5, batch_size = 256):
        super().__init__(data, id, None)
        self.__epochs = epochs
        self.__batch_size = batch_size
        self.structure = structure
        self.__data = data
        self.name = name

        self.__x_train = self.tec_data.x_train
        self.__x_test = self.tec_data.x_test

    def __createAutoencoder(self):
        input_img = Input(shape=(28,28,1))  # adapt this if using `channels_first` image data format

        x = input_img

        for layer in self.structure:
            if isinstance(layer, int):
                x = Conv2D(layer, (3, 3), activation='relu', padding="same",
                           activity_regularizer=regs.l2(1e-9))(x)
            elif layer == "max":
                x = MaxPooling2D((2, 2), padding="same")(x)
            elif layer == "average":
                x = AveragePooling2D((2, 2), padding="same")(x)
            else:
                print(layer, "is not recognized!")
                exit(0)

        encoded = x

        for layer in reversed(self.structure):
            if isinstance(layer, int):
                x = Conv2D(layer, (3, 3), activation='relu', padding="same",
                           activity_regularizer=regs.l2(1e-9))(x)
            elif layer == "max" or layer == "average":
                x = UpSampling2D((2, 2))(x)

        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same',
                         activity_regularizer=regs.l2(1e-9))(x)       

        self.autoencoder = Model(input_img, decoded)
        self.encoder = Model(input_img, encoded)

    def execute(self):
        self.__createAutoencoder()
        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        try:
            self.autoencoder.load_weights("./team_techniques/models/" + self.name + ".h5")
        except:
            x_train = self.__x_train
            x_test = self.__x_test

            self.autoencoder.fit(x_train, x_train,
                    epochs=self.__epochs,
                    batch_size=self.__batch_size,
                    shuffle=True,
                    validation_data=(x_test, x_test))
            
            self.autoencoder.save_weights("./team_techniques/models/" + self.name + ".h5")

    def predict(self, k_most_similar_images, only_encoder = False):
        if only_encoder:
            return self.encoder.predict(k_most_similar_images, batch_size=self.__batch_size)
        else:
            return self.autoencoder.predict(k_most_similar_images, batch_size=self.__batch_size)
            
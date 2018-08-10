# This code was adapted from:
# https://blog.keras.io/building-autoencoders-in-keras.html

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import utils.helpers as helpers
from team_techniques.techinique import Technique
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
import utils.helpers

class TF_SAE(Technique):
   
    def __init__(self, data, img_shape, epochs = 20, batch_size = 256, 
                 encoded_dim = 64, regularizer = 10e-9):
        super().__init__(data, None)
        self.__epochs = epochs
        self.__batch_size = batch_size
        self.__img_shape = img_shape
        self.__encoding_dim = encoded_dim
        self.__regularizer = regularizer

    def __createAutoencoder(self):
        input_img = Input(shape=(int(self.__img_shape),))
        encoded = Dense(int(self.__encoding_dim), activation = 'relu', 
                                                  activity_regularizer=regularizers.l1(self.__regularizer))(input_img)
        decoded = Dense(self.__img_shape, activation='sigmoid')(encoded)

        autoencoder = Model(input_img, decoded)
        encoder = Model(input_img, encoded)

        encoded_input = Input(shape=(self.__encoding_dim,))
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(encoded_input, decoder_layer(encoded_input))

        self.autoencoder = autoencoder
        self.encoder = encoder
        self.decoder = decoder

    def execute(self):
        self.__createAutoencoder()
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        x_train = self.tec_data.x_train
        x_test = self.tec_data.x_test

        self.autoencoder.fit(x_train, x_train, epochs=self.__epochs, batch_size=self.__batch_size, shuffle=True, 
                        validation_data=(x_test, x_test))

        #helpers.plot_encoding(self.tec_data.x_test, self.encoder, self.decoder)

    def predict(self, x, only_encoder=False):
        x = helpers.reshape_flatten(x)
        if only_encoder:
            return self.encoder.predict(x, batch_size=self.__batch_size)
        else:
            imgs = self.autoencoder.predict(x, batch_size=self.__batch_size)
            imgs = imgs.reshape(len(imgs), 28, 28, 1)
            return imgs
            
    
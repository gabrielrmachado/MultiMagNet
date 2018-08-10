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

class TF_StackAE(Technique):
   
    def __init__(self, data, img_shape, layers = [512, 256, 128, 64], epochs = 20, batch_size = 256, 
                 encoded_dim = 64):
        super().__init__(data, None)
        self.__epochs = epochs
        self.__batch_size = batch_size
        self.__img_shape = img_shape
        self.__encoding_dim = encoded_dim
        self.__layers = layers

    def __createAutoencoder(self):
        layers = self.__layers
        input_img = Input(shape=(int(self.__img_shape),))
        encoded = Dense(layers[0], activation='relu')(input_img)
        
        for i in range(len(layers)-1):
            encoded = Dense(layers[i+1], activation='relu')(encoded)

        decoded = Dense(layers[len(layers)-2], activation='relu')(encoded)
        for i in range(len(layers)-3, -1, -1):
            decoded = Dense(layers[i], activation='relu')(encoded)  

        decoded = Dense(int(self.__img_shape), activation='sigmoid')(encoded)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
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
        k_data = helpers.reshape_flatten(x)
        
        if only_encoder:
            return self.encoder.predict(k_data, batch_size=self.__batch_size)
        else: 
            imgs = self.autoencoder.predict(k_data, batch_size=self.__batch_size)
            imgs = imgs.reshape(len(imgs), 28, 28, 1)
            return imgs
            
    
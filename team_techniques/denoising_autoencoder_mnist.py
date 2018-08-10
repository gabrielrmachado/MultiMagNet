# This code was adapted from:
# https://blog.keras.io/building-autoencoders-in-keras.html

import tensorflow as tf
from keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import math
import utils.helpers as helpers
from team_techniques.techinique import Technique
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, BatchNormalization, Activation, Flatten, Reshape
from keras.models import Model
from keras import backend as K
from keras import regularizers as regs

class TF_DAE_MNIST(Technique):
    def __init__(self, data, name, epochs = 10, batch_size = 256, noise_factor = 0.5):
        super().__init__(data, None)
        self.__epochs = epochs
        self.__batch_size = batch_size

        self.__x_train = self.tec_data.x_train
        self.__x_test = self.tec_data.x_test

        # add noise to the data inputs
        self.__x_train_noisy = self.__x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, 
                                                                         size=self.__x_train.shape) 
        self.__x_test_noisy = self.__x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0,
                                                                          size=self.__x_test.shape) 

        self.__x_train_noisy = np.clip(self.__x_train_noisy, 0., 1.)
        self.__x_test_noisy = np.clip(self.__x_test_noisy, 0., 1.)

        self.name = name

    def __createAutoencoder(self):
        input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
        
        x = Conv2D(32, (3, 3), activation='relu', padding='same', activity_regularizer=regs.l2(1e-9))(input_img)
        x = AveragePooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same', activity_regularizer=regs.l2(1e-9))(x)
        encoded = AveragePooling2D((2, 2), padding='same')(x)

        # at this point the representation is (7, 7, 32)
        x = Conv2D(32, (3, 3), activation='relu', padding='same', activity_regularizer=regs.l2(1e-9))(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same', activity_regularizer=regs.l2(1e-9))(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', activity_regularizer=regs.l2(1e-9))(x)
            
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
            x_train_noisy = self.__x_train_noisy

            self.autoencoder.fit(x_train_noisy, x_train,
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
            
    
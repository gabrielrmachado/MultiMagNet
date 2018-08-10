import tensorflow as tf
from keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import math
import utils.helpers as helpers
from team_techniques.techinique import Technique
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Flatten, Reshape
from keras.models import Model
from keras import backend as K
from keras import regularizers

class TF_DAE_CIFAR(Technique):
    def __init__(self, data, name, opt = 1, epochs = 5, batch_size = 128, 
                    noise_factor = 0.5):
            super().__init__(data, None)
            self.__epochs = epochs
            self.__batch_size = batch_size
            self.__opt = opt

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
        input_img = Input(shape=(32, 32, 3))  # adapt this if using `channels_first` image data format

        if self.__opt == 1:
            x = Conv2D(32, (3, 3), padding='same')(input_img)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D((2, 2), padding='same')(x)
            x = Conv2D(64, (3, 3), padding='same')(input_img)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D((2, 2), padding='same')(x)
            x = Conv2D(128, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            encoded = MaxPooling2D((2, 2), padding='same')(x)

            x = Conv2D(128, (3, 3), padding='same')(encoded)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(64, (3, 3), padding='same')(encoded)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(32, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(3, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            decoded = Activation('sigmoid')(x)

        if self.__opt == 2:
            x = Conv2D(32, (3, 3), padding='same')(input_img)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D((2, 2), padding='same')(x)
            x = Conv2D(32, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            encoded = MaxPooling2D((2, 2), padding='same')(x)

            x = Conv2D(32, (3, 3), padding='same')(encoded)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(32, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(3, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            decoded = Activation('sigmoid')(x)

        self.autoencoder = Model(input_img, decoded)
        self.encoder = Model(input_img, encoded)
        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    def execute(self):
        self.__createAutoencoder()

        try:
            self.autoencoder.load_weights("./team_techniques/models/" + self.name + ".h5")
        
        except:
            x_train = self.__x_train
            x_test = self.__x_test
            x_train_noisy = self.__x_train_noisy
            x_test_noisy = self.__x_test_noisy

            self.autoencoder.fit(x_train_noisy, x_train,
                    epochs=self.__epochs,
                    batch_size=self.__batch_size,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test))

            self.autoencoder.save_weights("./team_techniques/models/" + self.name + ".h5")

    def predict(self, k_most_similar_images, only_encoder = False):
        if only_encoder:
            return self.encoder.predict(k_most_similar_images, batch_size=self.__batch_size)
        else:
            return self.autoencoder.predict(k_most_similar_images, batch_size=self.__batch_size)
            

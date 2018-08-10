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
    def __init__(self, data, name, opt = 1, epochs = 5, batch_size = 256):
        super().__init__(data, None)
        self.__epochs = epochs
        self.__batch_size = batch_size
        self.__opt = opt
        self.__data = data
        self.name = name

        self.__x_train = self.tec_data.x_train
        self.__x_test = self.tec_data.x_test

    def __createAutoencoder(self):
        input_img = Input(shape=(28,28,1))  # adapt this if using `channels_first` image data format
        
        if self.__opt == 1:            
            conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', activity_regularizer=regs.l2(1e-9))(input_img) #28 x 28 x 32
            pool1 = AveragePooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
            conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', activity_regularizer=regs.l2(1e-9))(pool1) #14 x 14 x 64
            pool2 = AveragePooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
            encoded = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)

            #decoder
            conv4 = Conv2D(128, (3, 3), activation='relu', padding='same', activity_regularizer=regs.l2(1e-9))(encoded) #7 x 7 x 128
            up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
            conv5 = Conv2D(64, (3, 3), activation='relu', padding='same', activity_regularizer=regs.l2(1e-9))(up1) # 14 x 14 x 64
            up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
            decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', activity_regularizer=regs.l2(1e-9))(up2) # 28 x 28 x 1
        
        if self.__opt == 2:
            x = Conv2D(32, (3, 3), activation='relu', padding='same', activity_regularizer=regs.l2(1e-9))(input_img) # 28 x 28 x 32
            x = MaxPooling2D((2, 2), padding='same')(x) # 14 x 14 x 32
            x = Conv2D(64, (3, 3), activation='relu', padding='same', activity_regularizer=regs.l2(1e-9))(x) # 14 x 14 x 64
            x = MaxPooling2D((2, 2), padding='same')(x) # 7 x 7 x 64
            x = Conv2D(128, (3, 3), activation='relu', padding='same', activity_regularizer=regs.l2(1e-9))(x) # 7 x 7 x 128
            x = MaxPooling2D((2, 2), padding='same')(x) # 4 x 4 x 128
            x = Conv2D(256, (3, 3), activation='relu', padding='same', activity_regularizer=regs.l2(1e-9))(x) # 7 x 7 x 256
            encoded = MaxPooling2D((2, 2), padding='same')(x) # 4 x 4 x 256

            # at this point the representation is (4, 4, 4) i.e. 128-dimensional
            x = Conv2D(256, (3, 3), activation='relu', padding='same', activity_regularizer=regs.l2(1e-9))(encoded) # 4 x 4 x 256
            x = UpSampling2D((2, 2))(x) # 8 x 8 x 256
            x = Conv2D(128, (3, 3), activation='relu', padding='same', activity_regularizer=regs.l2(1e-9))(x) # 4 x 4 x 128
            x = UpSampling2D((2, 2))(x) # 8 x 8 x 128
            x = Conv2D(64, (3, 3), activation='relu', padding='same', activity_regularizer=regs.l2(1e-9))(x) # 7 x 7 x 64
            x = UpSampling2D((2, 2))(x) # 14 x 14 x 64
            x = Conv2D(32, (3, 3), activation='relu', activity_regularizer=regs.l2(1e-9))(x) # 14 x 14 x 32
            x = UpSampling2D((2, 2))(x) # 28 x 28 x 32
            decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', activity_regularizer=regs.l2(1e-9))(x) # 28 x 28 x 1           

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
            
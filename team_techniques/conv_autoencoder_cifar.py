import tensorflow as tf
from keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import math
from team_techniques.techinique import Technique
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras.models import Model
from keras import backend as K
from keras import regularizers

class TF_CAE_CIFAR(Technique):
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
        input_img = Input(shape=(32,32,3))  # adapt this if using `channels_first` image data format

        if self.__opt == 1:
            # encoder            
            conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img) # 32 x 32 x 8
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #16 x 16 x 8
            conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool1) # 16 x 16 x 16
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) # 8 x 8 x 16
            conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool2) # 8 x 8 x 32
            encoded = MaxPooling2D(pool_size=(2, 2))(conv3) # 4 x 4 x 32
            #decoder
            conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded) # 4 x 4 x 256
            up1 = UpSampling2D((2,2))(conv4) # 8 x 8 x 256
            conv5 = Conv2D(16, (3, 3), activation='relu', padding='same')(up1) # 8 x 8 x 128
            up2 = UpSampling2D((2,2))(conv5) # 16 x 16 x 128
            conv6 = Conv2D(8, (3, 3), activation='relu', padding='same')(up2) # 16 x 16 x 64
            up3 = UpSampling2D((2,2))(conv6) # 32 x 32 x 64
            decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up3) # 32 x 32 x 3

        if self.__opt == 2:
            x = Conv2D(64, (3, 3), padding='same')(input_img)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D((2, 2), padding='same')(x)
            x = Conv2D(32, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D((2, 2), padding='same')(x)
            x = Conv2D(16, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            encoded = MaxPooling2D((2, 2), padding='same')(x)

            x = Conv2D(16, (3, 3), padding='same')(encoded)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(32, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(64, (3, 3), padding='same')(x)
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
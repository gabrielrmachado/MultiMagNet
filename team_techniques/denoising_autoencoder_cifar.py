import tensorflow as tf
from keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import math
import utils.helpers as helpers
from team_techniques.techinique import Technique
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, BatchNormalization, Activation, Flatten, Reshape
from keras.models import Model
from keras import backend as K
import keras.regularizers as regs
import keras

class TF_DAE_CIFAR(Technique):
    def __init__(self, data, structure, name, epochs = 5, batch_size = 128, 
                    noise_factor = 0.1, reg = 0.0, compiler='adam', batch_norm = False):
            super().__init__(data, None)
            self.__epochs = epochs
            self.__batch_size = batch_size
            self.structure = structure
            self.reg = reg
            self.compiler = compiler
            self.batch_norm = batch_norm

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
        x = input_img

        for layer in self.structure:
            if isinstance(layer, int):
                if self.reg != 0.0:
                    x = Conv2D(layer, (3,3), padding='same', activity_regularizer=regs.l2(self.reg))(x)
                else:
                    x = Conv2D(layer, (3,3), padding='same')(x)

                if self.batch_norm:
                    x = BatchNormalization()(x)
                x = Activation('relu')(x)
            
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
                if self.reg != 0.0:
                    x = Conv2D(layer, (3, 3), padding="same",
                               activity_regularizer=regs.l2(self.reg))(x)
                else:
                    x = Conv2D(layer, (3, 3), activation='relu', padding="same")(x)

                if self.batch_norm:
                    x = BatchNormalization()(x)
                x = Activation('relu')(x)

            elif layer == "max" or layer == "average":
                x = UpSampling2D((2, 2))(x)

        x = Conv2D(3, (3, 3), activation='sigmoid', padding='same',
                         activity_regularizer=regs.l2(self.reg))(x)
        x = BatchNormalization()(x)
        decoded = Activation('sigmoid')(x)

        self.autoencoder = Model(input_img, decoded)
        self.encoder = Model(input_img, encoded)
        
        if self.compiler == 'adam':
            self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        else:
            sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-4, nesterov=False)
            self.autoencoder.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

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
            

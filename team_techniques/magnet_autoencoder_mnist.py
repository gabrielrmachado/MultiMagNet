import matplotlib.pyplot as plt
import math
import utils.helpers as helpers
from team_techniques.techinique import Technique
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, BatchNormalization, Activation, Flatten, Reshape
from keras.models import Model
from keras import backend as K
from keras import regularizers as regs
import numpy as np
import os

class TF_DAE_MNIST(Technique):
    def __init__(self, data, name, structure, epochs = 100, v_noise=0.0, batch_size=256,
                 activation="relu", reg_strength=1e-9):
        super().__init__(data, None)
        self.image_shape = data.x_train.shape[1:]
        self.v_noise = v_noise
        self.activation = activation
        self.epochs = epochs
        self.batch = batch_size
        self.structure = structure
        self.reg_strength = reg_strength
        self.name = name

    def __createAutoencoder(self):
        input_img = Input(shape=self.image_shape)
        x = input_img

        for layer in self.structure:
            if isinstance(layer, int):
                x = Conv2D(layer, (3, 3), activation=self.activation, padding="same",
                           activity_regularizer=regs.l2(self.reg_strength))(x)
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
                x = Conv2D(layer, (3, 3), activation=self.activation, padding="same",
                           activity_regularizer=regs.l2(self.reg_strength))(x)
            elif layer == "max" or layer == "average":
                x = UpSampling2D((2, 2))(x)

        decoded = Conv2D(self.image_shape[2], (3, 3), activation='sigmoid', padding='same',
                         activity_regularizer=regs.l2(self.reg_strength))(x)
        
        self.model = Model(input_img, decoded)

    def execute(self):
        self.__createAutoencoder()
        self.model.compile(loss='mean_squared_error', metrics=['mean_squared_error'], optimizer='adam')

        try:
            self.model.load_weights("./team_techniques/models/" + self.name + ".h5")
        except:
            noise = self.v_noise * np.random.normal(size=np.shape(self.tec_data.x_train))
            noisy_train_data = self.tec_data.x_train + noise
            noisy_train_data = np.clip(noisy_train_data, 0.0, 1.0)

            self.model.fit(noisy_train_data, self.tec_data.x_train,
                        batch_size=self.batch,
                        validation_data=(self.tec_data.x_test, self.tec_data.x_test),
                        epochs=self.epochs,
                        shuffle=True)

            self.model.save_weights("./team_techniques/models/"+ self.name + ".h5")

    def predict(self, x):
        return self.model.predict(x, batch_size=self.batch)
        
       
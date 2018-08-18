from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.optimizers import SGD
import os
import tensorflow as tf
import numpy as np
import utils.helpers as helpers
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import LearningRateScheduler
from keras import optimizers
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
import keras

class Classifier:
    def __init__(self, sess, data, epochs = 10, batch_size = 64, learning_rate = 0.001, lr_decay = 1e-4, lr_drop = 20):
        self.__sess = sess
        self.__epochs = epochs
        self.__batch = batch_size
        self.__lr = learning_rate
        self.__lr_decay = lr_decay
        self.__lr_drop = lr_drop
        self.__data = data

    def __mnist_model(self):
        # build the main classifier
        # create model
        model = Sequential()
        model.add(Convolution2D(64, (5, 5), padding='valid', input_shape=self.__data.x_train.shape[1:], activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(10))
        
        return model

    def __cifar10_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=self.__data.x_train.shape[1:]))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10))

        # model = Sequential()
        # model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(self.__lr_decay), input_shape=self.__data.x_train.shape[1:]))
        # model.add(Activation('elu'))
        # model.add(BatchNormalization())
        # model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(self.__lr_decay)))
        # model.add(Activation('elu'))
        # model.add(BatchNormalization())
        # model.add(MaxPooling2D(pool_size=(2,2)))
        # model.add(Dropout(0.2))

        # model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(self.__lr_decay)))
        # model.add(Activation('elu'))
        # model.add(BatchNormalization())
        # model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(self.__lr_decay)))
        # model.add(Activation('elu'))
        # model.add(BatchNormalization())
        # model.add(MaxPooling2D(pool_size=(2,2)))
        # model.add(Dropout(0.3))

        # model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(self.__lr_decay)))
        # model.add(Activation('elu'))
        # model.add(BatchNormalization())
        # model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(self.__lr_decay)))
        # model.add(Activation('elu'))
        # model.add(BatchNormalization())
        # model.add(MaxPooling2D(pool_size=(2,2)))
        # model.add(Dropout(0.4))

        # model.add(Flatten())
        # model.add(Dense(10))

        return model

        #print("Main model parameters:\n{0}".format(model.summary()))

    def execute(self):
        """
        Fits and evaluates the main model, which is used for classifying input samples.

        # Attributes:
            dataset: The dataset name which is going to be used to train the main classifier. It can be MNIST, CIFAR10, GTSRB
        """
        x_train = self.__data.x_train
        x_test = self.__data.x_test
        y_train = self.__data.y_train
        y_test = self.__data.y_test

        model = self.get_model(logits=False)

        if self.__data.dataset_name.upper() == 'MNIST':
            # Builds the main model
            try:
                model.load_weights(os.path.dirname(os.path.realpath(__file__)) + '\\model_parameters\\mnist.h5')
                # Final evaluation of the model
                self.evaluate_model(model, x_test, y_test)
            except:
                print('\nTraining the MNIST main classifier...')

                # Fits the main model
                self.train_model(model)

                # Final evaluation of the model
                self.evaluate_model(model, x_test, y_test)                
                try:
                    model.save_weights(os.path.dirname(os.path.realpath(__file__)) + '\\model_parameters\\mnist.h5')                              
                except:
                    print("It has not been possible to save MNIST model's parameters.")
        
        if self.__data.dataset_name.upper() == 'CIFAR':
            try:
                model.load_weights(os.path.dirname(os.path.realpath(__file__)) + '\\model_parameters\\cifar10.h5')
                scores = model.evaluate(x_test, y_test, verbose=0)
                print("Baseline Error: %.2f%%" % (100-scores[1]*100))
            except:
                print('\nTraining the CIFAR10 main classifier...')

                datagen = ImageDataGenerator(
                    rotation_range=15,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    horizontal_flip=True,)
                datagen.fit(x_train)

                def lr_schedule(epoch):
                    lrate = self.__lr
                    if epoch > 75:
                        lrate = 0.0005
                    elif epoch > 100:
                        lrate = 0.0003        
                    return lrate

                model.fit_generator(datagen.flow(x_train, y_train, batch_size=self.__batch),\
                    steps_per_epoch=x_train.shape[0] // self.__batch, epochs=self.__epochs,\
                    verbose=1,validation_data=(x_test,y_test),callbacks=[LearningRateScheduler(lr_schedule)])
                
                # Final evaluation of the model
                scores = model.evaluate(x_test, y_test, verbose=0)
                print("Baseline Error: %.2f%%" % (100-scores[1]*100))

                try:
                    model.save_weights(os.path.dirname(os.path.realpath(__file__)) + '\\model_parameters\\cifar10.h5')
                except:
                    print("It has not been possible to save CIFAR-10 model's parameters.")
        
        return model

    def evaluate_model(self, model, x_test, y_test):
        scores = model.evaluate(x_test, y_test, verbose=0)
        print("Main classifier's baseline error: %.2f%%" % (100-scores[1]*100))

    def get_model(self, logits = True):
        if logits:
            if self.__data.dataset_name.upper() == "MNIST":
                model = self.__mnist_model()
                model.add(Activation('linear'))
                model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            else:
                model = self.__cifar10_model()
                model.add(Activation('linear'))
                opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
                model.compile(loss='sparse_categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])     
                model.load_weights(os.path.dirname(os.path.realpath(__file__)) + '\\model_parameters\\cifar10.h5')       
        else:        
            if self.__data.dataset_name.upper() == "MNIST":
                model = self.__mnist_model()
                model.add(Activation('softmax'))
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            else:
                model = self.__cifar10_model()
                model.add(Activation('softmax'))
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                # opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
                # model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
                
        self.__sess.run(tf.global_variables_initializer())
        return model        
        
    def train_model(self, model):
        keras.backend.set_session(self.__sess)
        model.fit(self.__data.x_train, self.__data.y_train, epochs=self.__epochs, verbose=1)

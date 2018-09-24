from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
import os
import tensorflow as tf
import numpy as np
import utils.helpers as helpers
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler
from keras import optimizers
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
import keras

class Classifier:
    def __init__(self, sess, data, epochs = 170, batch_size = 32, learning_rate = 0.01, lr_decay = 1e-4, lr_drop = 20):
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
        self.model = Sequential()
        self.model.add(Convolution2D(64, (5, 5), padding='valid', input_shape=self.__data.x_train.shape[1:], activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Convolution2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(Convolution2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(10))
        self.model.add(Activation("softmax"))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def __cifar10_model(self):
        # self.model = Sequential()
        # self.model.add(Conv2D(96, (3, 3), input_shape=self.__data.x_train.shape[1:]))
        # self.model.add(Activation('relu'))
        # self.model.add(Conv2D(96, (3, 3)))
        # self.model.add(Activation('relu'))
        # self.model.add(Conv2D(96, (3, 3)))
        # self.model.add(Activation('relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # self.model.add(Conv2D(192, (3, 3)))
        # self.model.add(Activation('relu'))
        # self.model.add(Conv2D(192, (3, 3)))
        # self.model.add(Activation('relu'))
        # self.model.add(Conv2D(192, (3, 3)))
        # self.model.add(Activation('relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # self.model.add(Conv2D(192, (3, 3)))
        # self.model.add(Activation('relu'))
        # self.model.add(Conv2D(192, (1, 1)))
        # self.model.add(Activation('relu'))
        # self.model.add(Conv2D(10, (1, 1)))
        # self.model.add(Activation('relu'))
        # self.model.add(GlobalAveragePooling2D())
        # self.model.add(Dense(10))
        # self.model.add(Activation("softmax"))

        self.model = Sequential()
        self.model.add(Conv2D(32, (3,3), padding='same', input_shape=self.__data.x_train.shape[1:]))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (3,3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(64, (3,3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3,3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, (3,3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(128, (3,3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.3))

        self.model.add(Flatten())
        self.model.add(Dense(10))
        self.model.add(Activation("softmax"))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def execute(self, logits=False):
        """
        Fits and evaluates the main model, which is used for classifying input samples.

        # Attributes:
            dataset: The dataset name which is going to be used to train the main classifier. It can be MNIST, CIFAR10, GTSRB
        """
        x_train = self.__data.x_train
        x_test = self.__data.x_test
        y_train = self.__data.y_train
        y_test = self.__data.y_test

        if self.__data.dataset_name.upper() == 'MNIST':
            # Builds the main model
            self.__mnist_model()
            try:
                self.model.load_weights(os.path.dirname(os.path.realpath(__file__)) + '\\model_parameters\\mnist.h5')
                # Final evaluation of the model
                self.evaluate_model(x_test, y_test)
            except:
                print('\nTraining the MNIST main classifier...')

                # Fits the main model
                self.train_model()

                # Final evaluation of the model
                self.evaluate_model(x_test, y_test)                
                try:
                    self.model.save_weights(os.path.dirname(os.path.realpath(__file__)) + '\\model_parameters\\mnist.h5')                              
                except:
                    print("It has not been possible to save MNIST model's parameters.")
        
        if self.__data.dataset_name.upper() == 'CIFAR':
            self.__cifar10_model()
            try:
                self.model.load_weights(os.path.dirname(os.path.realpath(__file__)) + '\\model_parameters\\cifar10.h5')
                scores = self.model.evaluate(x_test, y_test, verbose=0)
                print("Baseline Error: %.2f%%" % (100-scores[1]*100))
            except:
                print('\nTraining the CIFAR10 main classifier...')

                def lr_scheduler(epoch):
                    return self.__lr * (0.2 ** (epoch // 10))
                reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

                datagen = ImageDataGenerator(
                    featurewise_center=False,  # set input mean to 0 over the dataset
                    samplewise_center=False,  # set each sample mean to 0
                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                    samplewise_std_normalization=False,  # divide each input by its std
                    zca_whitening=False,  # apply ZCA whitening
                    zoom_range=0.2,
                    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
                    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                    horizontal_flip=True,  # randomly flip images
                    vertical_flip=False)  # randomly flip images
                # (std, mean, and principal components if ZCA whitening is applied).

                # datagen = ImageDataGenerator(
                #     width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
                #     height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
                #     horizontal_flip=True)  # randomly flip images

                datagen.fit(x_train)                
                
                # early_stop = EarlyStopping(monitor='val_acc', min_delta=0.00001, patience=10, \
                #                 verbose=1, mode='auto')

                historytemp = self.model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=self.__batch),
                            steps_per_epoch=x_train.shape[0] // self.__batch,
                            epochs=self.__epochs,
                            validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=1)

                
                # historytemp = self.model.fit_generator(datagen.flow(x_train, y_train,
                #                          batch_size=self.__batch),
                #             steps_per_epoch=x_train.shape[0] // self.__batch,
                #             epochs=self.__epochs,
                #             validation_data=(x_test, y_test),verbose=1)

                # Final evaluation of the model
                scores = self.model.evaluate(x_test, y_test, verbose=0)
                print("Baseline Error: %.2f%%" % (100-scores[1]*100))

                try:
                    self.model.save_weights(os.path.dirname(os.path.realpath(__file__)) + '\\model_parameters\\cifar10.h5')
                except:
                    print("It has not been possible to save CIFAR-10 model's parameters.")

    def evaluate_model(self, x_test, y_test):
        scores = self.model.evaluate(x_test, y_test, verbose=0)
        print("Main classifier's baseline error: %.2f%%" % (100-scores[1]*100))
        
    def train_model(self):
        keras.backend.set_session(self.__sess)
        self.model.fit(self.__data.x_train, self.__data.y_train, epochs=self.__epochs, verbose=1)

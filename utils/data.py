from keras.datasets import mnist
from keras.datasets import cifar10
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from keras.utils import np_utils

class Data(object):

    def __init__(self, dataset_name = 'MNIST', validation_data=5000):
        self.dataset_name = dataset_name        
        
        if dataset_name.upper() == 'MNIST':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
            
            self.x_train = self.x_train.astype('float32') / np.max(self.x_train)
            self.x_test = self.x_test.astype('float32') / np.max(self.x_test)

            # one hot encode outputs
            self.y_train = np_utils.to_categorical(self.y_train)
            self.y_test = np_utils.to_categorical(self.y_test)

            # reshape to be [samples][width][height][pixels]
            self.x_train = self.x_train.reshape(self.x_train.shape[0], 28, 28, 1).astype('float32')
            self.x_test = self.x_test.reshape(self.x_test.shape[0], 28, 28, 1).astype('float32')

        if dataset_name.upper() == 'CIFAR':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

            self.x_train = self.x_train.astype('float32') / np.max(self.x_train)
            self.x_test = self.x_test.astype('float32') / np.max(self.x_test)

            # mean = 120.707
            # std = 64.15

            # self.x_train = (self.x_train.astype('float32') - mean) / (std + 1e-7)
            # self.x_test = (self.x_test.astype('float32') - mean) / (std + 1e-7)

            # one hot encode outputs
            self.y_train = np_utils.to_categorical(self.y_train)
            self.y_test = np_utils.to_categorical(self.y_test)

        # creates validation set by spliting training set into the first 'validation_data' samples.
        if validation_data >= 0 and validation_data <= 8000:
            x_val = self.x_train[:validation_data]
            y_val = self.y_train[:validation_data]
            self.x_train = self.x_train[validation_data:]
            self.y_train = self.y_train[validation_data:]

            self.x_val = x_val
            self.y_val = y_val
        else:
            raise Exception("Validation set must have between 3000 and 8000 samples.")
            
        print('%s dataset loaded.' % self.dataset_name)
        print('x_train: ', self.x_train.shape)
        print('y_train: ', self.y_train.shape)
        print('x_test: ', self.x_test.shape)
        print('y_test: ', self.y_test.shape)
        if validation_data > 0.0: print('x_val: ', self.x_val.shape)
        if validation_data > 0.0: print('y_val: ', self.y_val.shape)
            



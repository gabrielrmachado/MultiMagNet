import numpy as np
import keras
from keras import backend
import tensorflow as tf
import utils.helpers as helpers
import keras
import winsound

from cleverhans.utils_keras import KerasModelWrapper
from art.classifiers import KerasClassifier

from classifiers.classifier import Classifier
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval
from art.attacks.deepfool import DeepFool
from art.attacks.carlini import CarliniL2Method
from art.attacks.fast_gradient import FastGradientMethod
from art.attacks.iterative_method import BasicIterativeMethod
from art.attacks.carlini import CarliniL2Method
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_keras import KerasModelWrapper
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from art.attacks.deepfool import DeepFool
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
import classifiers

class Adversarial_Attack:
    def __init__(self, sess, data, length, attack='FGSM',
                 num_filters = 64, batch_size = 128, epochs = 10):
        self.__data = data
        self.__image_rows = data.x_train.shape[1]
        self.__image_cols = data.x_train.shape[2]
        self.__channels = data.x_train.shape[3]
        self.__nb_classes = data.y_train.shape[1]
        self.__attack = attack
        self._length = length
        self.__sess = sess
        self.__batch = batch_size
        self.__epochs = epochs   
        self.__dataset = data.dataset_name 
        self.idx_adv = helpers.load_imgs_pkl('example_idx.pkl')

        self.surrogate_model = Sequential()
        self.surrogate_model.add(Conv2D(32, (3, 3), padding='same', input_shape=self.__data.x_train.shape[1:]))
        self.surrogate_model.add(Activation('relu'))
        self.surrogate_model.add(Conv2D(32, (3, 3)))
        self.surrogate_model.add(Activation('relu'))
        self.surrogate_model.add(MaxPooling2D(pool_size=(2, 2)))
        self.surrogate_model.add(Dropout(0.25))

        self.surrogate_model.add(Conv2D(64, (3, 3), padding='same'))
        self.surrogate_model.add(Activation('relu'))
        self.surrogate_model.add(Conv2D(64, (3, 3)))
        self.surrogate_model.add(Activation('relu'))
        self.surrogate_model.add(MaxPooling2D(pool_size=(2, 2)))
        self.surrogate_model.add(Dropout(0.25))

        self.surrogate_model.add(Flatten())
        self.surrogate_model.add(Dense(512))
        self.surrogate_model.add(Activation('relu'))
        self.surrogate_model.add(Dropout(0.5))
        self.surrogate_model.add(Dense(10))
        self.surrogate_model.add(Activation('softmax'))

        self.surrogate_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def attack(self, model=None, attack_str=""):
        imgs = self._load_images(attack_str)
        
        if type(imgs) != type(None) :
            print('\n{0} adversarial examples using {1} attack loaded...\n'.format(self.__dataset, self.__attack))
            return imgs

        if type(model) == type(None): 
            model = self.surrogate_model.fit(self.__data.x_train, self.__data.y_train, verbose=1, epochs=self.__epochs, batch_size=128)
            wrap = KerasClassifier((0., 1.), model = self.surrogate_model)
        else:
            wrap = KerasClassifier((0., 1.), model = model)
        
        if self.__attack == 'FGSM': 
            print('\nCrafting adversarial examples using FGSM attack...\n')
            fgsm = FastGradientMethod(wrap)

            if self.__data.dataset_name == 'MNIST':
                x_adv_images = fgsm.generate(x=self.__data.x_test[self.idx_adv][:self._length], eps = 0.2)
            else:
                x_adv_images = fgsm.generate(x=self.__data.x_test[self.idx_adv][:self._length], eps = 0.025)

            helpers.save_imgs_pkl(x_adv_images, self.__dataset.lower() + '_test_set_fgsm.pkl')
            return x_adv_images
        
        elif self.__attack.startswith("CW"):
            print('\nCrafting adversarial examples using CW attack...\n')
            cw = CarliniL2Method(wrap, confidence=0.0, targeted=False, binary_search_steps=3, learning_rate=0.5, initial_const=1, max_iter=100)
            x_adv_images = cw.generate(self.__data.x_test[self.idx_adv][:self._length])
            helpers.save_imgs_pkl(x_adv_images, self.__dataset.lower() + '_test_set_cw.pkl')
            return x_adv_images
            
        elif self.__attack == 'BIM':        
            print('\nCrafting adversarial examples using BIM attack...\n')

            if self.__dataset == 'MNIST':
                bim = BasicIterativeMethod(wrap, eps=0.2, eps_step=0.05, max_iter=50)
            if self.__dataset == 'CIFAR':
                bim = BasicIterativeMethod(wrap, eps=0.025, eps_step=0.01, max_iter=1000, norm=np.inf)

            x_adv_images = bim.generate(x = self.__data.x_test[self.idx_adv][:self._length])
            helpers.save_imgs_pkl(x_adv_images, self.__dataset.lower() + '_test_set_bim.pkl')
            return x_adv_images

        elif self.__attack == 'DEEPFOOL':
            print('\nCrafting adversarial examples using DeepFool attack...\n')
            
            deepfool = DeepFool(wrap)        
            x_adv_images = deepfool.generate(x = self.__data.x_test[self.idx_adv][:self._length])
            helpers.save_imgs_pkl(x_adv_images, self.__dataset.lower() + '_test_set_deepfool.pkl')
            return x_adv_images                

    def test_surrogate_model(self, x_adv_images, index):
        adv_x = x_adv_images[index].reshape(1, self.__image_rows, self.__image_cols, self.__channels)

        y_adv_probs = self.surrogate_model.predict(adv_x)
        y = np.argmax(y_adv_probs, axis=1)[0]

        leg_x = self.__data.x_test[index].reshape(1, self.__image_rows, self.__image_cols, self.__channels)
        y_leg_probs = self.surrogate_model.predict(leg_x)
        y_leg = np.argmax(y_leg_probs, axis=1)[0]
        
        print('\nLegitimate class of the adversarial sample: {0}\nClass of adversarial sample predicted by surrogate model: {1}\nClass of legitimate sample predicted by surrogate model: {2}\n'
                                .format(np.argmax(self.__data.y_test[index]), y, y_leg))
   

    def _train_surrogate_model(self, model):
        keras.backend.set_session(self.__sess)

        x = tf.placeholder(tf.float32, shape=(None, self.__image_rows, self.__image_cols, self.__channels))
        y = tf.placeholder(tf.float32, shape=(None, self.__nb_classes))

        preds = model(x)

        def evaluate():
            acc = model_eval(self.__sess, x, y, preds, self.__data.x_test, self.__data.y_test,
                            args={'batch_size': self.__batch})
            print('Test accuracy of surrogate model on legitimate examples: %0.4f' % acc)

        model_train(self.__sess, x, y, preds, X_train=self.__data.x_train, Y_train=self.__data.y_train,
                    evaluate=evaluate, args= {'nb_epochs': self.__epochs,
                                            'batch_size': self.__batch,
                                            'learning_rate': 0.001})

        wrap = KerasModelWrapper(model)
        return wrap
    
    def __evaluate_surrogate_model(self, x_test_adv, y_test=None):
        if type(None) == type(y_test):
            scores = self.surrogate_model.evaluate(x_test_adv, self.__data.y_test, verbose=0)
        else:
            scores = self.surrogate_model.evaluate(x_test_adv, y_test, verbose=0)
        print("Surrogate model's baseline error: %.2f%%" % (scores[1]*100))

    def _load_images(self, attack_str):
        if attack_str == "":
            imgs = helpers.load_imgs_pkl(self.__dataset.lower() + '_test_set_' + self.__attack.lower() + '.plk')
        else:
            imgs = helpers.load_imgs_pkl(self.__dataset.lower() + '_test_set_' + attack_str + '.plk')
        
        return imgs

    
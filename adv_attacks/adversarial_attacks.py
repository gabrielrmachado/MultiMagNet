import numpy as np
import keras
from keras import backend
import tensorflow as tf
import utils.helpers as helpers
import keras
import winsound

from cleverhans.utils_keras import KerasModelWrapper
from foolbox.attacks import DeepFoolL2Attack
from classifiers.classifier import Classifier
from cleverhans.utils_tf import model_train, model_eval
from nn_robust_attacks.l0_attack import CarliniL0
from nn_robust_attacks.l2_attack import CarliniL2
from nn_robust_attacks.li_attack import CarliniLi
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import DeepFool
from cleverhans.attacks import CarliniWagnerL2 as CW
from cleverhans.attacks import BasicIterativeMethod as BIM
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_keras import KerasModelWrapper
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
import classifiers

class Adversarial_Attack:
    def __init__(self, sess, data, attack='FGSM',
                 num_filters = 64, batch_size = 128, epochs = 10):
        self.__data = data
        self.__image_rows = data.x_train.shape[1]
        self.__image_cols = data.x_train.shape[2]
        self.__channels = data.x_train.shape[3]
        self.__nb_classes = data.y_train.shape[1]
        self.__attack = attack
        self.__sess = sess
        self.__batch = batch_size
        self.__epochs = epochs   
        self.__dataset = data.dataset_name 
        self.idx_adv = helpers.load_imgs_pkl('example_idx.pkl')

        self.surrogate_model = cnn_model(channels=self.__channels, img_rows=self.__image_rows, 	            
                        img_cols=self.__image_cols, nb_classes=self.__nb_classes)  

        self.surrogate_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def attack(self, model=None, logits=True, attack_str=""):
        imgs = self._load_images(attack_str)
        
        if type(imgs) != type(None) :
            print('\n{0} adversarial examples using {1} attack loaded...\n'.format(self.__dataset, self.__attack))
            return imgs

        if type(model) == type(None): 
            model = self._train_surrogate_model(self.surrogate_model)
        
        if self.__attack == 'FGSM': 
            print('\nCrafting adversarial examples using FGSM attack...\n')
            return self.__fgsm_attack(model)
        
        elif self.__attack.startswith("CW"):
            print('\nCrafting adversarial examples using CW attack...\n')
            return self.__cw_attack(model)
            
        elif self.__attack == 'BIM':        
            print('\nCrafting adversarial examples using BIM attack...\n')
            return self.__bim_attack(model)

        elif self.__attack == 'DEEPFOOL':
            print('\nCrafting adversarial examples using DeepFool attack...\n')
            return self.__deepfool_attack(model)      


    def __deepfool_attack(self, wrap):
        deepfool = DeepFool(wrap, sess=self.__sess)
        x_adv_images = deepfool.generate_np(self.__data.x_test[self.idx_adv][:500], over_shoot=0.02, max_iter=1000,
                                            nb_candidate=10)

        helpers.save_imgs_pkl(x_adv_images, self.__dataset.lower() + '_test_set_deepfool.pkl')
        return x_adv_images

    def __cw_attack(self, wrap):
        cw = CW(wrap, sess = self.__sess)

        x_adv_images = cw.generate_np(self.__data.x_test[self.idx_adv][:2000], y_target=None, 
                max_iterations=50, learning_rate=1e-2, confidence=20, binary_search_steps=3, initial_const=1e-3,
                abort_early=True)

        helpers.save_imgs_pkl(x_adv_images, self.__dataset.lower() + '_test_set_cw.pkl')
        return x_adv_images

    def __bim_attack(self, wrap):
        bim = BIM(wrap, sess = self.__sess)
        bim_params = {}

        if self.__dataset == 'MNIST':
            bim_params = {'eps': 0.15, 'eps_iter': 0.07, 'nb_iter': 50}
        if self.__dataset == 'CIFAR':
            bim_params = {'eps': 0.07, 'eps_iter': 0.03, 'nb_iter': 100}

        x_adv_images = bim.generate_np(self.__data.x_test[self.idx_adv][:2000], **bim_params)
        helpers.save_imgs_pkl(x_adv_images, self.__dataset.lower() + '_test_set_bim.pkl')
        return x_adv_images

    def __fgsm_attack(self, wrap):
        
        fgsm = FastGradientMethod(wrap, sess = self.__sess)
        x_adv_images = fgsm.generate_np(self.__data.x_test[self.idx_adv][:2000], eps = 0.05, clip_min = 0., clip_max = 1.)

        helpers.save_imgs_pkl(x_adv_images, self.__dataset.lower() + '_test_set_fgsm.pkl')
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
        print("Surrogate model's baseline error: %.2f%%" % (100-scores[1]*100))

    def _load_images(self, attack_str):
        if attack_str == "":
            imgs = helpers.load_imgs_pkl(self.__dataset.lower() + '_test_set_' + self.__attack.lower() + '.plk')
        else:
            imgs = helpers.load_imgs_pkl(self.__dataset.lower() + '_test_set_' + attack_str + '.plk')
        
        return imgs

    
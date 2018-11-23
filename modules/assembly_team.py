from team_techniques.sparse_autoencoder import TF_SAE as SPAE
from team_techniques.deep_autoencoder import TF_StackAE as STAE
from team_techniques.denoising_autoencoder_cifar import TF_DAE_CIFAR as DAE_CIFAR
from team_techniques.conv_autoencoder_mnist import TF_CAE_MNIST as CAE_MNIST
from team_techniques.conv_autoencoder_cifar import TF_CAE_CIFAR as CAE_CIFAR
from team_techniques.magnet_autoencoder_mnist import TF_DAE_MNIST as DAE_MNIST
from utils.helpers import JSD
from scipy.stats import entropy
from numpy.linalg import norm
from keras.models import Sequential
from keras.layers import Lambda
from keras.activations import softmax

import utils.helpers as helpers
from numpy import array
import numpy as np
import os

class Autoencoder_Params():
        def __init__(self, name, struct=None, model = "", activation="sigmoid", batch_size = 128, epochs=100, 
            noise=0.0, regularizer=0.0, compiler="adam", batch_normalization=False):
            self.name = name
            self.activation = activation
            self.model = model
            self.struct = struct
            self.batch_size = batch_size
            self.epochs = epochs
            self.noise = noise
            self.reg = regularizer
            self.compiler=compiler
            self.batch_norm = batch_normalization

class Assembly_Team():        
    def __init__(self, sess, data, number_team_members = 5):
        """
        This constructor creates the team R.

        # Attributes:
            data: the whole dataset
            number_team_members: the number of dimensionality reduction members which will compose the team.
        """
        self.__number = number_team_members
        self.__data = data

        if data.dataset_name == 'MNIST':
            m1 = Autoencoder_Params(model="DAE", name="MNIST_I", struct=[3, "average", 3], epochs=50, activation='sigmoid', noise=0.1)
            m2 = Autoencoder_Params(model="DAE", name="MNIST_II", struct=[3], epochs=50, activation='sigmoid', noise=0.1)
            m3 = Autoencoder_Params(model="DAE", name="mnist_dae1", struct=[3], epochs=50, activation='relu', noise=0.2)
            m4 = Autoencoder_Params(model="DAE", name="mnist_dae2", struct=[5, "max", 5], epochs=40, activation='sigmoid', noise=0.3)
            m5 = Autoencoder_Params(model="DAE", name="mnist_dae3", struct=[3, "average", 3], epochs=55, activation='sigmoid', noise=0.1)
            m6 = Autoencoder_Params(model="DAE", name="mnist_dae4", struct=[5], epochs=50, batch_size=128, noise=0.2)
            m7 = Autoencoder_Params(model="DAE", name="mnist_dae5", struct=[4, "max", 4], epochs=50, batch_size=128, noise=0.1)
            m8 = Autoencoder_Params(model="DAE", name="mnist_dae6", struct=[3, "max", 3], epochs=35, activation='sigmoid', noise=0.3)
            m9 = Autoencoder_Params(model="DAE", name="mnist_dae7", struct=[2, "max", 2], epochs=30, activation='sigmoid', noise=0.1)
            m10 = Autoencoder_Params(model="DAE", name="mnist_dae8", struct=[5, "average", 5], epochs=35, activation='sigmoid', noise=0.2)

        elif data.dataset_name == 'CIFAR':
            m1 = Autoencoder_Params(name='cifar_dae1',batch_size=32, epochs=30,struct=[16,"max",32,"max",32], 
                            regularizer=1e-9)

            m2 = Autoencoder_Params(name='cifar_dae2',batch_size=32, epochs=30,struct=[32,"max",32], 
                            regularizer=1e-9, batch_normalization=True)

            m3 = Autoencoder_Params(name='cifar_dae3',batch_size=32, epochs=30,struct=[16,"max",32,"max",32])

            m4 = Autoencoder_Params(name='cifar_dae4',batch_size=32, epochs=30,struct=[32,"max",32], 
                            batch_normalization=True)

            m5 = Autoencoder_Params(name='cifar_dae5',batch_size=32, epochs=30,struct=[32,"average",32], 
                            batch_normalization=True)

            m6 = Autoencoder_Params(name='cifar_dae6',batch_size=32, epochs=30,struct=[8,"max",16], 
                            regularizer=1e-9)

            m7 = Autoencoder_Params(name='cifar_dae7',batch_size=32, epochs=30,struct=[16,"average",16], 
                            regularizer=1e-9, batch_normalization=True)

            m8 = Autoencoder_Params(name='cifar_dae8',batch_size=36, epochs=30,
                            struct=[16,"max",24])

            m9 = Autoencoder_Params(name='cifar_dae9',batch_size=36, epochs=30,struct=[32,"average",32], 
                            batch_normalization=True, regularizer=1e-9)

            m10 = Autoencoder_Params(name='cifar_dae10',batch_size=36, epochs=30,struct=[16,"max",32], 
                            batch_normalization=True, regularizer=1e-9)

        self.repository = [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10]

    def get_team(self, number=0):
        s = array(self.repository)
        if (self.__number > s.size):
            raise Exception("Number_team_members: {0} is bigger than the number of avaiable models into repository: {1}."
                                           .format(self.__number, s.size))

        elif number > 0:
            print('\nTotal members into repository: {0}\nNumber of members chosen: {1}'.format(s.size, self.__number))
            team = np.random.choice(s, size=number, replace=False)
        else: 
            print('\nTotal members into repository: {0}\nNumber of members chosen: {1}'.format(s.size, self.__number))
            team = np.random.choice(s, size=self.__number, replace=False)
        return team

    def get_thresholds_pd(self, classifier, drop_rate=0.001, T = 10, p = 2, tau="RE", plot_rec_images=False, metric='JSD'):
        """
        Predicts the 'data' using the selected autoencoders, 
        returning their respective probability divergence thresholds.

        """
        self.team = self.get_team()
        thresholds = []
        num = round(drop_rate * len(self.__data.x_val))
        model = helpers.get_logits(classifier.model)
        sft = Sequential()
        sft.add(Lambda(lambda X: softmax(X, axis=1), input_shape=(10,)))

        for i in range(self.team.size):
            # load pre-computed autoencoder threshold (if it exists)
            path = os.path.join("./team_techniques/models/thresholds", self.team[i].name + "_" + metric + "_.plk")
            try:
                threshold = helpers.load_pkl(path)
                print("Threshold of autoencoder {0} loaded.".format(self.team[i].name))

            except:
                autoencoder = self.load_autoencoder(self.team[i], metric)
                rec = autoencoder.predict(self.__data.x_val)

                if plot_rec_images == True:
                    rec_ex = autoencoder.predict(self.__data.x_test[:10])
                    helpers.plot_images(self.__data.x_test[:10], rec_ex[:10], rec_ex.shape)
                    del rec_ex

                print('Reconstructing images using {0} model ({1}/{2}).'.format(self.team[i].name, i+1, self.team.size))

                if self.__data.x_val.shape[1:] != rec.shape[1:]:
                    rec = rec.reshape(rec.shape[0], self.__data.x_val.shape[1], self.__data.x_val.shape[2], self.__data.x_val.shape[3]).astype('float32')

                # marks = np.mean(np.power(np.abs(model.predict(self.__data.x_val) - model.predict(rec)), 1), axis=1)

                oc = sft.predict(model.predict(self.__data.x_val)/T)
                rc = sft.predict(model.predict(rec)/T)        

                # print("OC[0]: {0}\nRC[0]: {1}".format(oc[0], rc[0]))
                # print(oc.shape, rc.shape)

                if metric == 'JSD':    
                    marks = [JSD(oc[j], rc[j]) for j in range(len(rc))]                   
                
                elif metric == 'DKL':
                    from scipy.stats import entropy
                    marks = [entropy(pk=rc[j], qk=oc[j]) for j in range(len(rc))]                   

                marks_iset = np.sort(marks)
                threshold = marks_iset[-num]

                path = os.path.join("./team_techniques/models/thresholds", self.team[i].name + "_" + metric + "_.plk")
                try:
                    helpers.save_pkl(threshold, path)
                except:
                    print("It was not possible to save {0} autoencoder threshold.".format(self.team[i].name))
                del autoencoder

            thresholds.append(threshold)
        
        if tau == "minRE":
            thresholds = [np.min(thresholds)] * self.__number

        return thresholds

    def get_thresholds(self, drop_rate=0.001, p = 2, tau="RE", plot_rec_images=False):
        """
        Predicts the 'data' using the selected autoencoders, 
        returning their respective reconstruction errors thresholds.

        """
        s = array(self.repository)
        if (self.__number > s.size):
            raise Exception("Number_team_members: {0} is bigger than the number of avaiable models into repository: {1}."
                                            .format(self.__number, s.size))
        else: 
            print('\nTotal members into repository: {0}\nNumber of members chosen: {1}'.format(s.size, self.__number))
            self.team = np.random.choice(s, size=self.__number, replace=False)

        thresholds = []
        num = round(drop_rate * len(self.__data.x_val))

        for i in range(self.team.size):
            # load pre-computed autoencoder threshold (if it exists)
            path = os.path.join("./team_techniques/models/thresholds", self.team[i].name + "_" + metric + "_.plk")
            try:
                threshold = helpers.load_pkl(path)
                print("Threshold of autoencoder {0} loaded.".format(self.team[i].name))

            except:
                autoencoder = self.load_autoencoder(self.team[i], metric)
                rec = autoencoder.predict(self.__data.x_val)

                if plot_rec_images == True:
                    rec_ex = autoencoder.predict(self.__data.x_test[:10])
                    helpers.plot_images(self.__data.x_test[:10], rec_ex[:10], rec_ex.shape)

                print('Reconstructing images using {0} model ({1}/{2}).'.format(self.team[i].name, i+1, self.team.size))

                if self.__data.x_val.shape[1:] != rec.shape[1:]:
                    rec = rec.reshape(rec.shape[0], self.__data.x_val.shape[1], self.__data.x_val.shape[2], self.__data.x_val.shape[3]).astype('float32')

                diff = np.abs(self.__data.x_val - rec)
                marks = np.mean(np.power(diff, p), axis=(1,2,3))

                marks_iset = np.sort(marks)
                threshold = marks_iset[-num]
                path = os.path.join("./team_techniques/models/thresholds", self.team[i].name + "_" + "RE" + "_.plk")
                try:
                    helpers.save_pkl(threshold, path)
                except:
                    print("It was not possible to save {0} autoencoder threshold.".format(self.team[i].name))
                del autoencoder

            thresholds.append(threshold)
        
        if tau == "minRE":
            thresholds = [np.min(thresholds)] * self.__number

        return thresholds

    # def optimize_team_parameters(attack):

        # _, x, y, y_ori = helpers.join_test_sets(self._data, x_test_adv, length, idx=self._idx_adv[:length])

    def load_autoencoder(self, team_member, metric=""):
        model = team_member.model

        if self.__data.dataset_name == "MNIST":
            if model == "DAE":
                autoencoder = DAE_MNIST(self.__data, name=team_member.name, structure=team_member.struct, epochs=team_member.epochs, activation=team_member.activation, v_noise=team_member.noise)
            elif model == "CAE":
                autoencoder = CAE_MNIST(self.__data, name=team_member.name, structure=team_member.struct, epochs=team_member.epochs, batch_size=team_member.batch_size)
        else:
            autoencoder = DAE_CIFAR(self.__data, name=team_member.name, epochs=team_member.epochs,
                    batch_size=team_member.batch_size, noise_factor=team_member.noise, reg=team_member.reg,
                    structure=team_member.struct, compiler=team_member.compiler, 
                    batch_norm=team_member.batch_norm)
        
        print("\nLoading {0} autoencoder".format(team_member.name))
        autoencoder.execute()            
        return autoencoder
        
        

        
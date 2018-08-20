from team_techniques.sparse_autoencoder import TF_SAE as SPAE
from team_techniques.deep_autoencoder import TF_StackAE as STAE
from team_techniques.denoising_autoencoder_mnist import TF_DAE_MNIST as DAE_MNIST
from team_techniques.denoising_autoencoder_cifar import TF_DAE_CIFAR as DAE_CIFAR
from team_techniques.conv_autoencoder_mnist import TF_CAE_MNIST as CAE_MNIST
from team_techniques.conv_autoencoder_cifar import TF_CAE_CIFAR as CAE_CIFAR
from team_techniques.magnet_autoencoder_mnist import TF_DAE_MagNet as DAE_MagNet

import utils.helpers as helpers
from numpy import array
import numpy as np

class Assembly_Team():    
    class AuxData():
        def __init__(self, x_train, y_train, x_test, y_test):
            self.x_train = x_train
            self.x_test = x_test
            self.y_train = y_train
            self.y_test = y_test
    
    def __init__(self, sess, data, number_team_members = 5):
        """
        This constructor creates the team R.

        # Attributes:
            data: the whole dataset
            number_team_members: the number of dimensionality reduction members which will compose the team.
        """
        self.__number = number_team_members
        self.__data = data
        self.__team = []
        
        x_t = helpers.reshape_flatten(data.x_train)
        x_te = helpers.reshape_flatten(data.x_test)
        y_t = data.y_train
        y_te = data.y_test

        aux_data = self.AuxData(x_t, y_t, x_te, y_te)

        if data.dataset_name == 'MNIST':
            # spae = SPAE(aux_data, aux_data.x_train.shape[1])
            # self.__team.append(spae)

            # spae2 = SPAE(aux_data, aux_data.x_train.shape[1], batch_size=128, encoded_dim = 40, regularizer=10e-8)
            # self.__team.append(spae2)

            # stae = STAE(aux_data, aux_data.x_train.shape[1], layers=[512, 256, 128, 64])
            # self.__team.append(stae)

            # stae2 = STAE(aux_data, aux_data.x_train.shape[1], batch_size=128, layers=[512, 256, 128, 80], encoded_dim = 80)
            # self.__team.append(stae2)

            # magnet_i = DAE_MagNet(data, "MNIST_I", [3, "average", 3], epochs=100, activation = 'sigmoid', v_noise=0.1)
            # self.__team.append(magnet_i)

            # magnet_ii = DAE_MagNet(data, "MNIST_II", [3], epochs=50, activation = 'sigmoid', v_noise=0.1)
            # self.__team.append(magnet_ii)

            mag_dae3 = DAE_MagNet(data, "magnet_dae3", [3], epochs=50, activation = 'relu', v_noise=0.2)
            self.__team.append(mag_dae3)

            mag_dae2 = DAE_MagNet(data, "magnet_dae2", [5, "max", 5], epochs=100, activation = 'sigmoid', v_noise=0.3)
            self.__team.append(mag_dae2)

            mag_dae = DAE_MagNet(data, "magnet_dae", [3, "average", 3], epochs=100, activation = 'sigmoid', v_noise=0.1)
            self.__team.append(mag_dae)

            dae = DAE_MNIST(data, name='mnist_dae_opt1', epochs=7)
            self.__team.append(dae)

            dae2 = DAE_MNIST(data, name='mnist_dae_opt2', batch_size=256, noise_factor = 0.1)
            self.__team.append(dae2)            

            cae = CAE_MNIST(data, name='mnist_cae_opt1', batch_size=32, epochs=2)
            self.__team.append(cae)

            cae2 = CAE_MNIST(data, name='mnist_cae_opt2', batch_size=100, epochs=2)
            self.__team.append(cae2)            

            cae3 = CAE_MNIST(data, opt=2, name='mnist_cae_opt3', batch_size=64, epochs=5)
            self.__team.append(cae3)

            cae4 = CAE_MNIST(data, name='mnist_cae4_opt1', batch_size=128, epochs=5)
            self.__team.append(cae4)

            cae5 = CAE_MNIST(data, name='mnist_cae5_opt2', batch_size=128, epochs=5)
            self.__team.append(cae5) 
        
        if data.dataset_name == 'CIFAR':
            cae = CAE_CIFAR(data, name='cifar_cae1_opt1', batch_size=32, epochs=30)
            self.__team.append(cae)

            cae2 = CAE_CIFAR(data, name='cifar_cae2_opt1', batch_size=32, epochs=30)
            self.__team.append(cae2)

            cae3 = CAE_CIFAR(data, name='cifar_cae3_opt2', opt=2, batch_size=64, epochs=20)
            self.__team.append(cae3)

            dae = DAE_CIFAR(data, name='cifar_dae1_opt1', epochs=20, noise_factor=0.1)
            self.__team.append(dae)

            dae2 = DAE_CIFAR(data, name='cifar_dae1_opt2', opt=2, epochs=20, noise_factor=0.1)
            self.__team.append(dae2)

            dae3 = DAE_CIFAR(data, name='cifar_dae1_opt3', opt=3, epochs=20, noise_factor=0.1)
            self.__team.append(dae3)

            dae4 = DAE_CIFAR(data, name='cifar_dae1_opt4', opt=4, epochs=20, noise_factor=0.1)
            self.__team.append(dae4)

            dae5 = DAE_CIFAR(data, name='cifar_dae2_opt3', opt=3, epochs=30, noise_factor=0.1, batch_size=64)
            self.__team.append(dae5)

            dae6 = DAE_CIFAR(data, name='cifar_dae2_opt4', opt=4, epochs=30, noise_factor=0.1, batch_size=64)
            self.__team.append(dae6)

            cae4 = CAE_CIFAR(data, name='cifar_cae4_opt2', opt=2, batch_size=128, epochs=20)
            self.__team.append(cae4)


    def train_and_choose_team(self, plot_rec_images=False):
        """
        This method returns the created team R with its autoencoders randomly selected.
        """
        s = array(self.__team)
        if (self.__number > s.size):
            raise Exception("number_team_members: {0} is bigger than the number of avaiable models into repository: {1}."
                                            .format(self.__number, s.size))
        else: 
            print('\nTotal members into repository: {0}\nNumber of members chosen: {1}'.format(s.size, self.__number))
            self.r = np.random.choice(s, size=self.__number, replace=False)
            for i in range(self.r.size):
                print('\nTraining {0} model ({1}/{2})'.format(self.r[i].name, i+1, self.r.size))
                self.r[i].execute()

                if plot_rec_images == True:
                    rec = self.r[i].predict(self.__data.x_test[:100], False)
                    helpers.plot_images(self.__data.x_test[:100], rec, rec.shape)

    def get_thresholds(self, data, drop_rate=0.001, p = 2, tau="RE", plot_rec_images=False):
        """
        Predicts the 'data' using the selected autoencoders, 
        returning their respective reconstruction errors thresholds.

        """
        self.train_and_choose_team(plot_rec_images)
        thresholds = []
        num = round(drop_rate * len(data))

        for i in range(self.__number):
            print('Reconstructing images using {0} model ({1}/{2}).'.format(self.r[i].name, i+1, self.r.size))
            rec_set = self.r[i].predict(data)

            if data.shape[1:] != rec_set.shape[1:]:
                rec_set = rec_set.reshape(rec_set.shape[0], data.shape[1], data.shape[2], data.shape[3]).astype('float32')

            diff = np.abs(data - rec_set)
            marks = np.mean(np.power(diff, p), axis=(1,2,3))

            marks_iset = np.sort(marks)
            thresholds.append(marks_iset[-num])
        
        if tau == "RE":
            return thresholds
        elif tau == "minRE":
            return [np.min(thresholds)] * self.__number
            
        

        
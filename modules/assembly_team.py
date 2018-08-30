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

class Autoencoder_Params():
        def __init__(self, model, name, struct=None, activation=None, opt = 1, batch_size = 128, epochs=100, noise=0.0):
            self.model = model
            self.name = name
            self.struct = struct
            self.activation = activation
            self.opt = opt
            self.batch_size = batch_size
            self.epochs = epochs
            self.noise = noise

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
            m1 = Autoencoder_Params(model="DAE_MagNet", name="MNIST_I", struct=[3, "average", 3], epochs=100, activation='sigmoid', noise=0.1)
            m2 = Autoencoder_Params(model="DAE_MagNet", name="MNIST_II", struct=[3], epochs=50, activation='sigmoid', noise=0.1)
            m3 = Autoencoder_Params(model="DAE_MagNet", name="magnet_dae3", struct=[3], epochs=50, activation='relu', noise=0.2)
            m4 = Autoencoder_Params(model="DAE_MagNet", name="magnet_dae2", struct=[5, "max", 5], epochs=100, activation='sigmoid', noise=0.3)
            m5 = Autoencoder_Params(model="DAE_MagNet", name="magnet_dae", struct=[3, "average", 3], epochs=100, activation='sigmoid', noise=0.1)
            m6 = Autoencoder_Params(model="DAE", name="mnist_dae_opt1", epochs=7, batch_size=256, noise=0.5)
            m7 = Autoencoder_Params(model="DAE", name="mnist_dae_opt2", epochs=10, batch_size=256, noise=0.1)
            m8 = Autoencoder_Params(model="CAE", name="mnist_cae_opt1", opt=1, epochs=2, batch_size=32)
            m9 = Autoencoder_Params(model="CAE", name="mnist_cae_opt2", opt=1, epochs=2, batch_size=100)
            m10 = Autoencoder_Params(model="CAE", name="mnist_cae_opt3", opt=2, batch_size=64, epochs=5)

        elif data.dataset_name == 'CIFAR':
            m1 = Autoencoder_Params(model="CAE", name="cifar_cae1_opt1", opt=1, batch_size=32, epochs=30)
            m2 = Autoencoder_Params(model="CAE", name="cifar_cae2_opt1", opt=1, batch_size=32, epochs=20)
            m3 = Autoencoder_Params(model="CAE", name="cifar_cae3_opt2", opt=2, batch_size=64, epochs=20)
            m4 = Autoencoder_Params(model="DAE", name="cifar_dae1_opt1", opt=1, batch_size=128, epochs=20, noise=0.1)
            m5 = Autoencoder_Params(model="DAE", name="cifar_dae1_opt2", opt=2, epochs=20, batch_size=128, noise=0.1)
            m6 = Autoencoder_Params(model="DAE", name="cifar_dae1_opt3", opt=3, epochs=20, batch_size=128, noise=0.1)
            m7 = Autoencoder_Params(model="DAE", name="cifar_dae1_opt4", opt=4, epochs=20, batch_size=128, noise=0.1)
            m8 = Autoencoder_Params(model="DAE", name="cifar_dae2_opt3", opt=3, epochs=30, batch_size=64, noise=0.1)
            m9 = Autoencoder_Params(model="DAE", name="cifar_dae2_opt4", opt=4, epochs=30, batch_size=64, noise=0.1)
            m10 = Autoencoder_Params(model="CAE", name="cifar_cae4_opt2", opt=2, batch_size=128, epochs=20)

        self.repository = [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10]

        # if data.dataset_name == 'MNIST':
            #magnet_i = DAE_MagNet(data, "MNIST_I", [3, "average", 3], epochs=100, activation = 'sigmoid', v_noise=0.1)
            # self.__team.append(magnet_i)

            # magnet_ii = DAE_MagNet(data,"MNIST_II", [3], epochs=50, activation = 'sigmoid', v_noise=0.1)
            # self.__team.append(magnet_ii)

            # mag_dae3 = DAE_MagNet(data, "magnet_dae3", [3], epochs=50, activation = 'relu', v_noise=0.2)
            # self.__team.append(mag_dae3)

            # mag_dae2 = DAE_MagNet(data, "magnet_dae2", [5, "max", 5], epochs=100, activation = 'sigmoid', v_noise=0.3)
            # self.__team.append(mag_dae2)

            # mag_dae = DAE_MagNet(data, "magnet_dae", [3, "average", 3], epochs=100, activation = 'sigmoid', v_noise=0.1)
            # self.__team.append(mag_dae)

            # dae = DAE_MNIST(data, name='mnist_dae_opt1', epochs=7)
            # self.__team.append(dae)

            # dae2 = DAE_MNIST(data, name='mnist_dae_opt2', batch_size=256, noise_factor = 0.1)
            # self.__team.append(dae2)            

            # cae = CAE_MNIST(data, name='mnist_cae_opt1', batch_size=32, epochs=2)
            # self.__team.append(cae)

            # cae2 = CAE_MNIST(data, name='mnist_cae_opt2', batch_size=100, epochs=2)
            # self.__team.append(cae2)            

            # cae3 = CAE_MNIST(data, opt=2, name='mnist_cae_opt3', batch_size=64, epochs=5)
            # self.__team.append(cae3)

            # cae4 = CAE_MNIST(data, name='mnist_cae4_opt1', batch_size=128, epochs=5)
            # self.__team.append(cae4)

            # cae5 = CAE_MNIST(data, name='mnist_cae5_opt2', batch_size=128, epochs=5)
            # self.__team.append(cae5) 
        
        # if data.dataset_name == 'CIFAR':
            #cae = CAE_CIFAR(data, name='cifar_cae1_opt1', batch_size=32, epochs=30)
            # self.__team.append(cae)

            # cae2 = CAE_CIFAR(data, name='cifar_cae2_opt1', batch_size=32, epochs=30)
            # self.__team.append(cae2)

            # cae3 = CAE_CIFAR(data, name='cifar_cae3_opt2', opt=2, batch_size=64, epochs=20)
            # self.__team.append(cae3)

            # dae = DAE_CIFAR(data, name='cifar_dae1_opt1', epochs=20, noise_factor=0.1)
            # self.__team.append(dae)

            # dae2 = DAE_CIFAR(data, name='cifar_dae1_opt2', opt=2, epochs=20, noise_factor=0.1)
            # self.__team.append(dae2)

            # dae3 = DAE_CIFAR(data, name='cifar_dae1_opt3', opt=3, epochs=20, noise_factor=0.1)
            # self.__team.append(dae3)

            # dae4 = DAE_CIFAR(data, name='cifar_dae1_opt4', opt=4, epochs=20, noise_factor=0.1)
            # self.__team.append(dae4)

            # dae5 = DAE_CIFAR(data, name='cifar_dae2_opt3', opt=3, epochs=30, noise_factor=0.1, batch_size=64)
            # self.__team.append(dae5)

            # dae6 = DAE_CIFAR(data, name='cifar_dae2_opt4', opt=4, epochs=30, noise_factor=0.1, batch_size=64)
            # self.__team.append(dae6)

            # cae4 = CAE_CIFAR(data, name='cifar_cae4_opt2', opt=2, batch_size=128, epochs=20)
            # self.__team.append(cae4)

    def get_thresholds(self, data, drop_rate=0.001, p = 2, tau="RE", plot_rec_images=False):
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
        num = round(drop_rate * len(data))

        for i in range(self.team.size):
            autoencoder = self.load_autoencoder(self.team[i])
            rec = autoencoder.predict(data)

            if plot_rec_images == True:
                rec_ex = autoencoder.predict(self.__data.x_test[:10])
                helpers.plot_images(self.__data.x_test[:10], rec_ex[:10], rec_ex.shape)

            print('Reconstructing images using {0} model ({1}/{2}).'.format(self.team[i].name, i+1, self.team.size))

            if data.shape[1:] != rec.shape[1:]:
                rec = rec.reshape(rec.shape[0], data.shape[1], data.shape[2], data.shape[3]).astype('float32')

            diff = np.abs(data - rec)
            marks = np.mean(np.power(diff, p), axis=(1,2,3))

            marks_iset = np.sort(marks)
            thresholds.append(marks_iset[-num])
            del autoencoder
        
        if tau == "minRE":
            thresholds = [np.min(thresholds)] * self.__number

        return thresholds

        # for i in range(self.__number):
        #     print('Reconstructing images using {0} model ({1}/{2}).'.format(self.r[i].name, i+1, self.r.size))
        #     rec_set = self.r[i].predict(data)

        #     if data.shape[1:] != rec_set.shape[1:]:
        #         rec_set = rec_set.reshape(rec_set.shape[0], data.shape[1], data.shape[2], data.shape[3]).astype('float32')

        #     diff = np.abs(data - rec_set)
        #     marks = np.mean(np.power(diff, p), axis=(1,2,3))

        #     marks_iset = np.sort(marks)
        #     thresholds.append(marks_iset[-num])
        
        # if tau == "RE":
        #     return thresholds
        # elif tau == "minRE":
        #     return [np.min(thresholds)] * self.__number

    def load_autoencoder(self, team_member):
        model = team_member.model

        if self.__data.dataset_name == "MNIST":
            if model == "DAE_MagNet":
                autoencoder = DAE_MagNet(self.__data, name=team_member.name, structure=team_member.struct, epochs=team_member.epochs, activation=team_member.activation, v_noise=team_member.noise)
            elif model == "DAE":
                autoencoder = DAE_MNIST(self.__data, name=team_member.name, epochs=team_member.epochs, batch_size=team_member.batch_size, noise_factor=team_member.noise)
            elif model == "CAE":
                autoencoder = CAE_MNIST(self.__data, name=team_member.name, opt=team_member.opt, epochs=team_member.epochs, batch_size=team_member.batch_size)
        else:
            if model == "DAE":
                autoencoder = DAE_CIFAR(self.__data, name=team_member.name, opt=team_member.opt, epochs=team_member.epochs, batch_size=team_member.batch_size, noise_factor=team_member.noise)
            elif model == "CAE":
                autoencoder = CAE_CIFAR(self.__data, name=team_member.name, opt=team_member.opt, epochs=team_member.epochs, batch_size=team_member.batch_size)
        
        autoencoder.execute()
        return autoencoder
            
        

        
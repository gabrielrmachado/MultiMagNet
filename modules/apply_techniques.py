import random
import numpy as np
import tensorflow as tf
import utils.helpers as helpers
from scipy.stats import entropy
from numpy.linalg import norm
from keras.models import Sequential
from keras.layers import Lambda
from keras.activations import softmax
from utils.helpers import JSD

class Image_Reduction:
    @staticmethod
    def apply_techniques_encoder(i_set, team_obj):
        ir_set = []        
        for i in range(len(team_obj.team)):
            autoencoder = team_obj.load_autoencoder(team_obj.team[i])
            print('Reducing images using {0} model ({1}/{2}).'.format(team_obj.team[i].name, i+1, team_obj.team.size))
            k_ir = autoencoder.predict(i_set)
            ir_set.append(k_ir)
            print('IR set shape after reduction process using {0} model ({1}/{2}).'.format(team_obj.team[i].name, i+1, team_obj.team.size))

        print('\nReduction process finished.\n')        
        return ir_set

    @staticmethod
    def apply_techniques(x, team_obj, p = 2):
        """
        Apply reduction team members on input 'x' and returns 'x' reconstruction error on each model.
        """
        x_marks = []

        for i in range(len(team_obj.team)):
            autoencoder = team_obj.load_autoencoder(team_obj.team[i])
            print('Reconstructing test images using {0} model ({1}/{2}).'.format(autoencoder.name, i+1, team_obj.team.size))
            rec = autoencoder.predict(x)

            if x.shape[1:] != rec.shape[1:]:
                rec = rec.reshape(rec.shape[0], x.shape[1], x.shape[2], x.shape[3]).astype('float32')
           
            diff = np.abs(x - rec)
            x_mark = np.mean(np.power(diff, p), axis=(1,2,3))
            x_marks.append(x_mark)
            
            del autoencoder

        return x_marks

    @staticmethod
    def apply_techniques_pd(x, team_obj, classifier, T=10, p=1, metric='JSD'):
        """
        Apply reduction team members on input 'x' and returns the marks computed using JSD divergence.
        """
        x_marks = []
        model = helpers.get_logits(classifier.model)
        sft = Sequential()
        sft.add(Lambda(lambda X: softmax(X, axis=0), input_shape=(10,)))

        for i in range(len(team_obj.team)):
            autoencoder = team_obj.load_autoencoder(team_obj.team[i], metric)
            print('Reconstructing test images using {0} model ({1}/{2}).'.format(autoencoder.name, i+1, team_obj.team.size))
            rec = autoencoder.predict(x)

            if x.shape[1:] != rec.shape[1:]:
                rec = rec.reshape(rec.shape[0], x.shape[1], x.shape[2], x.shape[3]).astype('float32')

            #marks = np.mean(np.power(np.abs(model.predict(x) - model.predict(rec)), 1), axis=(1,2,3))
                
            oc = sft.predict(model.predict(x)/T)
            rc = sft.predict(model.predict(rec)/T)

            # print("OC[0]: {0}\nRC[0]: {1}".format(oc[0], rc[0]))
            # print(oc.shape, rc.shape)

            if metric == 'JSD':
                marks = [JSD(oc[j], rc[j]) for j in range(len(rc))]     
            elif metric == 'DKL':
                from scipy.stats import entropy
                marks = [entropy(pk=rc[j], qk=oc[j]) for j in range(len(rc))]
            # print("X_MARKS: \n{0}".format(marks))  
            x_marks.append(marks)

            del autoencoder
        return x_marks    
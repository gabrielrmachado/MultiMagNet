import random
import numpy as np
import tensorflow as tf
import utils.helpers as helpers
from scipy.stats import entropy
from numpy.linalg import norm
from keras.models import Sequential
from keras.layers import Lambda
from keras.activations import softmax

def JSD(P, Q):
        _P = P / norm(P, ord=1)
        _Q = Q / norm(Q, ord=1)
        _M = 0.5 * (_P + _Q)
        return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

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
            print('Reconstructing images using {0} model ({1}/{2}).'.format(autoencoder.name, i+1, team_obj.team.size))
            rec = autoencoder.predict(x)

            if x.shape[1:] != rec.shape[1:]:
                rec = rec.reshape(rec.shape[0], x.shape[1], x.shape[2], x.shape[3]).astype('float32')
           
            diff = np.abs(x - rec)
            x_mark = np.mean(np.power(diff, p), axis=(1,2,3))
            x_marks.append(x_mark)
            
            del autoencoder

        return x_marks

    @staticmethod
    def apply_techniques_jsd(x, team_obj, classifier, T=10, p=2):
        """
        Apply reduction team members on input 'x' and returns the marks computed using JSD divergence.
        """
        x_marks = []
        model = classifier.get_model(logits=True)
        sft = Sequential()
        sft.add(Lambda(lambda X: softmax(X, axis=1), input_shape=(10,)))

        for i in range(len(team_obj.team)):
            autoencoder = team_obj.load_autoencoder(team_obj.team[i])
            print('Reconstructing images using {0} model ({1}/{2}).'.format(autoencoder.name, i+1, team_obj.team.size))
            rec = autoencoder.predict(x)

            if x.shape[1:] != rec.shape[1:]:
                rec = rec.reshape(rec.shape[0], x.shape[1], x.shape[2], x.shape[3]).astype('float32')
                
                oc = sft.predict(model.predict(x)/T)
                rc = sft.predict(model.predict(rec)/T)

                marks = [(JSD(oc[i], rc[i])) for i in range(len(rc))]
                x_marks.append(marks)

                del autoencoder
            return x_marks    
import random
import numpy as np
import tensorflow as tf
import utils.helpers as helpers

class Image_Reduction:
    @staticmethod
    def apply_techniques_encoder(i_set, r_team):
        ir_set = []
        r = np.random.choice(r_team, size=r_team.size, replace=False)
        
        for i in range(len(r)):
            print('Reducing images using {0} model ({1}/{2}).'.format(r[i].__class__.__name__, i+1, r.size))
            k_ir = r[i].predict(i_set)
            ir_set.append(k_ir)
            print('IR set shape after reduction process using {0} model ({1}/{2}): {3}.'.format(r[i].__class__.__name__, i+1, r.size, ir_set[i].shape))

        print('\nReduction process finished.\n')        
        return ir_set

    @staticmethod
    def apply_techniques(x, r_team, p = 2):
        """
        Apply reduction team members on input 'x' and returns 'x' reconstruction error on each model.
        """
        x_marks = []

        for i in range(len(r_team)):
            print("Reconstructing 'x' using {0} model ({1}/{2}).".format(r_team[i].__class__.__name__, i+1, r_team.size))
            rec_set = r_team[i].predict(x)

            if x.shape[1:] != rec_set.shape[1:]:
                rec_set = rec_set.reshape(rec_set.shape[0], x.shape[1], x.shape[2], x.shape[3]).astype('float32')

            diff = np.abs(x - rec_set)
            x_mark = np.mean(np.power(diff, p), axis=(1,2,3))
            x_marks.append(x_mark)

        return x_marks
                
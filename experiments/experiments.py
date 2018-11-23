import tensorflow as tf
import cleverhans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import utils.helpers as helpers
import random

from utils.data import Data
from modules.retrieve_module import Retrieval
from modules.assembly_team import Assembly_Team
from modules.poll_votes import poll_votes as poll_votes
from modules.reformer import Reformer
from modules.poll_votes import poll_votes_each_x as poll_votes_each_x
from classifiers.classifier import Classifier
from adv_attacks.adversarial_attacks import Adversarial_Attack
from modules.apply_techniques import Image_Reduction
from keras.utils import np_utils
import time
import os
from utils.helpers import JSD
from datetime import timedelta
from sklearn.metrics import confusion_matrix

class Experiment:
    def __init__(self, dataset):
        """
        Realizes the MultiMagNet's experiments.

        # Attribute
            dataset: 'MNIST' or 'CIFAR'
        """
        path = os.path.join("./adv_attacks/adversarial_images/example_idx.pkl")
        self._idx_adv = helpers.load_pkl(path)

        self._sess = tf.Session()
        self._sess.as_default()              

        self._data = Data(dataset_name=dataset) 
        print("\nDataset loaded.")

    def create_adversarial_validation_images(self):
        classifier = Classifier(self._sess, self._data, epochs=350, learning_rate=0.01, batch_size=32)
        classifier.execute()
        length = 2000
        # # Creates surrogate model and returns the perturbed NumPy test set  
        x_val_adv = Adversarial_Attack(self._sess, self._data, dataset = "_x_val_set_", length=2000, attack="DEEPFOOL", epochs=12).attack(model=classifier.model)
        scores_leg = classifier.model.evaluate(self._data.x_val[self._idx_adv][:length], self._data.y_val[self._idx_adv][:length], verbose=1)
        scores = classifier.model.evaluate(x_val_adv[:length], self._data.y_val[self._idx_adv][:length], verbose=1)
        print("\nMain classifier's accuracy on legitimate examples: %.2f%%" % (scores_leg[1]*100))
        print("\nMain classifier's accuracy on adversarial examples: %.2f%%" % (scores[1]*100))

        helpers.plot_images(self._data.x_val[self._idx_adv][:length], x_val_adv[:length], x_val_adv.shape)

    def all_cases_experiment(self, *args, length=2000):
        """
        Creates an cartesian product with '*args' in order to make the experiments on several different scenarios. 
        All the experiments' results are saved in a .TXT file called 'all_cases_experiment.txt'

        # Attributes:
            *args: each '*args' parameter is a list containing all possible MultiMagNet's parameters: 
                NUMBER_EXPERIMENTS: how many times the code will run.
                DATASETS: ("MNIST" or "CIFAR"),
                ATTACKS: ("FGSM", "BIM", "DEEPFOOL", "CW_0.0"),
                DROP_RATE: (values below 1, preferably below 0.1),
                REDUCTION_MODELS: (1,3,5,7,9 for MNIST),
                TAU: ("RE" or "minRE")
                T: Temperature (>= 1)
                metric: "RE", "JSD", "DKL"
        """
        import itertools

        start = time.time()
        combinations = list(itertools.product(*args))
        att = ""
        
        classifier = Classifier(self._sess, self._data, epochs=350, learning_rate=0.01, batch_size=32)
        classifier.execute()

        for combination in combinations:
            n_experiments = combination[0]
            reduction_models = combination[1]
            attack = combination[2]
            drop_rate = combination[3]
            tau = combination[4]
            try:
                T = combination[5]
                metric = combination[6]
            except:
                T = 1
                metric = "RE"
            
            if att != attack:
                f = open("./experiments/experiments_logs/" + self._data.dataset_name + "_" + attack + "_all_cases_experiment.txt", "a+")

            if tau == "RE" and reduction_models == 1:
                continue
            else:     
                team_stats = np.zeros((n_experiments, 5))
                
                if att != attack:
                    x_test_adv = Adversarial_Attack(self._sess, self._data, length=length, attack=attack, epochs=5).attack()                
                    _, x, y, _ = helpers.join_test_sets(self._data, x_test_adv, length, idx=self._idx_adv[:length])
                    att = attack

                multiple_team = Assembly_Team(self._sess, self._data, reduction_models)

                scores_leg = classifier.model.evaluate(self._data.x_test[self._idx_adv][:length], self._data.y_test[self._idx_adv][:length], verbose=1)
                scores = classifier.model.evaluate(x_test_adv[:length], self._data.y_test[self._idx_adv][:length], verbose=1)
                print("\nMain classifier's accuracy on legitimate examples: %.2f%%" % (scores_leg[1]*100))
                print("\nMain classifier's accuracy on adversarial examples: %.2f%%" % (scores[1]*100))

                for exp in range(n_experiments):
                    if metric == "RE":
                        multiple_thresholds = multiple_team.get_thresholds(tau=tau, drop_rate=drop_rate, p = 1, plot_rec_images=False)
                        multiple_x_marks = Image_Reduction.apply_techniques(x, multiple_team, p = 1)
                    else:
                        multiple_thresholds = multiple_team.get_thresholds_pd(tau=tau, classifier = classifier, T=T, drop_rate=drop_rate, p = 1, plot_rec_images=False,metric=metric)
                        multiple_x_marks = Image_Reduction.apply_techniques_pd(x, multiple_team, classifier, T=T, p = 1, metric=metric)

                    y_pred_team, _ = poll_votes(x, y, multiple_x_marks, multiple_thresholds, reduction_models)
                    team_stats[exp,0], team_stats[exp,1], team_stats[exp,2], team_stats[exp,3], team_stats[exp,4], confusion_matrix_team = helpers.get_cm_and_statistics(y, y_pred_team)
                    
                    print("\nSCENARIO {0}/{1} FINISHED.\nTeam CM \n{2}\n".format(exp+1, n_experiments, confusion_matrix_team))

                print("\nEXPERIMENT TERMINATED. {0} DATASET: {1} Input Images 'x', {2} Attack, p = {3}, reduction models = {4}, drop_rate = {5}, tau = {6}, T = {7}\n"
                    .format(self._data.dataset_name, len(x), attack, 1, reduction_models, drop_rate, tau, T))

                s1 = helpers.get_statistics_experiments("Team", team_stats)

                if type(f) != type(None):
                    s0 = "EXPERIMENT TERMINATED. {0} DATASET: {1} Input Images 'x', {2} Attack, p = {3}, reduction models = {4}, drop_rate = {5}, tau = {6}, T = {7}\n\n".format(self._data.dataset_name, len(x), attack, 1, reduction_models, drop_rate, tau, T)
                    sep = '-' * len(s0)
                    helpers.write_txt(f, '\n','\n', s0, s1, '\n', sep, '\n', '\n')         

                helpers.write_txt(f, "\nExperiment's elapsed time: {0}".format(timedelta(seconds=time.time() - start)))
        f.close()

    def simple_experiment(self, reduction_models, attack="FGSM", drop_rate=0.001, tau="RE", p = 1, length=2000, T=1, metric='JSD'):
        """
        Evaluates MultiMagNet with test dataset containing half legitimate and adversarial images, and prints the its metrics.

        # Attributes:        
            length: the size of the test dataset containing legitimate images that will be used in the experiments. A final test dataset will be produced containing legitimate and adversarial images, with size length * 2.
            
            reduction_models: the number of autoencoders randomly chosen to form the MultiMagNet ensemble. 

            attack: can be 'FGSM', 'BIM', 'DEEPFOOL', 'CW_0.0', 'CW_10.0', 'CW_20.0', 'CW_30.0', 'CW_40.0'.

            drop_rate: the maximum percentage of legitimate images classified as 'adversarial'.

            tau: the approach used to compute the thresholds. It can be 'RE' which assigns a different threshold based on each autoencoder's reconstruction error or 'minRE', which assigns the minimum reconstruction error obtained for all the autoencoders. 
        """
        start = time.time()

        # test inputs on main classifier
        classifier = Classifier(self._sess, self._data, epochs=350, learning_rate=0.01, batch_size=32)
        classifier.execute()

        # # Creates surrogate model and returns the perturbed NumPy test set  
        x_test_adv = Adversarial_Attack(self._sess, self._data, length=length, attack=attack, epochs=12).attack(model=classifier.model)

        # Evaluates the brand-new adversarial examples on the main model.
        scores_leg = classifier.model.evaluate(self._data.x_test[self._idx_adv][:length], self._data.y_test[self._idx_adv][:length], verbose=1)
        scores = classifier.model.evaluate(x_test_adv[:length], self._data.y_test[self._idx_adv][:length], verbose=1)
        print("\nMain classifier's accuracy on legitimate examples: %.2f%%" % (scores_leg[1]*100))
        print("\nMain classifier's accuracy on adversarial examples: %.2f%%" % (scores[1]*100))

        # plots the adversarial images
        #helpers.plot_images(self._data.x_test[self._idx_adv][:length], x_test_adv[:length], x_test_adv.shape)

        # Creates a test set containing 'length * 2' input images 'x', where half are benign images and half are adversarial.
        _, x, y, y_ori = helpers.join_test_sets(self._data, x_test_adv, length, idx=self._idx_adv[:length])
        
        # # Creates, trains and returns the 'R' dimensionality reduction team
        team = Assembly_Team(self._sess, self._data, reduction_models)

        if metric == "RE":
            thresholds = team.get_thresholds(tau=tau, drop_rate=drop_rate, p = p, plot_rec_images=False)
            x_marks = Image_Reduction.apply_techniques(x, team, p = p)
        else:
            thresholds = team.get_thresholds_pd(tau=tau, classifier = classifier, T=T, drop_rate=drop_rate, p = p, plot_rec_images=False, metric=metric)
            x_marks = Image_Reduction.apply_techniques_pd(x, team, classifier, T=T, p = p, metric=metric)

        y_pred, filtered_indices = poll_votes(x, y, x_marks, thresholds, reduction_models)

        print("\nEXPERIMENT USING {0} DATASET: {1} Input Images 'x', {2} Attack, p = {3}, reduction models = {4}, drop_rate = {5}\n, T = {6}"
        .format(self._data.dataset_name, len(x), attack, p, reduction_models, drop_rate, T))

        acc, pp, nn, auc, f1, cm = helpers.get_cm_and_statistics(y, y_pred)

        print('Threshold used: {0}\nConfusion Matrix:\n{1}\nACC: {2}, Positive Precision: {3}, Negative Precision: {4}, AUC: {5:.3}, F1: {6:.3}'
            .format(thresholds, cm, acc, pp, nn, auc, f1))

        ori_acc, ref_acc = Reformer(classifier.model, team, x[filtered_indices], y_ori[filtered_indices])
        d_acc = classifier.model.evaluate(x, y_ori)[1]

        print("\nModel accuracy on D set: %.2f%%" % (d_acc*100))
        print("\nModel accuracy on filtered images: %.2f%%" % (ori_acc*100))
        print("Model accuracy on filtered and reformed images: %.2f%%" % (ref_acc*100))

        print("\nExperiment's elapsed time: {0}".format(timedelta(seconds=time.time() - start)))

    def choose_team_each_jump_experiment(self, jump=0, magnet=False, attack="FGSM", drop_rate=0.001, tau="RE", p = 1, length=2000, T=1, metric='JSD'):
        import math
        """
        Evaluates MultiMagNet with test dataset containing half legitimate and adversarial images, and prints the its metrics.

        # Attributes:        
            length: the size of the test dataset containing legitimate images that will be used in the experiments. A final test dataset will be produced containing legitimate and adversarial images, with size length * 2.

            jump: forms a different 'R' team at each jump.

            magnet: if True, it is chosen just one autoencoder; if False, it is chosen a random number of autoencoders.

            attack: can be 'FGSM', 'BIM', 'DEEPFOOL', 'CW_0.0', 'CW_10.0', 'CW_20.0', 'CW_30.0', 'CW_40.0'.

            drop_rate: the maximum percentage of legitimate images classified as 'adversarial'.

            tau: the approach used to compute the thresholds. It can be 'RE' which assigns a different threshold based on each autoencoder's reconstruction error or 'minRE', which assigns the minimum reconstruction error obtained for all the autoencoders. 
        """
        start = time.time()

        # test inputs on main classifier
        classifier = Classifier(self._sess, self._data, epochs=350, learning_rate=0.01, batch_size=32)
        classifier.execute()

        # # Creates surrogate model and returns the perturbed NumPy test set  
        x_test_adv, _, _ = Adversarial_Attack(self._sess, self._data, length=length, attack=attack, epochs=12).attack(model=classifier.model)

        # Evaluates the brand-new adversarial examples on the main model.
        scores_leg = classifier.model.evaluate(self._data.x_test[self._idx_adv][:length], self._data.y_test[self._idx_adv][:length], verbose=1)
        scores = classifier.model.evaluate(x_test_adv[:length], self._data.y_test[self._idx_adv][:length], verbose=1)
        print("\nMain classifier's accuracy on legitimate examples: %.2f%%" % (scores_leg[1]*100))
        print("\nMain classifier's accuracy on adversarial examples: %.2f%%" % (scores[1]*100))

        # plots the adversarial images
        #helpers.plot_images(self._data.x_test[self._idx_adv][:length], x_test_adv[:length], x_test_adv.shape)

        # Creates a test set containing 'length * 2' input images 'x', where half are benign images and half are adversarial.
        _, x, y, y_ori = helpers.join_test_sets(self._data, x_test_adv, length, idx=self._idx_adv[:length])
        team_stats = np.zeros((math.floor(len(x)/jump), 4))
        
        i = 0
        k = 0

        while i+jump <= len(x):
            reduction_models = random.choice([3,5,7,9]) if not magnet else 1
            print("\nInput images 'x' {0}-{1}/{2}\nNumber of autoencoders chosen: {3}".format(i+1, i+jump, len(x), reduction_models))
            print("==============================================")
            team = Assembly_Team(self._sess, self._data, reduction_models)
            
            if metric == "RE":
                thresholds = team.get_thresholds(tau=tau, drop_rate=drop_rate, p = p, plot_rec_images=False)
                x_marks = Image_Reduction.apply_techniques(x[i:i+jump], team, p = p)
            else:
                thresholds = team.get_thresholds_pd(tau=tau, classifier = classifier, T=T, drop_rate=drop_rate, p = p, plot_rec_images=False, metric=metric)
                x_marks = Image_Reduction.apply_techniques_pd(x[i:i+jump], team, classifier, T=T, p = p, metric=metric)

            y_pred, filtered_indices = poll_votes(x[i:i+jump], y[i:i+jump], x_marks, thresholds, reduction_models)

            print("\nEXPERIMENT USING {0} DATASET: {1} Input Images 'x', {2} Attack, p = {3}, reduction models = {4}, drop_rate = {5}\n, T = {6}"
            .format(self._data.dataset_name, len(x[i:i+jump]), attack, p, reduction_models, drop_rate, T))

            team_stats[k,0], team_stats[k,1], team_stats[k,2], _, _, cm = helpers.get_cm_and_statistics(y[i:i+jump], y_pred)
            team_stats[k,3] = reduction_models

            print('Threshold used: {0}\nConfusion Matrix:\n{1}\nACC: {2}, Positive Precision: {3}, Negative Precision: {4}'
                .format(thresholds, cm, team_stats[k,0], team_stats[k,1], team_stats[k,2]))

            ori_acc, ref_acc = Reformer(classifier.model, team, x[i:i+jump][filtered_indices], y_ori[i:i+jump][filtered_indices])
            d_acc = classifier.model.evaluate(x[i:i+jump], y_ori[i:i+jump])[1]

            print("\nModel accuracy on D set: %.2f%%" % (d_acc*100))
            print("\nModel accuracy on filtered images: %.2f%%" % (ori_acc*100))
            print("Model accuracy on filtered and reformed images: %.2f%%" % (ref_acc*100))

            print("\nExperiment's elapsed time: {0}\n".format(timedelta(seconds=time.time() - start)))

            i = i+jump
            k = k+1
        
        helpers.get_statistics_experiments("Team", team_stats)
        print("Number of autoencoders chosen on each experiment: {0}".format(team_stats[:,3]))

import tensorflow as tf
import cleverhans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import utils.helpers as helpers

from utils.data import Data
from modules.retrieve_module import Retrieval
from modules.assembly_team import Assembly_Team
from modules.poll_votes import poll_votes as poll_votes
from modules.poll_votes import poll_votes_each_x as poll_votes_each_x
from classifiers.classifier import Classifier
from adv_attacks.adversarial_attacks import Adversarial_Attack
from modules.apply_techniques import Image_Reduction
from keras.utils import np_utils
import time
from utils.helpers import JSD
from datetime import timedelta
from sklearn.metrics import confusion_matrix

class Experiment:
    def __init__(self, dataset):
        """
        Realizes the MultiMagNet's experiments.

        # Attribute
            dataset: 'MNIST' or 'CIFAR' (this not fully implemented yet!)
        """
        self._idx_adv = helpers.load_imgs_pkl('example_idx.pkl')

        self._sess = tf.Session()
        self._sess.as_default()              

        self._data = Data(dataset_name=dataset) 
        print("\nDataset loaded.")

    def test_logits(self):
        classifier = Classifier(self._sess, self._data, epochs=170)
        model = helpers.get_logits(classifier.model)
        
        print(model.predict(self._data.x_test[0:1]))
        print(np.sum(model.predict(self._data.x_test[0:1])))

        print(classifier.model.predict(self._data.x_test[0:1]))
        print(np.sum(classifier.model.predict(self._data.x_test[0:1])))

    def all_cases_experiment(self, *args, length=2000):
        """
        Creates an cartesian product with '*args' in order to make the experiments on several different scenarios. 
        All the experiments' results are saved in a .TXT file called 'all_cases_experiment.txt'

        # Attributes:
            *args: each '*args' parameter is a list containing all possible MultiMagNet's parameters: 
                NUMBER_EXPERIMENTS: how many times the code will run.
                DATASETS: ("MNIST" or "CIFAR"),
                ATTACKS: ("FGSM", "BIM", "DEEPFOOL", "CW_0.0", "CW_10.0", "CW_20.0", "CW_30.0", "CW_40.0"),
                DROP_RATE: (values below 1, preferably below 0.1),
                REDUCTION_MODELS: (1,3,5,7,9 for MNIST),
                TAU: ("RE" or "minRE")
        """
        import itertools

        start = time.time()
        combinations = list(itertools.product(*args))
        att = ""

        for combination in combinations:
            n_experiments = combination[0]
            reduction_models = combination[1]
            attack = combination[2]
            drop_rate = combination[3]
            tau = combination[4]
            
            if att != attack:
                f = open("./experiments/experiments_logs/" + self._data.dataset_name + "_" + attack + "_all_cases_experiment.txt", "a+")

            if tau == "minRE" and reduction_models == 1:
                continue
            else:     
                team_stats = np.zeros((n_experiments, 5))
                
                if att != attack:
                    x_test_adv = Adversarial_Attack(self._sess, self._data, length=length, attack=attack, epochs=5).attack()                
                    _, x, y = helpers.join_test_sets(self._data.x_test, x_test_adv, length)
                    att = attack

                multiple_team = Assembly_Team(self._sess, self._data, reduction_models)

                for exp in range(n_experiments):
                    multiple_thresholds = multiple_team.get_thresholds(self._data.x_val, tau=tau, drop_rate=drop_rate, p = 1, plot_rec_images=False)
                    multiple_x_marks = Image_Reduction.apply_techniques(x, multiple_team, p = 1)

                    y_pred_team = poll_votes(x, y, multiple_x_marks, multiple_thresholds, reduction_models)
                    team_stats[exp,0], team_stats[exp,1], team_stats[exp,2], team_stats[exp,3], team_stats[exp,4], confusion_matrix_team = helpers.get_cm_and_statistics(y, y_pred_team)
                    
                    print("\nSCENARIO {0}/{1} FINISHED.\nTeam CM \n{2}\n".format(exp+1, n_experiments, confusion_matrix_team))

                print("\nEXPERIMENT TERMINATED. {0} DATASET: {1} Input Images 'x', {2} Attack, p = {3}, reduction models = {4}, drop_rate = {5}, tau = {6}\n"
                    .format(self._data.dataset_name, len(x), attack, 1, reduction_models, drop_rate, tau))

                s1 = helpers.get_statistics_experiments("Team", team_stats)

                if type(f) != type(None):
                    s0 = "EXPERIMENT TERMINATED. {0} DATASET: {1} Input Images 'x', {2} Attack, p = {3}, reduction models = {4}, drop_rate = {5}, tau = {6}\n\n".format(self._data.dataset_name, len(x), attack, 1, reduction_models, drop_rate, tau)
                    sep = '-' * len(s0)
                    helpers.write_txt(f, '\n','\n', s0, s1, '\n', sep, '\n', '\n')         

                helpers.write_txt(f, "\nExperiment's elapsed time: {0}".format(timedelta(seconds=time.time() - start)))
        f.close()

    def testJSD(self, length, reduction_models, attack, logits=True):
        from keras.models import Sequential
        from keras.layers import Lambda
        from keras.activations import softmax

        print("Loading adversarial images...\n")

        idx = np.random.permutation(2000)[:length]

        x_test_adv = Adversarial_Attack(self._sess, self._data, length=length, attack=attack, epochs=12).attack()
        x_test_adv = x_test_adv[idx]

        print("Loading team of autoencoders...\n")
        team_obj = Assembly_Team(self._sess, self._data, reduction_models)
        team = team_obj.get_team()
        
        print("Loading classifier...\n")
        classifier = Classifier(self._sess, self._data, epochs=170)
        classifier.execute()
        sft = Sequential()
        sft.add(Lambda(lambda X: softmax(X, axis=1), input_shape=(10,)))
        
        x = self._data.x_test[self._idx_adv][idx]
        y = self._data.y_test[self._idx_adv][idx]

        helpers.plot_images(x, x_test_adv, shape=(10, 32, 32, 3))
        
        print("Reforming legitimate images...\n")        
        for i in range(len(team)):
            autoencoder = team_obj.load_autoencoder(team[i])
            rec = autoencoder.predict(x)
            rec_adv = autoencoder.predict(x_test_adv)

            sft = Sequential()
            sft.add(Lambda(lambda X: softmax(X, axis=1), input_shape=(10,)))

            out_leg = sft.predict(helpers.get_output_model_layer(x, classifier.model, logits=logits)/10)
            out_rec = sft.predict(helpers.get_output_model_layer(rec, classifier.model, logits=logits)/10)
            out_adv = sft.predict(helpers.get_output_model_layer(x_test_adv, classifier.model, logits=logits)/10)
            adv_rec = sft.predict(helpers.get_output_model_layer(rec_adv, classifier.model, logits=logits)/10)

            leg = JSD(out_leg, out_rec)
            adv = JSD(out_adv, adv_rec)

            # leg = np.mean(np.power(np.abs(out_leg - out_rec), 2), axis=1)
            # adv = np.mean(np.power(np.abs(out_adv - adv_rec), 2), axis=1)
            
            print("\nModel's outputs (original legitimate samples): \n{0}\n".format(out_leg))
            print("Model's outputs (rec legitimate samples): \n{0}\n".format(out_rec))
            print("Model's outputs (original adversarial samples): \n{0}\n".format(out_adv))
            print("Model's outputs (rec adversarial samples): \n{0}\n".format(adv_rec))

            print("Legitimate Class:\n{0}\nLeg. Predicted classes:\n{1}".format(np.argmax(y, axis=1), np.argmax(classifier.model.predict(x), axis=1)))
            print("\nLegitimate Class:\n{0}\nRec. Leg. Predicted classes:\n{1}".format(np.argmax(y, axis=1), np.argmax(classifier.model.predict(rec), axis=1)))
            print("\nLegitimate Class:\n{0}\nAdv. Predicted classes:\n{1}".format(np.argmax(y, axis=1), np.argmax(classifier.model.predict(x_test_adv), axis=1)))
            print("\nLegitimate Class:\n{0}\nRec. Adv. Predicted classes:\n{1}".format(np.argmax(y, axis=1), np.argmax(classifier.model.predict(rec_adv), axis=1)))

            np.set_printoptions(threshold=np.nan)
            float_formatter = lambda x: "%.4f" % x
            np.set_printoptions(formatter={'float_kind':float_formatter})
            print("\nLegitimate JSD: {0}\nAdversarial JSD: {1}".format(leg, adv))
            del autoencoder

    def simple_experiment(self, reduction_models = 3, attack="FGSM", drop_rate=0.001, tau="RE", p = 1, length=2000):
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
        classifier = Classifier(self._sess, self._data, epochs=170, learning_rate=0.01)
        classifier.execute()

        # # Creates surrogate model and returns the perturbed NumPy test set  
        x_test_adv = Adversarial_Attack(self._sess, self._data, length=length, attack=attack, epochs=12).attack(model=classifier.model)

        # Evaluates the brand-new adversarial examples on the main model.
        scores = classifier.model.evaluate(x_test_adv[:length], self._data.y_test[self._idx_adv][:length], verbose=1)
        print("\nMain classifier's baseline error: %.2f%%" % (100-scores[1]*100))

        # plots the adversarial images
        helpers.plot_images(self._data.x_test[self._idx_adv][:length], x_test_adv[:length], x_test_adv.shape)

        # Creates a test set containing 'length * 2' input images 'x', where half are benign images and half are adversarial.
        _, x, y = helpers.join_test_sets(self._data.x_test, x_test_adv, length)
          
        # # Creates, trains and returns the 'R' dimensionality reduction team
        team = Assembly_Team(self._sess, self._data, reduction_models)

        if self._data.dataset_name == "MNIST":
            thresholds = team.get_thresholds(tau=tau, drop_rate=drop_rate, p = p, plot_rec_images=False)
            x_marks = Image_Reduction.apply_techniques(x, team, p = p)
        else:
            thresholds = team.get_thresholds_jsd(tau=tau, classifier = classifier, T=20, drop_rate=drop_rate, p = p, plot_rec_images=True)
            x_marks = Image_Reduction.apply_techniques_jsd(x, team, classifier, T=20, p = p)

        y_pred = poll_votes(x, y, x_marks, thresholds, reduction_models)

        print("\nEXPERIMENT USING {0} DATASET: {1} Input Images 'x', {2} Attack, p = {3}, reduction models = {4}, drop_rate = {5}\n"
        .format(self._data.dataset_name, len(x), attack, p, reduction_models, drop_rate))

        acc, pp, nn, auc, f1, cm = helpers.get_cm_and_statistics(y, y_pred)

        print('Threshold used: {0}\nConfusion Matrix:\n{1}\nACC: {2}, Positive Precision: {3}, Negative Precision: {4}, AUC: {5:.3}, F1: {6:.3}'
            .format(thresholds, cm, acc, pp, nn, auc, f1))

        print("\nExperiment's elapsed time: {0}".format(timedelta(seconds=time.time() - start)))

    def choose_team_each_jump_experiment(self, n_experiments, reduction_models, attack, drop_rate, tau, jump = 50,  length=2000, p = 1):

        """
        Pìcks randomly different autoencoders for each jump and prints the final result.

        # Attributes:
            n_experiments: the number of experiments that will be performed.
            
            length: the size of the test dataset containing legitimate images that will be used in the experiments. A final test dataset will be produced containing legitimate and adversarial images, with size length * 2.

            jump: forms a different 'R' team at each jump.
            
            reduction_models: the number of autoencoders randomly chosen to form the MultiMagNet ensemble, which will be compared to its one-autoencoder version afterwards. 

            attack: can be 'FGSM', 'BIM', 'DEEPFOOL', 'CW_0.0', 'CW_10.0', 'CW_20.0', 'CW_30.0', 'CW_40.0'.

            drop_rate: the maximum percentage of legitimate images classified as 'adversarial'.

            tau: the approach used to compute the thresholds. It can be 'RE' which assigns a different threshold based on each autoencoder's reconstruction error or 'minRE', which assigns the minimum reconstruction error obtained for all the autoencoders.
        """
        start = time.time()

        team_stats = np.zeros((n_experiments, 5))
        unique_stats = np.zeros((n_experiments, 5))

        # test inputs on main classifier
        classifier = Classifier(self._sess, self._data, epochs=20)
        model = classifier.execute()

        # Creates surrogate model and returns the perturbed NumPy test set  
        x_test_adv = Adversarial_Attack(self._sess, self._data, length=length, attack=attack, epochs=5).attack(helpers.get_logits(classifier.model))

        # Evaluates the brand-new adversarial examples on the main model.
        scores = model.evaluate(x_test_adv[:length], self._data.y_test[self._idx_adv][:length], verbose=0)
        print("\nMain classifier's baseline error: %.2f%%" % (100-scores[1]*100))

        # plots the adversarial images
        #helpers.plot_images(data.x_test[idx_adv], x_test_adv, x_test_adv.shape)

        # Creates a test set containing 'length * 2' input images 'x', where half are benign images and half are adversarial.
        _, x, y = helpers.join_test_sets(self._data.x_test, x_test_adv, length)

        for exp in range(n_experiments):
            confusion_matrix_team = np.zeros((2,2))
            confusion_matrix_unique = np.zeros((2,2))

            unique_autoencoder = Assembly_Team(self._sess, self._data, 1)
            multiple_team = Assembly_Team(self._sess, self._data, reduction_models)

            # predicted labels
            y_pred_team = np.zeros((len(y)))
            y_pred_unique = np.zeros((len(y)))

            # Forms different 'R' teams for each input image 'x'
            for i in range(len(x)):
                print("\nInput image 'x' {0}/{1}".format(i+1, len(x)))
                print("==============================================")

                if i % jump == 0:
                    thresholds = multiple_team.get_thresholds(self._data.x_val, tau=tau, drop_rate=drop_rate, p = p, plot_rec_images=False)
                    threshold = unique_autoencoder.get_thresholds(self._data.x_val, tau=tau, drop_rate=drop_rate, p = p, plot_rec_images=False)

                x_ = x[i].reshape(1, x.shape[1], x.shape[2], x.shape[3])

                x_marks = Image_Reduction.apply_techniques(x_, multiple_team, p = p)
                x_marks_u = Image_Reduction.apply_techniques(x_, unique_autoencoder, p = p)

                y_pred_team[i] = poll_votes_each_x(x_, y[i], x_marks, thresholds, reduction_models)
                y_pred_unique[i] = poll_votes_each_x(x_, y[i], x_marks_u, threshold, 1)
            
            team_stats[exp,0], team_stats[exp,1], team_stats[exp,2], team_stats[exp,3], team_stats[exp,4], confusion_matrix_team = helpers.get_cm_and_statistics(y, y_pred_team)        
            unique_stats[exp,0], unique_stats[exp,1], unique_stats[exp,2], unique_stats[exp,3], unique_stats[exp,4], confusion_matrix_unique = helpers.get_cm_and_statistics(y, y_pred_unique)   

            print("\nExperiment {0}/{1}\nTeam CM \n{2}\nOne Autoencoder CM:\n{3}"
                        .format(exp+1, n_experiments, confusion_matrix_team, confusion_matrix_unique))

        print("\nEXPERIMENT 2 TERMINATED. {0} DATASET: {1} Input Images 'x', {2} Attack, p = {3}, reduction models = {4}, drop_rate = {5}\n"
            .format(self._data.dataset_name, len(x), attack, p, reduction_models, drop_rate))

        helpers.get_statistics_experiments("Team", team_stats)
        helpers.get_statistics_experiments("Unique", unique_stats)

        print("\nExperiment's elapsed time: {0}".format(timedelta(seconds=time.time() - start)))


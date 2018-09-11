import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import os
from sklearn.utils import shuffle
from scipy.stats import entropy
from numpy.linalg import norm
from keras.models import Sequential
from keras.layers import Lambda
from keras.activations import softmax

def save_imgs_pkl(imgs, name):
        try:
                if name.endswith(".pkl"): name = name[:-4]
                path = os.path.join("./adv_attacks/adversarial_images", name + '.plk')
                with open(path, 'wb') as f:
                        pickle.dump(imgs, f, pickle.HIGHEST_PROTOCOL)
                return True
        except:
                print('\nIt was not possible to save images in specified directory: {0}'.format(path))
                return False

def load_imgs_pkl(name):
        path = os.path.join("./adv_attacks/adversarial_images", name)
        try:
                with open(path, 'rb') as f:
                        return pickle.load(f)
        except:
                print('\nIt was not possible to load images in specified directory: {0}'.format(path))
                return None

def plot_images(orig, dec, shape, num=10):
    n = num
    plt.figure(figsize=(20, 4))

    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i+1)
        if shape[3] > 1: plt.imshow(orig[i].reshape(shape[1], shape[2], shape[3]))
        else: 
                plt.imshow(orig[i].reshape(shape[1], shape[2]))
                plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i +1 + n)
        if shape[3] > 1: plt.imshow(dec[i].reshape(shape[1], shape[2], shape[3]))
        else: 
                plt.imshow(dec[i].reshape(shape[1], shape[2]))
                plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def reshape_flatten(images):
        return images.reshape((len(images), np.prod(images.shape[1:])))

def get_input_images_x(x_test, y_test, length = 1, index=True):
        """
        This method randomly selects test samples and returns them.

        # Attributes:
           length: the amount of data randomly taken from the test set [1, len(data.x_test)];
           index: if True, returns only the index of the input image 'x', otherwise returns a dict containing the image's fetures and its label

        # Returns:
           the indices shuffled.
        """
        i = np.random.choice(len(x_test), length, replace=False)
        if index:
                return i
        else:
                x = {'image': x_test[i], 
                        'label': y_test[i]}
                return x

def compute_statistics(d_set, approach):
        d_set_whole = d_set.flatten()

        median = np.median(d_set_whole)
        mean = np.mean(d_set_whole, dtype=np.float64)
        minimum = np.min(d_set_whole)
        maximum = np.max(d_set_whole)
        p25 = np.percentile(d_set_whole, 25)
        p75 = np.percentile(d_set_whole, 75)

        print("\nStatistics from {6} 'd_set'\nMin: {0}\nP-25: {1}\nMedian: {2}\nMean: {3}\nP-75: {4}\nMax: {5}"
                        .format(minimum, p25, median, mean, p75, maximum, approach))

def compute_tau(d_set, mode="avg"):
        """
        Computes 'tau' using a pre-defined 'mode', flattening the whole distance matrix.

        # Attributes:
           d_set: the matrix containing all the computed distances;
           mode: the aproach used to compute 'tau':
              avg: defines 'tau' as the mean of the max distances of d_set;
              maximum: defines 'tau' as the max distance from d_set;
              minimum: defines 'tau' as the max distance of all the min distances from of d_set;
              p-25: defines 'tau' as the distance corresponding to the 1st quatile of all distances in ascending order;
              median: defines 'tau' as the median of all distances in ascending order;
              p-75: defines 'tau' as the distance corresponding to the 3rd quatile of all distances in ascending order.
              max_knn: defines 'tau' as the max distance computed by kNN (before dimensionality reduction process).

        # Returns: the corresponding chosen measure of the whole 'd_set'.
        """
        d_set_whole = d_set.flatten()

        if (mode == 'avg'): return np.mean(d_set_whole, dtype=np.float64)
        if (mode == 'minimum'): return np.min(d_set_whole)
        if (mode == 'p25'): return np.percentile(d_set_whole, 25)
        if (mode == 'median'): return np.median(d_set_whole)
        if (mode == 'p75'): return np.percentile(d_set_whole, 75)
        if (mode == 'maximum'): return np.max(d_set_whole)

def join_test_sets(leg_set, adv_set, length=100, idx = []):
        """
        Unifies the benign and adversarial test sets into an unique set and creates its corresponding labels set, where 
        '1' represents 'benign data' and '0' 'adversarial data'.

        # Attributes:
                length: the amount of data taken from the original test set. It can be a number between '1' and 'data.x_test.shape[0]'.

        # Returns: 
                idx: the indices retrieved from original test set. The indices are two-folded because each image on leg_set was turned into adversarial.
                x: the data (benign and adversarial);
                y: the labels.
        """

        leg_labels = np.ones(shape=length)
        adv_labels = np.zeros(shape=length)

        idx_l = np.random.permutation(len(leg_set))[:length]
        idx_a = np.random.permutation(len(adv_set))[:length]

        x = np.concatenate((leg_set[idx_l], adv_set[idx_a]))
        y = np.concatenate((leg_labels, adv_labels))

        i = np.random.permutation(len(x))
        return np.concatenate((idx_l, idx_a))[i], x[i], y[i]  

def assign_confusion_matrix(confusion_matrix, x_label, ans_label):
        if x_label == 1 and ans_label == 1: # 'x' is benign and it was classified as benign.
                confusion_matrix[0][0] += 1

        if x_label == 1 and ans_label == 0: # 'x' is benign and it was classified as adversarial. 
                confusion_matrix[0][1] += 1

        if x_label == 0 and ans_label == 1: # 'x' is adversarial and it was classified as benign. 
                confusion_matrix[1][0] += 1

        if x_label == 0 and ans_label == 0: # 'x' is adversarial and it was classified as adversarial.
                confusion_matrix[1][1] += 1

def get_cm_and_statistics(y, y_pred):
        """
        Gets the main metrics of a CM using the ground-truth and predicted labels.

        # Attributes:
                y: the ground-truth labels.
                y_pred: the predicted labels.
        """
        from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score

        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        n_p = cm[0,0] / (cm[0,0] + cm[0,1])
        p_p = cm[1,1] / (cm[1,0] + cm[1,1])
        auc = roc_auc_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        return acc, p_p, n_p, auc, f1, cm

def get_statistics_experiments(experiment, stats):
        """
        Get the main statistics from all confusion matrices of the experiments performed.

        # Atributes:
                experiment: a string that indentifies whether the metrics belong to a 'Team' formed by 'm' autoencoders,
                or a 'Unique' autoenconder.
                
                stats: a numpy matrix with shape (n_autoencoders, 5), where '5' represents the 'ACC', 
                'Positive Precision', 'Negative Precision', auc and f1 score, computed from the confusion matrix.                
        """

        acc_mean = np.mean(stats[:,0])
        pp_mean = np.mean(stats[:,1])
        nn_mean = np.mean(stats[:,2])
        auc_mean = np.mean(stats[:,3])
        f1_mean = np.mean(stats[:,4])

        s = "{0} Statistics: ACC Mean: {1:.2%}, Positive Mean: {2:.2%}, Negative Mean: {3:.2%}, AUC mean: {4:.3}, F1 Mean: {5:.3}".format(experiment, acc_mean, pp_mean, nn_mean, auc_mean, f1_mean)
        print(s)
        return s

def write_txt(f, *string):                        
        for s in string:
                f.write(s)

def JSD(P, Q):
        _P = P / norm(P, ord=1)
        _Q = Q / norm(Q, ord=1)
        _M = 0.5 * (_P + _Q)
        return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))        



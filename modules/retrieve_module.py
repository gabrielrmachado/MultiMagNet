import random
import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import operator
from PIL import Image
import utils.helpers as helpers

class Retrieval:
    __sess = None
    __dataset = None

    def __init__(self, sess, dataset):
        self.__sess = sess
        self.__dataset = dataset

    def retrieve_k_similar_images(self, x, y, k_neighbors = 10, metric='L2'):     
        """
        Performs kNN algorithm in the validation dataset.

        # Attributes:
            x: the input image 'x'.
            y: the 'x's label.
            k_neighbors: the number of most similar images to the input image 'x'
            metric: the metric of distance - L1, L2 or LM.
        
        # Returns: 
            distances: the computed distances by kNN algorithm;
            indexes: the predominant class' indexes computed by kNN;
            k_most_similar_images: the 'I' set formed by the 'k' most similar images to 'x', including 'x'. 
            The returned set are shaped as "'k, img_rows, img_cols'", for future steps.
        """
        x_val = helpers.reshape_flatten(self.__dataset.x_val)
        x_flatten = x.reshape(np.prod(x.shape[0:]))
        y_val = self.__dataset.y_val

        print("Input image 'x' label %d" % np.argmax(y))

        # Placeholders
        x_ = tf.placeholder(shape=x_flatten.shape, dtype=tf.float32)
        x_data = tf.placeholder(shape=[None, x_flatten.shape[0]], dtype=tf.float32)
        y_data = tf.placeholder(shape=[None, y_val.shape[1]], dtype=tf.float32)

        if metric == 'L1':
            distance = tf.negative(tf.reduce_sum(tf.abs(tf.subtract(x_data, x_)),reduction_indices=1))

        if metric == 'L2':
            distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x_data, x_)), reduction_indices=1))
        
        # manhattan distance
        if metric == 'LM':
            distance = tf.reduce_sum(tf.abs(tf.subtract(x_data, x_)),reduction_indices=1)

        # kNN
        _, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k_neighbors)
        prediction_indices = tf.gather(y_data, top_k_indices)

        distances, x_i, y_i = self.__sess.run([distance, top_k_indices, prediction_indices], feed_dict = {
                                                            x_data: x_val,
                                                            y_data: y_val,
                                                            x_: x_flatten})

        classes = []
        i_set = []

        for i in range(y_i.shape[0]):
            classes.append(y_i[i].argmax())

        # Creates an array containing the 'k' most similar images, using the computed indexes.
        for i in range(x_i.shape[0]):
            i_set.append(self.__dataset.x_train[x_i[i]])

        unique, counts = np.unique(classes, return_counts=True)
        occurrences = dict(zip(unique, counts))

        print(occurrences)

        # Retrieves from the 'I' set all the indexes of images which belongs to the predominant class 
        predominant_index = max(occurrences.items(), key=operator.itemgetter(1))[0]
        indexes = np.where(classes == predominant_index)[0]

        return distances[x_i], indexes, np.array(i_set)
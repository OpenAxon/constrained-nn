from config import config
import numpy as np
import tensorflow as tf


def generate_dataset_with_disjoint_classes(size, nb_disjoint, nb_others):
    ''' Generates a synthetic dataset for testing disjoint classes restrictions
    @param:size: the number of instances in the dataset
    @param:nb_disjoint: the number of disjoint classes to be generated
    @param:nb_others: the number of other classes to be generated'''
    np.random.seed = config['dataset_seed']
    x_disjoint = np.round(np.random.rand(size, nb_disjoint), 2)
    y_disjoint = np.zeros((size, nb_disjoint))
    max_indices = np.argmax(x_disjoint, axis=1)
    I = np.indices(max_indices.shape)
    y_disjoint[I, max_indices] = 1
    x = np.round(np.random.rand(size, nb_others), 2)
    y = (x > 0.5).astype(np.float)
    return np.hstack((x_disjoint, x)), np.hstack((y_disjoint, y))


def split(X, Y, test_size):
    return X[int(X.shape[0]*test_size):], Y[int(Y.shape[0]*test_size):], \
           X[:int(X.shape[0]*test_size)], Y[:int(Y.shape[0]*test_size)]


def generate_and_split(size, nb_disjoint, nb_others, test_size):
    X, Y = generate_dataset_with_disjoint_classes(size, nb_disjoint, nb_others)
    return split(X, Y, test_size)

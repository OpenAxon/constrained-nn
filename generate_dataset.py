# Copyright 2017 Axon Enterprise, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from config import config
import numpy as np

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

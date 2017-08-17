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

from generate_dataset import generate_and_split
from config import *
import json
import keras
from keras.activations import softmax
from keras.layers import Dense, Dropout, Input
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.metrics import binary_accuracy, categorical_accuracy
from keras.models import Model
from keras.utils import plot_model
import numpy as np
import os
import tensorflow as tf
import uuid

def mutually_exclusive_loss(y_true, y_pred):
    '''define a loss over a set of label. 
       Min loss is when one label is correct with val 1, and others have value 0.
       Max loss is when other labels have value 1 while correct label has value close to 0.'''
    return(categorical_crossentropy(y_true, softmax(y_pred)))

def multiple_loss(y_true, y_pred):
    '''Assuming disjoint classes are the first columns of the data'''
    y_disjoint_true = y_true[:, :config['nb_disjoint_classes']]
    y_disjoint_pred = y_pred[:, :config['nb_disjoint_classes']]
    y_other_true = y_true[:, config['nb_disjoint_classes']:]
    y_other_pred = y_pred[:, config['nb_disjoint_classes']:]
    loss1 = categorical_crossentropy(y_disjoint_true, softmax(y_disjoint_pred))
    loss2 = binary_crossentropy(y_other_true, y_other_pred)
    return config['disjoint_classes_output_weight'] * loss1 + loss2

def model(input_shape, nb_output):
    x = input = Input(shape=input_shape)
    x = Dense(config['total_neurons_per_layer'], activation='relu')(x)
    x = Dropout(config['dropout'])(x)
    x = Dense(config['total_neurons_per_layer'], activation='relu')(x)
    x = Dropout(config['dropout'])(x)
    x = Dense(config['total_neurons_per_layer'], activation='relu')(x)
    x = Dropout(config['dropout'])(x)
    x = Dense(nb_output, activation='sigmoid', name='main_output')(x)
    model = Model(inputs=input, outputs=x)
    plot_model(model, to_file=config['model_image_path'], show_shapes=True)
    return(model)

def constraint_accuracy(y_true, y_pred):
    y_disjoint_true = y_true[:, :config['nb_disjoint_classes']]
    y_disjoint_pred = y_pred[:, :config['nb_disjoint_classes']]
    return categorical_accuracy(y_disjoint_true, y_disjoint_pred)

def others_accuracy(y_true, y_pred):
    y_others_true = y_true[:, config['nb_disjoint_classes']:]
    y_others_pred = y_pred[:, config['nb_disjoint_classes']:]
    return binary_accuracy(y_others_true, y_others_pred)

def train_simple(model, x, y, tboard):
    model.compile(optimizer='adam', 
                  metrics=[constraint_accuracy, others_accuracy],
                  loss={'main_output': binary_crossentropy})
    model.fit(x, y,
              batch_size=config['batch_size'], epochs=config['epochs'],
              callbacks=[keras.callbacks.TensorBoard(log_dir=tboard)],
              validation_split=0.33,
              verbose=KERAS_VERBOSITY)
    return

def train_constrained(model, x, y, tboard):
    model.compile(optimizer='adam', 
                  metrics=[constraint_accuracy, others_accuracy],
                  loss={'main_output': multiple_loss})
    model.fit(x,
              y, # labels
              batch_size=config['batch_size'], epochs=config['epochs'], 
              callbacks=[keras.callbacks.TensorBoard(log_dir=tboard)],
              validation_split=0.33,
              verbose=KERAS_VERBOSITY)
    return

def test(m, x, y):
    return m.evaluate(x, 
                      {'disjoint_classes': y[:,:config['nb_disjoint_classes']], 'main_output': y[:,config['nb_disjoint_classes']:]},
                      batch_size=config['batch_size'],
                      verbose=KERAS_VERBOSITY)

def main():
    x_train, y_train, x_test, y_test = generate_and_split(config['dataset_size'], config['nb_disjoint_classes'], config['nb_other_classes'], config['test_size'])
    m = model((config['nb_disjoint_classes']+config['nb_other_classes'],), config['nb_disjoint_classes']+ config['nb_other_classes'])

    name = NAME
    net_name = name+str(uuid.uuid4())
    tboard = os.path.join(config['logdir_path'], net_name)
    train_simple(m, x_train, y_train, tboard)

    print("Network type: constrained")
    name = NAME+"_DISJOINT_WEIGHT_{}_".format(config['disjoint_classes_output_weight'])
    net_name = name+str(uuid.uuid4())
    tboard = os.path.join(config['logdir_path'], net_name)
    train_constrained(m, x_train, y_train, tboard)

if __name__ == '__main__':
    main()



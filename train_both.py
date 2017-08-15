from generate_dataset import generate_and_split
from config import *
import json
import keras
from keras import backend as K
from keras.activations import softmax
from keras.layers import Dense, Dropout, Input, concatenate
from keras.losses import categorical_crossentropy, mean_squared_error, binary_crossentropy
from keras.metrics import binary_accuracy, categorical_accuracy
from keras.models import Model
from keras.utils import plot_model
from math import log
import numpy as np
import os
import tensorflow as tf
import uuid

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
KERAS_VERBOSITY = 2

def mutually_exclusive_loss(y_true, y_pred):
    '''define a loss over a set of label. 
       Min loss is when one label is correct with val 1, and others have value 0.
       Max loss is when other labels have value 1 while corect label has value close to 0.'''
    return(categorical_crossentropy(y_true, softmax(y_pred)))

def multiple_loss(y_true, y_pred):
    '''Assuming disjoint classes are the first colomns of the data'''
    y_disjoint_true = y_true[:, :config['nb_disjoint_classes']]
    y_disjoint_pred = y_pred[:, :config['nb_disjoint_classes']]
    y_other_true = y_true[:, config['nb_disjoint_classes']:]
    y_other_pred = y_pred[:, config['nb_disjoint_classes']:]
    loss1 = categorical_crossentropy(y_disjoint_true, softmax(y_disjoint_pred))
    loss2 = binary_crossentropy(y_other_true, y_other_pred)
    return config['disjoint_classes_output_weight'] * loss1 + loss2

def model(input_shape, nb_output):
    x = input = Input(shape=input_shape)
    x = Dense(200, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(200, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(200, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(nb_output, activation='sigmoid', name='main_output')(x)
    model = Model(inputs=input, outputs=x)
    plot_model(model, to_file='images/constrained-network.png', show_shapes=True)
    return(model)

def constraint_accuracy(y_true, y_pred):
    y_disjoint_true = y_true[:, :config['nb_disjoint_classes']]
    y_disjoint_pred = y_pred[:, :config['nb_disjoint_classes']]
#    return tf.reduce_all(tf.equal(tf.cast(tf.cast(y_disjoint_pred+0.5, tf.int32), tf.bool), tf.cast(y_disjoint_true, tf.bool)))
    return categorical_accuracy(y_disjoint_true, y_disjoint_pred)

def others_accuracy(y_true, y_pred):
    y_others_true = y_true[:, config['nb_disjoint_classes']:]
    y_others_pred = y_pred[:, config['nb_disjoint_classes']:]
    return binary_accuracy(y_others_true, y_others_pred)

def train_simple(model, x, y, tboard):
    model.compile(optimizer='adam', 
                  metrics=[constraint_accuracy, others_accuracy],
                  loss={'main_output': binary_crossentropy})
    model.fit(x, # data
              y, # labels
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
    # print(m.input_shape)
    # print(m.output_shape)
    # print(x_train.shape)
    # print(y_train.shape)

#    print("Network type: simple")
#    name = NAME
#    net_name = name+str(uuid.uuid4())
#    tboard = os.path.join(config['logdir_path'], net_name)
#    train_simple(m, x_train, y_train, tboard)

    print("Network type: constrained")
    name = NAME+"_DISJOINT_WEIGHT_{}_".format(config['disjoint_classes_output_weight'])
    net_name = name+str(uuid.uuid4())
    tboard = os.path.join(config['logdir_path'], net_name)
    train_constrained(m, x_train, y_train, tboard)

if __name__ == '__main__':
    main()



from generate_dataset import generate_and_split
from config import *
import json
import keras
from keras import backend as K
from keras.activations import softmax
from keras.layers import Dense, Dropout, Input, concatenate
from keras.losses import categorical_crossentropy, mean_squared_error, binary_crossentropy
from keras.models import Model
from keras.utils import plot_model
from math import log
import numpy as np
import os
import tensorflow as tf
import uuid

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
KERAS_VERBOSITY = 0

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

def constraint_metric(y_true, y_pred):
    y_disjoint_true = y_true[:, :config['nb_disjoint_classes']]
    y_disjoint_pred = y_pred[:, :config['nb_disjoint_classes']]
    return tf.equal(tf.cast(y_disjoint_pred, tf.bool), tf.cast(y_disjoint_true, tf.bool))

def train_simple(model, x, y, tboard):
    model.compile(optimizer='adam', 
                  metrics=[constraint_metric],
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
                  metrics=[constraint_metric],
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
    print("Network type: constrained")
    name = NAME+"_DISJOINT_WEIGHT_{}_".format(config['disjoint_classes_output_weight'])
    net_name = name+str(uuid.uuid4())
    tboard = os.path.join(config['logdir_path'], net_name)
    train_constrained(m, x_train, y_train, tboard)
    total_loss, disjoint_classes_loss, main_loss = test(m, x_test, y_test)
    print("Total loss:{}\nMain loss:{}\nDisjoint classes loss:{}".format(total_loss, main_loss, disjoint_classes_loss))
    wrong_pred = [0, 0]
    incorrects = 0
    pred_data = []
    for i in range(len(x_test)):
        predictions = m.predict(np.reshape(x_test[i], (1, x_test[i].shape[0])), batch_size=1)
        binary_pred = [np.zeros(config['nb_disjoint_classes']), np.zeros(config['nb_other_classes'])]
        binary_pred[0][np.argmax(predictions[0][0])] = 1
        binary_pred[1] = (predictions[1][0] > 0.5).astype(np.float)
        pred_data.append({'prediction': list(np.concatenate(binary_pred)), 'truth': list(y_test[i])})
        incorrect = False
        k = 0
        for j in range(len(binary_pred[k])):
            if binary_pred[k][j] != y_test[i][j+(k*config['nb_disjoint_classes'])]:
                wrong_pred[k] += 1
#                print("{}\ninput:{}\npred:{}\npred_binary:{}\ntruth:      {}\n---------------------"\
#                      .format(k, x_test[i], predictions, binary_pred, y_test[i]))
                incorrect = True
                break
        k = 1
        for j in range(len(binary_pred[k])):
            if binary_pred[k][j] != y_test[i][j+(k*config['nb_disjoint_classes'])]:
                wrong_pred[k] += 1
#                print("{}\ninput:{}\npred:{}\npred_binary:{}\ntruth:      {}\n---------------------"\
#                      .format(k, x_test[i], predictions, binary_pred, y_test[i]))
                incorrect = True
        if incorrect:
            incorrects += 1


    acc = (len(x_test)-incorrects)/float(len(x_test))
    print("prediction accuracy: {}".format(acc))
    print("Number of times the disjointness constraint was violated: {}/{}".format(wrong_pred[0], len(x_test)))
    print("Generic classes classificaton accuracy: {}/{}".format(wrong_pred[1]/float(config['nb_other_classes']), len(x_test)))
    payload = {'network_type': "constrained",
               'config': config,
               'results': {'strict_accuracy': acc,
                        'nb_disjoint_constraint_broken': wrong_pred[0],
                        'nb_other_incorrect': wrong_pred[1],
                        'loss': total_loss,
                        'disjoint_classes_loss': disjoint_classes_loss,
                        'other_classes_loss': main_loss },
               'tboard_log_file': tboard,
               'predictions': pred_data }

    with open(os.path.join(config['results_path'], net_name), "wt") as f:
        json.dump(payload, f)

if __name__ == '__main__':
    main()


from generate_dataset import generate_and_split
from config import *
import json
import keras
from keras import backend as K
from keras.activations import softmax
from keras.layers import Dense, Dropout, Input, concatenate
from keras.losses import categorical_crossentropy, mean_squared_error, binary_crossentropy
from keras.models import Model
from math import log
import numpy as np
import os
import tensorflow as tf
import uuid


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def mutually_exclusive_loss(y_true, y_pred):
    '''define a loss over a set of label. 
       Min loss is when one label is correct with val 1, and others have value 0.
       Max loss is when other labels have value 1 while corect label has value close to 0.'''
    return(categorical_crossentropy(y_true, softmax(y_pred)))

def model(input_shape, nb_output):
    x = input = Input(shape=input_shape)
    x = Dense(200, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(200, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(nb_output, activation='sigmoid', name='main_output')(x)
    model = Model(inputs=input, outputs=x)
    return(model)

def train(model, x, y, tboard):
    model.compile(optimizer='adam', loss={'main_output': config['loss']})
    model.fit(x, # data
              y, # labels
              batch_size=config['batch_size'], epochs=config['epochs'],
              callbacks=[keras.callbacks.TensorBoard(log_dir=tboard)],
              validation_split=0.33,
              verbose=0)
    return

def test(m, x, y):
    return m.evaluate(x, 
                      y,
                      batch_size=config['batch_size'],
                      verbose=0)

def main():
    x_train, y_train, x_test, y_test = generate_and_split(config['dataset_size'], config['nb_disjoint_classes'], config['nb_other_classes'], config['test_size'])
    nb_classes = config['nb_disjoint_classes']+config['nb_other_classes']
    m = model((nb_classes,), nb_classes)
    # print(m.input_shape)
    # print(m.output_shape)
    # print(x_train.shape)
    # print(y_train.shape)
    net_name = NAME+str(uuid.uuid4())
    tboard = os.path.join(config['logdir_path'], net_name)
    train(m, x_train, y_train, tboard)
    total_loss = test(m, x_test, y_test)
    print("Network type: simple")
    print("Total loss:{}".format(total_loss))
    wrong_pred = [0, 0]
    incorrects = 0
    pred_data = []
    for i in range(len(x_test)):
        predictions = m.predict(np.reshape(x_test[i], (1, x_test[i].shape[0])), batch_size=1)
        binary_pred = [np.zeros(config['nb_disjoint_classes']), np.zeros(config['nb_other_classes'])]
        binary_pred[0][np.argmax(predictions[0][:config['nb_disjoint_classes']])] = 1
        binary_pred[1] = (predictions[0][config['nb_disjoint_classes']:] > 0.5).astype(np.float)
        pred_data.append({'prediction': list(np.concatenate(binary_pred)), 'truth': list(y_test[i])}) 
        incorrect = False
        for k in [0,1]:
            for j in range(len(binary_pred[k])):
                if binary_pred[k][j] != y_test[i][j+(k*config['nb_disjoint_classes'])]:
                    wrong_pred[k] += 1
#                    print("{}\ninput:{}\npred:{}\npred_binary:{}\ntruth:      {}\n---------------------"\
#                          .format(k, x_test[i], predictions, binary_pred, y_test[i]))
                    incorrect = True
                    break
        if incorrect:
            incorrects += 1
    acc = (len(x_test)-incorrects)/float(len(x_test))
    print("prediction accuracy (fully correct predicitions): {}".format(acc))
    print("Number of times the disjointness constraint was violated: {}/{}".format(wrong_pred[0], len(x_test)))
    print("Number of times the generic classes were misclassified: {}/{}".format(wrong_pred[1], len(x_test)))
    payload = {'network_type': 'simple',
               'config': config, 
               'results': {'strict_accuracy': acc,
                        'nb_disjoint_constraint_broken': wrong_pred[0],
                        'nb_other_incorrect': wrong_pred[1],
                        'loss': total_loss },
               'tboard_log_file': tboard,
               'predictions': pred_data }

    with open(os.path.join(config['results_path'], net_name), "wt") as f:
        json.dump(payload, f)
 
if __name__ == '__main__':
    main()
    
 

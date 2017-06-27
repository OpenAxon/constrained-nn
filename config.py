config = {
    'test_size': 0.3,
    'dataset_size': 100000,
    'nb_disjoint_classes': 5,
    'nb_other_classes': 50,
    'epochs': 1000,
    'nb_output': 10,
    'loss': 'binary_crossentropy',
    'batch_size': 32,
    'dataset_seed': 42,
    'disjoint_classes_output_weight': 2,
    'layers': 2,
    'total_neurons_per_layer': 100,
    'logdir_path': 'logs/3_layers',
    'results_path': 'results/3_layers'
}

NAME = "TEST_SIZE_{}-DATASET_SIZE_{}-NB_CLASSES_{}-EPOCHS_{}-LOSS_{}-2layers100neurons"\
       .format(config['test_size'], config['dataset_size'], config['nb_disjoint_classes']+config['nb_other_classes'], config['epochs'], config['loss'])

KERAS_VERBOSITY = 0

config = {
    'test_size': 0.3,
    'dataset_size': 10000,
    'nb_disjoint_classes': 5,
    'nb_other_classes': 50,
    'epochs': 10000,
    'batch_size': 32,
    'dataset_seed': 42,
    'disjoint_classes_output_weight': 2,
    'layers': 3,
    'total_neurons_per_layer': 200,
    'logdir_path': 'logs/test',
    'results_path': 'results/test'
}

NAME = "TEST_SIZE_{}-DATASET_SIZE_{}-NB_CLASSES_{}-EPOCHS_{}-3layers200neurons"\
       .format(config['test_size'], config['dataset_size'], config['nb_disjoint_classes']+config['nb_other_classes'], config['epochs'])

KERAS_VERBOSITY = 0

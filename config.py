config = {
    'test_size': 0.3,
    'dataset_size': 50000,
    'nb_disjoint_classes': 5,
    'nb_other_classes': 50,
    'epochs': 2500,
    'batch_size': 32,
    'dataset_seed': 42,
    'disjoint_classes_output_weight': 8,
    'layers': 3,
    'total_neurons_per_layer': 200,
    'logdir_path': 'logs/experiments2/dataset-50000',
    'results_path': 'results/experiments2'
}

NAME = "TEST_SIZE_{}-DATASET_SIZE_{}-NB_CLASSES_{}-EPOCHS_{}-BATCH_{batch}-3layers200neurons"\
       .format(config['test_size'], config['dataset_size'], config['nb_disjoint_classes']+config['nb_other_classes'], config['epochs'], batch=config['batch_size'])

KERAS_VERBOSITY = 0


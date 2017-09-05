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
    'dropout': 0.3,
    'logdir_path': 'logs/experiments2/dataset-50000',
    'results_path': 'results/experiments2',
    'model_image_path': 'images/constrained-network.png'
}

NAME = "TEST_SIZE_{}-DATASET_SIZE_{}-NB_CLASSES_{}-EPOCHS_{}-BATCH_{batch}-3layers{nbneurons}neurons"\
       .format(config['test_size'], config['dataset_size'], config['nb_disjoint_classes']+config['nb_other_classes'], 
               config['epochs'], batch=config['batch_size'], nbneurons=config['total_neurons_per_layer'])

KERAS_VERBOSITY = 0


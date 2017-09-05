Experiments towards adding logical constraints to multi-class, multi-label classifiers.
This is the code for the paper "Class disjointness constraints as specific objective functions in neural network classifiers" published at SemDeep (http://semdeep.iiia.csic.es/SemDeep-2/index.html).

### Prerequisites
See `requirements.txt`. This code is using the following libraries:
* Tensorflow
* Keras

### How to run
* Create the `model_image_path` and `results_path` directories specified in `config.py` (see below).
* `pip install -r requirements.txt`
* `python train_both.py`

Parameters can then be set in config.py (see below)

### `config.py`
* `test_size`: size of the test set wrt the dataset size
* `dataset_size`: size of the dataset
* `nb_disjoint_classes`: number of disjoint classes
* `nb_other_classes`: number of other classes
* `epochs`: number of training epochs
* `batch_size`: batch size
* `dataset_seed`: seed for the dataset random generation. Same seed will generate the same dataset.
* `disjoint_classes_output_weight`: weight of the disjoint classes loss.
* `logdir_path`: path for tensorboard logging. Logging can be disabled by removing the callback in the model training method.
* `results_path`: path to store the trained models

### Citations
If you find this code or the ideas useful, please reference the paper:

```Class disjointness constraints as specific objective functions in neural network classifiers. François Scharffe. 2nd Workshop on Semantic Deep Learning (SemDeep). Montpellier, France. 2017.```

### Disclaimer
This is experimental code, meant to accompany the above referenced paper, not meant to use in production systems.

### Copyright 
Copyright 2017 Axon Enterprise, Inc.

### License
See `LICENSE.txt` in this repository.
Apache (version 2.0) https://www.apache.org/licenses/LICENSE-2.0

# DeepSars - XR Multilabel Classification

This project aims to provide a number of utilities to address multilabel classification problems related to chest X-ray images. This project is currently under construction and is subject to multiple changes.

## Getting Started

Pull the DeepSars - XR Multilabel Classification code repository and follow the instructions below to get you a copy of the project up and running on your local machine for development and testing purposes. 

### Installation

All the packages are listed in `requirements-pip.txt`. You can install all of them by executing the next line in the terminal:

```
pip install -r requirements-pip.txt
```

## Usage

### Preparing datasets for training

The first thing to be done is the contruction of `*.tfrecords` files to store the datasets that will be used for training, validation and testing of the models built. These files can be stored somewhere of the user's preference and the path to these must be referenced in `config.py`. In future versions of this project it is planned to make available a tool to easily build these datasets.

### Execution

Once the datasets are constructed, a variety of functions can be executed that have as their purpose the training of neural networks for the classification of multi-label problems and the evaluation of these networks through many tasks and metrics. The general procedure to start an execution is as follows:

1. Edit `config.py` to specificy the dataset and process configuration by commenting/uncommenting/editing specific lines
2. Run `train.py`
3. The results are written into a newly created subdirectory under `config.result_dir`

The processes available for execution are the following:


#### Results folder structure

Depending on the process being executed, a new subdirectory will be created within `results` with a unique identifier (uid) for the experiment and a description given by the user, e.g. `uid-network-dataset`. The structure for each subdirectory looks something like

```
📦DeepSars_multilabel_rx
 ┗ 📂results
   ┗ 📂uid-network-dataset
      ┣ 📜log.txt
      ┣ 📜config.txt
      ┣ 📜best_auc_model.h5
      ┣ 📜best_loss_model.h5
      ┣ 📜metrics_on_training.png
      ┣ 📜evaluation_log.txt
      ┗ 📜metrics_on_evaluation.pkl
```


## To Do list

- Enhance current behaviour of existing features:
  - [ ] Make so that every function in `train.py` creates a `README.md` in the experiment subfolder with information about the training
  - [ ] Work on `utils.HistoryPlotter` to make it plot the metrics when there's no validation data
  - [ ] Modify the model save in `train_single_network` to match the DeepSars naming convention
- Add the following functions to `train.py`:
  - [x] **`train_single_network`:** trains a default `tf.keras.applications` network using a test and validation record
  - [ ] **`train_late_fusion_ensemble`:** trains two default `tf.keras.applications` networks using either a standard `Average` layer or a custom `WeightedAverage` layer
- Add the following functions to `util_scripts.py`:
  - [x] **`evaluate_single_network`:** evaluates a single model using a test set
  - [x] **`evaluate_late_fusion_ensemble`:** takes two trained models or a late fusion ensemble to evaluate using a test set
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import utils
import layers
import models
import config
import dataset
import callback


def train_single_network(epochs=5, initial_lr=1e-3, verbose=2, print_summary=True):
    """Trains a default tf.keras.applications network using the data
    from the train and validation TfRecords.

    Args:
        epochs (int): Number of epochs to train. Defaults to 5.
        initial_lr (float): Initial learning rate. Defaults to 1e-3.
        verbose (int): Verbosity for the train process. Defaults to 2.

    Raises:
        Exception: If the train or validation TfRecords do not exist
    """

    # Create result subdirectory to store the experiment results
    result_subdir = utils.create_result_subdir(config.result_dir, config.desc)

    # Read dataset
    if config.train_record is not None:
        train_dataset = tf.data.TFRecordDataset(config.train_record)
    else:
        raise Exception("No path for the Training TfRecord was given.")

    if config.train_record is not None:
        valid_dataset = tf.data.TFRecordDataset(config.valid_record)
    else:
        raise Exception("No path for the Validation TfRecord was given.")

    # Apply the stack of transformations uncommented in config.py
    train_dataset = dataset.map_functions(train_dataset, config.dataset.map_functions)
    valid_dataset = dataset.map_functions(valid_dataset, config.dataset.map_functions)

    # Shuffle, batch and prefetch both datasets
    train_batches = (
        train_dataset.shuffle(config.dataset.shuffle, reshuffle_each_iteration=True)
        .batch(config.dataset.batch_size)
        .prefetch(config.dataset.prefetch)
    )

    valid_batches = (
        valid_dataset.shuffle(config.dataset.shuffle, reshuffle_each_iteration=True)
        .batch(config.dataset.batch_size)
        .prefetch(config.dataset.prefetch)
    )

    # load the model and print summary
    print("Loading the model...\n")
    model = models.multiclass_model(**config.network_1)
    if print_summary:
        model.summary()

    # Optimizer + compile the model
    adam_optimizer = tf.keras.optimizers.Adam(
        lr=initial_lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False,
    )

    model.compile(
        optimizer=adam_optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    )

    # callbacks
    callbacks = []

    if "decay_lr_after_epoch" in config.callbacks:
        callbacks.append(
            tf.keras.callbacks.LearningRateScheduler(
                callback.decay_lr_on_epoch_end, verbose=1
            )
        )

    if "reduce_lr_on_plateau" in config.callbacks:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                **config.callbacks.reduce_lr_on_plateau
            )
        )

    if "model_checkpoint_callback" in config.callbacks:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(result_subdir, "best_loss_model.h5"),
                **config.callbacks.model_checkpoint_callback
            )
        )

    if "multiple_class_auroc" in config.callbacks:
        callbacks.append(
            callback.MultipleClassAUROC(
                dataset=valid_batches,
                result_subdir=result_subdir,
                **config.callbacks.multiple_class_auroc
            )
        )

    # train
    history = {}
    history[config.network_1.module_name] = model.fit(
        train_batches,
        epochs=epochs,
        validation_data=valid_batches,
        callbacks=callbacks,
        verbose=verbose,
    )

    # graph metrics
    plotter = utils.HistoryPlotter(metric="loss", result_subdir=result_subdir)

    try:
        plotter.plot(history)
    except:
        print("Error. Could not save metric's plot")

    # Create MD
    utils.generate_md_file(
        result_subdir,
        config.class_names,
        config.train_record,
        config.valid_record,
        config.network_1,
        config.dataset,
        config.train,
        config.callbacks,
        config.feature_dict,
    )


def train_ensemble_network(epochs=5, initial_lr=1e-3, verbose=2, print_summary=True):

    # Create result subdirectory to store the experiment results
    result_subdir = utils.create_result_subdir(config.result_dir, config.desc)

    # Read dataset with a return structure like this -> ((x1, x2), y)
    if config.train_record is not None:
        train_dataset = tf.data.TFRecordDataset(config.train_record)
    else:
        raise Exception("No path for the Training TfRecord was given.")

    if config.train_record is not None:
        valid_dataset = tf.data.TFRecordDataset(config.valid_record)
    else:
        raise Exception("No path for the Validation TfRecord was given.")

    # Apply the stack of transformations uncommented in config.py
    train_dataset = dataset.map_functions(train_dataset, config.dataset.map_functions)
    valid_dataset = dataset.map_functions(valid_dataset, config.dataset.map_functions)

    # Shuffle, batch and prefetch both datasets
    train_batches = (
        train_dataset.shuffle(config.dataset.shuffle, reshuffle_each_iteration=True)
        .batch(config.dataset.batch_size)
        .prefetch(config.dataset.prefetch)
    )

    valid_batches = (
        valid_dataset.shuffle(config.dataset.shuffle, reshuffle_each_iteration=True)
        .batch(config.dataset.batch_size)
        .prefetch(config.dataset.prefetch)
    )

    # Load the Ensemble Model
    ensemble = models.multiclass_ensemble(**config.networks)

    if print_summary:
        ensemble.summary()

    # Optimizer + compile the model
    adam_optimizer = tf.keras.optimizers.Adam(
        lr=initial_lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False,
    )

    ensemble.compile(
        optimizer=adam_optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    )

    # callbacks
    callbacks = []

    if "decay_lr_after_epoch" in config.callbacks:
        callbacks.append(
            tf.keras.callbacks.LearningRateScheduler(
                callback.decay_lr_on_epoch_end, verbose=1
            )
        )

    if "reduce_lr_on_plateau" in config.callbacks:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                **config.callbacks.reduce_lr_on_plateau
            )
        )

    if "model_checkpoint_callback" in config.callbacks:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(result_subdir, "best_loss_model.h5"),
                **config.callbacks.model_checkpoint_callback
            )
        )

    if "multiple_class_auroc" in config.callbacks:
        callbacks.append(
            callback.MultipleClassAUROC(
                dataset=valid_batches,
                result_subdir=result_subdir,
                **config.callbacks.multiple_class_auroc
            )
        )

    # train
    history = {}
    history[config.network_1.module_name] = ensemble.fit(
        train_batches,
        epochs=epochs,
        validation_data=valid_batches,
        callbacks=callbacks,
        verbose=verbose,
    )

    # graph metrics
    plotter = utils.HistoryPlotter(metric="loss", result_subdir=result_subdir)

    try:
        plotter.plot(history)
    except:
        print("Error. Could not save metric's plot")

    # Create Markdowm file with experiment configuration
    utils.generate_md_file(
        result_subdir,
        config.class_names,
        config.train_record,
        config.valid_record,
        config.network_1,
        config.dataset,
        config.train,
        config.callbacks,
        config.feature_dict,
        network_2_dict=config.network_2,
    )


if __name__ == "__main__":
    utils.init_output_logging()
    tf.random.set_seed(config.random_seed)
    np.random.seed(config.random_seed)
    print("Initializing execution...")
    os.environ.update(config.env)
    print("Running %s()..." % config.train["func"])
    utils.call_func_by_name(**config.train)
    print("\nEnding execution...")

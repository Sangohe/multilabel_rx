import os
import numpy as np
import tensorflow as tf

import utils
import models
import config
import dataset
import callback


def train_single_network(result_subdir=None, epochs=5, initial_lr=1e-3, verbose=0):
    """Trains a default tf.keras network only with the parent nodes from CheXpert"""

    # Make sure the result directory exists and is not none, raise exception if any
    # of the conditions are not met
    if result_subdir is None or not os.path.exists(result_subdir):
        raise Exception("Please make sure to point a directory to save your progress")

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
        .batch(config.dataset.batch)
        .prefetch(config.dataset.prefetch)
    )

    valid_batches = (
        valid_dataset.shuffle(config.dataset.shuffle, reshuffle_each_iteration=True)
        .batch(config.dataset.batch)
        .prefetch(config.dataset.prefetch)
    )

    # load the model and print summary
    model = models.multiclass_model(**config.network)
    model.summary()

    # optimizer + compile the model
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
            reduce_lr_on_plateau=tf.keras.callbacks.ReduceLROnPlateau(
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
    history = model.fit(
        train_batches,
        epochs=epochs,
        validation_data=valid_batches,
        callbacks=callbacks,
        verbose=verbose,
    )


if __name__ == "__main__":
    utils.init_output_logging()
    tf.random.set_seed(config.random_seed)
    np.random.seed(config.random_seed)
    print("Initializing execution...")
    os.environ.update(config.env)
    print("Running %s()..." % config.train["func"])
    result_subdir = utils.create_result_subdir(config.result_dir, config.desc)
    utils.call_func_by_name(result_subdir=result_subdir, **config.train)
    print("Ending execution...")

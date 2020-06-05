import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import utils
import models
import config
import dataset
import callback


def train_single_network(epochs=5, initial_lr=1e-3, verbose=1):
    """Trains a default tf.keras network only with the parent nodes from CheXpert"""

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
    history[config.network.model_name] = model.fit(
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


def evaluate_single_network(
    run_id, test_record=None, class_names=None, metrics=None, log=None
):
    """
    1. Locate the result_subdir using the run_id
    2. Load the test dataset using test_record. Make sure to apply the 
    same transformations read config.txt file in the result_subdir
    3. Feedforward the test dataset and compare the predictions to 
    the ground truth labels
    4. Create a dictionary for the metrics and save it as pickle
    """
    import pickle
    import tensorflow_addons as tfa
    from sklearn.metrics import roc_auc_score

    metrics_class_names = {
        "acc": tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        "precision": tf.keras.metrics.Precision(name="precision"),
        "recall": tf.keras.metrics.Recall(name="recall"),
        "f1": tfa.metrics.F1Score(name="f1_score"),
    }

    result_subdir = utils.locate_result_subdir(run_id)
    if log is not None:
        log_file = os.path.join(result_subdir, log)
    else:
        log_file = os.path.join(result_subdir, "evaluation_log.txt")
    print("Logging output to {}".format(log_file))

    print("\nLoading the model best AUC model...")
    model = tf.keras.models.load_model(os.path.join(result_subdir, "best_auc_model.h5"))
    model.summary()

    print("\nUsing the {} record to evaluate".format(test_record))
    if config.train_record is not None:
        test_dataset = tf.data.TFRecordDataset(test_record)
    else:
        raise Exception("No path for the Test TfRecord was given.")

    # Apply the stack of transformations uncommented in config.py
    test_dataset = dataset.map_functions(test_dataset, config.dataset.map_functions)

    # Shuffle, batch and prefetch both datasets
    test_batches = (
        test_dataset.shuffle(config.dataset.shuffle, reshuffle_each_iteration=True)
        .batch(config.dataset.batch_size)
        .prefetch(config.dataset.prefetch)
    )

    # evaluate metrics
    compile_metrics = [metrics_class_names[m] for m in metrics]
    model.compile(
        optimizer="Adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=compile_metrics,
    )

    evaluation_metrics = model.evaluate(test_batches)
    eval_metrics = {
        key: (round(value, 2) * 100)
        for (key, value) in zip(metrics, ["loss"] + list(evaluation_metrics))
    }

    # append auc score
    eval_metrics["auc"] = {}
    data = list(test_dataset.as_numpy_iterator())
    x = np.asarray([element[0] for element in data])
    y = np.asarray([element[1] for element in data])

    y_hat = model.predict(x)

    if class_names is None:
        class_names = ["Class {}".format(i) for i in range(y.shape[1])]

    current_auroc = []
    for i in range(len(class_names)):
        try:
            score = roc_auc_score(y[:, i], y_hat[:, i])
        except ValueError:
            score = 0
        eval_metrics["auc"][class_names[i]] = round(score, 2) * 100
        current_auroc.append(score)
        print("{:02d}. {:26s} -> {:>8}".format(i + 1, class_names[i], score))

    mean_auroc = np.mean(current_auroc)
    eval_metrics["auc"]["mean"] = round(mean_auroc, 2) * 100
    print("--------------------------------------------------------")
    print("MEAN AUROC: {:>40}".format(mean_auroc))
    print("--------------------------------------------------------\n")

    # save metrics with pickle
    pickle_path = os.path.join(result_subdir, "metrics_on_evaluation.pkl")
    print("Saving metrics to {}".format(pickle_path))
    with open(pickle_path, "wb") as handle:
        pickle.dump(eval_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    utils.init_output_logging()
    tf.random.set_seed(config.random_seed)
    np.random.seed(config.random_seed)
    print("Initializing execution...")
    os.environ.update(config.env)
    print("Running %s()..." % config.train["func"])
    utils.call_func_by_name(**config.train)
    print("\nEnding execution...")

import os
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import roc_auc_score

import utils
import config
import layers
import dataset


def evaluate_single_network(
    run_id, test_record=None, class_names=None, metrics=None, log=None,
):
    """Takes a single experiment id to locate the result subdirectory and
    load the model with the best AUC score for that experiment. Then, the
    model will be tested using the data from the test TfRecord and the
    final metrics will be stored in a pickle.

    Args:
        run_id (int or str): experiment id
        test_record (str): Path to evaluation TfRecord. Defaults to None.
        class_names (list): Ordered list of class names. Defaults to None.
        metrics (list): List of metrics to evaluate. Defaults to None.
        log (str, optional): Filename to write the log. Defaults to None.

    Raises:
        Exception: If the evaluation TfRecord does not exist
    """

    metrics_class_names = {
        "acc": tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        "precision": tf.keras.metrics.Precision(name="precision"),
        "recall": tf.keras.metrics.Recall(name="recall"),
        "f1": tfa.metrics.F1Score(
            num_classes=len(class_names), average="micro", name="f1_score"
        ),
    }

    result_subdir = utils.locate_result_subdir(run_id)
    if log is not None:
        log_file = os.path.join(result_subdir, log)
    else:
        log_file = os.path.join(result_subdir, "evaluation_log.txt")
    print("Logging output to {}".format(log_file))
    utils.set_output_log_file(log_file)

    print("Loading the best AUC model for this experiment...")
    model = tf.keras.models.load_model(os.path.join(result_subdir, "best_auc_model.h5"))

    print("Using the {} record to evaluate\n".format(test_record))
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
    if metrics is None:
        metrics = ["acc", "precision", "recall", "f1"]
    compile_metrics = [metrics_class_names[m] for m in metrics]
    model.compile(
        optimizer="Adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=compile_metrics,
    )

    evaluation_metrics = model.evaluate(test_batches, verbose=2)
    eval_metrics = {
        key: (round(value * 100, 2))
        for (key, value) in zip(["loss"] + metrics, list(evaluation_metrics))
    }

    eval_metrics["loss"] = round(eval_metrics["loss"] / 100.0, 2)

    # append auc score
    eval_metrics["auc"] = {}
    data = list(test_dataset.as_numpy_iterator())
    x = np.asarray([element[0] for element in data])
    y = np.asarray([element[1] for element in data])

    y_hat = model.predict(x)

    if class_names is None:
        class_names = ["Class {}".format(i) for i in range(y.shape[1])]

    current_auroc = []
    print("\n--------------------------------------------------------")
    print("AUROC Score Evaluation for each class in the dataset ")
    print("--------------------------------------------------------")
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


def evaluate_late_fusion_ensemble(
    first_exp_id=None,
    second_exp_id=None,
    test_record=None,
    class_names=None,
    metrics=None,
    log=None,
    use_weighted_average=False,
    valid_record=None,
):
    """Takes two experiment ids, locate both result subdirectories and load
    the model with the best AUC score from each subdirectory to make an
    ensemble. The ensemble can be done either with a normal Average Layer
    or a WeightedAverage Layer. If the WeightedAverage Layer is chosen, 
    the path to the validation TfRecord must be given to adjust the weights.
    At the end, the ensemble will be evaluated using the dataset from the
    test TfRecord and the resulting values will be stored in a pickle.

    Args:
        first_exp_id (int or str): First experiment id. Defaults to None.
        second_exp_id (int or str): Second experiment id. Defaults to None.
        test_record (str): Path to evaluation TfRecord. Defaults to None.
        class_names (list): Ordered list of class names. Defaults to None.
        metrics (list): List of metrics to evaluate. Defaults to None.
        log (str, optional): Filename to write the log. Defaults to None.
        use_weighted_average (bool, optional). Defaults to False.
        valid_record (str, optional): Path to valid TfRecord. Defaults to None.

    Raises:
        Exception: If the validation or evaluation TfRecords do not exist
    """

    result_subdir = utils.create_result_subdir(config.result_dir, config.desc)

    metrics_class_names = {
        "acc": tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        "precision": tf.keras.metrics.Precision(name="precision"),
        "recall": tf.keras.metrics.Recall(name="recall"),
        "f1": tfa.metrics.F1Score(
            num_classes=len(class_names), average="micro", name="f1_score"
        ),
    }

    # Load the trained models and give names to the model
    # and all the layers to avoid errors.
    first_subdir = utils.locate_result_subdir(first_exp_id)
    first_model_path = os.path.join(first_subdir, "best_auc_model.h5")
    print("Loading the first model from {}".format(first_model_path))
    first_model = tf.keras.models.load_model(first_model_path)
    first_model._name = "first_model"
    for layer in first_model.layers:
        layer._name = "first_model_" + layer.name
    first_model.trainable = False

    second_subdir = utils.locate_result_subdir(second_exp_id)
    second_model_path = os.path.join(second_subdir, "best_auc_model.h5")
    print("Loading the second model from {}".format(second_model_path))
    second_model = tf.keras.models.load_model(second_model_path)
    second_model._name = "second_model"
    for layer in second_model.layers:
        layer._name = "second_model_" + layer.name
    second_model.trainable = False

    # Create a layer that will average both predictions
    if use_weighted_average:
        print("Using the Custom WeightedAverage Layer")
        average_layer = layers.WeightedAverage(name="weighted_average_layer")(
            [first_model.output, second_model.output]
        )

        print("Using the {} record to adjust weights\n".format(valid_record))
        if config.train_record is not None:
            valid_dataset = tf.data.TFRecordDataset(valid_record)
        else:
            raise Exception("No path for the Validation TfRecord was given.")

        # Apply the stack of transformations uncommented in config.py
        valid_dataset = dataset.map_functions(
            valid_dataset, config.dataset.map_functions
        )

        # Shuffle, batch and prefetch both datasets
        valid_batches = (
            valid_dataset.shuffle(config.dataset.shuffle, reshuffle_each_iteration=True)
            .batch(config.dataset.batch_size)
            .prefetch(config.dataset.prefetch)
        )
    else:
        print("Using the standard Average Layer")
        average_layer = tf.keras.layers.Average(name="average_layer")(
            [first_model.output, second_model.output]
        )

    # Create the ensemble
    print("Creating the ensemble...")
    ensemble = tf.keras.Model(
        inputs=[first_model.input, second_model.input], outputs=[average_layer]
    )

    if use_weighted_average:
        # optimizer + compile the model
        adam_optimizer = tf.keras.optimizers.Adam(
            lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False,
        )

        ensemble.compile(
            optimizer=adam_optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        )

        # train
        print("Adjusting average weights for the ensemble")
        history = {}
        history["Ensemble"] = ensemble.fit(valid_batches, epochs=2, verbose=2,)

        # graph metrics
        plotter = utils.HistoryPlotter(metric="loss", result_subdir=result_subdir)

        try:
            plotter.plot(history)
        except:
            print("Error. Could not save metric's plot")

    print("Using the {} record to evaluate\n".format(test_record))
    if config.test_record is not None:
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
    if metrics is None:
        metrics = ["acc", "precision", "recall", "f1"]
    compile_metrics = [metrics_class_names[m] for m in metrics]
    ensemble.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=compile_metrics,
    )

    evaluation_metrics = ensemble.evaluate(test_batches, verbose=2)
    eval_metrics = {
        key: (round(value * 100, 2))
        for (key, value) in zip(["loss"] + metrics, list(evaluation_metrics))
    }

    eval_metrics["loss"] = round(eval_metrics["loss"] / 100.0, 2)

    # append auc score
    eval_metrics["auc"] = {}
    data = list(test_dataset.as_numpy_iterator())
    x1 = np.asarray([element[0][0] for element in data])
    x2 = np.asarray([element[0][1] for element in data])
    y = np.asarray([element[1] for element in data])

    y_hat = ensemble.predict([x1, x2])

    if class_names is None:
        class_names = ["Class {}".format(i) for i in range(y.shape[1])]

    current_auroc = []
    print("\n--------------------------------------------------------")
    print("AUROC Score Evaluation for each class in the dataset ")
    print("--------------------------------------------------------")
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

    model_path = os.path.join(result_subdir, "ensemble.h5")
    print("Saving model to {}".format(model_path))
    ensemble.save(model_path)

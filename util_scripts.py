import os
import cv2
import glob
import shutil
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import roc_auc_score

import misc
import config
import layers
import metrics
import dataset


def evaluate_multiclass_model(
    run_id=None,
    model_path="",
    train_record="",
    test_record="",
    class_names=None,
    visuals=False,
):
    # Load the model:
    # Use the run_id if given, otherwise use the model_path
    if run_id is not None:
        result_subdir = misc.locate_result_subdir(run_id)
        model_path = glob.glob(os.path.join(result_subdir, "*.h5"))
    elif os.path.exists(model_path):
        # Create the other_models dir in results if it doesn't exist
        other_path = os.path.join(config.result_dir, "other_models")
        if not os.path.exists(other_path):
            os.makedirs(other_path)
        # Create a subdirectory inside other_path with the model_name
        result_subdir = os.path.join(other_path, model_path.split("/")[-1][:-3])
        if not os.path.exists(result_subdir):
            os.makedirs(result_subdir)
        # Copy the model to result_subdir
        print("Copying model to result subdirectory...")
        shutil.copy(
            model_path, "{}/{}".format(result_subdir, model_path.split("/")[-1])
        )
    else:
        raise FileNotFoundError("Neither the model_path or run_id were provided")

    log_file = os.path.join(result_subdir, "evaluation_log.txt")
    print("Logging output to {}".format(log_file))
    misc.set_output_log_file(log_file)

    print(f"Loading a pretrained Model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    print(f"Loading datasets...")
    if os.path.exists(train_record):
        train_record = tf.data.TFRecordDataset(train_record)
        train_dataset = dataset.map_functions(
            train_record, config.dataset.map_functions
        )
        train_dataset_batches = train_dataset.batch(config.dataset.batch_size).prefetch(
            config.dataset.prefetch
        )

        if visuals:
            train_embeddings = dataset.map_functions(
                train_record, config.dataset.map_functions[:2]
            )
            train_embeddings_batches = train_embeddings.batch(
                config.dataset.batch_size
            ).prefetch(config.dataset.prefetch)
    else:
        print("No train record path were provided. Proceeding without it.")

    if os.path.exists(test_record):
        test_record = tf.data.TFRecordDataset(test_record)
        test_dataset = dataset.map_functions(test_record, config.dataset.map_functions)
        test_dataset_batches = test_dataset.batch(config.dataset.batch_size).prefetch(
            config.dataset.prefetch
        )

        if visuals:
            test_embeddings = dataset.map_functions(
                test_record, config.dataset.map_functions[:2]
            )
            test_embeddings_batches = test_embeddings.batch(
                config.dataset.batch_size
            ).prefetch(config.dataset.prefetch)
    else:
        raise FileNotFoundError(
            "The test record path doesn't exists. You must provide a valid test "
            "path for the evaluation"
        )

    # Evaluation metrics.
    eval_metrics = dict()
    y_true = []
    y_hat = []

    for x_batch, y_batch in test_dataset_batches.as_numpy_iterator():
        y_true.append(y_batch)
        y_hat.append(model.predict(x_batch))

    y_true = np.concatenate(y_true, axis=0)
    y_hat = np.concatenate(y_hat, axis=0)

    if class_names is None:
        class_names = ["Class {}".format(i) for i in range(y.shape[-1])]

    # Multiclass metrics.
    print("Calculating the classificaction metrics...")
    eval_metrics["metrics"] = metrics.multiclass_metrics(y_true, y_hat)
    for key, val in eval_metrics["metrics"].items():
        eval_metrics["metrics"][key] = round(np.mean(val) * 100, 2)

    # ROC Curve.
    fpr, tpr = metrics.multiclass_roc_curve(y_true, y_hat)
    eval_metrics["roc"] = {"fpr": fpr["macro"].tolist(), "tpr": tpr["macro"].tolist()}

    if visuals:
        print("Building model to calculate embeddings...")
        embedding_model = tf.keras.models.clone_model(model)
        embedding_model.set_weights(model.get_weights())

        if misc.is_composite_model(embedding_model):
            print("Removing sigmoid activation from all models' last layer...")
            for layer in embedding_model.layers:
                if isinstance(layer, tf.python.keras.engine.training.Model):
                    for sub_layer in layer.layers:
                        if "predictions" in sub_layer.name:
                            sub_layer.activation = tf.keras.activations.linear
        else:
            print("Removing sigmoid activation from the last layer")
            for layer in embedding_model.layers:
                if "predictions" in layer.name:
                    layer.activation = tf.keras.activations.linear

        labels = []
        umap_predictions = []
        test_predictions = []
        train_predictions = []

        print("Predicting on test batches...")
        for x_batch, y_batch in test_dataset_batches.as_numpy_iterator():
            umap_predictions.append(embedding_model.predict(x_batch))
            test_predictions.append(model.predict(x_batch))
        for x_batch, y_batch in test_embeddings_batches.as_numpy_iterator():
            labels.append(y_batch)

        print("Predicting on train batches...")
        for x_batch, y_batch in train_dataset_batches.as_numpy_iterator():
            umap_predictions.append(embedding_model.predict(x_batch))
            train_predictions.append(model.predict(x_batch))
        for x_batch, y_batch in train_embeddings_batches.as_numpy_iterator():
            labels.append(y_batch)

        labels = np.concatenate(labels, axis=0)
        umap_predictions = np.concatenate(umap_predictions, axis=0)
        test_predictions = np.concatenate(test_predictions, axis=0)
        train_predictions = np.concatenate(train_predictions, axis=0)

        print("Calculating embeddings for the train and test dataset...")
        transformer, umap_points = misc.umap_points(umap_predictions)
        eval_metrics["transformer"] = transformer
        eval_metrics["embedded"] = {
            "x": umap_points[:, 0].tolist(),
            "y": umap_points[:, 1].tolist(),
            "z": umap_points[:, 2].tolist(),
            "label": labels.tolist(),
        }

        print("Saving points for violin plot...")
        eval_metrics["violin"] = {
            "train": train_predictions.tolist(),
            "test": test_predictions.tolist(),
        }

        print("Saving points for gaussian plot...")
        joint_predictions = np.concatenate(
            (test_predictions, train_predictions), axis=0
        )
        eval_metrics["gaussian"] = {
            "mean": joint_predictions.mean(axis=0).tolist(),
            "std": joint_predictions.std(axis=0).tolist(),
            "probs": joint_predictions.tolist(),
        }

    # Serialize metrics with Pickle.
    pickle_path = os.path.join(result_subdir, "metrics_on_evaluation.pkl")
    print("Saving metrics to {}".format(pickle_path))
    with open(pickle_path, "wb") as handle:
        pickle.dump(eval_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)


def evaluate_single_network(
    run_id, test_record=None, class_names=None, log=None,
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

    result_subdir = misc.locate_result_subdir(run_id)
    if log is not None:
        log_file = os.path.join(result_subdir, log)
    else:
        log_file = os.path.join(result_subdir, "evaluation_log.txt")
    print("Logging output to {}".format(log_file))
    misc.set_output_log_file(log_file)

    print("Loading the best AUC model for this experiment...")
    model = tf.keras.models.load_model(os.path.join(result_subdir, "best_auc_model.h5"))

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

    # Evaluation metrics.
    y = []
    y_hat = []

    for x_batch, y_batch in test_batches.as_numpy_iterator():
        y.append(y_batch)
        y_hat.append(model.predict(x_batch))

    y = np.concatenate(y, axis=0)
    y_hat = np.concatenate(y_hat, axis=0)

    if class_names is None:
        class_names = ["Class {}".format(i) for i in range(y.shape[-1])]

    eval_metrics = metrics.multiclass_metrics(y, y_hat)

    # Serialize metrics with Pickle.
    pickle_path = os.path.join(result_subdir, "metrics_on_evaluation.pkl")
    print("Saving metrics to {}".format(pickle_path))
    with open(pickle_path, "wb") as handle:
        pickle.dump(eval_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)


def evaluate_late_fusion_ensemble(
    run_id=None,
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

    if run_id is not None:
        result_subdir = misc.locate_result_subdir(run_id)
        if log is not None:
            log_file = os.path.join(result_subdir, log)
        else:
            log_file = os.path.join(result_subdir, "evaluation_log.txt")
        print("Logging output to {}".format(log_file))
        misc.set_output_log_file(log_file)

        print("Loading the best AUC model for this experiment...")
        model_path = glob.glob(os.path.join(result_subdir, "*.h5"))[0]
        ensemble = tf.keras.models.load_model(
            model_path, custom_objects={"WeightedAverage": layers.WeightedAverage}
        )
    else:
        result_subdir = misc.create_result_subdir(config.result_dir, config.desc)

        # Load the trained models and give names to the model
        # and all the layers to avoid errors.
        first_subdir = misc.locate_result_subdir(first_exp_id)
        first_model_path = os.path.join(first_subdir, "best_auc_model.h5")
        print("Loading the first model from {}".format(first_model_path))
        first_model = tf.keras.models.load_model(first_model_path)
        first_model._name = "first_model"
        for layer in first_model.layers:
            layer._name = "first_model_" + layer.name
        first_model.trainable = False

        second_subdir = misc.locate_result_subdir(second_exp_id)
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
                valid_dataset.shuffle(
                    config.dataset.shuffle, reshuffle_each_iteration=True
                )
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
                lr=1e-4,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=None,
                decay=0.0,
                amsgrad=False,
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
            plotter = misc.HistoryPlotter(metric="loss", result_subdir=result_subdir)

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

    # Evaluation metrics.
    y = []
    y_hat = []

    for x_batch, y_batch in test_batches.as_numpy_iterator():
        y.append(y_batch)
        y_hat.append(ensemble.predict(x_batch))

    y = np.concatenate(y, axis=0)
    y_hat = np.concatenate(y_hat, axis=0)

    if class_names is None:
        class_names = ["Class {}".format(i) for i in range(y.shape[-1])]

    eval_metrics = metrics.multiclass_metrics(y, y_hat)

    # Serialize metrics with Pickle.
    pickle_path = os.path.join(result_subdir, "metrics_on_evaluation.pkl")
    print("Saving metrics to {}".format(pickle_path))
    with open(pickle_path, "wb") as handle:
        pickle.dump(eval_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

    model_path = os.path.join(result_subdir, "ensemble.h5")
    print("Saving model to {}".format(model_path))
    ensemble.save(model_path)


def generate_cams(
    model_path=None, run_id=None, image_path=None, scale_func=None, class_names=None,
):
    """This function generates Class Activation Maps using a single Model"""

    # Load the model:
    # Read directly from the model_path, otherwise use the run_id
    if os.path.exists(model_path):
        # Create the other_models dir in results if it doesn't exist
        other_path = os.path.join(config.result_dir, "other_models")
        if not os.path.exists(other_path):
            os.makedirs(other_path)
        # Create a subdirectory inside other_path with the model_name
        result_subdir = os.path.join(other_path, model_path.split("/")[-1][:-3])
        if not os.path.exists(result_subdir):
            os.makedirs(result_subdir)
    elif run_id is not None:
        result_subdir = misc.locate_result_subdir(run_id)
        model_path = glob.glob(os.path.join(result_subdir, "*.h5"))
    else:
        raise FileNotFoundError("Neither the model_path or run_id were provided")

    log_file = os.path.join(result_subdir, "cams.txt")
    print("Logging output to {}".format(log_file))
    misc.set_output_log_file(log_file)

    print(f"Loading the pretrained Model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Create the graph using the Model's layers.
    class_weights = model.layers[-1].get_weights()[0]
    final_conv_layer = model.get_layer("bn")
    get_output = kb.function(
        [model.layers[0].input], [final_conv_layer.output, model.layers[-1].output]
    )

    if os.path.exists(image_path):
        # Save path for CAMs
        save_path = os.path.join(
            result_subdir, image_path.split("/")[-1].replace(".", "_")
        )

        # Read the image and generate just the CAM.
        image_raw = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        scale_func = "dataset.scale_imagenet_np" if scale_func is None else scale_func
        print(f"Using {scale_func} function to preprocess the image.")
        image = misc.call_func_by_name(scale_func)(image_rgb)

        # Execute the tf function.
        [conv_outputs, predictions] = get_output(image)
        conv_outputs = conv_outputs[0, ...]

        # Create CAM for each class
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(class_weights))]
        for idx, cls in enumerate(class_names):
            cam = np.zeros(dtype=np.float32, shape=(conv_outputs.shape[:2]))
            for i, w in enumerate(class_weights[idx]):
                cam += w * conv_outputs[:, :, i]
            cam /= np.max(cam)
            cam = cv2.resize(cam, (image_rgb.shape[:2]))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap[np.where(cam < 0.2)] = 0
            image = heatmap * 0.5 + image_rgb

            # Put some text to recognize to what class the CAM belongs.
            cv2.putText(
                image,
                text=f"{cls}",
                org=(20, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2.0,
                color=(255, 255, 255),
                thickness=1,
            )

            cam_path = os.path.join(save_path, f"{cls}_cam.png")
            cv2.imwrite(cam_path, image)
            print(f"{cls} Class Activation Map saved to: {cam_path}")
    else:
        raise FileNotFoundError("Image path doesn't exists")


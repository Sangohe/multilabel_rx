import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


def continuous_to_binary(y_hat, threshold=0.5):
    """Maps each continuous predicted value from `y_hat` into a binarized value by 
    comparing it to a given threshold. If the value is greater than the threshold 
    the binarized value will be 1.0, otherwise the binarized value will be 0.0

    Args:
        y_hat (array): Array of continuous values
        threshold (float, optional): Defaults to 0.5.

    Returns:
        array: Numpy array of binarized values
    """
    return np.where(y_hat > threshold, 1.0, 0.0)


def multiclass_metrics(y_true, y_hat):
    """Compares the true and predicted values from a Model to calculate a set of 
    useful classification metrics. `y_true` and `y_hat` must be of the same size

    Args:
        y_true (array): ground truth values
        y_hat ([type]): predicted values

    Returns:
        dictionary: Python dictionary with the calculated metrics for each class
    """

    metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "specificity": [],
        "f1_score": [],
        "auc": [],
    }

    y_bin = continuous_to_binary(y_hat)
    for i in range(y_true.shape[1]):
        tn, fp, fn, tp = confusion_matrix(y_true[:, i], y_bin[:, i]).ravel()

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        f1_score = (precision * recall) / (precision + recall)
        auc = roc_auc_score(y_true[:, i], y_hat[:, i])

        metrics["accuracy"].append(accuracy)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["specificity"].append(specificity)
        metrics["f1_score"].append(f1_score)
        metrics["auc"].append(auc)

    return metrics

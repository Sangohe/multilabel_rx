import numpy as np
from sklearn.metrics import roc_curve
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


def multiclass_roc_curve(y_true, y_hat):
    """Compares the true and predicted values to calculate the ROC AUC 
    Curves for each class.

    Args:
        y_true (array): ground truth values
        y_hat ([type]): predicted values

    Returns:
        dictionaries: two dictionaries with both the false positive rates
        and true positive rates for each class and also the micro and 
        macro averages
    """

    fpr = dict()
    tpr = dict()

    n_classes = y_true.shape[-1]

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_hat[:, i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_hat.ravel())

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr

    return fpr, tpr

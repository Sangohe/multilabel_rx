import os
import numpy as np
import tensorflow as tf

import tensorflow.keras.backend as kb
from sklearn.metrics import roc_auc_score


def decay_lr_on_epoch_end(epoch):
    init_lr = 1e-3
    factor = 0.1
    return init_lr * (factor ** epoch)


class MultipleClassAUROC(tf.keras.callbacks.Callback):
    """
    Monitor mean AUROC and update model
    """

    def __init__(self, dataset, class_names=None, result_subdir=None, stats=None):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.dataset = dataset
        self.class_names = class_names
        self.save_model_path = os.path.join(result_subdir, "best_auc_model.h5")

        if stats:
            self.stats = stats
        else:
            self.stats = {"best_mean_auroc": 0}

        # aurocs log
        self.aurocs = {}
        for c in self.class_names:
            self.aurocs[c] = []

    def on_epoch_end(self, epoch, logs={}):
        """
        Calculate the average AUROC and save the best model weights according
        to this metric.
        """
        self.stats["lr"] = float(kb.eval(self.model.optimizer.lr))
        print("\n--------------------------------------------------------")
        print("Epoch #{} Validation AUROC. Current Lr: {:.9f}".format(epoch + 1, self.stats["lr"]))
        print("--------------------------------------------------------")
        
        data = list(self.dataset.unbatch().as_numpy_iterator())
        x = np.asarray([element[0] for element in data])
        y = np.asarray([element[1] for element in data])

        y_hat = self.model.predict(x)

        current_auroc = []
        for i in range(len(self.class_names)):
            try:
                score = roc_auc_score(y[:, i], y_hat[:, i])
            except ValueError:
                score = 0
            self.aurocs[self.class_names[i]].append(score)
            current_auroc.append(score)
            print("{:02d}. {:26s} -> {:>8}".format(i+1, self.class_names[i], score))
        
        mean_auroc = np.mean(current_auroc)
        print("--------------------------------------------------------")
        print("MEAN AUROC: {:>40}".format(mean_auroc))
        print("--------------------------------------------------------\n")

        # customize your multiple class metrics here
        if mean_auroc > self.stats["best_mean_auroc"]:
            print(
                "Update best AUROC from {} to {}. Saving model to:\n{}\n".format(
                    self.stats["best_mean_auroc"], mean_auroc, self.save_model_path
                )
            )
            self.model.save(self.save_model_path)
        return

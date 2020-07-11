import os
import sys
import cv2
import glob
import mdutils
import importlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


import config


# ------------------------------------------------------------------------------------------------------
# Utilities for importing modules and objects by name.


def import_module(module_or_obj_name):
    parts = module_or_obj_name.split(".")
    parts[0] = {"np": "numpy", "tf": "tensorflow"}.get(parts[0], parts[0])
    for i in range(len(parts), 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:i]))
            relative_obj_name = ".".join(parts[i:])
            return module, relative_obj_name
        except ImportError:
            pass
    raise ImportError(module_or_obj_name)


def find_obj_in_module(module, relative_obj_name):
    obj = module
    for part in relative_obj_name.split("."):
        obj = getattr(obj, part)
    return obj


def import_obj(obj_name):
    module, relative_obj_name = import_module(obj_name)
    return find_obj_in_module(module, relative_obj_name)


def call_func_by_name(*args, func=None, **kwargs):
    assert func is not None
    return import_obj(func)(*args, **kwargs)


# ------------------------------------------------------------------------------------------------------
# Create result subdirectory


def create_result_subdir(result_dir, desc):
    # Select run ID and create subdir.
    while True:
        run_id = 0
        for fname in glob.glob(os.path.join(result_dir, "*")):
            try:
                fbase = os.path.basename(fname)
                ford = int(fbase[: fbase.find("-")])
                run_id = max(run_id, ford + 1)
            except ValueError:
                pass

        result_subdir = os.path.join(result_dir, "%03d-%s" % (run_id, desc))
        try:
            os.makedirs(result_subdir)
            break
        except OSError:
            if os.path.isdir(result_subdir):
                continue
            raise

    print("Saving results to", result_subdir)
    set_output_log_file(os.path.join(result_subdir, "log.txt"))

    # Export config.
    try:
        with open(os.path.join(result_subdir, "config.txt"), "wt") as fout:
            for k, v in sorted(config.__dict__.items()):
                if not k.startswith("_"):
                    fout.write("%s = %s\n" % (k, str(v)))
    except:
        pass

    return result_subdir


def locate_result_subdir(run_id_or_result_subdir):
    if isinstance(run_id_or_result_subdir, str) and os.path.isdir(
        run_id_or_result_subdir
    ):
        return run_id_or_result_subdir

    searchdirs = []
    searchdirs += [""]
    searchdirs += ["results"]
    searchdirs += ["networks"]

    for searchdir in searchdirs:
        dir = (
            config.result_dir
            if searchdir == ""
            else os.path.join(config.result_dir, searchdir)
        )
        dir = os.path.join(dir, str(run_id_or_result_subdir))
        if os.path.isdir(dir):
            return dir
        prefix = (
            "%03d" % run_id_or_result_subdir
            if isinstance(run_id_or_result_subdir, int)
            else str(run_id_or_result_subdir)
        )
        dirs = sorted(
            glob.glob(os.path.join(config.result_dir, searchdir, prefix + "-*"))
        )
        dirs = [dir for dir in dirs if os.path.isdir(dir)]
        if len(dirs) == 1:
            return dirs[0]
    raise IOError("Cannot locate result subdir for run", run_id_or_result_subdir)


# ------------------------------------------------------------------------------------------------------
# Logging of stdout and stderr to a file.


class OutputLogger(object):
    def __init__(self):
        self.file = None
        self.buffer = ""

    def set_log_file(self, filename, mode="wt"):
        assert self.file is None
        self.file = open(filename, mode)
        if self.buffer is not None:
            self.file.write(self.buffer)
            self.buffer = None

    def write(self, data):
        if self.file is not None:
            self.file.write(data)
        if self.buffer is not None:
            self.buffer += data

    def flush(self):
        if self.file is not None:
            self.file.flush()


class TeeOutputStream(object):
    def __init__(self, child_streams, autoflush=False):
        self.child_streams = child_streams
        self.autoflush = autoflush

    def write(self, data):
        for stream in self.child_streams:
            stream.write(data)
        if self.autoflush:
            self.flush()

    def flush(self):
        for stream in self.child_streams:
            stream.flush()


output_logger = None


def init_output_logging():
    global output_logger
    if output_logger is None:
        output_logger = OutputLogger()
        sys.stdout = TeeOutputStream([sys.stdout, output_logger], autoflush=True)
        sys.stderr = TeeOutputStream([sys.stderr, output_logger], autoflush=True)


def set_output_log_file(filename, mode="wt"):
    if output_logger is not None:
        output_logger.set_log_file(filename, mode)


# ------------------------------------------------------------------------------------------------------
# Plot graphics -> copied from tensorflow_docs.plots
# https://github.com/tensorflow/docs/blob/master/tools/tensorflow_docs/plots/__init__.py

prop_cycle = plt.rcParams["axes.prop_cycle"]
COLOR_CYCLE = prop_cycle.by_key()["color"]


def _smooth(values, std):
    """Smooths a list of values by convolving with a gussian.
  Assumes equal spacing.
  Args:
    values: A 1D array of values to smooth.
    std: The standard devistion of the gussian. The units are array elements.
  Returns:
    The smoothed array.
  """
    width = std * 4
    x = np.linspace(-width, width, 2 * width + 1)
    kernel = np.exp(-((x / 5) ** 2))

    values = np.array(values)
    weights = np.ones_like(values)

    smoothed_values = np.convolve(values, kernel, mode="same")
    smoothed_weights = np.convolve(weights, kernel, mode="same")

    return smoothed_values / smoothed_weights


class HistoryPlotter(object):
    """A class for plotting named set of keras-histories.
  The class maintains colors for each key from plot to plot.
  """

    def __init__(self, metric=None, result_subdir=None, smoothing_std=None):
        self.color_table = {}
        self.metric = metric
        self.result_subdir = result_subdir
        self.smoothing_std = smoothing_std

    def plot(self, histories, metric=None, smoothing_std=None):
        """Plots a {name: history} dictionary of keras histories.
    Colors are assigned to the name-key, and maintained from call to call.
    Training metrics are shown as a solid line, validation metrics dashed.
    Args:
      histories: {name: history} dictionary of keras histories.
      metric: which metric to plot from all the histories.
      smoothing_std: the standard-deviaation of the smoothing kernel applied
        before plotting. The units are in array-indices.
    """
        if metric is None:
            metric = self.metric
        if smoothing_std is None:
            smoothing_std = self.smoothing_std

        for name, history in histories.items():
            # Remember name->color asociations.
            if name in self.color_table:
                color = self.color_table[name]
            else:
                color = COLOR_CYCLE[len(self.color_table) % len(COLOR_CYCLE)]
                self.color_table[name] = color

            train_value = history.history[metric]
            val_value = history.history["val_" + metric]

            if smoothing_std is not None:
                train_value = _smooth(train_value, std=smoothing_std)
                val_value = _smooth(val_value, std=smoothing_std)

            plt.plot(
                history.epoch, train_value, color=color, label=name.title() + " Train"
            )
            plt.plot(
                history.epoch, val_value, "--", label=name.title() + " Val", color=color
            )

        plt.xlabel("Epochs")
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend()

        plt.xlim([0, max([history.epoch[-1] for name, history in histories.items()])])
        plt.grid(True)
        if self.result_subdir is not None:
            plt.savefig(os.path.join(self.result_subdir, "metrics_on_training.png"))
            print(
                "\nSaving metric's plot in {}".format(
                    os.path.join(self.result_subdir, "metrics_on_training.png")
                )
            )


# ------------------------------------------------------------------------------------------------------
# Image utilities


def create_and_save_heatmap(
    image, cam, label, save_dir, filename="", threshold=0.2, annotations=None
):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < threshold)] = 0
    heatmap_image = heatmap * 0.5 + image

    cv2.putText(
        heatmap_image,
        text=label,
        org=(5, 20),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(255, 255, 255),
        thickness=1,
    )

    if annotations is not None:
        cam_path = os.path.join(save_dir, f"{filename}_{label}.png")
        cv2.rectangle(
            heatmap_image,
            (int(annotations["x1"]), int(annotations["y1"])),
            (int(annotations["x2"]), int(annotations["y2"])),
            (255, 0, 0),
            2,
        )
    else:
        cam_path = os.path.join(save_dir, f"{label}_cam.png")

    cv2.imwrite(cam_path, heatmap_image)
    print(f"Class Activation Map saved to: {cam_path}")


# ------------------------------------------------------------------------------------------------------
# Model utilities


def is_composite_model(model):
    """Takes a tf.keras.Model and check if there are multiple Models
    in its architecture to return True, otherwise returns False"""
    num_of_models = 0
    for layer in model.layers:
        if isinstance(layer, tf.python.keras.engine.training.Model):
            num_of_models += 1
        if num_of_models >= 2:
            return True
    return False


def get_layer_by_substring(model, substring):
    layer_dict = {layer.name: layer for layer in model.layers}
    for layer_name in layer_dict:
        if substring in layer_name:
            return layer_dict[layer_name]
    raise Exception(
        f"None of the layers from {model.name} have {substring} in their name"
    )


# ------------------------------------------------------------------------------------------------------
# Embeddings


def umap_points(predictions):
    """Transforms embeddings into points in a 3D space using UMAP"""
    from umap import UMAP

    transformer = UMAP(
        n_neighbors=20,
        min_dist=0.5,
        n_components=3,
        metric="euclidean",
        transform_seed=8128,
    )
    umap_points = transformer.fit_transform(predictions)

    return transformer, umap_points

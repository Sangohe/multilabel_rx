import os
import numpy as np
import tensorflow as tf

import utils
import models
import config
import dataset


def conditional_training():
    """Trains a default tf.keras network only with the parent nodes from CheXpert"""
    model = models.multiclass_model()
    model.summary()


if __name__ == "__main__":
    utils.init_output_logging()
    tf.random.set_seed(config.random_seed)
    np.random.seed(config.random_seed)
    print("Initializing execution...")
    os.environ.update(config.env)
    print("Running %s()..." % config.train["func"])
    result_subdir = utils.create_result_subdir(config.result_dir, config.desc)
    conditional_training()
    print("Ending execution...")

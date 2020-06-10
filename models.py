import importlib
import numpy as np
import tensorflow as tf


def multiclass_model(
    module_name="DenseNet121",
    input_shape=(224, 224, 3),
    n_classes=14,
    model_name="",
    use_base_weights=True,
    weights_path=None,
    freeze=False,
):
    """
    returns a multiclass model using as the base model one
    of the default tf.keras.networks
    """
    base_weights = "imagenet" if use_base_weights else None
    base_model_class = getattr(
        importlib.import_module("tensorflow.keras.applications"), module_name,
    )
    img_input = tf.keras.Input(shape=input_shape)
    base_model = base_model_class(
        include_top=False,
        input_tensor=img_input,
        input_shape=input_shape,
        weights=base_weights,
        pooling="avg",
    )

    x = base_model.output
    predictions = tf.keras.layers.Dense(
        n_classes, activation="sigmoid", name="predictions"
    )(x)
    model = tf.keras.Model(inputs=img_input, outputs=predictions)

    # Rename model
    model._name = model_name
    for layer in model.layers:
        layer._name = model_name + "_" + layer.name

    if weights_path is not None:
        print("Load model weights path: {}".format(weights_path))
        model.load_weights(weights_path)

    if freeze:
        for layer in model.layers[:-2]:
            layer.trainable = False

    return model

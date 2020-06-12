import importlib
import tensorflow as tf

import layers


def multiclass_model(
    module_name="DenseNet121",
    input_shape=(224, 224, 3),
    use_base_weights=True,
    n_classes=14,
    model_path=None,
    weights_path=None,
    model_name="",
    freeze=False,
):
    """
    returns a multiclass model using as the base model one
    of the default tf.keras.networks
    """

    if model_path is not None:
        print(f"Creating Model: Loading an already built model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
    else:
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

        if weights_path is not None:
            print(f"Creating Model: Using the weights from: {weights_path}")
            model.load_weights(weights_path)
        else:
            print(
                "Creating Model: Using the weights from a pretrained Model on Imagenet"
            )

    # Rename model
    model._name = model_name
    for layer in model.layers:
        layer._name = model_name + "_" + layer.name

    if freeze:
        for layer in model.layers[:-2]:
            layer.trainable = False

    return model


def multiclass_ensemble(
    networks=[],
    model_path=None,
    weights_path=None,
    use_weighted_average=False,
    freeze=False,
):
    """
    returns a multiclass ensemble model using N python dictionaries
    to create N default tf.keras.networks. The final layer will be
    a Custom WeightedAverage Layer if use_weighted_average is True.
    Otherwise, a Standard Average Layer will be used
    """

    if model_path is not None:
        print(f"\nLoading an already built model from: {model_path}")
        ensemble = tf.keras.models.load_model(
            model_path, custom_objects={"WeightedAverage": layers.WeightedAverage}
        )
    else:
        # Create all the models. Throw an error if there is no more than
        # one network in the list
        if len(networks) > 1:
            models = [multiclass_model(**net) for net in networks]
        else:
            raise Exception(
                "You need two or more networks to make an ensemble. "
                "Check config.networks"
            )

        # Create a layer that will average both predictions
        if use_weighted_average:
            print("\nUsing the Custom WeightedAverage Layer")
            average_layer = layers.WeightedAverage(name="weighted_average_layer")(
                [model.output for model in models]
            )
        else:
            print("\nUsing the standard Average Layer")
            average_layer = tf.keras.layers.Average(name="average_layer")(
                [model.output for model in models]
            )

        if weights_path is not None:
            print(f"Creating Ensemble: Using the weights from: {weights_path}")
            ensemble.load_weights(weights_path)
        else:
            print("Creating Ensemble: All the ensemble's models preserve their weights")

        # Create the ensemble
        ensemble = tf.keras.Model(
            inputs=[model.input for model in models], outputs=[average_layer]
        )

        print("The Ensemble Model has been successfully created")

    # Freeze the Ensemble Model and unfreeze the predictions and average layers
    if freeze:
        ensemble.trainable = False
        for layer in ensemble.layers:
            if (
                "average" in layer.name
                or "predictions" in layer.name
                or "avg_pool" in layer.name
            ):
                layer.trainable = True

    return ensemble

import importlib
import tensorflow as tf

import utils
import config

# --------------------------------------------------------------------------------
# Feature dictionaries


def rx_chest14(n_diseases=8):
    """Function to get the encoding dictionary for RX-ChestX-Ray14 dataset"""
    encode_dict = {
        "img": tf.io.FixedLenFeature([], tf.string),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "channels": tf.io.FixedLenFeature([], tf.int64),
        "diseases": tf.io.FixedLenFeature([n_diseases], tf.int64),
    }
    return encode_dict


def rx_chexpert(n_diseases=8):
    """Function to get the encoding dictionary for RX-CheXpert dataset"""
    encode_dict = {
        "img": tf.io.FixedLenFeature([], tf.string),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "channels": tf.io.FixedLenFeature([], tf.int64),
        "diseases": tf.io.FixedLenFeature([n_diseases], tf.int64),
    }
    return encode_dict


def rx_chexpert_multiview(n_diseases=8):
    """Function to get the encoding dictionary for RX-CheXpert dataset"""
    encode_dict = {
        "img_frontal": tf.io.FixedLenFeature([], tf.string),
        "img_lateral": tf.io.FixedLenFeature([], tf.string),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "channels": tf.io.FixedLenFeature([], tf.int64),
        "diseases": tf.io.FixedLenFeature([n_diseases], tf.int64),
    }
    return encode_dict


encoding_dictionary = utils.call_func_by_name(**config.feature_dict)

# --------------------------------------------------------------------------------
# Mapping functions


def from_bytes_to_dict(example_bytes):
    """Transform bytes to dictionary"""
    return tf.io.parse_single_example(example_bytes, encoding_dictionary)


def extract_data_from_dict(example_dict):
    """Takes an example dictionary and returns an image with its corresponding labels"""
    image = tf.io.decode_raw(example_dict["img"], tf.uint8)
    image = tf.reshape(
        image, (example_dict["height"], example_dict["width"], example_dict["channels"])
    )

    return image, example_dict["diseases"]


def scale_0(image, label):
    """Takes and image and scale its values between [0, 1]"""
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.image.resize(
        image, [config.network.input_shape[0], config.network.input_shape[0]]
    )
    return image, tf.cast(label, tf.float32)


def scale_minus1_1(image, label):
    """Takes an image and scale its values between [-1, 1]"""
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(
        image, [config.network.input_shape[0], config.network.input_shape[0]]
    )

    return image, tf.cast(label, tf.float32)


def scale_imagenet(image, label):
    """
    Takes an image and scale its values using the Imagenet dataset mean and 
    standard deviation
    """
    imagenet_mean = tf.constant([0.485, 0.456, 0.406])
    imagenet_std = tf.constant([0.229, 0.224, 0.225])

    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.image.resize(
        image, [config.network.input_shape[0], config.network.input_shape[0]]
    )
    image = (image - imagenet_mean) / imagenet_std

    return image, tf.cast(label, tf.float32)


def horizontal_flipping_aug(image, label):
    """Applies a random horizontal flipping to the image"""
    transformed_image = tf.image.random_flip_left_right(image)
    return transformed_image, label


def upolicy(image, label):
    """
    Takes a vector of labels and maps all values equal to -1 to
    0 or 1, depending on the chosen policy in config
    """
    return (
        image,
        tf.where(
            label == -1.0,
            tf.constant(config.minval, dtype=tf.float32),
            tf.cast(label, dtype=tf.float32),
        ),
    )


def label_smoothing(image, label):
    """
    Takes a label value of -1 and maps it into a [minval, maxval] 
    depending on the chosen policy in config
    """
    return (
        image,
        tf.where(
            label == -1.0,
            tf.random.uniform(
                [], minval=config.minval, maxval=config.maxval, dtype=tf.float32
            ),
            tf.cast(label, tf.float32),
        ),
    )


# --------------------------------------------------------------------------------
# Multiview Mapping functions


def extract_data_from_dict_multiview(example_dict):
    """Takes an example dictionary and returns an image with its corresponding labels"""
    frontal_image = tf.io.decode_raw(example_dict["img_frontal"], tf.uint8)
    frontal_image = tf.reshape(
        frontal_image,
        (example_dict["height"], example_dict["width"], example_dict["channels"]),
    )

    lateral_image = tf.io.decode_raw(example_dict["img_lateral"], tf.uint8)
    lateral_image = tf.reshape(
        lateral_image,
        (example_dict["height"], example_dict["width"], example_dict["channels"]),
    )

    return frontal_image, lateral_image, example_dict["diseases"]


def scale_0_multiview(frontal_image, lateral_image, label):
    """Takes and image and scale its values between [0, 1]"""
    frontal_image = tf.cast(frontal_image, tf.float32)
    frontal_image = frontal_image / 255.0
    frontal_image = tf.image.resize(
        frontal_image, [config.network.input_shape[0], config.network.input_shape[0]]
    )

    lateral_image = tf.cast(lateral_image, tf.float32)
    lateral_image = lateral_image / 255.0
    lateral_image = tf.image.resize(
        lateral_image, [config.network.input_shape[0], config.network.input_shape[0]]
    )
    return frontal_image, lateral_image, tf.cast(label, tf.float32)


def scale_minus1_1_multiview(frontal_image, lateral_image, label):
    """Takes an image and scale its values between [-1, 1]"""
    frontal_image = tf.cast(frontal_image, tf.float32)
    frontal_image = (frontal_image / 127.5) - 1
    frontal_image = tf.image.resize(
        frontal_image, [config.network.input_shape[0], config.network.input_shape[0]]
    )

    lateral_image = tf.cast(lateral_image, tf.float32)
    lateral_image = (lateral_image / 127.5) - 1
    lateral_image = tf.image.resize(
        lateral_image, [config.network.input_shape[0], config.network.input_shape[0]]
    )

    return frontal_image, lateral_image, tf.cast(label, tf.float32)


def scale_imagenet_multiview(frontal_image, lateral_image, label):
    """
    Takes an image and scale its values using the Imagenet dataset mean and 
    standard deviation
    """
    imagenet_mean = tf.constant([0.485, 0.456, 0.406])
    imagenet_std = tf.constant([0.229, 0.224, 0.225])

    frontal_image = tf.cast(frontal_image, tf.float32)
    frontal_image = frontal_image / 255.0
    frontal_image = tf.image.resize(
        frontal_image, [config.network.input_shape[0], config.network.input_shape[0]]
    )
    frontal_image = (frontal_image - imagenet_mean) / imagenet_std

    lateral_image = tf.cast(lateral_image, tf.float32)
    lateral_image = lateral_image / 255.0
    lateral_image = tf.image.resize(
        lateral_image, [config.network.input_shape[0], config.network.input_shape[0]]
    )
    lateral_image = (lateral_image - imagenet_mean) / imagenet_std

    return frontal_image, lateral_image, tf.cast(label, tf.float32)


def horizontal_flipping_aug_multiview(frontal_image, lateral_image, label):
    """Applies a random horizontal flipping to the image"""
    frontal_image = tf.image.random_flip_left_right(frontal_image)
    lateral_image = tf.image.random_flip_left_right(lateral_image)
    return frontal_image, lateral_image, label


def upolicy_multiview(frontal_image, lateral_image, label):
    """
    Takes a vector of labels and maps all values equal to -1 to
    0 or 1, depending on the chosen policy in config
    """
    return (
        frontal_image,
        lateral_image,
        tf.where(
            label == -1.0,
            tf.constant(config.minval, dtype=tf.float32),
            tf.cast(label, dtype=tf.float32),
        ),
    )


def label_smoothing_multiview(frontal_image, lateral_image, label):
    """
    Takes a label value of -1 and maps it into a [minval, maxval] 
    depending on the chosen policy in config
    """
    return (
        frontal_image,
        lateral_image,
        tf.where(
            label == -1.0,
            tf.random.uniform(
                [], minval=config.minval, maxval=config.maxval, dtype=tf.float32
            ),
            tf.cast(label, tf.float32),
        ),
    )


# --------------------------------------------------------------------------------
# Apply functions


def map_functions(dataset, funcs, num_parallel_calls=24):
    """Map all the elements in dataset using each function in funcs"""
    for func in funcs:
        dataset = dataset.map(utils.import_obj(func), num_parallel_calls)
    return dataset

import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils


class WeightedAverage(tf.keras.layers.Average):
    """Layer that computes an element-wise weighted average for a list of inputs

    It takes as input a list of tensors, all of the same shape, and returns a 
    single tensor (also of the same shape)

    Raises:
    ValueError: If there is a shape mismatch between the inputs and the shapes
    cannot be broadcasted to match.
    """

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape[0], tuple):
            raise ValueError("A merge layer should be called on a list of inputs.")
        if len(input_shape) < 2:
            raise ValueError(
                "A merge layer should be called "
                "on a list of at least 2 inputs. "
                "Got " + str(len(input_shape)) + " inputs."
            )
        batch_sizes = {s[0] for s in input_shape if s is not None} - {None}
        if len(batch_sizes) > 1:
            raise ValueError(
                "Can not merge tensors with different "
                "batch sizes. Got tensors with shapes : " + str(input_shape)
            )
        if input_shape[0] is None:
            output_shape = None
        else:
            output_shape = input_shape[0][1:]
        for i in range(1, len(input_shape)):
            if input_shape[i] is None:
                shape = None
            else:
                shape = input_shape[i][1:]
            output_shape = self._compute_elemwise_op_output_shape(output_shape, shape)
        # If the inputs have different ranks, we have to reshape them
        # to make them broadcastable.
        if None not in input_shape and len(set(map(len, input_shape))) == 1:
            self._reshape_required = False
        else:
            self._reshape_required = True

        # Create weights
        self.w = self.add_weight(
            name="w",
            shape=[len(input_shape), input_shape[0][-1]],
            initializer="ones",
            constraint=tf.keras.constraints.NonNeg(),
        )

    def _merge_function(self, inputs):
        output = inputs[0] * self.w[0]
        for i in range(1, len(inputs)):
            output += inputs[i] * self.w[i]
        return output / tf.reduce_sum(self.w, axis=0)

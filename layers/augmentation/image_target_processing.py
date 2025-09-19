from typing import List
from typing import Tuple, Union, Optional, Dict, Type, Any

import keras
import numpy as np
import tensorflow as tf
from keras import backend
from keras.src.engine import base_layer
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from keras.src.utils import image_utils
from tensorflow.keras.layers import Layer
from tensorflow.tools.docs import doc_controls

H_AXIS = -3
W_AXIS = -2

HORIZONTAL = "horizontal"
VERTICAL = "vertical"
HORIZONTAL_AND_VERTICAL = "horizontal_and_vertical"


def check_fill_mode_and_interpolation(fill_mode, interpolation):
    """Checks that the supplied fill mode and interpolation are correct."""
    if fill_mode not in {"reflect", "wrap", "constant", "nearest"}:
        raise NotImplementedError(
            "Unknown `fill_mode` {}. Only `reflect`, `wrap`, "
            "`constant` and `nearest` are supported.".format(fill_mode)
        )
    if interpolation not in {"nearest", "bilinear"}:
        raise NotImplementedError(
            "Unknown `interpolation` {}. Only `nearest` and "
            "`bilinear` are supported.".format(interpolation)
        )

def transform(images,
              transforms,
              fill_mode="reflect",
              fill_value=0.0,
              interpolation="bilinear",
              output_shape=None,
              name=None):
    """Applies the given transform(s) to the image(s).

    Args:
      images: A tensor of shape
        `(num_images, num_rows, num_columns, num_channels)` (NHWC). The rank
        must be statically known (the shape is not `TensorShape(None)`).
      transforms: Projective transform matrix/matrices. A vector of length 8 or
        tensor of size N x 8. If one row of transforms is [a0, a1, a2, b0, b1,
        b2, c0, c1], then it maps the *output* point `(x, y)` to a transformed
        *input* point
        `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
        where `k = c0 x + c1 y + 1`. The transforms are *inverted* compared
        to the transform mapping input points to output points. Note that
        gradients are not backpropagated into transformation parameters.
      fill_mode: Points outside the boundaries of the input are filled according
        to the given mode (one of `{"constant", "reflect", "wrap", "nearest"}`).
      fill_value: a float represents the value to be filled outside the
        boundaries when `fill_mode="constant"`.
      interpolation: Interpolation mode. Supported values: `"nearest"`,
        `"bilinear"`.
      output_shape: Output dimension after the transform, `[height, width]`.
        If `None`, output is the same size as input image.
      name: The name of the op.

    Fill mode behavior for each valid value is as follows:

    - reflect (d c b a | a b c d | d c b a)
    The input is extended by reflecting about the edge of the last pixel.

    - constant (k k k k | a b c d | k k k k)
    The input is extended by filling all
    values beyond the edge with the same constant value k = 0.

    - wrap (a b c d | a b c d | a b c d)
    The input is extended by wrapping around to the opposite edge.

    - nearest (a a a a | a b c d | d d d d)
    The input is extended by the nearest pixel.

    Input shape:
      4D tensor with shape: `(samples, height, width, channels)`,
        in `"channels_last"` format.

    Output shape:
      4D tensor with shape: `(samples, height, width, channels)`,
        in `"channels_last"` format.

    Returns:
      Image(s) with the same type and shape as `images`, with the given
      transform(s) applied. Transformed coordinates outside of the input image
      will be filled with zeros.

    Raises:
      TypeError: If `image` is an invalid type.
      ValueError: If output shape is not 1-D int32 Tensor.
    """
    with backend.name_scope(name or "transform"):
        if output_shape is None:
            output_shape = tf.shape(images)[1:3]
            if not tf.executing_eagerly():
                output_shape_value = tf.get_static_value(output_shape)
                if output_shape_value is not None:
                    output_shape = output_shape_value

        output_shape = tf.convert_to_tensor(
            output_shape, tf.int32, name="output_shape"
        )

        if not output_shape.get_shape().is_compatible_with([2]):
            raise ValueError(
                "output_shape must be a 1-D Tensor of 2 elements: "
                "new_height, new_width, instead got "
                "{}".format(output_shape)
            )

        fill_value = tf.convert_to_tensor(
            fill_value, tf.float32, name="fill_value"
        )

        return tf.raw_ops.ImageProjectiveTransformV3(
            images=images,
            output_shape=output_shape,
            fill_value=fill_value,
            transforms=transforms,
            fill_mode=fill_mode.upper(),
            interpolation=interpolation.upper(),
        )

def get_rotation_matrix(angles, image_height, image_width, name=None):
    """Returns projective transform(s) for the given angle(s).

    Args:
      angles: A scalar angle to rotate all images by, or (for batches of images)
        a vector with an angle to rotate each image in the batch. The rank must
        be statically known (the shape is not `TensorShape(None)`).
      image_height: Height of the image(s) to be transformed.
      image_width: Width of the image(s) to be transformed.
      name: The name of the op.

    Returns:
      A tensor of shape (num_images, 8). Projective transforms which can be
        given to operation `image_projective_transform_v2`. If one row of
        transforms is [a0, a1, a2, b0, b1, b2, c0, c1], then it maps the
        *output* point `(x, y)` to a transformed *input* point
        `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
        where `k = c0 x + c1 y + 1`.
    """
    with backend.name_scope(name or "rotation_matrix"):
        x_offset = (
                           (image_width - 1)
                           - (
                                   tf.cos(angles) * (image_width - 1)
                                   - tf.sin(angles) * (image_height - 1)
                           )
                   ) / 2.0
        y_offset = (
                           (image_height - 1)
                           - (
                                   tf.sin(angles) * (image_width - 1)
                                   + tf.cos(angles) * (image_height - 1)
                           )
                   ) / 2.0
        num_angles = tf.shape(angles)[0]
        return tf.concat(
            values=[
                tf.cos(angles)[:, None],
                -tf.sin(angles)[:, None],
                x_offset[:, None],
                tf.sin(angles)[:, None],
                tf.cos(angles)[:, None],
                y_offset[:, None],
                tf.zeros((num_angles, 2), tf.float32),
            ],
            axis=1,
        )

def check_input(inputs: Union[int, float, list, tuple], name: str = None,
                input_range: Union[list, tuple] = (0, 200)):
    """Checks if an input is of a certain type and within a specified range."""

    # confirm that inputs and name are not None
    if all(value is not None for value in (inputs, name)):
        # ensure that input_range is a list or tuple
        if isinstance(input_range, (list, tuple)) and len(input_range) == 2:
            min_value = input_range[0]
            max_value = input_range[1]

        else:
            raise ValueError(
                f"`input_range` must be a list or tuple containing "
                f"two values, received {input_range} of type {type(input_range)}")

        if not isinstance(name, str):
            raise TypeError(f"`name` must be a string, received {name} of type {type(name)}.")

        if isinstance(inputs, (int, float)):
            if inputs > max_value or inputs < min_value:
                raise ValueError(
                    f"{name} must be a value between [{min_value}, {max_value}], "
                    f"received {inputs}.")

        elif isinstance(inputs, (list, tuple)):
            inputs = sorted(inputs)
            if inputs[1] > max_value or inputs[0] < min_value:
                raise ValueError(
                    f"The input `{name}` must be a value between [{min_value}, {max_value}], "
                    f"received {inputs}.")
        else:
            raise TypeError("inputs must be a int, float, list, or tuple, receive {}".format(type(inputs)))
    else:
        raise ValueError(
            "`inputs` must either be a [int, float, list or tuple], and "
            "`name` must be a string.")

def center_transformation_matrix(matrices, image_height, image_width):
    """
    Centers a batch of transformation matrices.

    Args:
        matrices: [batch_size, 3, 3] tensor of transform matrices.
        image_height: float or scalar tensor.
        image_width: float or scalar tensor.

    Returns:
        [batch_size, 8] tensor of transforms for ImageProjectiveTransformV3
    """
    height_offset = image_height / 2.0 - 0.5
    width_offset = image_width / 2.0 - 0.5

    offset = tf.convert_to_tensor([[1.0, 0.0, width_offset],
                                   [0.0, 1.0, height_offset],
                                   [0.0, 0.0, 1.0]], dtype=tf.float32)

    reset = tf.convert_to_tensor([[1.0, 0.0, -width_offset],
                                  [0.0, 1.0, -height_offset],
                                  [0.0, 0.0, 1.0]], dtype=tf.float32)

    offset = tf.expand_dims(offset, 0)  # [1, 3, 3]
    reset = tf.expand_dims(reset, 0)    # [1, 3, 3]

    # Apply: M_centered = offset @ matrix @ reset (batched matmul)
    centered = tf.matmul(offset, tf.matmul(matrices, reset))  # [batch, 3, 3]
    flat = tf.reshape(centered[:, :2, :], [-1, 6])  # [batch_size, 6]

    # Pad to 8 by adding [0, 0]
    zeros = tf.zeros((tf.shape(flat)[0], 2), dtype=tf.float32)
    return tf.concat([flat, zeros], axis=1)  # [batch_size, 8]

def get_shear_matrix(angles, image_height, image_width, name=None):
    """Returns a batch of shear matrices for the given angles (in radians)."""
    with tf.name_scope(name or "shear_matrix"):
        angles = tf.convert_to_tensor(angles, dtype=tf.float32)
        batch_size = tf.shape(angles)[0]

        ones = tf.ones_like(angles)
        zeros = tf.zeros_like(angles)
        shear = -tf.sin(angles)
        cos = tf.cos(angles)

        # [batch_size, 3, 3]
        matrices = tf.stack([
            tf.stack([ones, shear, zeros], axis=1),
            tf.stack([zeros, cos, zeros], axis=1),
            tf.stack([zeros, zeros, ones], axis=1)
        ], axis=1)

        return center_transformation_matrix(matrices, image_height, image_width)  # [batch_size, 8]

def interpolate_to_tensorflow_value(custom_value: Union[int, float, list, tuple],
                                    custom_datapoints: Union[list, tuple, None],
                                    tensorflow_datapoints: Union[list, tuple, None]):
    """
    Produces a corresponding TensorFlow value when given a custom value. This is done by
    interpolating between the custom values and their corresponding TensorFlow values.

    :param custom_value: The custom value to be converted Tensorflow equivalent.
    :param custom_datapoints: tuple or list, The   custom data points (x-coordinates),
        must be increasing if argument `period` is not specified. Otherwise,
        `custom_datapoints` is internally sorted after normalizing the periodic boundaries
        with ``custom_datapoints = custom_datapoints % period``.
    :param tensorflow_datapoints: The TF datapoints (y-coordinates) of the data points, same
        length as `custom_datapoints`.
    :return: The interpolated values, same shape as custom_datapoints.
    :raises: ValueError – If custom_datapoints and tensorflow_datapoints have different length If
        custom_datapoints and tensorflow_datapoints not 1-D sequences If period == 0
    """
    if custom_datapoints is not None and tensorflow_datapoints is not None:
        if not all(isinstance(dps, (list, tuple)) for dps in (custom_datapoints, tensorflow_datapoints)):
            raise TypeError(
                f"custom_datapoints and tensorflow_datapoints must be a list or tuple.\n"
                f"Current datatype: \ncustom_datapoints is a {type(custom_datapoints).__name__}, "
                f"while tensorflow_datapoints is a {type(tensorflow_datapoints).__name__}")

        if len(custom_datapoints) != len(tensorflow_datapoints):
            raise ValueError(
                f"custom_datapoints and tensorflow_datapoints must be the same length. "
                f"custom_datapoints and tensorflow_datapoints have {len(custom_datapoints)} "
                f"and {len(tensorflow_datapoints)} values, respectively")

        if not isinstance(custom_value, (int, float)):
            return tuple(np.interp(x=custom_value, xp=custom_datapoints, fp=tensorflow_datapoints))
        else:
            return np.interp(x=custom_value, xp=custom_datapoints, fp=tensorflow_datapoints)

    else:
        return custom_value

def get_zoom_matrix(zooms, image_height, image_width, name=None):
    """Returns projective transform(s) for the given zoom(s).

    Args:
      zooms: A matrix of 2-element lists representing `[zx, zy]` to zoom for
        each image (for a batch of images).
      image_height: Height of the image(s) to be transformed.
      image_width: Width of the image(s) to be transformed.
      name: The name of the op.

    Returns:
      A tensor of shape `(num_images, 8)`. Projective transforms which can be
        given to operation `image_projective_transform_v2`.
        If one row of transforms is
         `[a0, a1, a2, b0, b1, b2, c0, c1]`, then it maps the *output* point
         `(x, y)` to a transformed *input* point
         `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
         where `k = c0 x + c1 y + 1`.
    """
    with backend.name_scope(name or "zoom_matrix"):
        num_zooms = tf.shape(zooms)[0]
        # The zoom matrix looks like:
        #     [[zx 0 0]
        #      [0 zy 0]
        #      [0 0 1]]
        # where the last entry is implicit.
        # Zoom matrices are always float32.
        x_offset = ((image_width - 1.0) / 2.0) * (1.0 - zooms[:, 0:1])
        y_offset = ((image_height - 1.0) / 2.0) * (1.0 - zooms[:, 1:2])

        return tf.concat(
            values=[
                zooms[:, 0:1],
                tf.zeros((num_zooms, 1), tf.float32),
                x_offset,
                tf.zeros((num_zooms, 1), tf.float32),
                zooms[:, 1:2],
                y_offset,
                tf.zeros((num_zooms, 2), tf.float32),
            ],
            axis=1,
    )


@keras.saving.register_keras_serializable(package='Augmentation', name='AugmentationLayer')
class AugmentationLayer(base_layer.BaseRandomLayer):

    def __init__(self,
                 active: bool = True,
                 custom_interp_datapoints: Union[List, Tuple, None] = None,
                 tensorflow_interp_datapoints: Union[List, Tuple, None] = None,
                 image_key: str = 'images',
                 target_key: str = 'targets',
                 bounding_box_key: str = "bounding_boxes",
                 batches_before_aug: Optional[int] = None,
                 name: str = 'dual_augmentation_layer',
                 **kwargs):
        """
        Initializes the base augmentation layer for image, mask, and bounding box transformations.

        This layer serves as a foundation for implementing custom augmentation behaviors that
        may operate on different input types (images, targets, bounding boxes). It also supports
        optional interpolation between custom and TensorFlow-defined datapoints, useful for
        mapping custom augmentation scales to standardized outputs.

        :param active: If False, the augmentation logic is disabled (passthrough mode).
                       Stored as a `tf.Variable` to allow dynamic switching at runtime.
        :param custom_interp_datapoints: Optional list or tuple of reference x-values
                       for interpolation from custom-defined space.
        :param tensorflow_interp_datapoints: Optional list or tuple of reference y-values
                       for interpolation into the TensorFlow space.
        :param image_key: Dictionary key under which the image tensor is expected in the input.
        :param target_key: Dictionary key under which the target (e.g., mask) tensor is expected.
        :param bounding_box_key: Dictionary key under which the bounding box tensor is expected.
        :param batches_before_aug: int, optional.
            Number of batches to pass through the layer without augmentation at the start of each epoch.
            Augmentation is disabled for these batches to ensure the model sees the unaltered original data first.
            After this threshold, augmentation is applied for the remaining batches in the epoch.

        :param name: Name of the layer.
        :param kwargs: Additional keyword arguments passed to `BaseRandomLayer`.
        """
        super().__init__(trainable=False, name=name, **kwargs)

        self.check_parameter_type(params_and_types={
            "active": (active, bool, False),
            "custom_interp_datapoints": (custom_interp_datapoints, (list, tuple), True),
            "tensorflow_interp_datapoints": (tensorflow_interp_datapoints, (list, tuple), True),
            "image_key": (image_key, str, False),
            "target_key": (target_key, str, False),
            "bounding_box_key": (bounding_box_key, str, False),
            "batches_before_aug": (batches_before_aug, int, True),
            "name": (name, str, False),
        })

        self.active = tf.Variable(tf.constant(active, dtype=tf.bool))
        self.tensorflow_interp_datapoints = tensorflow_interp_datapoints
        self.custom_interp_datapoints = custom_interp_datapoints
        self.image_key = image_key
        self.target_key = target_key
        self.bounding_box_key = bounding_box_key
        self.target_compute_dtype = tf.int32

        if batches_before_aug is None:
            self.batches_before_aug = tf.constant(value=0, dtype=tf.int32)
        else:
            self.batches_before_aug = tf.constant(batches_before_aug, dtype=tf.int32, name='batches_before_aug')

        self.batch_counter = tf.Variable(0, trainable=False, dtype=tf.int32, name='batch_counter')

    def __repr__(self):
        return f"{self.name}"

    @doc_controls.for_subclass_implementers
    def augment_image(self, image, transformation):
        """Augment a batch of image(s) during training.

        Args:
          image: 4D image input tensor to the layer. Forwarded from
            `layer.call()`.
          transformation: The transformation object produced by
            `get_random_transformation`. Used to coordinate the randomness
            between image, label and bounding box.

        Returns:
          output 4D tensor, which will be forward to `layer.call()`.
        """
        raise NotImplementedError()

    @doc_controls.for_subclass_implementers
    def augment_label(self, label, transformation):
        """Augment a batch of label(s) during training.

        Args:
          label: 4D [B,H,W,C] label to the layer. Forwarded from `layer.call()`.
          transformation: The transformation object produced by
            `get_random_transformation`. Used to coordinate the randomness
            between image, label and bounding box.

        Returns:
          output 4D tensor, which will be forward to `layer.call()`.
        """
        raise NotImplementedError()

    def augment_target(self, target, transformation):
        """Augment a batch of target(s) during training.

        Args:
          target: 4D label to the layer. Forwarded from `layer.call()`.
          transformation: The transformation object produced by
            `get_random_transformation`. Used to coordinate the randomness
            between image, label and bounding box.

        Returns:
          output 4D tensor, which will be forward to `layer.call()`.
        """
        return self.augment_label(target, transformation)

    @doc_controls.for_subclass_implementers
    def augment_bounding_boxes(self,
                               images,
                               bounding_boxes,
                               transformation: Optional[Dict[str, Any]] = None):
        """Augment bounding boxes for one image during training.

        Args:
          images: 4D image input tensor to the layer. Forwarded from
            `layer.call()`.
          bounding_boxes: 3D Tensor [B, N, 4] of bounding boxes to the layer. Forwarded from
            `call()`.
          transformation: The transformation object produced by
            `get_random_transformation`. Used to coordinate the randomness
            between image, label and bounding box.

        Returns:
          output 2D tensor, which will be forward to `layer.call()`.
        """
        raise NotImplementedError()

    @doc_controls.for_subclass_implementers
    def get_random_transformation(self, image=None, label=None, bounding_box=None):
        """Produce random transformation config for batched input.

        This is used to produce same randomness between
        image/label/bounding_box.

        Args:
          image: 4D image tensor from inputs.
          label: optional 1D/3D/4D label tensor from inputs.
          bounding_box: optional 2D bounding boxes tensor from inputs.

        Returns:
          Any type of object, which will be forwarded to `augment_image`,
          `augment_label` and `augment_bounding_box` as the `transformation`
          parameter.
        """
        return None

    def check_inputs_type(self, inputs):
        """Ensures the input is a Python dictionary"""
        if not isinstance(inputs, dict):
            raise TypeError(f"The inputs must be a dict, with keys ['{self.image_key}', "
                            f"'{self.target_key}', {self.bounding_box_key}]. "
                            f"Got {inputs} of type {type(inputs).__name__}")

    @staticmethod
    def check_parameter_type(params_and_types: Dict[str, Tuple[Any, Union[Type, Tuple[Type, ...]], bool]]):
        """
        Validates that each parameter value matches its expected type, and handles optional parameters.

        :param params_and_types: A dictionary where each key is a parameter name, and the value is a tuple:
            (value, expected_type(s), is_optional), where:
                - value: The actual value to validate.
                - expected_type(s): A type or tuple of types that the value is expected to match.
                - is_optional: If True, allows the value to be None.

            Example:
                check_type({
                    "name": ("Alice", str, False),
                    "age": (None, int, True),
                    "scores": ([90, 95], list, False),
                })

        :raises TypeError:
            - If a value is None and is_optional is False.
            - If a value does not match the expected type(s).
        """
        for param_name, (value, expected_types, is_optional) in params_and_types.items():
            if value is None:
                if is_optional:
                    continue
                else:
                    raise TypeError(f"{param_name} must not be None.")

            if not isinstance(value, expected_types):
                allowed_types = (
                    expected_types.__name__
                    if isinstance(expected_types, type)
                    else ', '.join(t.__name__ for t in expected_types)
                )
                raise TypeError(f"{param_name} must be of type {allowed_types}. Got {type(value).__name__}.")

    def check_dict_keys(self, inputs):
        """Checks that the inputs have the correct dict keys"""
        if not all([key in inputs for key in [self.image_key, self.target_key]]):
            raise KeyError(f"Invalid dictionary keys for training examples. "
                           f"Got keys: {list(inputs.keys())}"
                           f"The Images and segmentation mask must be assigned to the "
                           f"keys '{self.image_key}' and '{self.target_key}', respectively. ")

    def _ensure_inputs_are_compute_dtype(self, inputs):
        """Ensures the input is a Tensor, SparseTensor or RaggedTensor."""
        self.check_inputs_type(inputs)
        self.check_dict_keys(inputs)
        inputs[self.image_key] = utils.ensure_tensor(inputs=inputs[self.image_key],
                                                     dtype=self.compute_dtype)

        inputs[self.target_key] = utils.ensure_tensor(inputs=inputs[self.target_key],
                                                      dtype=self.target_compute_dtype)
        return inputs

    @staticmethod
    def interpolate_to_tensorflow_value(custom_value: Union[int, float, list, tuple],
                                        custom_datapoints: Union[list, tuple, None],
                                        tensorflow_datapoints: Union[list, tuple, None]):
        """
        Produces TensorFlow equivalent of a given custom value. This is done by
        interpolating between custom values and their corresponding TensorFlow values.

        :param custom_value: The custom value to be converted Tensorflow equivalent.
        :param custom_datapoints: tuple or list, The   custom data points (x-coordinates).
            These values must be presented in ascending order.
        :param tensorflow_datapoints: The TF equivalent (y-coordinates) of the custom datapoints, same
            length as `custom_datapoints`.
        :return: The interpolated value (The Tensorflow equivalent)
        :raises: ValueError – If custom_datapoints and tensorflow_datapoints have different length If
            custom_datapoints and tensorflow_datapoints not 1-D sequences If period == 0
        """
        if custom_datapoints is not None and tensorflow_datapoints is not None:
            if not all(isinstance(dps, (list, tuple)) for dps in (custom_datapoints, tensorflow_datapoints)):
                raise TypeError(
                    f"custom_datapoints and tensorflow_datapoints must be a list or tuple.\n"
                    f"Current datatype: \ncustom_datapoints is a {type(custom_datapoints).__name__}, "
                    f"while tensorflow_datapoints is a {type(tensorflow_datapoints).__name__}")

            if len(custom_datapoints) != len(tensorflow_datapoints):
                raise ValueError(
                    f"custom_datapoints and tensorflow_datapoints must be the same length. "
                    f"custom_datapoints and tensorflow_datapoints have {len(custom_datapoints)} "
                    f"and {len(tensorflow_datapoints)} values, respectively")

            if not isinstance(custom_value, (int, float)):
                return tuple(np.interp(x=custom_value, xp=custom_datapoints, fp=tensorflow_datapoints))
            else:
                return np.interp(x=custom_value, xp=custom_datapoints, fp=tensorflow_datapoints)

        else:
            return custom_value

    @staticmethod
    def is_unbatched(tensor: tf.Tensor) -> bool:
        """
        Determines if the input tensor is unbatched (i.e., 3D).

        Args:
            tensor: A TensorFlow tensor, typically with shape [B, H, W, C] or [H, W, C].

        Returns:
            True if the tensor is unbatched (rank == 3), False otherwise.
        """
        return tensor.shape.rank == 3

    @staticmethod
    def is_batched(tensor: tf.Tensor) -> bool:
        """
        Determines if the input tensor is batched (i.e., 4D).

        Args:
            tensor: A TensorFlow tensor, typically with shape [B, H, W, C] or [H, W, C].

        Returns:
            True if the tensor is batched (rank == 4), False otherwise.
        """
        return tensor.shape.rank == 4

    def ensure_batched(self, tensor: tf.Tensor):
        """
        Ensures that the input tensor is batched (rank 4).

        If it's unbatched (rank 3), this method expands its dimensions.
        Also returns a boolean indicating whether the tensor was originally unbatched.

        Args:
            tensor: A TensorFlow tensor with rank 3 or 4.

        Returns:
            A tuple:
                - The tensor in batched form (rank 4)
                - A boolean flag `was_unbatched` indicating if batching was applied
        """
        was_unbatched = self.is_unbatched(tensor)
        if was_unbatched:
            tensor = tf.expand_dims(input=tensor, axis=0)
        return tensor, was_unbatched

    def reset_counter(self):
        self.batch_counter.assign(0)

    def call(self, inputs, training=None):
        inputs = self._ensure_inputs_are_compute_dtype(inputs)

        if training:
            # Increment the batch counter and decide whether to apply augmentation
            # based on whether the model has already seen enough original (non-augmented) batches.
            current_batch = self.batch_counter.assign_add(1)

            def augment():
                return tf.cond(
                    pred=tf.convert_to_tensor(self.active),
                    true_fn=lambda: self._augment(inputs),
                    false_fn=lambda: inputs)

            def no_augment():
                return inputs

            return tf.cond(
                pred=tf.greater(current_batch, self.batches_before_aug),
                true_fn=augment,
                false_fn=no_augment
            )
        else:
            return inputs

    def _augment(self, inputs):
        """Augments the image, target and bounding box."""
        image = inputs.get(self.image_key, None)
        target = inputs.get(self.target_key, None)
        bounding_box = inputs.get(self.bounding_box_key, None)

        original_image_shape = image.shape
        # The transform op only accepts rank 4 inputs,
        # so if we have an unbatched image,
        # we need to temporarily expand dims to a batch.
        image, image_was_unbatched = self.ensure_batched(image)

        transformation = self.get_random_transformation(image=image, label=target,
                                                        bounding_box=bounding_box)

        image = self.augment_image(image, transformation=transformation)

        if image_was_unbatched:
            image = tf.squeeze(image, 0)
            image.set_shape(original_image_shape)

        result = {self.image_key: image}

        # apply target augmentation only if target exist, and it is at least 3-dimensional
        if target is not None:
            original_target_shape = target.shape
            target, target_was_unbatched = self.ensure_batched(target)

            if self.is_batched(target):
                target = self.augment_target(target, transformation=transformation)
                if target_was_unbatched:
                    target = tf.squeeze(input=target, axis=0)
                    target.set_shape(original_target_shape)

                target = tf.round(target)
                result[self.target_key] = tf.cast(target, dtype=self.target_compute_dtype)
            elif target.shape.rank > 2 and target.shape[-1] > 1:
                raise ValueError(f"The shape of the target(s) must be in the form "
                                 f"[Height, Width, 1], got {target.shape}.")
            else:
                # incase the target is not a 3-dimensional tensor(e.g. a categorical target)
                result[self.target_key] = target

        if bounding_box is not None:
            bounding_box = self.augment_bounding_boxes(image, bounding_box,
                                                       transformation=transformation)
            result[self.bounding_box_key] = bounding_box

        return result

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'active': self.active,
                'custom_interp_datapoints': self.custom_interp_datapoints,
                'tensorflow_interp_datapoints': self.tensorflow_interp_datapoints,
                'image_key': self.image_key,
                'target_key': self.target_key,
                'bounding_box_key': self.bounding_box_key,
                'batches_before_aug': self.batches_before_aug
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package='Augmentation', name='SampleWeight')
class SampleWeight(AugmentationLayer):
    """
    A preprocessing layer that computes a per-pixel sample weight map based on the inverse class frequency
    of the segmentation mask.

    This is typically used to address class imbalance in semantic segmentation tasks by assigning higher
    weights to pixels belonging to underrepresented classes.

    Example:
        >>> layer = SampleWeight(num_classes=5)
        >>> inputs = {
        >>>     'images': tf.ones([1, 128, 128, 3]),
        >>>     'targets': tf.random.uniform([1, 128, 128, 1], minval=0, maxval=5, dtype=tf.int32)
        >>> }
        >>> outputs = layer(inputs)

    Args:
        num_classes (int): Total number of distinct classes in the segmentation task.
        active (bool): Whether the layer should be active during training/inference.
        image_key (str): Dictionary key to access the input image tensor.
        target_key (str): Dictionary key to access the segmentation mask tensor.
        bounding_box_key (str): Dictionary key to access optional bounding box data.
        sample_weight_key (str): Key under which to store the computed sample weight map.
        name (str): Name of the layer.
    """
    def __init__(self,
                 num_classes: int,
                 active: bool = False,
                 image_key: str = 'images',
                 target_key: str = 'targets',
                 bounding_box_key: str = "bounding_boxes",
                 name: str = 'sample_weight',
                 ):
        super().__init__(active=active,
                         name=name,
                         image_key=image_key,
                         target_key=target_key,
                         bounding_box_key=bounding_box_key)

        self.sample_weight_key = 'sample_weight'
        self.num_classes = num_classes

    def __repr__(self):
        return f"{self.sample_weight_key}"

    def _augment(self, inputs: Dict):
        """
        Computes the sample weight map and injects it into the input dictionary under `self.sample_weight_key`.

        If the target mask is batched, weights are computed per-sample using `tf.map_fn`.
        If the target is a categorical vector or label, no weight map is generated.

        Args:
            inputs (dict): A dictionary containing keys for images and targets.

        Returns:
            dict: The same dictionary with an additional key for the computed sample weight map.
        """

        target = inputs.get(self.target_key, None)

        if target is not None and isinstance(target, tf.Tensor):
            original_target_shape = target.shape
            temp_target, target_was_unbatched = self.ensure_batched(target)

            if self.is_batched(temp_target):
                sample_weight = tf.map_fn(fn=self.compute_single_sample_weight_map,
                                          elems=temp_target,
                                          fn_output_signature=tf.TensorSpec(shape=temp_target.shape[1:],
                                                                            dtype=tf.float32))
                if target_was_unbatched:
                    sample_weight = tf.squeeze(sample_weight, axis=0)
                    sample_weight.set_shape(original_target_shape)

                inputs[self.sample_weight_key] = sample_weight
            elif target.shape.rank > 2 and target.shape[-1] > 1:
                raise ValueError(f"The shape of the target(s) must be in the form "
                                 f"[Height, Width, 1], got {target.shape}.")
            else:
                # Skip sample weight computation if it's a 1D/2D categorical target
                inputs[self.sample_weight_key] = None

        return inputs

    def compute_single_sample_weight_map(self, mask):
        """
        Computes a per-pixel weight map for a single mask using inverse class frequency.

        Args:
            mask (tf.Tensor): A 3D tensor [H, W, 1] or [H, W] containing class indices.

        Returns:
            tf.Tensor: A 3D tensor [H, W, 1] where each pixel's value represents its class weight.
        """

        # Flatten mask to 1D
        flat_mask = tf.cast(x=tf.reshape(tensor=mask, shape=[-1]), dtype=tf.int32)

        # Count frequency of each class
        class_counts = tf.math.bincount(arr=flat_mask,
                                        minlength=self.num_classes,
                                        maxlength=self.num_classes,
                                        dtype=tf.float32)

        # Mask out zero entries to avoid division by zero
        present_mask = class_counts > 0
        inv_freq = tf.where(condition=present_mask, x=1.0/class_counts, y=tf.zeros_like(class_counts))

        # Normalize weights so that present class weights sum to 1.
        norm_weights = inv_freq / tf.reduce_sum(inv_freq)

        # Map each pixel label to its class weight
        weight_map = tf.gather(norm_weights, mask)
        return weight_map


@keras.saving.register_keras_serializable(package='Augmentation', name='RandomRotation')
class RandomRotation(AugmentationLayer):
    """
    A preprocessing layer which randomly rotates images during training.

    This layer applies random rotations to each image and its segmentation mask (if available),
    filling empty regions according to the specified `fill_mode` and `fill_value`. The rotation
    angle is sampled uniformly from a specified range. At inference time or when `active=False`,
    the layer behaves as a no-op, passing the input (image and label) through unchanged.

    Input shape:
        {'images': 3D (unbatched) or 4D (batched) tensor,
         'targets': 3D (unbatched) or 4D (batched) tensor}

        Image and target tensors must follow the shape `(..., height, width, channels)`
        using the `"channels_last"` format.

    Output shape:
        {'images': 3D (unbatched) or 4D (batched) tensor,
         'targets': 3D (unbatched) or 4D (batched) tensor}

        Output shape and format are identical to input. NOTE: if the shape of the target is 2D,
        it will not be augmented.

    Arguments:
      angle: A float, int, tuple, or list representing the angle by which the image and mask
         will be rotated. If a single value (int or float) is provided, the rotation range is
         between [-angle, angle]. If a tuple or list of two values is provided, the angle is
         sampled between (angle[0], angle[1]).
      active: Boolean indicating whether to apply augmentation during training. If False,
         inputs are returned unchanged.
      fill_mode: Points outside the boundaries of the input are filled according to this mode.
         One of: `{"constant", "reflect", "wrap", "nearest"}`.

         - *reflect*: `(d c b a | a b c d | d c b a)` — reflects about the edge of the last pixel.
         - *constant*: `(k k k k | a b c d | k k k k)` — fills with constant value `k` (default 0).
         - *wrap*: `(a b c d | a b c d | a b c d)` — wraps around to the opposite edge.
         - *nearest*: `(a a a a | a b c d | d d d d)` — extends using nearest edge pixel.
      interpolation: Interpolation mode used during resampling. Supported values: `"nearest"`, `"bilinear"`.
      fill_value: A float representing the pixel value used to fill empty regions when `fill_mode="constant"`.
      image_key: The key used to retrieve the image tensor from the input dictionary.
      target_key: The key used to retrieve the mask or label tensor from the input dictionary.
      bounding_box_key: The key used to retrieve bounding box data from the input dictionary.
      name: The name of the layer instance.

    Raises:
      TypeError: If `angle` is not a float, int, list, or tuple.
      ValueError: If `angle` is a list or tuple not of length 2, or if the upper angle is less than the lower.
    """


    def __init__(self,
                 angle: Union[int, float, List[int], Tuple[int, int]] = 180,
                 active: bool = True,
                 fill_mode="reflect",
                 interpolation="bilinear",
                 fill_value=0.0,
                 image_key: str = 'images',
                 target_key: str = 'targets',
                 bounding_box_key: str = "bounding_boxes",
                 batches_before_aug: Optional[int] = None,
                 name:str = 'random_rotation',
                 **kwargs):


        super().__init__(active=active,
                         image_key=image_key,
                         target_key=target_key,
                         bounding_box_key=bounding_box_key,
                         batches_before_aug=batches_before_aug,
                         name=name,
                         **kwargs)
        self.angle = angle
        if isinstance(angle, (tuple, list)):
            if len(angle) != 2:
                raise ValueError(f"The {type(angle).__name__} `angle` must have only two values, got {angle}")
            self.lower = np.deg2rad(angle[0])
            self.upper = np.deg2rad(angle[1])

        elif isinstance(angle, (int, float)):
            self.lower = np.deg2rad(-angle)
            self.upper = np.deg2rad(angle)
        else:
            raise TypeError(
                f"The angle must be a list, tuple, int or float. Got {angle} of type `{type(angle).__name__}`"
            )

        if self.upper < self.lower:
            raise ValueError(
                "The upper boundary cannot be less than the lower boundary. Got angle = {}".format(angle)
            )

        check_fill_mode_and_interpolation(fill_mode, interpolation)
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.interpolation = interpolation

    def get_random_transformation(self, image=None, label=None, bounding_box=None):
        # min_angle = self.lower * 2.0 * np.pi
        # max_angle = self.upper * 2.0 * np.pi
        batch_size = tf.shape(image)[0]
        angles = tf.random.uniform(shape=[batch_size], minval=self.lower, maxval=self.upper)

        # # 50% chance to apply rotation: mask = 1 if apply, 0 if skip
        # apply_mask = tf.cast(tf.random.uniform(shape=[batch_size], minval=0, maxval=1) < 0.5,
        #                      dtype=angles.dtype)
        #
        # # Zero out angles for samples where augmentation should be skipped
        # angles = angles * apply_mask

        return {"angles": angles}

    def augment_image(self, image, transformation):
        image = utils.ensure_tensor(image, self.compute_dtype)
        image_shape = tf.shape(image)

        img_hd = tf.cast(image_shape[H_AXIS], tf.float32)
        img_wd = tf.cast(image_shape[W_AXIS], tf.float32)

        angle = transformation["angles"]

        output = transform(images=image,
                           transforms=get_rotation_matrix(angle, img_hd, img_wd),
                           fill_mode=self.fill_mode,
                           fill_value=self.fill_value,
                           interpolation=self.interpolation,
                           )
        return output

    def augment_label(self, label, transformation):
        label = utils.ensure_tensor(label, self.compute_dtype)
        image_shape = tf.shape(label)

        label_hd = tf.cast(image_shape[H_AXIS], tf.float32)
        label_wd = tf.cast(image_shape[W_AXIS], tf.float32)

        angles = transformation["angles"]
        output = transform(label,
                           get_rotation_matrix(angles, label_hd, label_wd),
                           fill_mode=self.fill_mode,
                           fill_value=self.fill_value,
                           interpolation='nearest',
                           )
        return output

    def augment_bounding_boxes(self, images, bounding_boxes, transformation=None):
        """
        Applies rotation to bounding boxes for a batch of images.

        :param images: 4D tensor of shape (B, H, W, C)
        :param bounding_boxes: RaggedTensor of shape (B, None, 4), each row is [x_min, y_min, x_max, y_max]
        :param transformation: Dictionary containing 'angles': 1D tensor of shape (B,)
        :return: RaggedTensor of shape (B, None, 4) with updated bounding boxes
        """
        angles = -transformation["angles"]  # Clockwise for TF transform
        batch_size = tf.shape(images)[0]
        image_shapes = tf.shape(images)

        def rotate_boxes_per_image(args):
            boxes, angle, img = args
            h = tf.cast(tf.shape(img)[H_AXIS], tf.float32)
            w = tf.cast(tf.shape(img)[W_AXIS], tf.float32)
            cx, cy = w / 2.0, h / 2.0

            x1, y1, x2, y2 = tf.split(boxes, 4, axis=1)
            corners = tf.concat([
                tf.stack([x1, y1], axis=2),
                tf.stack([x2, y1], axis=2),
                tf.stack([x2, y2], axis=2),
                tf.stack([x1, y2], axis=2)
            ], axis=1)  # shape (N, 4, 2)

            translated = corners - tf.constant([[cx, cy]], dtype=tf.float32)

            cos_theta = tf.cos(angle)
            sin_theta = tf.sin(angle)
            rot_matrix = tf.convert_to_tensor([[cos_theta, -sin_theta], [sin_theta, cos_theta]], dtype=tf.float32)

            rotated = tf.einsum('nij,jk->nik', translated, rot_matrix) + tf.constant([[cx, cy]], dtype=tf.float32)

            min_coords = tf.reduce_min(rotated, axis=1)
            max_coords = tf.reduce_max(rotated, axis=1)
            return tf.concat([min_coords, max_coords], axis=1)

        batched_outputs = tf.map_fn(
            fn=rotate_boxes_per_image,
            elems=(bounding_boxes, angles, images),
            fn_output_signature=tf.TensorSpec(shape=(None, 4), dtype=tf.float32)
        )

        return batched_outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"angle": self.angle,
                       "fill_mode": self.fill_mode,
                       "interpolation": self.interpolation,
                       "fill_value": self.fill_value})

        return config


@keras.saving.register_keras_serializable(package='Augmentation', name='RandomGaussianNoise')
class RandomGaussianNoise(AugmentationLayer):

    """
    A preprocessing layer which randomly applies Gaussian noise to images during training.

    This layer adds pixel-wise Gaussian noise to the input image tensor. The standard deviation
    is sampled uniformly from a specified range. Noise is applied only during training when
    `active=True`. During inference or when inactive, the layer behaves as a no-op: all inputs,
    including images, targets, and bounding boxes, are returned unmodified.

    Input shape:
        {'images': 3D (unbatched) or 4D (batched) tensor,
         'targets': 1D, 2D, 3D, or 4D tensor}

        Image tensors must follow the shape `(..., height, width, channels)` using
        the `"channels_last"` format. Targets can be any of the following:
        - 1D: single class label
        - 2D: single-channel binary mask
        - 3D: multi-channel mask or label image
        - 4D: batched masks or labels

        Only the image is augmented; the target is passed through unchanged.

    Output shape:
        {'images': 3D (unbatched) or 4D (batched) tensor,
         'targets': 1D, 2D, 3D, or 4D tensor}

        Output shape and format are identical to input.

    Arguments:
      stddev: A float, list, or tuple representing the lower and upper bounds of the
          standard deviation to use for sampling Gaussian noise. If a single float is provided,
          it is treated as the upper bound with the lower bound defaulting to 0.0. Both values
          must be ≥ 0.
      name: The name assigned to the layer instance.
      active: Boolean indicating whether this layer is active during training. If False,
          the input is returned unchanged.
      image_key: The key used to retrieve the image tensor from the input dictionary.
      target_key: The key used to retrieve the mask or label tensor from the input dictionary.
      bounding_box_key: The key used to retrieve bounding box data from the input dictionary.
      kwargs: Additional keyword arguments passed to the base `AugmentationLayer`.

    Raises:
      TypeError: If `stddev` is not a float, int, list, or tuple.
      ValueError: If `stddev` contains negative values or if the upper bound is less than the lower.
    """

    def __init__(self,
                 stddev: Union[float, List[float], Tuple[float, float]] = (0.0, 3.0),
                 name='random_gaussian_noise',
                 active: bool = True,
                 image_key: str = 'images',
                 target_key: str = 'targets',
                 bounding_box_key: str = "bounding_boxes",
                 batches_before_aug: Optional[int] = None,
                 **kwargs):

        base_preprocessing_layer.keras_kpl_gauge.get_cell("RandomDualGaussianNoise").set(True)

        super().__init__(active=active,
                         image_key=image_key,
                         target_key=target_key,
                         bounding_box_key=bounding_box_key,
                         batches_before_aug=batches_before_aug,
                         name=name,
                         **kwargs)
        self.stddev = stddev

        if isinstance(stddev, (tuple, list)):
            self.lower = stddev[0]
            self.upper = stddev[1]
        elif isinstance(stddev, (int, float)):
            self.lower = 0.0
            self.upper = stddev
        else:
            raise TypeError(
                "Allowed datatypes  for stddev: tuple, list, int, "
                "float. Receive {} of type - {}.".format(stddev, type(stddev).__name__))

        if self.lower < 0 or self.upper < 0:
            raise ValueError("stddev must be greater than 0, got {}".format(stddev))

        if self.upper < self.lower:
            raise ValueError(
                "The upper limit of stddev must be greater than the lower limit. "
                "{} is less than {}".format(self.upper, self.lower))

    def get_random_transformation(self, image=None, label=None, bounding_box=None):
        batch_size = tf.shape(image)[0]
        min_stddev = self.lower
        max_stddev = self.upper

        # Set shape of stddev to [batch_size, 1, 1, 1], so that the noise is added to each image
        # in the batch using a different standard deviation.
        stddev = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=min_stddev, maxval=max_stddev)

        return {"stddev": stddev}

    def augment_image(self, image, transformation):
        image = utils.ensure_tensor(image, self.compute_dtype)
        stddev = transformation["stddev"]

        output = image + tf.random.normal(shape=tf.shape(image),
                                          mean=0.0,
                                          stddev=stddev,
                                          dtype=image.dtype, )

        return output

    def augment_label(self, label, transformation):
        return label

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"stddev": self.stddev ,})
        return config


@keras.saving.register_keras_serializable(package='Augmentation', name='RandomFlip')
class RandomFlip(AugmentationLayer):
    """
    A preprocessing layer that randomly flips images, segmentation masks, and bounding boxes during training.

    This layer supports horizontal, vertical, or both types of flips. Each sample in the batch is independently
    considered for flipping based on a random condition. No flipping occurs during inference or if the layer is inactive.

    Attributes:
        mode (str): One of {'horizontal', 'vertical', 'horizontal_and_vertical'} specifying the flip direction(s).
        active (bool): Whether to activate the augmentation during training.
        image_key (str): Key to access image tensors in the input dictionary. Default value = 'images'
        target_key (str): Key to access target tensors. Default value = 'targets'
        bounding_box_key (str): Key to access bounding box tensors (optional). default value = 'bounding_box'
        batches_before_aug (int, optional): Number of batches to skip before applying augmentation.
        auto_vectorize (bool): Whether the augmentation is vectorized over the batch (enabled by default).

    Input:
        A dictionary containing:
            - 'images': 3D or 4D tensor of shape [H, W, C] or [B, H, W, C].
            - 'targets': 1D to 4D tensor (e.g., segmentation masks or class labels).
            - 'bounding_boxes' (optional): 2D or 3D tensor of shape [B, N, 4] with format [x_min, y_min, x_max, y_max].

    Output:
        A dictionary with the same structure, where 'images', 'targets', and optionally 'bounding_boxes' are randomly flipped.
    """

    def __init__(self,
                 mode='horizontal_and_vertical',
                 active: bool = True,
                 image_key: str = 'images',
                 target_key: str = 'targets',
                 bounding_box_key: str = "bounding_boxes",
                 batches_before_aug: Optional[int] = None,
                 name='random_flip',
                 **kwargs):
        super().__init__(name=name,
                         active=active,
                         image_key=image_key,
                         target_key=target_key,
                         bounding_box_key=bounding_box_key,
                         batches_before_aug=batches_before_aug,
                         **kwargs)

        mode = mode.lower()
        if mode == 'horizontal':
            self.horizontal = True
            self.vertical = False
        elif mode == 'vertical':
            self.horizontal = False
            self.vertical = True
        elif mode == 'horizontal_and_vertical':
            self.horizontal = True
            self.vertical = True
        else:
            raise ValueError(
                f"Invalid flip mode '{mode}' in layer '{name}'. Must be 'horizontal', 'vertical', or 'horizontal_and_vertical'."
            )

        self.mode = mode
        self.auto_vectorize = True

    def get_random_transformation(self, image=None, label=None, bounding_box=None):
        batch_size = tf.shape(image)[0]

        flip_horizontal = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0) < 0.5 if self.horizontal else tf.zeros([batch_size], dtype=tf.bool)
        flip_vertical = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0) < 0.5 if self.vertical else tf.zeros([batch_size], dtype=tf.bool)

        return {
            "flip_horizontal": flip_horizontal,
            "flip_vertical": flip_vertical,
        }

    def augment_image(self, image, transformation):
        flip_horizontal = transformation["flip_horizontal"]
        flip_vertical = transformation["flip_vertical"]

        if self.horizontal:
            image = tf.where(flip_horizontal, tf.image.flip_left_right(image), image)
        if self.vertical:
            image = tf.where(flip_vertical, tf.image.flip_up_down(image), image)

        return image

    def augment_label(self, label, transformation):
        flip_horizontal = transformation["flip_horizontal"]
        flip_vertical = transformation["flip_vertical"]

        if self.horizontal:
            label = tf.where(flip_horizontal, tf.image.flip_left_right(label), label)
        if self.vertical:
            label = tf.where(flip_vertical, tf.image.flip_up_down(label), label)

        return label

    def augment_bounding_boxes(self, images, bounding_boxes, transformation=None):
        """
        Applies horizontal and/or vertical flip to bounding boxes in a batched and GPU-friendly way.

        :param images: 4D Tensor of shape (B, H, W, C)
        :param bounding_boxes: 3D Tensor of shape (B, N, 4), in [x_min, y_min, x_max, y_max] format
        :param transformation: Dict containing 'flip_horizontal' and 'flip_vertical' (both shape [B])
        :return: Transformed bounding_boxes, shape (B, N, 4)
        """
        image_shape = tf.shape(images)
        batch_size = image_shape[0]
        height = tf.cast(image_shape[1], tf.float32)
        width = tf.cast(image_shape[2], tf.float32)

        x_min, y_min, x_max, y_max = tf.split(bounding_boxes, 4, axis=-1)

        flip_h = tf.cast(tf.reshape(transformation["flip_horizontal"], [batch_size, 1, 1]), tf.bool)
        flip_v = tf.cast(tf.reshape(transformation["flip_vertical"], [batch_size, 1, 1]), tf.bool)

        new_x_min = tf.where(flip_h, width - x_max, x_min)
        new_x_max = tf.where(flip_h, width - x_min, x_max)

        new_y_min = tf.where(flip_v, height - y_max, y_min)
        new_y_max = tf.where(flip_v, height - y_min, y_max)

        flipped_bboxes = tf.concat([new_x_min, new_y_min, new_x_max, new_y_max], axis=-1)
        return flipped_bboxes

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"mode": self.mode})
        return config


@keras.saving.register_keras_serializable(package='Augmentation', name='RandomShear')
class RandomShear(AugmentationLayer):

    """
    A preprocessing layer which randomly applies shear transformations during training.

    This layer performs a random shear on each image and its corresponding segmentation mask
    (if available), filling any resulting empty regions according to the specified `fill_mode`
    and `interpolation` strategy. Shearing is only applied during training when `active=True`.
    At inference time or when inactive, the layer behaves as a no-op and all inputs—including
    images, targets, and bounding boxes—are passed through unchanged.

    Input shape:
        {'images': 3D (unbatched) or 4D (batched) tensor,
         'targets': 1D, 2D, 3D, or 4D tensor}

        `images` must follow shape `(..., height, width, channels)` using `"channels_last"` format.
        `targets` can be 1D–4D. If 2D or lower, the target will not be augmented.

    Output shape:
        {'images': 3D (unbatched) or 4D (batched) tensor,
         'targets': 1D, 2D, 3D, or 4D tensor}

        Output shape and format are identical to input.

    Arguments:
      shear_angle: A float, list, or tuple specifying the shear angle in degrees. If a scalar
          is provided, the angle will be sampled from [-shear_angle, +shear_angle]. If a tuple
          or list of two values is provided, the shear angle is sampled from (angle[0], angle[1]).
      active: Whether the layer is active during training. If False, input is returned unchanged.
      name: Name assigned to the layer instance.
      fill_mode: Strategy used to fill pixels outside the transformed image boundaries.
         One of: `{"constant", "reflect", "wrap", "nearest"}`.
         - *reflect*: `(d c b a | a b c d | d c b a)` — reflects about the edge of the last pixel.
         - *constant*: `(k k k k | a b c d | k k k k)` — fills with constant value `k` (default 0).
         - *wrap*: `(a b c d | a b c d | a b c d)` — wraps around to the opposite edge.
         - *nearest*: `(a a a a | a b c d | d d d d)` — extends using nearest edge pixel.
      interpolation: Interpolation mode used for resampling. One of: `"nearest"`, `"bilinear"`.
      image_key: Key used to retrieve the image tensor from the input dictionary.
      target_key: Key used to retrieve the mask or label tensor from the input dictionary.
      bounding_box_key: Key used to retrieve bounding box data from the input dictionary.
      kwargs: Additional keyword arguments passed to the base `AugmentationLayer`.

    Raises:
      TypeError: If `shear_angle` is not a float, int, list, or tuple.
      ValueError: If `shear_angle` is a list/tuple of length not equal to 2, or if upper < lower.
    """

    def __init__(self,
                 shear_angle: Union[list, tuple, float] = (0, 5),
                 active: bool = True,
                 name: str = 'random_shear',
                 fill_mode="reflect",
                 interpolation="bilinear",
                 image_key: str = 'images',
                 target_key: str = 'targets',
                 bounding_box_key: str = "bounding_boxes",
                 batches_before_aug: Optional[int] = None,
                 **kwargs):

        super().__init__(name=name,
                         active=active,
                         image_key=image_key,
                         target_key=target_key,
                         bounding_box_key=bounding_box_key,
                         batches_before_aug=batches_before_aug,
                         **kwargs)

        self.shear_angle = shear_angle

        check_input(inputs=shear_angle, name='shear_angle', input_range=(-45, 45))

        if isinstance(shear_angle, (tuple, list)):
            self.lower = np.deg2rad(shear_angle[0])
            self.upper = np.deg2rad(shear_angle[1])

        elif isinstance(shear_angle, (int, float)):
            self.lower = np.deg2rad(-shear_angle)
            self.upper = np.deg2rad(shear_angle)

        else:
            raise TypeError(
                "The shear_angle must be a list, tuple, int or float. Got {} of type {}".format(shear_angle,
                                                                                                type(shear_angle))
            )

        if self.upper < self.lower:
            raise ValueError(
                "The upper boundary cannot be less than the lower boundary. Got shear_angle = {}".format(shear_angle)
            )

        check_fill_mode_and_interpolation(fill_mode, interpolation)
        self.fill_mode = fill_mode
        self.interpolation = interpolation

    def get_random_transformation(self, image=None, label=None, bounding_box=None):
        batch_size = tf.shape(image)[0]
        angles = tf.random.uniform(shape=[batch_size], minval=self.lower, maxval=self.upper)
        return {"angles": angles}

    def augment_image(self, image, transformation):
        image = utils.ensure_tensor(image, self.compute_dtype)
        image_shape = tf.shape(image)

        img_hd = tf.cast(image_shape[H_AXIS], tf.float32)
        img_wd = tf.cast(image_shape[W_AXIS], tf.float32)
        angles = transformation["angles"]

        output = transform(images=image,
                           transforms=get_shear_matrix(angles, img_hd, img_wd),
                           fill_mode=self.fill_mode,
                           interpolation=self.interpolation,
                           )

        return output

    def augment_label(self, label, transformation):
        label = utils.ensure_tensor(label, self.compute_dtype)
        image_shape = tf.shape(label)

        label_hd = tf.cast(image_shape[H_AXIS], tf.float32)
        label_wd = tf.cast(image_shape[W_AXIS], tf.float32)

        angles = transformation["angles"]
        output = transform(images=label,
                           transforms=get_shear_matrix(angles, label_hd, label_wd),
                           fill_mode=self.fill_mode,
                           interpolation='nearest')

        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"shear_angle": self.shear_angle,
                       "fill_mode": self.fill_mode,
                       "interpolation": self.interpolation})
        return config


@keras.saving.register_keras_serializable(package='Augmentation', name='RandomZoom')
class RandomZoom(AugmentationLayer):

    """
    A preprocessing layer which randomly zooms images and their segmentation masks during training.

    This layer zooms in or out along the height and/or width axes by a random factor. You can
    specify zoom bounds as a single value (symmetric zoom in both directions) or as a range.
    If `width_factor` is not provided, the zoom preserves the aspect ratio and applies the same
    transformation to both axes. This layer only applies transformations during training when
    `active=True`. At inference time or when inactive, the layer behaves as a no-op and returns
    all inputs (images, targets, bounding boxes) unchanged.

    Input shape:
        {'images': 3D (unbatched) or 4D (batched) tensor,
         'targets': 1D, 2D, 3D, or 4D tensor}

        `images` must follow shape `(..., height, width, channels)` using `"channels_last"` format.
        `targets` can be 1D–4D. If 2D or lower, the target will not be augmented.

    Output shape:
        {'images': 3D (unbatched) or 4D (batched) tensor,
         'targets': 1D, 2D, 3D, or 4D tensor}

        Output shape and format are identical to input.

    Arguments:
      height_factor: A float, int, list, or tuple specifying the zoom range along the height axis.
         Values are interpreted as percentages. For example:
         - `(90, 110)` samples a zoom between +10% in and −10% out.
         - A scalar `x` will be treated as `(-x, x)` symmetrically.
      width_factor: Optional float, int, list, or tuple specifying zoom range along the width axis.
         Same format as `height_factor`. If `None`, the same factor is applied to both height and width.
      fill_mode: Strategy used to fill pixels outside the transformed image boundaries.
         One of: `{"constant", "reflect", "wrap", "nearest"}`.
         - *reflect*: `(d c b a | a b c d | d c b a)` — reflects about the edge of the last pixel.
         - *constant*: `(k k k k | a b c d | k k k k)` — fills with constant value `k` (default 0).
         - *wrap*: `(a b c d | a b c d | a b c d)` — wraps around to the opposite edge.
         - *nearest*: `(a a a a | a b c d | d d d d)` — extends using nearest edge pixel.
      interpolation: Interpolation method used for resampling. One of: `"nearest"`, `"bilinear"`.
      fill_value: Float value to use when `fill_mode="constant"` to fill empty space.
      image_key: Key used to retrieve the image tensor from the input dictionary.
      target_key: Key used to retrieve the mask or label tensor from the input dictionary.
      bounding_box_key: Key used to retrieve bounding box data from the input dictionary.
      name: Name assigned to the layer instance.
      kwargs: Additional keyword arguments passed to the base `AugmentationLayer`.

    Raises:
      TypeError: If `height_factor` or `width_factor` is not a float, int, list, or tuple.
      ValueError: If zoom bounds contain invalid values (e.g., below 0 or upper < lower).
    """

    def __init__(self,
                 height_factor: Union[int, list, tuple] = (90, 110),
                 width_factor: Union[int, list, tuple] = None,
                 fill_mode="reflect",
                 interpolation="bilinear",
                 fill_value=0.0,
                 image_key: str = 'images',
                 target_key: str = 'targets',
                 bounding_box_key: str = "bounding_boxes",
                 batches_before_aug: Optional[int] = None,
                 name: str = 'random_zoom',
                 **kwargs):

        super().__init__(custom_interp_datapoints=[-200.0, -100.0, 0.0, 100.0, 200.0],
                         tensorflow_interp_datapoints=[-1.0, -0.5, 0.0, 0.5, 1.0],
                         image_key=image_key,
                         target_key=target_key,
                         bounding_box_key=bounding_box_key,
                         batches_before_aug=batches_before_aug,
                         name=name,
                         **kwargs)

        check_input(inputs=height_factor, name='height_factor', input_range=(1, 200))

        # convert custom value to Tensorflow value
        self.height_factor = interpolate_to_tensorflow_value(custom_value=height_factor,
                                                             custom_datapoints=self.custom_interp_datapoints,
                                                             tensorflow_datapoints=self.tensorflow_interp_datapoints)

        if isinstance(height_factor, (tuple, list)):
            self.height_lower = self.height_factor[0]
            self.height_upper = self.height_factor[1]
        else:
            self.height_lower = -self.height_factor
            self.height_upper = self.height_factor

        if width_factor is not None:
            check_input(inputs=width_factor, name='width_factor',
                        input_range=(0, 200))
            if isinstance(width_factor, (tuple, list)):
                self.width_lower = width_factor[0]
                self.width_upper = width_factor[1]
            else:
                self.width_lower = -width_factor
                self.width_upper = width_factor

            self.width_factor = interpolate_to_tensorflow_value(custom_value=width_factor,
                                                                custom_datapoints=self.custom_interp_datapoints,
                                                                tensorflow_datapoints=self.tensorflow_interp_datapoints)
        else:
            self.width_factor = None

        check_fill_mode_and_interpolation(fill_mode, interpolation)

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.interpolation = interpolation

    def get_random_transformation(self, image=None, label=None, bounding_box=None):
        batch_size = tf.shape(image)[0]

        height_zoom = tf.random.uniform(
            shape=[batch_size, 1],
            minval=1.0 - self.height_lower,
            maxval=1.0 - self.height_upper,
        )

        if self.width_factor is not None:
            width_zoom = tf.random.uniform(
                shape=[batch_size, 1],
                minval=1.0 - self.width_lower / 100,
                maxval=1.0 - self.width_upper / 100,
            )
        else:
            width_zoom = height_zoom

        return {"height_zoom": height_zoom, "width_zoom": width_zoom}

    def augment_image(self, image, transformation):
        image = utils.ensure_tensor(image, self.compute_dtype)
        image_shape = tf.shape(image)

        img_hd = tf.cast(image_shape[H_AXIS], tf.float32)
        img_wd = tf.cast(image_shape[W_AXIS], tf.float32)

        width_zoom = transformation["width_zoom"]
        height_zoom = transformation["height_zoom"]
        zooms = tf.cast(tf.concat([width_zoom, height_zoom], axis=1), dtype=tf.float32)

        output = transform(image,
                           get_zoom_matrix(zooms, img_hd, img_wd),
                           fill_mode=self.fill_mode,
                           fill_value=self.fill_value,
                           interpolation=self.interpolation
                           )

        return output

    def augment_label(self, label, transformation):
        label = utils.ensure_tensor(label, self.compute_dtype)
        label_shape = tf.shape(label)

        img_hd = tf.cast(label_shape[H_AXIS], tf.float32)
        img_wd = tf.cast(label_shape[W_AXIS], tf.float32)

        width_zoom = transformation["width_zoom"]
        height_zoom = transformation["height_zoom"]

        zooms = tf.cast(tf.concat([width_zoom, height_zoom], axis=1), dtype=tf.float32)
        output = transform(label,
                           get_zoom_matrix(zooms, img_hd, img_wd),
                           fill_mode=self.fill_mode,
                           fill_value=self.fill_value,
                           interpolation='nearest',
                           )
        return output

    def augment_bounding_boxes(self, image, bounding_boxes, transformation=None):
        """
        Adjusts bounding boxes to match the zoomed image.

        :param image: 4D tensor (B, H, W, C)
        :param bounding_boxes: 3D tensor (B, N, 4) where each box is [x_min, y_min, x_max, y_max]
        :param transformation: Dict containing 'height_zoom' and 'width_zoom' (shape: [B, 1])
        :return: Adjusted bounding boxes of shape (B, N, 4)
        """
        image_shape = tf.shape(image)
        height = tf.cast(image_shape[1], tf.float32)
        width = tf.cast(image_shape[2], tf.float32)

        # Get zoom factors
        zoom_h = tf.reshape(transformation["height_zoom"], [-1, 1, 1])  # (B, 1, 1)
        zoom_w = tf.reshape(transformation["width_zoom"], [-1, 1, 1])   # (B, 1, 1)

        # Compute center of image (B, 1)
        center_x = width / 2.0
        center_y = height / 2.0

        # Expand to (B, 1, 1)
        center_x = tf.reshape(center_x, [1, 1, 1])
        center_y = tf.reshape(center_y, [1, 1, 1])

        # Center box coordinates
        x_min, y_min, x_max, y_max = tf.split(bounding_boxes, 4, axis=-1)
        x_centered = tf.stack([x_min - center_x, x_max - center_x], axis=-1)  # (B, N, 2, 1)
        y_centered = tf.stack([y_min - center_y, y_max - center_y], axis=-1)

        # Apply zoom
        x_scaled = x_centered / zoom_w
        y_scaled = y_centered / zoom_h

        # Re-center
        new_x_min = x_scaled[:, :, 0] + center_x
        new_x_max = x_scaled[:, :, 1] + center_x
        new_y_min = y_scaled[:, :, 0] + center_y
        new_y_max = y_scaled[:, :, 1] + center_y

        # Clip to image bounds
        new_x_min = tf.clip_by_value(new_x_min, 0.0, width)
        new_x_max = tf.clip_by_value(new_x_max, 0.0, width)
        new_y_min = tf.clip_by_value(new_y_min, 0.0, height)
        new_y_max = tf.clip_by_value(new_y_max, 0.0, height)

        return tf.concat([new_x_min, new_y_min, new_x_max, new_y_max], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
            "height_factor": self.height_factor,
            "width_factor": self.width_factor,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
            "interpolation": self.interpolation,
        })
        return config


@keras.saving.register_keras_serializable(package='Augmentation', name='AffineTransform')
class AffineTransform(AugmentationLayer):

    """
    A preprocessing layer that applies affine transformations—rotation, shear, and zoom—
    to images and, if provided, their segmentation masks and bounding boxes.

    Each transformation can be toggled independently using the `rotate`, `shear`, and `zoom` flags.
    Transformations are only applied during training when `active=True`. During inference or when
    inactive, the layer behaves as a no-op and returns all inputs unchanged.

    Input shape:
        {'images': 3D (unbatched) or 4D (batched) tensor,
         'targets': 1D, 2D, 3D, or 4D tensor}

        `images` must follow shape `(..., height, width, channels)` using `"channels_last"` format.
        `targets` can be 1D–4D. If 2D or lower, the target will not be augmented.

    Output shape:
        {'images': 3D (unbatched) or 4D (batched) tensor,
         'targets': 1D, 2D, 3D, or 4D tensor}

        Output shape and format are identical to input.

    Arguments:
      rotate: Boolean indicating whether to apply random rotation during training.
      theta: A float, int, tuple, or list specifying the rotation angle in degrees.
         - Range: -180 to +180
         - If scalar: sampled from [-theta, +theta]
         - If tuple/list: sampled from [theta[0], theta[1]]
      shear: Boolean indicating whether to apply random shear during training.
      shear_angle: A float, list, or tuple specifying the shear angle in degrees.
         - Range: -45 to +45
         - If scalar: sampled from [-shear_angle, +shear_angle]
         - If tuple/list: sampled from [shear_angle[0], shear_angle[1]]
      zoom: Boolean indicating whether to apply random zoom during training.
      zoom_factor: A float, int, tuple, or list specifying the zoom scale in percentage.
         - Range: -200 to 200 (percent of original size)
         - If scalar `x`: interpreted as (-x, +x), applied as percent deviation from original. For instance if zoom_factor is set to 20, it means zoom into the image by a random percentage between -20% and +20%. -30 mean zoom-out of the image by 30% of its original size.
         - If tuple/list: sampled from [zoom_factor[0], zoom_factor[1]]. E.g. [20 - 40] means, zoom in by a random percentage between 20%  to 40%.
      fill_mode: Strategy used to fill pixels outside the transformed image boundaries.
         One of: `{"constant", "reflect", "wrap", "nearest"}`.
         - *reflect*: `(d c b a | a b c d | d c b a)` — reflects about the edge of the last pixel.
         - *constant*: `(k k k k | a b c d | k k k k)` — fills with constant value `k` (default 0).
         - *wrap*: `(a b c d | a b c d | a b c d)` — wraps around to the opposite edge.
         - *nearest*: `(a a a a | a b c d | d d d d)` — extends using nearest edge pixel.
      interpolation: Interpolation method used for resampling. One of: `"nearest"`, `"bilinear"`.
      fill_value: Float value to use when `fill_mode="constant"` to fill empty space.
      image_key: Key used to retrieve the image tensor from the input dictionary.
      target_key: Key used to retrieve the mask or label tensor from the input dictionary.
      bounding_box_key: Key used to retrieve bounding box data from the input dictionary.
      name: Name assigned to the layer instance.
      kwargs: Additional keyword arguments passed to the base `AugmentationLayer`.

    Raises:
      TypeError: If any transformation parameter is of an invalid type.
      ValueError: If theta, shear_angle, or zoom_factor are out of range or malformed
        (e.g., upper < lower or negative zoom).
    """

    def __init__(self,
                 rotate: bool = False,
                 theta: Union[int, List[int], Tuple[int, int], None] = (-180, 180),
                 shear: bool = False,
                 shear_angle: Union[list, tuple, float, None] = (0, 4),
                 zoom: bool = False,
                 zoom_factor: Union[int, list, tuple, None] = (1, 90),
                 fill_mode="reflect",
                 interpolation="bilinear",
                 fill_value=0.0,
                 image_key: str = 'images',
                 target_key: str = 'targets',
                 bounding_box_key: str = "bounding_boxes",
                 batches_before_aug: Optional[int] = None,
                 name='affine_transform',
                 **kwargs):

        super().__init__(image_key=image_key,
                         target_key=target_key,
                         bounding_box_key=bounding_box_key,
                         batches_before_aug=batches_before_aug,
                         name=name,
                         **kwargs)

        if any([rotate, shear, zoom]):
            self.active = tf.Variable(tf.constant(True), dtype=tf.bool, trainable=False)
        else:
            self.active = tf.Variable(tf.constant(False), dtype=tf.bool, trainable=False)

        check_fill_mode_and_interpolation(fill_mode, interpolation)
        self.rotate = tf.Variable(tf.constant(rotate), dtype=tf.bool, trainable=False)
        self.theta = theta
        self.rotate_initially_active = rotate

        self.shear = tf.Variable(tf.constant(shear), dtype=tf.bool, trainable=False)
        self.shear_angle = shear_angle
        self.shear_initially_active = shear

        self.zoom = tf.Variable(tf.constant(zoom), dtype=tf.bool, trainable=False)
        self.zoom_factor = zoom_factor
        self.zoom_initially_active = zoom

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.interpolation = interpolation

        # Check input value
        self.check_rotate_values()
        self.check_shear_values()
        self.check_zoom_values()

    def get_random_transformation(self, image=None, label=None, bounding_box=None):
        """Generate transformation parameters for rotation, shear and zoom"""
        batch_size = tf.shape(image)[0]

        # Default theta, shear, and zoom to 0, 0, 1, so that image is not rotated, sheared or zoomed.
        # when their (zoom, rotation or shear) matrix is multiplied by transformed matrix in
        # .get_transform_matrix method.
        transform_parameters = {
            'theta': tf.zeros(shape=[batch_size], dtype=tf.float32),
            'shear': tf.zeros(shape=[batch_size], dtype=tf.float32),
            'zoom': tf.ones(shape=[batch_size], dtype=tf.float32)
        }

        # Rotation
        if self.apply_rotate:
            transform_parameters['theta'] = tf.random.uniform(
                shape=[batch_size], minval=self.rotate_lower, maxval=self.rotate_upper)

        # Shear
        if self.apply_shear:
            transform_parameters['shear'] = tf.random.uniform(
                shape=[batch_size], minval=self.shear_lower, maxval=self.shear_upper)

        # Zoom
        if self.apply_zoom:
            transform_parameters['zoom'] = tf.random.uniform(
                shape=[batch_size], minval=1.0 - self.zoom_lower, maxval=1.0 - self.zoom_upper)

        return transform_parameters

    @staticmethod
    def get_transform_matrix(transform_parameters):
        """
        Produces the transform matrix for conducting all the geometric
        transformed that have been chosen.
        """
        theta = transform_parameters['theta']
        shear = transform_parameters['shear']
        zoom = transform_parameters['zoom']

        cos_theta = tf.cos(theta)
        sin_theta = tf.sin(theta)

        batch_size = tf.shape(theta)[0]
        zeros = tf.zeros([batch_size], dtype=tf.float32)
        ones = tf.ones([batch_size], dtype=tf.float32)

        rotation_matrices = tf.stack([
            tf.stack([cos_theta, -sin_theta, zeros], axis=1),
            tf.stack([sin_theta, cos_theta,  zeros], axis=1),
            tf.stack([zeros,     zeros,      ones], axis=1)
        ], axis=1)  # shape: [batch_size, 3, 3]

        shear_matrices = tf.stack([
            tf.stack([ones, -tf.sin(shear), zeros], axis=1),
            tf.stack([zeros, tf.cos(shear), zeros], axis=1),
            tf.stack([zeros, zeros,          ones], axis=1)
        ], axis=1)

        zoom_matrices = tf.stack([
            tf.stack([zoom, zeros, zeros], axis=1),
            tf.stack([zeros, zoom, zeros], axis=1),
            tf.stack([zeros, zeros, ones], axis=1)
        ], axis=1)


        # Combine
        return tf.matmul(tf.matmul(rotation_matrices, shear_matrices), zoom_matrices)  # shape: [batch, 3, 3]

    def augment_image(self, image, transformation):
        image = utils.ensure_tensor(image, self.compute_dtype)
        image_shape = tf.shape(image)

        img_hd = tf.cast(image_shape[H_AXIS], tf.float32)
        img_wd = tf.cast(image_shape[W_AXIS], tf.float32)

        transform_matrix = self.get_transform_matrix(transform_parameters=transformation)
        centered_matrix = center_transformation_matrix(transform_matrix, img_hd, img_wd)

        output = transform(images=image,
                           transforms=centered_matrix,
                           fill_mode=self.fill_mode,
                           fill_value=self.fill_value,
                           interpolation=self.interpolation,
                           )
        return output

    def augment_label(self, label, transformation):
        label = utils.ensure_tensor(label, self.compute_dtype)
        image_shape = tf.shape(label)

        label_hd = tf.cast(image_shape[H_AXIS], tf.float32)
        label_wd = tf.cast(image_shape[W_AXIS], tf.float32)

        transform_matrix = self.get_transform_matrix(transform_parameters=transformation)
        centered_matrix = center_transformation_matrix(transform_matrix, label_hd, label_wd)

        output = transform(images=label,
                           transforms=centered_matrix,
                           fill_mode=self.fill_mode,
                           fill_value=self.fill_value,
                           interpolation='nearest',
                           )
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def check_rotate_values(self):
        """Check the rotation range and values"""
        if self.rotate and self.theta is not None:
            self.apply_rotate = True

            check_input(inputs=self.theta, name='theta', input_range=(-180, 180))

            if isinstance(self.theta, (tuple, list)):
                self.rotate_lower = np.deg2rad(self.theta[0])
                self.rotate_upper = np.deg2rad(self.theta[1])

            elif isinstance(self.theta, (int, float)):
                self.rotate_lower = np.deg2rad(-self.theta)
                self.rotate_upper = np.deg2rad(self.theta)
            else:
                raise TypeError(
                    "`theta` must be a list, tuple, int or float. "
                    "Got {} of type {}".format(self.theta, type(self.theta))
                )
        else:
            self.apply_rotate = False

    def check_shear_values(self):
        """Check the shear state and range"""
        if self.shear and self.shear_angle is not None:
            self.apply_shear = True

            check_input(inputs=self.shear_angle, name='shear_angle', input_range=(-45, 45))

            if isinstance(self.shear_angle, (tuple, list)):
                self.shear_lower = np.deg2rad(self.shear_angle[0])
                self.shear_upper = np.deg2rad(self.shear_angle[1])

            elif isinstance(self.shear_angle, (int, float)):
                self.shear_lower = np.deg2rad(-self.shear_angle)
                self.shear_upper = np.deg2rad(self.shear_angle)
            else:
                raise TypeError(
                    "The shear_angle must be a list, tuple, int or float. Got {} of type {}".format(self.shear_angle,
                                                                                                    type(
                                                                                                        self.shear_angle))
                )

        else:
            self.apply_shear = False

    def check_zoom_values(self):
        """Check the zoom state and range"""
        custom_interp_datapoints = [-200.0, -100.0, 0.0, 100.0, 200.0]
        tensorflow_interp_datapoints = [-1.0, -0.5, 0.0, 0.5, 1.0]

        if self.zoom and self.zoom_factor is not None:
            self.apply_zoom = True

            check_input(inputs=self.zoom_factor, name='zoom_factor', input_range=(-200, 200))
            zoom_factor = interpolate_to_tensorflow_value(custom_value=self.zoom_factor,
                                                          custom_datapoints=custom_interp_datapoints,
                                                          tensorflow_datapoints=tensorflow_interp_datapoints)

            if isinstance(self.zoom_factor, (tuple, list)):
                self.zoom_lower = zoom_factor[0]
                self.zoom_upper = zoom_factor[1]
            else:
                self.zoom_lower = -abs(zoom_factor)
                self.zoom_upper = zoom_factor

        else:
            self.apply_zoom = False

    def get_config(self):
        config = super().get_config()
        config.update({"rotate": self.rotate,
                       "theta": self.theta,
                       "shear": self.shear,
                       "shear_angle": self.shear_angle,
                       "zoom": self.zoom,
                       "zoom_factor": self.zoom_factor,
                       "fill_mode": self.fill_mode,
                       "fill_value": self.fill_value,
                       "interpolation": self.interpolation
                       })

        return config


@keras.saving.register_keras_serializable(package='Augmentation', name='RandomJitter')
class RandomJitter(AugmentationLayer):
    """
    A preprocessing layer that applies random **spatial jitter** to images and their corresponding
    segmentation masks during training.

    This layer simulates small camera shifts or object displacements by padding the image and then
    cropping a randomly shifted region of the original size. It does not modify individual pixel
    intensities (i.e., not pixel-level jitter).

    Jitter is only applied during training. During inference, the input is returned unchanged.

    ---
    Input:
        A dictionary with keys:
            - 'images': Tensor of shape 3D or 4D (`[H, W, C]` or `[B, H, W, C]`), dtype float or int
            - 'targets': Tensor of shape 3D or 4D. If target is 1D or 2D, it is passed through unchanged.

    Output:
        A dictionary with the same keys and shapes as input. Spatially jittered versions of input tensors.

    ---
    Example:
        >>> jitter = RandomJitter(jitter_range=(5, 10))
        >>> images = tf.random.uniform((4, 224, 224, 3))
        >>> masks = tf.random.uniform((4, 224, 224, 1))
        >>> output = jitter({'images': images, 'targets': masks}, training=True)
        >>> output['images'].shape
        TensorShape([4, 224, 224, 3])

    ---
    Args:
        jitter_range: Tuple of 2 integers specifying min and max spatial offset in pixels.
                      A random offset is sampled in [min, max] for height and width.
                      - Range allowed: (0, 60)
                      - Example: (5, 15) allows up to 15-pixel shifts

        fill_mode: Pixel fill method for padded areas. One of:
                   - "constant": Pads with `fill_value`
                   - "reflect": Reflects image border
                   - "wrap": Wraps image content
                   - "nearest": Replicates nearest pixels

        interpolation: Interpolation method when cropping. One of: "nearest", "bilinear"

        fill_value: Value to use for padding if `fill_mode="constant"`

        image_key: Key to access image tensor in the input dictionary (default: "images")

        target_key: Key to access mask/label tensor in the input dictionary (default: "targets")

        bounding_box_key: Optional key to access bounding boxes (default: "bounding_boxes")

        name: Name of the layer.

        kwargs: Additional keyword arguments passed to base layer.

    ---
    Raises:
        TypeError: If `jitter_range` is not a list or tuple of two integers.
        ValueError: If `jitter_range[1]` is less than `jitter_range[0]`.
    """

    def __init__(self,
                 jitter_range: Union[List[int], Tuple[int, int]] = (0, 60),
                 name: str = 'random_jitter',
                 image_key: str = 'images',
                 target_key: str = 'targets',
                 bounding_box_key: str = "bounding_boxes",
                 batches_before_aug: Optional[int] = None,
                 **kwargs):
        base_preprocessing_layer.keras_kpl_gauge.get_cell("RandomDualJitter").set(True)

        super().__init__(name=name,
                         image_key=image_key,
                         target_key=target_key,
                         bounding_box_key=bounding_box_key,
                         batches_before_aug=batches_before_aug,
                         **kwargs)

        self.jitter_range = jitter_range

        check_input(inputs=jitter_range, name='jitter_range', input_range=(0, 60))

        if isinstance(jitter_range, (tuple, list)):
            self.lower = jitter_range[0]
            self.upper = jitter_range[1]
        else:
            raise TypeError(
                "The jitter_range must be a list or tuple tuple. Got {} of type {}".format(jitter_range,
                                                                                           type(jitter_range))
            )
        if not all(isinstance(value, int) for value in (self.lower, self.upper)):
            raise TypeError(
                "The min and max values within the jitter_range must be integer, "
                "got {}: [{}, {}]".format(jitter_range, type(self.lower), type(self.upper))
            )
        if self.upper < self.lower:
            raise ValueError(
                "The upper boundary cannot be less than the lower boundary. Got jitter_range = {}".format(jitter_range)
            )

    def get_random_transformation(self, image=None, label=None, bounding_box=None):
        batch_size = tf.shape(image)[0]

        jitter_h = tf.random.uniform(shape=[batch_size], minval=self.lower,
                                     maxval=self.upper + 1, dtype=tf.int32)

        jitter_w = tf.random.uniform(shape=[batch_size], minval=self.lower,
                                     maxval=self.upper + 1, dtype=tf.int32)

        return {"jitter_h": jitter_h,
                "jitter_w": jitter_w}

    def _resize_image(self, image, jitter_h, jitter_w):
        image = tf.convert_to_tensor(image)
        height = tf.shape(image)[0]
        width = tf.shape(image)[1]

        # Pad to max size
        max_width = width + self.upper
        max_height = height + self.upper
        padded = tf.image.resize_with_crop_or_pad(image, max_height, max_width)

        # Crop the jittered region
        cropped = tf.image.crop_to_bounding_box(
            padded,
            offset_height=jitter_h,
            offset_width=jitter_w,
            target_height=height,
            target_width=width
        )
        return cropped

    def augment_image(self, image, transformation):
        image = utils.ensure_tensor(image, self.compute_dtype)
        jitter_h = transformation['jitter_h']
        jitter_w = transformation['jitter_w']

        def apply_jitter(inputs):
            img, jh, jw = inputs
            return self._resize_image(image=img, jitter_h=jh, jitter_w=jw)

        output = tf.map_fn(fn=apply_jitter,
                           elems=(image, jitter_h, jitter_w),
                           fn_output_signature=tf.TensorSpec(shape=image.shape[1:],
                                                             dtype=image.dtype))

        return output

    def augment_label(self, label, transformation):
        label = utils.ensure_tensor(label, self.compute_dtype)
        jitter_h = transformation['jitter_h']
        jitter_w = transformation['jitter_w']

        def apply_jitter(inputs):
            lbl, jh, jw = inputs
            return self._resize_image(image=lbl, jitter_h=jh, jitter_w=jw)

        output = tf.map_fn(fn=apply_jitter,
                           elems=(label, jitter_h, jitter_w),
                           fn_output_signature=tf.TensorSpec(shape=label.shape[1:],
                                                             dtype=label.dtype))

        return output

    def augment_bounding_boxes(self, image, bounding_boxes, transformation=None):
        """
        Shift bounding boxes according to the jitter offsets.

        :param image: Batched image tensor, used to determine batch size
        :param bounding_boxes: Tensor of shape (B, N, 4)
        :param transformation: Dict containing 'jitter_h' and 'jitter_w' of shape (B,)
        :return: Tensor of shape (B, N, 4) with adjusted coordinates
        """
        bbox_dtype = bounding_boxes.dtype
        jitter_h = tf.cast(tf.expand_dims(transformation["jitter_h"], axis=1), bbox_dtype)  # (B, 1)
        jitter_w = tf.cast(tf.expand_dims(transformation["jitter_w"], axis=1), bbox_dtype)  # (B, 1)

        # Broadcast to match number of bboxes per image
        jitter_shift = tf.concat([jitter_w, jitter_h, jitter_w, jitter_h], axis=-1)  # (B, 4)

        # If bounding_boxes has shape (B, N, 4), tile jitter to match (B, N, 4)
        jitter_shift = tf.expand_dims(jitter_shift, axis=1)  # (B, 1, 4)
        jitter_shift = tf.broadcast_to(jitter_shift, tf.shape(bounding_boxes))  # (B, N, 4)

        # Subtract the shift from bounding box coordinates
        adjusted_bboxes = bounding_boxes - jitter_shift
        return adjusted_bboxes

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"jitter_range": self.jitter_range})
        return config


@keras.saving.register_keras_serializable(package='Augmentation', name='RandomBrightness')
class RandomBrightness(AugmentationLayer):
    """
    A preprocessing layer that randomly adjusts the brightness of input images during training.

    This layer applies a random brightness shift to each image in the batch by adding an offset
    proportional to the input intensity range. A different brightness factor is sampled for each image.
    At inference time, the layer behaves as a no-op.

    Input shape:
        {'images': 3D (unbatched) or 4D (batched) tensor,
         'targets': 3D (unbatched), 4D (batched), 2D, or 1D tensor}

        Both `images` and `targets` should follow `(..., height, width, channels)` format,
        using `"channels_last"` convention.

    Output shape:
        {'images': 3D (unbatched) or 4D (batched) tensor,
         'targets': 3D (unbatched), 4D (batched), 2D, or 1D tensor}

        Only the `images` tensor is modified. Brightness is adjusted and clipped to the provided
        `image_intensity_range`. `targets` are passed through unchanged.

    Example:
        >>> layer = RandomBrightness(factor=(0.1, 0.5), image_intensity_range=(0.0, 255.0))
        >>> img = tf.random.uniform((2, 128, 128, 3))  # Batch of RGB images
        >>> mask = tf.random.uniform((2, 128, 128, 1)) # Corresponding masks
        >>> output = layer({'images': img, 'targets': mask}, training=True)
        >>> output['images'].shape
        TensorShape([2, 128, 128, 3])

    Arguments:
        factor: A float, or tuple/list of two floats between -1.0 and 1.0. The
            factor is used to determine the lower bound and upper bound of the
            brightness adjustment.  A float will be randomly chosen between the limits.
                - Values above 0 brighten the image; values below 0 darken it.
                - When only one float is provided, eg, 0.2, then -0.2 will be used for lower bound
                  and 0.2 will be used for upper bound.
                - If a tuple/list is provided, the factor is sampled from `[factor[0], factor[1]]`.

        image_intensity_range: A tuple or list specifying the lower and upper bounds of pixel
            values (e.g., `[0, 1]` or `[0, 255]`). If the input is normalized to 0.0 - 1.0, use
            image_intensity_range of (0, 1) , else, use (0, 255)
                - The brightness offset is computed relative to this range.
                - Output is clipped to stay within this range after transformation.

        image_key: Dictionary key used to access the image tensor. Default is `'images'`.
        target_key: Dictionary key used to access the target tensor. Default is `'targets'`.
        bounding_box_key: Dictionary key used to access the bounding box tensor. Default is `'bounding_boxes'`.
        name: Name of the layer instance.
        kwargs: Additional keyword arguments passed to the base `AugmentationLayer`.

    Raises:
        ValueError: If `factor` is outside `[-1.0, 1.0]`, or if `image_intensity_range` is invalid.
        TypeError: If `factor` is not a float, tuple, or list.
    """


    _FACTOR_VALIDATION_ERROR = (
        "The `factor` argument should be a number (or a list of two numbers) "
        "in the range [-1.0, 1.0]. ")

    _VALUE_RANGE_VALIDATION_ERROR = (
        "The `image_intensity_range` argument should be a tuple of two numbers. ")

    def __init__(self,
                 active: bool = False,
                 factor: Union[float, List, Tuple] = (0.0, 0.15),
                 image_intensity_range=(0, 255),
                 image_key: str = 'images',
                 target_key: str = 'targets',
                 bounding_box_key: str = "bounding_boxes",
                 batches_before_aug: Optional[int] = None,
                 name: str = 'random_brightness',
                 **kwargs):
        super().__init__(active=active,
                         image_key=image_key,
                         target_key=target_key,
                         bounding_box_key=bounding_box_key,
                         batches_before_aug=batches_before_aug,
                         name=name,
                         **kwargs)

        # check if factor is outside the acceptable range
        check_input(factor, name='factor', input_range=(-1.0, 1.0))

        # convert the custom input factor to be within the tensorflow range
        factor = interpolate_to_tensorflow_value(
            custom_value=factor,
            # custom_datapoints=[-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
            custom_datapoints=[-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            tensorflow_datapoints=[-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])

        self.image_intensity_range = image_intensity_range
        self._set_factor(factor)
        self._set_value_range(image_intensity_range)

    def get_random_transformation(self, image=None, label=None, bounding_box=None):

        batch_size = tf.shape(image)[0]
        rgb_delta_shape = [batch_size, 1, 1, 1]

        random_rgb_delta = tf.random.uniform(
            shape=rgb_delta_shape,
            minval=self._factor[0],
            maxval=self._factor[1])

        random_rgb_delta = random_rgb_delta * (self._value_range[1] - self._value_range[0])
        random_rgb_delta = tf.cast(random_rgb_delta, dtype=image.dtype)
        return {"rgb_delta": random_rgb_delta}

    def augment_image(self, image, transformation):
        rgb_delta = transformation['rgb_delta']

        image = tf.math.add(image, rgb_delta)
        return tf.clip_by_value(t=image,
                                clip_value_min=self._value_range[0],
                                clip_value_max=self._value_range[1])

    def augment_label(self, label, transformation):
        return label

    def _set_value_range(self, value_range):
        if not isinstance(value_range, (tuple, list)):
            raise ValueError(
                self._VALUE_RANGE_VALIDATION_ERROR + f"Got {value_range}")

        if len(value_range) != 2:
            raise ValueError(
                self._VALUE_RANGE_VALIDATION_ERROR + f"Got {value_range}")

        self._value_range = sorted(value_range)

    def _set_factor(self, factor):

        if isinstance(factor, (tuple, list)):
            if len(factor) != 2:
                raise ValueError(
                    self._FACTOR_VALIDATION_ERROR + f"Got {factor}"
                )
            self._check_factor_range(factor[0])
            self._check_factor_range(factor[1])
            self._factor = sorted(factor)
        elif isinstance(factor, (int, float)):
            self._check_factor_range(factor)
            factor = abs(factor)
            self._factor = [-factor, factor]
        else:
            raise ValueError(self._FACTOR_VALIDATION_ERROR + f"Got {factor}")

    def _check_factor_range(self, input_number):
        if input_number > 1.0 or input_number < -1.0:
            raise ValueError(
                self._FACTOR_VALIDATION_ERROR + f"Got {input_number}"
            )

    def get_config(self):
        config = super().get_config()
        config.update({"factor": self._factor,
                       "image_intensity_range": self.image_intensity_range,
                       })
        return config


@keras.saving.register_keras_serializable(package='Augmentation', name='RandomContrast')
class RandomContrast(AugmentationLayer):
    """
    A preprocessing layer that randomly adjusts the contrast of images during training.

    This layer performs contrast adjustment independently per channel. For each channel,
    it computes the mean pixel value and adjusts every pixel using the formula:
        `(x - mean) * contrast_factor + mean`.

    Contrast adjustment helps the model generalize to variations in lighting and exposure.
    At inference time, the layer is a no-op and passes inputs unchanged.

    Inputs:
        A dictionary containing:
            - 'images': 3D (unbatched) or 4D (batched) tensor in `"channels_last"` format
            - 'targets': 3D (unbatched), 4D (batched), 2D, or 1D tensor

    Outputs:
        A dictionary with the same structure:
            - 'images': Tensor with adjusted contrast, clipped to `image_intensity_range`
            - 'targets': Unchanged

    Example:
        >>> layer = RandomContrast(factor=(0.2, 0.8), image_intensity_range=(0.0, 255.0))
        >>> img = tf.random.uniform((1, 128, 128, 3))
        >>> mask = tf.random.uniform((1, 128, 128, 1))
        >>> result = layer({'images': img, 'targets': mask}, training=True)
        >>> result['images'].shape
        TensorShape([1, 128, 128, 3])

    Arguments:
        :param factor: a positive float represented as fraction of value, or a tuple of
            size 2 representing lower and upper bound.
            When represented as a single float, lower = upper.
            The contrast factor will be randomly picked between
            `[1.0 - lower, 1.0 + upper]`. For any pixel x in the channel,
            the output will be `(x - mean) * factor + mean`
            where `mean` is the mean value of the channel.
        :param active: If False, the layer performs no augmentation and passes inputs unchanged.
        :param seed: Optional random seed for reproducibility.
        :param image_intensity_range: Tuple or list of two floats, specifying the minimum and maximum pixel intensity.
            Example: `[0.0, 1.0]` for normalized inputs or `[0.0, 255.0]` for raw RGB inputs.
        :param image_key: Dictionary key for image tensors. Default is `'images'`.
        :param target_key: Dictionary key for target tensors. Default is `'targets'`.
        :param bounding_box_key: Dictionary key for bounding boxes. Default is `'bounding_boxes'`.
        :param name: Name of the layer.
        :param kwargs: Additional keyword arguments for the base layer.

    Raises:
        ValueError: If `factor` is outside the allowed range `[0.1, 1.0]`.
        TypeError: If `factor` is not a float, list, or tuple.
    """

    _VALUE_RANGE_VALIDATION_ERROR = (
        "The `value_range` argument should be a tuple of two numbers. ")

    def __init__(self,
                 factor: Union[float, List[float], Tuple[float, float]]=(0.1, 1.0),
                 active: bool = True,
                 image_intensity_range= (0, 255),
                 image_key: str = 'images',
                 target_key: str = 'targets',
                 bounding_box_key: str = "bounding_boxes",
                 batches_before_aug: Optional[int] = None,
                 name='random_contrast',
                 **kwargs):
        super().__init__(active=active,
                         image_key=image_key,
                         target_key=target_key,
                         bounding_box_key=bounding_box_key,
                         batches_before_aug=batches_before_aug,
                         name=name,
                         **kwargs)

        self.check_parameter_type({
            'factor': (factor, (int, tuple), False),
            'image_intensity_range': (image_intensity_range, tuple, False),
            'active': (active, bool, False),
            'name': (name, str, False),
            'image_key': (image_key, str, False),
            'target_key': (target_key, str, False),
            'bounding_box_key': (bounding_box_key, str, False),
            'batches_before_aug': (batches_before_aug, int, True),
        })

        self.factor = factor
        if isinstance(factor, (tuple, list)):
            self.lower = factor[0]
            self.upper = factor[1]
        else:
            self.lower = self.upper = factor

        if self.lower < 0.0 or self.upper < 0.0 or self.lower > 1.0:
            raise ValueError(
                "`factor` argument cannot have negative values or values "
                "greater than 1."
                f"Received: factor={factor}"
            )

        if not isinstance(image_intensity_range, (tuple, list)):
            raise ValueError(
                self._VALUE_RANGE_VALIDATION_ERROR + f"Got {image_intensity_range}")

        if len(image_intensity_range) != 2:
            raise ValueError(
                self._VALUE_RANGE_VALIDATION_ERROR + f"Got {image_intensity_range}")

        self.clip_value_range = sorted(image_intensity_range)

    def get_random_transformation(self, image=None, label=None, bounding_box=None):
        batch_size = tf.shape(image)[0]
        adjust_contrast = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1.0) < 0.5
        contrast_factor = tf.random.uniform(shape=[batch_size, 1, 1, 1],
                                            minval=1.0 - self.lower,
                                            maxval=1.0 + self.upper)

        return {'adjust_contrast': adjust_contrast,
                'contrast_factor': contrast_factor}

    def augment_image(self, image, transformation):
        adjust_contrast = transformation['adjust_contrast']
        contrast_factor = transformation['contrast_factor']

        mean = tf.reduce_mean(image, axis=[1, 2], keepdims=True)
        contrasted = (image - mean) * contrast_factor + mean
        contrasted = tf.clip_by_value(contrasted,
                                      clip_value_min=self.clip_value_range[0],
                                      clip_value_max=self.clip_value_range[1])

        return tf.where(condition=adjust_contrast, x=contrasted, y=image)


    def augment_label(self, label, transformation):
        return label

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"factor": self.factor,
                       "image_intensity_range": self.clip_value_range,
                       })
        return config


@keras.saving.register_keras_serializable(package='Augmentation', name='RandomCrop')
class RandomCrop(AugmentationLayer):
    """
    A preprocessing layer that randomly crops images and segmentation masks during training.

    This layer selects a random region of the input image and crops it to a specified zoom level.
    If used with segmentation masks, the masks will be cropped using the same region as the image.

    During inference or if the layer is inactive, it acts as a no-op (i.e., passes data through unchanged).

    Inputs:
        A dictionary containing:
            - 'images': 3D (unbatched) or 4D (batched) tensor in `"channels_last"` format
            - 'targets': 3D, 4D, 2D, or 1D tensor

    Outputs:
        A dictionary with:
            - 'images': Cropped image tensor
            - 'targets': Cropped target tensor, or unchanged if shape is not compatible

        All outputs maintain `"channels_last"` format.

    Fill modes (used for padding if needed):
        - *reflect*: `(d c b a | a b c d | d c b a)` The input is extended by reflecting about the edge of the last pixel.
        - *constant*: `(k k k k | a b c d | k k k k)` The input is extended by filling all values beyond the edge with the same constant value k = 0.
        - *wrap*: `(a b c d | a b c d | a b c d)` The input is extended by wrapping around to the opposite edge.
        - *nearest*: `(a a a a | a b c d | d d d d)` The input is extended by the nearest pixel.

    Example:
        >>> layer = RandomCrop(zoom_factor=(20, 50))
        >>> img = tf.random.uniform((1, 224, 224, 3))
        >>> mask = tf.random.uniform((1, 224, 224, 1))
        >>> result = layer({'images': img, 'targets': mask}, training=True)
        >>> result['images'].shape
        TensorShape([1, target_height, target_width, 3])

    Arguments:
        :param zoom_factor: Float or tuple/list of two floats. Must be in range [1, 100].
            Represents the percentage (relative to original size) to which the image will be cropped.
            For example, a zoom_factor of (20, 50) will crop a region sized between 20% and 50% of the original.
        :param active: Boolean flag to indicate whether the layer is active during training. Defaults to True.
        :param image_key: String key to access the image tensor from the input dictionary. Default is `'images'`.
        :param target_key: String key to access the target (e.g., mask) tensor from the input dictionary. Default is `'targets'`.
        :param name: Name of the layer.
        :param kwargs: Additional keyword arguments passed to the base `AugmentationLayer`.

    Raises:
        ValueError: If zoom_factor has invalid structure or values outside the allowed range.
        TypeError: If zoom_factor is not a float, int, list, or tuple.
    """

    def __init__(self,
                 zoom_factor: Union[float, List[float], Tuple[float, float]] = (20, 50),
                 active: bool = True,
                 image_key: str = 'images',
                 target_key: str = 'targets',
                 batches_before_aug: Optional[int] = None,
                 name='random_crop',
                 **kwargs):

        super().__init__(name=name,
                         image_key=image_key,
                         target_key=target_key,
                         active=active,
                         batches_before_aug=batches_before_aug,
                         **kwargs
                         )
        self.zoom_factor = zoom_factor

        check_input(inputs=zoom_factor, name='zoom factor', input_range=(1, 100))

        if isinstance(zoom_factor, (int, float)):
            self.min_factor = self.max_factor = zoom_factor
        elif isinstance(zoom_factor, (list, tuple)):
            if len(zoom_factor) != 2:
                raise ValueError(
                    f"The {type(zoom_factor).__name__} `zoom_factor` must have only two values, got {zoom_factor}")

            self.min_factor = zoom_factor[0]
            self.max_factor = zoom_factor[1]
        else:
            raise TypeError(
                f"The zoom_factor must be a list, tuple, int or float. Got {zoom_factor} of type `{type(zoom_factor).__name__}`"
            )

        if self.max_factor < self.min_factor:
            raise ValueError(
                "The upper boundary cannot be less than the lower boundary. Got angle = {}".format(zoom_factor)
            )
        if self.min_factor < 0:
            raise ValueError(f"Zoom factors must be positive values. Got {zoom_factor}")

    def _compute_crop_percent(self):
        return tf.random.uniform(shape=[],
                                 minval=self.min_factor / 100.0,
                                 maxval=self.max_factor / 100.0,
                                 dtype=tf.float32)

    def get_random_transformation(self, image=None, label=None, bounding_box=None):
        batch_shape = tf.shape(image)
        original_height = batch_shape[H_AXIS]
        original_width = batch_shape[W_AXIS]

        crop_percent = self._compute_crop_percent()
        target_height = tf.cast(tf.cast(original_height, tf.float32) * crop_percent, tf.int32)
        target_width = tf.cast(tf.cast(original_width, tf.float32) * crop_percent, tf.int32)

        # compute the offset height and width
        dtype = image.dtype
        rands = self._random_generator.random_uniform(shape=[2],
                                                      minval=0,
                                                      maxval=tf.int32.max,
                                                      dtype=tf.int32)
        h_diff = original_height - target_height
        w_diff = original_width - target_width

        offset_height = rands[0] % tf.maximum(h_diff + 1, 1)
        offset_width = rands[1] % tf.maximum(w_diff + 1, 1)
        return {
            'offset_height': offset_height,
            'offset_width': offset_width,
            'target_height': target_height,
            'target_width': target_width,
            'original_height': original_height,
            'original_width': original_width,
        }

    def augment_image(self, image, transformation):
        cropped = tf.image.crop_to_bounding_box(
            image=image,
            offset_height=transformation['offset_height'],
            offset_width=transformation['offset_width'],
            target_height=transformation['target_height'],
            target_width=transformation['target_width']
        )

        resized = tf.image.resize(
            images=cropped,
            size=[transformation['original_height'], transformation['original_width']],
            method=tf.image.ResizeMethod.BILINEAR
        )
        return resized

    def augment_label(self, label, transformation):
        cropped = tf.image.crop_to_bounding_box(
            image=label,
            offset_height=transformation['offset_height'],
            offset_width=transformation['offset_width'],
            target_height=transformation['target_height'],
            target_width=transformation['target_width']
        )

        resized = tf.image.resize(
            images=cropped,
            size=[transformation['original_height'], transformation['original_width']],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        return resized

    def augment_bounding_boxes(self, image, bounding_boxes, transformation=None):
        """
        Adjust bounding boxes to reflect cropping and resizing.

        :param image: Tensor of shape (B, H, W, C)
        :param bounding_boxes: Tensor of shape (B, N, 4) with [x_min, y_min, x_max, y_max]
        :param transformation: Dict with keys:
            - offset_height, offset_width
            - target_height, target_width
            - original_height, original_width
        :return: Adjusted bounding_boxes of shape (B, N, 4)
        """
        x_min, y_min, x_max, y_max = tf.split(bounding_boxes, 4, axis=-1)

        # Cast and expand transformation tensors
        offset_x = tf.cast(tf.expand_dims(transformation['offset_width'], axis=-1), bounding_boxes.dtype)  # (B, 1)
        offset_y = tf.cast(tf.expand_dims(transformation['offset_height'], axis=-1), bounding_boxes.dtype)
        crop_w = tf.cast(tf.expand_dims(transformation['target_width'], axis=-1), bounding_boxes.dtype)
        crop_h = tf.cast(tf.expand_dims(transformation['target_height'], axis=-1), bounding_boxes.dtype)
        orig_w = tf.cast(tf.expand_dims(transformation['original_width'], axis=-1), bounding_boxes.dtype)
        orig_h = tf.cast(tf.expand_dims(transformation['original_height'], axis=-1), bounding_boxes.dtype)

        # Adjust coordinates: shift by crop offset
        x_min = (x_min - offset_x) / crop_w * orig_w
        x_max = (x_max - offset_x) / crop_w * orig_w
        y_min = (y_min - offset_y) / crop_h * orig_h
        y_max = (y_max - offset_y) / crop_h * orig_h

        # Recombine
        return tf.concat([x_min, y_min, x_max, y_max], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({'zoom_factor': self.zoom_factor})
        return config


@keras.saving.register_keras_serializable(package='Augmentation', name='CenterCrop')
class CenterCrop(AugmentationLayer):
    """
    A preprocessing layer that crops the central region of input images and their
    corresponding targets (e.g., segmentation masks) to a specified size.

    During training, this layer crops the center of the input tensors to a fixed
    `(height, width)` shape. If the input is smaller than the target size, it will
    be resized first to preserve the aspect ratio before cropping. This ensures a
    consistent output shape across varying input sizes.

    Inputs:
        A dictionary with any of the following keys:
        - `'images'`: A 3D (H, W, C) or 4D (B, H, W, C) float/integer tensor. Pixel values can be in any range (e.g., `[0., 1.]`, `[0, 255]`).
        - `'targets'`: A 3D or 4D tensor. If the target is not at least 3D, it is passed through unchanged.
        - `'bounding_boxes'` (optional): A 2D or 3D tensor specifying bounding boxes.

    Outputs:
        A dictionary with the same keys:
        - `'images'` and `'targets'` are center-cropped or resized to `(height, width)`.
        - `'bounding_boxes'` are optionally adjusted if present.
        - Non-augmentable targets (e.g., 1D labels) are returned unchanged.

    Example:
        >>> center_crop = CenterCrop(height=256, width=256, active=True)

        >>> inputs = {
        >>>     "images": tf.random.uniform(shape=(512, 512, 3)),
        >>>     "targets": tf.random.uniform(shape=(512, 512, 1), maxval=2, dtype=tf.int32),
        >>> }

        >>> outputs = center_crop(inputs, training=True)

        >>> print(outputs["images"].shape)   # (256, 256, 3)
        >>> print(outputs["targets"].shape)  # (256, 256, 1)


    Notes:
        - The layer supports both batched and unbatched inputs.
        - If the input and target height/width differ in parity (even vs. odd), symmetric padding may occur.
        - Non-3D/4D targets (e.g., class indices) are passed through without cropping.
        - Uses `tf.image.crop_to_bounding_box` and `smart_resize` to ensure robust handling of varied input sizes.

    :param height: `int`. Target height after cropping. Must be a positive integer.
    :param width: `int`. Target width after cropping. Must be a positive integer.
    :param active: `bool`. Whether to apply center cropping during training. Defaults to `False`.
    :param name: `str`. Name of the layer. Defaults to `'center_crop'`.
    :param image_key: `str`. Dictionary key for accessing the image tensor. Defaults to `'images'`.
    :param target_key: `str`. Dictionary key for accessing the target tensor. Defaults to `'targets'`.
    :param bounding_box_key: `str`. Dictionary key for accessing bounding boxes. Defaults to `'bounding_boxes'`.
    """

    def __init__(self,
                 height: int = None,
                 width: int = None,
                 active: bool = True,
                 image_key: str = 'images',
                 target_key: str = 'targets',
                 bounding_box_key: str = "bounding_boxes",
                 batches_before_aug: Optional[int] = None,
                 name: str = 'center_crop',
                 **kwargs):

        super().__init__(name=name,
                         active=active,
                         image_key=image_key,
                         target_key=target_key,
                         bounding_box_key=bounding_box_key,
                         batches_before_aug=batches_before_aug,
                         **kwargs)

        if height is None or width is None:
            self.active.assign(False)

        self.check_parameter_type(params_and_types={
            'height': (height, int, True),
            'width': (width, int, True),
            'active': (active, bool, False),
            'name': (name, str, False),
            'image_key': (image_key, str, False),
            'target_key': (target_key, str, False),
            'bounding_box_key': (bounding_box_key, str, False),
        })

        if all(value is not None for value in [height, width]):
            if all(isinstance(value, int) for value in [height, width]):
                self.width = width
                self.height = height
            else:
                raise TypeError(
                    "The `height` and `width` parameter for the center crop layer "
                    "must be of type int. Got height="
                    "{} and width={} of type {} and {}".format(height, width, type(height), type(width))
                )

    def _augment(self, inputs):
        """Augments the image, target and bounding box."""
        image = inputs.get(self.image_key, None)
        target = inputs.get(self.target_key, None)
        bounding_box = inputs.get(self.bounding_box_key, None)

        original_image_shape = image.shape
        # The transform op only accepts rank 4 inputs,
        # so if we have an unbatched image,
        # we need to temporarily expand dims to a batch.
        image, image_was_unbatched = self.ensure_batched(image)

        transformation = self.get_random_transformation(image=image, label=target,
                                                        bounding_box=bounding_box)

        image = self.augment_image(image, transformation=transformation)

        if image_was_unbatched:
            image = tf.squeeze(image, 0)
            image.set_shape((self.height, self.width, original_image_shape[-1]))

        result = {self.image_key: image}

        # apply target augmentation only if target exist, and it is at least 3-dimensional
        if target is not None:
            original_target_shape = target.shape
            target, target_was_unbatched = self.ensure_batched(target)

            if self.is_batched(target):
                target = self.augment_target(target, transformation=transformation)
                if target_was_unbatched:
                    target = tf.squeeze(input=target, axis=0)
                    target.set_shape((self.height, self.width, original_target_shape[-1]))

                target = tf.round(target)
                result[self.target_key] = tf.cast(target, dtype=self.target_compute_dtype)
            elif target.shape.rank > 2 and target.shape[-1] > 1:
                raise ValueError(f"The shape of the target(s) must be in the form "
                                 f"[Height, Width, 1], got {target.shape}.")
            else:
                # incase the target is not a 3-dimensional tensor(e.g. a categorical target)
                result[self.target_key] = target

        if bounding_box is not None:
            bounding_box = self.augment_bounding_boxes(image, bounding_box,
                                                       transformation=transformation)
            result[self.bounding_box_key] = bounding_box

        return result

    def augment_image(self, image, transformation):
        image = utils.ensure_tensor(image, self.compute_dtype)
        input_shape = tf.shape(image)
        h_diff = input_shape[H_AXIS] - self.height
        w_diff = input_shape[W_AXIS] - self.width

        def center_crop():
            h_start = tf.cast(h_diff / 2, tf.int32)
            w_start = tf.cast(w_diff / 2, tf.int32)
            cropped = tf.image.crop_to_bounding_box(
                image, h_start, w_start, self.height, self.width
            )
            return tf.cast(cropped, self.compute_dtype)

        def upsize():
            resized = image_utils.smart_resize(
                image, [self.height, self.width]
            )
            # smart_resize will always output float32, so we need to re-cast.
            return tf.cast(resized, self.compute_dtype)

        return tf.cond(
            tf.reduce_all((h_diff >= 0, w_diff >= 0)), center_crop, upsize
        )

    def augment_label(self, label, transformation):
        label = utils.ensure_tensor(label, self.compute_dtype)
        label = tf.cast(label, dtype=tf.float32)

        input_shape = tf.shape(label)
        h_diff = input_shape[H_AXIS] - self.height
        w_diff = input_shape[W_AXIS] - self.width

        def center_crop():
            h_start = tf.cast(h_diff / 2, tf.int32)
            w_start = tf.cast(w_diff / 2, tf.int32)
            cropped = tf.image.crop_to_bounding_box(
                label, h_start, w_start, self.height, self.width
            )
            return cropped

        def upsize():
            resized = image_utils.smart_resize(
                x=label,
                size=[self.height, self.width],
                interpolation='nearest'
            )
            return resized

        return tf.cond(
            pred=tf.reduce_all((h_diff >= 0, w_diff >= 0)),
            true_fn=center_crop,
            false_fn=upsize
        )

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        input_shape[H_AXIS] = self.height
        input_shape[W_AXIS] = self.width
        return tf.TensorShape(input_shape)

    def get_config(self):
        config = {
            "height": self.height,
            "width": self.width,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@keras.saving.register_keras_serializable(package='Augmentation', name='RandomAugmentationLayer')
class RandomAugmentationLayer(Layer):
    """
    A comprehensive GPU-friendly preprocessing layer that applies multiple random augmentations
    to images and their corresponding targets (e.g., segmentation masks). This is designed for
    tasks like semantic segmentation where identical spatial transformations must be applied
    to both image and label.

    Notes:

    The `targets` must be either 3D or 4D tensors (i.e., shape `[H, W, C]` or `[B, H, W, C]`)
    to undergo spatial augmentations. If `targets` are scalar (e.g., shape `[B]`), they will
    be passed through unchanged.

    Supported Transformations:
        - Horizontal Flip
        - Vertical Flip
        - Rotation
        - Shear
        - Zoom
        - Gaussian Noise
        - Brightness Adjustment
        - Contrast Adjustment

    Each augmentation can be selectively enabled via constructor arguments.

    Example Usage:
            >>> augmenter = RandomAugmentationLayer(
            >>>     image_key = 'images',
            >>>     target_key= 'targets',
            >>>     bounding_box_key= 'bounding_boxes',
            >>>     flip=True,
            >>>     rotate=True,
            >>>     theta=30,
            >>>     shear=True,
            >>>     shear_angle=(0, 5),
            >>>     zoom=True,
            >>>     zoom_factor=(90, 110),
            >>>     use_gaussian_noise=True,
            >>>     gaussian_range=(0.0, 0.02),
            >>>     adjust_brightness=True,
            >>>     brightness_range=(0.9, 1.1),
            >>>     adjust_contrast=True,
            >>>     contrast_range=(0.9, 1.1),
            >>>     apply_jitter=True,
            >>>     jitter_intensity=(1, 5),
            >>>     image_intensity_range=(0.0, 255.0)
            >>> )

            >>> output = augmenter({'images': image_batch, 'targets': mask_batch}, training=True)

    :param image_key: Key in the input dictionary corresponding to the image tensor. Default `'images'`.
    :param target_key: Key for the target/mask tensor. Default `'targets'`.
    :param bounding_box_key: Key for the optional bounding box tensor. Default `'bounding_boxes'`.

    :param flip: If `True`, applies random horizontal flips with 50% probability.

    :param flip_mode: (str): Flip direction. Options:
            - 'HORIZONTAL': left-right flip
            - 'VERTICAL': up-down flip
            - 'HORIZONTAL_AND_VERTICAL': both directions randomly

    :param rotate: Enables random image rotation.
    :param theta: (int or Tuple[int, int]): Rotation angle in degrees. Can be:
                  - A single number → rotation in `[-theta, +theta]`
                  - A tuple/list → rotation in `[min_angle, max_angle]`. For instance, (-40, 45).
                  Valid range: [-180°, 180°]

    :param shear: Enables shear transformation.
    :param shear_angle: (float or Tuple[float, float]): Shear angles in degrees. eg.: (-10.0, 10.0).
        The valid range is (-45.0, 45).

    :param zoom: Enables random zoom.
    :param zoom_factor: A float, int, tuple, or list specifying the zoom scale in percentage.
         - Range: -200 to 200 (percent of original size)
         - If scalar `x`: interpreted as (-x, +x), applied as percent deviation from original. For instance if zoom_factor is set to 20, it means zoom into the image by a random percentage between -20% and +20%. -30 mean zoom-out of the image by 30% of its original size.
         - If tuple/list: sampled from [zoom_factor[0], zoom_factor[1]]. E.g. [20 - 40] means, zoom in by a random percentage between 20%  to 40%.

    :param use_gaussian_noise: If `True`, adds Gaussian noise to image only.
    :param gaussian_range: A float, list, or tuple representing the lower and upper bounds of the
          standard deviation to use for sampling Gaussian noise. If a single float is provided,
          it is treated as the upper bound with the lower bound defaulting to 0.0. Both values
          must be ≥ 0.

    :param adjust_brightness: Enables brightness changes.
    :param brightness_range: A float, or tuple/list of two floats between -1.0 and 1.0. The
            factor is used to determine the lower bound and upper bound of the
            brightness adjustment.  A float will be randomly chosen between the limits.
                - Values above 0 brighten the image; values below 0 darken it.
                - When only one float is provided, eg, 0.2, then -0.2 will be used for lower bound
                  and 0.2 will be used for upper bound.
                - If a tuple/list is provided, the factor is sampled from `[factor[0], factor[1]]`.

    :param adjust_contrast: Enables contrast adjustment.
    :param contrast_range: a positive float represented as fraction of value, or a tuple of
            size 2 representing lower and upper bound.
            When represented as a single float, lower = upper.
            The contrast factor will be randomly picked between
            `[1.0 - lower, 1.0 + upper]`. For any pixel x in the channel,
            the output will be `(x - mean) * factor + mean`
            where `mean` is the mean value of the channel.

    :param apply_jitter: Whether to apply spatial jitter (random spatial shift of the image and mask).
    :param jitter_intensity: Tuple of two integers specifying the minimum and maximum pixel offset
                             for spatial jitter. A random shift is applied independently along height
                             and width in the range [min, max].
                             - Allowed range: (0, 60)
                             - Example: (5, 15) allows shifts up to ±15 pixels in both dimensions.


    :param center_crop: If `True`, crops the center of the final image.
    :param output_height: Output height in pixels. Required if `center_crop=True`.
    :param output_width: Output width in pixels. Required if `center_crop=True`.

    :param fill_mode: Padding method for affine transforms. One of:
                      - `"constant"`: Fill with `fill_value`
                      - `"reflect"`: Mirror edges
                      - `"wrap"`: Wrap around
                      - `"nearest"`: Use nearest pixel

    :param interpolation: Image resampling method: `"nearest"` or `"bilinear"`. Segmentation targets always use `"nearest"`.

    :param fill_value: Float used if `fill_mode="constant"`. Default: `0.0`.

    :param image_intensity_range: Min/max pixel values of input. Common values:
                                  - `(0.0, 1.0)` for normalized
                                  - `(0.0, 255.0)` for raw inputs

    :param name: Name of the layer instance.
    :param kwargs: Additional layer keyword arguments.
    :returns: Dictonary with keys: image_key, target_key and optionally sample_weight.
    """

    def __init__(self,
                 image_key: str = 'images',
                 target_key: str = 'targets',
                 bounding_box_key: str = 'bounding_boxes',
                 batches_before_aug: Optional[int] = None,
                 flip: bool = False,
                 flip_mode: str = 'HORIZONTAL_AND_VERTICAL',
                 rotate: bool = False,
                 theta: Union[int, List[int], Tuple[int, int], None] = 180,
                 shear: bool = False,
                 shear_angle: Union[list, tuple, float, None] = (0, 20),
                 zoom: bool = False,
                 zoom_factor: Union[int, list, tuple, None] = (80, 110),
                 use_gaussian_noise: bool = False,
                 gaussian_range: Union[float, List[float], Tuple[float, float]] = (0.0, 0.01),
                 adjust_brightness: bool = False,
                 brightness_range: Union[float, list, tuple] = (0.90, 1.0),
                 adjust_contrast: bool = False,
                 contrast_range: Union[float, List[float], Tuple[float, float]]=(0.1, 1.0),
                 jitter: bool = False,
                 jitter_range: Union[List[int], Tuple[int, int]] = (0, 60),
                 fill_mode: str = "reflect",
                 interpolation: str = "bilinear",
                 image_intensity_range: Tuple[float, float] = (0.0, 255.0),
                 fill_value: float = 0.0,
                 name='multi_dual_augmentation_layer',
                 **kwargs):

        super().__init__(name=name, trainable=False, **kwargs)
        self.image_key = image_key
        self.target_key = target_key
        self.bounding_box_key = bounding_box_key
        self.batches_before_aug = batches_before_aug

        self.value_range = image_intensity_range
        self.flip = flip
        self.flip_mode = flip_mode
        self.rotate = rotate
        self.theta = theta
        self.shear = shear
        self.shear_angle = shear_angle
        self.zoom = zoom
        self.zoom_factor = zoom_factor
        self.use_gaussian_noise = use_gaussian_noise
        self.gaussian_range = gaussian_range
        self.adjust_brightness = adjust_brightness
        self.brightness_range = brightness_range
        self.adjust_contrast = adjust_contrast
        self.contrast_range = contrast_range
        self.apply_jitter = jitter
        self.jitter_range = jitter_range
        self.fill_mode = fill_mode
        self.interpolation = interpolation
        self.fill_value = fill_value


        self.flip_layer = RandomFlip(active=self.flip,
                                     mode=self.flip_mode,
                                     image_key = self.image_key,
                                     target_key= self.target_key,
                                     bounding_box_key= self.bounding_box_key,
                                     batches_before_aug=self.batches_before_aug,
                                     name='horizontal_flip')

        self.affine_layer = AffineTransform(rotate=self.rotate,
                                            theta=theta,
                                            shear=self.shear,
                                            shear_angle=shear_angle,
                                            zoom=self.zoom,
                                            zoom_factor=zoom_factor,
                                            interpolation=interpolation,
                                            fill_mode=fill_mode,
                                            fill_value=fill_value,
                                            image_key=self.image_key,
                                            target_key=self.target_key,
                                            bounding_box_key=self.bounding_box_key,
                                            batches_before_aug=self.batches_before_aug,
                                            name='affine_transform')

        self.noise_layer = RandomGaussianNoise(active=self.use_gaussian_noise,
                                               stddev=gaussian_range,
                                               image_key=self.image_key,
                                               target_key=self.target_key,
                                               bounding_box_key=self.bounding_box_key,
                                               batches_before_aug=self.batches_before_aug,
                                               name='gaussian_noise')

        self.brightness_layer = RandomBrightness(active=self.adjust_brightness,
                                                 factor=brightness_range,
                                                 image_intensity_range=image_intensity_range,
                                                 image_key=self.image_key,
                                                 target_key=self.target_key,
                                                 bounding_box_key=self.bounding_box_key,
                                                 batches_before_aug=self.batches_before_aug,
                                                 name='brightness')

        self.contrast_layer = RandomContrast(active=self.adjust_contrast,
                                             factor=contrast_range,
                                             image_key=self.image_key,
                                             target_key=self.target_key,
                                             bounding_box_key=self.bounding_box_key,
                                             batches_before_aug=self.batches_before_aug,
                                             name='contrast')

        self.jitter_layer = RandomJitter(active=self.apply_jitter,
                                         jitter_range=self.jitter_range,
                                         image_key=self.image_key,
                                         target_key=self.target_key,
                                         bounding_box_key=self.bounding_box_key,
                                         batches_before_aug=self.batches_before_aug,
                                         name='jitter')


        # keep track of all layers that were activated.
        self.register = []

        if flip:
            self.register.append(self.flip_layer)
        if any([shear, zoom, rotate]):
            self.register.append(self.affine_layer)
        if use_gaussian_noise:
            self.register.append(self.noise_layer)
        if adjust_brightness:
            self.register.append(self.brightness_layer)
        if adjust_contrast:
            self.register.append(self.contrast_layer)
        if jitter:
            self.register.append(self.jitter_layer)

    def call(self, inputs, training=None):
        """
        :param inputs: a tuple containing the images and targets
        :return: a dictionary of {'images': inputs[0], 'targets': inputs[1]}
        """
        x = self.flip_layer(inputs, training=training)
        x = self.affine_layer(x, training=training)
        x = self.contrast_layer(x, training=training)
        x = self.noise_layer(x, training=training)
        x = self.jitter_layer(x, training=training)
        x = self.brightness_layer(x, training=training)
        return x

    def reset_counter(self):
        self.flip_layer.reset_counter()
        self.affine_layer.reset_counter()
        self.noise_layer.reset_counter()
        self.jitter_layer.reset_counter()
        self.brightness_layer.reset_counter()

    def get_config(self):
        base_config = super().get_config()
        config = {"image_key":self.image_key,
                  "target_key": self.target_key,
                  "bounding_box_key": self.bounding_box_key,
                  "batches_before_augmentation": self.batches_before_aug,
                  "flip": self.flip,
                  "flip_mode": self.flip_mode,
                  "rotate": self.rotate,
                  "theta": self.theta,
                  "shear": self.shear,
                  "shear_angle": self.shear_angle,
                  "zoom": self.zoom,
                  "zoom_factor": self.zoom_factor,
                  "use_gaussian_noise": self.use_gaussian_noise,
                  "gaussian_range": self.gaussian_range,
                  "adjust_brightness": self.adjust_brightness,
                  "brightness_range": self.brightness_range,
                  "jitter": self.apply_jitter,
                  "jitter_range": self.jitter_range,
                  "fill_mode": self.fill_mode,
                  "interpolation": self.interpolation,
                  'image_intensity_range': self.value_range,
                  "fill_value": self.fill_value}
        base_config.update(config)
        return base_config

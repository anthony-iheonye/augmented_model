from typing import Dict
from typing import Optional

import keras
import tensorflow as tf
from keras.src.layers.preprocessing import preprocessing_utils as utils
from tensorflow.keras.layers import Layer

from layers.augmentation.image_target_processing import AugmentationLayer


@keras.saving.register_keras_serializable(package='Augmentation', name='SplitImageTarget')
class SplitImageTarget(tf.keras.layers.Layer):
    """
    A utility Keras layer that extracts and returns image and target tensors from a dictionary input.

    This layer is useful in preprocessing pipelines where inputs are passed as dictionaries
    (e.g., from tf.data or custom augmentation layers), and models expect separate (image, target) tuples.

    Example:
        input = {'images': <image_tensor>, 'targets': <mask_tensor>}
        output = (<image_tensor>, <mask_tensor>)

    :param image_key: str, the key used to retrieve the image tensor from the input dictionary.
    :param target_key: str, the key used to retrieve the target (e.g., mask) tensor from the input dictionary.
    :param name: str, name of the layer.
    :param kwargs: Additional keyword arguments passed to the base Layer class.
    """

    def __init__(self,
                 image_key: str = 'images',
                 target_key: str = 'targets',
                 name='splitImageMask',
                 **kwargs):
        super().__init__(name=name, trainable=False, **kwargs)
        self.image_key = image_key
        self.target_key = target_key

    def call(self, inputs, **kwargs):
        """
        Splits a dictionary input into a tuple of (image, target) tensors.

        :param inputs: dict, a dictionary containing image and target tensors.
                       Expected to have keys matching `self.image_key` and `self.target_key`.
        :return: Tuple (image_tensor, target_tensor)
        """
        image = inputs.get(self.image_key, None)
        target = inputs.get(self.target_key, None)
        return image, target

    def get_config(self):
        config = super().get_config()
        config.update({'image_key': self.image_key,
                       'target_key': self.target_key})
        return config


@keras.saving.register_keras_serializable(package='Augmentation', name='SplitImageTargetSampleWeight')
class SplitImageTargetSampleWeight(tf.keras.layers.Layer):
    """
    A utility Keras layer that extracts and returns image, target and sample weight tensors from a dictionary input.

    This layer is useful in preprocessing pipelines where inputs are passed as dictionaries
    (e.g., from tf.data or custom augmentation layers), and models expect separate (image, target) tuples.

    Example:
        input = {'images': <image_tensor>, 'targets': <mask_tensor>}
        output = (<image_tensor>, <mask_tensor>)

    :param image_key: str, the key used to retrieve the image tensor from the input dictionary.
    :param target_key: str, the key used to retrieve the target (e.g., mask) tensor from the input dictionary.
    :param sample_weight_key: str, the key used to retrieve the sample weight (e.g., 'sample_weight') tensor
        from the input dictionary.
    :param name: str, name of the layer.
    :param kwargs: Additional keyword arguments passed to the base Layer class.
    """

    def __init__(self,
                 image_key: str = 'images',
                 target_key: str = 'targets',
                 sample_weight_key: str = 'sample_weight',
                 name='splitImageMask',
                 **kwargs):
        super().__init__(name=name, trainable=False, **kwargs)
        self.image_key = image_key
        self.target_key = target_key
        self.sample_weight_key = sample_weight_key

    def call(self, inputs, **kwargs):
        """
        Splits a dictionary input into a tuple of (image, target) tensors.

        :param inputs: dict, a dictionary containing image and target tensors.
                       Expected to have keys matching `self.image_key` and `self.target_key`.
        :return: Tuple (image_tensor, target_tensor)
        """
        image = inputs.get(self.image_key, None)
        target = inputs.get(self.target_key, None)
        sample_weight = inputs.get(self.sample_weight_key, None)
        return image, target, sample_weight

    def get_config(self):
        config = super().get_config()
        config.update({'image_key': self.image_key,
                       'target_key': self.target_key,
                       'sample_weight_key': self.sample_weight_key})
        return config


@keras.saving.register_keras_serializable(package='Augmentation', name='PassThroughLayer')
class PassThroughLayer(Layer):
    """Passes"""

    def __init__(self, name='pass_through', **kwargs):
        super().__init__(name=name, **kwargs)

    def __repr__(self):
        return f"{self.name}"

    def call(self, inputs, *args, **kwargs):
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({'name': self.name})
        return config


@keras.saving.register_keras_serializable(package='Augmentation', name='NormalizeImage')
class NormalizeImage(AugmentationLayer):
    """
    A preprocessing layer that normalizes input images to a standardized float range.

    This layer scales image pixel values either to the range [0, 1] or [-1, 1] depending
    on the configuration. The normalization is only applied during training if `active=True`.

    Inputs:
        A dictionary with at least the key `'images'`, pointing to a 3D or 4D tensor
        representing the input image(s). The pixel values can be integers or floats.

    Outputs:
        The dictionary with the `'images'` tensor normalized.
        Other keys (e.g., `'targets'`) are returned unmodified.

    Example:
    >>> layer = NormalizeImage(scale_to_0_1=True)
    >>> inputs = {'images': tf.random.uniform((256, 256, 3), maxval=255, dtype=tf.int32)}
    >>> outputs = layer(inputs, training=True)
    """

    def __init__(self,
                 image_key: str = 'images',
                 active: bool = True,
                 scale_to_0_1: bool = True,
                 name: str = 'normalize_layer'):
        """
        Initializes the normalization layer.

        Args:
            active (bool): If False, the layer returns the inputs unchanged.
            scale_to_0_1 (bool): If True, normalize to [0, 1] using division by 255.
                                 If False, normalize to [-1, 1] using (x - 127.5) / 127.5.
            name (str): Name of the layer.
        """
        super().__init__(name=name, active=active, image_key=image_key)
        self.scale_to_0_1 = scale_to_0_1

        if self.scale_to_0_1:
            self.norm_fn = lambda image: tf.divide(x=tf.cast(image, tf.float32), y=255.0)
        else:
            self.norm_fn = lambda image: tf.divide(x=tf.cast(image, tf.float32) - 127.5, y=127.5)

    def __repr__(self):
        """Returns the name of the layer as its string representation."""
        return f"{self.name}"

    def call(self, inputs, *args, **kwargs):
        """
        Applies normalization during training if `active` is True.

        Args:
            inputs (dict): Dictionary containing at least the key `'images'`.
            training (bool, optional): Whether the layer is in training mode.

        Returns:
            dict: The inputs with the `'images'` tensor normalized.
        """
        inputs = self._ensure_inputs_are_compute_dtype(inputs)

        return tf.cond(
            pred=tf.convert_to_tensor(self.active),
            true_fn=lambda: self.norm_fn(inputs),
            false_fn=lambda: inputs)

    def check_inputs_type(self, inputs):
        """Ensures the input is a Python dictionary"""
        if not isinstance(inputs, (dict, tf.Tensor)):
            raise TypeError(f"The inputs must be a tf.Tensor or a dict, with "
                            f"key '{self.image_key}'. "
                            f"Got {inputs} of type {type(inputs).__name__}")

    def check_dict_keys(self, inputs):
        """Checks that the inputs have the correct dict keys"""
        if not self.image_key in inputs:
            raise KeyError(f"Invalid dictionary for inputs. "
                           f"Got keys: {list(inputs.keys())}"
                           f"The Images  must be assigned to the "
                           f"key '{self.image_key}'. ")

    def _ensure_inputs_are_compute_dtype(self, inputs):
        """Ensures the input is a Tensor, SparseTensor or RaggedTensor."""
        self.check_inputs_type(inputs)

        if isinstance(inputs, dict):
            self.check_dict_keys(inputs)
            image = utils.ensure_tensor(inputs=inputs[self.image_key],
                                        dtype=self.compute_dtype)
            return image
        else:
            image = utils.ensure_tensor(inputs=inputs, dtype=self.compute_dtype)
            return image

    def get_config(self):
        """
        Returns the config dictionary for serialization.

        Returns:
            dict: Configuration of the layer.
        """
        config = super().get_config()
        config.update({'scale_to_0_1': self.scale_to_0_1})
        return config


@keras.saving.register_keras_serializable(package='Augmentation', name='SampleWeight')
class SampleWeight(Layer):
    """
    A preprocessing layer that computes a per-pixel sample weight map based on inverse class frequency
    in a segmentation mask tensor. Useful for addressing class imbalance in semantic segmentation tasks.

    If `active=True` during training, the layer generates a weight map where each pixel’s weight is
    inversely proportional to the frequency of its class in the mask. The weights are normalized such that
    the total sum across the present classes equals 1. If either `active` = False, or the input shape is not (H, W, C) or (B, H, W, C),
    a tensor of ones with the same shape is returned.

    This layer is typically used by passing the resulting weight map as the `sample_weight` during model training.

    Example:
        >>> layer = SampleWeight(num_classes=4, active=True)
        >>> mask = tf.constant([0, 0, 0, 1, 2, 2, 3, 3, 3, 3], dtype=tf.int32, shape=(1, 1, 10, 1))
        >>> weights = layer(mask, training=True)

    Args:
        num_classes (int): Total number of distinct classes in the segmentation task.
        active (bool): Whether to compute weights during training. If False, the output is None.
        name (str): Name of the layer instance.
    """
    def __init__(self,
                 num_classes: int,
                 active: bool = False,
                 name: str = 'sample_weight',
                 **kwargs,
                 ):
        super().__init__(trainable=False, name=name, **kwargs)

        self.num_classes = num_classes
        self.sample_weight_key = 'sample_weight'
        self.active = tf.constant(active, dtype=tf.bool)

    def __repr__(self):
        return f"{self.name}"

    @staticmethod
    def check_inputs_type(mask_tensor):
        """
        Ensures the input is a Tensor.

        Args:
            mask_tensor (tf.Tensor): The input tensor representing segmentation masks.

        Raises:
            TypeError: If input is not a tf.Tensor.
        """
        if not isinstance(mask_tensor, tf.Tensor):
            raise TypeError(f"The `mask_tensor` must be a tf.Tensor. "
                            f"Got {mask_tensor} of type {type(mask_tensor).__name__}")

    def _ensure_inputs_are_compute_dtype(self, inputs):
        """
        Converts the input to the correct dtype and ensures it's a tensor.

        Args:
            inputs (tf.Tensor): A tensor representing segmentation masks.

        Returns:
            tf.Tensor: A compute-dtype tensor.
        """
        self.check_inputs_type(inputs)
        inputs = utils.ensure_tensor(inputs=inputs, dtype=self.compute_dtype)
        return inputs

    def call(self, inputs, training=None)-> Dict[str, Optional[tf.Tensor]]:
        inputs = self._ensure_inputs_are_compute_dtype(inputs)

        if training:
            def compute_weights():
                return self._compute_weights(inputs)

            def no_weights():
                return tf.ones_like(inputs, dtype=tf.float32)

            return tf.cond(
                pred=self.active,
                true_fn=compute_weights,
                false_fn=no_weights
            )
        else:
            return tf.ones_like(inputs, dtype=tf.float32)

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

    def _compute_weights(self, mask: tf.Tensor):
        """
        Computes per-pixel sample weights for a batch of segmentation masks.

        Args:
            mask (tf.Tensor): A 3D or 4D tensor representing one or more segmentation masks.

        Returns:
            tf.Tensor: A tensor of the same shape as `mask`, containing per-pixel weights.
        """


        original_target_shape = mask.shape
        temp_target, target_was_unbatched = self.ensure_batched(mask)

        if self.is_batched(temp_target):
            sample_weight = tf.map_fn(fn=self.compute_single_sample_weight_map,
                                      elems=temp_target,
                                      fn_output_signature=tf.TensorSpec(shape=temp_target.shape[1:],
                                                                        dtype=tf.float32))
            if target_was_unbatched:
                sample_weight = tf.squeeze(sample_weight, axis=0)
                sample_weight.set_shape(original_target_shape)

            return sample_weight
        elif mask.shape.rank > 2 and mask.shape[-1] > 1:
            raise ValueError(f"The shape of the target(s) must be in the form "
                             f"[Height, Width, 1], got {mask.shape}.")
        else:
            # Skip sample weight computation if it's a 1D/2D categorical target
            return tf.ones_like(input=mask, dtype=tf.float32)

    def compute_single_sample_weight_map(self, mask):
        """
        Computes a per-pixel weight map for a single segmentation mask using inverse class frequency.

        Args:
            mask (tf.Tensor): A 3D tensor [H, W, 1] or [H, W] of class labels.

        Returns:
            tf.Tensor: A tensor of shape [H, W, 1], where each pixel value corresponds to the normalized inverse frequency of its class.
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
        inv_freq = tf.where(condition=present_mask, x=1.0 / class_counts, y=tf.zeros_like(class_counts))

        # Normalize weights so that present class weights sum to 1.
        norm_weights = inv_freq / tf.reduce_sum(inv_freq)

        # Map each pixel label to its class weight
        mask = tf.cast(mask, tf.int32)
        weight_map = tf.gather(norm_weights, mask)
        return weight_map

    def get_config(self):
        config = super().get_config()
        config.update({
            'active': self.active,
            'sample_weight_key': self.sample_weight_key,
            'num_classes': self.num_classes
        })
        return config


@keras.saving.register_keras_serializable(package='Augmentation', name='ReturnNoneLayer')
class ReturnNoneLayer(Layer):
    """A Keras Layer that returns None, regardless for input into a layer."""
    def __init__(self, name='none_layer', **kwargs):
        super().__init__(name=name, **kwargs)

    def __repr__(self):
        return f"{self.name}"

    def call(self, inputs, training=None, *args, **kwargs):
        return None

    def get_config(self):
        return super().get_config()
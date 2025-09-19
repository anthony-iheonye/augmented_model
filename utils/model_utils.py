from typing import Union, Optional, Dict, List, Tuple

import tensorflow as tf
from keras import Model
from keras.layers import Layer


class XLayer(Layer):
    def __init__(self, name='', **kwargs):
        super().__init__(name=name, **kwargs)
        self.trainable_layers = None


class ModelUtilities:
    def __init__(self):
        self.name: Optional[str] = None
        self.trainable_layers: Optional[list[XLayer]] = None
        self.built = False

    def get_inner_layers(self) -> Dict[str, XLayer]:
        """
        Recursively retrieve and return all basic (non-nested) layers in the model.
        Example of basic layers includes Dense, Conv2D, Dropout, MaxPool, Concat, etc.
        """
        layers = {}
        if not self.trainable_layers:
            return layers

        for layer in self.trainable_layers:
            self._collect_inner_layers(layer, layers)

        return layers

    def _collect_inner_layers(self, layer: XLayer, layers: Dict[str, XLayer]) -> None:
        """
        Recursively collects the most basic trainable layers from a nested layer structure.
        """

        # Base case: If the layer does not contain nested layers.
        if not hasattr(layer, 'trainable_layers') or not layer.trainable_layers:
            layers[layer.name] = layer
            return

        # Recursive case: dig into nested layers
        for sublayer in layer.trainable_layers:
            self._collect_inner_layers(sublayer, layers)

    def view_trainable_status(self, line_length=50):
        """
        View which layers in the model have frozen (trainable) or unfrozen
        (un-trainable) weights.

        :return: A printout containing the index (position of the layer
            in the model), the name, and trainable status of all the layers in the model
        """
        length = 50
        name_length = len(self.name)
        extra_ticks = line_length - length
        num_ticks = (length + extra_ticks - (name_length + 13)) // 2

        print(f"\n\n{'-' * num_ticks + self.name.upper() + ' LAYER STATUS' + '-' * num_ticks :^{length}}\n"
              f"{'INDEX': <7s}{'LAYER_NAME':<{28 + extra_ticks}s}{'TRAINABLE LAYER'}")

        for idx, (name, layer) in enumerate(self.get_inner_layers().items()):
            print(f"{str(idx) + '.':<7}{name:<{28 + extra_ticks}s}{layer.trainable}")

    @staticmethod
    def is_batchnorm(layer: Layer,
                     batchnorm_identifiers: tuple = ('_bn', 'batch_normalization')) -> bool:
        """
        Determines whether the given layer is a batch normalization layer based on name patterns.

        This is useful when you want to freeze or skip BatchNorm layers separately, which is a common
        practice during fine-tuning due to their internal running statistics.

        :param layer: A Keras Layer instance to evaluate.
        :param batchnorm_identifiers: A tuple of substrings commonly found in BatchNorm layer names.
                                       If any substring is found in the layer's name, the layer is treated as batchnorm.
        :return: True if the layer appears to be a BatchNorm layer based on its name; False otherwise.
        """
        return any(identifier in layer.name for identifier in batchnorm_identifiers)

    def freeze_layers(self,
                      layers: Union[str, list, tuple] = None,
                      positions: Union[str, int, list, tuple] = None,
                      batchnorm_ids: Tuple[str] = ('_bn', 'batch_normalization')):
        """
        Freezes the weights of selected layers in the model.

        This method supports three use cases:
          - If `layers=None` and `positions=None`, all non-batchnorm layers will be unfrozen,
            and all batchnorm layers will be frozen.
          - If `layers='all'` or `positions='all'`, all layers will be frozen including batchnorm layers.
          - If `layers` is a list/tuple of layer names, only those layers will be frozen.
          - If `positions` is an int, list, or tuple of indices, the corresponding layers will be frozen.

        In all cases, batch normalization layers will be frozen.

        :param layers: A string ('all'), or list/tuple of layer names to freeze.
        :param positions: An int (layer index), list/tuple of indices, or 'all' to freeze all layers by index.
                          Cannot be used with `layers`.
        :param batchnorm_ids: Substrings typically found in batch normalization layer names, for instance ('_bn',
                            'batch_normalization'). Batchnorm layers are always frozen if detected.
        :raises ValueError: If both `layers` and `positions` are provided or invalid types are passed.
        """

        if layers is not None and positions is not None:
            raise ValueError('Use either `layers` or `positions`, not both.')

        # Dict mapping layer name to layer object.
        model_layers = self.get_inner_layers()

        # Case1: Unfreeze all layers except batchnorms
        if layers is None and positions is None:
            for layer in model_layers.values():
                layer.trainable = True if not self.is_batchnorm(layer, batchnorm_ids) else False
            return

        # Case 2: Freeze by layer name
        if layers is not None:
            error_message = ValueError(f"layers must either be set to 'None', 'all' a list, or "
                                       f"a tuple, got `{layers}` of type {type(layers).__name__}.")

            # Freeze all layers (if layers = 'all')
            if isinstance(layers, str):
                if layers.lower() == 'all':
                    for layer in model_layers.values():
                        layer.trainable = False
                    return
                raise error_message

            # Unfreeze all layers
            for layer in model_layers.values():
                layer.trainable = True if not self.is_batchnorm(layer, batchnorm_ids) else False

            if isinstance(layers, (list, tuple)):
                for name in layers:
                    if not isinstance(name, str):
                        raise ValueError(f'Layer name must be a string - got {name}, of type {type(name).__name__}.')

                # Freeze specified layers
                for name in layers:
                    layer = model_layers.get(name)
                    if layer is None:   # Layer does not exist
                        continue
                    layer.trainable = False
                return
            else:
                raise error_message


        # Case 3: Freeze by layer index/position
        if positions is not None:
            error_message = ValueError(f"positions must either be set to 'None', 'all', "
                                       f"an int, a list, or a tuple, got `{positions}` of "
                                       f"type {type(positions).__name__}.")

            all_layers = list(model_layers.values())

            # Freeze all layers (if layers = 'all').
            if isinstance(positions, str):
                if positions.lower() == 'all':
                    for layer in model_layers.values():
                        layer.trainable = False
                    return
                raise error_message

            # Unfreeze the weights of all layers
            for layer in all_layers:
                layer.trainable = True if not self.is_batchnorm(layer, batchnorm_ids) else False

            if isinstance(positions, (list, tuple)):
                for idx in positions:
                    if not isinstance(idx, int):
                        raise ValueError(f"Each position must be an int — got {type(idx).__name__}.")

                    if idx < 0 or idx >= len(all_layers):
                        raise IndexError(f"Layer index {idx} is out of range (model has {len(all_layers)} layers).")

                # Freeze layers at specified positions
                for position in positions:
                    layer = all_layers[position]
                    layer.trainable = False
                return

            # Freeze a layer at a specific index
            if isinstance(positions, int):
                layer = all_layers[positions]
                layer.trainable = False
                return
            raise error_message



    def set_weights_from_layers(self, layers: Union[List[Layer], Tuple[Layer]]) -> None:
        """
        Load weights from a list of layers into the current model layers (by matching layer names).
        Args:
            layers (Union[List[Layer], Tuple[Layer]]): A collection of Keras Layer objects with weights to load.
        """

        if not self.built:
            raise ValueError(
                "You called `set_weights_from_layers` on a model that has not been built.\n"
                "Please run `model.build((None, height, width, channels))` before calling this method, "
                "or pass `input_shape` when instantiating the model to ensure it is built automatically."
            )

        inner_layers = self.get_inner_layers()

        for layer in layers:
            name = layer.name
            if name in inner_layers:
                inner_layers.get(name).set_weights(layer.get_weights())


class L2Regularizer:

    def __init__(self):
        """
        Returns an L2 regularization layer, set to the assign regularization value.
        """

    def __call__(self, factor: Optional[float] = None):
        """

        :param factor: The regularization value
        :return: None, if factor is set to None, or a regularization layer, if factor is a float.
        """
        if factor is None:
            return None
        else:
            return tf.keras.regularizers.l2(l2=factor)


class LayerExtractor:
    """
    Returns the lower_level layers in a model. It the input is a
    layer, this class returns the sub-layers, if any.
    """

    def __call__(self, model: Union[Model, XLayer]):
        """

        :param model: A tf.Keras Model or Layer object
        :return: A list containing the low-level layers.
        """

        layers = {}
        for layer in model.trainable_layers:
            layers.update(self._get_layers(layer, {}))
        return layers

    def _get_layers(self, layer, result):
        if not hasattr(layer, 'trainable_layers'):
            result[layer.name] = layer
            return result

        if len(layer.trainable_layers) == 0:
            return None

        for child_layer in layer.trainable_layers:
            self._get_layers(child_layer, result)

        return result


class UNetTFModule(tf.Module):
    """Create a tf.Module version of a UNet tf.keras model. The UNet module would have a segment method."""

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def segment(self, image: tf.Tensor):
        """Returns an RGB image containing only the objects of interest or foreground"""
        return self.model.segment(image)

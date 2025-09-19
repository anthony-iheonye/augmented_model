from typing import Optional, Tuple

from tensorflow.keras import Model
from tensorflow.keras.saving import register_keras_serializable

from layers.segmenter import Decoder, Encoder, BottleNeck, PostProcessMask
from layers.segmenter import ForegroundExtractor, MergeLabelAxis
from utils import ModelUtilities, UNetTFModule


@register_keras_serializable(package='Unet', name='Unet')
class UNet(Model, ModelUtilities):
    """UNet semantic segmentation model."""
    def __init__(self,
                 num_classes: int,
                 input_shape: Optional[Tuple[int, int, int]] = None,
                 dropout_rate: float = 0.0,
                 l2_reg_factor: float = None,
                 post_process: bool = False,
                 opening_radius: int = 5,
                 closing_radius: int = 3,
                 erosion_radius: int = 2,
                 dilation_radius: int = 1,
                 min_size: int = 2000,
                 area_threshold: int = 300,
                 return_dict: bool = True,
                 input_name: str = 'image',
                 output_name: str = 'predicted_mask',
                 name='UNET',
                 **kwargs):
        """

        :param num_classes: (int) Number of classes
        :param input_shape: (Tuple) A tuple specifying the full shape of a single input sample,
            including height, width, and number of channels (e.g., `(1024, 1024, 3)` for RGB images).
            This excludes the batch dimension, which is added automatically as `None` during model building.
            If input_shape is provided, the model is automatically built on initialization.
        :param dropout_rate: (float) The dropout rate for the UNet Model.
        :param l2_reg_factor: (float) The L2-regularization factor that will be applied during model training.
        :param post_process: (bool) Whether to apply morphological postprocessing after segmentation or not.
        :param opening_radius: (int) The radius of the disk-shaped structuring element that would be used to
            apply morphological opening on the mask.
        :param closing_radius: (int) The radius of the disk-shaped structuring element that would be used to
            apply morphological closing on the mask.
        :param erosion_radius: (int) The radius of the disk-shaped structuring element that would be used to
            apply morphological erosion on the mask.
        :param dilation_radius: (int) The radius of the disk-shaped structuring element that would be used to
            apply morphological dilation on the mask.
        :param min_size: The smallest allowable object size, that will not be treated as noise when applying
            morphological (post-segmentation) operation on predicted mask.
        :param area_threshold: (int) The maximum area, in pixels, of a contiguous hole that will be filled,
            during post-processing.
        :param return_dict: If `True`, the predicted outputs are returned as a
              dict, with each key being the name of an output. If return_dict is set
              to `False`, the outputs are returned as a list.
              While return_dict is `True`, if `post_process` is `False`, the dict key
              will be `predicted_mask`, if `post_process` is `True`, the dict keys will
              be [`predicted_mask`, `processed_mask`, `processed_2d_mask`].
        :param name: Name of the model, set as `UNET` by default.
        """
        Model.__init__(self, name=name, **kwargs)

        self.image_shape = input_shape
        self.output_name = output_name
        self.input_name = input_name
        self.post_process = post_process
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.l2_reg_factor = l2_reg_factor

        self.opening_radius = opening_radius
        self.closing_radius = closing_radius
        self.erosion_radius = erosion_radius
        self.dilation_radius = dilation_radius
        self.min_size = min_size
        self.area_threshold = area_threshold
        self.return_dict = return_dict

        # The Layers that make up the UNET model
        self.encoder = Encoder(dropout_rate=dropout_rate, l2_reg_factor=l2_reg_factor)
        self.bottleneck = BottleNeck(l2_reg_factor=l2_reg_factor)
        self.decoder = Decoder(num_classes=num_classes,
                               dropout_rate=dropout_rate,
                               l2_reg_factor=l2_reg_factor)

        # if post_process:
        #     # Post-processing layer
        self.morphological_layer = PostProcessMask(num_classes=num_classes,
                                                   opening_radius=opening_radius,
                                                   closing_radius=closing_radius,
                                                   erosion_radius=erosion_radius,
                                                   dilation_radius=dilation_radius,
                                                   min_size=min_size,
                                                   area_threshold=area_threshold,
                                                   name='processed_2D_mask')

        # Layer to generate RGB version of the segmented result
        self.foreground = ForegroundExtractor(name='infocus_peas')

        # Layer to convert a 3D masks to 2D, by merging the label axis.
        self.merge_label_axis = MergeLabelAxis(name='merged_mask')

        self.built = False

        if input_shape is not None:
            self.build((None, *input_shape))
            self.built = True

    def __repr__(self):
        return f"{self.name}"

    def call(self, inputs):
        """Segments an image and returns the segmentation mask."""
        if isinstance(inputs, dict):
            inputs = inputs[self.input_name]

        encoder_output, convs = self.encoder(inputs)
        bottleneck = self.bottleneck(encoder_output)
        predicted_mask = self.decoder([bottleneck, convs])

        # if self.post_process:
        #     # apply morphological operation on the predicted mask.
        #     processed_2d_mask, processed_3d_mask = self.morphological_layer(predicted_mask)
        # else:
        #     processed_2d_mask = tf.ones(tf.shape(predicted_mask), dtype=tf.float32)
        #     processed_3d_mask = tf.ones(tf.shape(predicted_mask), dtype=tf.float32)

        # return output as a dict if the inputs are supplied as dict in the tf.Dataset
        if self.return_dict:
            return {self.output_name: predicted_mask,
                    # 'processed_mask': processed_3d_mask,
                    # 'processed_2d_mask': processed_2d_mask,
                    }
        # return predicted_mask, processed_3d_mask, processed_2d_mask
        return predicted_mask

    def segment(self, image_dict: dict):
        """Returns an RGB image containing only the objects of interest or foreground"""
        if not isinstance(image_dict, dict):
            image_dict = {self.input_name: image_dict}

        # Generate segmentation mask
        mask = self(image_dict)

        # produce RGB image containing only foreground/objects of interest (e.g., infocus peas)
        mask = mask[self.output_name] if self.return_dict else mask
        if self.post_process:
            # apply morphological operation on the predicted mask.
            processed_2d_mask, processed_3d_mask = self.morphological_layer(mask)
            foreground = self.foreground([processed_2d_mask, image_dict[self.input_name]])
        else:
            predicted_2d_mask = self.merge_label_axis(mask)
            foreground = self.foreground([predicted_2d_mask, image_dict[self.input_name]])

        return {'infocus_peas': foreground}

    @property
    def trainable_layers(self):
        return self.encoder, self.bottleneck, self.decoder

    @property
    def tf_module(self):
        """
        Returns the UNet keras model as a tf.Module. Here,
        the `Unet.segment()` method wrapped in a `tf.function`,
        thus, increasing the computational speed.
        """
        return UNetTFModule(self)

    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes,
                       'dropout_rate': self.dropout_rate,
                       'l2_reg_factor': self.l2_reg_factor,
                       'post_process': self.post_process,
                       'opening_radius': self.opening_radius,
                       'closing_radius': self.closing_radius,
                       "erosion_radius": self.erosion_radius,
                       "dilation_radius": self.dilation_radius,
                       "min_size": self.min_size,
                       "area_threshold": self.area_threshold,
                       'name': self.name})
        return config

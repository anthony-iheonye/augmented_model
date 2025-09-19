from typing import Optional

from tensorflow.keras.layers import Layer
from tensorflow.keras.saving import register_keras_serializable

from layers.segmenter.conv_blocks import Conv2dBlock


@register_keras_serializable(package='Unet', name='BottleNeck')
class BottleNeck(Layer):
    def __init__(self,
                 num_filters: int = 1024,
                 l2_reg_factor: Optional[float] = None,
                 name='bottle_neck', **kwargs):
        """
        This class defines the bottleneck for the UNET model. The bottleneck convolutional
        block extract more features from the input tensor before the result are fed into the
        upsampling layers.

        :param num_filters: int, the number of filters for the 2D convolutional layers
            within the Conv2dBlock that make up the BottleNeck layer.
        :param l2_reg_factor: float, The L2 regularization factor for the regularizer
            function that is applied to the `kernel` weights matrix
        """
        super().__init__(name=name, **kwargs)

        self.num_filters = num_filters
        self.l2_reg_factor = l2_reg_factor
        self.bottle_neck = Conv2dBlock(num_filters=num_filters, l2_reg_factor=l2_reg_factor,
                                       name=name)

    def __repr__(self):
        return f"{self.name}"

    def call(self, inputs, *args, **kwargs):
        """
        :param inputs: The output from the last encoder block within the encoder.
            Recall that the encoder returns two values - a tensor (the encoder output) and a tuple
            containing the outputs from each Conv2dBlock within the four Encoder blocks.
        :return:
        """
        bottleneck = self.bottle_neck(inputs)
        return bottleneck

    @property
    def trainable_layers(self):
        return [self.bottle_neck]

    def get_config(self):
        config = super().get_config()
        config.update({'num_filters': self.num_filters,
                       'l2_reg_factor': self.l2_reg_factor,
                       'name' :self.name})
        return config


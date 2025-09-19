from tensorflow.keras.layers import Layer
from tensorflow.keras.saving import register_keras_serializable
from typing import Optional

from layers.segmenter.conv_blocks import EncoderBlock

@register_keras_serializable(package='Unet', name='Encoder')
class Encoder(Layer):
    """The Encoder for the UNet Model"""
    def __init__(self,
                 dropout_rate: float = 0.0,
                 l2_reg_factor: Optional[float] = None,
                 name: str = 'Encoder',
                 **kwargs):
        """
        This Encoder class implements the down-sampling section of the UNet model.

        :param dropout_rate: (float) dropout rate/probability
        :param l2_reg_factor: float, The L2 regularization factor for the regularizer
            function that is applied to the `kernel` weights matrix
        :return: This returns a tuple containing p4 - output maxpooled features of the last encoder block; and
            (f1, f2, f3, f4) - the output features of all the encoder blocks
        """
        super().__init__(name=name, **kwargs)

        self.dropout_rate = dropout_rate
        self.l2_reg_factor = l2_reg_factor

        self.encoder_block1 = EncoderBlock(num_filters=64, pool_size=(2 ,2), dropout_rate=dropout_rate,
                                           l2_reg_factor=l2_reg_factor, name='encod_blk_1')

        self.encoder_block2 = EncoderBlock(num_filters=128, pool_size=(2 ,2), dropout_rate=dropout_rate,
                                           l2_reg_factor=l2_reg_factor, name='encod_blk_2')

        self.encoder_block3 = EncoderBlock(num_filters=256, pool_size=(2 ,2), dropout_rate=dropout_rate,
                                           l2_reg_factor=l2_reg_factor, name='encod_blk_3')

        self.encoder_block4 = EncoderBlock(num_filters=512, pool_size=(2 ,2), dropout_rate=dropout_rate,
                                           l2_reg_factor=l2_reg_factor, name='encod_blk_4')


    def __repr__(self):
        return f"{self.name}"

    @property
    def trainable_layers(self):
        return self.encoder_block1, self.encoder_block2, self.encoder_block3, self.encoder_block4

    def call(self, inputs, *args, **kwargs):
        # f1, f2, f3, f4 are the outputs from the Conv2dBlock within each EncoderBlock,
        # before Maxpooling and dropout was applied. p1, p2, p3, and p4 as the results
        # after maxpooling and dropout have been applied.
        f1, p1 = self.encoder_block1(inputs)
        f2, p2 = self.encoder_block2(p1)
        f3, p3 = self.encoder_block3(p2)
        f4, p4 = self.encoder_block4(p3)
        return p4, (f1, f2, f3, f4)

    def get_config(self):
        config = super().get_config()
        config.update({'name' :self.name,
                       'dropout_rate': self.dropout_rate,
                       'l2_reg_factor': self.l2_reg_factor})
        return config



from typing import Union

import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, MaxPool2D, Dropout, Concatenate)
from tensorflow.keras.layers import Layer
from tensorflow.keras.saving import register_keras_serializable

from utils.model_utils import L2Regularizer


@register_keras_serializable(package='Unet', name='Conv2dBlock')
class Conv2dBlock(Layer):
    def __init__(self,
                 num_filters: int,
                 kernel_size: Union[list, tuple] = (3, 3),
                 l2_reg_factor: float = None,
                 name: str = 'block', **kwargs):
        """
        Produces a custom Keras layer that passes the input through two 2D
        convolutional layers and a RELU activation layer.

        :param num_filters: (int) -- Number of filters
        :param kernel_size: kernel size for convolution
        :param l2_reg_factor: float, The L2 regularization factor for the regularizer
            function that is applied to the `kernel` weights matrix
        :param name: name of the convolutional block
        :return: (tensor) -- Tensor of output features
        """
        super().__init__(name=name, **kwargs)

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.regularizer = L2Regularizer()
        self.reg_factor = l2_reg_factor

        self.conv_1 =  Conv2D(filters=num_filters, kernel_size=kernel_size, padding='same',
                              kernel_initializer='he_normal', name=f"conv1_{name}",
                              kernel_regularizer=self.regularizer(l2_reg_factor) ,)

        self.conv_2 =  Conv2D(filters=num_filters, kernel_size=kernel_size, padding='same',
                              kernel_initializer='he_normal', name=f"conv2_{name}",
                              kernel_regularizer=self.regularizer(l2_reg_factor) ,)

        self.activation = tf.keras.layers.Activation('relu', name=f'relu_{name}')

    def __repr__(self):
        return f"{self.name}"

    def call(self, inputs, *args, **kwargs):
        """Passes the inputs through a 2-D convolutional block."""
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = self.activation(x)
        return x

    @property
    def trainable_layers(self):
        return self.conv_1, self.conv_2, self.activation

    def get_config(self):
        config = super().get_config()
        config.update({'num_filters' :self.num_filters,
                       'kernel_size': self.kernel_size,
                       'reg_factor': self.reg_factor,
                       'name': self.name})
        return config


@register_keras_serializable(package='Unet', name='EncoderBlock')
class EncoderBlock(Layer):
    def __init__(self,
                 num_filters: int,
                 pool_size: Union[list, tuple, int] = (2 ,2),
                 dropout_rate: float = 0.0,
                 l2_reg_factor: float = None,
                 name: str = '',
                 **kwargs):
        """
        The EncoderBlock passes the input tensor through a Conv2dBlock layer, a MaxPooling layer and a dropout layer.

        :param inputs: (tensor) --  The input tensor
        :param n_filters: (int) -- number of filters
        :param pool_size: (int) -- kernel size for the Max pooling layer
        :param dropout_rate: (float) -- probability value for dropout
        :param l2_reg_factor: float, The L2 regularization factor for the regularizer
            function that is applied to the `kernel` weights matrix
        :param name: (str) -- Name for the EncoderBlock
        :returns: f- The output features of the convolution block, and
                  p- The maxpooled features with dropout
        """
        super().__init__(name=name, **kwargs)

        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        self.l2_reg_factor = l2_reg_factor
        self.pool_size = pool_size

        self.conv2d_block = Conv2dBlock(num_filters=num_filters,
                                        l2_reg_factor=l2_reg_factor,
                                        name=name)

        self.max_pool = MaxPool2D(pool_size=pool_size, name=f"maxpool_{name}")
        self.dropout = Dropout(rate=dropout_rate, name=f"dropout_{name}")

    def __repr__(self):
        return f"{self.name}"

    def call(self, inputs, *args, **kwargs):
        f = self.conv2d_block(inputs)
        p = self.max_pool(f)
        p = self.dropout(p)
        return f, p

    @property
    def trainable_layers(self):
        return self.conv2d_block, self.max_pool, self.dropout

    def get_config(self):
        config = super().get_config()
        config.update({'num_filters' :self.num_filters,
                       'pool_size': self.pool_size,
                       'dropout_rate': self.dropout_rate,
                       'l2_reg_factor': self.l2_reg_factor,
                       'name': self.name})
        return config


@register_keras_serializable(package='Unet', name='DecoderBlock')
class DecoderBlock(Layer):
    def __init__(self,
                 num_filters: int,
                 kernel_size: Union[list, tuple, int] = (3, 3),
                 pool_size: Union[list, tuple, int] = (2, 2),
                 strides: Union[list, tuple, int] = (2, 2),
                 padding: str = 'same',
                 dropout_rate: float = 0.0,
                 l2_reg_factor: float = None,
                 name: str = '',
                 **kwargs):
        """
        Defines a decoder block for the UNet model.

        :param num_filters: (int): number of filters
        :param kernel_size: (int): kernel size
        :param strides: (int): strides for the deconvolution/upsampling
        :param dropout_rate: (float): dropout probability
        :param padding: (str): 'same' or 'valid'. same retains the original shape of the image.
        :param l2_reg_factor: float, The L2 regularization factor for the regularizer
            function that is applied to the `kernel` weights matrix
        :param name: (int): name of convolutional block
        :return: Tensor, output features of the decoder block
        """
        super().__init__(name=name, **kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.l2_reg_factor = l2_reg_factor
        self.regularizer = L2Regularizer()

        self.conv_transpose = Conv2DTranspose(filters=num_filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                              kernel_regularizer=self.regularizer(l2_reg_factor), name=f"deconv_{name}")

        self.concatenate = Concatenate(name=f'concat_{name}')
        self.dropout = Dropout(rate=dropout_rate, name=f"dropout_{name}")
        self.conv2d_block = Conv2dBlock(num_filters=num_filters, kernel_size=kernel_size, l2_reg_factor=l2_reg_factor,
                                        name=name)

    def __repr__(self):
        return f"{self.name}"

    def call(self, inputs, *args, **kwargs):
        inputs, conv_outputs = inputs

        u = self.conv_transpose(inputs)
        c = self.concatenate([u, conv_outputs])
        c = self.dropout(c)
        c = self.conv2d_block(c)
        return c

    @property
    def trainable_layers(self):
        return self.conv_transpose, self.concatenate, self.dropout, self.conv2d_block

    def get_config(self):
        config = super().get_config()
        config.update({'num_filters': self.num_filters,
                       'kernel_size': self.kernel_size,
                       'pool_size': self.pool_size,
                       'strides': self.strides,
                       'padding': self.padding,
                       'dropout_rate': self.dropout_rate,
                       'l2_reg_factor': self.l2_reg_factor,
                       'name': self.name})
        return config

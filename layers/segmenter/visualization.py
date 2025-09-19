import tensorflow as tf
from keras.layers import Layer
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable(package='Unet', name='ForegroundExtractor')
class ForegroundExtractor(Layer):
    """
    Produces an RGB image containing the objects of interest.


    Input to this layer is a list containing the 2D-mask and the original rgb image.

    :returns: A 3-dimensional tensor of the foreground (objects of interest).
    """

    def __init__(self, name='infocus_peas'):
        super().__init__(name=name, trainable=False, dtype=tf.float32)

    def __repr__(self):
        return f"{self.name}"

    def call(self, inputs, *args, **kwargs):
        """
        :param inputs: a list containing the 3D-mask and the original rgb image.
        """
        mask_2d, rgb_image = inputs

        if len(mask_2d.shape) < 4:
            mask_2d = tf.expand_dims(mask_2d, axis=-1)

        x = tf.multiply(x=rgb_image, y=tf.cast(mask_2d, tf.float32))
        return tf.image.convert_image_dtype(image=x, dtype=tf.uint8)

    def get_config(self):
        config = super().get_config()
        config.update({'name': self.name})
        return config


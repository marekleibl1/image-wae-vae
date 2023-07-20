import logging
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose
from tensorflow.keras import regularizers

from autoencoders.model.autoencoder import Autoencoder
from autoencoders.model.tf_utils import conv2d_trans_weights

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ConvAE(Autoencoder):
    """
    A simple convolutional autoencoder.
    Encodes the given image into an image with lower resolution and more features.
    """

    def _build_model(self, **kwargs):
        # --- parameters

        learning_rate = kwargs.get('learning_rate')
        l2_weight = kwargs.get('l2_weight')
        nf = kwargs.get('nbasefilters', 32)
        z_dim = kwargs.get('zdim', 10)
        out_dim = 1

        conv_params = {
            'kernel_size': 3,
            'padding': 'same',
            'activation': tf.nn.leaky_relu,
            'kernel_initializer': tf.initializers.glorot_normal(seed=0),
            'kernel_regularizer': regularizers.l2(l2_weight),
        }

        # sigmoid activation to producte latent images with values in [0, 1] 
        conv_params_last = conv_params.copy()
        # conv_params_last['activation'] = tf.nn.sigmoid

        # ---  encoder

        deconv_params = {
            'strides': [2, 2],
            'kernel_size': 4,
            'padding': 'same',
            'kernel_regularizer': regularizers.l2(l2_weight)
        }

        # see this https://github.com/tensorlayer/tensorlayer/issues/53
        init1 = tf.keras.initializers.Constant(conv2d_trans_weights(4 * nf, 2 * nf))
        init2 = tf.keras.initializers.Constant(conv2d_trans_weights(2 * nf, nf))

        self._encoder = tf.keras.Sequential([
            Conv2D(nf, **conv_params),
            Conv2D(2 * nf, **conv_params),
            MaxPool2D(),
            Conv2D(2 * nf, **conv_params),
            Conv2D(4 * nf, **conv_params),
            MaxPool2D(),
            Conv2D(4 * nf, **conv_params),
            Conv2D(z_dim, **conv_params_last),
        ])

        self._decoder = tf.keras.Sequential([
            Conv2D(4 * nf, **conv_params),
            Conv2D(4 * nf, **conv_params),
            Conv2DTranspose(2 * nf, **deconv_params, kernel_initializer=init1),
            Conv2D(2 * nf, **conv_params),
            Conv2DTranspose(nf, **deconv_params, kernel_initializer=init2),
            Conv2D(out_dim, **conv_params)
        ])

        # note that: there might be a better way to implement this is tf 2.12
        self._encoder.compile(optimizer='adam', loss='mse', metrics=[])
        self._decoder.compile(optimizer='adam', loss='mse', metrics=[])

        self._optimizer = tf.keras.optimizers.Adam(learning_rate)
        self._loss = tf.keras.losses.MeanSquaredError()

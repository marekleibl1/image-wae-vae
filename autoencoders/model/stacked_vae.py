import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.python.keras.losses import MeanSquaredError

from autoencoders.model.autoencoder import Autoencoder
from autoencoders.model.stacked_ae import StackedAE


class StackedVAE(StackedAE):
    """
    Variational autoencoder.

    Based on:
    https://www.tensorflow.org/tutorials/generative/cvae
    https://www.tensorflow.org/guide/keras/custom_layers_and_models#putting_it_all_together_an_end-to-end_example
    """

    def __init__(self):
        super(StackedVAE, self).__init__()
        self.add_logvar = True

    def _compute_loss(self, x):
        # --- encode & decode

        z, mean, logvar = self.encode(x)
        x_rec = self.decode(z)

        rec_loss = MeanSquaredError()(x, x_rec)

        # --- VAE loss

        # TODO add to config 
        w_zreg_loss = 0.001
        # w_l2reg = 0.0001 # TODO this might be too large 

        # KL divergence regularization loss
        # zreg_loss = - 0.5 * tf.reduce_mean(logvar - tf.square(mean) - tf.exp(logvar) + 1)
        zreg_loss = 0.5 * tf.reduce_mean(tf.square(mean) + tf.exp(logvar) - logvar - 1)
        
        zreg_loss *= w_zreg_loss
        # zreg_loss = 0 

        # --- regularization

        # add regularization term (L2 regularization)
        reg_loss = tf.math.add_n(self._encoder.losses) + tf.math.add_n(self._decoder.losses)
        # reg_loss = w_l2reg * reg_loss
        
        total_loss = rec_loss + reg_loss + zreg_loss

        return total_loss, rec_loss, reg_loss, zreg_loss

    @tf.function
    def encode(self, x):
        """
        Unlike vanilla autoencoder, encoding is stochastic.

        :return a tuple (a sample from N(mean, var), mean, var)
        """
        # compute mean and logvar
        encoded = self._encoder(x)
        mean, logvar = tf.split(encoded, num_or_size_splits=2, axis=1)

        # reparametrize
        eps = tf.random.normal(shape=mean.shape)
        z = eps * tf.exp(logvar * .5) + mean

        return z, mean, logvar

    @tf.function
    def decode(self, z, *args):
        return self._decoder(z)

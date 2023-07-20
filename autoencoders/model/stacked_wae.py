import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.python.keras.losses import MeanSquaredError

from autoencoders.model.autoencoder import Autoencoder
from autoencoders.model.stacked_ae import StackedAE
from autoencoders.model.wae_utils.wae import mmd_penalty


class StackedWAE(StackedAE):
    """
    Wasserstein Autoencoder.

    Based on:
    https://github.com/tolstikhin/wae
    """

    def __init__(self):
        super(StackedWAE, self).__init__()
        self.pretrain = True

    def _sample_pz(self, batch_size, z_dim):
        return tf.random.normal([batch_size, z_dim])

    def _compute_loss(self, x):
        encoder, decoder = self._encoder, self._decoder

        z = self.encode(x)
        x_rec = self.decode(z)

        rec_loss = MeanSquaredError()(x, x_rec)

        reg_loss = tf.math.add_n(encoder.losses) + tf.math.add_n(decoder.losses)

        # --- MMD loss
        # based on this https://github.com/tolstikhin/wae/blob/master/wae.py

        # sample from Pz
        sample_pz = self._sample_pz(x.shape[0], z.shape[-1])

        mmd_loss = mmd_penalty(z, sample_pz)
        w_mmd = self.wae_w_mmd

        zreg_loss = w_mmd * mmd_loss

        total_loss = rec_loss + reg_loss + zreg_loss
        return total_loss, rec_loss, reg_loss, zreg_loss

    def _pretrain_loss(self, sample_qz, sample_pz):
        """

        :param sample_qz: encoded batch
        :param sample_pz: sample from prior distribution Pz (e.q. multivar. normal distr.)
        :return:
        """
        e_pretrain_sample_size = sample_qz.shape[0]

        mean_pz = tf.reduce_mean(sample_pz, axis=0, keepdims=True)
        mean_qz = tf.reduce_mean(sample_qz, axis=0, keepdims=True)
        mean_loss = tf.reduce_mean(tf.square(mean_pz - mean_qz))

        cov_pz = tf.matmul(sample_pz - mean_pz, sample_pz - mean_pz, transpose_a=True)
        cov_pz /= e_pretrain_sample_size - 1.
        cov_qz = tf.matmul(sample_qz - mean_qz, sample_qz - mean_qz, transpose_a=True)
        cov_qz /= e_pretrain_sample_size - 1.
        cov_loss = tf.reduce_mean(tf.square(cov_pz - cov_qz))
        return mean_loss + cov_loss

    @tf.function
    def pretrain_step(self, x):
        """
        Pretrain the encoder so that mean and covariance of Qz will try to match those of Pz
        """

        # TODO can we just minimize mmd loss during the pretraining? -> try both

        encoder, optimizer = self._encoder, self._optimizer

        with tf.GradientTape() as tape:
            z = self.encode(x)
            sample_pz = self._sample_pz(x.shape[0], z.shape[-1])
            loss = self._pretrain_loss(z, sample_pz)

        gradients = tape.gradient(loss, encoder.trainable_variables)
        optimizer.apply_gradients(zip(gradients, encoder.trainable_variables))

        return loss

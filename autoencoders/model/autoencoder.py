import logging
import os
from os.path import join
import tensorflow as tf
from tensorflow.python.keras.losses import MeanSquaredError

logger = logging.getLogger(__name__)


class Autoencoder:
    """
    Abstract autoencoder class.
    """

    def __init__(self):
        """
        Use 'create_new_model' to create new model.
        """
        self._encoder = None
        self._decoder = None
        self._optimizer = None

        # pretrain encoder
        self.pretrain = False

        # extra neurons for logvar (e.q. for VAE)
        self.add_logvar = False

    def _build_model(self, **kwargs):
        """
        Write your custom encoder and decoder here (by overriding this function).
        """
        raise NotImplementedError('Not implemented')

    def _compute_loss(self, x):
        encoder, decoder = self._encoder, self._decoder

        x_rec = self.decode(self.encode(x))

        rec_loss = MeanSquaredError()(x, x_rec)
        reg_loss = tf.math.add_n(encoder.losses) + tf.math.add_n(decoder.losses)
        zreg_loss = 0.0

        total_loss = rec_loss + reg_loss + zreg_loss

        return total_loss, rec_loss, reg_loss, zreg_loss

    # --- PUBLIC

    @tf.function
    def train_step(self, x):
        encoder, decoder, optimizer = self._encoder, self._decoder, self._optimizer

        with tf.GradientTape() as tape:
            total_loss, rec_loss, reg_loss, zreg_loss = self._compute_loss(x)

        trainable_variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(total_loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

        return total_loss, rec_loss, reg_loss, zreg_loss

    @tf.function
    def test_step(self, x):
        return self._compute_loss(x)

    @tf.function
    def encode(self, x):
        return self._encoder(x)

    @tf.function
    def decode(self, z, *args):
        return self._decoder(z)

    @tf.function
    def reconstruct(self, x):
        z = self.encode(x)
        return self.decode(z)

    def save(self, model_path):
        os.makedirs(model_path, exist_ok=True)
        self._encoder.save(join(model_path, 'encoder'))
        self._decoder.save(join(model_path, 'decoder'))
        logger.info('Model saved to ' + model_path)

    @classmethod
    def load_model(cls, model_path, **kwargs):
        logger.info('Loading model from ' + model_path)
        model = cls()
        model._encoder = tf.keras.models.load_model(join(model_path, 'encoder'))
        model._decoder = tf.keras.models.load_model(join(model_path, 'decoder'))
        logger.info('Loading model - done ')
        return model

    @classmethod
    def create_model(cls, **kwargs):
        model = cls()
        model._build_model(**kwargs)
        return model

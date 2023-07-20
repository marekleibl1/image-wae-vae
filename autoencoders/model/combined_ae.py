import logging
from os.path import join

import tensorflow as tf

from autoencoders import default_setting
from autoencoders.model.conv_ae import ConvAE
from autoencoders.model.stacked_ae import StackedAE
from autoencoders.model.stacked_vae import StackedVAE
from autoencoders.model.stacked_wae import StackedWAE

logger = logging.getLogger(__name__)


class CombinedAE:
    """
    Only for inference with trained model.

    Combined convolutional and stacked autoencoder.
    """

    def __init__(self, convae, stackedae):
        self._convae = convae
        self._stackedae = stackedae

    @tf.function
    def encode(self, x):
        z = self._stackedae.encode(self._convae.encode(x))
        return z

    def deterministic_encode(self, x):
        z = self.encode(x)
        z = z[1] if type(z) == tuple else z
        return z

    @tf.function
    def decode(self, z):
        x_rec = self._convae.decode(self._stackedae.decode(z))
        return x_rec

    @tf.function
    def reconstruct(self, x):
        z = self.deterministic_encode(x)

        return self.decode(z)

    @classmethod
    def load_model(cls, convAE_model_or_dir, stackedAE_dir, model_type):

        models = {
            'Vanilla': StackedAE,
            'VAE': StackedVAE,
            'WAE': StackedWAE
        }

        if type(convAE_model_or_dir) == str:
            logger.info('Loading convae from ' + convAE_model_or_dir)
            convae = ConvAE.load_model(convAE_model_or_dir)
        else:
            convae = convAE_model_or_dir

        logger.info('Loading stackedae from ' + stackedAE_dir)
        stackedae = models[model_type].load_model(stackedAE_dir)

        logger.info('Loading models - done ')
        return CombinedAE(convae, stackedae)

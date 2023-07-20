import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Flatten, Reshape

from autoencoders.model.autoencoder import Autoencoder


class StackedAE(Autoencoder):
    """
    Stacked autoencoder.
    Encodes the latent images of the original larger AE into fewer latent parameters.
    """

    def _build_model(self, **kwargs):
        
        input_shape = kwargs['input_shape'] #  input shape = output shape
        nfeatures = kwargs['nfeatures'] 
        z_dim =  kwargs['z_dim'] 
        w_l2_reg = kwargs['w_l2_reg'] 
        learning_rate = kwargs['learning_rate']
        self.wae_w_mmd = kwargs['wae_w_mmd']

        dense_params = {
            'activation': tf.nn.leaky_relu,
            'kernel_initializer': tf.initializers.glorot_normal(),
            'kernel_regularizer': regularizers.l2(w_l2_reg)
        }

        # encoder outputs (mu, logvar) in case of VAE
        encoder_output_size = 2 * z_dim if self.add_logvar else z_dim
        decoder_out_features = input_shape[0] * input_shape[1] * input_shape[2]

        self._encoder = tf.keras.Sequential([
            Flatten(),
            Dense(nfeatures, **dense_params),
            Dense(nfeatures, **dense_params),
            Dense(encoder_output_size, **dense_params)
        ])

        self._decoder = tf.keras.Sequential([
            Dense(nfeatures, **dense_params),
            Dense(nfeatures, **dense_params),
            Dense(decoder_out_features, **dense_params),
            Reshape(input_shape)
        ])

        # we use legacy Adam to make it compatible with older tf code 
        self._optimizer = tf.keras.optimizers.legacy.Adam(learning_rate)
        self._loss = tf.keras.losses.MeanSquaredError()


import argparse
from os.path import join
import numpy as np
import os


# Disable Tensorflow debugging information
# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR') 

from autoencoders.default_paths import default_model_dir
from autoencoders.default_setting import fill_hps_with_defaults
from autoencoders.utils.utils import generate_batches, load_dataset
from autoencoders.utils.logging_setup import get_logger
from autoencoders.model.stacked_ae import StackedAE
from autoencoders.model.stacked_vae import StackedVAE
from autoencoders.model.stacked_wae import StackedWAE

logger = get_logger(__name__, 'INFO')

parser = argparse.ArgumentParser(description="Hyper-parameters for ConvAE")
parser.add_argument('--dataset_name', type=str, default=None, help="Dataset name (mnist or circles)")


def _pretrain_encoder(model, train_generator, pretrain_maxsteps, pretrain_stop_at_loss):
    """
    Pretrain encoder (of e.q. Wasserstein autoencoder). 
    """
    logger.info('Pretraining encoder ...')

    for step in range(pretrain_maxsteps):
        loss = model.pretrain_step(next(train_generator))

        if step % 100 == 0:
            logger.info('loss {}'.format(loss))

        if loss < pretrain_stop_at_loss:
            break

    logger.info('Pretraining encoder - done')


def train_autoencoder(dataset_name, model_type = 'VAE', **kwargs):
    """
    Train Variational, Wasserstein or Vanilla autoencoder.
    """

    stackedAE_models = {
        'Vanilla': StackedAE,
        'VAE': StackedVAE,
        'WAE': StackedWAE
    }

    assert model_type in stackedAE_models

    hparams = fill_hps_with_defaults(dataset_name, model_type, custom_hps=kwargs)

    pretrain_maxsteps = hparams['wae_pretrain_maxsteps']
    pretrain_stop_at_loss = hparams['wae_pretrain_stop_at_loss']
    batch_size = hparams['batch_size']
    training_steps = hparams['training_steps']

    model_dir = kwargs.get('model_dir', default_model_dir(dataset_name, model_type))

    # load latent images
    # alternatively, we could train directly on the original images, 
    # but it would require a deeper model to get reasonable results
    train_valid_data = load_dataset(dataset_name + '_zimages')

    # batches generator
    train_generator = generate_batches(train_valid_data['train'], batch_size)
    valid_generator = generate_batches(train_valid_data['valid'], batch_size)

    # create model
    model = stackedAE_models[model_type].create_model(**hparams)

    # Pretraing encoder (depending on the model type)

    avg_valid_loss, best_valid_loss = None, 9999
    
    if model.pretrain:
        _pretrain_encoder(model, train_generator, pretrain_maxsteps, pretrain_stop_at_loss)

    # Main training

    # estimate loss before training
    losses = [model.test_step(next(valid_generator))[0] for _ in range(10)]
    logger.info(f'Step 0 | loss before training {np.mean(losses):1.4}')

    train_losses, valid_losses = [], []

    for step in range(training_steps):

        # Train step
        total_loss, rec_loss, reg_loss, zreg_loss = model.train_step(next(train_generator))
        train_losses.append([total_loss, rec_loss, reg_loss, zreg_loss])

        # Compute validation loss
        if step % 10 == 0:
            total_loss, rec_loss, reg_loss, zreg_loss = model.test_step(next(valid_generator))
            valid_losses.append([total_loss, rec_loss, reg_loss, zreg_loss])

        # Print stats & save the model
        if step % 500 == 0 and step > 0:
            # total loss = sum of reconstuction loss + weights regularization + 
            # + (optional) z regularization

            def print_loss(losses, name):
                total_loss, rec_loss, reg_loss, zreg_loss = np.mean(losses, axis=0)
                logger.info(f"""Step {step} {name} loss: total {total_loss:1.4} rec {rec_loss:1.4} 
                reg {reg_loss:1.4} zreg {zreg_loss:1.4}""")

            print_loss(train_losses, 'train')
            print_loss(valid_losses, 'valid')

            avg_valid_loss = np.mean(valid_losses, axis=0)[0]
            train_losses, valid_losses = [], []

            # Save the model if improved
            if avg_valid_loss < best_valid_loss * 0.99:
                best_valid_loss = avg_valid_loss
                model.save(model_dir)


def train_stacked_autoencoders(dataset_names, model_types, z_dims, training_steps):
    """
    Train multiple models with different latent dimension and type of regularized latent space. 
    """

    for dataset_name in dataset_names:
        for model_type in model_types:
            for z_dim in z_dims:

                logger.info(f'Training {dataset_name}-{model_type} z_dim {z_dim} ...')

                model_dir = default_model_dir(dataset_name, model_type, suffix=f'zdim{z_dim}')

                train_autoencoder(dataset_name, model_type, z_dim = z_dim, 
                    training_steps=training_steps, model_dir = model_dir)

                logger.info(f'Training {dataset_name}-{model_type} z_dim {z_dim} - done')


if __name__ == '__main__':
    
    args = parser.parse_args()
    
    # Dataset: mnist or circles
    dataset_name = getattr(args.__dict__, 'dataset_name', 'mnist')

    # model_types = ['WAE', 'VAE', 'Vanilla']
    
    z_dims = [2, 3, 5, 8, 13, 21]

    training_steps = 80000 
    # training_steps = 200000

    model_types = ['WAE']
    z_dims = [2, 3, 5, 8, 13, 21]

    train_stacked_autoencoders([dataset_name], model_types, z_dims, training_steps)

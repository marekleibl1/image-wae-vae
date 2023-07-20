import argparse
import numpy as np
from os.path import join

from autoencoders.default_paths import default_model_dir
from autoencoders.default_setting import get_default_cnn_hparams
from autoencoders.model.conv_ae import ConvAE
from autoencoders.utils.logging_setup import get_logger
from autoencoders.utils.utils import generate_batches, load_dataset, update_ema_loss

logger = get_logger(__name__, 'INFO')


parser = argparse.ArgumentParser(description="Hyper-parameters for ConvAE")
parser.add_argument('--dataset_name', type=str, default=None, help="Dataset name (mnist or circles)")
parser.add_argument('--learning_rate', type=float, default=None, help="Learning rate")
parser.add_argument('--ntraining_steps', type=int, default=None, help="Number of training steps")
parser.add_argument('--model_dir', type=str, default=None, help="Path to save the trained model")


def _get_hyperparameters(dataset_name):
    args = parser.parse_args()
    
    # default hyper-parameters for the convolutional autoencoder 
    hparams = get_default_cnn_hparams(dataset_name)
    
    if args.learning_rate:
        hparams.learning_rate = args.learning_rate

    if args.ntraining_steps:
        hparams.ntraining_steps = args.ntraining_steps

    if args.model_dir:
        hparams.model_dir = args.model_dir

    return hparams


def train_conv_autoencoder():
    """
    Train a fully convolutional autoencoder to encode images 
    into lower resolution images of shape [W, H, n_features].
    """

    hparams = _get_hyperparameters()
    dataset_name = hparams['dataset_name']
    
    batch_size = hparams['batch_size']
    model_dir = hparams.get('model_dir', default_model_dir(dataset_name, 'convAE')) 
    
    # create new model
    model = ConvAE.create_model(**hparams)

    # load train and valid set
    train_valid_data = load_dataset(dataset_name)

    # batches generator
    train_generator = generate_batches(train_valid_data['train'], batch_size)
    valid_generator = generate_batches(train_valid_data['valid'], batch_size)

    # Main training

    train_losses, valid_losses = [], []

    avg_valid_loss, best_valid_loss = None, 9999

    # estimate loss before training
    losses = [model.test_step(next(valid_generator))[0] for _ in range(10)]
    logger.info(f'Step 0 | loss before training {np.mean(losses):1.4}')

    for step in range(hparams['ntraining_steps']):

        # Train step
        train_loss = model.train_step(next(train_generator))[0]
        train_losses.append(train_loss)

        # Validation loss
        if step % 10 == 0:
            valid_loss = model.test_step(next(valid_generator))[0]
            valid_losses.append(valid_loss)

        # Print stats & save the model
        if step % 500 == 0 and step > 0:
            train, valid = np.mean(train_losses), np.mean(valid_losses)
            logger.info(f'Step {step} | reconstruction loss - train {train:1.4} valid {valid:1.4}')

            avg_valid_loss = np.mean(valid_losses)
            train_losses, valid_losses = [], []

            # Save the model if improved
            if avg_valid_loss < best_valid_loss * 0.99:
                best_valid_loss = avg_valid_loss
                model.save(model_dir)


if __name__ == '__main__':
    args = parser.parse_args()

    # Dataset: mnist or circles
    dataset_name = getattr(args.__dict__, 'dataset_name', 'mnist')

    train_conv_autoencoder(dataset_name)

from os.path import join
import os 
import numpy as np
from autoencoders.default_paths import default_dataset_path
from autoencoders.utils.logging_setup import get_logger
from autoencoders.datasets.synth_data import random_circle
from autoencoders import default_setting

logger = get_logger(__name__, 'INFO')


def load_dataset(dataset_name, split_names=['train', 'valid', 'test']):

    data = {
        split_name: np.load(default_dataset_path(dataset_name, split_name))
        for split_name in split_names
    }

    return data


def generate_batches(images, batch_size):
    """
    Generate batches as random samples from all given (training or validation) images. 
    """

    indexes = np.arange(images.shape[0])

    while True:
        ixs = np.random.choice(indexes, batch_size)
        yield images[ixs]


# def generate_batches(batch_size, **kwargs):
#     """
#     Return a generator of batches with randomly generated circles.
#     """
#     return_tuple = kwargs.get('return_tuple')

#     image_shape = kwargs.get('input_shape', [16, 16])

#     # estimate mu, sigma on a sample
#     sample_size = 200
#     sample = np.stack([random_circle(image_shape) for _ in range(sample_size)], axis=0)
#     mu, sigma = np.mean(sample), np.std(sample)
#     assert sigma > 0

#     while True:
#         # generate the next batch
#         batch = np.stack([random_circle(image_shape) for _ in range(batch_size)], axis=0)

#         # normalize
#         batch = (batch[:, :, :, np.newaxis] - mu) / sigma

#         if return_tuple:
#             yield batch, batch
#         else:
#             yield batch


def update_ema(avg_loss, new_loss, q):
    """
    Update exponential moving average.
    """
    if avg_loss is None:
        return new_loss

    return (1 - q) * avg_loss + q * new_loss


def update_ema_loss(avg_loss, avg_loss_sigma_sq, new_loss, q):
    """
    Update ema loss and ema estimate of variance.
    """
    avg_loss = update_ema(avg_loss, new_loss, q)
    avg_loss_sigma_sq = update_ema(avg_loss_sigma_sq, (new_loss - avg_loss) ** 2, q)
    return avg_loss, avg_loss_sigma_sq


def deterministic_train_test_split(xx, nfolds=5):
    """
    Split data into train / test set deterministically along zero axis.
    """
    is_test = np.array([hash(i) % nfolds == 0 for i in range(xx.shape[0])])
    xx_train, xx_test = xx[np.logical_not(is_test)], xx[is_test]
    assert xx_train.shape[0] + xx_test.shape[0] == xx.shape[0]
    assert xx_train.shape[0] > 0 and xx_test.shape[0] > 0
    return xx_train, xx_test

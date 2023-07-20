from os.path import join
import os 
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from autoencoders.datasets.synth_data import generate_and_export_synth_data

from autoencoders.utils.logging_setup import get_logger
from autoencoders import default_setting

logger = get_logger(__name__, 'INFO')


def export_mnist(dst_dir):
    (x_train_and_valid, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    assert x_train_and_valid.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)

    # further split into train and valid 
    n_valid = 10000
    x_train, x_valid = train_test_split(x_train_and_valid, test_size=n_valid)

    data = {
        'train': x_train[:, :, :, np.newaxis],
        'valid': x_valid[:, :, :, np.newaxis],
        'test': x_test[:, :, :, np.newaxis]
    }

    # export 
    os.makedirs(dst_dir, exist_ok=True)

    for name, arr in data.items():
        # convert to float32 + normalize to [0,1]
        arr = arr.astype(np.float32) / 255 
        
        # export 
        dst_path = join(dst_dir, name)
        np.save(dst_path, arr)

        logger.info(f'''Exported {name} data to {dst_dir}
            shape: {arr.shape}
            dtype: {arr.dtype}
            ''' )

def export_all_datasets():
    """
    Export mnist and synth data (circles) as numpy arrays. 
    """
    dst_dir = join(default_setting.data_dir, 'datasets/mnist')
    export_mnist(dst_dir)

    dst_dir = join(default_setting.data_dir, 'datasets/circles')
    generate_and_export_synth_data(dst_dir)


if __name__ == '__main__':
    export_all_datasets()


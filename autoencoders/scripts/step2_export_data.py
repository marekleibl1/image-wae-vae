
from matplotlib import pyplot as plt
import numpy as np
import argparse
import os
from tqdm import tqdm

from autoencoders.default_paths import default_dataset_path, default_model_dir
from autoencoders.model.conv_ae import ConvAE
from autoencoders.utils.utils import load_dataset
from autoencoders.utils.logging_setup import get_logger

plt.style.use('dark_background')
logger = get_logger(__name__, 'INFO')

parser = argparse.ArgumentParser(description="Hyper-parameters for ConvAE")
parser.add_argument('--dataset_name', type=str, default=None, help="Dataset name (mnist or circles)")


def export_latent_images(dataset_name):

    # load model
    model_dir = default_model_dir(dataset_name, 'convAE')
    model = ConvAE.load_model(model_dir)

    # load data
    dataset = load_dataset(dataset_name)

    max_chunk_size = 128

    logger.info(f'Exporting {dataset_name} latent images ...')

    for split_name, arr in dataset.items():
        # split into chunks
        nchunks = np.ceil(arr.shape[0] / max_chunk_size)
        chunks = np.array_split(arr, nchunks)

        # encode into lower resolution images of shape [W, H, nfeatures]
        encoded_chunks = [model.encode(chunk) for chunk in tqdm(chunks, split_name)]
        zimages = np.concatenate(encoded_chunks, axis=0)

        # export 
        dst_path = default_dataset_path(dataset_name + '_zimages', split_name)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        np.save(dst_path, zimages)
        logger.info(f'Exported to {dst_path}')

    logger.info(f'Exporting {dataset_name} latent images - done')


if __name__ == '__main__':
    args = parser.parse_args()
    
    # Dataset: mnist or circles
    dataset_name = getattr(args.__dict__, 'dataset_name', 'mnist')

    export_latent_images(dataset_name)

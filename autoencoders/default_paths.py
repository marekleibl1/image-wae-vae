"""
For convenience we define most of paths here so that we don't need to pass them to Python scripts.
"""

import os
from os.path import join, abspath

project_dir = abspath(join(__file__, os.pardir, os.pardir))

data_dir = join(project_dir, 'data')
models_dir = join(data_dir, 'models')
datasets_dir = join(data_dir, 'datasets')


def default_model_dir(dataset_name, model_type, suffix=None):
    assert model_type in {'convAE', 'VAE', 'WAE', 'Vanilla'}

    model_name = model_type if suffix is None else model_type + '_' + suffix 
    return join(models_dir, dataset_name, model_name)


def default_dataset_path(dataset_name, split_name):
    return join(datasets_dir, dataset_name, split_name + '.npy')

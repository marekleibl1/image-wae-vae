"""
For convenience we define most of default 
hyper-parameters and paths here. 
"""

from os.path import join, abspath

# Default hyper parameters

# convolutional autoencoder
# circles_cnn_hparams = {
#     'input_shape': [16, 16],
#     'learning_rate': 1e-4,
#     'batch_size': 128,  # 128,
#     'l2_weight': 1e-6,
#     'nbasefilters': 64,  # 32,
#     'zdim': 20,
#     'ntraining_steps': int(10 ** 5)
# }

circles_cnn_hparams = {
    'input_shape': [24, 24],
    'learning_rate': 1e-4,
    'batch_size': 128,  
    'l2_weight': 1e-6,
    'nbasefilters': 32,  
    'zdim': 8,
    'ntraining_steps': 100000 # 40000
}


mnist_cnn_hparams = {
    'input_shape': [28, 28],
    'learning_rate': 1e-4,
    'batch_size': 128,  # 128,
    'l2_weight': 1e-6,
    'nbasefilters': 32, 
    'zdim': 10,
    'ntraining_steps': 100000 # 40000
}

cnn_hparams_dict = dict(
    circles=circles_cnn_hparams,
    mnist=mnist_cnn_hparams, 
)


# default hyper-parameters for stacked autoencoder (Vanilla, VAE or WAE)

mnist_sae_hparams = {
    'input_shape': [7, 7, 10], # 7x7 images with 10 channels 
    'batch_size': 128, 
    'training_steps': 200000, 
    'nfeatures': 512, 
    'z_dim': 5, 
    'w_l2_reg': 1e-6, # l2 regularization
    'learning_rate': 1e-4, 
    'wae_pretrain_maxsteps': 200, 
    'wae_pretrain_stop_at_loss' : 0.1, 
    # weight of mmd loss  
    # - different datasets and zdims may require different values
    # - higher value usually leads to higher reconstruction error, but nicer latent structure
    # - tested with values from range [0.001, 1]
    'wae_w_mmd': 0.3  # this should give roughly similar reconstruction error to VAE
}

circles_sae_hparams = mnist_cnn_hparams.copy()

sae_hparams_dict = dict(
    circles=circles_sae_hparams,
    mnist=mnist_sae_hparams, 
)


def get_default_hparams(dataset_name, model_type):
    # convAE
    if model_type == 'convAE':
        return cnn_hparams_dict[dataset_name]

    # stackedAE
    return sae_hparams_dict[dataset_name]


def fill_hps_with_defaults(dataset_name, model_type, custom_hps={}):
    """
    Return all required hyper-parameters: 
    - use specified custom hyper-parameters,
    - fill the rest from default values for the given dataset and model type.  

    Hyper-parameters not relavant to the given model and None values are ignored. 
    """

    #  get default hps 
    default_hparams = get_default_hparams(dataset_name, model_type)

    # replace values with the given custom values 
    hparams = default_hparams.copy()

    for name, val in custom_hps.items():
        if name in hparams and val is not None:
            hparams[name] = val

    return hparams



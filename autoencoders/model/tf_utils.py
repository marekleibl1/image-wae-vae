import numpy as np


def conv2d_trans_weights(in_size, out_size, kernel_size=4, stride=2):
    """
    Initialize with bilinear interpolations (for kernel size 4 and stride 2).

    Based on this code https://github.com/MarvinTeichmann/tensorflow-fcn/blob/master/fcn8_vgg.py
    """
    # Xavier initialization
    n_inputs = in_size * (kernel_size / stride) ** 2
    stddev = np.sqrt(2 / n_inputs)

    # bilinear filter
    f_bilinear = np.ones([kernel_size, kernel_size, 1, 1], dtype=np.float32)

    w = np.random.randn(*[1, 1, out_size, in_size]) * stddev

    # shape [height, width, output_channels, in_channels]
    W = w * f_bilinear
    return W

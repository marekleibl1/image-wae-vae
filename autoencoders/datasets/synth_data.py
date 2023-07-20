import numpy as np
from numpy.random import rand
from os.path import join 
import os

from autoencoders.utils.logging_setup import get_logger
from autoencoders import default_setting

logger = get_logger(__name__, 'INFO')


def random_circle(im_shape=[16, 16]):
    """
    Generate an image with a circle of random position, radius and thickness.
    """

    xsize, ysize = im_shape

    # --- generate random params

    # center of the circle
    sx = xsize / 2 + (xsize / 4) * (2 * rand() - 1)
    sy = ysize / 2 + (ysize / 4) * (2 * rand() - 1)

    # radius
    msize = min(xsize, ysize)
    r = (msize / 4) + (msize / 4) * rand()

    # relative thickness of the circle
    b = 0.1 + 0.4 * rand()
    assert 0 < b < 1

    # --- create circle image

    # 2d grid
    xx, yy = np.meshgrid(np.arange(xsize), np.arange(ysize))

    # positive inside the circle, negative outside
    dd = (xx - sx) ** 2 + (yy - sy) ** 2 - r ** 2
    dd2 = (xx - sx) ** 2 + (yy - sy) ** 2 - (r * (1 - b)) ** 2

    # "soft" indicator of whether the pixel is inside innet/outer circle
    inside_outer = 1 / (1 + np.exp(-dd))
    inside_inner = 1 / (1 + np.exp(-dd2))

    # the final image: close to 1 between inner and outer circle, otherwise close to 0
    im = inside_inner - inside_outer 

    return im.astype(np.float32)


def plot_circles():
    from matplotlib import pyplot as plt
    plt.style.use('dark_background')

    while True:
        circle = random_circle()

        plt.imshow(circle, cmap='gray')
        plt.show()


def _generate_and_export(dst_dir, n_samples, name):
    samples = [random_circle() for _ in range(n_samples)]
    samples = np.stack(samples, axis=0)[:, :, :, np.newaxis]

    # export 
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = join(dst_dir, name)
    np.save(dst_path, samples)

    logger.info(f'''Exported {name} data to {dst_dir}
    shape: {samples.shape}
    dtype: {samples.dtype}
    ''' )

def generate_and_export_synth_data(dst_dir):
    """
    Generate random circles and export as a single numpy array
    for training, validation and test set. 
    """

    # the same as mnist 
    n_samples = {
        'train': 50000,
        'valid': 10000,
        'test': 10000    
    }

    for name, n in n_samples.items():
        _generate_and_export(dst_dir, n, name)
    

if __name__ == '__main__':
    dst_dir = join(default_setting.data_dir, 'datasets/circles')
    generate_and_export_synth_data(dst_dir)

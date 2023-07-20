import sys
import time
import os
import numpy as np
import tensorflow as tf
import logging


def sample_pz(zdim, num):
    mean, cov = np.zeros(zdim), np.identity(zdim)
    sample = np.random.multivariate_normal(mean, cov, num).astype(np.float32)
    return sample


def pretrain_loss(encoded, sample_noise):
    # TODO can we just minimize mmd loss during the pretraining? -> try both
    e_pretrain_sample_size = None

    # Adding ops to pretrain the encoder so that mean and covariance
    # of Qz will try to match those of Pz
    mean_pz = tf.reduce_mean(sample_noise, axis=0, keep_dims=True)
    mean_qz = tf.reduce_mean(encoded, axis=0, keep_dims=True)
    mean_loss = tf.reduce_mean(tf.square(mean_pz - mean_qz))
    cov_pz = tf.matmul(sample_noise - mean_pz, sample_noise - mean_pz, transpose_a=True)
    cov_pz /= e_pretrain_sample_size - 1.
    cov_qz = tf.matmul(encoded - mean_qz, encoded - mean_qz, transpose_a=True)
    cov_qz /= e_pretrain_sample_size - 1.
    cov_loss = tf.reduce_mean(tf.square(cov_pz - cov_qz))
    return mean_loss + cov_loss


def pretrain_encoder(self, data):
    opts = self.opts
    steps_max = 200
    batch_size = opts['e_pretrain_sample_size']

    for step in range(steps_max):
        train_size = data.num_points
        data_ids = np.random.choice(train_size, min(train_size, batch_size),
                                    replace=False)
        batch_images = data.data[data_ids].astype(np.float)
        batch_noise = self.sample_pz(batch_size)

        [_, loss_pretrain] = self.sess.run([self.pretrain_opt, self.loss_pretrain],
                                           feed_dict={self.sample_points: batch_images,
                                                      self.sample_noise: batch_noise,
                                                      self.is_training: True})

        if opts['verbose']:
            logging.error('Step %d/%d, loss=%f' % (step, steps_max, loss_pretrain))

        if loss_pretrain < 0.1:
            break


def mmd_penalty(sample_qz, sample_pz):
    # batch size
    n = sample_qz.shape[0]
    n, nf = tf.cast(n, tf.int32), tf.cast(n, tf.float32)
    half_size = (n * n - n) // 2

    norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keepdims=True)
    dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
    distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz

    norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keepdims=True)
    dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
    distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

    dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
    distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods

    # --- RBF Kernel

    # Median heuristic for the sigma^2 of Gaussian kernel
    sigma2_k = tf.nn.top_k(tf.reshape(distances, [-1]), half_size).values[half_size - 1]
    sigma2_k += tf.nn.top_k(tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
    # Maximal heuristic for the sigma^2 of Gaussian kernel
    # sigma2_k = tf.nn.top_k(tf.reshape(distances_qz, [-1]), 1).values[0]
    # sigma2_k += tf.nn.top_k(tf.reshape(distances, [-1]), 1).values[0]
    # sigma2_k = opts['latent_space_dim'] * sigma2_p
    res1 = tf.exp(- distances_qz / 2. / sigma2_k)
    res1 += tf.exp(- distances_pz / 2. / sigma2_k)
    res1 = tf.multiply(res1, 1. - tf.eye(n))
    res1 = tf.reduce_sum(res1) / (nf * nf - nf)
    res2 = tf.exp(- distances / 2. / sigma2_k)
    res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
    stat = res1 - res2

    return stat

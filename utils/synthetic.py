"""
The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import scipy as scp

import logging

import skimage
import skimage.transform

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def np_onehot(label, num_classes):
    return np.eye(num_classes)[label]


def augment_label(label, num_classes, scale=8, keep_prop=0.8):
    """
    Add noise to label for synthetic benchmark.
    """

    shape = label.shape
    label = label.reshape(shape[0], shape[1])

    onehot = np_onehot(label, num_classes)
    lower_shape = (shape[0] // scale, shape[1] // scale)

    label_down = skimage.transform.resize(
        onehot, (lower_shape[0], lower_shape[1], num_classes),
        order=1, preserve_range=True, mode='constant')

    onehot = skimage.transform.resize(label_down,
                                      (shape[0], shape[1], num_classes),
                                      order=1, preserve_range=True,
                                      mode='constant')

    noise = np.random.randint(0, num_classes, lower_shape)

    noise = np_onehot(noise, num_classes)

    noise_up = skimage.transform.resize(noise,
                                        (shape[0], shape[1], num_classes),
                                        order=1, preserve_range=True,
                                        mode='constant')

    mask = np.floor(keep_prop + np.random.rand(*lower_shape))
    mask_up = skimage.transform.resize(mask, (shape[0], shape[1], 1),
                                       order=1, preserve_range=True,
                                       mode='constant')

    noised_label = mask_up * onehot + (1 - mask_up) * noise_up

    return noised_label


if __name__ == '__main__':
    logging.info("Hello World.")

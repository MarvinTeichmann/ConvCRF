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
import math

import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral
from pydensecrf.utils import create_pairwise_gaussian

import torch
import torch.nn as nn
from torch.nn import functional as nnfun
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import gc

from pydensecrf import densecrf as dcrf
from pydensecrf import utils


default_conf = {
    'blur': 4,
    'merge': False,
    'norm': 'none',
    'trainable': False,
    'weight': 'scalar',
    'weight_init': 0.2,
    'convcomp': False,

    'pos_feats': {
        'sdims': 3,
        'compat': 3,
    },
    'col_feats': {
        'sdims': 80,
        'schan': 13,
        'compat': 10,
        'use_bias': False
    },
    "trainable_bias": False,
}


test_config = {
    'filter_size': 5,
    'blur': 1,
    'merge': False,
    'norm': 'sym',
    'trainable': False,
    'weight': None,
    'weight_init': 5,
    'convcomp': False,

    'pos_feats': {
        'sdims': 3,
        'compat': 3,
    },

    'col_feats': {
        'sdims': 80,
        'schan': 13,
        'compat': 10,
        'use_bias': True
    },
    "trainable_bias": False,
}


class FullCRF():

    def __init__(self, conf, shape, num_classes=None):
        self.crf = None
        self.conf = conf
        self.num_classes = num_classes
        self.shape = shape

    def compute_lattice(self, img, num_classes=None):

        if num_classes is not None:
            self.num_classes = num_classes

        assert self.num_classes is not None

        npixels = self.shape[0] * self.shape[1]
        crf = dcrf.DenseCRF(npixels, self.num_classes)

        sdims = self.conf['pos_feats']['sdims']

        feats = utils.create_pairwise_gaussian(
            sdims=(sdims, sdims),
            shape=img.shape[:2])

        self.smooth_feats = feats

        sdims = self.conf['col_feats']['sdims']
        schan = self.conf['col_feats']['schan']

        feats = utils.create_pairwise_bilateral(sdims=(sdims, sdims),
                                                schan=(schan, schan, schan),
                                                img=img, chdim=2)

        self.appear_feats = feats

        self.crf = crf

    def compute_dcrf(self, unary):

        self.crf.addPairwiseEnergy(
            self.smooth_feats, compat=self.conf['pos_feats']['compat'])

        self.crf.addPairwiseEnergy(
            self.appear_feats, compat=self.conf['pos_feats']['compat'])

        eps = 1e-20
        unary = unary + eps
        unary = unary.reshape(-1, self.num_classes)
        unary = np.transpose(unary)
        unary = np.ascontiguousarray(unary, dtype=np.float32)
        self.crf.setUnaryEnergy(-np.log(unary))

        # Run five inference steps.
        crfout = self.crf.inference(5)
        crfout = np.transpose(crfout)
        crfout = crfout.reshape(self.shape[0], self.shape[1], -1)

        return crfout

    def compute(self, unary, img, softmax=False):
        if softmax:
            unary = torch.nn.functional.softmax(
                Variable(torch.Tensor(unary)), dim=2)
            unary = unary.data.numpy()
        self.create_gaussians(img)
        return self.compute_dcrf(unary)

    def batched_compute(self, unary, img, softmax=False):
        img = img.data.cpu().numpy()
        unary = unary.data.cpu().numpy()

        img = img.transpose(0, 2, 3, 1)
        unary = unary.transpose(0, 2, 3, 1)

        results = []

        for d in range(img.shape[0]):
            img_d = img[d]
            unary_d = unary[d]
            res = self.compute(unary_d, img_d, softmax)
            results.append(res)

        return results

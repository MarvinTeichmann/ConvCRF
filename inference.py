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
import scipy.misc

import argparse

import logging

from convcrf import convcrf

import torch
from torch.autograd import Variable

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def do_crf_inference(image, unary):

    # get basic hyperparameters
    num_classes = unary.shape[2]
    shape = image.shape[0:2]
    config = convcrf.default_conf

    ##
    # make input pytorch compatible
    image = image.transpose(2, 0, 1)  # shape: [3, hight, width]
    # Add batch dimension to image: [1, 3, height, width]
    image = image.reshape([1, 3, shape[0], shape[1]])
    img_var = Variable(torch.Tensor(image), volatile=True).cuda()

    unary = unary.transpose(2, 0, 1)  # shape: [3, hight, width]
    # Add batch dimension to unary: [1, 21, height, width]
    unary = unary.reshape([1, num_classes, shape[0], shape[1]])
    unary_var = Variable(torch.Tensor(unary), volatile=True).cuda()

    ##
    # Create CRF module
    gausscrf = convcrf.GaussCRF(conf=config, shape=shape, nclasses=num_classes)
    # Cuda computation is required.
    # A CPU implementation of our message passing is not provided.
    gausscrf.cuda()

    # Perform CRF inference
    prediction = gausscrf.forward(unary=unary_var, img=img_var)

    return prediction


def plot_results():
    return


def get_parser():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("image", type=str,
                        help="input image")

    parser.add_argument("unary", type=str,
                        help="unary for input")

    parser.add_argument("label", type=str, nargs='?',
                        help="Label file (optional).")

    parser.add_argument("--gpu", type=str, default='0',
                        help="which gpu to use")

    parser.add_argument('--output', type=str,
                        help="Optionally save output as img.")

    # parser.add_argument('--compare', action='store_true')
    # parser.add_argument('--embed', action='store_true')

    # args = parser.parse_args()

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    image = scp.misc.imread(args.image)
    unary = np.load(args.unary)['arr_0']

    prediction = do_crf_inference(image, unary)
    plot_results(args, image, unary, prediction)
    logging.info("Thank you for trying ConvCRFs.")

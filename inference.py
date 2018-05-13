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
import imageio
# import scipy as scp
# import scipy.misc

import argparse

import logging
import time

from convcrf import convcrf

import torch
from torch.autograd import Variable

from utils import pascal_visualizer as vis

try:
    import matplotlib.pyplot as plt
    figure = plt.figure()
    matplotlib = True
    plt.close(figure)
except:
    matplotlib = False
    pass

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def do_crf_inference(image, unary, speed_test, pyinn=False):

    if pyinn or not hasattr(torch.nn.functional, 'unfold'):
        # pytorch 0.3 or older requires pyinn.
        pyinn = True
        # Cheap and easy trick to make sure that pyinn is loadable.
        import pyinn

    # get basic hyperparameters
    num_classes = unary.shape[2]
    shape = image.shape[0:2]
    config = convcrf.default_conf
    config['filter_size'] = 7
    config['pyinn'] = pyinn

    ##
    # make input pytorch compatible
    image = image.transpose(2, 0, 1)  # shape: [3, hight, width]
    # Add batch dimension to image: [1, 3, height, width]
    image = image.reshape([1, 3, shape[0], shape[1]])
    img_var = Variable(torch.Tensor(image)).cuda()

    unary = unary.transpose(2, 0, 1)  # shape: [3, hight, width]
    # Add batch dimension to unary: [1, 21, height, width]
    unary = unary.reshape([1, num_classes, shape[0], shape[1]])
    unary_var = Variable(torch.Tensor(unary)).cuda()

    logging.info("Build ConvCRF.")
    ##
    # Create CRF module
    gausscrf = convcrf.GaussCRF(conf=config, shape=shape, nclasses=num_classes)
    # Cuda computation is required.
    # A CPU implementation of our message passing is not provided.
    gausscrf.cuda()

    logging.info("Start Computation.")
    # Perform CRF inference
    prediction = gausscrf.forward(unary=unary_var, img=img_var)

    if speed_test:
        # Evaluate inference speed
        logging.info("Doing speed evaluation.")
        start_time = time.time()
        for i in range(10):
            # Running ConvCRF 10 times and average total time
            pred = gausscrf.forward(unary=unary_var, img=img_var)

        pred.cpu()  # wait for all GPU computations to finish

        duration = (time.time() - start_time) * 1000 / 10

        logging.info("Finished running 10 predictions.")
        logging.info("Avg. Computation time: {} ms".format(duration))

    return prediction.data.cpu().numpy()


def plot_results(image, unary, prediction, label, args):

    logging.info("Plot results.")

    # Create visualizer
    myvis = vis.PascalVisualizer()

    if label is not None:
        # Transform id image to coloured labels
        coloured_label = myvis.id2color(id_image=label)
        # Plot parameters
        num_rows = 2
        num_cols = 2
        off = 0
    else:
        # Plot parameters
        num_cols = 3
        num_rows = 1
        off = 1

    unary_hard = np.argmax(unary, axis=2)
    coloured_unary = myvis.id2color(id_image=unary_hard)

    prediction = prediction[0]  # Remove Batch dimension
    prediction_hard = np.argmax(prediction, axis=0)
    coloured_crf = myvis.id2color(id_image=prediction_hard)

    if matplotlib:
        # Plot results using matplotlib
        figure = plt.figure()
        figure.tight_layout()

        ax = figure.add_subplot(num_rows, num_cols, 1)
        # img_name = os.path.basename(args.image)
        ax.set_title('Image ')
        ax.axis('off')
        ax.imshow(image)

        ax = figure.add_subplot(num_rows, num_cols, 2)
        ax.set_title('Label')
        ax.axis('off')
        ax.imshow(coloured_label.astype(np.uint8))

        ax = figure.add_subplot(num_rows, num_cols, 3 - off)
        ax.set_title('Unary')
        ax.axis('off')
        ax.imshow(coloured_unary.astype(np.uint8))

        ax = figure.add_subplot(num_rows, num_cols, 4 - off)
        ax.set_title('CRF Output')
        ax.axis('off')
        ax.imshow(coloured_crf.astype(np.uint8))

        plt.show()
    else:
        if args.output is None:
            args.output = "out.png"

        logging.warning("Matplotlib not found.")
        logging.info("Saving output to {} instead".format(args.output))

    if args.output is not None:
        # Save results to disk
        if label is not None:
            out_img = np.concatenate(
                (image, coloured_label, coloured_unary, coloured_crf),
                axis=1)
        else:
            out_img = np.concatenate(
                (image, coloured_unary, coloured_crf),
                axis=1)

        imageio.imwrite(args.output, out_img.astype(np.uint8))

        logging.info("Plot has been saved to {}".format(args.output))

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
                        help="Label file (Optional: Used for plotting only"
                        ". Recommended).")

    parser.add_argument("--gpu", type=str, default='0',
                        help="which gpu to use")

    parser.add_argument('--output', type=str,
                        help="Optionally save output as img.")

    parser.add_argument('--speed_test', action='store_true',
                        help="Evaluate and print inference speed.")

    parser.add_argument('--pyinn', action='store_true',
                        help="Use pyinn based Cuda implementation"
                             "for message passing.")

    # parser.add_argument('--compare', action='store_true')
    # parser.add_argument('--embed', action='store_true')

    # args = parser.parse_args()

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    logging.info("Load and uncompress data.")

    image = imageio.imread(args.image)
    unary = np.load(args.unary)['arr_0']
    if args.label is not None:
        label = imageio.imread(args.label)
    else:
        label = args.labels

    prediction = do_crf_inference(image, unary, args.speed_test, args.pyinn)
    plot_results(image, unary, prediction, label, args)
    logging.info("Thank you for trying ConvCRFs.")

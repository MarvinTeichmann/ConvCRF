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

from convcrf import convcrf
from fullcrf import fullcrf

import torch
from torch.autograd import Variable

from utils import pascal_visualizer as vis

import time

try:
    import matplotlib.pyplot as plt
    matplotlib = True
    figure = plt.figure()
    plt.close(figure)
except:
    matplotlib = False
    pass

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def do_crf_inference(image, unary):

    # get basic hyperparameters
    num_classes = unary.shape[2]
    shape = image.shape[0:2]
    config = convcrf.default_conf
    config['filter_size'] = 11

    logging.info("Doing speed benchmark with filter size: {}"
                 .format(config['filter_size']))

    ##
    # make input pytorch compatible
    img = image.transpose(2, 0, 1)  # shape: [3, hight, width]
    # Add batch dimension to image: [1, 3, height, width]
    img = img.reshape([1, 3, shape[0], shape[1]])
    img_var = Variable(torch.Tensor(img)).cuda()

    un = unary.transpose(2, 0, 1)  # shape: [3, hight, width]
    # Add batch dimension to unary: [1, 21, height, width]
    un = un.reshape([1, num_classes, shape[0], shape[1]])
    unary_var = Variable(torch.Tensor(un)).cuda()

    logging.debug("Build ConvCRF.")
    ##
    # Create CRF module
    gausscrf = convcrf.GaussCRF(conf=config, shape=shape, nclasses=num_classes)
    # Cuda computation is required.
    # A CPU implementation of our message passing is not provided.
    gausscrf.cuda()

    # Perform CRF inference
    """
    'Warm up': Our implementation compiles cuda kernels during runtime.
    The first inference call thus comes with some overhead.
    """
    prediction = gausscrf.forward(unary=unary_var, img=img_var)

    logging.info("Start Computation.")
    logging.info("Running multiple iteration. This may take a while.")
    start_time = time.time()

    for i in range(10):
        # Running ConvCRF 10 times and report average total time
        prediction = gausscrf.forward(unary=unary_var, img=img_var)

    prediction.cpu()  # wait for all GPU computations to finish

    duration = (time.time() - start_time) * 1000 / 10

    logging.debug("Finished running 10 predictions.")
    logging.debug("Avg Computation time: {} ms".format(duration))

    myfullcrf = fullcrf.FullCRF(config, shape, num_classes)

    # Initialize permutohedral lattice with image features
    # myfullcrf.compute_lattice(image)
    """
    Computing the lattice is actually part of the processing time.
    However, in our implementation the features are generated in python,
    making it an unfairly slow step.
    """

    for i in range(3):
        #  'Warm up'
        fullprediction = myfullcrf.compute(unary, image, softmax=False)

    start_time = time.time()
    for i in range(5):
        # Running FullCRF 5 times and report average total time
        fullprediction = myfullcrf.compute(unary, image, softmax=False)

    fullduration = (time.time() - start_time) * 1000 / 5

    logging.debug("Finished running 2 predictions.")
    logging.debug("Avg Computation time: {} ms".format(fullduration))

    logging.info("Using FullCRF took {:4.0f} ms ({:2.2f} s)".format(
        fullduration, fullduration / 1000))

    logging.info("Using ConvCRF took {:4.0f} ms ({:2.2f} s)".format(
        duration, duration / 1000))

    logging.info("Congratulation. Using ConvCRF provids a speed-up"
                 " of {:.0f}.".format(fullduration / duration))

    logging.info("")

    return prediction.data.cpu().numpy(), fullprediction


def plot_results(image, unary, conv_out, full_out, label, args):

    logging.debug("Plot results.")

    # Create visualizer
    myvis = vis.PascalVisualizer()

    if label is not None:
        # Transform id image to coloured labels
        coloured_label = myvis.id2color(id_image=label)
        # Plot parameters
        num_rows = 1
        num_cols = 5
        off = 0
    else:
        # Plot parameters
        num_cols = 2
        num_rows = 2
        off = 1

    unary_hard = np.argmax(unary, axis=2)
    coloured_unary = myvis.id2color(id_image=unary_hard)

    conv_out = conv_out[0]  # Remove Batch dimension
    conv_hard = np.argmax(conv_out, axis=0)
    coloured_conv = myvis.id2color(id_image=conv_hard)

    full_hard = np.argmax(full_out, axis=2)
    coloured_full = myvis.id2color(id_image=full_hard)

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
        ax.set_title('ConvCRF Output')
        ax.axis('off')
        ax.imshow(coloured_conv.astype(np.uint8))

        ax = figure.add_subplot(num_rows, num_cols, 5 - off)
        ax.set_title('FullCRF Output')
        ax.axis('off')
        ax.imshow(coloured_full.astype(np.uint8))

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
                (image, coloured_label, coloured_unary, coloured_conv),
                axis=1)
        else:
            out_img = np.concatenate(
                (image, coloured_unary, coloured_conv),
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

    # parser.add_argument('--compare', action='store_true')
    # parser.add_argument('--embed', action='store_true')

    # args = parser.parse_args()

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    logging.debug("Load and uncompress data.")

    image = imageio.imread(args.image)
    unary = np.load(args.unary)['arr_0']
    if args.label is not None:
        label = imageio.imread(args.label)
    else:
        label = args.label

    conv_out, full_out = do_crf_inference(image, unary)
    plot_results(image, unary, conv_out, full_out, label, args)
    logging.info("Thank you for trying ConvCRFs.")

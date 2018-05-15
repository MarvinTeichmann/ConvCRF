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
from utils import synthetic

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


def do_crf_inference(image, unary, args):

    if args.pyinn or not hasattr(torch.nn.functional, 'unfold'):
        # pytorch 0.3 or older requires pyinn.
        args.pyinn = True
        # Cheap and easy trick to make sure that pyinn is loadable.
        import pyinn

    # get basic hyperparameters
    num_classes = unary.shape[2]
    shape = image.shape[0:2]
    config = convcrf.default_conf
    config['filter_size'] = 7
    config['pyinn'] = args.pyinn

    if args.normalize:
        # Warning, applying image normalization affects CRF computation.
        # The parameter 'col_feats::schan' needs to be adapted.

        # Normalize image range
        #     This changes the image features and influences CRF output
        image = image / 255
        # mean substraction
        #    CRF is invariant to mean subtraction, output is NOT affected
        image = image - 0.5
        # std normalization
        #       Affect CRF computation
        image = image / 0.3

        # schan = 0.1 is a good starting value for normalized images.
        # The relation is f_i = image / schan
        config['col_feats']['schan'] = 0.1

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

    # Perform ConvCRF inference
    """
    'Warm up': Our implementation compiles cuda kernels during runtime.
    The first inference call thus comes with some overhead.
    """
    logging.info("Start Computation.")
    prediction = gausscrf.forward(unary=unary_var, img=img_var)

    if args.nospeed:

        logging.info("Doing speed benchmark with filter size: {}"
                     .format(config['filter_size']))
        logging.info("Running multiple iteration. This may take a while.")

        # Our implementation compiles cuda kernels during runtime.
        # The first inference run is those much slower.
        # prediction = gausscrf.forward(unary=unary_var, img=img_var)

        start_time = time.time()
        for i in range(10):
            # Running ConvCRF 10 times and report average total time
            prediction = gausscrf.forward(unary=unary_var, img=img_var)

        prediction.cpu()  # wait for all GPU computations to finish
        duration = (time.time() - start_time) * 1000 / 10

        logging.debug("Finished running 10 predictions.")
        logging.debug("Avg Computation time: {} ms".format(duration))

    # Perform FullCRF inference
    myfullcrf = fullcrf.FullCRF(config, shape, num_classes)
    fullprediction = myfullcrf.compute(unary, image, softmax=False)

    if args.nospeed:

        start_time = time.time()
        for i in range(5):
            # Running FullCRF 5 times and report average total time
            fullprediction = myfullcrf.compute(unary, image, softmax=False)

        fullduration = (time.time() - start_time) * 1000 / 5

        logging.debug("Finished running 5 predictions.")
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

    # Transform id image to coloured labels
    coloured_label = myvis.id2color(id_image=label)

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
        # Plot parameters
        num_rows = 2
        num_cols = 3
        off = 0

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

        # plt.subplots_adjust(left=0.02, right=0.98,
        #                    wspace=0.15, hspace=0.15)

        plt.show()
    else:
        if args.output is None:
            args.output = "out.png"

        logging.warning("Matplotlib not found.")
        logging.info("Saving output to {} instead".format(args.output))

    if args.output is not None:
        # Save results to disk
        out_img = np.concatenate(
            (image, coloured_label, coloured_unary, coloured_conv),
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

    parser.add_argument("label", type=str,
                        help="Label file.")

    parser.add_argument("--gpu", type=str, default='0',
                        help="which gpu to use")

    parser.add_argument('--output', type=str,
                        help="Optionally save output as img.")

    parser.add_argument('--nospeed', action='store_false',
                        help="Skip speed evaluation.")

    parser.add_argument('--normalize', action='store_true',
                        help="Normalize input image before inference.")

    parser.add_argument('--pyinn', action='store_true',
                        help="Use pyinn based Cuda implementation"
                             "for message passing.")

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # Load data
    image = imageio.imread(args.image)
    label = imageio.imread(args.label)

    # Produce unary by adding noise to label
    unary = synthetic.augment_label(label, num_classes=21)
    # Compute CRF inference

    conv_out, full_out = do_crf_inference(image, unary, args)
    plot_results(image, unary, conv_out, full_out, label, args)
    logging.info("Thank you for trying ConvCRFs.")

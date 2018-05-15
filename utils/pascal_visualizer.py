import os
import collections
from collections import OrderedDict
import json
import logging
import sys
import random

import numpy as np
import scipy as scp
import scipy.misc

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

from . import visualization as vis

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

voc_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
             'bottle', 'bus', 'car', 'cat',
             'chair', 'cow', 'diningtable', 'dog',
             'horse', 'motorbike', 'person', 'potted-plant',
             'sheep', 'sofa', 'train', 'tv/monitor']

color_list = [(0, 0, 0),
              (128, 0, 0),
              (0, 128, 0),
              (128, 128, 0),
              (0, 0, 128),
              (128, 0, 128),
              (0, 128, 128),
              (128, 128, 128),
              (64, 0, 0),
              (192, 0, 0),
              (64, 128, 0),
              (192, 128, 0),
              (64, 0, 128),
              (192, 0, 128),
              (64, 128, 128),
              (192, 128, 128),
              (0, 64, 0),
              (128, 64, 0),
              (0, 192, 0),
              (128, 192, 0),
              (0, 64, 128)]


class PascalVisualizer(vis.SegmentationVisualizer):

    def __init__(self):
        super(PascalVisualizer, self).__init__(
            color_list=color_list, name_list=voc_names)

    def plot_sample(self, sample):

        image = sample['image'].transpose(1, 2, 0)
        label = sample['label']
        mask = label != -100

        idx = eval(sample['load_dict'])['idx']

        coloured_label = self.id2color(id_image=label,
                                       mask=mask)

        figure = plt.figure()
        figure.tight_layout()

        ax = figure.add_subplot(1, 2, 1)
        ax.set_title('Image #{}'.format(idx))
        ax.axis('off')
        ax.imshow(image)

        ax = figure.add_subplot(1, 2, 2)
        ax.set_title('Label')
        ax.axis('off')
        ax.imshow(coloured_label.astype(np.uint8))

        return figure

    def plot_segmentation_batch(self, sample_batch, prediction):
        figure = plt.figure()
        figure.tight_layout()

        batch_size = len(sample_batch['load_dict'])
        figure.set_size_inches(12, 3 * batch_size)

        for d in range(batch_size):
            image = sample_batch['image'][d].numpy().transpose(1, 2, 0)
            label = sample_batch['label'][d].numpy()

            mask = label != -100

            pred = prediction[d].cpu().data.numpy().transpose(1, 2, 0)
            pred_hard = np.argmax(pred, axis=2)

            idx = eval(sample_batch['load_dict'][d])['idx']

            coloured_label = self.id2color(id_image=label,
                                           mask=mask)

            coloured_prediction = self.pred2color(pred_image=pred,
                                                  mask=mask)

            coloured_hard = self.id2color(id_image=pred_hard,
                                          mask=mask)

            ax = figure.add_subplot(batch_size, 4, batch_size * d + 1)
            ax.set_title('Image #{}'.format(idx))
            ax.axis('off')
            ax.imshow(image)

            ax = figure.add_subplot(batch_size, 4, batch_size * d + 2)
            ax.set_title('Label')
            ax.axis('off')
            ax.imshow(coloured_label.astype(np.uint8))

            ax = figure.add_subplot(batch_size, 4, batch_size * d + 3)
            ax.set_title('Prediction (hard)')
            ax.axis('off')
            ax.imshow(coloured_hard.astype(np.uint8))

            ax = figure.add_subplot(batch_size, 4, batch_size * d + 4)
            ax.set_title('Prediction (soft)')
            ax.axis('off')
            ax.imshow(coloured_prediction.astype(np.uint8))

        return figure

    def plot_batch(self, sample_batch):

        figure = plt.figure()
        figure.tight_layout()

        batch_size = len(sample_batch['load_dict'])

        for d in range(batch_size):

            image = sample_batch['image'][d].numpy().transpose(1, 2, 0)
            label = sample_batch['label'][d].numpy()
            mask = label != -100

            idx = eval(sample_batch['load_dict'][d])['idx']

            coloured_label = self.id2color(id_image=label,
                                           mask=mask)

            ax = figure.add_subplot(2, batch_size, d + 1)
            ax.set_title('Image #{}'.format(idx))
            ax.axis('off')
            ax.imshow(image)

            ax = figure.add_subplot(2, batch_size, d + batch_size + 1)
            ax.set_title('Label')
            ax.axis('off')
            ax.imshow(coloured_label.astype(np.uint8))

        return figure

import sys
import scipy
import cv2 
import numpy as np

import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import losses
from tensorflow.keras import layers


import os, sys, argparse
import numpy as np

from math import ceil 
from imgaug import augmenters as iaa

from helper_code import *

# segmentation model


def batch_activate(x):
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def convolution_block(x,
                      filters,
                      size,
                      strides=(1, 1),
                      padding='same',
                      activation=True):
    x = layers.Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation:
        x = batch_activate(x)
    return x

def residual_block(block_input,
                   num_filters=16,
                   use_batch_activate=False):
    x = batch_activate(block_input)
    x = convolution_block(x, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = layers.Add()([x, block_input])
    if use_batch_activate:
        x = batch_activate(x)
    return x


def unet(img_size=(512, 512),
                no_channels=3,
                start_neurons=32,
                dropout_rate=0.25):

    # inner
    input_layer = layers.Input(name='the_input',
                               shape=(*img_size, no_channels),  # noqa
                               dtype='float32')

    # down 1
    conv1 = layers.Conv2D(start_neurons * 1, (3, 3),
                          activation=None, padding="same")(input_layer)
    conv1 = residual_block(conv1, start_neurons * 1)
    conv1 = residual_block(conv1, start_neurons * 1, True)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    pool1 = layers.Dropout(dropout_rate)(pool1)

    # down 2
    conv2 = layers.Conv2D(start_neurons * 2, (3, 3),
                          activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2, start_neurons * 2)
    conv2 = residual_block(conv2, start_neurons * 2, True)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    pool2 = layers.Dropout(dropout_rate)(pool2)

    # down 3
    conv3 = layers.Conv2D(start_neurons * 4, (3, 3),
                          activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3, start_neurons * 4)
    conv3 = residual_block(conv3, start_neurons * 4, True)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    pool3 = layers.Dropout(dropout_rate)(pool3)

    # middle
    middle = layers.Conv2D(start_neurons * 8, (3, 3),
                           activation=None, padding="same")(pool3)
    middle = residual_block(middle, start_neurons * 8)
    middle = residual_block(middle, start_neurons * 8, True)

    # up 1
    deconv3 = layers.Conv2DTranspose(
        start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(middle)
    uconv3 = layers.concatenate([deconv3, conv3])
    uconv3 = layers.Dropout(dropout_rate)(uconv3)

    uconv3 = layers.Conv2D(start_neurons * 4, (3, 3),
                           activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_neurons * 4)
    uconv3 = residual_block(uconv3, start_neurons * 4, True)

    # up 2
    deconv2 = layers.Conv2DTranspose(
        start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = layers.concatenate([deconv2, conv2])
    uconv2 = layers.Dropout(dropout_rate)(uconv2)

    uconv2 = layers.Conv2D(start_neurons * 2, (3, 3),
                           activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_neurons * 2)
    uconv2 = residual_block(uconv2, start_neurons * 2, True)

    # up 3
    deconv1 = layers.Conv2DTranspose(
        start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = layers.concatenate([deconv1, conv1])
    uconv1 = layers.Dropout(dropout_rate)(uconv1)

    uconv1 = layers.Conv2D(start_neurons * 1, (3, 3),
                           activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_neurons * 1)
    uconv1 = residual_block(uconv1, start_neurons * 1, True)

    # output mask
    output_layer = layers.Conv2D(
        2, (1, 1), padding="same", activation=None)(uconv1)

    # 2 classes: character mask & center point mask
    output_layer = layers.Activation('sigmoid')(output_layer)

    model = models.Model(inputs=[input_layer], outputs=output_layer)
    return model


def get_iou_vector(A, B):
    # Numpy version
    batch_size = A.shape[0]
    metric = 0.0
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        true = np.sum(t)
        pred = np.sum(p)

        # deal with empty mask first
        if true == 0:
            metric += (pred == 0)
            continue

        # non empty mask case.  Union is never empty
        # hence it is safe to divide by its number of pixels
        intersection = np.sum(t * p)
        union = true + pred - intersection
        iou = intersection / union

        # iou metrric is a stepwise approximation of the real iou over 0.5
        iou = np.floor(max(0, (iou - 0.45) * 20)) / 10

        metric += iou

    # teake the average over all images in batch
    metric /= batch_size
    return metric


def my_iou_metric(label, pred):
    
    # Tensorflow version
    return tf.py_function(get_iou_vector, [label, pred > 0.5 ], tf.float64)

# loss function
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score

# loss function

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / \
        (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def bce_dice_loss(y_true, y_pred):
    return losses.binary_crossentropy(y_true, y_pred) + \
        dice_loss(y_true, y_pred)


def bce_logdice_loss(y_true, y_pred):
    return losses.binary_crossentropy(y_true, y_pred) - \
        K.log(1. - dice_loss(y_true, y_pred))
        

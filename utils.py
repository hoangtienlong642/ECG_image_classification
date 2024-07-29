import sys
import scipy
import cv2 
import numpy as np
import joblib

from scipy import ndimage
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont


from matplotlib.patches import Rectangle

import torch 
import torchvision

import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import losses



from torchvision.utils import *

from torch.utils.data import Dataset

import os, sys, argparse
import numpy as np
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from math import ceil 
import wfdb
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from helper_code import *
from model import *
from dataset import *



def label_encoder(Labels):
    label_dict = {
        'NORM': 0,
        'Acute MI': 1,
        'Old MI': 2,
        'STTC': 3,
        'CD': 4,
        'HYP': 5,
        'PAC': 6,
        'PVC': 7,
        'AFIB/AFL': 8,
        'TACHY': 9,
        'BRADY': 10
        }
    labels_encoded = np.zeros(len(label_dict), dtype=int)
    for label in Labels:
        if label in label_dict.keys():
            labels_encoded[label_dict[label]] = 1
    return labels_encoded
        


def apply_mask (image, mask):
    img_size = image.shape[0]
    mask = cv2.resize(mask, (img_size, img_size))
    # image = image.astype(np.uint8)
    mask = mask.astype(np.float32)
    # result = cv2.bitwise_and(image,image,mask = mask)
    result = image * np.expand_dims(mask, axis=-1)
    return result




def prepare_data(model, data_dir, img_size, no_chqannels):
    data = SegmentData(data_dir, img_size, no_channels = no_channels)
    for i in range (len(data)):
        X,mask = data[i]
        y = model.predict(X)
        y[y>=0.9] = 255
        y[y<0.9] = 0
        img_name  = data.getimagename(i)
        image = apply_mask(X[0,:,:,:], y[0,:,:,0])
        
        img,extension = os.path.splitext(img_name)
        cv2.imwrite(os.path.join(data_dir,(img+ '_masked' + extension)), image)
    return data_dir 
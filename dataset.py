import sys
import scipy
import cv2 
import numpy as np
import joblib

import tensorflow as tf
import os, sys, argparse
import numpy as np

from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from helper_code import *
from model import *

# bounding box extraction
def lead_bbox_extract(image_name,data_dir):
    image_name = os.path.splitext(image_name)[0]
    dir = os.path.join(data_dir,'lead_bounding_box',image_name)
    dir += '.txt'
    bbs = read_bounding_box_txt(dir)   
    return bbs 

def text_bbox_extract(image_name,data_dir):
    image_name = os.path.splitext(image_name)[0]
    dir = os.path.join(data_dir,'text_bounding_box',image_name)
    dir += '.txt'
    bbs = read_bounding_box_txt(dir)
    return bbs
    
def read_bounding_box_txt(filename):
    bbs = []

    with open(filename, 'r') as text_file:
        lines = text_file.readlines()
    
    for i, line in enumerate(lines):
        line = line.split('\n')[0]
        
        parts = line.split(',')
        x1 = float(parts[0])
        y1 = float(parts[1])
        x2 = float(parts[2])
        y2 = float(parts[3])
        try:
           label = float(parts[4])
        except ValueError:
            label = str(parts[4])

        box = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label)
        bbs.append(box)

    return bbs


def padding(img): #padding a cv2 image to square 
    """https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    """
    height, width = img.shape[:2]
    desired_size = max(height, width)

    ratio = float(desired_size) / max(img.shape)
    new_size = tuple([int(dim * ratio) for dim in img.shape[:2]])

    # resize img
    rimg = cv2.resize(img, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # make padding
    color = [0, 0, 0]
    rimg = cv2.copyMakeBorder(rimg, top, bottom, left,
                              right, cv2.BORDER_CONSTANT, value=color)

    return rimg
# image resize and padding
def load_image(img,
               resize = True,
               img_size=(512, 512),
               return_hw=False):

    if isinstance(img, str):
        img = cv2.imread(img)[:, :, ::-1]

    h, w, _ = img.shape
    img = padding(img)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    if resize:
        img = cv2.resize(img, img_size)


    if return_hw:
        return img, h, w
    return img

# segmentation dataset

class SegmentData (tf.keras.utils.Sequence):
    def __init__(self,
                 data_dir,
                 no_channels = 3,
                 batch_size = 1,
                 n_classes = 2,
                 mask_false = True,    # bounding box bị lật ngược
                 img_size=(512, 512),
                 mask_thres=0.5,
                 shuffle = True,
                 debug=False):
        """
            Arguments:
                data_dir: str, path to the data directory
                no_channels: int, number of channels in the image
                batch_size: int, number of images in a batch
                n_classes: int, number of classes in the mask
                mask_false: bool, if True, the mask will be flipped
                img_size: tuple, size of the image
                mask_thres: float, threshold for the mask
                shuffle: bool, if True, shuffle the data

        """
        self.mask_false = mask_false 
        self.data_dir = data_dir
        self.img_labels = find_records(self.data_dir)
        self.mask_thres = mask_thres
        self.img_size= img_size
        self.no_channels = no_channels
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.ids = range(len(self.img_labels))
        
        print(">>> Batch_size: {} images".format(self.batch_size))
        
        self.ids = range(len(self.img_labels))
        
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index):
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]

        temp_ids = [self.ids[k] for k in indexes]
        
        X, y = self.__data_generation(temp_ids)
        
        return X, y
    
    def getimagename(self, index):
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]

        temp_ids = [self.ids[k] for k in indexes]
        for index, idx in enumerate(temp_ids):
            img_name = get_image_files(os.path.join(self.data_dir,self.img_labels[idx]))[0]
        
        return img_name
    
    def on_epoch_end(self):
        
        self.indexes = np.arange(len(self.ids))
        
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, ids):
        
        X = np.empty((0, *self.img_size, self.no_channels))  # noqa
        Y = np.empty((0, *self.img_size, self.n_classes))
        
        for index, idx in enumerate(ids):
            img_name = get_image_files(os.path.join(self.data_dir,self.img_labels[idx]))[0]
            img_dir = os.path.join(self.data_dir,img_name) 
            
            file = open('items.txt','w')
            for item in self.img_labels:
                file.write(item+"\n")
            file.close()
            
            # print("\t",img_name)
            
            img = cv2.imread(img_dir)[:,:,::-1]  # BGR2RGB
            
            lead_bbs = lead_bbox_extract(img_name,self.data_dir)
            text_bbs = text_bbox_extract(img_name,self.data_dir)
            lead_bbs = BoundingBoxesOnImage(lead_bbs , shape=img.shape)
            text_bbs = BoundingBoxesOnImage(text_bbs , shape=img.shape)
            
            mask = np.zeros((img.shape[0], img.shape[1], 2), dtype='float32')
            
            for bbx in lead_bbs.bounding_boxes:
                if self.mask_false:  
                    bbx.y1 = img.shape[0] - bbx.y1
                    bbx.y2 = img.shape[0] - bbx.y2

                w = int(bbx.x2 - bbx.x1)
                h = int(bbx.y1 - bbx.y2)
                x, y = int(bbx.x1), int(bbx.y2)
                
                mask[y: y + h, x: x + w, 0] = 1
                
            
            for bbx in text_bbs.bounding_boxes:
                
                if self.mask_false:  
                    bbx.y1 = img.shape[0] - bbx.y1
                    bbx.y2 = img.shape[0] - bbx.y2
            
                w = int(bbx.x2 - bbx.x1)
                h = int(bbx.y1 - bbx.y2)
                x, y = int(bbx.x1), int(bbx.y2)

                mask[y: y + h, x: x + w, 1] = 1
            
            mask = mask.astype(np.float32)
            
            mask = padding(mask)
            print("\t",mask.shape)
        
            mask = cv2.resize(mask, self.img_size)
            
            
            mask[mask >= self.mask_thres] = 1
            mask[mask < self.mask_thres] = 0
            
            img = load_image(img, img_size = self.img_size)
        
            img = img.astype(np.float32)
            mask = mask.astype(np.float32)
            
            X = np.vstack((X, np.expand_dims(img, axis=0)))
            Y = np.vstack((Y, np.expand_dims(mask, axis=0)))
        
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)

        assert X.shape[0] == Y.shape[0]
        return X, Y


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
    data = SegmentData(data_dir, img_size= (img_size, img_size), no_channels = no_channels)
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


# classify dataset

class ClassifyData(tf.keras.utils.Sequence):
    def __init__(self,
                 data_dir,
                 no_channels = 3,
                 batch_size = 1,
                 img_size=(512, 512),
                 shuffle = True):
        """
            Arguments:
                data_dir: str, path to the data directory
                no_channels: int, number of channels in the image
                batch_size: int, number of images in a batch
                n_classes: int, number of classes in the mask
                img_size: tuple, size of the image
                shuffle: bool, if True, shuffle the data

        """
        self.data_dir = data_dir
        self.img_labels = find_records(self.data_dir)
        self.img_size= img_size
        self.no_channels = no_channels
        self.batch_size = batch_size
        self.ids = range(len(self.img_labels))
        self.labels, self.classes = self.__label_generation()
        
        print(">>> Batch_size: {} images".format(self.batch_size))
        
        self.ids = range(len(self.img_labels))
        
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index):
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]

        temp_ids = [self.ids[k] for k in indexes]
        
        X, y = self.__data_generation(temp_ids)
        
        return X, y
    def return_classes(self):
        return self.classes
    
    def getimagename(self, index):
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]

        temp_ids = [self.ids[k] for k in indexes]
        for index, idx in enumerate(temp_ids):
            img_name = get_image_files(os.path.join(self.data_dir,self.img_labels[idx]))[0]
        
        return img_name
    
    def on_epoch_end(self):
        
        self.indexes = np.arange(len(self.ids))
        
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __label_generation(self):
        num_records = len(self.img_labels)
        classification_labels = list()
        for i in range(num_records):
            record = os.path.join(self.data_dir, self.img_labels[i])
            labels = load_labels(record)
            if any(label for label in labels):
                classification_labels.append(labels)
        classes = sorted(set.union(*map(set, classification_labels)))
        classification_labels = compute_one_hot_encoding(classification_labels, classes)
        return classification_labels, classes

        
    def __data_generation(self, ids):
        

        for index, idx in enumerate(ids):
            img_name = get_image_files(os.path.join(self.data_dir,self.img_labels[idx]))[0]
            # labels = load_labels(os.path.join(self.data_dir,self.img_labels[idx]))
            
            name,extension = os.path.splitext(img_name)
            img_dir = os.path.join(self.data_dir,(name + '_masked'+ extension)) 
    
            img = cv2.imread(img_dir)[:,:,::-1]  # BGR2RGB
            
            Y = self.labels[idx]
            
            X = np.empty((0, *self.img_size, self.no_channels),dtype=img.dtype)  # noqa
            X = np.vstack((X, np.expand_dims(img, axis=0)))
            Y = tf.cast(Y, tf.float32)
        
        # X = X.astype(np.float32)

        return X, Y
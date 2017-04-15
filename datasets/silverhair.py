import os
import numpy as np
from PIL import Image
import six
import json
import cv2
import glob
from io import BytesIO
import common.paths as paths
import numpy as np
from .datasets_base import datasets_base

class silverhair_train(datasets_base):
    def __init__(self, dataset_path=paths.root_silverhair, flip=1, resize_to=280, crop_to=256):
        super(silverhair_train, self).__init__(flip=flip, resize_to=resize_to, crop_to=crop_to)
        self.dataset_path = dataset_path
        self.trainAkey = glob.glob(dataset_path + "silver_hairs/*.jpg")
        self.trainBkey = glob.glob(dataset_path + "others/*.jpg")

    def __len__(self):
        return len(self.trainAkey)

    def do_resize(self, img):
        #print(img.shape)
        img = cv2.resize(img, (280, 336), interpolation=cv2.INTER_AREA)
        #print(img.shape)
        return img

    def do_random_crop(self, img, crop_to=256):
        w, h, ch = img.shape
        limx = w - crop_to
        limy = h - crop_to
        x = np.random.randint(0,limx)
        y = np.random.randint(0,limy)
        img = img[x:x+crop_to, y:y+crop_to]
        return img

    def do_augmentation(self, img):
        if self.flip > 0:
            img = self.do_flip(img)

        if self.resize_to > 0:
            img = self.do_resize(img)

        if self.crop_to > 0:
            img = self.do_random_crop(img, self.crop_to)
        return img

    def get_example(self, i):
        np.random.seed(None)
        idA = self.trainAkey[np.random.randint(0,len(self.trainAkey))]
        idB = self.trainBkey[np.random.randint(0,len(self.trainBkey))]
        #print(idA)

        imgA = cv2.imread(idA, cv2.IMREAD_COLOR)
        imgB = cv2.imread(idB, cv2.IMREAD_COLOR)

        imgA = self.do_augmentation(imgA)
        imgB = self.do_augmentation(imgB)

        imgA = self.preprocess_image(imgA)
        imgB = self.preprocess_image(imgB)

        return imgA, imgB

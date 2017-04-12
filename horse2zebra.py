import os
import numpy as np
from PIL import Image
import six
import json
import cv2
from io import BytesIO
from chainer.dataset import dataset_mixin
import paths
import numpy as np

def preprocess_image(img):
    img = img.astype("f")
    img = img/127.5 - 1
    img = img.transpose((2, 0, 1))
    return img

def postprocess_image(img):
    img = (img + 1) *127.5
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    img.transpose((1, 2, 0))
    return img

def random_crop(img, crop_to):
    sz = img.shape[0]
    if sz > crop_to:
        lim = sz - crop_to
        x = np.random.randint(0,lim)
        y = np.random.randint(0,lim)
        img = img[x:x+crop_to, y:y+crop_to]
    return img

class horse2zebra_Dataset_train(dataset_mixin.DatasetMixin):
    def __init__(self, dataset_path=paths.root_horse2zebra, flip=1, resize_to=256, crop_to=128):
        self.dataset_path = dataset_path
        self.trainAkey = paths.readAllString(dataset_path + paths.horse2zebra_trainA_key)
        self.trainBkey = paths.readAllString(dataset_path + paths.horse2zebra_trainB_key)
        self.flip = flip
        self.resize_to = resize_to
        self.crop_to  = crop_to

    def __len__(self):
        return len(self.trainAkey)


    def get_example(self, i):
        idA = self.trainAkey[np.random.randint(0,len(self.trainAkey))]
        idB = self.trainBkey[np.random.randint(0,len(self.trainBkey))]
        imgA = cv2.imread(self.dataset_path+'trainA/'+idA, cv2.IMREAD_COLOR)
        imgB = cv2.imread(self.dataset_path+'trainB/'+idB, cv2.IMREAD_COLOR)

        if self.flip > 0:
            if np.random.rand() > 0.5:
                imgA = cv2.flip(imgA, 1)
            if np.random.rand() > 0.5:
                imgB = cv2.flip(imgB, 1)

        if self.resize_to > 0:
            imgA = cv2.resize(imgA, (self.resize_to, self.resize_to), interpolation=cv2.INTER_AREA)
            imgB = cv2.resize(imgB, (self.resize_to, self.resize_to), interpolation=cv2.INTER_AREA)

        if self.crop_to > 0:
            imgA = random_crop(imgA, self.crop_to)
            imgB = random_crop(imgB, self.crop_to)

        imgA = preprocess_image(imgA)
        imgB = preprocess_image(imgB)

        return imgA, imgB


class horse2zebra_Dataset_test(dataset_mixin.DatasetMixin):
    def __init__(self, dataset_path=paths.root_horse2zebra, flip=1, resize_to=256, crop_to=128):# augmentation=False):
        self.dataset_path = dataset_path
        self.testAkey = paths.readAllString(dataset_path + paths.horse2zebra_testA_key)
        self.testBkey = paths.readAllString(dataset_path + paths.horse2zebra_testB_key)
        self.flip = flip
        self.resize_to = resize_to
        self.crop_to  = crop_to


    def __len__(self):
        return len(self.testAkey)

    def get_example(self, i):
        idA = self.testAkey[i]
        idB = self.testBkey[i]
        imgA = cv2.imread(self.dataset_path+'testA/'+idA, cv2.IMREAD_COLOR)
        imgB = cv2.imread(self.dataset_path+'testB/'+idB, cv2.IMREAD_COLOR)

        if self.flip > 0:
            if np.random.rand() > 0.5:
                imgA = cv2.flip(imgA, 1)
            if np.random.rand() > 0.5:
                imgB = cv2.flip(imgB, 1)

        if self.resize_to > 0:
            imgA = cv2.resize(imgA, (self.resize_to, self.resize_to), interpolation=cv2.INTER_AREA)
            imgB = cv2.resize(imgB, (self.resize_to, self.resize_to), interpolation=cv2.INTER_AREA)

        if self.crop_to > 0:
            imgA = random_crop(imgA, self.crop_to)
            imgB = random_crop(imgB, self.crop_to)

        imgA = preprocess_image(imgA)
        imgB = preprocess_image(imgB)

        return imgA, imgB

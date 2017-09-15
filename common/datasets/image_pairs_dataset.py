import numpy as np
import chainer
import os
import glob
from chainer import cuda, optimizers, serializers, Variable
import cv2
import json
from .datasets_base import datasets_base

class image_pairs_train(datasets_base):
    def __init__(self, dataset_a, dataset_b, flip=1, resize_to=280, crop_to=256):
        self.train_a_key = glob.glob(dataset_a + "/*.jpg")
        self.train_b_key = glob.glob(dataset_b + "/*.jpg")
        self.train_a_key.append(glob.glob(dataset_a + "/*.png"))
        self.train_b_key.append(glob.glob(dataset_b + "/*.png"))

        super(image_pairs_train, self).__init__(flip=flip, resize_to=resize_to, crop_to=crop_to)

    def __len__(self):
        return min(len(self.train_a_key), len(self.train_b_key))

    def get_example(self, i):
        np.random.seed(None)
        idA = self.train_a_key[np.random.randint(0, len(self.train_a_key))]
        idB = self.train_b_key[np.random.randint(0, len(self.train_b_key))]
        #print(idA)

        imgA = cv2.imread(id_a, cv2.IMREAD_COLOR)
        imgB = cv2.imread(id_b, cv2.IMREAD_COLOR)

        imgA = self.do_augmentation(imgA)
        imgB = self.do_augmentation(imgB)

        imgA = self.preprocess_image(imgA)
        imgB = self.preprocess_image(imgB)

        return imgA, imgB

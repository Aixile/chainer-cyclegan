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
        if os.path.isdir(dataset_a):
            self.train_a_key = glob.glob(dataset_a + "/*.jpg")
            self.train_a_key.append(glob.glob(dataset_a + "/*.png"))
        elif dataset_a.lower().endswith(('.json')):
            with open(dataset_a,'r') as f:
                self.train_a_key = json.load(f)
        else:
            self.train_a_key = []

        if os.path.isdir(dataset_b):
            self.train_b_key = glob.glob(dataset_b + "/*.jpg")
            self.train_b_key.append(glob.glob(dataset_b + "/*.png"))
        elif dataset_a.lower().endswith(('.json')):
            with open(dataset_b,'r') as f:
                self.train_b_key = json.load(f)
        else:
            self.train_b_key = []

        super(image_pairs_train, self).__init__(flip=flip, resize_to=resize_to, crop_to=crop_to, keep_aspect_ratio=True)

    def __len__(self):
        return min(len(self.train_a_key), len(self.train_b_key))

    def get_example(self, i):
        np.random.seed(None)
        idA = self.train_a_key[np.random.randint(0, len(self.train_a_key))]
        idB = self.train_b_key[np.random.randint(0, len(self.train_b_key))]
        #print(idA)

        imgA = cv2.imread(idA, cv2.IMREAD_COLOR)
        imgB = cv2.imread(idB, cv2.IMREAD_COLOR)

        imgA = self.do_augmentation(imgA)
        imgB = self.do_augmentation(imgB)

        imgA = self.preprocess_image(imgA)
        imgB = self.preprocess_image(imgB)
        return imgA, imgB

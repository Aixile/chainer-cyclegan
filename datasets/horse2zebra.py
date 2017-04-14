import os
import numpy as np
from PIL import Image
import six
import json
import cv2
from io import BytesIO
import common.paths as paths
import numpy as np
from .datasets_base import datasets_base

class horse2zebra_train(datasets_base):
    def __init__(self, dataset_path=paths.root_horse2zebra, flip=1, resize_to=172, crop_to=128):
        super(horse2zebra_train, self).__init__(flip=flip, resize_to=resize_to, crop_to=crop_to)
        self.dataset_path = dataset_path
        self.trainAkey = self.read_image_key_file_plaintext(dataset_path + paths.horse2zebra_trainA_key)
        self.trainBkey = self.read_image_key_file_plaintext(dataset_path + paths.horse2zebra_trainB_key)

    def __len__(self):
        return len(self.trainAkey)

    def get_example(self, i):
        idA = self.trainAkey[np.random.randint(0,len(self.trainAkey))]
        idB = self.trainBkey[np.random.randint(0,len(self.trainBkey))]

        imgA = cv2.imread(self.dataset_path+'trainA/'+idA, cv2.IMREAD_COLOR)
        imgB = cv2.imread(self.dataset_path+'trainB/'+idB, cv2.IMREAD_COLOR)

        imgA = self.do_augmentation(imgA)
        imgB = self.do_augmentation(imgB)

        imgA = self.preprocess_image(imgA)
        imgB = self.preprocess_image(imgB)

        return imgA, imgB


class horse2zebra_test(datasets_base):
    def __init__(self, dataset_path=paths.root_horse2zebra, flip=1, resize_to=172, crop_to=128):
        super(horse2zebra_train, self).__init__(flip=flip, resize_to=resize_to, crop_to=crop_to)
        self.dataset_path = dataset_path
        self.testAkey = self.read_image_key_file_plaintext(dataset_path + paths.horse2zebra_testA_key)
        self.testBkey = self.read_image_key_file_plaintext(dataset_path + paths.horse2zebra_testB_key)

    def __len__(self):
        return len(self.testAkey)

    def get_example(self, i):
        idA = self.testAkey[i]
        idB = self.testBkey[i]
        imgA = cv2.imread(self.dataset_path+'testA/'+idA, cv2.IMREAD_COLOR)
        imgB = cv2.imread(self.dataset_path+'testB/'+idB, cv2.IMREAD_COLOR)

        imgA = self.do_augmentation(imgA)
        imgB = self.do_augmentation(imgB)

        imgA = self.preprocess_image(imgA)
        imgB = self.preprocess_image(imgB)

        return imgA, imgB

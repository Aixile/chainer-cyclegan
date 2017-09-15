import numpy as np
import chainer
import os
import glob

from chainer import cuda, optimizers, serializers, Variable
import cv2
from .datasets_base import datasets_base

class celeba_train(datasets_base):
    def __init__(self, path, img_size=64):
        self._paths = glob.glob(path + "/img_align_celeba/*.jpg")
        super(celeba_train, self).__init__(flip=1, resize_to=img_size, crop_to=0)

    def __len__(self):
        return len(self._paths)

    def do_resize(self, img, resize_to):
        img = cv2.resize(img[20:198],(resize_to, resize_to),interpolation=cv2.INTER_AREA)
        return img

    def get_example(self, i):
        img = cv2.imread(self._paths[i])
        img = self.do_augmentation(img)
        img = self.preprocess_image(img)
        return img

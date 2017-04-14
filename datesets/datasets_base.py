import os
import numpy as np
from PIL import Image
import six
import json
import cv2
from io import BytesIO
from chainer.dataset import dataset_mixin
import numpy as np

class datasets_base(dataset_mixin.DatasetMixin):
    def __init__(self, flip=1, resize_to=128, crop_to=0):
        self.flip = flip
        self.resize_to = resize_to
        self.crop_to  = crop_to

    def preprocess_image(self, img):
        img = img.astype("f")
        img = img / 127.5 - 1
        img = img.transpose((2, 0, 1))
        return img

    def postprocess_image(self, img):
        img = (img + 1) *127.5
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
        img.transpose((1, 2, 0))
        return img

    def batch_postprocess_images(self, img, batch_w, batch_h):
        b, ch, w, h = img.shape
        img = img.reshape((batch_w, batch_h, ch, w, h))
        img = img.transpose(0,1,3,4,2)
        img = (img + 1) *127.5
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
        img = img.reshape((batch_w, batch_h, w, h, ch)).transpose(0,2,1,3,4).reshape((w*batch_w, h*batch_h, ch))[:,:,::-1]
        return img



    def read_image_key_file_plaintext(self, file):
        with open(file,'r') as f:
            lines = f.readlines()
        return [i.strip() for i in lines]

    def read_image_key_file_json(self, file):
        return json.load(open(file ,"r"))

    # ensure the image is square before doing random crop
    def do_random_crop(self, img, crop_to):
        sz = img.shape[0]
        if sz > crop_to:
            lim = sz - crop_to
            x = np.random.randint(0,lim)
            y = np.random.randint(0,lim)
            img = img[x:x+crop_to, y:y+crop_to]
        return img

    def do_resize(self, img, resize_to):
        img = cv2.resize(img, (resize_to, resize_to), interpolation=cv2.INTER_AREA)
        return img

    def do_flip(self, img):
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)
        return img

    def do_augmentation(self, img):
        if self.flip > 0:
            img = do_flip(img)

        if self.resize_to > 0:
            img = do_resize(img)

        if self.crop_to > 0:
            img = do_random_crop(img)

        return img

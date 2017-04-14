import os
import numpy as np
from PIL import Image
import six
import lmdb
import cv2
from io import BytesIO
import ..utils.XDoG as xdog
import ..common.paths as paths
import numpy as np
from datasets_base import datasets_base

class lsun_bedroom_line2color_train(datasets_base):
    def __init__(self, dataset_path=paths.lsun_bedroom, flip=1, resize_to=172, crop_to=128):
        super(lsun_bedroom_line2color_dataset, self).__init__(flip=flip, resize_to=resize_to, crop_to=crop_to)
        self.all_keys = self.read_image_key_file_json(paths.all_keys_lsun_bedroom_train)
        self.db = lmdb.open(dataset_path, readonly=True).begin(write=False)

    def __len__(self):
        return len(self.all_keys)

    def get_example(self, i):
        id = self.all_keys[i]
        img = None
        val = self.db.get(id.encode())

        img = cv2.imdecode(np.fromstring(val, dtype=np.uint8), 1)
        img = self.do_augmentation(img)

        img_color = img
        img_color = self.preprocess_image(img_color)

        img_line = xdog.XDoG(img)
        img_line = cv2.cvtColor(img_line, cv.CV_GRAY2RGB)
        #if img_line.ndim == 2:
        #    img_line = img_line[:, :, np.newaxis]
        img_line = self.preprocess_image(img_line)

        return img_line, img_color

import numpy as np
import chainer
import os
import glob
import pandas
import pickle
from chainer import cuda, optimizers, serializers, Variable
import cv2
import datetime
import json
import pickle
from .datasets_base import datasets_base

class game_faces_tags_generator():
    def __init__(self, attr_file):
        self._attr = json.load(open(attr_file))
        self._len = self.get_length()

    def get_length(self):
        ans=0
        for name, val in self._attr:
            if name!='gender':
                ans+=len(val)
            else:
                ans+=1
        return ans

    def get_tag_vector(self, prob, threshold=0.25):
        tags = np.zeros((self._len))
        start=0
        for name, val in self._attr:
            if name=='attribute':
                for i in range(len(val)):
                    tags[start] = 1 if prob[val[i][1]] >= threshold else 0
                    start+=1
            elif name=='gender':
                tags[start] = 1 if prob[val[1][1]] >= prob[val[0][1]] else 0
                start+=1
            else:
                ids = [v[1] for v in val]
                tag_len=len(val)
                tags[start+np.argmax(prob[ids])] = 1
                start+=tag_len
        return tags.astype("f")

    def get_fake_tag_vector(self, threshold=0.75):
        tags = np.zeros((self._len)).astype("f")
        tags[:] = -1
        prob = np.random.rand((self._len))
        start=0
        for name, val in self._attr:
            if name=='attribute':
                for i in range(len(val)):
                    tags[start] = 1 if prob[start] >= threshold else -1
                    start+=1
            elif name=='gender':
                tags[start] = 1 if prob[start] >= prob[start] else -1
                start+=1
            else:
                tag_len = len(val)
                ids = [i for i in range(start,start+tag_len)]
                tags[start+np.argmax(prob[ids])] = 1
                start+=tag_len
        return tags


class game_faces_tags_train(datasets_base):
    def __init__(self, path, img_size=64, flip=1, crop_to=0, threshold=0.25, attr_json_path='attr.json'):
        self._paths = glob.glob(path + "/images/*")
        self._datapath = path
        self._gamedata=pickle.load(open(path + "/games.pickle", 'rb'))
        self._tags = pickle.load(open(path + "/tags.pickle", 'rb'))
        self.tags_generator = game_faces_tags_generator(attr_json_path)
        self._threshold = threshold
        super(game_faces_tags_train, self).__init__(flip=1, resize_to=img_size, crop_to=crop_to)

    def __len__(self):
        return len(self._paths)

    def get_getchu_id(self, path):
        path = os.path.basename(path)
        ans = path.split("chara")
        return int(ans[0])

    def get_tags(self, path):
        path = os.path.basename(path)
        prob = self._tags[path]
        tags = self.tags_generator.get_tag_vector(prob, self._threshold)
        return tags

    def try_get_example(self, id=None):
        if id is None:
            id = np.random.randint(0, self.__len__())
        path = self._paths[id]
        g_id = self.get_getchu_id(path)
        year = self._gamedata.loc[self._gamedata['getchu_id']==g_id].iloc[0]['sellday'].year
        if year < 2003 or year > 2018:
            return None, None, None
        img = cv2.imread(path)
        if img.shape[0] < 180 or img.shape[0] != img.shape[1]:
            return None, None, None
        tags = self.get_tags(path)
        year = (year-2010) / 7.0
        return img, tags, year

    def count_tag(self):
        ans = np.zeros((self._len_attr))
        for i in range(self.__len__()):
            img, tag = self.try_get_example(i)
            if not img is None:
                ans += tag
        return ans

    def get_example(self, i, preprocessing=True):
        np.random.seed(None)
        while True:
            img, tags, year = self.try_get_example()
            if not img is None:
                break
        if preprocessing:
            img = self.do_augmentation(img)
            img = self.preprocess_image(img)
        return img, tags, year

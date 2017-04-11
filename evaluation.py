import os
import copy

import chainer
from chainer.training import extension
from chainer import Variable, cuda
import chainer.functions as F

import numpy as np
import os
import cv2

from PIL import Image

def evaluation(gen_g, gen_f, test_image_folder, image_size=256, side=2):
    @chainer.training.make_extension()
    def _eval(trainer, it):
        xp = gen_g.xp
        batch = it.next()
        batchsize = len(batch)

        #x = []
        x = xp.zeros((batchsize, 3, image_size, image_size)).astype("f")
        t = xp.zeros((batchsize, 3, image_size, image_size)).astype("f")
        for i in range(batchsize):
            x[i, :] = xp.asarray(batch[i][0])
            t[i, :] = xp.asarray(batch[i][1])

        x = Variable(x)
        result = gen_g(x, test=True)
        img = result.data.get()

        img_c = img.reshape((side, side, 3, image_size, image_size))
        img_c = img_c.transpose(0,1,3,4,2)
        img_c = img_c*255.0 + 127.5
        img_c = np.clip(img_c, 0, 255)
        img_c = img_c.astype(np.uint8)
        img_c = img_c.reshape((side, side, image_size, image_size, 3)).transpose(0,2,1,3,4).reshape((side*image_size, side*image_size, 3))[:,:,::-1]
        Image.fromarray(img_c).save(test_image_folder+"/iter_"+str(trainer.updater.iteration)+"_G.jpg")

        t = Variable(t)
        result = gen_f(t, test=True)
        img_t = result.data.get()
        img_t = img_t.reshape( (side, side, 3, image_size, image_size))
        img_t = img_t.transpose(0,1,3,4,2)
        img_t = img_t*255.0 + 127.5
        img_t = np.clip(img_t, 0, 255)
        img_t = img_t.astype(np.uint8)
        img_t = img_t.reshape((side, side, image_size, image_size, 3)).transpose(0,2,1,3,4).reshape((side*image_size, side*image_size, 3))[:,:,::-1]
        #print(img_t)
        Image.fromarray(img_t).save(test_image_folder+"/iter_"+str(trainer.updater.iteration)+"_F.jpg")

    def evaluation(trainer):
        it = trainer.updater.get_iterator('test')
        _eval(trainer, it)

    return evaluation

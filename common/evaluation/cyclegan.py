import os
import chainer
from chainer.training import extension
from chainer import Variable, cuda
import chainer.functions as F
import numpy as np
import os
import cv2
from common.utils.save_images import save_images_grid

def cyclegan_sampling(gen_g, gen_f, test_iter, eval_folder=".", batch_size=1):
    @chainer.training.make_extension()
    def samples_generation(trainer):
        if not os.path.exists(eval_folder):
            os.makedirs(eval_folder)

        xp = gen_f.xp
        batch = test_iter.next()
        img_shape = batch[0][0].shape
        x = xp.zeros((batch_size,) + img_shape).astype("f")
        y = xp.zeros((batch_size,) + img_shape).astype("f")

        for i in range(batch_size):
            x[i, :] = batch[i][0]
            y[i, :] = batch[i][1]

        x = Variable(x)
        y = Variable(y)

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x_y = gen_g(x)
            y_x = gen_f(y)
            x_y_x = gen_f(x_y)
            y_x_y = gen_g(y_x)

        imgs = xp.concat([x.data, x_y.data, x_y_x.data, y.data, y_x.data, y_x_y.data])
        filename="iter_" + str(trainer.updater.iteration) + ".jpg"
        
        save_images_grid(imgs, path=eval_folder+'/'+filename,
            grid_w=batch_size*3, grid_h=3)
        return samples_generation

import os
import chainer
from chainer.training import extension
from chainer import Variable, cuda
import chainer.functions as F
import numpy as np
import os
import cv2
from common.utils.save_images import save_images_grid

def unlabeled_gan_sampling(gen, eval_folder=".",
                            gpu=0, rows=6, cols=6, latent_len=128):
    @chainer.training.make_extension()

    def samples_generation(trainer):
        if not os.path.exists(eval_folder):
            os.makedirs(eval_folder)
        z = np.random.normal(size=(rows*cols, latent_len)).astype("f")
        if gpu>=0:
            z = cuda.to_gpu(z)

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            imgs = gen(Variable(z))

        filename="iter_" + str(trainer.updater.iteration) + ".jpg"

        save_images_grid(imgs, path=eval_folder+'/'+filename,
            grid_w=rows, grid_h=cols)

    return samples_generation

def labeled_gan_sampling(gen, tag_gen, eval_folder=".",
                            gpu=0, rows=6, cols=6, latent_len=128):
    @chainer.training.make_extension()

    def get_fake_tag_batch():
        xp = gen.xp
        batch = rows*cols
        tags = xp.zeros((batch, tag_gen._len)).astype("f")
        for i in range(batch):
            tags[i] = xp.asarray(tag_gen.get_fake_tag_vector())
        return tags

    def samples_generation(trainer):
        if not os.path.exists(eval_folder):
            os.makedirs(eval_folder)
        z = np.random.normal(size=(rows*cols, latent_len)).astype("f")
        if gpu>=0:
            z = cuda.to_gpu(z)
        tags =get_fake_tag_batch()

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                imgs = gen(F.concat([Variable(z), Variable(tags)]))

        filename="iter_" + str(trainer.updater.iteration) + ".jpg"

        save_images_grid(imgs, path=eval_folder+'/'+filename,
            grid_w=rows, grid_h=cols)

    return samples_generation



def labeled_gan_sampling_with_year(gen, tag_gen,  eval_folder='.',
                                gpu=0, rows=6, cols=6, latent_len=128):
    @chainer.training.make_extension()
    def get_fake_tag_batch():
        xp = gen.xp
        batch = rows*cols
        tags = xp.zeros((batch, tag_gen._len)).astype("f")
        for i in range(batch):
            tags[i] = xp.asarray(tag_gen.get_fake_tag_vector())
        return tags

    def get_fake_year_batch():
        xp = gen.xp
        year = xp.random.rand(rows*cols, 1).astype("f")*2.0 - 1.0
        return year

    def samples_generation(trainer):
        if not os.path.exists(eval_folder):
            os.makedirs(eval_folder)
        z = np.random.normal(size=(rows*cols, latent_len)).astype("f")
        if gpu>=0:
            z = cuda.to_gpu(z)
        tags =get_fake_tag_batch()
        year = get_fake_year_batch()
        z = Variable(z)
        tags = Variable(tags)
        year = Variable(year)
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            imgs = gen(F.concat([z, tags, year]))
        filename="iter_" + str(trainer.updater.iteration) + ".jpg"
        save_images_grid(imgs, path=eval_folder+'/'+filename,
            grid_w=rows, grid_h=cols)

    return samples_generation

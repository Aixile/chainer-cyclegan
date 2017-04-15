import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.datasets.image_dataset as ImageDataset
import six
import os

from chainer import cuda, optimizers, serializers, Variable
from chainer import training

from PIL import Image

def cal_l2_sum(h, t):
    return F.sum((h-t)**2)/ np.prod(h.data.shape)

def loss_func_rec_l1(x_out, t):
    return F.mean_absolute_error(x_out, t)

def loss_func_rec_l2(x_out, t):
    return F.mean_squared_error(x_out, t)

def loss_func_adv_dis_fake(y_fake):
    return cal_l2_sum(y_fake, 0.1)

def loss_func_adv_dis_real(y_real):
    return cal_l2_sum(y_real, 0.9)

def loss_func_adv_gen(y_fake):
    return cal_l2_sum(y_fake, 0.9)

def loss_func_tv(x_out):
    xp = cuda.get_array_module(x_out.data)
    b, ch, h, w = x_out.data.shape
    Wx = xp.zeros((ch, ch, 2, 2), dtype="f")
    Wy = xp.zeros((ch, ch, 2, 2), dtype="f")
    for i in range(ch):
        Wx[i,i,0,0] = -1
        Wx[i,i,0,1] = 1
        Wy[i,i,0,0] = -1
        Wy[i,i,1,0] = 1
    return F.sum(F.convolution_2d(x_out, W=Wx) ** 2) + F.sum(F.convolution_2d(x_out, W=Wy) ** 2)


class Updater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen_g, self.gen_f, self.dis_x, self.dis_y = kwargs.pop('models')
        params = kwargs.pop('params')
        self._lambda1 = params['lambda1']
        self._lambda2 = params['lambda2']
        self._learning_rate_anneal = params['learning_rate_anneal']
        self._learning_rate_anneal_interval = params['learning_rate_anneal_interval']
        self._image_size = params['image_size']
        self._eval_foler = params['eval_folder']
        self._dataset = params['dataset']
        self._iter = 0
        self._max_buffer_size = 50
        xp = self.gen_g.xp
        self._buffer_x = xp.zeros((self._max_buffer_size , 3, self._image_size, self._image_size)).astype("f")
        self._buffer_y = xp.zeros((self._max_buffer_size , 3, self._image_size, self._image_size)).astype("f")
        super(Updater, self).__init__(*args, **kwargs)

    def getAndUpdateBufferX(self, data):
        if  self._iter < self._max_buffer_size:
            self._buffer_x[self._iter, :] = data[0]
            return data

        self._buffer_x[0:self._max_buffer_size-2, :] = self._buffer_x[1:self._max_buffer_size-1, :]
        self._buffer_x[self._max_buffer_size-1, : ]=data[0]

        if np.random.rand() < 0.5:
            return data
        id = np.random.randint(0, self._max_buffer_size)
        return self._buffer_x[id, :].reshape((1, 3, self._image_size, self._image_size))


    def getAndUpdateBufferY(self, data):

        if  self._iter < self._max_buffer_size:
            self._buffer_y[self._iter, :] = data[0]
            return data

        self._buffer_y[0:self._max_buffer_size-2, :] = self._buffer_y[1:self._max_buffer_size-1, :]
        self._buffer_y[self._max_buffer_size-1, : ]=data[0]

        if np.random.rand() < 0.5:
            return data
        id = np.random.randint(0, self._max_buffer_size)
        return self._buffer_y[id, :].reshape((1, 3, self._image_size, self._image_size))
        """
    def save_images(self,img, w=2, h=3):
        img = cuda.to_cpu(img)
        img = img.reshape((w, h, 3, self._image_size, self._image_size))
        img = img.transpose(0,1,3,4,2)
        img = (img + 1) *127.5
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
        img = img.reshape((w, h, self._image_size, self._image_size, 3)).transpose(0,2,1,3,4).reshape((w*self._image_size, h*self._image_size, 3))[:,:,::-1]
        Image.fromarray(img).save(self._eval_foler+"/iter_"+str(self._iter)+".jpg")
        """

    def update_core(self):
        xp = self.gen_g.xp
        self._iter += 1
        batch = self.get_iterator('main').next()

        batchsize = len(batch)

        w_in = self._image_size

        x = xp.zeros((batchsize, 3, w_in, w_in)).astype("f")
        y = xp.zeros((batchsize, 3, w_in , w_in)).astype("f")

        for i in range(batchsize):
            x[i, :] = xp.asarray(batch[i][0])
            y[i, :] = xp.asarray(batch[i][1])

        x = Variable(x)
        y = Variable(y)

        x_y = self.gen_g(x)
        x_y_copy = self.getAndUpdateBufferX(x_y.data)
        x_y_copy = Variable(x_y_copy)
        x_y_x = self.gen_f(x_y)

        y_x = self.gen_f(y)
        y_x_copy = self.getAndUpdateBufferY(y_x.data)
        y_x_copy = Variable(y_x_copy)
        y_x_y = self.gen_g(y_x)

        opt_g = self.get_optimizer('gen_g')
        opt_f = self.get_optimizer('gen_f')
        opt_x = self.get_optimizer('dis_x')
        opt_y = self.get_optimizer('dis_y')

        if self._learning_rate_anneal > 0 and self._iter % self._learning_rate_anneal_interval == 0:
            if opt_g.alpha > self._learning_rate_anneal:
                opt_g.alpha -= self._learning_rate_anneal
            if opt_f.alpha > self._learning_rate_anneal:
                opt_f.alpha -= self._learning_rate_anneal
            if opt_x.alpha > self._learning_rate_anneal:
                opt_x.alpha -= self._learning_rate_anneal
            if opt_y.alpha > self._learning_rate_anneal:
                opt_y.alpha -= self._learning_rate_anneal

        opt_g.zero_grads()
        opt_f.zero_grads()
        opt_x.zero_grads()
        opt_y.zero_grads()

        loss_dis_y_fake = loss_func_adv_dis_fake(self.dis_y(x_y_copy))
        loss_dis_y_real = loss_func_adv_dis_real(self.dis_y(y))
        loss_dis_y = loss_dis_y_fake + loss_dis_y_real
        chainer.report({'loss': loss_dis_y}, self.dis_y)

        loss_dis_x_fake = loss_func_adv_dis_fake(self.dis_x(y_x_copy))
        loss_dis_x_real = loss_func_adv_dis_real(self.dis_x(x))
        loss_dis_x = loss_dis_x_fake + loss_dis_x_real
        chainer.report({'loss': loss_dis_x}, self.dis_x)

        loss_dis_y.backward()
        loss_dis_x.backward()

        opt_y.update()
        opt_x.update()

        loss_gen_g_adv = loss_func_adv_gen(self.dis_y(x_y))
        loss_gen_f_adv = loss_func_adv_gen(self.dis_x(y_x))

        loss_cycle_x = self._lambda1 * loss_func_rec_l1(x_y_x, x)
        loss_cycle_y = self._lambda1 * loss_func_rec_l1(y_x_y, y)
        loss_gen = self._lambda2*loss_gen_g_adv + self._lambda2*loss_gen_f_adv + loss_cycle_x + loss_cycle_y
        loss_gen.backward()
        opt_f.update()
        opt_g.update()

        chainer.report({'loss_rec': loss_cycle_y}, self.gen_g)
        chainer.report({'loss_rec': loss_cycle_x}, self.gen_f)
        chainer.report({'loss_adv': loss_gen_g_adv}, self.gen_g)
        chainer.report({'loss_adv': loss_gen_f_adv}, self.gen_f)

        if self._iter%100 ==0:
            img = xp.zeros((6, 3, w_in, w_in)).astype("f")
            img[0, : ] = x.data
            img[1, : ] = x_y.data
            img[2, : ] = x_y_x.data
            img[3, : ] = y.data
            img[4, : ] = y_x.data
            img[5, : ] = y_x_y.data
            img = cuda.to_cpu(img)
            img = self._dataset.batch_postprocess_images(img, 2, 3)
            Image.fromarray(img).save(self._eval_foler+"/iter_"+str(self._iter)+".jpg")

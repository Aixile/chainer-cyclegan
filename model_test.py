#!/usr/bin/env python

import os
import glob
import numpy as np

import chainer
import chainer.cuda
from chainer import cuda, serializers, Variable
from chainer import training
import chainer.functions as F
import cv2
import argparse
import common.net as net

import datasets
from PIL import Image

class testUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen_g, self.gen_f = kwargs.pop('models')
        params = kwargs.pop('params')
        self._image_size = params['image_size']
        self._eval_foler = params['eval_folder']
        self._iter = 0
        super(testUpdater, self).__init__(*args, **kwargs)

    def save_images(self,img, w=2, h=3):
        img = cuda.to_cpu(img)
        img = img.reshape((w, h, 3, self._image_size, self._image_size))
        img = img.transpose(0,1,3,4,2)
        img = (img + 1) *127.5
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
        img = img.reshape((w, h, self._image_size, self._image_size, 3)).transpose(0,2,1,3,4).reshape((w*self._image_size, h*self._image_size, 3))[:,:,::-1]
        Image.fromarray(img).save(self._eval_foler+"/test_"+str(self._iter)+".jpg")

    def update_core(self):
        self._iter += 1
        print(self._iter)
        xp = self.gen_g.xp
        batch = self.get_iterator('main').next()
        batchsize = 1
        w_in = self._image_size
        x = xp.zeros((1, 3, w_in, w_in)).astype("f")
        y = xp.zeros((1, 3, w_in , w_in)).astype("f")

        for i in range(batchsize):
            x[i, :] = xp.asarray(batch[i][0])
            y[i, :] = xp.asarray(batch[i][1])

        x = Variable(x)
        y = Variable(y)

        x_y = self.gen_g(x)
        y_x = self.gen_f(y)
        x_y_x = self.gen_f(x_y)
        y_x_y = self.gen_g(y_x)

        img = xp.zeros((6, 3, w_in, w_in)).astype("f")
        img[0, : ] = x.data
        img[1, : ] = x_y.data
        img[2, : ] = x_y_x.data
        img[3, : ] = y.data
        img[4, : ] = y_x.data
        img[5, : ] = y_x_y.data
        self.save_images(img)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model testing')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gen_class', default='Generator_ResBlock_6', help='Default generator class')
    parser.add_argument("--load_gen_f_model", default='', help='load generator model')
    parser.add_argument('--rows', type=int, default=4, help='rows')
    parser.add_argument('--cols', type=int, default=4, help='cols')
    parser.add_argument("--load_gen_g_model", default='', help='load generator model')
    parser.add_argument('--eval_folder', '-e', default='test', help='directory to output the evaluation result')
    parser.add_argument('--out', '-o', default='output' ,help='saved file')
    parser.add_argument("--resize_to", type=int, default=128, help='resize the image to')

    #parser.add_argument('--test_count', '-t', type=int, default=10, help='')

    args = parser.parse_args()
    print(args)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()

    if not os.path.exists(args.eval_folder):
        os.makedirs(args.eval_folder)

    gen_g = getattr(net, args.gen_class)()
    gen_f = getattr(net, args.gen_class)()


    if args.load_gen_g_model != '':
        serializers.load_npz(args.load_gen_g_model, gen_g)
        print("Generator G model loaded")

    if args.load_gen_f_model != '':
        serializers.load_npz(args.load_gen_f_model, gen_f)
        print("Generator F model loaded")

    if args.gpu >= 0:
        gen_g.to_gpu()
        gen_f.to_gpu()
        print("use gpu {}".format(args.gpu))

    test_dataset = datasets.silverhair_train(flip=0)
#    result = []

    cnt = args.rows * args.cols
    xp = gen_g.xp
    w_in = 256

    input = xp.zeros((cnt, 3, w_in, w_in)).astype("f")

    for i in range(0, args.rows):
        for j in range(0,args.cols):
            x, y = test_dataset.get_example(0)
            input[i*args.cols + j, :] = xp.asarray(y)

    output = gen_f(Variable(input))
    if args.gpu >= 0:
        output = cuda.to_cpu(output.data)
        input = cuda.to_cpu(input)
    else:
        output = output.data

    result = np.zeros((cnt*2, 3, w_in, w_in))

    for i in range(0, args.rows):
        for j in range(0,args.cols):
            id = i*args.cols + j
            result[id*2, :] = input[id]
            result[id*2+1, :] = output[id]

    result=test_dataset.batch_postprocess_images(result, args.rows, args.cols*2)
    Image.fromarray(result).save(args.eval_foler+"/"+args.out+".jpg")


    """
    test_dataset = horse2zebra_Dataset_test(flip=0, resize_to=args.resize_to, crop_to=0)
    test_iter = chainer.iterators.SerialIterator(test_dataset, 1)
    updater = testUpdater(
        models=(gen_g, gen_f),
        iterator={
            'main': test_iter
        },
        optimizer={
                },
        device=args.gpu,
        params={
            'image_size' : args.resize_to,
            'eval_folder' : args.eval_folder,
        })

    trainer = training.Trainer(updater, (args.test_count, 'iteration'))
    trainer.run()
    """

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
from utils import save_images_grid

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CycleGAN model testing script')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gen_class', '-c', default='Generator_ResBlock_9', help='Default generator class')
    parser.add_argument("--load_gen_f_model", default='', help='load generator model')
    parser.add_argument("--load_gen_g_model", default='', help='load generator model')
    parser.add_argument('--direction', '-d', type=int, default=1, help='direction: 0 for G(X), 1 for F(Y)')
    parser.add_argument('--input_channels',  type=int, default=3, help='number of input channels')
    parser.add_argument('--rows', type=int, default=5, help='rows')
    parser.add_argument('--cols', type=int, default=5, help='cols')
    parser.add_argument('--eval_folder', '-e', default='evaldata', help='directory to output the evaluation result')
    parser.add_argument('--out', '-o', default='output' ,help='saved file name')
    parser.add_argument("--resize_to", type=int, default=256, help='resize the image to')
    parser.add_argument("--crop_to", type=int, default=256, help='crop the resized image to')
    parser.add_argument("--load_dataset", default='silverhair_train', help='load dataset')
    parser.add_argument("--recurrent", type=int, default=1, help='apply the function recursively')

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

    test_dataset = getattr(datasets, args.load_dataset)(flip=0, resize_to=args.resize_to, crop_to=args.crop_to)

    cnt = args.rows * args.cols
    xp = gen_g.xp

    input = xp.zeros((cnt, args.input_channels, args.crop_to, args.crop_to)).astype("f")

    for i in range(0, args.rows):
        for j in range(0,args.cols):
            x, y = test_dataset.get_example(0)
            if args.direction == 1:
                input[i*args.cols + j, :] = xp.asarray(y)
            else:
                input[i*args.cols + j, :] = xp.asarray(x)

    input = input
    save_images_grid(input,path=args.eval_folder+"/"+args.out+".0.jpg", grid_w=args.rows, grid_h=args.cols)

    for i in range(args.recurrent):
        if args.direction == 1:
            output = gen_f(input,  volatile=True)
        else:
            output = gen_g(input, volatile=True)
        del input
        save_images_grid(output,path=args.eval_folder+"/"+args.out+"."+str(i+1)+".jpg", grid_w=args.rows, grid_h=args.cols)
        output.unchain_backward()
        input = output.data

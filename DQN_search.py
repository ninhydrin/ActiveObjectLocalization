#!/usr/bin/env python

from __future__ import print_function
import argparse

import numpy as np
import six
import random
import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import cv2
from skimage import io
from chainer import optimizers
from chainer import serializers
import pickle

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--initmodel', '-m', default='mlp.model',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=200000, type=int,
                    help='number of epochs to learn')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='learning minibatch size')
args = parser.parse_args()

batchsize = args.batchsize
n_epoch = args.epoch
alpha = 0.2
eta = 0.6


print('GPU: {}'.format(args.gpu))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('')


def load_image_list(path):
    tuples = []
    for line in open(path):
        pair = line.strip().split()
        tuples.append("imTrain/" + pair[0])
    return tuples

train_list = load_image_list("imagenet.txt")
val_list = load_image_list("imagenetVal.txt")
train_board = ["OnlyBoard/" + i[:-1] for i in open("train_board.txt")]
valid_board = ["OnlyBoard/" + i[:-1] for i in open("valid_board.txt")]


def read_image(path, is_val=False, flip=False):
    timage = np.asarray(io.imread(path))

    if len(timage.shape) != 3:
        timage = np.r_[np.array([timage]), np.array(
            [timage]), np.array([timage])].transpose(1, 2, 0)

    if timage.shape[2] == 4:
        timage = timage[:, :, :3]

    bimage = np.asarray(io.imread(random.sample(train_board, 1)[0]))

    if is_val:
        bimage = np.asarray(io.imread(random.sample(valid_board, 1)[0]))

    timage = cv2.resize(timage, (1280, 960))
    bo_shape = bimage.shape

    if random.randint(0, 1):
        bimage = bimage[:, ::-1, :]

    if random.randint(0, 2) > 1:
        myfilter = np.random.randn(bo_shape[0], bo_shape[1], bo_shape[
                                   2]) * (random.randint(1, 3) + random.random())
        #myfilter /=max(myfilter.max(),abs(myfilter.min()))
        myfilter = myfilter.astype(np.float32)
        bimage = bimage.astype(np.float32)
        bimage += myfilter

    if random.randint(0, 2) > 1:
        kari = timage.transpose(2, 0, 1)
        kari = kari.astype(np.float32)
        num = 1. + random.random()
        kari[1] /= num
        kari[0] /= num
        timage = kari.transpose(1, 2, 0)

    if random.randint(0, 2) > 1:
        kari = bimage.transpose(2, 0, 1)
        kari = kari.astype(np.float32)
        num = 1. + random.random()
        kari[1] /= num
        kari[0] /= num
        bimage = kari.transpose(1, 2, 0)

    bo_shape = bimage.shape
    im_shape = timage.shape

    x = random.randint(0, im_shape[1] - bo_shape[1])
    y = random.randint(0, im_shape[0] - bo_shape[0])
    ans = (x, y, x + bo_shape[1], y + bo_shape[0])
    timage[y:y + bo_shape[0], x:x + bo_shape[1], :] = bimage
    timage = timage.transpose(2, 0, 1)
    timage = timage / 255.  # timage_max

    if random.randint(0, 1) == 0:
        ans = (im_shape[1] - (x + bo_shape[1]), y,
               im_shape[1] - x, y + bo_shape[0])
        return timage[:, :, ::-1], ans
    else:
        return timage, ans

import net
alex = net.Alex()
serializers.load_hdf5("alex.param", alex)
alex.train = False
model = net.DQN()

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy


# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)


f_batch = np.ndarray(
    (1, 3, alex.insize, alex.insize), dtype=np.float32)


def get_feature(img, pos):
    img = img.transpose(1, 2, 0)
    n_img = cv2.resize(img[pos[1]:pos[3], pos[0]:pos[2], :], (227, 227))
    f_batch[0] = n_img.transpose(2, 0, 1)
    f = chainer.Variable(np.asarray(f_batch))
    return alex(f).data


def load_train(train=True):
    if train:
        im_path = random.choice(train_list)
    else:
        im_path = random.choice(val_list)
    return read_image(im_path)

x_batch = np.ndarray((1, model.insize), dtype=np.float32)


def get_score(pos, ans):
    x_l = max(pos[0], ans[0])
    y_t = max(pos[1], ans[1])
    x_r = min(pos[2], ans[2])
    y_b = min(pos[3], ans[3])
    common_area = float(max((y_b - y_t), 0) * max((x_r - x_l), 0))
    ans_are = max((pos[2] - pos[0]), 0) * max((pos[3] - pos[1]), 0)
    out_are = max((ans[2] - ans[0]), 0) * max((ans[3] - ans[1]), 0)
    #print (common_area,(ans_are+out_are-common_area))
    return (common_area / (ans_are + out_are - common_area))


def set_data(feat):
    x_batch[0][:4096] = feat
    for i in range(10):
        x_batch[0][4096 + i * 9:4096 + (i + 1) * 9] = past[i]


# Setup optimizer
optimizer = optimizers.SGD()
optimizer.setup(model)

optimizer.lr = 0.0098


replay = []
replay_count = 0
y = 10
from DqnActions import *
# Learning loop
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)
    # training
    past = np.zeros((10, 9), dtype=np.float32)
    img, ans = load_train()
    pos = [0, 0, img.shape[2] * 0.75, img.shape[1] * 0.75]
    action_batch = np.ndarray((1, 9), dtype=np.float32)
    for i in xrange(50):
        feat = get_feature(img, pos)
        set_data(feat)
        x = chainer.Variable(xp.asarray(x_batch))
        train_action = np.array(best_action(pos, ans))
        action_batch[0] = train_action
        a = train_action.argmax()
        if random.randint(0, 100) > 90:
            a = random.randint(0, 8)
        actions[a](pos)
        after_score = get_score(pos, ans)
        t = chainer.Variable(xp.array(action_batch))
        print ("Action:", a, "Score:", after_score)
        # Pass the loss function (Classifier defines it) and its arguments
        optimizer.update(model, x, t)
        if after_score > eta and a == 8:
            print ("ok")
            break
        past[1:] = past[:-1]
        past[0] = train_action
        replay.append([x.data.copy(), t.data.copy()])
        if len(replay) > 10000:
            print ("dump replay")
            pickle.dump(replay, open("dataset_" + str(replay_count), "w"))
            del replay
            replay = []
            replay_count += 1
    y *= 1.2
    if epoch % 100 == 0:
        print('save the model')
        serializers.save_npz('mlp.model', model)
        print('save the optimizer')
        serializers.save_npz('mlp.state', optimizer)

        past = np.zeros((10, 9), dtype=np.float32)
        img, ans = load_train()
        pos = [0, 0, img.shape[2] * 0.75, img.shape[1] * 0.75]
        action_batch = np.ndarray((1, 9), dtype=np.float32)
        for j in xrange(20):
            feat = get_feature(img, pos)
            set_data(feat)
            x = chainer.Variable(xp.asarray(x_batch))
            a = cuda.to_cpu(model.action(x).data).argmax()
            print ("Action:", a)
            if a == 8:
                break
            actions[a](pos)
        im = img[:, pos[1]:pos[3], pos[0]:pos[2]]
        im = im / im.max()
        io.imsave("dqn_image/" + str(epoch) + "img.png", im.transpose(1, 2, 0))
    optimizer.lr *= 0.99

print('save the model')
serializers.save_npz('mlp.model', model)
print('save the optimizer')
serializers.save_npz('mlp.state', optimizer)

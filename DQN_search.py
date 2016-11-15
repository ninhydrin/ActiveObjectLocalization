# 
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

from DqnActions import Action
import pickle

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--initmodel', '-m', default='',
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

import BlackboardTools as BT

import net
feature_extractor = net.Alex()
serializers.load_npz("alex_npz.model", feature_extractor)
feature_extractor.train = False

model = net.DQN()

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)



def get_feature(window_ob):
    img = window_ob.now_img.transpose(1, 2, 0)
    n_img = cv2.resize(img, (227, 227))
    f_batch[0] = n_img.transpose(2, 0, 1)
    f = chainer.Variable(np.asarray(f_batch))
    return feature_extractor(f).data[0]

def load_train(train=True):
    if train:
        im_path = random.choice(train_list)
    else:
        im_path = random.choice(val_list)
    return BT.read_image(im_path)

def set_data(feat):
    x_batch[0][:4096] = feat
    for i in range(10):
        x_batch[0][4096 + i * 9:4096 + (i + 1) * 9] = past_action[i]
    return chainer.Variable(xp.asarray(x_batch))

#optimizer = optimizers.MomentumSGD()
optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, momentum=0.95, eps=0.01)
optimizer.setup(model)

optimizer.lr = 0.0098

replay = []
replay_count = 0
y = 10

f_batch = np.ndarray(
    (1, 3, feature_extractor.insize, feature_extractor.insize), dtype=np.float32)
x_batch = np.ndarray((args.batchsize, model.insize), dtype=np.float32)
action_batch = np.ndarray((args.batchsize, 9), dtype=np.float32)
test_batch = np.ndarray((1, model.insize), dtype=np.float32)

# Learning loop
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)

    past_action = np.zeros((10, 9), dtype=np.float32)
    img, ans = load_train()
    window = Action(img, ans)

    for i in range(args.batchsize):
        feature = get_feature(window)
        x_batch[i] =np.append(feature,past_action.flatten())
        action = window.best_action()
        action_batch[i] = action
        max_list = [n for n, x in enumerate(action) if x == 1]
        if not max_list:
            break
        window.action(np.random.choice(max_list))
        after_score = window.score
        #print ("Action:", action.argmax(), "Score:", after_score())
        past_action[1:] = past_action[:-1]
        past_action[0] = action
        replay.append([x_batch[i].copy(), action_batch[i].copy()])
        if len(replay) > 1000:
            print ("dump replay")
            pickle.dump(replay, open("dataset/dataset_" + str(replay_count), "w"))
            del replay
            replay = []
            replay_count += 1

        if action.argmax() == 8:
            break
    if i:
        x = chainer.Variable(xp.array(x_batch[:i]))
        t = chainer.Variable(xp.array(action_batch[:i]))
        optimizer.update(model, x, t)
        print (model.loss.data)

    if epoch % 100 == 0:
        print('save the model')
        serializers.save_npz('mlp.model', model)
        print('save the optimizer')
        serializers.save_npz('mlp.state', optimizer)

        past_action = np.zeros((10, 9), dtype=np.float32)
        img, ans = load_train()
        window = Action(img, ans)
        for j in range(20):
            feature = get_feature(window)
            test_batch[0] =np.append(feature,past_action.flatten())
            x = chainer.Variable(xp.asarray(test_batch))
            action = cuda.to_cpu(model.action(x).data)
            print ("Action:", action)
            action=action.argmax()
            if action == 8:
                break
            window.action(action)
            past_action[1:] = past_action[:-1]
            past_action[0][action] = 1
            io.imsave("dqn_image/{}img.png".format(j), window.now_img.transpose(1, 2, 0)/ window.now_img.max())
        optimizer.lr *= 0.97

print('save the model')
serializers.save_npz('mlp.model', model)
print('save the optimizer')
serializers.save_npz('mlp.state', optimizer)

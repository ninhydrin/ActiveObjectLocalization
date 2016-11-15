# -*- coding: utf-8 -*-
import chainer
import chainer.functions as F
import chainer.links as L



class Alex(chainer.Chain):

    insize = 227

    def __init__(self):
        super(Alex, self).__init__(
            conv1=L.Convolution2D(3,  96, 11, stride=4),
            conv2=L.Convolution2D(96, 256,  5, pad=2),
            conv3=L.Convolution2D(256, 384,  3, pad=1),
            conv4=L.Convolution2D(384, 384,  3, pad=1),
            conv5=L.Convolution2D(384, 256,  3, pad=1),
            fc6=L.Linear(9216, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 1000),
        )
        self.train = False
        self.fc6_dropout_rate=0.5
        self.fc7_dropout_rate=0.5


    def __call__(self, x):

        h = F.max_pooling_2d(
            F.local_response_normalization(F.relu(self.conv1(x)),alpha=0.00002,k=1), 3, stride=2)
        h = F.max_pooling_2d(
            F.local_response_normalization(F.relu(self.conv2(h)),alpha=0.00002,k=1), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)), ratio=self.fc6_dropout_rate, train=self.train)
        return h

class DQN(chainer.Chain):
    insize = 4096 + 9 * 10

    def __init__(self):
        super(DQN, self).__init__(
            fc1=L.Linear(4096 + 9 * 10, 1024),
            fc2=L.Linear(1024, 1024),
            fc3=L.Linear(1024, 9),
        )
        self.train = True

    def __call__(self, x, y):
        h = F.dropout(F.tanh(self.fc1(x)), train=self.train)
        h = F.dropout(F.tanh(self.fc2(h)), train=self.train)
        h = F.tanh(self.fc3(h))

        self.loss = F.mean_squared_error(h, y)
        return self.loss

    def action(self, x):
        h = F.tanh(self.fc1(x))
        h = F.tanh(self.fc2(h))
        h = F.tanh(self.fc3(h))
        return F.softmax(h)

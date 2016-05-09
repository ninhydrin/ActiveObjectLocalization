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

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        self.feat = F.dropout(F.relu(self.fc6(h)), train=self.train)
        #self.loss = F.softmax_cross_entropy(h, t)
        #self.accuracy = F.accuracy(h, t)
        return self.feat


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
        h = F.relu(self.fc1(x))
        h = F.dropout(F.relu(self.fc2(h)), train=self.train)
        h = F.dropout(F.relu(self.fc3(h)), train=self.train)
        self.loss = F.mean_squared_error(h, y)  # F.softmax_cross_entropy(h,y)
        #self.accuracy = F.accuracy(h, y)
        return self.loss

    def action(self, x):
        h = F.relu(self.fc1(x))
        h = F.dropout(F.relu(self.fc2(h)), train=self.train)
        h = F.dropout(F.relu(self.fc3(h)), train=self.train)
        return F.softmax(h)

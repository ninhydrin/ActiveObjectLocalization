import copy
import numpy as np
import cv2


class Action(object):
    def __init__(self, img, gt, alpha=0.2, ita=0.6):
        self.img = img
        self.gt = gt
        self._position=[0, 0, img.shape[2]*0.75, img.shape[1]*0.75]
        self.alpha = alpha
        self.ita = ita
        self.x = self.img.shape[2]
        self.y = self.img.shape[1]

    @property
    def now_img(self):
        return self.img[:, self.position[1]:self.position[3], self.position[0]:self.position[2]]

    @property
    def actions(self):
        return [self.right, self.left, self.up, self.down,
                self.bigger, self.smaller, self.fatter, self.taller, self.trriger]

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, pos):
        self._position = [int(i) for i in pos]

    def score(self, pos=None):
        if not pos:
            pos = self._position
        x_l = max(pos[0], self.gt[0])
        y_t = max(pos[1], self.gt[1])
        x_r = min(pos[2], self.gt[2])
        y_b = min(pos[3], self.gt[3])
        common_area = float(max((y_b - y_t), 0) * max((x_r - x_l), 0))
        ans_are = max((pos[2] - pos[0]), 0) * max((pos[3] - pos[1]), 0)
        out_are = max((self.gt[2] - self.gt[0]), 0) * max((self.gt[3] - self.gt[1]), 0)
        return (common_area / (ans_are + out_are - common_area))

    def feature(self, extractor):
        pass

    def best_action(self):
        before = self.score()
        reward = [-before] * 9
        for i in range(9):
            score, pos = self.actions[i]()
            reward[i] = 1 if reward[i]+score > 0 else -1
            if i == 8:
                reward[i] = 3 if score >= 0.6 else -3
        return np.array(reward)

    def action(self, num):
        _, pos = self.actions[num]()
        self.position = pos

    def right(self):
        pos = self._position[:]
        movement = self.alpha * (pos[2] - pos[0])
        if movement + pos[2] > self.x:
            movement = self.x - pos[2]
        pos[0] += movement
        pos[2] += movement
        return self.score(pos),pos

    def left(self):
        pos = self._position[:]
        movement = self.alpha * (pos[2] - pos[0])
        if pos[0] - movement < 0:
            movement = pos[0]
        pos[0] -= movement
        pos[2] -= movement
        return self.score(pos),pos

    def down(self):
        pos = self._position[:]
        movement = self.alpha * (pos[3] - pos[1])
        if movement + pos[3] > self.y:
            movement = self.y - pos[3]
        pos[1] += movement
        pos[3] += movement
        return self.score(pos),pos

    def up(self):
        pos = self._position[:]
        movement = self.alpha * (pos[3] - pos[1])
        if pos[1] - movement < 0:
            movement = pos[1]
        pos[1] -= movement
        pos[3] -= movement
        return self.score(pos),pos

    def bigger(self):
        pos = self._position[:]
        scale_x = int(self.alpha * (pos[2] - pos[0]) / 2)
        scale_y = int(self.alpha * (pos[3] - pos[1]) / 2)
        can_list = [pos[0] - scale_x, pos[1] - scale_y]
        if min(can_list) >= 0 and pos[2] + scale_x <= self.x and pos[3] + scale_y <= self.y:
            pos[0] -= scale_x
            pos[1] -= scale_y
            pos[2] += scale_x
            pos[3] += scale_y
        return self.score(pos),pos

    def smaller(self):
        pos = self._position[:]
        if pos[2] - pos[0] > 20 and pos[3] - pos[1] > 18:
            scale_x = int(self.alpha * (pos[2] - pos[0]) / 2)
            scale_y = int(self.alpha * (pos[3] - pos[1]) / 2)
            pos[0] += scale_x
            pos[1] += scale_y
            pos[2] -= scale_x
            pos[3] -= scale_y
        return self.score(pos),pos

    def fatter(self):
        pos = self._position[:]
        if pos[3] - pos[1] > 18:
            scale_y = int(self.alpha * (pos[3] - pos[1]) / 2)
            pos[1] += scale_y
            pos[3] -= scale_y
        return self.score(pos),pos

    def taller(self):
        pos = self._position[:]
        if pos[2] - pos[0] > 20:
            scale_x = int(self.alpha * (pos[2] - pos[0]) / 2)
            pos[0] += scale_x
            pos[2] -= scale_x
        return self.score(pos),pos

    def trriger(self):
        return self.score(self._position), self._position

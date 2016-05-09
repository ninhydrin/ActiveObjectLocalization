import copy

alpha = 0.2
x_max = 1280
y_max = 960


def ac_score(pos, ans):
    x_l = max(pos[0], ans[0])
    y_t = max(pos[1], ans[1])
    x_r = min(pos[2], ans[2])
    y_b = min(pos[3], ans[3])
    common_area = float(max((y_b - y_t), 0) * max((x_r - x_l), 0))
    ans_are = max((pos[2] - pos[0]), 0) * max((pos[3] - pos[1]), 0)
    out_are = max((ans[2] - ans[0]), 0) * max((ans[3] - ans[1]), 0)
    #print (common_area,(ans_are+out_are-common_area))
    return (common_area / (ans_are + out_are - common_area))


def best_action(pos, ans):
    before = ac_score(pos, ans)
    reward = [-before] * 9
    for i in xrange(9):
        kari = copy.copy(pos)
        actions[i](kari)
        score = ac_score(kari, ans)
        reward[i] += score
        reward[i] = 1 if reward[i] > 0 else -1
        if i == 8:
            reward[i] = 3 if score >= 0.6 else -3
        max_score = -1
        for j in xrange(8):
            kari2 = copy.copy(kari)
            actions[j](kari2)
            score = ac_score(kari2, ans)
            max_score = max(score, max_score)
        reward[i] += max_score / 2.
    return reward


def move_right(pos):
    movement = alpha * (pos[2] - pos[0])
    if movement + pos[2] > x_max:
        movement = x_max - pos[2]
    pos[0] += movement
    pos[2] += movement


def move_left(pos):
    movement = alpha * (pos[2] - pos[0])
    if pos[0] - movement < 0:
        movement = pos[0]
    pos[0] -= movement
    pos[2] -= movement


def move_down(pos):
    movement = alpha * (pos[3] - pos[1])
    if movement + pos[3] > y_max:
        movement = y_max - pos[3]
    pos[1] += movement
    pos[3] += movement


def move_up(pos):
    movement = alpha * (pos[3] - pos[1])
    if pos[1] - movement < 0:
        movement = pos[1]
    pos[1] -= movement
    pos[3] -= movement


def bigger(pos):
    scale_x = int(alpha * (pos[2] - pos[0]) / 2)
    scale_y = int(alpha * (pos[3] - pos[1]) / 2)
    can_list = [pos[0] - scale_x, pos[1] - scale_y]
    if min(can_list) >= 0 and pos[2] + scale_x <= x_max and pos[3] + scale_y <= y_max:
        pos[0] -= scale_x
        pos[1] -= scale_y
        pos[2] += scale_x
        pos[3] += scale_y


def smaller(pos):
    if pos[2] - pos[0] > 20 and pos[3] - pos[1] > 18:
        scale_x = int(alpha * (pos[2] - pos[0]) / 2)
        scale_y = int(alpha * (pos[3] - pos[1]) / 2)
        pos[0] += scale_x
        pos[1] += scale_y
        pos[2] -= scale_x
        pos[3] -= scale_y


def fatter(pos):
    if pos[3] - pos[1] > 18:
        scale_y = int(alpha * (pos[3] - pos[1]) / 2)
        pos[1] += scale_y
        pos[3] -= scale_y


def taller(pos):
    if pos[2] - pos[0] > 20:
        scale_x = int(alpha * (pos[2] - pos[0]) / 2)
        pos[0] += scale_x
        pos[2] -= scale_x


def trriger(pos):
    pass

actions = [move_right, move_left, move_up, move_down,
           bigger, smaller, fatter, taller, trriger]

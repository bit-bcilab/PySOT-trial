

import numpy as np
import random


def random_sys(size=1):
    if size > 1:
        rd = []
        for i in range(size):
            rd.append(random.random())
        rd = np.array(rd, dtype=np.float32)
        return rd
    else:
        return random.random()


def rand(a=0., b=1.):
    """
    随机变化比例
    :param a: 变化随机值下限
    :param b: 变化范围的上限
    :return:
    """
    return random.uniform(a, b)
    # return random_sys() * (b - a) + a


def select(position, keep_num=16):
    num = position[0].shape[0]
    if num <= keep_num:
        return position, num, []
    slt = np.arange(num)
    random.shuffle(slt)
    slt = slt[:keep_num]
    return tuple(p[slt] for p in position), keep_num, slt

import numpy as np


def onehot2int(array):
    return np.asarray([np.where(row == 1)[0][0] for row in array], dtype=np.uint8)

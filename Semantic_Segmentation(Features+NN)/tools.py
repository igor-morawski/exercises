import numpy as np
from sklearn.preprocessing import LabelBinarizer
import pickle


def onehot2int(array):
    return np.asarray([np.where(row == 1)[0][0] for row in array], dtype=np.uint8)


def int2onehot(array):
    binazer = LabelBinarizer()
    return binazer.fit_transform(array).astype(np.uint8)


def load_pickle(fn):
    with open(fn, 'rb') as handle:
        content = pickle.load(handle)
    return content


def dump2pickle(content, fn):
    with open(fn, 'wb') as handle:
        pickle.dump(content, handle)
    return


def prob2class(array):
    return np.argmax(array, axis=1)


def prob2onehot(array):
    result = np.zeros(array.shape, dtype=np.uint8)
    result[(np.arange(len(result)), np.argmax(result, axis=1))] = 1
    return result


def class2bgr(img, palette_bgr):
    result = np.zeros([img.shape[0], img.shape[1], 3], dtype=np.uint8)
    for label in range(img.max()+1):
        y, x, _ = np.where(img == label)
        result[(y, x)] = palette_bgr[label]
    return result


def bgr2class(img, palette_bgr):
    result = np.zeros([img.shape[0], img.shape[1], 1], dtype=np.uint8)
    for label in range(img.max()+1):
        y, x, _ = np.where(img == label)
        result[(y, x)] = palette_bgr[label]
    return result

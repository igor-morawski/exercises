import os
import progressbar
from glob import glob
import pickle
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

import cv2
import mahotas as mt
from skimage import feature
from numpy.lib import stride_tricks

from settings import image_dir, label_dir, model_dir, cache_dir
from settings import scaler_fn, model_fn
from settings import palette_bgr, palette_rgb
from settings import oversample, random_state
from settings import classes, class_sep
from settings import h_neigh, h_ind, lbp_radius, lbp_points
from settings import test_size

from tools import onehot2int, int2onehot, load_pickle, dump2pickle


def test_model(X, y, model):
    pred = model.predict(X)
    precision = metrics.precision_score(
        y, pred, average='weighted', labels=np.unique(pred))
    recall = metrics.recall_score(
        y, pred, average='weighted', labels=np.unique(pred))
    f1 = metrics.f1_score(y, pred, average='weighted', labels=np.unique(pred))
    accuracy = metrics.accuracy_score(y, pred)

    print('--------------------------------')
    print('[RESULTS] Accuracy: %.2f' % accuracy)
    print('[RESULTS] Precision: %.2f' % precision)
    print('[RESULTS] Recall: %.2f' % recall)
    print('[RESULTS] F1: %.2f' % f1)
    print('--------------------------------')
    return


def train_model(X, y):
    print('[INFO] Training MLP model.')
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, warm_start=True,
                          hidden_layer_sizes=(30), learning_rate='adaptive', early_stopping=True, max_iter=200, random_state=random_state)
    model.fit(X, y)

    print('[INFO] Model training complete.')
    print('[INFO] Training Accuracy: %.2f' % model.score(X, y))
    return model


def calc_haralick(roi):
    feature_vec = []

    texture_features = mt.features.haralick(roi)
    mean_ht = texture_features.mean(axis=0)

    [feature_vec.append(i) for i in mean_ht[0:9]]
    return np.array(feature_vec)


def harlick_features(img, h_neigh):

    print('[INFO] Computing haralick features.')
    size = h_neigh
    shape = (img.shape[0] - size + 1, img.shape[1] - size + 1, size, size)
    strides = 2 * img.strides
    patches = stride_tricks.as_strided(img, shape=shape, strides=strides)
    patches = patches.reshape(-1, size, size)

    bar = progressbar.ProgressBar(maxval=img.size,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    bar.start()

    h_features = []

    for i, p in enumerate(patches):
        bar.update(i+1)
        h_features.append(calc_haralick(p))

    return np.array(h_features)


def create_binary_pattern(img, p, r):

    print('[INFO] Computing local binary pattern features.')
    lbp = feature.local_binary_pattern(img, p, r)
    return (lbp-np.min(lbp))/(np.max(lbp)-np.min(lbp))


def create_features(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feature_img = np.zeros((img.shape[0], img.shape[1], 4))
    feature_img[:, :, :3] = img

    hsv = np.zeros((img.shape[0], img.shape[1], 3))
    hsv[:, :, :3] = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = hsv[h_ind:-h_ind, h_ind:-h_ind]

    feature_img[:, :, 3] = create_binary_pattern(
        img_gray, lbp_points, lbp_radius)
    feature_img = feature_img[h_ind:-h_ind, h_ind:-h_ind]
    features = feature_img.reshape(
        feature_img.shape[0]*feature_img.shape[1], feature_img.shape[2])
    hsv = hsv.reshape(
        feature_img.shape[0]*feature_img.shape[1], 3)

    features = np.hstack((hsv, features))

    h_features = harlick_features(img_gray, h_neigh)

    features = np.hstack((features, h_features))

    return features


def create_dataset(image_dict, label_dict):
    print('[INFO] Creating training dataset on %d image(s).' %
          len(image_dict.keys()))

    X = []
    y = []

    for fn in image_dict.keys():
        img = image_dict[fn]
        label = label_dict[fn]

        cached_fn = os.path.join(cache_dir, fn.split(
            os.path.sep)[-1].split('.')[0]+".pkl")
        if os.path.isfile(cached_fn):
            features = load_pickle(cached_fn)
        else:
            features = create_features(img)
            dump2pickle(features, cached_fn)

        label = label[h_ind:-h_ind, h_ind:-h_ind]
        labels = label.reshape(label.shape[0]*label.shape[1], -1)
        X.append(features)
        y.append(labels)
    X = np.array(X)
    X = X.reshape(X.shape[0]*X.shape[1], -1)

    y = np.array(y)
    y = (y > 0) * 1
    y = y.astype(dtype=np.uint8)
    y = y.reshape(y.shape[0]*y.shape[1], -1)

    # delete ambigous samples
    indices = np.where(y.sum(axis=1) != 1)
    y = np.delete(y, indices, axis=0)
    X = np.delete(X, indices, axis=0)

    # unused
    def add_background_label(y):
        expanded = np.c_[y, np.zeros(y.shape[0])].astype(np.uint8)
        bg_label = np.zeros(len(classes)+1, dtype=np.uint8)
        bg_label[-1] = 1
        expanded[np.where(~expanded.any(axis=1))] = bg_label
        return expanded

    y_int = onehot2int(y)

    from collections import Counter
    print("Original dataset shape {}".format(Counter(y_int)))
    if oversample:
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=random_state)
        X, y_int = ros.fit_resample(X, y_int)
        print("Resampled dataset shape {}".format(Counter(y_int)))
        y = int2onehot(y_int)

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    dump2pickle(scaler, os.path.join(model_dir, scaler_fn))

    return X, y


def read_data(image_dir, label_dir):

    print('[INFO] Reading image data.')

    filelist = glob(os.path.join(image_dir, '*.jpg'))
    image_dict = dict()
    label_dict = dict()

    for fn in filelist:
        image_dict[fn] = cv2.imread(fn, cv2.IMREAD_COLOR)
        label_fn = os.path.join(label_dir, fn.split(
            os.path.sep)[-1].split('.')[0]+".png")
        label_map = cv2.imread(label_fn, cv2.IMREAD_COLOR)
        label_layers = []
        if not os.path.isfile(label_fn):
            raise ValueError(
                "Label file {} does not exist".format(label_fn))
        for i in classes:
            indices = np.where(label_map == palette_bgr[i])[:2]
            ones = np.zeros((label_map.shape[0], label_map.shape[1]), np.uint8)
            ones[indices] = 1
            label_layers.append(ones)
        label_dict[fn] = np.stack([layer
                                   for layer in label_layers], axis=-1)
    return image_dict, label_dict


if __name__ == "__main__":
    cache_dir = os.path.join(cache_dir, "store")

    image_dict, label_dict = read_data(image_dir, label_dir)
    a = label_dict['data\\images\\pots_block_2_27.jpg']
    X, y = create_dataset(image_dict, label_dict)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    print('[INFO] Feature vector size:', X_train.shape)
    model = train_model(X_train, y_train)
    test_model(X_test, y_test, model)
    dump2pickle(model, os.path.join(model_dir, model_fn))

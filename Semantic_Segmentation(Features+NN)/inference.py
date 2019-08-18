import os
from glob import glob
import pickle
import numpy as np

from settings import infere_dir, label_dir, model_dir, cache_dir, output_dir
from settings import scaler_fn, model_fn
from settings import h_neigh
from settings import classes, palette_bgr

from tools import load_pickle, dump2pickle
from tools import prob2onehot, prob2class, onehot2int, int2onehot
from tools import class2bgr
import train

import cv2


def compute_prediction(fn, model):
    border = int((h_neigh-1)/2)

    img = cv2.imread(fn, cv2.IMREAD_COLOR)
    cached_fn = os.path.join(cache_dir, fn.split(
        os.path.sep)[-1].split('.')[0]+".pkl")
    if os.path.isfile(cached_fn):
        features = load_pickle(cached_fn)
    else:
        features = train.create_features(img)
        dump2pickle(features, cached_fn)
    scaler = load_pickle(os.path.join(model_dir, scaler_fn))
    features = features.reshape(-1, features.shape[1])
    features = scaler.transform(features)
    model_predictions = model.predict_proba(features)
    model_predictions = prob2class(model_predictions)
    predictions_image = model_predictions.reshape(
        [img.shape[0]-2*border, img.shape[1]-2*border, -1])
    return predictions_image


if __name__ == '__main__':
    filelist = glob(os.path.join(infere_dir, '*.jpg'))

    print('[INFO] Running inference on %s test images' % len(filelist))
    model = load_pickle(os.path.join(model_dir, model_fn))

    for fn in filelist:
        print('[INFO] Processing images:', fn.split(os.path.sep)[-1])
        inference_img = compute_prediction(fn, model)
        mapped_img = class2bgr(inference_img, palette_bgr)
        output_fn = os.path.join(output_dir, fn.split(
            os.path.sep)[-1].split('.')[0]+".png")
        cv2.imwrite(output_fn, mapped_img)

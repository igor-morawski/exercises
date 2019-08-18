from settings import evaluate_dir, label_dir
from glob import glob
import os
import cv2
import numpy as np

from settings import classes, class_sep
from settings import palette_bgr, palette_rgb
from settings import h_neigh, h_ind, lbp_radius, lbp_points

from tools import onehot2int, int2onehot, load_pickle, dump2pickle

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score


def read_evaluation_data(test_dir, label_dir):

    print('[INFO] Reading test data and labels.')

    filelist = glob(os.path.join(test_dir, '*.png'))
    test_dict = dict()
    label_dict = dict()

    for fn in filelist:
        test_map = cv2.imread(fn, cv2.IMREAD_COLOR)
        label_fn = os.path.join(label_dir, fn.split(
            os.path.sep)[-1].split('.')[0]+".png")
        label_map = cv2.imread(label_fn, cv2.IMREAD_COLOR)
        if not os.path.isfile(label_fn):
            raise ValueError(
                "Label file {} does not exist".format(label_fn))

        def read_classes(map):
            label_layers = []
            for i in classes:
                indices = np.where(map == palette_bgr[i])[:2]
                ones = np.zeros(
                    (map.shape[0], map.shape[1]), np.uint8)
                ones[indices] = 1
                label_layers.append(ones)
            return np.stack([layer for layer in label_layers], axis=-1)

        label_dict[fn] = read_classes(label_map)
        test_dict[fn] = read_classes(test_map)

    return test_dict, label_dict


best_accuracy = -1
best_accuracy_model = " "
best_f1 = -1
best_f1_model = " "

dirs = glob(os.path.join(evaluate_dir, '*', ''))
for test_dir in dirs:
    print(test_dir)
    test_dict, label_dict = read_evaluation_data(test_dir, label_dir)
    for key in test_dict.keys():
        test_2D = test_dict[key]
        label_2D = label_dict[key][h_ind:-h_ind, h_ind:-h_ind]
        test = test_2D.reshape(-1, 3)
        label = label_2D.reshape(-1, 3)
        indices = np.where(label.sum(axis=1) != 1)

        test = np.delete(test, indices, axis=0)
        label = np.delete(label, indices, axis=0)

        test = onehot2int(test)
        label = onehot2int(label)

        metrics = dict()

        cm = confusion_matrix(test, label)
        metrics['cm'] = cm
        f1 = f1_score(test, label, average='weighted')
        metrics['f1'] = f1
        accuracy = accuracy_score(test, label)
        metrics['accuracy'] = accuracy
        precision = precision_score(test, label, average='weighted')
        metrics['precision'] = precision
        recall = recall_score(test, label, average='weighted')
        metrics['recall'] = recall

        with open(os.path.join(test_dir, 'results.txt'), 'w+') as handle:
            for metric in metrics.keys():
                handle.write(metric+":\n")
                handle.write(str(metrics[metric])+"\n")
                handle.write("--------------------------------"+"\n")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_accuracy_model = test_dir
        if f1 > best_f1:
            best_f1 = f1
            best_f1_model = test_dir

print("[INFO] Best (accuracy) model: {}".format(best_accuracy_model))
print("[INFO] Best (f1) model: {}".format(best_f1_model))

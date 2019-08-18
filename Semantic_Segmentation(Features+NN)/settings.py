import numpy as np
import os

oversample = False

dir_prefix = "data"

cache_dir = os.path.join(dir_prefix, "cache")

image_dir = os.path.join(dir_prefix, "images")
label_dir = os.path.join(dir_prefix, "labels")

model_dir = os.path.join(dir_prefix, "model")

output_dir = os.path.join(dir_prefix, "output")
infere_dir = os.path.join(dir_prefix, "infere")

evaluate_dir = os.path.join(dir_prefix, "evaluate")

scaler_fn = "scaler.pkl"
model_fn = "model.pkl"

classes = [0, 1, 2]
class_sep = "#"


def encode1hot(idx):
    vec = np.zeros(len(classes), dtype=np.uint8)
    vec[idx] = 1
    return vec


classes1hot = {idx: encode1hot(idx) for idx in classes}


h_neigh = 11  # haralick neighbourhood
h_ind = int((h_neigh - 1) / 2)

lbp_radius = 24  # local binary pattern neighbourhood
lbp_points = lbp_radius*8

test_size = 0.2
hidden_layers = (5, 5)

random_state = 42


palette_rgb = np.array(
    [
        [128,  64, 128],
        [250, 170,  30],
        [107, 142,  35],
        [70,  70,  70],
        [0,  60, 100],
        [153, 153, 153],
        [220, 220,   0],
        [152, 251, 152],
        [70, 130, 180],
        [220,  20,  60],
        [255,   0,   0],
        [0,   0, 142],
        [0,   0,  70],
        [0,  80, 100],
        [0,   0, 230],
        [119,  11,  32],
        [0,   0,   0],
        [255, 255, 255]
    ], np.uint8)

palette_bgr = palette_rgb[:, ::-1]

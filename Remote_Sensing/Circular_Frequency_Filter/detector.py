"""Implementation of [1].
Args:
    --input:    input file
    --output:   output file
Pipeline:
    Stage 1 (Candidate location extraction):
        1' Filtering with CF filter.
        2' Thresholding filtered image.
        3' Simple shape analyis.
    Stage 2 (Aircraft identification):
        1' Multilevel feature modeling
            A) HUE descriptor
            B) HOG descriptor
        2' SVM
            A) apply classifier to images in spatial pyramid
            B) overlapped boxes suppresing
[1] Feng Gao, Qizhi Xu, and Bo Li, “Aircraft Detection from VHR Images Based on Circle-Frequency Filter and Multilevel Features,”
The Scientific World Journal, vol. 2013, Article ID 917928, 7 pages, 2013.
"""
import imutils
import argparse
import numpy as np
import cv2
from skimage import feature, transform
from scipy.signal import convolve2d


PI = 3.14159265359
SQRT3 = 1.73205080757
# HUE_MIN = -1.5685322119609821
# HUE_MAX = 1.5685322119609821
HUE_MIN = -1.57
HUE_MAX = 1.57

PAPER_WINDOW_SIZE = [32, 32]
PAPER_STEP = 8
PAPER_RADIUS = 12
PAPER_N = 50
PAPER_BINS = 6
PAPER_REPRESENTATION_LEVEL = 3

AREA_THRESHOLD_MAX = 12
AREA_THRESHOLD_MIN = 0

PYRAMID_SCALE = 1.5
PYRAMID_MIN = (300, 300)


# Candidate location extraction functions


def stage1(source: np.ndarray, radius: int = PAPER_RADIUS, N: int = PAPER_N) -> np.ndarray:
    kernel_cos = np.zeros([2*radius + 1, 2*radius + 1])
    kernel_sin = np.zeros([2*radius + 1, 2*radius + 1])
    for k in range(N):
        x = (int)(radius*np.cos(k*2*PI/N)) + radius
        y = (int)(radius*np.sin(k*2*PI/N)) + radius
        kernel_cos[y][x] = np.cos(8*PI*k/N)
        kernel_sin[y][x] = np.sin(8*PI*k/N)
    cos_filtered = convolve2d(source, kernel_cos, fillvalue=0, mode="same")
    sin_filtered = convolve2d(source, kernel_sin, fillvalue=0, mode="same")
    # 1' Filtering with CF filter.
    '''
    image_filtered = 1/N * \
        np.sqrt(np.square(cos_filtered) + np.square(sin_filtered))
    '''
    image_filtered = np.square(cos_filtered) + np.square(sin_filtered)
    # 2' Thresholding filtered image.
    """
    Using thershold value as in [2] instead of OTSU as in [1]. 
    [2] Liu, Liu, and Zhenwei Shi. "Airplane detection based on rotation invariant and sparse coding in remote sensing images."
        Optik-International Journal for Light and Electron Optics 125.18 (2014): 5327-5333.
    """
    _, image_filtered = cv2.threshold(
        image_filtered, image_filtered.max()*0.5, 255, cv2.THRESH_BINARY)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = image_filtered
    result = cv2.dilate(result, structuringElement)
    result = cv2.erode(result, structuringElement)
    result = result.astype(np.uint8)
    _, contours, hierarchy = cv2.findContours(result, 1, 2)
    area_discriminated = np.zeros(result.shape, dtype=np.uint8)
    cnt_thresholded = list()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > AREA_THRESHOLD_MIN:
            cnt_thresholded.append(cnt)
    cv2.drawContours(area_discriminated, cnt_thresholded, -
                     1, color=(255), thickness=cv2.FILLED)
    return area_discriminated


def patch_extractor(image: np.ndarray, window_size: int = PAPER_WINDOW_SIZE,
                    step: int = PAPER_STEP) -> (list, list):
    def sliding_window(image, window_size,
                       step):
        s_y, s_x = window_size[0], window_size[1]
        for i in range(0, image.shape[0] - s_y, step):
            for j in range(0, image.shape[1] - s_x, step):
                patch = image[i:i + s_y, j:j + s_x]
                yield (i, j),  patch
    indices, patch_extractor = zip(*sliding_window(image, window_size=window_size,
                                                   step=step))
    return indices, patch_extractor


def pyramid(image: np.ndarray, scale: float = 2., minSize: tuple = (1, 1)) -> np.ndarray:
    k = 0
    while True:
        scale_k = scale**(-k)
        resized = cv2.resize(image, None, fx=scale_k, fy=scale_k)
        if resized.shape[0] < minSize[0] or resized.shape[1] < minSize[1]:
            break
        k += 1
        yield resized


def non_max_suppression_slow(boxes: tuple, overlapThresh):
    if len(boxes) == 0:
        return []
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]
        for pos in range(0, last):
            j = idxs[pos]
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            overlap = float(w * h) / area[j]
            if overlap > overlapThresh:
                suppress.append(pos)
        idxs = np.delete(idxs, suppress)
    return boxes[pick]

# Stage 2 functions

# prepare HUE descriptor lookup table


def calculate_lookup() -> dict:
    import os
    import pickle
    fn = "lookup_table.pkl"
    if os.path.isfile(fn):
        with open(fn, 'rb') as handle:
            return pickle.load(handle)
    print("Calculating lookup table...")
    lookup = dict()
    for B in range(256):
        for G in range(256):
            for R in range(256):
                numerator = (SQRT3 * (R - G))
                denumerator = (R+G-2*B)
                if (denumerator == 0):
                    lookup[(B, G, R)] = 0
                    continue
                atan = np.arctan(numerator/denumerator)
                lookup[(B, G, R)] = (int)(
                    256*(atan - HUE_MIN)/(HUE_MAX-HUE_MIN))
    with open(fn, 'wb') as handle:
        pickle.dump(lookup, handle)
    return lookup


lookup_hue = calculate_lookup()


def bgr2HUE(image: np.ndarray) -> np.ndarray:
    '''
    Calculates HUE descritptor as described in [1] for every pixel in the input image
    Returns array of HUE descriptors
    Instead of (-1.57;1.57) range uint8 is implemented
    '''
    result = np.zeros([image.shape[0], image.shape[1]], dtype=np.uint8)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            bgr = image[y][x]
            result[y][x] = lookup_hue[(bgr[0], bgr[1], bgr[2])]
    return result


def feature_extractor(image: np.ndarray, hue_map: np.ndarray, hue_bins: int = PAPER_BINS):
    '''
    Multilevel features:
    A) HUE descriptor
    B) HOG descriptor
    '''
    def hue_histogram(hue_map: np.ndarray, bins: hue_bins) -> np.ndarray:
        histogram, _ = np.histogram(hue_map, bins=bins, range=(0, 256))
        return histogram

    def hog(image: np.ndarray):
        return feature.hog(image, pixels_per_cell=(image.shape[0], image.shape[1]), cells_per_block=(1, 1), block_norm="L1")

    representation_level = PAPER_REPRESENTATION_LEVEL - 1

    result = np.hstack([hue_histogram(hue_map, hue_bins), hog(image)])

    def mulitlevel(image, hue_map, result, level, levels):
        if level < 1:   # base case
            return result
        else:
            subcell_size = (int)(image.shape[0]/2**(level))
            _, image_cells = patch_extractor(image, window_size=[
                subcell_size, subcell_size], step=subcell_size-1)
            _, hue_cells = patch_extractor(hue_map, window_size=[
                subcell_size, subcell_size], step=subcell_size-1)
            for image_cell, hue_cell in zip(image_cells, hue_cells):
                result = np.hstack(
                    [hue_histogram(hue_cell, hue_bins), hog(image_cell), result])
            return mulitlevel(image, hue_map, result, level-1, levels)

    result = mulitlevel(image, hue_map, result,
                        representation_level, representation_level)

    return result


if __name__ == "__main__":
    print(__doc__)
    parser = argparse.ArgumentParser(
        description="Aircraft detector")
    parser.add_argument("--input", type=str, default='input/test.bmp')
    parser.add_argument("--output", type=str, default='output/test.bmp')
    args = parser.parse_args()

    from sklearn.externals import joblib
    scaler = joblib.load("scaler.pkl")
    classifier = joblib.load("classifier.pkl")

    image = cv2.imread(args.input, 0)
    image_bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)

    s_y, s_x = (int)(PAPER_WINDOW_SIZE[0]), (int)(PAPER_WINDOW_SIZE[1])
    features_length = feature_extractor(
        np.zeros(PAPER_WINDOW_SIZE), np.zeros(PAPER_WINDOW_SIZE)).shape[0]

    # Get an array of aircraft candidates
    print("Stage 1: Candidate location extraction... ")
    candidates_location = dict()
    images_resized = dict()
    images_filtered = dict()
    image_pyramid = pyramid(image, scale=PYRAMID_SCALE, minSize=PYRAMID_MIN)
    for (k, resized) in enumerate(image_pyramid):
        h, w = resized.shape
        print(resized.shape)
        resized_filtered = stage1(resized, radius=PAPER_RADIUS, N=PAPER_N)
        images_resized[k] = resized
        images_filtered[k] = resized_filtered
        candidate_indices = list()
        indices, patches = patch_extractor(
            resized_filtered, PAPER_WINDOW_SIZE, PAPER_STEP)
        for indice, patch in zip(indices, patches):
            if len(np.where(patch > 0)[0]):
                candidate_indices.append((indice[0], indice[1]))
        candidates_location[k] = candidate_indices

    # stage 2
    print("Stage 2: SVM detection in candidate regions... ")
    gray_resized = images_resized

    predictions = dict()
    results = dict()
    detections = dict()
    for (k, bgr_resized) in enumerate(pyramid(image_bgr, scale=PYRAMID_SCALE, minSize=PYRAMID_MIN)):
        detections_k = list()
        print(bgr_resized.shape)
        hue_resized = bgr2HUE(bgr_resized)
        indices = candidates_location[k]
        features = np.array([]).reshape([0, features_length])
        for (i, j) in indices:
            i2 = i + s_y
            j2 = j + s_x
            patch = gray_resized[k][i:i2, j:j2]
            hue_patch = hue_resized[i:i2, j:j2]
            features = np.vstack(
                [features, feature_extractor(patch, hue_patch)])
        if not indices:
            predictions[k] = None
            results[k] = image_bgr.copy()
            detections[k] = None
            continue
        predictions[k] = classifier.predict(scaler.transform(features))
        scale = 1/PYRAMID_SCALE**(-k)
        bounding_boxes = np.array([], dtype=np.int).reshape(0, 4)
        for _, idx in np.ndenumerate(np.where(predictions[k] > 0)):
            i, j, i2, j2 = (indices[idx][0], indices[idx]
                            [1], indices[idx][0]+s_x, indices[idx][1]+s_y)
            i, j, i2, j2, = (int)(scale * i), (int)(scale *
                                                    j), (int)(scale * i2), (int)(scale * j2)
            bounding_boxes = np.vstack([bounding_boxes, [j, i, j2, i2]])
        non_overlapping_boxes = non_max_suppression_slow(bounding_boxes, 1)
        detections[k] = non_overlapping_boxes
    import random

    def random_color():
        rgbl = [255, 0, 0]
        random.shuffle(rgbl)
        return tuple(rgbl)
    color = random_color()
    '''
    result = image_bgr.copy()
    for k in detections.keys():
        color = random_color()
        for (startX, startY, endX, endY) in detections[k]:
            cv2.rectangle(result, (startX, startY),
                          (endX, endY), color, 2)
    '''
    result = image_bgr.copy()
    for (startX, startY, endX, endY) in non_max_suppression_slow(np.vstack([array for array in detections.values()]), 0.5):
        cv2.rectangle(result, (startX, startY), (endX, endY), color, 2)
    cv2.imwrite(args.output, result)

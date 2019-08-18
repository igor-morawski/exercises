import cv2
import sys
import os
import numpy as np


def SAM(vec1, vec2):
    """Spectral angle mapper
    The spectral angle has a maximum value 1.57 and minimum value of 0.
    Parameters
    ----------
    vec1, vec2 : array-like, dtype=float, shape=n
    """
    # manually compute cosine similarity
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    dot = np.dot(vec1, vec2)
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)
    return np.arccos(dot/(vec1_norm*vec2_norm))


def SID(vec1, vec2):
    """Spectral information divergence
    ----------
    vec1, vec2 : array-like, dtype=float, shape=n
    """
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)

    def D_KL(p, q):
        """Kullback-Leibler divergence D(P || Q) for discrete distributions
        Parameters
        ----------
        p, q : array-like, dtype=float, shape=n
        Discrete probability distributions.
        https://gist.github.com/swayson/86c296aa354a555536e6765bbe726ff7
        """
        p = np.asarray(p)
        q = np.asarray(q)
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    return D_KL(vec1, vec2) + D_KL(vec2, vec1)


def OPD(vec1, vec2):
    """Orthogonal projection divergence
    ----------
    vec1, vec2 : array-like, dtype=float, shape=n
    """
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)

    def P(vec):
        return np.eye(vec.shape[0]) - vec.T/np.dot(vec.T, vec)
    return np.dot(vec1.T, np.dot(P(vec2), vec1)) + np.dot(vec1.T, np.dot(P(vec2), vec1))


def cov_for(img_cube):
    X = img_cube.astype(np.float32)
    mean = np.zeros(X.shape[0])
    N = X.shape[1] * X.shape[2]
    for band in range(img_cube.shape[0]):
        mean[band] = np.sum(img_cube[band])/N
    for band in range(img_cube.shape[0]):
        X[band] -= mean[band]
    X = X.reshape([img_cube.shape[0], -1])
    return np.dot(X, X.T)/(N-1)


def cov(img_cube):
    X = img_cube.reshape([img_cube.shape[0], -1]).astype(np.float32)
    N = X.shape[1]
    X -= X.mean(axis=1)[(slice(None), np.newaxis)]
    return np.dot(X, X.T)/(N-1)


def corr(img_cube):
    c = cov(img_cube)
    c /= np.sqrt(np.multiply.outer(np.diag(c), np.diag(c)))
    return c


def mean(img_cube):
    X = img_cube.reshape([img_cube.shape[0], -1])
    return X.mean(axis=1)


def CMD(vec1, vec2, cov):
    """Covariance-Mahalanobis distance
    ----------
    vec1, vec2: array-like, dtype = float, shape = n
    cov: array_like, shape = [n, n]
    """
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    vec_diff = vec1 - vec2
    cov_inv = np.linalg.inv(cov)
    return np.dot(vec_diff.T, np.dot(cov_inv, vec_diff))


def RMD(vec1, vec2, corr):
    """Correlation-Mahalanobis distance
    ----------
    vec1, vec2: array-like, dtype = float, shape = n
    corr: array_like, shape = [n, n]
    """
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    return CMD(vec1, vec2, corr)


def CMFD(vec1, vec2, cov, mean):
    """Covariance matched filter-based distance
    ----------
    vec1, vec2: array-like, dtype = float, shape = n
    corr: array_like, shape = [n, n]
    mean: array_like, shape = n
    """
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    cov_inv = np.linalg.inv(cov)
    u = vec1 - mean
    v = vec2 - mean
    return np.dot(u.T, np.dot(cov_inv, v))


def RMFD(vec1, vec2, corr, mean):
    """Correlation matched filter-based distance
    ----------
    vec1, vec2: array-like, dtype = float, shape = n
    corr: array_like, shape = [n, n]
    mean: array_like, shape = n
    """
    corr_inv = np.linalg.inv(corr)
    return np.dot(vec1.T, np.dot(corr_inv, vec2))


def SCM(vec1, vec2, mean):
    """Spectral Correlation  Mapper
    Parameters
    ----------
    vec1, vec2, mean : array-like, dtype=float, shape=n
    Reference: https://pdfs.semanticscholar.org/4a3b/dae7ce9ed62751e1d415687aac763f062310.pdf?_ga=2.7871739.89944739.1559639587-1285175875.1559639587
    """
    vec1 = np.asarray(vec1) - mean
    vec2 = np.asarray(vec2) - mean
    dot = np.dot(vec1, vec2)
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)
    return np.arccos(dot/(vec1_norm*vec2_norm))


def normalized_crosscorelation(vec1, vec2):
    assert(len(vec1) == len(vec2))
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    vec1 = (vec1 - np.mean(vec1)) / (np.std(vec1))
    vec2 = (vec2 - np.mean(vec1)) / (np.std(vec2))
    return float(np.correlate(vec1, vec2, 'valid'))


if __name__ == "__main__":
    from osgeo import gdal
    print("[INFO] Reading image...")
    ds = gdal.Open(os.path.join("052298844010_01_P001_MUL",
                                "09DEC10103019-M2AS-052298844010_01_P001.TIF"))
    img_cube = ds.ReadAsArray()

    (channels, height, width) = img_cube.shape
    print("[INFO] Loaded in {} x {} x {} image cube.".format(
        channels, height, width))
    pt1, pt2 = (670, 40), (730, 300)
    (y1, x1), (y2, x2) = pt1, pt2

    ''''''

    display_band = 8
    display_img = (img_cube[display_band-1]/(2 ** (10-8))).astype(np.uint8)
    display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)
    # draw circle
    cv2.circle(display_img, pt1, 10, (0, 255, 0))
    cv2.circle(display_img, pt2, 10, (0, 255, 0))
    cv2.putText(display_img, "{}, {}".format(y1, x1), pt1,
                cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 255, 0))
    cv2.putText(display_img, "{}, {}".format(y2, x2), pt2,
                cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 255, 0))
    display_img = cv2.resize(display_img, None, fx=0.8, fy=0.8)
    cv2.imshow("Band {}".format(display_band), display_img)
    cv2.waitKey(100)

    ''''''

    print("[INFO] Calculating similarity measures for pts {} and {}.".format(pt1, pt2))
    vec1, vec2 = img_cube[:, y1, x1], img_cube[:, y2, x2]
    print("vec1: {} \nvec2: {}".format(vec1, vec2))
    print("Similarity measures:")
    print("1° SAM:  {}".format(SAM(vec1, vec2)))
    print("2° SID:  {}".format(SID(vec1, vec2)))
    print("3° OPD:  {}".format(OPD(vec1, vec2)))
    print("4° CMD:  {}".format(CMD(vec1, vec2, cov(img_cube))))
    print("5° RMD:  {}".format(RMD(vec1, vec2, corr(img_cube))))
    print("6° CMFD:  {}".format(CMFD(vec1, vec2, cov(img_cube), mean(img_cube))))
    print("7° RMFD:  {}".format(RMFD(vec1, vec2, corr(img_cube), mean(img_cube))))
    print("\n and additional similariy measures:")
    print("8° SCM:  {}".format(SCM(vec1, vec2, mean(img_cube))))
    print("8° Normalized Cross-corelation:  {}".format(normalized_crosscorelation(vec1, vec2)))

    ''''''

    print("\n\n cov_for and cov equal: {}".format(
        np.allclose(cov(img_cube), cov_for(img_cube))))

    import timeit
    setup = '''
from osgeo import gdal
import os
import numpy as np
ds = gdal.Open(os.path.join("052298844010_01_P001_MUL",
                            "09DEC10103019-M2AS-052298844010_01_P001.TIF"))
img_cube = ds.ReadAsArray()

def cov_for(img_cube):
    X = img_cube.astype(np.float32)
    mean = np.zeros(X.shape[0])
    N = X.shape[1] * X.shape[2]
    for band in range(img_cube.shape[0]):
        mean[band] = np.sum(img_cube[band])/N
    for band in range(img_cube.shape[0]):
        X[band] -= mean[band]
    X = X.reshape([img_cube.shape[0], -1])
    return np.dot(X, X.T)/(N-1)


def cov(img_cube):
    X = img_cube.reshape([img_cube.shape[0], -1]).astype(np.float32)
    N = X.shape[1]
    X -= X.mean(axis=1)[(slice(None), np.newaxis)]
    return np.dot(X, X.T)/(N-1)
    '''
    print("cov() execution time: {0:.5f}".format(
        min(timeit.Timer('cov(img_cube)', setup=setup).repeat(5, 5))))
    print("cov_for() execution time: {0:.5f}".format(
        min(timeit.Timer('cov_for(img_cube)', setup=setup).repeat(5, 5))))

    print("Press key to continue...")
    cv2.waitKey()
    cv2.destroyAllWindows()

import cv2
import sys
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from itertools import cycle, combinations

n_components = None
visualize = False
numVectors = 10000
# numVector = 100
numEM = 4
vis_max = 10

fx, fy = 1, 1
colors = ["b", "orange", "g", "r", "c", "m", "y"]

fn = "Urban_F210"
#fn = "samson_1"
res_dir = fn+"_results"


def PPI(img_pca, numVectors, numEM):
    '''
    https://pdfs.semanticscholar.org/c518/0325ca012a2b2bd78016ec21122e4b855292.pdf
    '''
    M = img_pca.T
    M = np.matrix(M, dtype=np.float64)
    components, N = M.shape
    Mm = (M - M.mean(axis=1))/M.std()
    vecs = np.random.rand(components, numVectors)
    votes = np.zeros((N, 1))
    '''
    for i in range(vecs.shape[1]):
        projection = abs(vecs[:, i]*Mm)
        idx = np.argmax(projection)
        votes[idx] = votes[idx] + 1
    '''
    for i in range(vecs.shape[1]):
        vec = vecs[:, i]
        # projection = np.abs(np.dot(Mm.T, vec) / np.linalg.norm(vec))
        # looking for maxima, dividing doesn't change the result
        projection = np.abs(np.dot(Mm.T, vec))
        idx = np.argmax(projection)
        votes[idx] = votes[idx] + 1

    max_idx = np.argsort(votes, axis=None)
    endmember_idx = max_idx[-numEM:][::-1]
    return endmember_idx


def plot_mnf(img_src, band, m, n):
    img_abs = np.abs(img_src.T[band-1].reshape(m, n))
    plot_img = (img_abs/img_abs.max() * 255).astype(np.uint8)
    # plot_img = cv2.cvtColor(plot_img, cv2.COLOR_GRAY2RGB)
    plot_img = cv2.resize(plot_img, None, fx=fx, fy=fy)
    '''
    cv2.imshow("Band {}".format(band), plot_img)
    cv2.waitKey()
    '''
    cv2.imwrite(os.path.join(
        res_dir, "MNF_band{}.png".format(int(band))), plot_img)
    return


def save_filtered(img_src, m, n, name):
    img_abs = np.abs(img_src.T.reshape(m, n))
    img = (img_abs/img_abs.max() * 255).astype(np.uint8)
    img = cv2.resize(img, None, fx=fx, fy=fy)
    cv2.imwrite(os.path.join(
        res_dir, "{}.png".format(name)), img)
    return


def plot_EM(img_flattened, indices):
    bandNum = img_flattened.shape[0]
    color = cycle(colors)
    M = img_flattened.T
    # normalize reflectance
    M = M/M.max()
    endmembers = []
    bands = [int(x) for x in range(bandNum)]
    for i, endmember_idx in enumerate(indices):
        endmembers.append(M[endmember_idx])
        plt.plot(bands, M[endmember_idx], next(color), label='EM{}'.format(i))
    plt.xlabel('Band №')
    plt.ylabel('Reflectance normalized')
    plt.title('Spectral Profile')
    plt.legend()
    plt.savefig(os.path.join(res_dir, "EM_profiles.png"))
    plt.clf()
    return


def cov(img_flattened):
    X = img_flattened.astype(np.float32)
    N = X.shape[1]
    X -= X.mean(axis=1)[(slice(None), np.newaxis)]
    return np.dot(X, X.T)/(N-1)


def corr(img_flattened):
    c = cov(img_flattened)
    c /= np.sqrt(np.multiply.outer(np.diag(c), np.diag(c)))
    return c


def whiten(img_flattened):
    U, S, V = np.linalg.svd(cov(img_flattened))
    S_sqrt_diag = np.diag(np.sqrt(S).T)
    Aw = np.dot(V, np.dot(S_sqrt_diag, V.T))
    return np.dot(img_flattened.T, Aw)


def MNF(img_flattened):
    M = whiten(img_flattened)
    transform = PCA(n_components=n_components, whiten=True)
    return transform.fit_transform(M)


def plot_datacloud(img_flattened, bandX=0, bandY=1, endmember_idx=None):

    X = img_flattened[bandX]
    Y = img_flattened[bandY]
    plt.plot(X, Y, '.', color="gray")
    if endmember_idx is not None:
        color = cycle(colors)
        for i, idx in enumerate(endmember_idx):
            plt.plot(X[idx], Y[idx], '.', color=next(
                color), label='EM{}'.format(i))
    plt.xlabel('Band {}'.format(int(bandX)))
    plt.ylabel('Band {}'.format(int(bandY)))
    plt.legend()
    plt.title('Spectral Data Cloud')
    plt.savefig(os.path.join(
        res_dir, "spectral_datacloud_{}_{}.png".format(int(bandX), int(bandY))))
    plt.clf()
    return


def endmembers_idx2vec(img_flattened, indices):
    result = []
    for idx in indices:
        result.append(img_flattened[:, idx])
    return result


def OSP(img_flattened, U, d):
    '''
    d_OSP = d.T dot Pu_ort dot r
    '''
    Pu_ort = np.dot(U, np.linalg.pinv(U))
    A = d.reshape(1, -1)
    B = np.einsum('ij,ik->jk', Pu_ort, img_flattened)
    result = np.einsum('ij, jk -> ik', A, B)
    return result


def CEM(img_flattened, d):
    R = corr(img_flattened)
    num = np.dot(np.linalg.inv(R), d)
    denom = np.dot(d.T, num)
    w_cem = num/denom
    w_cem_T = w_cem.reshape(1, -1)
    result = np.einsum('ij, jk -> ik', w_cem_T, img_flattened)
    return result


def corr(img_cube):
    c = cov(img_cube)
    c /= np.sqrt(np.multiply.outer(np.diag(c), np.diag(c)))
    return c


if __name__ == "__main__":
    from osgeo import gdal
    print("[INFO] Reading image...")
    ds = gdal.Open(os.path.join(fn, fn+".img"))
    img_cube = ds.ReadAsArray()

    (channels, height, width) = img_cube.shape
    print("[INFO] Loaded in {} x {} x {} image cube.".format(
        channels, height, width))

    img_flattened = img_cube.reshape([img_cube.shape[0], -1])

    # img_pca = pca.fit_transform(img_flattened.T)

    img_mnf = MNF(img_flattened)

    ''''''
    if visualize:
        for band in range(min(vis_max, img_mnf.shape[1])):
            plot_mnf(img_mnf, band, height, width)
    ''''''

    endmember_idx = PPI(img_mnf, numVectors, numEM)

    if visualize:
        plot_EM(img_flattened, endmember_idx)
        interesting_bands = [0, 1, 50, 118]
        for bandX, bandY in combinations(interesting_bands, 2):
            plot_datacloud(img_flattened, bandX=bandX, bandY=bandY,
                           endmember_idx=endmember_idx)

    endmembers = endmembers_idx2vec(img_flattened, endmember_idx)

    d = endmembers[0]
    U = []
    for endmember in endmembers:
        if np.array_equal(endmember, d):
            continue
        U.append(endmember.reshape(1, -1))
    U = np.vstack(U).T

    OSP_filtered_flattened = OSP(img_flattened, U, d)
    CEM_filtered_flattened = CEM(img_flattened, d)
    if True:
        save_filtered(OSP_filtered_flattened, height, width, "OSP")
        save_filtered(CEM_filtered_flattened, height, width, "CEM")

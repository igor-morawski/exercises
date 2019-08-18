import numpy as np
import math
from scipy.spatial.distance import euclidean as measure
from scipy.spatial.distance import cdist as distance
eps = 0.001


def ED(series1, series2):
    return np.linalg.norm(series1-series2)


def L_inf(series1, series2):
    return max(abs(series1-series2))


def LCSS(series1, series2, eps):
    S = np.zeros([len(series1)+1, len(series2)+1])
    for i in range(1, len(series1)+1):
        for j in range(1, len(series2)+1):
            if np.all(np.abs(series1[i-1]-series2[j-1]) < eps) and (
                    np.abs(i-j) < 0.01):
                S[i, j] = S[i-1, j-1]+1
            else:
                S[i, j] = max(S[i, j-1], S[i-1, j])
    return 1-S[len(series1), len(series2)]/min(len(series1), len(series2))


def EDR(series1, series2, eps):
    N = len(series1)+1
    M = len(series2)+1
    C = np.zeros((N, M))
    for i in range(1, N):
        for j in range(1, M):
            x0 = series1[i-1]
            x1 = series2[j-1]
            if ED(x0, x1) < eps:
                subcost = 0
            else:
                subcost = 1
            C[i, j] = min(min(C[i, j-1]+1, C[i-1, j]+1), C[i-1, j-1]+subcost)
    return (C[N-1, M-1])/max(N-1, M-1)


def ERP(series1, series2, g):
    N = len(series1)+1
    M = len(series2)+1
    C = np.zeros((N, M))
    edge = 0
    for i in range(1, N):
        edge += np.abs(ED(series1[i-1], g))
    C[1:, 0] = edge
    for i in range(1, N):
        for j in range(1, M):
            x0 = series1[i-1]
            x1 = series2[j-1]
            erp0 = C[i-1, j] + ED(x0, g)
            erp1 = C[i, j-1] + ED(x1, g)
            erp01 = C[i-1, j-1] + ED(x0, x1)
            C[i, j] = min(erp0, min(erp1, erp01))
    return C[N-1, M-1]


def DTW(series1, series2, warp=1):
    series1, series2 = series1.reshape(-1, 1), series2.reshape(-1, 1)
    M = len(series1)+1
    N = len(series2)+1
    D0 = np.zeros([M, N])
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    D1 = D0[1:, 1:]
    D0[1:, 1:] = distance(series1, series2, measure)
    for i in range(M-1):
        for j in range(N-1):
            mins = [D0[i, j]]
            for k in range(1, warp + 1):
                mins += [D0[min(i + k, M-1), j],
                         D0[i, min(j + k, N-1)]]
            D1[i, j] += min(mins)
    return D1[-1, -1] / sum(D1.shape)


if __name__ == "__main__":
    ts1 = np.array([0, 1, -1, 0, 0])
    ts2 = np.array([0, 0, 1, -1, 0])
    ts3 = ts1 + np.random.normal(0, 0.1, len(ts1))

    print("Similarity measures for time series.")
    print("Time series:\n ts1 = {}\n ts2 = {}\n ts3 = ts1 + gauss_noise = {}".format(ts1, ts2, ts3))
    print("")

    print("1° ED_1,2 = {}".format(ED(ts1, ts2)))
    print("1° ED_1,3 = {}".format(ED(ts1, ts3)))
    print("1° ED_2,3 = {}".format(ED(ts2, ts3)))

    print("")

    print("2° L-inf_1,2 = {}".format(L_inf(ts1, ts2)))
    print("2° L-inf_1,3 = {}".format(L_inf(ts1, ts3)))
    print("2° L-inf_2,3 = {}".format(L_inf(ts2, ts3)))

    print("")

    print("3° LCSS_1,2 = {}".format(LCSS(ts1, ts2, eps)))
    print("3° LCSS_1,3 = {}".format(LCSS(ts1, ts3, eps)))
    print("3° LCSS_2,3 = {}".format(LCSS(ts2, ts3, eps)))

    print("")

    print("4° EDR_1,2 = {}".format(EDR(ts1, ts2, eps)))
    print("4° EDR_1,3 = {}".format(EDR(ts1, ts3, eps)))
    print("4° EDR_2,3 = {}".format(EDR(ts2, ts3, eps)))

    print("")

    g = np.zeros(len(ts1))

    print("5° ERP_1,2 = {}".format(ERP(ts1, ts2, g)))
    print("5° ERP_1,3 = {}".format(ERP(ts1, ts3, g)))
    print("5° ERP_2,3 = {}".format(ERP(ts2, ts3, g)))

    print("")

    print("6° DTW_1,2 = {}".format(DTW(ts1, ts2)))
    print("6° DTW_1,3 = {}".format(DTW(ts1, ts3)))
    print("6° DTW_2,3 = {}".format(DTW(ts2, ts3)))

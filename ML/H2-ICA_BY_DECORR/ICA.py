'''
ICA by Maximum Likelihood (infomax according to Bell and Sejnowski )

Assume linear mixing
        x = As
we search for W=A^-1, so
        s = Wx

* assuming
p_s(s_i) can be approximated by sigmoid(s_i)
'''
import numpy as np
import re 
import matplotlib.pyplot as plt
from numpy.linalg import det, pinv
import argparse
from scipy.linalg import sqrtm

DATA_FN = 'ICA_2_mess.dat'
ITER_MAX = 300
LEARNING_RATE_0 = 1

COLOR_MIXED = 'g'
COLOR_UNMIXED = 'r'

COLOR_SIGNAL1 = 'b'
COLOR_SIGNAL2 = 'm'

SUBSAMPLE_STEP = 100

def sigmoid(z):
    g = 1.0 / (1.0 + np.exp(-z))
    return g

def sigmoid_grad(z):
    g = 1.0 / (1.0 + np.exp(-z))
    return g*(1-g)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', action='store_true')
    args = parser.parse_args()
    COMPUTE = not args.load

    #read data
    with open(DATA_FN, 'r') as handle:
        data2parse = handle.read()

    def data_parser(str, separator='\n'):
        str = re.sub("\[|\]","", str)
        separator = '\n'
        str = str.split(separator)
        array = []
        for line in str:
            row = []
            if line:
                for value in line.split(','):
                    row.append(value)
                array.append(tuple(row))
        return np.asarray(array).astype(np.float64)
    
    y = data_parser(data2parse)
    '''
    testing
    t = np.linspace(0, 10, len(y[:,0]))
    y[:,0] = np.sin(5*t)
    y[:,1] = np.power(np.sin(np.sqrt(10)*t), 15)
    A = np.random.rand(2,2)
    y = np.dot(y, A.T)
    '''
    # normalize
    y = y/np.abs(y).max()
    N, M = y.shape

    # plot data
    plt.title('Scatter plot of mixed data')
    plt.scatter(y[:,0], y[:,1], c=COLOR_MIXED)
    plt.xlabel('y_0')
    plt.ylabel('y_1')
    plt.savefig('mixed_data')
    plt.clf()

    # plot signals
    plt.title('Mixed signals (subsampled)')
    plt.plot(y[::SUBSAMPLE_STEP][:,0],c=COLOR_SIGNAL1)
    plt.plot(y[::SUBSAMPLE_STEP][:,1],c=COLOR_SIGNAL2)
    plt.xlabel('t')
    plt.ylabel('s')
    plt.savefig('mixed_signals')
    plt.clf()


    # assume linear mixing x = As, then demixing matrix is W = A^-1 which we search for
    W = np.eye(M,M)
    
    if COMPUTE: 
        # just save time by taking just some data, maybe randomly but for now the first n samples
        _y = y
        y =  (y - y.mean(axis=0))
        # SVD
        U, s, Vt = np.linalg.svd(y, full_matrices=False)
        v_whitened = np.dot(U, Vt)
        theta = np.radians(np.arange(0, 180, 5))
        covs_c = []
        angles = []
        for angle in theta:
            c, s = np.cos(angle), np.sin(angle)
            rot = np.array([[c,-s],[s,c]])
            v_fin = np.dot(v_whitened, rot)
            # arbitrary function f = tanh
            cov_f = np.abs(np.cov(np.tanh(v_fin)))
            covs_c.append(cov_f.sum()-np.diag(cov_f).sum())
            angles.append(angle)
        covs_c = np.array(covs_c)
        best_angle = angles[np.argmin(covs_c)]
        c, s = np.cos(best_angle), np.sin(best_angle)
        rot = np.array([[c,-s],[s,c]])
        s = np.dot(v_whitened, rot)
        
        
    # ICA by decorrelation

    if not COMPUTE:
        pass
    
    #y = _y
    #s = np.dot(y, W.T)
    s = s/np.abs(s).max()
    s[::1][:,0] = s[::1][:,0]/np.abs(s[::1][:,0]).max()
    s[::2][:,0] = s[::2][:,0]/np.abs(s[::2][:,0]).max()

    # plot result
    plt.title('Scatter plot of unmixed data')
    plt.scatter(s[:,0], s[:,1], c=COLOR_UNMIXED)
    plt.xlabel('s_0')
    plt.ylabel('s_1')
    plt.savefig('unmixed_data')
    plt.clf()

plt.title('Unmixed signals (subsampled)')
plt.plot(s[::1][:,0],c=COLOR_SIGNAL1)
plt.plot(s[::2][:,0],c=COLOR_SIGNAL1)
plt.xlabel('t')
plt.ylabel('s')
plt.savefig('unmixed_signals')
plt.clf()

    
    

    

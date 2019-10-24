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

    DATA_FN = 'ICA_2_mess.dat'
    
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
        # SG ascent
        '''
        L(W) = \Sigma_(i=1)^m ( \Sigma_(j=1)^n log g'(w^T x^((j)) + log(|W|)) )
        L(W) = \Sigma_(i=1)^m ( \Sigma_(j=1)^n log pdf_approx_grad + log(|W|)) )
        a_n+1 = a_n - LEARNING_RATE * grad(F(a_n))
        LEARNING_RATE = LEARNING_RATE_0/(1+DECAY_RATE*EPOCH)
        '''

        def decay(eta, current_iter, decay_rate):
            return eta/(1+decay_rate*current_iter)

        LEARNING_RATE = LEARNING_RATE_0
        s = np.matmul(y,W)
        for iter in range(ITER_MAX):
            y_shuffled = np.take(y, np.random.permutation(y.shape[0]), axis=0)
            for sample in y_shuffled:
                W = W + LEARNING_RATE*(np.outer(1-2*sigmoid(np.dot(W,sample.T)), sample) + pinv(W.T))
            LEARNING_RATE = decay(LEARNING_RATE_0, iter, 1e2)
        np.save("W.npy", W)

    if not COMPUTE:
        W = np.load("W.npy")
        
    s = np.dot(y, W.T)

    s = s/np.abs(s).max()
    
    # normalize signals separately!
    s[:,0] = s[:,0]/np.abs(s[:,0]).max()
    s[:,1] = s[:,1]/np.abs(s[:,1]).max()
    

    # plot result
    plt.title('Scatter plot of unmixed data')
    plt.scatter(s[:,0], s[:,1], c=COLOR_UNMIXED)
    plt.xlabel('s_0')
    plt.ylabel('s_1')
    plt.savefig('unmixed_data')
    plt.clf()

        # plot signals
    plt.title('Unmixed signals (subsampled)')
    plt.plot(s[::1][:,0],c=COLOR_SIGNAL1)
    plt.plot(s[::1][:,1],c=COLOR_SIGNAL2)
    plt.xlabel('t')
    plt.ylabel('s')
    plt.savefig('unmixed_signals')
    plt.clf()

    
    

    

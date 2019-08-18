# Python module written for homework

# import numpy to get dimensions, etc.
import numpy as np
from math import exp

ROBERTS = 0
PREWITT = 1
SOBEL = 2


'''def clip(value):
    if value > 255:
        return 255
    if value < 0:
        return 0
    return abs(int(value))
'''
def clip(value):
    if value > 255:
        return 255
    elif value < 0:
        return 0
    else:
        return abs(int(value))

def convolve(src, operator, no_abs=False):
    img = np.zeros(src.shape)
    kernel = np.copy(operator)
    y_middle = int(kernel.shape[0]/2)
    x_middle = int(kernel.shape[1]/2)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            accumulator = 0.
            for t in range(-y_middle, kernel.shape[0]-y_middle):
                for s in range(-x_middle, kernel.shape[1]-x_middle):
                    yPad = 0
                    xPad = 0
                    while (y + t + yPad < 0):
                        yPad += 1
                    while (x + s + xPad < 0):
                        xPad += 1
                    while (y + t + yPad >= img.shape[0]):
                        yPad -= 1
                    while (x + s + xPad >= img.shape[1]):
                        xPad -= 1
                    accumulator += src[y + t + yPad, x + s + xPad]*kernel[t + y_middle, s + x_middle]
            #by default clip, for 2D filtering e.g. Gaussian, mean filtering, etc.
            if no_abs is False:
                img[y,x] = clip(accumulator)
            #for gradient calculation leave unclipped
            if no_abs is True:
                img[y,x] = accumulator
    return img


def getGradientOperator(type):
    if type is ROBERTS:
        return np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]], np.int8), \
        np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]], np.int8)
        return np.array([[1, 0], [0, -1]], np.int8), np.array([[0, 1], [-1, 0]], np.int8)
    if type is PREWITT:
        return np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], np.int8), \
        np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.int8)
    if type is SOBEL:
        # KERNELS IN Y- AND X- DIRECTIONS
        return np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int8), \
        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int8)

def color2gray(src):
    img = np.zeros((src.shape[0], src.shape[1]))
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            accumulator = 0
            for ch in range(src.shape[2]):
                accumulator += src[y, x, ch]
            accumulator/=3
            img[y, x] = clip(accumulator)
    return img
    
def getGaussianKernel(size, sigma):
    #get Gaussian kernel and normalize it
    kernel = np.zeros(size)
    y_middle = int(kernel.shape[0]/2)
    x_middle = int(kernel.shape[1]/2)
    two_sigma_squared =(2*sigma*sigma)
    accumulator = 0.
    for t in range(-y_middle, y_middle+1):
        for s in range(-x_middle, x_middle+1):
            value = exp(-(s*s+t*t)/two_sigma_squared)
            kernel[t+y_middle, s+x_middle] = value
            accumulator+= value
    return kernel/accumulator

def getGsGd(y, x, type=SOBEL):
    iy = np.copy(y)
    ix = np.copy(x)
    return np.abs(ix)+np.abs(iy), np.zeros(iy.shape)
    #directions non needed for the homework
    #so empty array is returned instead of calculation
    directions = np.arctan2(iy,ix)    
    if type is ROBERTS:
        directions-=(3/4*np.pi)
    #round directions to 0, 45, 90, 135 degrees DIRECTION QUANTIZATION
    directions = (np.round(directions*180/np.pi/45)*45)
    for y in range(directions.shape[0]):
        for x in range(directions.shape[1]):
            if directions[y,x] == 180:
                directions[y,x]=0
            if directions[y,x] < 0:
                directions[y,x]+=180
    return np.abs(ix)+np.abs(iy), directions
    
def threshold(image, threshold):
    result = np.zeros(image.shape)
    for y in range(0, image.shape[0]):
        for x in range(0, image.shape[1]):
            if image[y,x] > threshold:
                result[y,x] = 255
    return result



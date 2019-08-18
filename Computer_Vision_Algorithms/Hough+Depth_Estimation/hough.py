#Hough Lines implementation for HW3
from math import ceil, sqrt
import numpy as np
from imageprocessing import gray2color
from cv2 import line as openCV_line
from random import randint

def lines(image, threshold, demo = True, threshold_Relative = True, rho_resolution = 1, theta_resolution = np.deg2rad(1)):
    '''
    Hough Lines implementation

    Input: binary image of edges
    if threshold_Relative is set to True pass relative threshold in range [0; 255]
    Output: list of lines detected (rho, theta)
    if demonstrative is set to True returns accumulator for display also
    '''
    #1. Calculate (rho_range = [-rho_max, rho_max], theta_range = [-pi/2; pi/2])

    rho_max = ceil(sqrt(image.shape[0]**2+image.shape[1]**2))
    theta_max = np.deg2rad(90)

    theta_range = np.arange(-theta_max, theta_max+theta_resolution, theta_resolution)
    rho_range = np.arange(-rho_max, rho_max+rho_resolution, rho_resolution, dtype=np.float32)

    #2. Create accumulator
    accumulator = np.zeros((rho_range.shape[0],theta_range.shape[0]), dtype=np.uint32)

    #3. We're using binary image to speed up calculations (instead of going through
    #all rhos and thetas we calculate it's rho and all thetas for existing edges) and increasing
    #accumulator
    for y in range(0, image.shape[0]):
        for x in range(0, image.shape[1]):
                if image[y,x] != 0:
                    #Calculate lines and increase accumulator
                    for id_theta in range(0,theta_range.shape[0]):
                        theta = theta_range[id_theta]
                        rho =  x*np.cos(theta)+y*np.sin(theta)
                        id_rho = (np.abs(rho_range - rho)).argmin()
                        accumulator[id_rho,id_theta] += 1

    #4.Find local maxima
    acc_pad = np.pad(accumulator, (1,1), 'symmetric')
    acc_max = np.zeros(acc_pad.shape, dtype=np.uint32)
    for y in range(1,acc_pad.shape[0]-1):
        for x in range(1,acc_pad.shape[1]-1):
            acc_val = acc_pad[y, x]
            if ((acc_val >= acc_pad[y, x - 1]) and (acc_val >= acc_pad[y, x + 1]) and \
            (acc_val >= acc_pad[y - 1, x]) and (acc_val >= acc_pad[y + 1, x]) and \
            (acc_val >= acc_pad[y - 1, x + 1]) and (acc_val >= acc_pad[y + 1, x - 1]) and \
            (acc_val >= acc_pad[y - 1, x - 1]) and (acc_val >= acc_pad[y + 1, x + 1])):
                acc_max[y,x] = acc_val
    acc_max = acc_max[1:acc_pad.shape[0]+1,1:acc_pad.shape[1]+1]

    if threshold_Relative:
        acc_max = 255*(acc_max/acc_max.max())
    id_rho, id_theta = np.where(acc_max > threshold)


    rhos = list()
    for id in id_rho:
        rhos.append(rho_range[id])
    thetas = list()
    for id in id_theta:
        thetas.append(theta_range[id])

    if demo:
        accumulator = 255*(accumulator/accumulator.max())
        acc_max = 255*(acc_max/acc_max.max())
        return (rhos, thetas), accumulator.astype(np.uint8), acc_max.astype(np.uint8)
    else:
        return (rhos, thetas)

def draw(source, lines, color = -1):
    if len(source.shape) != 3:
        #convert to color if grayscale, use function from HW2 (HW1) -- imported from imageprocessing library written for homework 2
        image = gray2color(source)
    else:
        image = np.copy(source)
    if color == -1:
        color_random = True
    rhos = lines[0]
    thetas = lines[1]
    theta_sin = np.sin(lines[1])
    theta_cos = np.cos(lines[1])
    for id in range(len(thetas)):
        if color_random:
            color = (randint(0,255),randint(0,255), randint(0,255))
        a = theta_cos[id]
        b = theta_sin[id]
        rho = rhos[id]
        theta = thetas[id]
        x0 = a*rho
        y0 = b*rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        image = openCV_line(image, pt1, pt2, color)
    return image

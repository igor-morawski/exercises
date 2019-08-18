#Canny Edge Detector implementation written for homework
import numpy as np
import imageprocessing as ip


def detector(src, size, sigma, threshold1, threshold2):
    #Step 1: Filter out noise by Gaussian filtering
    g_filter = ip.getGaussianKernel(size, sigma)
    img_smoothed = ip.convolve(src, g_filter)
    #Step 2: Find gradient intensity and directions:
    gradient_intensity, gradient_direction = calculateGradient(img_smoothed)
    #Step 3: Non-max Suppresion (thinning):
    nonmax_suppressed = nonmax_suppr(gradient_intensity, gradient_direction)
    #Step 4 and 5: Double Thresholding and Edge Tracking
    lnkd, thresholded = hysteresis(nonmax_suppressed, gradient_direction, threshold1, threshold2)
    return img_smoothed, gradient_intensity, gradient_direction, nonmax_suppressed, lnkd, thresholded

def calculateGradient(image):
    operator_y, operator_x = ip.getGradientOperator(ip.SOBEL)

    intensity_y = ip.convolve(image, operator_y, True)
    intensity_x = ip.convolve(image, operator_x, True)

    #I_xy=sqrt(I_x^2+I_y^2) can be approximated as I_xy=abs(I_x)+abs(I_y)
    #approximation: gradient_strength = np.abs(intensity_y) + np.abs(intensity_x)
    gradient_intensity = np.hypot(intensity_y, intensity_x)
    #gradient_intensity = np.abs(intensity_y)+np.abs(intensity_x)
    gradient_direction = np.arctan2(intensity_y, intensity_x)

    #round directions to 0, 45, 90, 135 degrees DIRECTION QUANTIZATION
    gradient_direction = (np.round(gradient_direction*180/np.pi/45)*45)

    for y in range(gradient_direction.shape[0]):
        for x in range(gradient_direction.shape[1]):
            if gradient_direction[y,x] == 180:
                gradient_direction[y,x]=0
            if gradient_direction[y,x] < 0:
                gradient_direction[y,x]+=180

    return gradient_intensity, gradient_direction

def nonmax_suppr(strenght, direct):
    #simplified maximum suppresion, choose the highest gradient intensity along direction without interpolating
    intensities = np.pad(strenght, (1,1), 'symmetric')
    directions = np.pad(direct, (1,1), 'symmetric')
    nonmax = np.zeros(directions.shape)
    for y in range(1, intensities.shape[0]-1):
        for x in range(1, intensities.shape[1]-1):
            direction = directions[y,x]
            intensity = intensities[y,x]
            if direction == 0:
                if (intensity >= intensities[y, x - 1]) and (intensity >= intensities[y, x + 1]):
                    nonmax[y,x] = intensity
            elif direction == 90:
                if (intensity >= intensities[y - 1, x]) and (intensity >= intensities[y + 1, x]):
                    nonmax[y,x] = intensity
            elif direction == 45:
                if (intensity >= intensities[y - 1, x + 1]) and (intensity >= intensities[y + 1, x - 1]):
                    nonmax[y,x] = intensity
            elif direction == 135:
                if (intensity >= intensities[y - 1, x - 1]) and (intensity >= intensities[y + 1, x + 1]):
                    nonmax[y,x] = intensity
    return nonmax[1:direct.shape[0]+1,1:direct.shape[1]+1]


def hysteresis(image, directions, threshold_weak, threshold_strong):
    WEAK_EDGEL = 100
    STRONG_EDGEL = 200
    intensities = np.copy(image)
    result = np.zeros(intensities.shape)
    for y in range(0, intensities.shape[0]):
        for x in range(0, intensities.shape[1]):
            if intensities[y,x] > threshold_strong:
                intensities[y,x] = STRONG_EDGEL
            elif intensities[y,x] > threshold_weak:
                intensities[y,x] = WEAK_EDGEL
            else: 
                intensities[y,x] = 0
    thresh = np.copy(intensities)
    #Grass-Fire Algorithm
    j, i = np.where(intensities == STRONG_EDGEL)
    while (j.any()):
        for N in range(0,j.shape[0]):
            #mark location in result as an edge
            result[j[N],i[N]]=255
            #burn out the edgel
            intensities[j[N],i[N]]=0

        #commented out is following edge direction to check the neighborhood
        #instead I check 8-neighborhood irrespectively to the edge direction
        #this turned out to give better results
            #direction=directions[j[N], i[N]]
        #if direction == 0:  
            try: 
                if (intensities[j[N], i[N] - 1] == WEAK_EDGEL):
                    intensities[j[N], i[N] - 1] = STRONG_EDGEL
            except IndexError:
                None
            try:
                if (intensities[j[N], i[N] + 1] == WEAK_EDGEL):
                    intensities[j[N], i[N] + 1] = STRONG_EDGEL
            except IndexError:
                None
        #elif direction == 90:
            try: 
                if (intensities[j[N] - 1, i[N]] == WEAK_EDGEL):
                    intensities[j[N] - 1, i[N]] = STRONG_EDGEL
            except IndexError:
                None
            try: 
                if (intensities[j[N] + 1, i[N]] == WEAK_EDGEL):
                    intensities[j[N] + 1, i[N]] = STRONG_EDGEL
            except IndexError:
                None
        #elif direction == 45:
            try: 
                if (intensities[j[N] + 1, i[N] - 1] == WEAK_EDGEL):
                    intensities[j[N] + 1, i[N] - 1] = STRONG_EDGEL
            except IndexError:
                None
            try: 
                if (intensities[j[N] - 1, i[N] + 1] == WEAK_EDGEL):
                    intensities[j[N] - 1, i[N] + 1] = STRONG_EDGEL
            except IndexError:
                None
        #elif direction == 135:
            try: 
                if (intensities[j[N] - 1, i[N] - 1] == WEAK_EDGEL):
                    intensities[j[N] - 1, i[N] - 1] = STRONG_EDGEL
            except IndexError:
                None
            try: 
                if (intensities[j[N] + 1, i[N] + 1] == WEAK_EDGEL):
                    intensities[j[N] + 1, i[N] + 1] = STRONG_EDGEL
            except IndexError:
                None
        j, i = np.where(intensities == STRONG_EDGEL)
    return result, thresh

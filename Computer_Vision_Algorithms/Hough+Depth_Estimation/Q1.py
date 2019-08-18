#import reading and saving functions from OpenCV,
from cv2 import imread, imwrite, waitKey, imshow
#import numpy to define image matrices, get size, etc.
import numpy as np
#import module written for homework: convolution, operators, etc.
import imageprocessing as ip
#import module written for homework: Canny edge detection implementation
import cannyedge as canny
import hough

if __name__ == "__main__":

    #read input
    img1 = imread('input/house.jpg')
    img2 = imread('input/road.jpg')

    #this part is very slow because canny detector is implemented for HW2 in Python, openCV uses C++ (Python library is a wrapper)
    #nonetheless I'm using my implementation
    img1_edges = canny.detector(ip.color2gray(img1), (3,3), 1.0, 30, 90)
    img2_edges = canny.detector(ip.color2gray(img2), (3,3), 1.0, 30, 90)

    lines1, accumulator1, acc_max1 = hough.lines(img1_edges, 200, demo = True, rho_resolution = 1)
    lines2, accumulator2, acc_max2 = hough.lines(img2_edges, 128, demo = True, rho_resolution = 10)

    img1_lines = hough.draw(img1, lines1)
    img2_lines = hough.draw(img2, lines2)

    imwrite("output/house.jpg", img1_lines)
    imwrite("output/road.jpg", img2_lines)
    imwrite("output/house_acc.jpg", accumulator1)
    imwrite("output/road_acc.jpg", accumulator2)
    imwrite("output/house_accmax.jpg", acc_max1)
    imwrite("output/road_accmax.jpg", acc_max2)

#import reading and saving functions from OpenCV, 
from cv2 import imread, imwrite
#import numpy to define image matrices, get size, etc.
import numpy as np
#import module written for homework: convolution, operators, etc.
import imageprocessing as ip
#import module written for homework: Canny edge detection implementation
import cannyedge as canny

if __name__ == "__main__":
    
    #read input
    img1 = imread('data/I1.jpg')
    img2 = imread('data/I2.jpg')

    #convert to grayscale, color features are irrelevant for edge detection
    img1 = ip.color2gray(img1)
    img2 = ip.color2gray(img2)
    img2x = np.copy(img2)
    img2y = np.copy(img2)

    #get operators
    roberts_y, roberts_x = ip.getGradientOperator(ip.ROBERTS)
    prewitt_y, prewitt_x = ip.getGradientOperator(ip.PREWITT)
    sobel_y, sobel_x = ip.getGradientOperator(ip.SOBEL)

    #convolve with operators
    img1_roberts_x = ip.convolve(img1,roberts_x, no_abs=True)
    img1_roberts_y= ip.convolve(img1,roberts_y, no_abs=True)
    img1_prewitt_x = ip.convolve(img1,prewitt_x, no_abs=True)
    img1_prewitt_y = ip.convolve(img1,prewitt_y, no_abs=True)
    img1_sobel_x = ip.convolve(img1,sobel_x, no_abs=True)
    img1_sobel_y = ip.convolve(img1,sobel_y, no_abs=True)

    #get gradient strength and directions
    img1_roberts_Gs, img1_roberts_Gd = ip.getGsGd(ip.convolve(img1,roberts_y, no_abs=True), ip.convolve(img1,roberts_x, no_abs=True), ip.ROBERTS)
    img1_prewitt_Gs, img1_prewitt_Gd = ip.getGsGd(img1_prewitt_y, img1_prewitt_x, ip.PREWITT)
    img1_sobel_Gs, img1_sobel_Gd = ip.getGsGd(img1_sobel_y, img1_sobel_x, ip.SOBEL)

    #threshold Gs
    img1_roberts_Gs_thresh = ip.threshold(img1_roberts_Gs, 35)
    img1_prewitt_Gs_thresh = ip.threshold(img1_prewitt_Gs, 100)
    img1_sobel_Gs_thresh = ip.threshold(img1_sobel_Gs, 130)


    #Canny Edge detection
    img1_smoothed, img1_Canny_Sobel_Gs, \
    img1_Canny_Sobel_Gd, img1_nonmax_suppressed, \
    img1_linked, img1_thresholded = canny.detector(img1, (3,3), sigma=1.0, threshold1=30, threshold2=90)

    #save the outputs

    imwrite('output/img1_roberts_Gs.jpg',img1_roberts_Gs)
    imwrite('output/img1_prewitt_Gs.jpg',img1_prewitt_Gs)
    imwrite('output/img1_sobel_Gs.jpg',img1_sobel_Gs)

    imwrite('output/img1_roberts_Gs_thresh.jpg',img1_roberts_Gs_thresh)
    imwrite('output/img1_prewitt_Gs_thresh.jpg',img1_prewitt_Gs_thresh)
    imwrite('output/img1_sobel_Gs_thresh.jpg',img1_sobel_Gs_thresh)

    imwrite('output/img1_smoothed.jpg',img1_smoothed)
    imwrite('output/img1_Canny_Sobel_Gs.jpg',img1_Canny_Sobel_Gs)
    imwrite('output/img1_nonmax_suppressed.jpg',img1_nonmax_suppressed)
    imwrite('output/img1_linked.jpg',img1_linked)
    imwrite('output/img1_thresholded.jpg',img1_thresholded)
    

    #do the same for 2nd image ...

    #convolve with operators
    img2_roberts_x = ip.convolve(img2,roberts_x, no_abs=True)
    img2_roberts_y= ip.convolve(img2,roberts_y, no_abs=True)
    img2_prewitt_x = ip.convolve(img2,prewitt_x, no_abs=True)
    img2_prewitt_y = ip.convolve(img2,prewitt_y, no_abs=True)
    img2_sobel_x = ip.convolve(img2,sobel_x, no_abs=True)
    img2_sobel_y = ip.convolve(img2,sobel_y, no_abs=True)

    #get gradient strength and directions
    img2_roberts_Gs, img2_roberts_Gd = ip.getGsGd(ip.convolve(img2,roberts_y, no_abs=True), ip.convolve(img2,roberts_x, no_abs=True), ip.ROBERTS)
    img2_prewitt_Gs, img2_prewitt_Gd = ip.getGsGd(img2_prewitt_y, img2_prewitt_x, ip.PREWITT)
    img2_sobel_Gs, img2_sobel_Gd = ip.getGsGd(img2_sobel_y, img2_sobel_x, ip.SOBEL)

    #threshold Gs
    img2_roberts_Gs_thresh = ip.threshold(img2_roberts_Gs, 45)
    img2_prewitt_Gs_thresh = ip.threshold(img2_prewitt_Gs, 135)
    img2_sobel_Gs_thresh = ip.threshold(img2_sobel_Gs, 180)


    #Canny Edge detection
    img2_smoothed, img2_Canny_Sobel_Gs, \
    img2_Canny_Sobel_Gd, img2_nonmax_suppressed, \
    img2_linked, img2_thresholded = canny.detector(img2, (3,3), sigma=1.0, threshold1=50, threshold2=120)

    #save the outputs

    imwrite('output/img2_roberts_Gs.jpg',img2_roberts_Gs)
    imwrite('output/img2_prewitt_Gs.jpg',img2_prewitt_Gs)
    imwrite('output/img2_sobel_Gs.jpg',img2_sobel_Gs)

    imwrite('output/img2_roberts_Gs_thresh.jpg',img2_roberts_Gs_thresh)
    imwrite('output/img2_prewitt_Gs_thresh.jpg',img2_prewitt_Gs_thresh)
    imwrite('output/img2_sobel_Gs_thresh.jpg',img2_sobel_Gs_thresh)

    imwrite('output/img2_smoothed.jpg',img2_smoothed)
    imwrite('output/img2_Canny_Sobel_Gs.jpg',img2_Canny_Sobel_Gs)
    imwrite('output/img2_nonmax_suppressed.jpg',img2_nonmax_suppressed)
    imwrite('output/img2_linked.jpg',img2_linked)
    imwrite('output/img2_thresholded.jpg',img2_thresholded)
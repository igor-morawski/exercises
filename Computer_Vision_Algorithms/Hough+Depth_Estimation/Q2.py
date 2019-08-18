import cv2
import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":
    #generate list of files to load
    directory = "stereoData/"
    left_prefix = "left"
    right_prefix = "right"
    suffix = ".jpg"

    #using glob will not maintain order of files
    left_collection = list()
    right_collection = list()
    for i in range(1,15):
        f = "%02d"%i
        left_collection.append(directory+left_prefix+f+suffix)
        right_collection.append(directory+right_prefix+f+suffix)

    #callibrate left and right cameras

    #criteria EPS - accuracy, MAX_ITER - maximum iterations
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.001)

    #size of chessboard square [mm]
    size = 30

    #just following OpenCV documentation tutorial
    objp_left = np.zeros((6*7,3), np.float32)
    objp_right = np.zeros((6*7,3), np.float32)
    objp_left[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    objp_right [:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    objpoints_left = [] # 3d point in real world space
    objpoints_right  = [] # 3d point in real world space
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right  = [] # 2d points in image plane.

    #ret_left, mtx_left, dist_left, rvecs_left, tvecs_left
    #ret_right, mtx_right, dist_right, rvecs_right, tvecs_right
    for id in range(len(left_collection)):
        img_left = cv2.imread(left_collection[id],0)
        img_right = cv2.imread(left_collection[id],0)
        ret_left, corners_left = cv2.findChessboardCorners(img_left, (7,6), None)
        ret_right, corners_right = cv2.findChessboardCorners(img_right, (7,6), None)
        if ret_left and ret_right:
            objpoints_left.append(size*objp_left)
            corners2_left = cv2.cornerSubPix(img_left,corners_left,(11,11),(-1,-1),criteria)
            imgpoints_left.append(corners2_left)
            objpoints_right.append(size*objp_right)
            corners2_right = cv2.cornerSubPix(img_right,corners_right,(11,11),(-1,-1),criteria)
            imgpoints_right.append(corners2_right)


    ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(objpoints_left , imgpoints_left, img_left.shape[::-1],None,None)
    ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(objpoints_right , imgpoints_right , img_right.shape[::-1],None,None)

    ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(objpoints_left, imgpoints_left, imgpoints_right, mtx_left, dist_left, mtx_right, dist_right, img_right.shape[::-1])

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(mtx_left, dist_left, mtx_right, dist_right, img_right.shape[::-1], R, T)



    for id in range(len(left_collection)):
        img_left = cv2.imread(left_collection[id],0)
        img_right = cv2.imread(right_collection[id],0)

        h_left,  w_left = img_left.shape[:2]
        newcameramtx_left, roi_left=cv2.getOptimalNewCameraMatrix(mtx_left,dist_left,(w_left,h_left),1,(w_left,h_left))

        # undistort
        mapx_left,mapy_left = cv2.initUndistortRectifyMap(mtx_left,dist_left,R,newcameramtx_left,(w_left,h_left),5)
        img_left = cv2.remap(img_left,mapx_left,mapy_left,cv2.INTER_LINEAR)

        h_right,  w_right = img_right.shape[:2]
        newcameramtx_right, roi_right=cv2.getOptimalNewCameraMatrix(mtx_right,dist_right,(w_right,h_right),1,(w_right,h_right))

        # undistort
        mapx_right,mapy_right = cv2.initUndistortRectifyMap(mtx_right,dist_right,R,newcameramtx_right,(w_right,h_right),5)
        img_right = cv2.remap(img_right,mapx_right,mapy_right,cv2.INTER_LINEAR)

        if id is 0:
            cv2.imwrite('stereoOutput/imgL_calibrated.jpg', img_left)
            cv2.imwrite('stereoOutput/imgR_calibrated.jpg', img_right)

        #SGBM usage from http://timosam.com/python_opencv_depthimage
        # SGBM Parameters -----------------
        window_size = 3

        left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=160,
            blockSize=5,
            P1=8 * 3 * window_size ** 2,
            P2=32 * 3 * window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=7,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
        wls_filter.setLambda(80000)
        wls_filter.setSigmaColor(0.8)
        displ = left_matcher.compute(img_left, img_right)
        dispr = right_matcher.compute(img_right, img_left)
        filteredImg = wls_filter.filter(displ, img_left, None, dispr)
        for i in range(15):
            filteredImg=cv2.medianBlur(filteredImg, 3)
        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
        filteredImg = np.uint8(filteredImg)
        cv2.imwrite('stereoOutput/'+str(id+1)+suffix, filteredImg)

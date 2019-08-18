/*
   OpenCV version 4.0.0, using only:
    - cv::imread
    - cv::imwrite
    - cv::imshow
    - cv::waitKey
    - cv::Mat
    - cv::Mat.cols, rows, size(), channels()
    - cv::Mat.data to check if image is found
 */
#include <opencv2/highgui.hpp>
#include <iostream>

#include "imageprocessing.hpp" //functions written for homework

int main( int argc, char** argv ) {
        cv::Mat image1;
        std::string fn1 = "I1";
        image1 = cv::imread("data/"+fn1+".jpg", cv::IMREAD_ANYCOLOR);

        if(!image1.data ) {
                std::cout <<  "Could not open or find the image" << std::endl;
                return -1;
        }

        cv::Mat image1_gray = imageprocessing::convert2gray(image1);
        cv::Mat image1_thresholded = imageprocessing::threshold(image1);
        cv::Mat image1_rotated90right = imageprocessing::rotate90right(image1);

        cv::imwrite("output/"+fn1+"_gray"+".jpg", image1_gray);
        cv::imwrite("output/"+fn1+"_thresholded"+".jpg", image1_thresholded);
        cv::imwrite("output/"+fn1+"_rotated90right"+".jpg", image1_rotated90right);
        ////////////////////////////////////////////

        cv::Mat image2;
        std::string fn2 = "I2";
        image2 = cv::imread("data/"+fn2+".jpg", cv::IMREAD_ANYCOLOR);

        if(!image2.data ) {
                std::cout <<  "Could not open or find the image" << std::endl;
                return -1;
        }

        cv::Mat image2_gray = imageprocessing::convert2gray(image2);
        cv::Mat image2_thresholded = imageprocessing::threshold(image2);
        cv::Mat image2_rotated90right = imageprocessing::rotate90right(image2);

        cv::imwrite("output/"+fn2+"_gray"+".jpg", image2_gray);
        cv::imwrite("output/"+fn2+"_thresholded"+".jpg", image2_thresholded);
        cv::imwrite("output/"+fn2+"_rotated90right"+".jpg", image2_rotated90right);
        ////////////////////////////////////////////

        cv::Mat image3;
        std::string fn3 = "N1";
        image3 = cv::imread("data/"+fn3+".jpg", cv::IMREAD_ANYCOLOR);

        if(!image3.data ) {
                std::cout <<  "Could not open or find the image" << std::endl;
                return -1;
        }

        cv::Mat image3_mean = imageprocessing::filter2D(image3, IP_MEAN, cv::Size(3,3));
        cv::Mat image3_median = imageprocessing::filter2D(image3, IP_MEDIAN, cv::Size(3,3));
        cv::Mat image3_gauss = imageprocessing::filter2D(image3, IP_GAUSS, cv::Size(3,3), 1.);


        cv::imwrite("output/"+fn3+"_mean"+".jpg", image3_mean);
        cv::imwrite("output/"+fn3+"_median"+".jpg", image3_median);
        cv::imwrite("output/"+fn3+"_gauss"+".jpg", image3_gauss);
        ////////////////////////////////////////////

        cv::Mat image4;
        std::string fn4 = "N3";
        image4 = cv::imread("data/"+fn4+".jpg", cv::IMREAD_ANYCOLOR);

        if(!image4.data ) {
                std::cout <<  "Could not open or find the image" << std::endl;
                return -1;
        }

        cv::Mat image4_mean = imageprocessing::filter2D(image4, IP_MEAN, cv::Size(3,3));
        cv::Mat image4_median = imageprocessing::filter2D(image4, IP_MEDIAN, cv::Size(3,3));
        cv::Mat image4_gauss = imageprocessing::filter2D(image4, IP_GAUSS, cv::Size(5,5), 1.);
        cv::Mat result = image4;
        result = imageprocessing::filter2D(result, IP_MEDIAN, cv::Size(3,3));
        result = imageprocessing::filter2D(result, IP_MEDIAN, cv::Size(5,5));
        result = imageprocessing::filter2D(result, IP_MEAN, cv::Size(3,3));
        result = imageprocessing::filter2D(result, IP_MEAN, cv::Size(3,3));



        cv::imwrite("output/"+fn4+"_result"+".jpg", result);
        cv::imwrite("output/"+fn4+"_mean"+".jpg", image4_mean);
        cv::imwrite("output/"+fn4+"_median"+".jpg", image4_median);
        cv::imwrite("output/"+fn4+"_gauss"+".jpg", image4_gauss);
        ////////////////////////////////////////////

        return 0;
}

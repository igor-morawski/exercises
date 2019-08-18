#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>
#include <vector>

#define IP_COLOR 16
#define IP_GRAYSCALE 0
#define IP_MEAN 0
#define IP_MEDIAN 1
#define IP_GAUSS 2

namespace imageprocessing {

cv::Mat convert2gray(cv::Mat source) {
        if (source.type() == CV_8UC1) return source;
        if (source.type() != CV_8UC3) throw std::invalid_argument("Unsupported Mat type (CV_8UC1 and CV_8UC3 supported");
        cv::Mat result = cv::Mat(source.rows, source.cols, CV_8UC1);
        for (int y=0; y<source.rows; y++) {
                for (int x=0; x<source.cols; x++) {
                        int accumulator=0;
                        for (int ch = 0; ch < source.channels(); ch++) {
                                accumulator+=source.at<cv::Vec3b>(y,x)[ch];
                        }
                        accumulator/=3;
                        result.at<uchar>(y,x) = (uchar)accumulator;
                }
        }
        return result;
}

cv::Mat threshold(cv::Mat input, cv::Scalar thresValue = cv::Scalar(100), int color=IP_GRAYSCALE) {
        /*returns thresholded input image of grayscale image if color==IP_GRAYSCALE
           return thresholded input image of color image if color==IP_COLOR*/
        if (input.type() != CV_8UC1 && input.type() != CV_8UC3) throw std::invalid_argument("Unsupported Mat type (CV_8UC1 and CV_8UC3 supported");
        cv::Mat source = (color==IP_GRAYSCALE) ? convert2gray(input) : input;
        cv::Mat result;
        if (color==IP_GRAYSCALE) {
                source = convert2gray(input);
                return result = source > thresValue[0];
        }
        if (color==IP_COLOR) {
                if (source.type() != CV_8UC3) throw std::invalid_argument("Unsupported Mat type for thresholding in three channels");
                cv::Mat result = cv::Mat(source.rows, source.cols, CV_8UC1);
                for (int y=0; y<source.rows; y++) {
                        for (int x=0; x<source.cols; x++) {
                                for (int ch = 0; ch < source.channels(); ch++) {
                                        result.at<uchar>(y,x) = (source.at<cv::Vec3b>(y,x)[0] > thresValue[0] && source.at<cv::Vec3b>(y,x)[1] > thresValue[1] && source.at<cv::Vec3b>(y,x)[2] > thresValue[2]) ? 255 : 0;
                                }
                        }
                }
        }
        return result;
}

cv::Mat rotate90right(cv::Mat source) {
        if (source.type() != CV_8UC1 && source.type() != CV_8UC3) throw std::invalid_argument("Unsupported Mat type (CV_8UC1 and CV_8UC3 supported");
        cv::Mat result = cv::Mat(source.cols, source.rows, source.type());
        if (source.type() == CV_8UC1) {
                for (int y=0; y<source.rows; y++) {
                        for (int x=0; x<source.cols; x++) {
                                result.at<uchar>(x,result.cols-y-1) = source.at<uchar>(y,x);
                        }
                }
        }
        if (source.type() == CV_8UC3) {
                for (int y=0; y<source.rows; y++) {
                        for (int x=0; x<source.cols; x++) {
                                for (int ch = 0; ch < source.channels(); ch++) {
                                        result.at<cv::Vec3b>(x,result.cols-y-1)[ch] = source.at<cv::Vec3b>(y,x)[ch];
                                }
                        }
                }
        }
        return result;
        throw std::invalid_argument("Unsupported Mat type (CV_8UC1 and CV_8UC3 supported");
}

cv::Mat getMeanKernel(cv::Size size) {
        return cv::Mat(size, CV_32FC1, cv::Scalar(1));
}

cv::Mat getMedianKernel(cv::Size size) {
        return cv::Mat(size, CV_32FC1, cv::Scalar(1));
}

cv::Mat getGaussianKernel(cv::Size size, float sigma) {
        cv::Mat gaussKenrel = cv::Mat(size, CV_32FC1);
        int x_middle = (size.width)/2.;
        int y_middle = (size.height)/2.;
        float e_sigma = 1./(2*sigma*sigma);
        for (int t=-y_middle; t<y_middle+1; t++) {
                for (int s=-x_middle; s<x_middle+1; s++) {
                        int y = t+y_middle;
                        int x = s+x_middle;
                        float e_power = -(s*s+t*t) * e_sigma;
                        gaussKenrel.at<float>(y,x) = std::exp(-(s*s + t*t) * e_sigma);
                }
        }
        return gaussKenrel;
}

cv::Mat normalizeKernel(cv::Mat kernel) {
        cv::Mat result = cv::Mat(kernel.size(), kernel.type());
        float accumulator = 0.;
        for (int y=0; y<kernel.rows; y++) {
                for (int x=0; x<kernel.cols; x++) {
                        accumulator+=kernel.at<float>(y,x);
                }
        }
        float normalizeConstant = 1./accumulator;
        for (int y=0; y<kernel.rows; y++) {
                for (int x=0; x<kernel.cols; x++) {
                        result.at<float>(y,x)=kernel.at<float>(y,x)*normalizeConstant;
                }
        }
        return result;
}

cv::Mat getKernel(cv::Size size, int type, float sigma = 1.) {
        cv::Mat kernel;
        if (type == IP_MEAN) {
                kernel = getMeanKernel(size);
        }
        if (type == IP_GAUSS) {
                kernel=getGaussianKernel(size, sigma);
        }
        return normalizeKernel(kernel);
}

cv::Mat convolve(cv::Mat source, cv::Mat kernel){
/*convolves grayscale image with 2D mxm filter,
   mirroring on image's border*/
        if (source.type() != CV_8UC1 && source.type() != CV_8UC3) throw std::invalid_argument("Unsupported Mat type (CV_8UC1 and CV_8UC3 supported");
        cv::Mat result = cv::Mat(source.size(), source.type());
        int x_middle = (kernel.size().width)/2.;
        int y_middle = (kernel.size().height)/2.;
        if (source.type() == CV_8UC1) {
                for (int y=0; y<source.rows; y++) {
                        for (int x=0; x<source.cols; x++) {
                                float accumulator = 0.;
                                for (int t=-y_middle; t<y_middle+1; t++) {
                                        for (int s=-x_middle; s<x_middle+1; s++) {
                                                int xPad = 0;
                                                int yPad = 0;
                                                while (x + s + xPad < 0) { xPad++; }
                                                while (y + t + yPad < 0) { yPad++; }
                                                while (x + s + xPad >= source.cols) { xPad--; }
                                                while (y + t + yPad >= source.rows) { yPad--; }
                                                accumulator+=source.at<uchar>(y + t + yPad, x + s + xPad ) * kernel.at<float>(t+y_middle, s+x_middle);
                                        }
                                }
                                result.at<uchar>(y,x) = (uchar)accumulator;
                        }
                }
        }
        if (source.type() == CV_8UC3) {
                for (int y=0; y<source.rows; y++) {
                        for (int x=0; x<source.cols; x++) {
                                for (int ch=0; ch<source.channels(); ch++) {
                                        float accumulator = 0.;
                                        for (int t=-y_middle; t<y_middle+1; t++) {
                                                for (int s=-x_middle; s<x_middle+1; s++) {
                                                        int xPad = 0;
                                                        int yPad = 0;
                                                        while (x + s + xPad < 0) { xPad++; }
                                                        while (y + t + yPad < 0) { yPad++; }
                                                        while (x + s + xPad >= source.cols) { xPad--; }
                                                        while (y + t + yPad >= source.rows) { yPad--; }
                                                        accumulator+=source.at<cv::Vec3b>(y + t + yPad, x + s + xPad )[ch] * kernel.at<float>(t+y_middle, s+x_middle);
                                                }
                                        }
                                        result.at<cv::Vec3b>(y,x)[ch] = (uchar)accumulator;
                                }
                        }
                }
        }
        return result;
}
uchar calculateMedian(std::vector<uchar> values) {
        std::sort(values.begin(), values.end());
        if (values.size() % 2 == 0)
        {
                return (values[values.size() / 2 - 1] + values[values.size() / 2]) / 2;
        }
        else
        {
                return values[values.size() / 2];
        }
}

cv::Mat medianFiltering(cv::Mat source, cv::Size size) {
        if (source.type() != CV_8UC1 && source.type() != CV_8UC3) throw std::invalid_argument("Unsupported Mat type (CV_8UC1 and CV_8UC3 supported");
        cv::Mat result = cv::Mat(source.size(), source.type());
        int x_middle = (size.width)/2.;
        int y_middle = (size.height)/2.;
        if (source.type() == CV_8UC1) {
                for (int y=y_middle-1; y<source.rows-y_middle+1; y++) {
                        for (int x=x_middle-1; x<source.cols-x_middle+1; x++) {
                                std::vector<uchar> pixelValues;
                                for (int t=-y_middle; t<y_middle+1; t++) {
                                        for (int s=-x_middle; s<x_middle+1; s++) {
                                                pixelValues.push_back(source.at<uchar>(y + t, x + s));
                                        }
                                }
                                result.at<uchar>(y,x) = calculateMedian(pixelValues);
                        }
                }
        }
        if (source.type() == CV_8UC3) {
                for (int y=y_middle-1; y<source.rows-y_middle+1; y++) {
                        for (int x=x_middle-1; x<source.cols-x_middle+1; x++)  {
                                for (int ch=0; ch<source.channels(); ch++) {
                                        std::vector<uchar> pixelValues;
                                        for (int t=-y_middle; t<y_middle+1; t++) {
                                                for (int s=-x_middle; s<x_middle+1; s++) {

                                                        pixelValues.push_back(source.at<cv::Vec3b>(y + t, x + s)[ch]);
                                                }
                                        }
                                        result.at<cv::Vec3b>(y,x)[ch] = calculateMedian(pixelValues);
                                }
                        }
                }
        }
        return result;

}

cv::Mat filter2D(cv::Mat source, int type, cv::Size size, float sigma = 1.){
        if (source.type() != CV_8UC1 && source.type() != CV_8UC3) throw std::invalid_argument("Unsupported Mat type (CV_8UC1 and CV_8UC3 supported");
        cv::Mat result;
        cv::Mat kernel;
        if (type == IP_MEAN) {
                kernel = getKernel(size, type);
                result = convolve(source, kernel);
        }
        if (type == IP_MEDIAN) {
                result = medianFiltering(source, size);
        }
        if (type == IP_GAUSS) {
                kernel = getKernel(size, type, sigma);
                result = convolve(source, kernel);
        }
        return result;
}

}

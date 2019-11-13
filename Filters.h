#pragma once
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"

#define __WHITE_NOISE_ 1
#define __SALT_PEPPER_ 3
#define __NONE_ 0
#define __OPENCV_ 2
#define __MEDIAN_ 4

void WhiteNoise(const cv::Mat& input, cv::Mat& output);

void MedianFilter(const cv::Mat& input, cv::Mat& output);

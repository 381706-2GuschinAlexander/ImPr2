#pragma once
#include <ctime>
#include <random>
#include <math.h>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

#define __CHANCE_ 33

#define __WHITE_NOISE_ 1
#define __SALT_PEPPER_ 3
#define __NONE_ 0
#define __OPENCV_ 2
#define __MEDIAN_ 4
#define __ATRIM_ 6

#define __OTSU_ 7

void WhiteNoise(const cv::Mat& input, cv::Mat& output);

// 2-nd lab
void MedianFilter(const cv::Mat& input, cv::Mat& output, int radius);
void ATrimFilter(const cv::Mat& input, cv::Mat& output, int radius, int alpha);

// 3-rd lab
void Monochrome(const cv::Mat& input, cv::Mat& output);
unsigned char OtsuThreshold(const cv::Mat& input);
void Binarization(const cv::Mat& input, cv::Mat& output, unsigned char threshold);
void OtsuFilter(const cv::Mat& input, cv::Mat& output);

// unity
double ConditionalExp(const cv::Mat& input);
double Dispersion(const cv::Mat& input, const double conditional_expectation);
double covFuncion(double& mW1, double& mW2, cv::Mat& a, cv::Mat& b);
double ssim(double& cE1, double& cE2, double& dis1, double& dis2, double cov);
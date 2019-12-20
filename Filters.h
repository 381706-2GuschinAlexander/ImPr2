#pragma once
#include <math.h>
#include <ctime>
#include <random>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

#define _USE_MATH_DEFINES

#define __CHANCE_ 33
#define __MAX_DISTANCE_ 441.67295593 + 1

#define __WHITE_NOISE_ 1
#define __SALT_PEPPER_ 3
#define __NONE_ 0
#define __OPENCV_ 2
#define __MEDIAN_ 4
#define __ATRIM_ 6

#define __OTSU_ 7

#define __HUFFMAN 8
#define __GROW_ 9

typedef std::vector<std::vector<int>> Matrix;


void WhiteNoise(const cv::Mat& input, cv::Mat& output);

// 2-nd lab
void MedianFilter(const cv::Mat& input, cv::Mat& output, int radius);
void ATrimFilter(const cv::Mat& input, cv::Mat& output, int radius, int alpha);

// 3-rd lab
void Monochrome(const cv::Mat& input, cv::Mat& output);
unsigned char OtsuThreshold(const cv::Mat& input);
void Binarization(const cv::Mat& input, cv::Mat& output,
                  unsigned char threshold);
void OtsuFilter(const cv::Mat& input, cv::Mat& output);

// 4-th lab
int fib(int n);
void histPull(const cv::Mat& sourse, int* hist, unsigned char& count);
void strconcat(char* str, char* paterncode, char add);
void probabilitisOfIntensity(int* hist, double* res, int countOfPixels);
void HuffmanCompression(const cv::Mat& input, cv::Mat output);
void GrowFilter(cv::Mat& image, int thr);
void Merge(cv::Mat& image, int reg1, int reg2, int nI, int end_x, int end_y,
           Matrix& A, int count);
int ToGray(const cv::Vec3b& color);

// unity
double ConditionalExp(const cv::Mat& input);
double Dispersion(const cv::Mat& input, const double conditional_expectation);
double covFuncion(double& mW1, double& mW2, cv::Mat& a, cv::Mat& b);
double ssim(double& cE1, double& cE2, double& dis1, double& dis2, double cov);
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include "Filters.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"


/*long long factorial(int number)
{
    if (number == 1)
        return 1;
    else
        return number * factorial(number - 1);
}*/



/*void covFuncion(double& cov, double* mW, int i1, int i2, Mat& a, Mat& b) {
  for (int i = 0; i < a.rows; i++)
    for (int j = 0; j < a.cols; j++)
      cov = ((a.at<cv::Vec3b>(i, j)[0] + a.at<cv::Vec3b>(i, j)[1] +
              a.at<cv::Vec3b>(i, j)[2] / 3) -
             mW[i1]) *
            ((b.at<cv::Vec3b>(i, j)[0] + b.at<cv::Vec3b>(i, j)[1] +
              b.at<cv::Vec3b>(i, j)[2] / 3) -
             mW[i2]);
  cov = sqrt(cov);
}

double covFuncion1(int& mW1, int& mW2, Mat& a, Mat& b) {
  double cov = 0;
  for (int i = 0; i < a.rows; i++)
    for (int j = 0; j < a.cols; j++)
      cov = ((a.at<cv::Vec3b>(i, j)[0] + a.at<cv::Vec3b>(i, j)[1] +
              a.at<cv::Vec3b>(i, j)[2] / 3) -
             mW1) *
            ((b.at<cv::Vec3b>(i, j)[0] + b.at<cv::Vec3b>(i, j)[1] +
              b.at<cv::Vec3b>(i, j)[2] / 3) -
             mW2);
  cov = sqrt(cov);
  return cov;
}

void ssimFunctuon(double& cov, double& ssim, double* mW, double* dis, int i1,
                  int i2) {
  ssim = ((2 * mW[i1] * mW[i2] + 2.55 * 2.55) * (2 * cov + 7.65 * 7.65)) /
         ((mW[i1] * mW[i1] + mW[i2] * mW[i2] + 2.55 * 2.55) *
          (dis[i1] * dis[i1] + dis[i2] * dis[i2] + 7.65 * 7.65));
}*/

void applyFilter(std::string source, int noise_type, int filter_type) {
  //original image
  cv::Mat Image = cv::imread(source, 1);
  cv::namedWindow("Source image", cv::WINDOW_AUTOSIZE);
  imshow("Source image", Image);
  cv::waitKey(0);

  //noise
  cv::Mat NoiseImage(Image);
  if (noise_type == __WHITE_NOISE_) WhiteNoise(Image, NoiseImage);
  if (noise_type != 0) {
    cv::namedWindow("Source image", cv::WINDOW_AUTOSIZE);
    imshow("Source image", NoiseImage);
    cv::waitKey(0);
  }

  //filter
  cv::Mat FiltredImage(Image);
  if (filter_type == __MEDIAN_) MedianFilter(NoiseImage, FiltredImage);
}


int main(int argc, char** argv) {
  srand(time(NULL));
  int default_noise = __WHITE_NOISE_;
  try {
    std::string source = argv[1];
    std::string typeOfFilter = argv[2];
    if (typeOfFilter == "Median")
      applyFilter(source, default_noise, __MEDIAN_);
  } catch (const std::exception&) {
    std::cout << "Error! Wrong image source or type of filter" << std::endl;
  }
  return 0;
}
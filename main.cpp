#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include "Filters.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

void applyFilter(std::string source, int noise_type, int filter_type) {
  //original image
  cv::Mat Image = cv::imread(source, 1);
  cv::namedWindow("Source image", cv::WINDOW_AUTOSIZE);
  imshow("Source image", Image);
  cv::waitKey(0);

  //noise
  cv::Mat NoiseImage;
  Image.copyTo(NoiseImage);

  if (noise_type == __WHITE_NOISE_) WhiteNoise(Image, NoiseImage);
  if (noise_type != __NONE_) {
    cv::namedWindow("Source image", cv::WINDOW_AUTOSIZE);
    imshow("Source image", NoiseImage);
    cv::waitKey(0);
  }

  //filter
  cv::Mat FiltredImage;
  Image.copyTo(FiltredImage);

  if (filter_type == __ATRIM_) ATrimFilter(NoiseImage, FiltredImage, 1, 4);
  if (filter_type != __NONE_) {
    //calculation
    double C_E_filtred = ConditionalExp(FiltredImage);
    double C_E_noise = ConditionalExp(NoiseImage);
    double dispersion_filtred = Dispersion(FiltredImage, C_E_filtred);
    double dispersion_noise = Dispersion(NoiseImage, C_E_noise);
    //statistic 
    std::cout << "Conditional Expectation of filterd image= " << C_E_filtred
              << "\n";  
    std::cout << "Conditional Expectation of noise image= " << C_E_noise
              << "\n";
    std::cout << "Dispersion of filterd image= " << dispersion_filtred
              << "\n";
    std::cout << "Dispersion of noise image= " << dispersion_noise
              << "\n";
    //pic
    cv::namedWindow("Source image", cv::WINDOW_AUTOSIZE);
    imshow("Source image", FiltredImage);
    cv::waitKey(0);
  }
}


int main(int argc, char** argv) {
  srand(time(NULL));
  int default_noise = __WHITE_NOISE_;
  try {
    std::string source = argv[1];
    std::string typeOfFilter = argv[2];
    if (typeOfFilter == "Median")
      applyFilter(source, default_noise, __MEDIAN_);
    if (typeOfFilter == "ATrim") 
      applyFilter(source, default_noise, __ATRIM_);
  } catch (const std::exception&) {
    std::cout << "Error! Wrong image source or type of filter" << std::endl;
  }
  return 0;
}
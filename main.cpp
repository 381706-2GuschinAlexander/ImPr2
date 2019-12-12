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
  // original image
  cv::Mat Image = cv::imread(source, 1);
  cv::namedWindow("Source image", cv::WINDOW_AUTOSIZE);
  imshow("Source image", Image);
  cv::waitKey(0);

  // noise
  cv::Mat NoiseImage;
  Image.copyTo(NoiseImage);

  if (noise_type == __WHITE_NOISE_) WhiteNoise(Image, NoiseImage);
  if (noise_type != __NONE_) {
    cv::namedWindow("Noise image", cv::WINDOW_AUTOSIZE);
    imshow("Noise image", NoiseImage);
    cv::waitKey(0);
  }

  // filter
  cv::Mat FiltredImage;
  Image.copyTo(FiltredImage);

  if (filter_type == __ATRIM_) ATrimFilter(NoiseImage, FiltredImage, 1, 4);
  if (filter_type == __OTSU_) OtsuFilter(Image, FiltredImage);
//  if (filter_type == __GROW_) GrowFilter(Image, FiltredImage, 10);
    if (filter_type != __NONE_ && filter_type < 7) {
      // calculation
      double C_E_filtred = ConditionalExp(FiltredImage);
      double C_E_source = ConditionalExp(Image);
      double dispersion_filtred = Dispersion(FiltredImage, C_E_filtred);
      double dispersion_source = Dispersion(Image, C_E_source);
      double cov = covFuncion(C_E_filtred, C_E_source, FiltredImage, Image);
      double ssim_stat = ssim(C_E_filtred, C_E_source, dispersion_filtred,
                              dispersion_source, cov);
      // statistic
      std::cout << "Conditional Expectation of filterd image= " << C_E_filtred
                << "\n";
      std::cout << "Conditional Expectation of source image= " << C_E_source
                << "\n";
      std::cout << "Dispersion of filterd image= " << dispersion_filtred
                << "\n";
      std::cout << "Dispersion of source image= " << dispersion_source << "\n";
      std::cout << "Covariation of images= " << cov << "\n";
      std::cout << "ssim of images= " << ssim_stat << "\n";
    }
  // pic
  if (filter_type != __NONE_) {
    cv::namedWindow("Filtred image", cv::WINDOW_AUTOSIZE);
    imshow("Filtred image", FiltredImage);
    cv::waitKey(0);
  }
}

int main(int argc, char** argv) {
  srand(time(NULL));
  int default_noise = __WHITE_NOISE_;
  try {
    std::string source = argv[1];
    std::string typeOfFilter = argv[2];
    if (typeOfFilter == "Median") applyFilter(source, default_noise, __MEDIAN_);
    if (typeOfFilter == "ATrim") applyFilter(source, default_noise, __ATRIM_);
    if (typeOfFilter == "Otsu") applyFilter(source, __NONE_, __OTSU_);
    if (typeOfFilter == "Grow") applyFilter(source, __NONE_, __GROW_);
  } catch (const std::exception&) {
    std::cout << "Error! Wrong image source or type of filter" << std::endl;
  }
  return 0;
}